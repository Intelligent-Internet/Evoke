#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.server
import json
import os
import re
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import psycopg

from ii42_test_support import (
    extension_control_root,
    vacuum_with_session_maintenance_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ITEM_POINTER_BYTES = 6
SEMANTIC_FIXED_WORKSPACE_BYTES = 32 * 1024 * 1024


def rebuild_estimate(cache_state: dict[str, object], name: str) -> int:
    raw_state = str(cache_state['raw_state'])
    match = re.search(rf'{name}=([0-9]+)', raw_state)
    if match is None:
        raise AssertionError(
            f'could not parse {name} from generation state: {raw_state}'
        )
    return int(match.group(1))


def assert_cache_state_snapshot(cache_state: dict[str, object]) -> None:
    generation = cache_state['generation']
    debt = cache_state['debt']
    if not isinstance(generation, dict) or not isinstance(debt, dict):
        raise AssertionError(f'invalid runtime state: {cache_state}')
    expected = {
        'cache_epoch': int(cache_state['cache_epoch']),
        'docs': int(generation['docs']),
        'pending_writes': int(debt['pending_writes']),
        'pending_deletes': int(debt['pending_deletes']),
        'delta_records': int(debt['delta_records']),
        'delta_bytes': int(debt['delta_bytes']),
    }
    observed = {
        name: rebuild_estimate(cache_state, name)
        for name in expected
    }
    if observed != expected:
        raise AssertionError(
            'runtime state JSON mixed multiple state snapshots: '
            f'observed={observed}, expected={expected}, state={cache_state}'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the unified runtime-service product smoke.',
    )
    parser.add_argument(
        '--pg-bin',
        default='/opt/homebrew/opt/postgresql@18/bin',
        help='Directory containing initdb, pg_ctl, and psql.',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Production model checkout using the current runtime contract.',
    )
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share directory containing extension/ii42.control, '
            'or the extension directory itself.'
        ),
    )
    parser.add_argument(
        '--worker-count',
        type=int,
        default=2,
        choices=range(2, 17),
        metavar='2..16',
        help='Restart-only runtime worker count for this isolated cluster.',
    )
    parser.add_argument(
        '--use-packaged-model',
        action='store_true',
        help=(
            'Do not configure ii42.sae_model_path; require the extension '
            'binary to resolve --model-path as its packaged checkout.'
        ),
    )
    parser.add_argument(
        '--runtime-server-binary',
        type=Path,
        help='Optional C++ ii42-runtime-server binary to smoke over HTTP.',
    )
    parser.add_argument(
        '--runtime-liveness-timeout-ms',
        type=int,
        default=300_000,
        help='Restart-only runtime liveness timeout for this temp cluster.',
    )
    parser.add_argument(
        '--failover-backpressure-only',
        action='store_true',
        help=(
            'Run only the remote-deadline/local-queue backpressure probe. '
            'Requires --runtime-server-binary and a short liveness timeout.'
        ),
    )
    parser.add_argument(
        '--planner-native-scale-docs',
        type=int,
        default=0,
        help=(
            'Optionally run a planner-native selectivity ladder with this '
            'many synthetic documents.'
        ),
    )
    return parser.parse_args()


def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        **kwargs,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result


def psql(
    psql_bin: Path,
    socket_dir: Path,
    port: int,
    sql: str,
) -> str:
    result = run(
        [
            str(psql_bin),
            '-X',
            '-q',
            '-t',
            '-A',
            '-h',
            str(socket_dir),
            '-p',
            str(port),
            '-d',
            'postgres',
            '-v',
            'ON_ERROR_STOP=1',
        ],
        input=sql,
    )
    return result.stdout.strip()


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 0))
        return int(probe.getsockname()[1])


def http_json(
    method: str,
    url: str,
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    data = None
    headers = {'Accept': 'application/json'}
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode('utf-8')
            return int(response.status), json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8')
        return int(exc.code), json.loads(body)


def wait_http_ready(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30.0
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = ''
            if process.stderr is not None:
                stderr = process.stderr.read()
            raise AssertionError(
                'C++ runtime server exited before becoming ready: '
                f'code={process.returncode} stderr={stderr}'
            )
        try:
            status, body = http_json('GET', f'{base_url}/health')
            if status == 200 and body.get('ok') is True:
                return
        except Exception as exc:  # pragma: no cover - diagnostic path
            last_error = exc
        time.sleep(0.05)
    raise AssertionError(
        f'C++ runtime server did not become ready: {last_error}'
    )


def runtime_http_request(
    base_url: str,
    checkout_signature: str,
    text: str,
) -> dict[str, object]:
    status, body = http_json(
        'POST',
        f'{base_url}/v1/encode',
        {
            'mode': 'document',
            'checkout_signature': checkout_signature,
            'runtime_precision': 'fp16',
            'texts': [text],
        },
    )
    if status != 200:
        raise AssertionError(f'HTTP encode failed: status={status} body={body}')
    return body


def assert_runtime_connection_capacity(port: int, capacity: int) -> None:
    idle_sockets: list[socket.socket] = []
    overflow: socket.socket | None = None

    try:
        for _ in range(capacity):
            idle_sockets.append(
                socket.create_connection(('127.0.0.1', port), timeout=2)
            )
        time.sleep(0.1)
        overflow = socket.create_connection(
            ('127.0.0.1', port),
            timeout=2,
        )
        response = overflow.recv(4096)
        if b' 503 ' not in response:
            raise AssertionError(
                'runtime server did not enforce its connection cap: '
                f'{response!r}'
            )
        time.sleep(1.2)
        idle_sockets[0].settimeout(2)
        expired_response = idle_sockets[0].recv(4096)
        if expired_response and b' 400 ' not in expired_response:
            raise AssertionError(
                'runtime server did not expire an idle connection: '
                f'{expired_response!r}'
            )
    finally:
        if overflow is not None:
            overflow.close()
        for connection in idle_sockets:
            connection.close()
    time.sleep(0.1)


def run_cpp_runtime_server_probe(
    runtime_server_binary: Path,
    socket_dir: Path,
    port: int,
    model_path: Path,
    *,
    failover_backpressure_only: bool = False,
    runtime_liveness_timeout_ms: int = 300_000,
) -> None:
    if not runtime_server_binary.is_file():
        raise FileNotFoundError(
            f'C++ runtime server was not found: {runtime_server_binary}'
        )
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    runtime_port = free_tcp_port()
    base_url = f'http://127.0.0.1:{runtime_port}'
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                'SELECT ii42_index_runtime_plan_internal(%s)',
                (str(model_path),),
            )
            plan = cursor.fetchone()[0]

    checkout = str(plan['checkout_signature'])
    process = subprocess.Popen(
        [
            str(runtime_server_binary),
            '--model-path',
            str(model_path),
            '--checkout-signature',
            checkout,
            '--host',
            '127.0.0.1',
            '--port',
            str(runtime_port),
            '--max-delay-ms',
            '100',
            '--max-batch-size',
            '192',
            '--worker-count',
            '1',
            '--request-timeout-s',
            '120',
            '--max-connections',
            '8',
            '--connection-idle-timeout-s',
            '1',
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_http_ready(base_url, process)
        if failover_backpressure_only:
            run_accelerator_local_failover_backpressure_probe(
                dsn,
                base_url,
                model_path,
                runtime_liveness_timeout_ms,
            )
            return
        assert_runtime_connection_capacity(runtime_port, 8)
        status, health = http_json('GET', f'{base_url}/health')
        if (
            status != 200
            or health.get('runtime_backend') != 'direct'
            or health.get('checkout_signature') != checkout
            or health.get('runtime_precision') != 'fp16'
        ):
            raise AssertionError(f'bad runtime health: {status} {health}')

        status, models = http_json('GET', f'{base_url}/v1/models')
        if (
            status != 200
            or models.get('model_id') != plan['model_id']
            or models.get('runtime_abi') != plan['runtime_abi']
            or models.get('runtime_precision') != 'fp16'
            or models.get('runtime_backend') != 'direct'
            or int(models.get('max_batch_size', 0)) != 192
        ):
            raise AssertionError(f'bad runtime model metadata: {models}')

        status, encoded = http_json(
            'POST',
            f'{base_url}/v1/encode',
            {
                'mode': 'document',
                'checkout_signature': checkout,
                'runtime_precision': 'fp16',
                'texts': [
                    'cpp runtime server document sentinel',
                    'cpp runtime server second document sentinel',
                ],
            },
        )
        if (
            status != 200
            or encoded.get('checkout_signature') != checkout
            or encoded.get('runtime_precision') != 'fp16'
            or encoded.get('mode') != 'document'
            or encoded.get('runtime_backend') != 'direct'
            or len(list(encoded.get('results', []))) != 2
        ):
            raise AssertionError(f'bad runtime encode response: {encoded}')

        status, rejected = http_json(
            'POST',
            f'{base_url}/v1/encode',
            {
                'mode': 'document',
                'checkout_signature': checkout,
                'runtime_precision': 'fp16',
                'texts': [
                    f'cpp runtime server oversized batch document {index}'
                    for index in range(193)
                ],
            },
        )
        if status != 400 or rejected.get('error') != 'invalid text batch size':
            raise AssertionError(
                f'runtime server did not enforce the service batch cap: '
                f'{status} {rejected}',
            )

        encoded_provider = str(encoded.get('active_provider', ''))
        if encoded_provider not in {'cpu', 'cuda', 'coreml', 'tensorrt'}:
            raise AssertionError(
                f'bad runtime provider after HTTP encode: {encoded}'
            )

        status, mismatch = http_json(
            'POST',
            f'{base_url}/v1/encode',
            {
                'mode': 'document',
                'checkout_signature': 'not-the-current-checkout',
                'runtime_precision': 'fp16',
                'texts': ['must be rejected'],
            },
        )
        if status != 409 or mismatch.get('error') != 'model checkout mismatch':
            raise AssertionError(
                f'checkout mismatch was not rejected: {status} {mismatch}'
            )

        status, mismatch = http_json(
            'POST',
            f'{base_url}/v1/encode',
            {
                'mode': 'document',
                'checkout_signature': checkout,
                'runtime_precision': 'fp32',
                'texts': ['must be rejected'],
            },
        )
        if status != 409 or mismatch.get('error') != 'runtime precision mismatch':
            raise AssertionError(
                f'precision mismatch was not rejected: {status} {mismatch}'
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    runtime_http_request,
                    base_url,
                    checkout,
                    f'cpp runtime dynamic batch row {index}',
                )
                for index in range(4)
            ]
            responses = [future.result(timeout=120) for future in futures]

        if any(len(list(response.get('results', []))) != 1
               for response in responses):
            raise AssertionError(f'bad dynamic batch split: {responses}')

        status, models = http_json('GET', f'{base_url}/v1/models')
        if (
            status != 200
            or models.get('active_provider') != encoded_provider
        ):
            raise AssertionError(
                'runtime model metadata returned stale active_provider: '
                f'models={models}, encoded={encoded}'
            )
        run_accelerator_dispatcher_probe(dsn, base_url)
        run_accelerator_resilience_probe(dsn, base_url)
        run_accelerator_build_cancellation_probe(
            dsn,
            base_url,
            model_path,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def accelerator_proxy(
    base_url: str,
    bind_port: int = 0,
    response_chunk_bytes: int = 0,
    response_chunk_delay_s: float = 0.0,
    request_delay_s: float = 0.0,
) -> tuple[
    http.server.ThreadingHTTPServer,
    threading.Thread,
    str,
]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get('Content-Length', '0'))
            payload = self.rfile.read(length)
            with self.server.request_lock:
                self.server.request_received_count += 1
            self.server.request_received_event.set()
            if self.server.request_delay_s > 0:
                time.sleep(float(self.server.request_delay_s))
            request = urllib.request.Request(
                f'{base_url}{self.path}',
                data=payload,
                method='POST',
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    status = response.status
                    body = response.read()
            except urllib.error.HTTPError as error:
                status = error.code
                body = error.read()
            except (urllib.error.URLError, TimeoutError, OSError):
                status = 503
                body = b'{"error":"upstream disconnected"}'
            with self.server.request_lock:
                self.server.request_count += 1
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                chunk_bytes = int(self.server.response_chunk_bytes)
                if 0 < chunk_bytes < len(body):
                    self.wfile.write(body[:chunk_bytes])
                    self.wfile.flush()
                    self.server.partial_response_event.set()
                    time.sleep(float(self.server.response_chunk_delay_s))
                    self.wfile.write(body[chunk_bytes:])
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            return

    server = http.server.ThreadingHTTPServer(
        ('127.0.0.1', bind_port),
        Handler,
    )
    server.request_count = 0
    server.request_received_count = 0
    server.request_lock = threading.Lock()
    server.request_received_event = threading.Event()
    server.partial_response_event = threading.Event()
    server.response_chunk_bytes = response_chunk_bytes
    server.response_chunk_delay_s = response_chunk_delay_s
    server.request_delay_s = request_delay_s
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f'http://{host}:{port}'


def run_accelerator_local_failover_backpressure_probe(
    dsn: str,
    base_url: str,
    model_path: Path,
    runtime_liveness_timeout_ms: int,
) -> None:
    if runtime_liveness_timeout_ms <= 0:
        raise ValueError('failover backpressure probe needs a finite timeout')
    request_delay_s = max(runtime_liveness_timeout_ms / 1000.0 + 3.0, 4.0)
    proxy, thread, proxy_url = accelerator_proxy(
        base_url,
        request_delay_s=request_delay_s,
    )
    config = json.dumps(
        [{'url': proxy_url, 'weight': 64, 'max_batch_size': 1}],
        separators=(',', ':'),
    )
    escaped_config = config.replace("'", "''")
    escaped_model_path = str(model_path).replace("'", "''")
    expected_remote_requests = 40

    try:
        with psycopg.connect(dsn, autocommit=True) as control:
            with control.cursor() as cursor:
                cursor.execute(
                    f"ALTER SYSTEM SET ii42.runtime_accelerators = "
                    f"'{escaped_config}'"
                )
                cursor.execute(
                    "ALTER SYSTEM SET "
                    "ii42.runtime_document_pipeline_depth = '64'"
                )
                cursor.execute('SELECT pg_reload_conf()')
                wait_accelerator_service_count(cursor, 1)
                cursor.execute('SELECT ii42_runtime_service_status()')
                before = dict(cursor.fetchone()[0])
                cursor.execute(
                    """
                    CREATE TABLE accelerator_backpressure_docs (
                        id int PRIMARY KEY,
                        body text NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO accelerator_backpressure_docs
                    SELECT value,
                           'local fallback backpressure document ' || value ||
                           ' ' || repeat('bounded recovery sentinel ', 8)
                    FROM pg_catalog.generate_series(1, 96) AS value
                    """
                )

        def build_index() -> None:
            with psycopg.connect(dsn, autocommit=True) as builder:
                with builder.cursor() as cursor:
                    cursor.execute(
                        f"""
                        CREATE INDEX accelerator_backpressure_docs_idx
                        ON accelerator_backpressure_docs
                        USING ii42 (body)
                        WITH (
                            sae = true,
                            model_path = '{escaped_model_path}'
                        )
                        """
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            build = executor.submit(build_index)
            deadline = time.monotonic() + 30.0
            received = 0
            while time.monotonic() < deadline:
                with proxy.request_lock:
                    received = int(proxy.request_received_count)
                if received >= expected_remote_requests:
                    break
                if build.done():
                    break
                time.sleep(0.01)
            if received < expected_remote_requests:
                raise AssertionError(
                    'probe did not exceed local response capacity: '
                    f'remote_requests={received}, '
                    f'needed={expected_remote_requests}'
                )
            build.result(timeout=120)

        with psycopg.connect(dsn, autocommit=True) as control:
            with control.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM ii42_query(
                        'accelerator_backpressure_docs_idx'::regclass,
                        'bounded recovery sentinel 42',
                        5
                    )
                    """
                )
                if int(cursor.fetchone()[0]) <= 0:
                    raise AssertionError('recovered index returned no hits')
                cursor.execute('SELECT ii42_runtime_service_status()')
                after = dict(cursor.fetchone()[0])
                cursor.execute(
                    'DROP TABLE accelerator_backpressure_docs CASCADE'
                )

        before_local = next(
            (
                entry
                for entry in list(before.get('accelerator_metrics', []))
                if isinstance(entry, dict) and entry.get('url') == 'local'
            ),
            {},
        )
        after_local = next(
            (
                entry
                for entry in list(after.get('accelerator_metrics', []))
                if isinstance(entry, dict) and entry.get('url') == 'local'
            ),
            {},
        )
        if int(after_local.get('successes', 0)) <= int(
            before_local.get('successes', 0)
        ):
            raise AssertionError(
                'expired remote requests did not complete through local '
                f'fallback: before={before}, after={after}'
            )
        for key in (
            'response_slots_in_use',
            'response_slots_ready',
            'response_slots_writing',
            'document_response_slots_in_use',
        ):
            if int(after.get(key, -1)) != 0:
                raise AssertionError(
                    f'local fallback leaked {key}: {after}'
                )
    finally:
        with psycopg.connect(dsn, autocommit=True) as control:
            with control.cursor() as cursor:
                cursor.execute('ALTER SYSTEM RESET ii42.runtime_accelerators')
                cursor.execute(
                    'ALTER SYSTEM RESET ii42.runtime_document_pipeline_depth'
                )
                cursor.execute('SELECT pg_reload_conf()')
                wait_accelerator_service_count(cursor, 0)
                cursor.execute(
                    'DROP TABLE IF EXISTS '
                    'accelerator_backpressure_docs CASCADE'
                )
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=5)


def run_accelerator_build_cancellation_probe(
    dsn: str,
    base_url: str,
    model_path: Path,
) -> None:
    proxy, thread, proxy_url = accelerator_proxy(
        base_url,
        response_chunk_bytes=64,
        response_chunk_delay_s=1.0,
    )
    config = json.dumps(
        [{'url': proxy_url, 'weight': 8, 'max_batch_size': 4}],
        separators=(',', ':'),
    )
    escaped_config = config.replace("'", "''")
    escaped_model_path = str(model_path).replace("'", "''")
    victim = psycopg.connect(dsn, autocommit=True)
    control = psycopg.connect(dsn, autocommit=True)

    try:
        with control.cursor() as cursor:
            cursor.execute(
                f"ALTER SYSTEM SET ii42.runtime_accelerators = "
                f"'{escaped_config}'"
            )
            cursor.execute('SELECT pg_reload_conf()')
            wait_accelerator_service_count(cursor, 1)
            cursor.execute(
                """
                CREATE TABLE accelerator_cancel_docs (
                    id int PRIMARY KEY,
                    body text NOT NULL
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO accelerator_cancel_docs
                SELECT value,
                       'accelerator cancellation document ' || value || ' ' ||
                       repeat('partial response ownership sentinel ', 8)
                FROM pg_catalog.generate_series(1, 2048) AS value
                """
            )
        with victim.cursor() as cursor:
            cursor.execute('SELECT pg_backend_pid()')
            victim_pid = int(cursor.fetchone()[0])

        def build_index() -> None:
            with victim.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE INDEX accelerator_cancel_docs_idx
                    ON accelerator_cancel_docs
                    USING ii42 (body)
                    WITH (
                        sae = true,
                        model_path = '{escaped_model_path}'
                    )
                    """
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            build = executor.submit(build_index)
            if not proxy.partial_response_event.wait(timeout=30):
                raise AssertionError(
                    'SAE build did not receive a partial accelerator response'
                )
            time.sleep(0.1)
            with control.cursor() as cursor:
                cursor.execute('SELECT pg_cancel_backend(%s)', (victim_pid,))
                if cursor.fetchone()[0] is not True:
                    raise AssertionError('failed to cancel partial SAE build')
            try:
                build.result(timeout=30)
            except psycopg.errors.QueryCanceled:
                pass
            except psycopg.Error as exc:
                raise AssertionError(
                    'SAE build cancellation replaced QueryCanceled with '
                    f'{exc.sqlstate}: {exc}'
                ) from exc
            else:
                raise AssertionError('partial SAE build ignored cancellation')

        with victim.cursor() as cursor:
            cursor.execute('SELECT 1')
            if cursor.fetchone()[0] != 1:
                raise AssertionError('canceled build backend is unusable')
        with control.cursor() as cursor:
            cursor.execute(
                "SELECT pg_catalog.to_regclass("
                "'accelerator_cancel_docs_idx')"
            )
            if cursor.fetchone()[0] is not None:
                raise AssertionError('canceled CREATE INDEX left an artifact')
            cursor.execute('DROP TABLE accelerator_cancel_docs CASCADE')
    finally:
        victim.close()
        with control.cursor() as cursor:
            cursor.execute('ALTER SYSTEM RESET ii42.runtime_accelerators')
            cursor.execute('SELECT pg_reload_conf()')
            wait_accelerator_service_count(cursor, 0)
        control.close()
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=5)


def wait_accelerator_service_count(
    cursor: psycopg.Cursor[object],
    expected: int,
) -> dict[str, object]:
    deadline = time.monotonic() + 15
    status: dict[str, object] = {}
    while time.monotonic() < deadline:
        cursor.execute('SELECT ii42_runtime_service_status()')
        status = dict(cursor.fetchone()[0])
        if int(status.get('accelerator_service_count', -1)) == expected:
            return status
        time.sleep(0.1)
    raise AssertionError(
        f'accelerator service count did not become {expected}: {status}'
    )


def run_accelerator_dispatcher_probe(dsn: str, base_url: str) -> None:
    def metric_successes(
        status: dict[str, object],
        url: str,
    ) -> int:
        metrics = status.get('accelerator_metrics')
        if not isinstance(metrics, list):
            return 0
        for entry in metrics:
            if isinstance(entry, dict) and entry.get('url') == url:
                return int(entry.get('successes', 0))
        return 0

    proxy, thread, proxy_url = accelerator_proxy(base_url)
    config = json.dumps(
        [{'url': proxy_url, 'weight': 16}],
        separators=(',', ':'),
    )
    escaped_config = config.replace("'", "''")
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"ALTER SYSTEM SET ii42.runtime_accelerators = "
                    f"'{escaped_config}'"
                )
                cursor.execute('SELECT pg_reload_conf()')
                status = wait_accelerator_service_count(cursor, 1)
                if status.get('accelerator_services') != [
                    {'url': proxy_url, 'weight': 16},
                ]:
                    raise AssertionError(
                        f'accelerator config was not visible: {status}'
                    )
                cursor.execute('SELECT ii42_runtime_service_status()')
                before_status = dict(cursor.fetchone()[0])
                cursor.execute(
                    """
                    CREATE TABLE accelerator_docs (
                        id int PRIMARY KEY,
                        body text NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO accelerator_docs
                    SELECT value,
                           'accelerator dispatcher document ' || value
                    FROM pg_catalog.generate_series(1, 96) AS value
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX accelerator_docs_body_idx
                    ON accelerator_docs
                    USING ii42 (body)
                    WITH (
                        sae = true,
                        consistency = eventual,
                        auto_preload = 1
                    )
                    """
                )
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM ii42_query(
                        'accelerator_docs_body_idx'::regclass,
                        'accelerator dispatcher document 42',
                        5
                    )
                    """
                )
                if int(cursor.fetchone()[0]) <= 0:
                    raise AssertionError('accelerator index returned no hits')
                cursor.execute('SELECT ii42_runtime_service_status()')
                after_status = dict(cursor.fetchone()[0])
                remote_successes_before_maintenance = metric_successes(
                    after_status,
                    proxy_url,
                )
                cursor.execute(
                    """
                    INSERT INTO accelerator_docs
                    SELECT value,
                           'accelerator maintenance document ' || value
                    FROM pg_catalog.generate_series(97, 192) AS value
                    """
                )
                cursor.execute('SELECT ii42_index_touch_maintenance()')

                deadline = time.monotonic() + 30.0
                maintenance_status: dict[str, object] = {}
                maintenance_runtime_status: dict[str, object] = {}
                while time.monotonic() < deadline:
                    cursor.execute(
                        "SELECT ii42_index_status("
                        "'accelerator_docs_body_idx'::regclass)"
                    )
                    maintenance_status = dict(cursor.fetchone()[0])
                    cursor.execute('SELECT ii42_runtime_service_status()')
                    maintenance_runtime_status = dict(cursor.fetchone()[0])
                    completion = maintenance_status.get('generation', {})
                    if not isinstance(completion, dict):
                        completion = {}
                    semantic_completion = completion.get('delta', {})
                    if not isinstance(semantic_completion, dict):
                        semantic_completion = {}
                    semantic_completion = semantic_completion.get(
                        'semantic_completion',
                        {},
                    )
                    accelerator = completion.get(
                        'semantic_accelerator',
                        {},
                    )
                    if (
                        isinstance(semantic_completion, dict)
                        and semantic_completion.get('converged') is True
                        and isinstance(accelerator, dict)
                        and accelerator.get('eligible') is True
                        and maintenance_status.get('performance_ready') is True
                        and metric_successes(
                            maintenance_runtime_status,
                            proxy_url,
                        ) > remote_successes_before_maintenance
                    ):
                        break
                    time.sleep(0.05)
                else:
                    raise AssertionError(
                        'background semantic maintenance did not converge '
                        'through the remote accelerator: '
                        f'index={maintenance_status}, '
                        f'runtime={maintenance_runtime_status}'
                    )

                if maintenance_status.get('performance_ready') is not True:
                    raise AssertionError(
                        'semantic accelerator publication did not qualify '
                        f'the query fast path: {maintenance_status}'
                    )
        with proxy.request_lock:
            request_count = int(proxy.request_count)
        if request_count <= 0:
            raise AssertionError(
                'runtime accelerator dispatcher did not call the HTTP service'
            )
        if int(after_status.get('document_worker_limit', 0)) <= 0:
            raise AssertionError(
                'runtime accelerator dispatcher disabled the local '
                f'document lane: before={before_status}, '
                f'after={after_status}'
            )
        metrics = after_status.get('accelerator_metrics')
        if not isinstance(metrics, list):
            raise AssertionError(f'accelerator metrics missing: {after_status}')
        metrics_by_url = {
            entry.get('url'): entry
            for entry in metrics
            if isinstance(entry, dict)
        }
        for required_url in ('local', proxy_url):
            if required_url not in metrics_by_url:
                raise AssertionError(
                    f'accelerator metrics did not include {required_url}: '
                    f'{after_status}'
                )
            if int(metrics_by_url[required_url].get('successes', 0)) <= 0:
                raise AssertionError(
                    f'accelerator metrics did not record success for '
                    f'{required_url}: {after_status}'
                )
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute('ALTER SYSTEM RESET ii42.runtime_accelerators')
                cursor.execute('SELECT pg_reload_conf()')
                wait_accelerator_service_count(cursor, 0)
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=5)


def run_accelerator_resilience_probe(dsn: str, base_url: str) -> None:
    unavailable_port = free_tcp_port()
    unavailable_url = f'http://127.0.0.1:{unavailable_port}'
    live_proxy, live_thread, live_url = accelerator_proxy(base_url)
    recovered_proxy: http.server.ThreadingHTTPServer | None = None
    recovered_thread: threading.Thread | None = None
    config = json.dumps(
        [{'url': unavailable_url}, {'url': live_url}],
        separators=(',', ':'),
    )
    escaped_config = config.replace("'", "''")
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"ALTER SYSTEM SET ii42.runtime_accelerators = "
                    f"'{escaped_config}'"
                )
                cursor.execute('SELECT pg_reload_conf()')
                status = wait_accelerator_service_count(cursor, 2)
                if status.get('accelerator_services') != [
                    {'url': unavailable_url},
                    {'url': live_url},
                ]:
                    raise AssertionError(
                        f'accelerator config was not visible: {status}'
                    )
                cursor.execute(
                    """
                    CREATE TABLE accelerator_failover_docs (
                        id int PRIMARY KEY,
                        body text NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO accelerator_failover_docs
                    SELECT value,
                           'accelerator failover document ' || value
                    FROM pg_catalog.generate_series(1, 64) AS value
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX accelerator_failover_docs_body_idx
                    ON accelerator_failover_docs
                    USING ii42 (body)
                    WITH (
                        sae = true,
                        consistency = eventual
                    )
                    """
                )
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM ii42_query(
                        'accelerator_failover_docs_body_idx'::regclass,
                        'accelerator failover document 42',
                        5
                    )
                    """
                )
                if int(cursor.fetchone()[0]) <= 0:
                    raise AssertionError('failover index returned no hits')
        with live_proxy.request_lock:
            live_count = int(live_proxy.request_count)
        if live_count <= 0:
            raise AssertionError(
                'runtime accelerator failover did not use live service'
            )

        recovered_proxy, recovered_thread, _ = accelerator_proxy(
            base_url,
            unavailable_port,
        )
        time.sleep(1.5)
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE accelerator_rejoin_docs (
                        id int PRIMARY KEY,
                        body text NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO accelerator_rejoin_docs
                    SELECT value,
                           'accelerator rejoin document ' || value
                    FROM pg_catalog.generate_series(1, 96) AS value
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX accelerator_rejoin_docs_body_idx
                    ON accelerator_rejoin_docs
                    USING ii42 (body)
                    WITH (
                        sae = true,
                        consistency = eventual
                    )
                    """
                )
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM ii42_query(
                        'accelerator_rejoin_docs_body_idx'::regclass,
                        'accelerator rejoin document 42',
                        5
                    )
                    """
                )
                if int(cursor.fetchone()[0]) <= 0:
                    raise AssertionError('rejoin index returned no hits')
        with recovered_proxy.request_lock:
            recovered_count = int(recovered_proxy.request_count)
        if recovered_count <= 0:
            raise AssertionError(
                'recovered runtime accelerator did not rejoin dispatch'
            )
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute('ALTER SYSTEM RESET ii42.runtime_accelerators')
                cursor.execute('SELECT pg_reload_conf()')
                wait_accelerator_service_count(cursor, 0)
        if recovered_proxy is not None:
            recovered_proxy.shutdown()
            recovered_proxy.server_close()
        if recovered_thread is not None:
            recovered_thread.join(timeout=5)
        live_proxy.shutdown()
        live_proxy.server_close()
        live_thread.join(timeout=5)


def runtime_identity(model_path: Path) -> tuple[str, str, str]:
    manifest = json.loads(
        (model_path / 'manifest.json').read_text(encoding='utf-8')
    )
    if (
        manifest.get('schema_version') != 1
        or manifest.get('api_version') != 'evoke_model_v1'
        or manifest.get('runtime_abi') != 'evoke_p2_unified_text_atoms_v2'
    ):
        raise ValueError(
            'runtime service smoke requires the current Evoke model contract'
        )
    artifacts = manifest.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError('P2 manifest artifacts are missing')

    def artifact_json(name: str) -> dict[str, object]:
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict) or not isinstance(
            artifact.get('path'),
            str,
        ):
            raise ValueError(f'P2 manifest artifact is missing: {name}')
        value = json.loads(
            (model_path / str(artifact['path'])).read_text(encoding='utf-8')
        )
        if not isinstance(value, dict):
            raise ValueError(f'P2 manifest artifact is invalid: {name}')
        return value

    atom_space = artifact_json('atom_space').get('atom_space')
    scoring_profile = artifact_json('scoring_profile').get('profile')
    model_id = manifest.get('model_id')
    if not all(
        isinstance(value, str) and value
        for value in (model_id, atom_space, scoring_profile)
    ):
        raise ValueError('P2 runtime identity is incomplete')
    return str(model_id), str(atom_space), str(scoring_profile)


def runtime_service_sql(
    model_path: Path,
    worker_count: int,
    model_path_source: str = 'environment',
    runtime_liveness_timeout_ms: int = 300_000,
) -> str:
    escaped_path = str(model_path).replace("'", "''")
    escaped_model_path_source = model_path_source.replace("'", "''")
    model_id, atom_space, scoring_profile = runtime_identity(model_path)
    escaped_model_id = model_id.replace("'", "''")
    escaped_atom_space = atom_space.replace("'", "''")
    escaped_scoring_profile = scoring_profile.replace("'", "''")
    return f'''
    CREATE EXTENSION ii42;
    CREATE TABLE docs (
        id text PRIMARY KEY,
        body text NOT NULL,
        category text NOT NULL DEFAULT 'keep'
    );
    INSERT INTO docs (id, body, category) VALUES
        ('doc-a', 'alpha cuda', 'keep'),
        ('doc-b', 'semantic gpu', 'keep'),
        ('doc-c', 'optimization graph', 'drop');
    CREATE TABLE allowed_docs (id text PRIMARY KEY);
    INSERT INTO allowed_docs (id) VALUES ('doc-a'), ('doc-b');
    CREATE INDEX docs_category_idx ON docs (category);

    CREATE INDEX docs_body_idx
    ON docs
    USING ii42 (body)
    WITH (
        sae = true,
        consistency = eventual
    )
    WHERE btrim(coalesce(body, '')) <> '';

    CREATE ROLE ii42_app_user;
    GRANT USAGE ON SCHEMA public TO ii42_app_user;
    GRANT SELECT ON TABLE docs TO ii42_app_user;

    DO $$
    DECLARE
        service_status jsonb;
        options jsonb;
        status jsonb;
        generation jsonb;
        shared_encoding jsonb;
        batch_encoding jsonb;
        wrapper_encoding jsonb;
        hit_count int;
        hit_ids text[];
        function_ordered_ids text[];
        planner_ordered_ids text[];
        function_ordered_scores float8[];
        planner_ordered_scores float8[];
        field_function_ordered_ids text[];
        field_planner_ordered_ids text[];
        filtered_function_ordered_ids text[];
        filtered_planner_ordered_ids text[];
        empty_filtered_count int;
        planner_probe_plan json;
        function_probe_trace jsonb;
        planner_probe_trace jsonb;
        session_cache_context text;
    BEGIN
        SELECT context
        INTO session_cache_context
        FROM pg_settings
        WHERE name = 'ii42.onnxruntime_session_cache_size';
        IF session_cache_context <> 'postmaster' THEN
            RAISE EXCEPTION
                'runtime session cache must be restart-only: %',
                session_cache_context;
        END IF;

        service_status := ii42_runtime_service_status();
        IF (service_status->>'shared_memory_available')::boolean
                IS DISTINCT FROM true
            OR (service_status->>'worker_ready')::boolean
                IS DISTINCT FROM true
            OR service_status->>'queue_policy'
                <> 'bounded_affinity_worker_pool'
            OR service_status->>'queue_timeout_policy' <> 'none'
            OR (service_status->>'queue_capacity')::int <> 32
            OR (service_status->>'response_capacity')::int <> 32
            OR service_status->>'control_database' <> 'template1'
            OR (service_status->>'runtime_liveness_timeout_ms')::int
                <> {runtime_liveness_timeout_ms}
            OR (service_status->>'response_slots_in_use')::int <> 0
            OR (service_status->>'response_slots_ready')::int <> 0
            OR (service_status->>'response_slots_writing')::int <> 0
            OR (
                service_status->>'document_response_slots_in_use'
            )::int <> 0
            OR (service_status->>'document_queue_limit')::int
                <> (service_status->>'document_worker_limit')::int
            OR (service_status->>'document_worker_limit')::int
                <> {worker_count - 1}
            OR jsonb_array_length(
                service_status->'accelerator_services'
            ) <> 0
            OR (service_status->>'accelerator_service_count')::int <> 0
            OR (service_status->>'reserved_query_worker_slots')::int <> 1
            OR (
                service_status->>'query_execution_lane_reserved'
            )::boolean IS DISTINCT FROM true
            OR jsonb_array_length(service_status->'workers')
                <> {worker_count} THEN
            RAISE EXCEPTION 'runtime service is not ready: %',
                service_status;
        END IF;
        IF EXISTS (
            SELECT 1
            FROM jsonb_array_elements(service_status->'workers') AS worker
            WHERE NOT (
                worker ? 'starts'
                AND worker ? 'recoveries'
                AND worker ? 'successes'
                AND worker ? 'failures'
                AND worker ? 'runtime_total_us'
                AND worker ? 'runtime_max_us'
                AND worker ? 'runtime_terminations'
                AND worker ? 'runtime_liveness_timeouts'
                AND worker ? 'terminate_requested'
                AND worker ? 'processing_started_at'
                AND worker ? 'last_progress_at'
            )
        ) THEN
            RAISE EXCEPTION 'runtime worker counters are incomplete: %',
                service_status;
        END IF;
        IF NOT (service_status ? 'canceled_requests')
            OR NOT (service_status ? 'orphan_responses')
            OR NOT (service_status ? 'worker_recoveries')
            OR NOT (service_status ? 'runtime_runs')
            OR NOT (service_status ? 'encoded_texts')
            OR NOT (service_status ? 'batch_successes')
            OR NOT (service_status ? 'runtime_terminations')
            OR NOT (service_status ? 'runtime_liveness_timeouts')
            OR NOT (service_status ? 'last_batch_size')
            OR NOT (service_status ? 'max_observed_batch_size') THEN
            RAISE EXCEPTION 'runtime recovery counters missing: %',
                service_status;
        END IF;
        IF (service_status->>'encoded_texts')::int8 <> 3
            OR (service_status->>'runtime_runs')::int8 <> 1
            OR (service_status->>'batch_successes')::int8 <> 1
            OR (service_status->>'last_batch_size')::int <> 3 THEN
            RAISE EXCEPTION 'index build did not use explicit batching: %',
                service_status;
        END IF;

        options := ii42_index_options('docs_body_idx'::regclass);
        IF options->>'index_type' <> 'semantic'
            OR options->>'model_path_source'
                <> '{escaped_model_path_source}'
            OR options->>'model_path' <> '{escaped_path}'
            OR options->>'model' <> '{escaped_model_id}'
            OR options->>'atom_space' <> '{escaped_atom_space}'
            OR options->>'scoring_profile' <> '{escaped_scoring_profile}'
            OR options->>'runtime_precision' <> 'fp16'
            OR options->>'payload_owner' <> 'index_relation'
            OR options->>'lifecycle' <> 'postgresql_index' THEN
            RAISE EXCEPTION 'unified SAE options mismatch: %', options;
        END IF;

        EXECUTE format(
            'ALTER INDEX docs_body_idx SET (model_path = %L)',
            '{escaped_path}'
        );
        options := ii42_index_options('docs_body_idx'::regclass);
        IF options->>'model_path_source' <> 'index'
            OR options->>'model_path' <> '{escaped_path}' THEN
            RAISE EXCEPTION
                'index model checkout did not override the default: %',
                options;
        END IF;
        EXECUTE 'ALTER INDEX docs_body_idx RESET (model_path)';
        options := ii42_index_options('docs_body_idx'::regclass);
        IF options->>'model_path_source'
                <> '{escaped_model_path_source}'
            OR options->>'model_path' <> '{escaped_path}' THEN
            RAISE EXCEPTION
                'default model checkout was not restored: %', options;
        END IF;

        status := ii42_index_status('docs_body_idx'::regclass);
        generation := status->'generation';
        IF status->>'api_version' <> 'ii42_index_v1'
            OR (status->>'query_ready')::boolean IS DISTINCT FROM true
            OR status->>'blocker' <> 'none'
            OR (generation->>'atomic')::boolean IS DISTINCT FROM true
            OR (generation->>'valid')::boolean IS DISTINCT FROM true
            OR NULLIF(generation->>'generation_id', '') IS NULL
            OR generation#>>'{{primary,role}}' <> 'unified_posting'
            OR (generation#>>'{{primary,payload_bytes}}')::int8 <= 0
            OR (generation#>>'{{primary,physical_blocks}}')::int <= 0
            OR generation#>>'{{posting,role}}' <> 'unified_posting'
            OR (generation#>>'{{posting,present}}')::boolean
                IS DISTINCT FROM true
            OR (generation#>>'{{posting,active}}')::boolean
                IS DISTINCT FROM true
            OR (generation#>>'{{posting,bytes}}')::int8 <>
                (generation#>>'{{primary,payload_bytes}}')::int8
            OR (generation#>>'{{posting,record_count}}')::int <> 3 THEN
            RAISE EXCEPTION 'unified generation mismatch: %', status;
        END IF;

        SELECT count(*), array_agg(docs.id ORDER BY docs.id)
        INTO hit_count, hit_ids
        FROM ii42_query(
            'docs_body_idx'::regclass,
            'alpha semantic optimization',
            3
        ) AS hit
        JOIN docs ON docs.ctid = hit.ctid;
        IF hit_count <> 3
            OR hit_ids IS DISTINCT FROM ARRAY['doc-a', 'doc-b', 'doc-c']
        THEN
            RAISE EXCEPTION
                'runtime product query mismatch: count=%, ids=%',
                hit_count,
                hit_ids;
        END IF;

        PERFORM pg_catalog.set_config(
            'ii42.enable_planner_native',
            'on',
            true
        );
        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false, VERBOSE true)
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
        $plan$ INTO planner_probe_plan;
        IF planner_probe_plan::text NOT LIKE '%Custom Scan%'
            OR planner_probe_plan::text NOT LIKE '%II42 Search%'
        THEN
            RAISE EXCEPTION
                'planner-native probe did not select II42 CustomScan: %',
                planner_probe_plan;
        END IF;
        SELECT array_agg(docs.id ORDER BY hit.rank)
        INTO function_ordered_ids
        FROM ii42_query(
            'docs_body_idx'::regclass,
            'alpha semantic optimization',
            3
        ) WITH ORDINALITY AS hit(ctid, doc_id, score, rank)
        JOIN docs ON docs.ctid = hit.ctid;
        SELECT array_agg(ranked.id)
        INTO planner_ordered_ids
        FROM (
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
        ) AS ranked;
        IF planner_ordered_ids IS DISTINCT FROM function_ordered_ids THEN
            RAISE EXCEPTION
                'planner-native TID order mismatch: % versus %',
                planner_ordered_ids,
                function_ordered_ids;
        END IF;
        SELECT array_agg(hit.score ORDER BY hit.rank)
        INTO function_ordered_scores
        FROM ii42_query(
            'docs_body_idx'::regclass,
            'alpha semantic optimization',
            3
        ) WITH ORDINALITY AS hit(ctid, doc_id, score, rank);
        SELECT array_agg(ranked.score ORDER BY ranked.score DESC)
        INTO planner_ordered_scores
        FROM (
            SELECT ii42_query(
                       'docs_body_idx'::regclass,
                       'alpha semantic optimization'
                   ) AS score
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
        ) AS ranked;
        IF planner_ordered_scores IS DISTINCT FROM function_ordered_scores THEN
            RAISE EXCEPTION
                'planner-native score mismatch: % versus %',
                planner_ordered_scores,
                function_ordered_scores;
        END IF;
        IF (
            SELECT count(*)
            FROM (
                SELECT aliased_docs.id
                FROM docs AS aliased_docs
                ORDER BY ii42_query(
                    'docs_body_idx'::regclass,
                    'alpha semantic optimization'
                ) DESC
                LIMIT 0
            ) AS no_rows
        ) <> 0 THEN
            RAISE EXCEPTION 'planner-native LIMIT 0 returned rows';
        END IF;
        IF (
            SELECT count(*)
            FROM (
                SELECT ctid, tableoid
                FROM docs
                ORDER BY ii42_query(
                    'docs_body_idx'::regclass,
                    'alpha semantic optimization'
                ) DESC
                LIMIT 3
            ) AS ranked
            WHERE ranked.ctid IS NULL
               OR ranked.tableoid <> 'docs'::regclass
        ) <> 0 THEN
            RAISE EXCEPTION
                'planner-native scan did not preserve system identity';
        END IF;

        EXECUTE 'CREATE INDEX docs_bm25_idx ON docs USING ii42 (body)';
        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false)
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_bm25_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
        $plan$ INTO planner_probe_plan;
        IF planner_probe_plan::text LIKE '%II42 Search%' THEN
            RAISE EXCEPTION
                'planner-native semantic path captured a BM25 index: %',
                planner_probe_plan;
        END IF;
        EXECUTE 'DROP INDEX docs_bm25_idx';

        EXECUTE $ddl$
            CREATE INDEX docs_field_idx
            ON docs
            USING ii42 (id, body)
            WITH (
                sae = true,
                field_aware = true,
                consistency = eventual
            )
        $ddl$;
        SELECT array_agg(docs.id ORDER BY hit.rank)
        INTO field_function_ordered_ids
        FROM ii42_query(
            'docs_field_idx'::regclass,
            'alpha semantic optimization',
            3
        ) WITH ORDINALITY AS hit(ctid, doc_id, score, rank)
        JOIN docs ON docs.ctid = hit.ctid;
        SELECT array_agg(ranked.id)
        INTO field_planner_ordered_ids
        FROM (
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_field_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
        ) AS ranked;
        IF field_planner_ordered_ids IS DISTINCT FROM
                field_function_ordered_ids THEN
            RAISE EXCEPTION
                'field-aware planner-native order mismatch: % versus %',
                field_planner_ordered_ids,
                field_function_ordered_ids;
        END IF;
        SELECT array_agg(docs.id ORDER BY hit.rank)
        INTO field_function_ordered_ids
        FROM ii42_query(
            'docs_field_idx'::regclass,
            'alpha semantic optimization',
            ARRAY['id', 'body']::text[],
            ARRAY[0.25, 1.75]::real[],
            3
        ) WITH ORDINALITY AS hit(ctid, doc_id, score, rank)
        JOIN docs ON docs.ctid = hit.ctid;
        SELECT array_agg(ranked.id)
        INTO field_planner_ordered_ids
        FROM (
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_field_idx'::regclass,
                'alpha semantic optimization',
                ARRAY['id', 'body']::text[],
                ARRAY[0.25, 1.75]::real[]
            ) DESC
            LIMIT 3
        ) AS ranked;
        IF field_planner_ordered_ids IS DISTINCT FROM
                field_function_ordered_ids THEN
            RAISE EXCEPTION
                'weighted field planner-native order mismatch: % versus %',
                field_planner_ordered_ids,
                field_function_ordered_ids;
        END IF;
        EXECUTE 'DROP INDEX docs_field_idx';

        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false)
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) ASC
            LIMIT 3
        $plan$ INTO planner_probe_plan;
        IF planner_probe_plan::text LIKE '%II42 Search%' THEN
            RAISE EXCEPTION
                'planner-native probe accepted ascending order: %',
                planner_probe_plan;
        END IF;
        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false)
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC,
            id
            LIMIT 3
        $plan$ INTO planner_probe_plan;
        IF planner_probe_plan::text LIKE '%II42 Search%' THEN
            RAISE EXCEPTION
                'planner-native probe accepted a secondary sort key: %',
                planner_probe_plan;
        END IF;
        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false)
            SELECT docs.id
            FROM docs
            JOIN allowed_docs AS allowed ON allowed.id = docs.id
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
        $plan$ INTO planner_probe_plan;
        IF planner_probe_plan::text LIKE '%II42 Search%' THEN
            RAISE EXCEPTION
                'planner-native probe accepted a join: %',
                planner_probe_plan;
        END IF;
        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false)
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                body
            ) DESC
            LIMIT 3
        $plan$ INTO planner_probe_plan;
        IF planner_probe_plan::text LIKE '%II42 Search%' THEN
            RAISE EXCEPTION
                'planner-native probe accepted a row-dependent query: %',
                planner_probe_plan;
        END IF;
        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false)
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            FETCH FIRST 3 ROWS WITH TIES
        $plan$ INTO planner_probe_plan;
        IF planner_probe_plan::text LIKE '%II42 Search%' THEN
            RAISE EXCEPTION
                'planner-native probe accepted WITH TIES: %',
                planner_probe_plan;
        END IF;
        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false)
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
            FOR UPDATE
        $plan$ INTO planner_probe_plan;
        IF planner_probe_plan::text LIKE '%II42 Search%' THEN
            RAISE EXCEPTION
                'planner-native probe accepted row locking: %',
                planner_probe_plan;
        END IF;
        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false)
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
        $plan$ INTO planner_probe_plan;
        IF planner_probe_plan::text LIKE '%II42 Search%' THEN
            RAISE EXCEPTION
                'planner-native probe accepted an unbounded query: %',
                planner_probe_plan;
        END IF;
        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false)
            SELECT id
            FROM docs
            WHERE id = 'doc-a'
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
        $plan$ INTO planner_probe_plan;
        IF planner_probe_plan::text NOT LIKE '%II42 Search%' THEN
            RAISE EXCEPTION
                'planner-native filtered query missed II42 CustomScan: %',
                planner_probe_plan;
        END IF;
        IF planner_probe_plan::text NOT LIKE '%II42 Index%'
            OR planner_probe_plan::text NOT LIKE '%docs_body_idx%'
            OR planner_probe_plan::text NOT LIKE '%Filtered%'
        THEN
            RAISE EXCEPTION
                'planner-native EXPLAIN is missing route identity: %',
                planner_probe_plan;
        END IF;
        SELECT array_agg(docs.id ORDER BY hit.rank)
        INTO filtered_function_ordered_ids
        FROM ii42_query(
            'docs_body_idx'::regclass,
            'alpha semantic optimization',
            ARRAY(
                SELECT ctid
                FROM docs
                WHERE category = 'keep'
                ORDER BY ctid
            ),
            3
        ) WITH ORDINALITY AS hit(ctid, doc_id, score, rank)
        JOIN docs ON docs.ctid = hit.ctid;
        function_probe_trace := ii42_query_trace_internal();
        EXECUTE $plan$
            EXPLAIN (FORMAT JSON, COSTS false, VERBOSE true)
            SELECT id
            FROM docs
            WHERE category = 'keep'
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
        $plan$ INTO planner_probe_plan;
        SELECT array_agg(ranked.id)
        INTO filtered_planner_ordered_ids
        FROM (
            SELECT id
            FROM docs
            WHERE category = 'keep'
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
        ) AS ranked;
        planner_probe_trace := ii42_query_trace_internal();
        IF filtered_planner_ordered_ids IS DISTINCT FROM
                filtered_function_ordered_ids THEN
            RAISE EXCEPTION
                'planner-native filtered order mismatch: % versus %, plan=%, function_trace=%, planner_trace=%',
                filtered_planner_ordered_ids,
                filtered_function_ordered_ids,
                planner_probe_plan,
                function_probe_trace,
                planner_probe_trace;
        END IF;
        SELECT count(*)
        INTO empty_filtered_count
        FROM (
            SELECT id
            FROM docs
            WHERE category = 'missing'
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization'
            ) DESC
            LIMIT 3
        ) AS ranked;
        IF empty_filtered_count <> 0 THEN
            RAISE EXCEPTION
                'planner-native empty filter returned % rows',
                empty_filtered_count;
        END IF;

        EXECUTE $prepare$
            PREPARE ii42_planner_parameter_probe(text, bigint) AS
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                $1
            ) DESC
            LIMIT $2
        $prepare$;
        EXECUTE $execute$
            EXECUTE ii42_planner_parameter_probe(
                'alpha semantic optimization',
                3
            )
        $execute$;
        PERFORM pg_catalog.set_config(
            'plan_cache_mode',
            'force_generic_plan',
            true
        );
        EXECUTE $execute$
            EXECUTE ii42_planner_parameter_probe(
                'alpha semantic optimization',
                2
            )
        $execute$;
        PERFORM pg_catalog.set_config(
            'plan_cache_mode',
            'auto',
            true
        );
        EXECUTE 'DEALLOCATE ii42_planner_parameter_probe';
        EXECUTE $prepare$
            PREPARE ii42_planner_filtered_probe(text, text, bigint) AS
            SELECT id
            FROM docs
            WHERE category = $2
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                $1
            ) DESC
            LIMIT $3
        $prepare$;
        EXECUTE $execute$
            EXECUTE ii42_planner_filtered_probe(
                'alpha semantic optimization',
                'keep',
                3
            )
        $execute$;
        EXECUTE 'DEALLOCATE ii42_planner_filtered_probe';
        PERFORM pg_catalog.set_config(
            'ii42.enable_planner_native',
            'off',
            true
        );

        batch_encoding := ii42_runtime_service_query_atoms_batch(
            '{escaped_path}',
            'fp16',
            ARRAY[
                'runtime parity sentinel',
                'runtime second batch row'
            ]
        );
        wrapper_encoding := ii42_runtime_service_atoms_batch_internal(
            '{escaped_path}',
            'fp16',
            'query',
            ARRAY[
                'runtime parity sentinel',
                'runtime second batch row'
            ]
        );
        shared_encoding := ii42_runtime_service_query_atoms(
            '{escaped_path}',
            'fp16',
            'runtime parity sentinel'
        );
        IF batch_encoding->>'runtime_precision' <> 'fp16'
            OR wrapper_encoding->>'runtime_precision' <> 'fp16'
            OR shared_encoding->>'runtime_precision' <> 'fp16' THEN
            RAISE EXCEPTION
                'runtime precision was not preserved: batch=%, wrapper=%, single=%',
                batch_encoding,
                wrapper_encoding,
                shared_encoding;
        END IF;
        IF batch_encoding#>'{{results,0,atoms}}' IS DISTINCT FROM
                shared_encoding->'atoms'
            OR batch_encoding#>'{{results,0,weights}}' IS DISTINCT FROM
                shared_encoding->'weights' THEN
            RAISE EXCEPTION
                'single/batch runtime parity mismatch: batch=%, single=%',
                batch_encoding,
                shared_encoding;
        END IF;
        IF wrapper_encoding#>'{{results,0,atoms}}' IS DISTINCT FROM
                batch_encoding#>'{{results,0,atoms}}'
            OR wrapper_encoding#>'{{results,0,weights}}' IS DISTINCT FROM
                batch_encoding#>'{{results,0,weights}}' THEN
            RAISE EXCEPTION
                'shared wrapper runtime parity mismatch: wrapper=%, batch=%',
                wrapper_encoding,
                batch_encoding;
        END IF;

        service_status := ii42_runtime_service_status();
        IF service_status->>'provider' <> 'auto'
            OR service_status->>'active_provider'
                NOT IN ('cpu', 'cuda', 'coreml', 'tensorrt')
            OR service_status->>'runtime_precision' <> 'fp16'
            OR (service_status->>'last_batch_size')::int <> 1
            OR (service_status->>'batch_successes')::int8 < 2 THEN
            RAISE EXCEPTION 'runtime service provider mismatch: %',
                service_status;
        END IF;
    END;
    $$;

    '''


def run_concurrent_runtime_queries(socket_dir: Path, port: int) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    client_count = 64

    def query_once(worker_id: int) -> int:
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                query = (
                    'alpha semantic optimization '
                    f'concurrent runtime worker {worker_id}'
                )
                if worker_id % 2 == 0:
                    cur.execute(
                        """
                        SELECT count(*)
                        FROM (
                            SELECT id
                            FROM docs
                            ORDER BY ii42_query(
                                'docs_body_idx'::regclass,
                                %s
                            ) DESC
                            LIMIT 3
                        ) AS ranked
                        """,
                        (query,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT count(*)
                        FROM ii42_query(
                            'docs_body_idx'::regclass,
                            %s,
                            3
                        )
                        """,
                        (query,),
                    )
                row = cur.fetchone()
        if row is None:
            raise AssertionError('concurrent query returned no row')
        return int(row[0])

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=client_count,
    ) as executor:
        futures = [
            executor.submit(query_once, worker_id)
            for worker_id in range(client_count)
        ]
        counts = [future.result() for future in futures]

    if counts != [3] * client_count:
        raise AssertionError(f'unexpected concurrent row counts: {counts}')

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT ii42_runtime_service_status()')
            status = cur.fetchone()[0]
    if status.get('queue_policy') != 'bounded_affinity_worker_pool':
        raise AssertionError(f'worker-pool queue policy missing: {status}')
    if status.get('queue_timeout_policy') != 'none':
        raise AssertionError(f'queue timeout policy changed: {status}')
    if int(status.get('queue_capacity', 0)) != 32:
        raise AssertionError(f'worker queue capacity mismatch: {status}')
    if int(status.get('response_capacity', 0)) != 32:
        raise AssertionError(f'response slot capacity mismatch: {status}')
    if int(status.get('queue_capacity', 0)) != int(
        status.get('response_capacity', -1)
    ):
        raise AssertionError(
            f'queue/response capacity contract diverged: {status}'
        )
    if int(status.get('response_slots_in_use', -1)) != 0:
        raise AssertionError(f'response slots leaked after queries: {status}')
    if int(status.get('response_slots_ready', -1)) != 0:
        raise AssertionError(f'ready response slots were not consumed: {status}')
    if int(status.get('response_slots_writing', -1)) != 0:
        raise AssertionError(f'writing response slots were not released: {status}')
    if int(status.get('document_response_slots_in_use', -1)) != 0:
        raise AssertionError(
            f'document response slots were not released: {status}'
        )
    if int(status.get('document_queue_limit', 0)) != int(
        status.get('document_worker_limit', -1)
    ):
        raise AssertionError(f'document queue limit missing: {status}')
    if int(status.get('document_worker_limit', 0)) <= 0:
        raise AssertionError(f'document worker limit missing: {status}')
    if status.get('accelerator_services') != []:
        raise AssertionError(f'accelerator services default changed: {status}')
    if int(status.get('accelerator_service_count', -1)) != 0:
        raise AssertionError(f'accelerator service count mismatch: {status}')
    if int(status.get('reserved_query_worker_slots', 0)) != 1:
        raise AssertionError(f'query execution reservation missing: {status}')
    if status.get('query_execution_lane_reserved') is not True:
        raise AssertionError(f'query execution lane is not reserved: {status}')
    if int(status.get('reserved_query_response_slots', 0)) != 1:
        raise AssertionError(f'query response reservation missing: {status}')
    if int(status.get('document_response_limit', 0)) != 31:
        raise AssertionError(f'document response limit mismatch: {status}')
    for key in (
        'canceled_requests',
        'orphan_responses',
        'worker_recoveries',
        'runtime_terminations',
        'runtime_liveness_timeouts',
    ):
        if key not in status:
            raise AssertionError(f'runtime recovery counter missing: {status}')
    if int(status.get('requests', 0)) < client_count:
        raise AssertionError(f'concurrent queries did not hit service: {status}')
    if int(status.get('successes', 0)) < client_count:
        raise AssertionError(f'concurrent queries did not complete: {status}')


def run_planner_native_cancellation_probe(
    socket_dir: Path,
    port: int,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'

    with (
        psycopg.connect(dsn, autocommit=True) as victim,
        psycopg.connect(dsn, autocommit=True) as controller,
    ):
        with victim.cursor() as cursor:
            cursor.execute('SELECT pg_backend_pid()')
            victim_pid = int(cursor.fetchone()[0])
            cursor.execute(
                "SET ii42.test_convergent_root_snapshot_pause_ms = '5000'"
            )

        def run_query() -> None:
            with victim.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id
                    FROM docs
                    ORDER BY ii42_query(
                        'docs_body_idx'::regclass,
                        'planner native cancellation sentinel'
                    ) DESC
                    LIMIT 3
                    """
                )
                cursor.fetchall()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_query)
            time.sleep(0.5)
            with controller.cursor() as cursor:
                cursor.execute('SELECT pg_cancel_backend(%s)', (victim_pid,))
                if not bool(cursor.fetchone()[0]):
                    raise AssertionError(
                        'planner-native cancellation request failed'
                    )
            try:
                future.result(timeout=10.0)
            except psycopg.errors.QueryCanceled:
                pass
            else:
                raise AssertionError(
                    'planner-native query ignored backend cancellation'
                )

        with victim.cursor() as cursor:
            cursor.execute('RESET ii42.test_convergent_root_snapshot_pause_ms')
            cursor.execute(
                """
                SELECT count(*)
                FROM (
                    SELECT id
                    FROM docs
                    ORDER BY ii42_query(
                        'docs_body_idx'::regclass,
                        'planner native post-cancellation sentinel'
                    ) DESC
                    LIMIT 3
                ) AS ranked
                """
            )
            if int(cursor.fetchone()[0]) != 3:
                raise AssertionError(
                    'planner-native query did not recover after cancellation'
                )


def run_runtime_batch_cap_reload_probe(
    socket_dir: Path,
    port: int,
    model_path: Path,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'

    def fetch_status(cursor: psycopg.Cursor[object]) -> dict[str, object]:
        cursor.execute('SELECT ii42_runtime_service_status()')
        row = cursor.fetchone()
        if row is None:
            raise AssertionError('runtime status returned no row')
        return dict(row[0])

    def wait_batch_cap(
        cursor: psycopg.Cursor[object],
        expected: int,
        *,
        excluded_pids: set[int] | None = None,
    ) -> dict[str, object]:
        deadline = time.monotonic() + 15.0
        last_status: dict[str, object] = {}
        excluded_pids = excluded_pids or set()
        while time.monotonic() < deadline:
            last_status = fetch_status(cursor)
            worker_pids = {
                int(item.get('pid', 0))
                for item in list(last_status.get('workers', []))
                if isinstance(item, dict) and int(item.get('pid', 0)) > 0
            }
            if (
                int(last_status.get('max_supported_batch_size', -1))
                    == expected
                and int(last_status.get('worker_count_ready', 0))
                    == int(last_status.get('worker_count_configured', -1))
                and last_status.get('ready_for_text_encoding') is True
                and not worker_pids.intersection(excluded_pids)
            ):
                return last_status
            time.sleep(0.1)
        raise AssertionError(
            f'runtime batch cap did not reload to {expected}: '
            f'{last_status}'
        )

    def encode_document_batch(cursor: psycopg.Cursor[object], count: int) -> None:
        texts = [
            (
                f'runtime batch reload row {row} '
                + ('semantic reload sentinel ' * 24)
            )
            for row in range(count)
        ]
        cursor.execute(
            'SELECT ii42_runtime_service_document_atoms_batch(%s, %s)',
            (str(model_path), texts),
        )
        result = cursor.fetchone()[0]
        if (
            not isinstance(result, dict)
            or len(list(result.get('results', []))) != count
        ):
            raise AssertionError(f'batch reload encode failed: {result}')

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "ALTER SYSTEM SET ii42.runtime_max_batch_size = '2'",
            )
            cursor.execute('SELECT pg_reload_conf()')
            low_status = wait_batch_cap(cursor, 2)
            workers = [
                item
                for item in list(low_status.get('workers', []))
                if isinstance(item, dict) and int(item.get('pid', 0)) > 0
            ]
            if not workers:
                raise AssertionError(f'no runtime workers found: {low_status}')
            terminated_pids = {int(worker['pid']) for worker in workers}
            for worker in workers:
                cursor.execute(
                    'SELECT pg_terminate_backend(%s)',
                    (int(worker['pid']),),
                )
            low_status = wait_batch_cap(
                cursor,
                2,
                excluded_pids=terminated_pids,
            )
            encode_document_batch(cursor, 2)

            cursor.execute(
                "ALTER SYSTEM SET ii42.runtime_max_batch_size = '4'",
            )
            cursor.execute('SELECT pg_reload_conf()')
            wait_batch_cap(cursor, 4)
            encode_document_batch(cursor, 4)

            cursor.execute('ALTER SYSTEM RESET ii42.runtime_max_batch_size')
            cursor.execute('SELECT pg_reload_conf()')


def run_parallel_worker_probe(
    socket_dir: Path,
    port: int,
    model_path: Path,
    worker_count: int,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    barrier = threading.Barrier(2)

    def encode_batch(worker_id: int) -> bool:
        texts = [
            (
                f'parallel runtime worker {worker_id} row {row} '
                + ('semantic posting concurrency sentinel ' * 12)
            )
            for row in range(32)
        ]
        with psycopg.connect(dsn, autocommit=True) as conn:
            barrier.wait(timeout=10)
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT ii42_runtime_service_query_atoms_batch(%s, %s)',
                    (str(model_path), texts),
                )
                return cursor.fetchone() is not None

    with psycopg.connect(dsn, autocommit=True) as control:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(encode_batch, worker_id)
                for worker_id in range(2)
            ]
            deadline = time.monotonic() + 20.0
            observed_parallel = False
            last_status: dict[str, object] = {}
            while time.monotonic() < deadline:
                with control.cursor() as cursor:
                    cursor.execute('SELECT ii42_runtime_service_status()')
                    last_status = dict(cursor.fetchone()[0])
                if int(last_status.get('worker_count_busy', 0)) >= 2:
                    observed_parallel = True
                    break
                if all(future.done() for future in futures):
                    break
                time.sleep(0.002)
            results = [future.result(timeout=20) for future in futures]

    if results != [True, True]:
        raise AssertionError(f'parallel runtime requests failed: {results}')
    if not observed_parallel:
        raise AssertionError(
            'runtime requests did not overlap on two workers: '
            f'{last_status}'
        )
    if int(last_status.get('worker_count_configured', 0)) != worker_count:
        raise AssertionError(f'configured worker count mismatch: {last_status}')
    if int(last_status.get('worker_count_ready', 0)) != worker_count:
        raise AssertionError(f'worker pool was not fully ready: {last_status}')


def run_model_affinity_probe(
    socket_dir: Path,
    port: int,
    model_path: Path,
    scratch_dir: Path,
) -> None:
    aliases = [
        scratch_dir / 'runtime-model-a',
        scratch_dir / 'runtime-model-b',
    ]
    for alias in aliases:
        alias.symlink_to(model_path, target_is_directory=True)

    dsn = f'host={socket_dir} port={port} dbname=postgres'
    barrier = threading.Barrier(2)

    def query_once(alias: Path, round_id: int) -> bool:
        with psycopg.connect(dsn, autocommit=True) as conn:
            barrier.wait(timeout=10)
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT ii42_runtime_service_query_atoms(%s, %s)',
                    (
                        str(alias),
                        f'model affinity round {round_id} for {alias.name}',
                    ),
                )
                result = cursor.fetchone()[0]
        return isinstance(result, dict) and bool(result.get('atoms'))

    def query_batch(alias: Path, lane: int) -> bool:
        texts = [
            (
                f'affinity steal lane {lane} row {row} '
                + ('semantic model affinity backlog sentinel ' * 32)
            )
            for row in range(32)
        ]
        with psycopg.connect(dsn, autocommit=True) as conn:
            barrier.wait(timeout=10)
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT ii42_runtime_service_query_atoms_batch(%s, %s)',
                    (str(alias), texts),
                )
                result = cursor.fetchone()[0]
        return (
            isinstance(result, dict)
            and len(list(result.get('results', []))) == len(texts)
        )

    with psycopg.connect(dsn, autocommit=True) as control:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for round_id in range(4):
                warm_results = [
                    future.result(timeout=30)
                    for future in (
                        executor.submit(query_once, aliases[0], round_id),
                        executor.submit(query_once, aliases[1], round_id),
                    )
                ]
                if warm_results != [True, True]:
                    raise AssertionError(
                        f'model affinity warmup failed: {warm_results}'
                    )

            with control.cursor() as cursor:
                cursor.execute('SELECT ii42_runtime_service_status()')
                before = dict(cursor.fetchone()[0])

            results = []
            for round_id in range(4, 28):
                results.extend(
                    future.result(timeout=30)
                    for future in (
                        executor.submit(query_once, aliases[0], round_id),
                        executor.submit(query_once, aliases[1], round_id),
                    )
                )

            steal_futures = [
                executor.submit(query_batch, aliases[0], lane)
                for lane in range(2)
            ]
            observed_affinity_steal = False
            steal_deadline = time.monotonic() + 20.0
            last_steal_status: dict[str, object] = {}
            while time.monotonic() < steal_deadline:
                with control.cursor() as cursor:
                    cursor.execute('SELECT ii42_runtime_service_status()')
                    last_steal_status = dict(cursor.fetchone()[0])
                if int(last_steal_status.get('worker_count_busy', 0)) >= 2:
                    observed_affinity_steal = True
                    break
                if all(future.done() for future in steal_futures):
                    break
                time.sleep(0.002)
            if not all(
                future.result(timeout=30)
                for future in steal_futures
            ):
                raise AssertionError('model affinity steal batches failed')
            if not observed_affinity_steal:
                raise AssertionError(
                    'same-model backlog did not use an idle non-affinity '
                    f'worker: {last_steal_status}'
                )

        with control.cursor() as cursor:
            cursor.execute('SELECT ii42_runtime_service_status()')
            after = dict(cursor.fetchone()[0])

    if not all(results):
        raise AssertionError('model affinity requests returned invalid atoms')
    load_delta = int(after.get('session_cache_loads', 0)) - int(
        before.get('session_cache_loads', 0)
    )
    if load_delta > 2:
        raise AssertionError(
            'model affinity caused repeated session reloads: '
            f'load_delta={load_delta}, before={before}, after={after}'
        )
    if int(after.get('affinity_dispatches', 0)) <= int(
        before.get('affinity_dispatches', 0)
    ):
        raise AssertionError(
            f'model affinity dispatches did not advance: {after}'
        )


def run_same_path_checkout_reload_probe(
    socket_dir: Path,
    port: int,
    model_path: Path,
    scratch_dir: Path,
) -> None:
    checkout_path = scratch_dir / 'runtime-model-same-path'
    checkout_path.mkdir()
    for child in model_path.iterdir():
        destination = checkout_path / child.name
        if child.name == 'manifest.json':
            destination.write_bytes(child.read_bytes())
        elif child.name == 'calibration':
            shutil.copytree(child, destination)
        else:
            destination.symlink_to(
                child,
                target_is_directory=child.is_dir(),
            )

    dsn = f'host={socket_dir} port={port} dbname=postgres'

    def status(cursor: psycopg.Cursor[object]) -> dict[str, object]:
        cursor.execute('SELECT ii42_runtime_service_status()')
        return dict(cursor.fetchone()[0])

    def query(
        cursor: psycopg.Cursor[object],
    ) -> tuple[list[object], str, float]:
        cursor.execute(
            'SELECT ii42_runtime_service_query_atoms(%s, %s)',
            (
                str(checkout_path),
                'same path checkout cache identity probe',
            ),
        )
        result = cursor.fetchone()[0]
        atoms = result.get('atoms') if isinstance(result, dict) else None
        signature = (
            result.get('checkout_signature')
            if isinstance(result, dict)
            else None
        )
        compiler = result.get('compiler') if isinstance(result, dict) else None
        lexical_proxy = (
            compiler.get('lexical_proxy')
            if isinstance(compiler, dict)
            else None
        )
        if not isinstance(atoms, list) or not atoms:
            raise AssertionError(
                f'same-path checkout returned invalid atoms: {result}'
            )
        if (
            not isinstance(signature, str)
            or len(signature) != 64
            or any(character not in '0123456789abcdef'
                   for character in signature)
        ):
            raise AssertionError(
                'same-path checkout returned an invalid manifest identity: '
                f'{result}'
            )
        if not isinstance(lexical_proxy, (float, int)) or lexical_proxy <= 0:
            raise AssertionError(
                'same-path checkout returned invalid compiler diagnostics: '
                f'{result}'
            )
        return atoms, signature, float(lexical_proxy)

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cursor:
            first_atoms, first_signature, first_lexical_proxy = query(cursor)
            after_first = status(cursor)
            (
                second_atoms,
                second_signature,
                second_lexical_proxy,
            ) = query(cursor)
            before_update = status(cursor)
            if (
                second_atoms != first_atoms
                or second_signature != first_signature
                or second_lexical_proxy != first_lexical_proxy
            ):
                raise AssertionError(
                    'unchanged checkout produced an unstable response'
                )
            if int(before_update.get('session_cache_loads', 0)) != int(
                after_first.get('session_cache_loads', 0)
            ):
                raise AssertionError(
                    'unchanged checkout did not reuse its ONNX session: '
                    f'first={after_first}, second={before_update}'
                )

            manifest_path = checkout_path / 'manifest.json'
            updated_manifest = json.loads(
                manifest_path.read_text(encoding='utf-8')
            )
            updated_manifest['session_cache_identity_revision'] = 1
            replacement_path = checkout_path / 'manifest.json.next'
            replacement_path.write_text(
                json.dumps(
                    updated_manifest,
                    indent=2,
                    sort_keys=True,
                )
                + '\n',
                encoding='utf-8',
            )
            os.replace(replacement_path, manifest_path)

            (
                updated_atoms,
                updated_signature,
                updated_lexical_proxy,
            ) = query(cursor)
            after_update = status(cursor)
            (
                cached_atoms,
                cached_signature,
                cached_lexical_proxy,
            ) = query(cursor)
            after_cached = status(cursor)

            calibration_relpath = updated_manifest[
                'artifacts'
            ]['query_calibration_runtime']['path']
            calibration_path = checkout_path / calibration_relpath
            calibration = bytearray(calibration_path.read_bytes())
            lexical_dims = int(
                updated_manifest['runtime_output']['lexical_dims']
            )
            for atom_id in range(lexical_dims):
                offset = 24 + atom_id * 4
                value = struct.unpack_from('<f', calibration, offset)[0]
                struct.pack_into('<f', calibration, offset, value * 1.25)
            calibration_next = calibration_path.with_suffix('.next')
            calibration_next.write_bytes(calibration)
            os.replace(calibration_next, calibration_path)
            updated_manifest[
                'artifacts'
            ]['query_calibration_runtime']['sha256'] = hashlib.sha256(
                calibration
            ).hexdigest()
            updated_manifest['compiler_cache_identity_revision'] = 1
            replacement_path.write_text(
                json.dumps(
                    updated_manifest,
                    indent=2,
                    sort_keys=True,
                )
                + '\n',
                encoding='utf-8',
            )
            os.replace(replacement_path, manifest_path)

            (
                calibration_atoms,
                calibration_signature,
                calibration_lexical_proxy,
            ) = query(cursor)
            after_calibration = status(cursor)
            (
                calibration_cached_atoms,
                calibration_cached_signature,
                calibration_cached_lexical_proxy,
            ) = query(cursor)
            after_calibration_cached = status(cursor)

    if updated_atoms != first_atoms or cached_atoms != first_atoms:
        raise AssertionError(
            'manifest-only checkout update changed deterministic atoms'
        )
    if (
        updated_signature == first_signature
        or cached_signature != updated_signature
        or updated_lexical_proxy != first_lexical_proxy
        or cached_lexical_proxy != updated_lexical_proxy
    ):
        raise AssertionError(
            'manifest-only checkout update did not publish a new stable '
            'worker identity'
        )
    if (
        calibration_signature == updated_signature
        or calibration_cached_signature != calibration_signature
        or calibration_atoms != first_atoms
        or calibration_cached_atoms != calibration_atoms
        or calibration_lexical_proxy == updated_lexical_proxy
        or calibration_cached_lexical_proxy != calibration_lexical_proxy
    ):
        raise AssertionError(
            'same-path calibration update did not replace and retain the '
            'posting compiler cache'
        )
    load_delta = int(after_update.get('session_cache_loads', 0)) - int(
        before_update.get('session_cache_loads', 0)
    )
    if load_delta != 1:
        raise AssertionError(
            'same-path checkout update did not load one new ONNX session: '
            f'load_delta={load_delta}, before={before_update}, '
            f'after={after_update}'
        )
    if int(after_cached.get('session_cache_loads', 0)) != int(
        after_update.get('session_cache_loads', 0)
    ):
        raise AssertionError(
            'updated checkout did not reuse its replacement ONNX session: '
            f'updated={after_update}, cached={after_cached}'
        )
    calibration_load_delta = int(
        after_calibration.get('session_cache_loads', 0)
    ) - int(after_cached.get('session_cache_loads', 0))
    if calibration_load_delta != 1:
        raise AssertionError(
            'same-path calibration update did not load one new ONNX session: '
            f'load_delta={calibration_load_delta}, '
            f'before={after_cached}, after={after_calibration}'
        )
    if int(
        after_calibration_cached.get('session_cache_loads', 0)
    ) != int(after_calibration.get('session_cache_loads', 0)):
        raise AssertionError(
            'calibration-updated checkout did not reuse its replacement '
            f'ONNX session: updated={after_calibration}, '
            f'cached={after_calibration_cached}'
        )


def run_mixed_query_document_probe(
    socket_dir: Path,
    port: int,
    model_path: Path,
    worker_count: int,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    expected_document_workers = worker_count - 1
    document_request_count = expected_document_workers + 1
    barrier = threading.Barrier(document_request_count)

    def encode_document_batch(lane: int) -> bool:
        texts = [
            (
                f'document lane {lane} row {row} '
                + ('long semantic document inference sentinel ' * 36)
            )
            for row in range(32)
        ]
        with psycopg.connect(dsn, autocommit=True) as conn:
            barrier.wait(timeout=10)
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT ii42_runtime_service_document_atoms_batch(%s, %s)',
                    (str(model_path), texts),
                )
                result = cursor.fetchone()[0]
        return isinstance(result, dict) and bool(result.get('results'))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=document_request_count
    ) as executor:
        document_futures = [
            executor.submit(encode_document_batch, lane)
            for lane in range(document_request_count)
        ]
        with psycopg.connect(dsn, autocommit=True) as control:
            deadline = time.monotonic() + 20.0
            observed_reserved_lane = False
            last_status: dict[str, object] = {}
            while time.monotonic() < deadline:
                with control.cursor() as cursor:
                    cursor.execute('SELECT ii42_runtime_service_status()')
                    last_status = dict(cursor.fetchone()[0])
                if (
                    int(last_status.get('document_workers_busy', 0))
                        == expected_document_workers
                    and (
                        int(last_status.get('queue_depth', 0)) >= 1
                        or int(
                            last_status.get('response_slots_in_use', 0)
                        ) >= int(
                            last_status.get('document_response_limit', 0)
                        )
                    )
                ):
                    observed_reserved_lane = True
                    break
                if all(future.done() for future in document_futures):
                    break
                time.sleep(0.002)

            if not observed_reserved_lane:
                raise AssertionError(
                    'document work did not preserve one execution lane: '
                    f'{last_status}'
                )
            if int(
                last_status.get('document_response_slots_in_use', -1)
            ) > int(last_status.get('document_response_limit', 0)):
                raise AssertionError(
                    'document work consumed the reserved query response '
                    f'slot: {last_status}'
                )

            with control.cursor() as cursor:
                cursor.execute(
                    'SELECT ii42_runtime_service_query_atoms(%s, %s)',
                    (
                        str(model_path),
                        'latency-sensitive query lane sentinel',
                    ),
                )
                query_result = cursor.fetchone()[0]
            query_preempted_waiting_document = not all(
                future.done()
                for future in document_futures
            )
            document_results = [
                future.result(timeout=30)
                for future in document_futures
            ]

    if not isinstance(query_result, dict) or not query_result.get('atoms'):
        raise AssertionError(
            f'reserved query lane returned invalid output: {query_result}'
        )
    if not query_preempted_waiting_document:
        raise AssertionError(
            'query completed only after all document requests; '
            'the reserved execution lane was ineffective'
        )
    if document_results != [True] * document_request_count:
        raise AssertionError(
            f'mixed document batches failed: {document_results}'
        )


def run_completed_response_reservation_probe(
    socket_dir: Path,
    port: int,
    model_path: Path,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    document_connections = [
        psycopg.connect(dsn, autocommit=True)
        for _ in range(8)
    ]
    backend_pids: list[int] = []
    stopped_pids: list[int] = []
    futures: list[concurrent.futures.Future[bool]] = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    control = psycopg.connect(dsn, autocommit=True)

    def runtime_status() -> dict[str, object]:
        with control.cursor() as cursor:
            cursor.execute('SELECT ii42_runtime_service_status()')
            return dict(cursor.fetchone()[0])

    def encode_document_batch(
        connection: psycopg.Connection[tuple[object, ...]],
        lane: int,
    ) -> bool:
        texts = [
            (
                f'completed response reservation lane {lane} row {row} '
                + ('semantic document response occupancy sentinel ' * 39)
            )
            for row in range(32)
        ]
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT ii42_runtime_service_document_atoms_batch(%s, %s)',
                (str(model_path), texts),
            )
            result = cursor.fetchone()[0]
        return isinstance(result, dict) and bool(result.get('results'))

    try:
        for connection in document_connections:
            with connection.cursor() as cursor:
                cursor.execute('SELECT pg_backend_pid()')
                backend_pids.append(int(cursor.fetchone()[0]))

        initial_status = runtime_status()
        metrics = initial_status.get('accelerator_metrics')
        local_capacity = 1
        if isinstance(metrics, list):
            for entry in metrics:
                if isinstance(entry, dict) and entry.get('url') == 'local':
                    local_capacity = max(1, int(entry.get('capacity', 1)))
                    break
        paused_target = min(7, local_capacity)

        for lane in range(paused_target):
            future = executor.submit(
                encode_document_batch,
                document_connections[lane],
                lane,
            )
            futures.append(future)
            deadline = time.monotonic() + 20.0
            status: dict[str, object] = {}
            while time.monotonic() < deadline:
                status = runtime_status()
                if (
                    status.get('request_processing') is True
                    and int(status.get('processing_owner_pid', 0))
                        == backend_pids[lane]
                ):
                    break
                if future.done():
                    raise AssertionError(
                        'document response completed before its owner could '
                        f'be paused: lane={lane}, status={status}'
                    )
                time.sleep(0.001)
            else:
                raise AssertionError(
                    'document response owner never entered processing: '
                    f'lane={lane}, status={status}'
                )

            os.kill(backend_pids[lane], signal.SIGSTOP)
            stopped_pids.append(backend_pids[lane])
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                status = runtime_status()
                if (
                    int(status.get('response_slots_ready', 0)) >= lane + 1
                    and int(
                        status.get(
                            'document_response_slots_in_use',
                            0,
                        )
                    ) == lane + 1
                ):
                    break
                time.sleep(0.002)
            else:
                raise AssertionError(
                    'paused document owner did not retain its completed '
                    f'response: lane={lane}, status={status}'
                )

        blocked = executor.submit(
            encode_document_batch,
            document_connections[paused_target],
            paused_target,
        )
        futures.append(blocked)
        deadline = time.monotonic() + 1.0
        status = {}
        while time.monotonic() < deadline:
            status = runtime_status()
            if int(
                status.get('document_response_slots_in_use', 0)
            ) > int(status.get('document_response_limit', 0)):
                raise AssertionError(
                    'completed document responses consumed the reserved '
                    f'query slot: {status}'
                )
            if blocked.done():
                raise AssertionError(
                    'document work bypassed the completed-response lane '
                    f'capacity: {status}'
                )
            time.sleep(0.005)

        with control.cursor() as cursor:
            cursor.execute("SET statement_timeout = '15s'")
            cursor.execute(
                'SELECT ii42_runtime_service_query_atoms(%s, %s)',
                (
                    str(model_path),
                    'query admitted beside completed document responses',
                ),
            )
            query_result = cursor.fetchone()[0]
            cursor.execute('RESET statement_timeout')
        if (
            not isinstance(query_result, dict)
            or not query_result.get('atoms')
        ):
            raise AssertionError(
                f'reserved query response was invalid: {query_result}'
            )
        for backend_pid in stopped_pids:
            os.kill(backend_pid, signal.SIGCONT)
        stopped_pids.clear()
        document_results = [
            future.result(timeout=30)
            for future in futures
        ]
        if document_results != [True] * len(futures):
            raise AssertionError(
                'document responses did not drain after reserved query: '
                f'{document_results}'
            )
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_error: Exception | None = None
        for backend_pid in stopped_pids:
            try:
                os.kill(backend_pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        for future in futures:
            try:
                future.result(timeout=30)
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        executor.shutdown(wait=True, cancel_futures=True)
        for connection in document_connections:
            connection.close()
        control.close()
        if not active_error and cleanup_error is not None:
            raise cleanup_error


def run_single_build_pipeline_probe(
    socket_dir: Path,
    port: int,
    model_path: Path,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    escaped_model_path = str(model_path).replace("'", "''")

    with psycopg.connect(dsn, autocommit=True) as control:
        with control.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE runtime_pipeline_docs (
                    id integer PRIMARY KEY,
                    body text NOT NULL
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO runtime_pipeline_docs (id, body)
                SELECT
                    value,
                    'pipeline document ' || value || ' ' ||
                    repeat(
                        'semantic lexical posting concurrency sentinel ',
                        24
                    )
                FROM generate_series(1, 512) AS value
                """
            )
            cursor.execute('SELECT ii42_runtime_service_status()')
            before = dict(cursor.fetchone()[0])

        def build_index(index_name: str) -> None:
            with psycopg.connect(dsn, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        CREATE INDEX {index_name}
                        ON runtime_pipeline_docs
                        USING ii42 (body)
                        WITH (
                            sae = true,
                            model_path = '{escaped_model_path}'
                        )
                        """
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            build = executor.submit(
                build_index,
                'runtime_pipeline_docs_a_idx',
            )
            observed_parallel = False
            last_status: dict[str, object] = {}
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                with control.cursor() as cursor:
                    cursor.execute('SELECT ii42_runtime_service_status()')
                    last_status = dict(cursor.fetchone()[0])
                if int(last_status.get('document_workers_busy', 0)) >= 2:
                    observed_parallel = True
                    break
                if build.done():
                    break
                time.sleep(0.002)
            build.result(timeout=120)

        if not observed_parallel:
            raise AssertionError(
                'one CREATE INDEX did not keep two document batches in '
                f'flight: {last_status}'
            )

        build_index('runtime_pipeline_docs_b_idx')
        with control.cursor() as cursor:
            cursor.execute('SELECT ii42_runtime_service_status()')
            after = dict(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT
                    index_name::text,
                    (ii42_index_status(index_name)->>'query_ready')::boolean,
                    (
                        ii42_index_status(index_name)
                        #>> '{generation,posting,record_count}'
                    )::integer
                FROM unnest(
                    ARRAY[
                        'runtime_pipeline_docs_a_idx'::regclass,
                        'runtime_pipeline_docs_b_idx'::regclass
                    ]
                ) AS index_name
                ORDER BY index_name::text
                """
            )
            index_rows = cursor.fetchall()

        if any(
            ready is not True or int(record_count) != 512
            for _, ready, record_count in index_rows
        ):
            raise AssertionError(
                f'pipelined index generations are invalid: {index_rows}'
            )
        if int(after.get('document_dispatches', 0)) - int(
            before.get('document_dispatches', 0)
        ) < 4:
            raise AssertionError(
                'pipelined builds did not dispatch multiple document batches: '
                f'before={before}, after={after}'
            )

        queries = (
            'semantic posting concurrency',
            'lexical pipeline document',
            'sentinel 417',
        )
        for query in queries:
            rankings: list[list[tuple[int, float]]] = []
            with control.cursor() as cursor:
                for index_name in (
                    'runtime_pipeline_docs_a_idx',
                    'runtime_pipeline_docs_b_idx',
                ):
                    cursor.execute(
                        """
                        SELECT source.id, hit.score::float8
                        FROM ii42_query(%s::regclass, %s, 25) AS hit
                        JOIN runtime_pipeline_docs AS source
                            ON source.ctid = hit.ctid
                        ORDER BY hit.score DESC, source.id
                        """,
                        (index_name, query),
                    )
                    rankings.append(
                        [
                            (int(document_id), float(score))
                            for document_id, score in cursor.fetchall()
                        ]
                    )
            if rankings[0] != rankings[1]:
                raise AssertionError(
                    f'pipelined build ranking is nondeterministic for {query!r}'
                )

        with control.cursor() as cursor:
            cursor.execute('DROP TABLE runtime_pipeline_docs CASCADE')


def run_runtime_owner_cancellation_probe(
    socket_dir: Path,
    port: int,
    model_path: Path,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    victim = psycopg.connect(dsn, autocommit=True)
    control = psycopg.connect(dsn, autocommit=True)

    try:
        with victim.cursor() as cursor:
            cursor.execute('SELECT pg_backend_pid()')
            victim_pid = int(cursor.fetchone()[0])
        with control.cursor() as cursor:
            cursor.execute('SELECT ii42_runtime_service_status()')
            before = dict(cursor.fetchone()[0])

        texts = [
            (
                f'cancellation batch row {row} '
                + ('semantic runtime owner cancellation sentinel ' * 40)
            )
            for row in range(32)
        ]

        def submit_victim() -> str:
            with victim.cursor() as cursor:
                cursor.execute(
                    'SELECT ii42_runtime_service_query_atoms_batch(%s, %s)',
                    (str(model_path), texts),
                )
                row = cursor.fetchone()
            return 'completed' if row is not None else 'missing'

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(submit_victim)
            deadline = time.monotonic() + 15.0
            observed_processing = False
            while time.monotonic() < deadline:
                with control.cursor() as cursor:
                    cursor.execute('SELECT ii42_runtime_service_status()')
                    current = dict(cursor.fetchone()[0])
                if (
                    current.get('request_processing') is True
                    and int(current.get('processing_owner_pid', 0))
                        == victim_pid
                    and int(current.get('processing_batch_count', 0)) == 32
                ):
                    observed_processing = True
                    break
                if future.done():
                    break
                time.sleep(0.005)
            if not observed_processing:
                raise AssertionError(
                    'large runtime batch completed before cancellation '
                    f'could observe it: status={current}'
                )

            with control.cursor() as cursor:
                cursor.execute('SELECT pg_cancel_backend(%s)', (victim_pid,))
                canceled = bool(cursor.fetchone()[0])
            if not canceled:
                raise AssertionError(
                    f'pg_cancel_backend({victim_pid}) returned false'
                )
            try:
                future.result(timeout=15)
            except psycopg.errors.QueryCanceled:
                pass
            else:
                raise AssertionError(
                    'runtime owner request completed despite cancellation'
                )
        victim.close()

        with control.cursor() as cursor:
            cursor.execute(
                'SELECT ii42_runtime_service_query_atoms(%s, %s)',
                (str(model_path), 'post cancellation service sentinel'),
            )
            if cursor.fetchone() is None:
                raise AssertionError(
                    'post-cancellation runtime request returned no row'
                )

        deadline = time.monotonic() + 15.0
        after: dict[str, object] = {}
        while time.monotonic() < deadline:
            with control.cursor() as cursor:
                cursor.execute('SELECT ii42_runtime_service_status()')
                after = dict(cursor.fetchone()[0])
            if (
                after.get('request_processing') is False
                and int(after.get('response_slots_in_use', -1)) == 0
                and int(after.get('response_slots_ready', -1)) == 0
                and int(after.get('response_slots_writing', -1)) == 0
            ):
                break
            time.sleep(0.01)
        else:
            raise AssertionError(
                f'runtime slots did not drain after cancellation: {after}'
            )

        if int(after.get('canceled_requests', 0)) <= int(
            before.get('canceled_requests', 0)
        ):
            raise AssertionError(
                f'caller cancellation was not recorded: before={before}, '
                f'after={after}'
            )
        if int(after.get('orphan_responses', 0)) <= int(
            before.get('orphan_responses', 0)
        ):
            raise AssertionError(
                f'orphan response was not discarded: before={before}, '
                f'after={after}'
            )
        if int(after.get('runtime_terminations', 0)) <= int(
            before.get('runtime_terminations', 0)
        ):
            raise AssertionError(
                'in-flight ONNX execution was not cooperatively terminated: '
                f'before={before}, after={after}'
            )
        if int(after.get('runtime_liveness_timeouts', 0)) != int(
            before.get('runtime_liveness_timeouts', 0)
        ):
            raise AssertionError(
                'caller cancellation was misclassified as a liveness timeout: '
                f'before={before}, after={after}'
            )
    finally:
        victim.close()
        control.close()


def run_batch_request_validation_probe(
    socket_dir: Path,
    port: int,
    model_path: Path,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    cases = [
        (
            'empty batch',
            [],
            'must be one-dimensional',
        ),
        (
            'null row',
            ['valid row', None],
            'must not contain nulls',
        ),
        (
            'row limit',
            [f'row {index}' for index in range(513)],
            'invalid ii42 runtime text batch size',
        ),
        (
            'packed text limit',
            ['a' * 600_000, 'b' * 600_000],
            'transport limit',
        ),
    ]
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for label, texts, expected_error in cases:
                try:
                    cur.execute(
                        'SELECT ii42_runtime_service_query_atoms_batch(%s, %s)',
                        (str(model_path), texts),
                    )
                except psycopg.Error as exc:
                    if expected_error not in str(exc):
                        raise AssertionError(
                            f'{label} returned unexpected error: {exc}'
                        ) from exc
                else:
                    raise AssertionError(f'{label} should have failed')

            cur.execute('SELECT ii42_runtime_service_status()')
            status = cur.fetchone()[0]
    if status.get('worker_ready') is not True:
        raise AssertionError(
            f'batch validation errors damaged the runtime worker: {status}'
        )


def expect_database_error(
    cursor: psycopg.Cursor[tuple[object, ...]],
    sql: str,
    expected_error: str,
) -> None:
    try:
        cursor.execute(sql)
    except psycopg.Error as exc:
        if expected_error not in str(exc):
            raise AssertionError(
                f'unexpected database error for {sql!r}: {exc}'
            ) from exc
    else:
        raise AssertionError(f'statement should have failed: {sql}')


def run_semantic_product_contract_probe(
    socket_dir: Path,
    port: int,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    with psycopg.connect(dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            expect_database_error(
                cursor,
                "SELECT ii42_index_options('docs_pkey'::regclass)",
                'must use ii42 access method',
            )
            expect_database_error(
                cursor,
                """
                CREATE INDEX invalid_sae_k1_idx
                ON docs USING ii42 (body)
                WITH (sae = true, k1 = 1.2)
                """,
                'not valid with sae=true',
            )
            cursor.execute(
                """
                CREATE INDEX valid_sae_automatic_idx
                ON docs USING ii42 (body)
                WITH (
                    sae = true,
                    consistency = eventual
                )
                """
            )
            cursor.execute(
                """
                SELECT
                    consistency,
                    pending_writes,
                    pending_deletes,
                    delta_records,
                    delta_bytes
                FROM ii42_index_details(
                    'valid_sae_automatic_idx'::regclass
                )
                """
            )
            automatic_details = cursor.fetchone()
            if automatic_details != ('eventual', 0, 0, 0, 0):
                raise AssertionError(
                    'SAE eventual details do not expose clean durable debt: '
                    f'{automatic_details}'
                )
            cursor.execute('DROP INDEX valid_sae_automatic_idx')

            cursor.execute(
                """
                SELECT
                    details.index_bytes,
                    (
                        status#>>'{generation,primary,bytes}'
                    )::int8,
                    (
                        status#>>'{generation,posting,bytes}'
                    )::int8,
                    (
                        status#>>'{generation,primary,physical_blocks}'
                    )::int8 * current_setting('block_size')::int8
                FROM ii42_index_details(
                    'docs_body_idx'::regclass
                ) AS details
                CROSS JOIN LATERAL (
                    SELECT jsonb_build_object(
                        'generation',
                        ii42_index_generation_audit_internal(
                            'docs_body_idx'::regclass
                        )
                    ) AS status
                ) AS current_status
                """
            )
            (
                reported_bytes,
                primary_bytes,
                posting_bytes,
                physical_bytes,
            ) = cursor.fetchone()
            if (
                reported_bytes <= 0
                or reported_bytes < primary_bytes
                or reported_bytes > physical_bytes
                or posting_bytes <= 0
                or posting_bytes > primary_bytes
            ):
                raise AssertionError(
                    'SAE index_bytes does not bound the authoritative v3 '
                    f'working set: details={reported_bytes} '
                    f'primary={primary_bytes} '
                    f'posting={posting_bytes} '
                    f'physical={physical_bytes}'
                )

            cursor.execute(
                """
                SELECT
                    recommended_options,
                    recommended_consistency
                FROM ii42_index_policy_recommend(
                    'docs_body_idx'::regclass,
                    'balanced'
                )
                """
            )
            balanced = cursor.fetchone()
            if balanced != (
                "WITH (consistency = 'eventual')",
                'eventual',
            ):
                raise AssertionError(
                    f'illegal SAE balanced recommendation: {balanced}'
                )

            cursor.execute(
                """
                SELECT
                    recommended_options,
                    recommended_consistency
                FROM ii42_index_policy_recommend(
                    'docs_body_idx'::regclass,
                    'query_first'
                )
                """
            )
            query_first = cursor.fetchone()
            if query_first != (
                "WITH (consistency = 'eventual')",
                'eventual',
            ):
                raise AssertionError(
                    f'illegal SAE query-first recommendation: {query_first}'
                )

            expect_database_error(
                cursor,
                """
                SELECT *
                FROM ii42_index_policy_recommend(
                    'docs_body_idx'::regclass,
                    'heavy_insert_skew'
                )
                """,
                'applies only to sae=false indexes',
            )

            cursor.execute(
                """
                CREATE TABLE semantic_policy_docs (
                    id integer PRIMARY KEY,
                    body text NOT NULL
                )
                """
            )
            for consistency in ('realtime', 'manual'):
                expect_database_error(
                    cursor,
                    f"""
                    CREATE INDEX invalid_sae_{consistency}_idx
                    ON semantic_policy_docs USING ii42 (body)
                    WITH (sae = true, consistency = {consistency})
                    """,
                    "SAE indexes require consistency = 'eventual'",
                )
            cursor.execute(
                """
                SELECT count(*)
                FROM pg_catalog.pg_class
                WHERE relname IN (
                    'invalid_sae_realtime_idx',
                    'invalid_sae_manual_idx'
                )
                """
            )
            if int(cursor.fetchone()[0]) != 0:
                raise AssertionError(
                    'rejected SAE policy left an index relation'
                )
            cursor.execute('DROP TABLE semantic_policy_docs')


def run_low_memory_semantic_completion_probe(
    socket_dir: Path,
    port: int,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    with psycopg.connect(dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                'SHOW ii42.maintenance_rebuild_memory_budget',
            )
            previous_budget = str(cursor.fetchone()[0])
            cursor.execute(
                "ALTER SYSTEM SET "
                "ii42.maintenance_rebuild_memory_budget = '1MB'",
            )
            cursor.execute('SELECT pg_reload_conf()')
            deadline = time.monotonic() + 5.0
            while True:
                cursor.execute(
                    'SHOW ii42.maintenance_rebuild_memory_budget',
                )
                if str(cursor.fetchone()[0]) == '1MB':
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError(
                        'semantic rebuild memory budget did not reload',
                    )
                time.sleep(0.05)

            try:
                cursor.execute("SET work_mem = '64kB'")
                cursor.execute(
                    """
                    CREATE TABLE semantic_spill_docs (
                        id integer PRIMARY KEY,
                        body text NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO semantic_spill_docs
                    SELECT
                        value,
                        'semantic spill batch document ' || value || ' group '
                            || (value % 97)
                    FROM pg_catalog.generate_series(1, 4096) AS value
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX semantic_spill_docs_idx
                    ON semantic_spill_docs
                    USING ii42 (body)
                    WITH (sae = true)
                    """
                )
                cursor.execute(
                    """
                    SELECT ii42_index_status(
                        'semantic_spill_docs_idx'::regclass
                    )
                    """
                )
                initial_status = cursor.fetchone()[0]
                cursor.execute(
                    """
                    SELECT ii42_index_runtime_state_json(
                        'semantic_spill_docs_idx'::regclass
                    )
                    """
                )
                initial_cache = cursor.fetchone()[0]
                assert_cache_state_snapshot(initial_cache)
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM ii42_query(
                        'semantic_spill_docs_idx'::regclass,
                        'semantic spill batch group 31',
                        10
                    )
                    """
                )
                initial_hits = int(cursor.fetchone()[0])
                cursor.execute(
                    "SET ii42.test_convergent_l0_rotation_records = '256'"
                )
                cursor.execute(
                    """
                    UPDATE semantic_spill_docs
                    SET body = body || ' rebuilt'
                    WHERE id <= 256
                    """
                )
                cursor.execute(
                    """
                    SELECT ii42_index_runtime_state_json(
                        'semantic_spill_docs_idx'::regclass
                    )
                    """
                )
                pending_cache = cursor.fetchone()[0]
                assert_cache_state_snapshot(pending_cache)
                budget_results: list[str] = []
                budget_status = initial_status
                for _ in range(80):
                    cursor.execute(
                        """
                        SELECT ii42_index_try_maintain(
                            'semantic_spill_docs_idx'::regclass
                        )
                        """
                    )
                    budget_results.append(str(cursor.fetchone()[0]))
                    cursor.execute(
                        """
                        SELECT ii42_index_status(
                            'semantic_spill_docs_idx'::regclass
                        )
                        """
                    )
                    budget_status = cursor.fetchone()[0]
                    cursor.execute(
                        """
                        SELECT ii42_index_runtime_state_json(
                            'semantic_spill_docs_idx'::regclass
                        )
                        """
                    )
                    assert_cache_state_snapshot(cursor.fetchone()[0])
                    semantic_completion = budget_status['generation'][
                        'delta'
                    ]['semantic_completion']
                    if (
                        int(budget_status['details']['delta_records']) == 0
                        and int(
                            budget_status['details']['pending_writes']
                        ) == 0
                        and bool(semantic_completion['converged'])
                    ):
                        break
                    time.sleep(0.05)
                else:
                    raise AssertionError(
                        'low-memory semantic maintenance did not converge: '
                        f'results={budget_results}, status={budget_status}'
                    )
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM ii42_query(
                        'semantic_spill_docs_idx'::regclass,
                        'semantic spill batch rebuilt',
                        100
                    ) AS hit
                    JOIN semantic_spill_docs AS docs
                      ON docs.ctid = hit.ctid
                    WHERE docs.id <= 256
                    """
                )
                budget_hits = int(cursor.fetchone()[0])
                cursor.execute('REINDEX INDEX semantic_spill_docs_idx')
                cursor.execute(
                    """
                    SELECT ii42_index_status(
                        'semantic_spill_docs_idx'::regclass
                    )
                    """
                )
                rebuilt_status = cursor.fetchone()[0]
                cursor.execute(
                    """
                    SELECT ii42_index_runtime_state_json(
                        'semantic_spill_docs_idx'::regclass
                    )
                    """
                )
                rebuilt_cache = cursor.fetchone()[0]
                assert_cache_state_snapshot(rebuilt_cache)
                cursor.execute('DROP TABLE semantic_spill_docs')
            finally:
                cursor.execute(
                    'RESET ii42.test_convergent_l0_rotation_records'
                )
                cursor.execute(
                    'ALTER SYSTEM RESET '
                    'ii42.maintenance_rebuild_memory_budget',
                )
                cursor.execute('SELECT pg_reload_conf()')
                deadline = time.monotonic() + 5.0
                while True:
                    cursor.execute(
                        'SHOW ii42.maintenance_rebuild_memory_budget',
                    )
                    if str(cursor.fetchone()[0]) == previous_budget:
                        break
                    if time.monotonic() >= deadline:
                        raise AssertionError(
                            'semantic rebuild memory budget did not reset',
                        )
                    time.sleep(0.05)

    initial_generation = initial_status['generation']
    budget_generation = budget_status['generation']
    budget_completion = budget_generation['delta']['semantic_completion']
    rebuilt_generation = rebuilt_status['generation']
    initial_rebuild_estimate = rebuild_estimate(
        initial_cache,
        'spill_estimated_bytes',
    )
    pending_rebuild_estimate = rebuild_estimate(
        pending_cache,
        'spill_estimated_bytes',
    )
    expected_initial_estimate = (
        SEMANTIC_FIXED_WORKSPACE_BYTES
        + 64 * 1024
        + 4096 * ITEM_POINTER_BYTES * 8
    )
    pending_delta_bytes = rebuild_estimate(
        pending_cache,
        'delta_bytes',
    )
    if initial_hits != 10:
        raise AssertionError(
            f'low-work-mem semantic query returned {initial_hits} rows'
        )
    if (
        not initial_generation['atomic']
        or not initial_generation['valid']
        or int(initial_generation['docs']) != 4096
        or initial_cache['generation']['rebuild_builder']
            != 'semantic_stream'
        or bool(initial_cache['generation']['rebuild_admitted'])
        or bool(pending_cache['generation']['rebuild_admitted'])
        or initial_rebuild_estimate != expected_initial_estimate
        or pending_delta_bytes <= 0
        or pending_rebuild_estimate
            != expected_initial_estimate + pending_delta_bytes * 4
    ):
        raise AssertionError(
            'low-work-mem semantic rebuild workload is invalid: '
            f'initial_status={initial_status}, initial_cache={initial_cache}, '
            f'pending_cache={pending_cache}'
        )
    completion_progress_observed = (
        any(
            reason in result
            for result in budget_results
            for reason in (
                'reason=active_l0_rotated',
                'reason=semantic_completed',
            )
        )
        or int(budget_completion['telemetry']['completed']) > 0
    )
    if (
        not completion_progress_observed
        or budget_generation['generation_id']
            == initial_generation['generation_id']
        or int(budget_status['details']['delta_records']) != 0
        or int(budget_status['details']['pending_writes']) != 0
        or not bool(budget_completion['converged'])
        or budget_hits <= 0
    ):
        raise AssertionError(
            'incremental semantic completion did not converge under the low '
            f'memory gate: results={budget_results}, '
            f'status={budget_status}, hits={budget_hits}'
        )
    if (
        not rebuilt_generation['atomic']
        or not rebuilt_generation['valid']
        or int(rebuilt_generation['docs']) != 4096
        or rebuilt_cache['generation']['rebuild_builder']
            != 'semantic_stream'
        or rebuilt_generation['generation_id']
            == initial_generation['generation_id']
    ):
        raise AssertionError(
            f'low-work-mem rebuilt generation is invalid: {rebuilt_status}'
        )


def run_database_lifecycle_probe(
    socket_dir: Path,
    port: int,
    worker_count: int,
) -> None:
    database_name = 'ii42_worker_lifecycle_probe'
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    with psycopg.connect(dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT application_name, count(*)
                FROM pg_catalog.pg_stat_activity
                WHERE datname = 'template1'
                  AND pid <> pg_catalog.pg_backend_pid()
                GROUP BY application_name
                """
            )
            template_sessions = {
                str(application_name): int(count)
                for application_name, count in cursor.fetchall()
            }
            expected_control_sessions = {
                'ii42 runtime service': worker_count,
                'ii42 background supervisor': 1,
            }
            if template_sessions != expected_control_sessions:
                raise AssertionError(
                    'ii42 workers did not attach only to the configured '
                    f'control database: {template_sessions}'
                )

            cursor.execute(
                f'CREATE DATABASE {database_name} TEMPLATE template0'
            )

        probe_dsn = (
            f'host={socket_dir} port={port} dbname={database_name}'
        )
        with psycopg.connect(probe_dsn, autocommit=True) as probe_connection:
            with probe_connection.cursor() as probe_cursor:
                probe_cursor.execute('CREATE EXTENSION ii42')
                probe_cursor.execute(
                    """
                    CREATE TABLE docs (
                        id integer PRIMARY KEY,
                        body text NOT NULL
                    )
                    """
                )
                probe_cursor.execute(
                    """
                    INSERT INTO docs
                    SELECT value, 'worker lifecycle document ' || value
                    FROM pg_catalog.generate_series(1, 100) AS value
                    """
                )
                probe_cursor.execute(
                    """
                    CREATE INDEX docs_body_idx
                    ON docs
                    USING ii42 (body)
                    WITH (
                        sae = false,
                        consistency = eventual
                    )
                    """
                )
                probe_cursor.execute(
                    """
                    UPDATE docs
                    SET body = body || ' updated'
                    WHERE id <= 10
                    """
                )
                probe_cursor.execute('SELECT ii42_index_touch_maintenance()')

        # Give the supervisor enough time to visit this database and verify
        # that its one-shot worker releases the database connection afterward.
        time.sleep(2.0)
        deadline = time.monotonic() + 10.0
        while True:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT application_name
                    FROM pg_catalog.pg_stat_activity
                    WHERE datname = %s
                    """,
                    (database_name,),
                )
                sessions = [row[0] for row in cursor.fetchall()]
            if not sessions:
                break
            if time.monotonic() >= deadline:
                raise AssertionError(
                    'ii42 worker kept application database pinned: '
                    f'{sessions}'
                )
            time.sleep(0.1)

        with connection.cursor() as cursor:
            cursor.execute(f'DROP DATABASE {database_name}')


def run_control_database_independence_probe(
    socket_dir: Path,
    port: int,
    model_path: Path,
) -> None:
    control_dsn = f'host={socket_dir} port={port} dbname=template1'
    primary_database = 'ii42_application_primary'
    second_database = 'ii42_application_second'
    database_names = (primary_database, second_database)

    with psycopg.connect(control_dsn, autocommit=True) as control:
        with control.cursor() as cursor:
            cursor.execute(
                f'ALTER DATABASE postgres RENAME TO {primary_database}'
            )
            cursor.execute(
                f'CREATE DATABASE {second_database} TEMPLATE template0'
            )

        try:
            for database_name in database_names:
                dsn = (
                    f'host={socket_dir} port={port} '
                    f'dbname={database_name}'
                )
                with psycopg.connect(dsn, autocommit=True) as application:
                    with application.cursor() as cursor:
                        if database_name == second_database:
                            cursor.execute('CREATE EXTENSION ii42')
                        cursor.execute(
                            'SELECT ii42_runtime_service_query_atoms(%s, %s)',
                            (
                                str(model_path),
                                f'control database probe {database_name}',
                            ),
                        )
                        encoding = cursor.fetchone()[0]
                        if not isinstance(encoding, dict) or not encoding.get(
                            'atoms'
                        ):
                            raise AssertionError(
                                'runtime encoding failed without a postgres '
                                f'database: {database_name}={encoding}'
                            )
                        cursor.execute(
                            """
                            CREATE TABLE control_database_docs (
                                id integer PRIMARY KEY,
                                body text NOT NULL
                            )
                            """
                        )
                        cursor.execute(
                            """
                            INSERT INTO control_database_docs
                            SELECT
                                value,
                                'control database maintenance row ' || value
                            FROM generate_series(1, 32) AS value
                            """
                        )
                        cursor.execute(
                            """
                            CREATE INDEX control_database_docs_idx
                            ON control_database_docs
                            USING ii42 (body)
                            WITH (
                                sae = false,
                                consistency = eventual
                            )
                            """
                        )
                        cursor.execute(
                            """
                            UPDATE control_database_docs
                            SET body = body || ' changed'
                            WHERE id <= 4
                            """
                        )
                        cursor.execute(
                            'SELECT ii42_index_touch_maintenance()'
                        )

            deadline = time.monotonic() + 20.0
            last_states: dict[str, tuple[int, int]] = {}
            while time.monotonic() < deadline:
                last_states = {}
                for database_name in database_names:
                    dsn = (
                        f'host={socket_dir} port={port} '
                        f'dbname={database_name}'
                    )
                    with psycopg.connect(dsn, autocommit=True) as application:
                        with application.cursor() as cursor:
                            cursor.execute(
                                """
                                SELECT
                                    (
                                        ii42_index_status(
                                            'control_database_docs_idx'
                                                ::regclass
                                        )
                                        #>> '{details,delta_records}'
                                    )::integer,
                                    (
                                        ii42_index_status(
                                            'control_database_docs_idx'
                                                ::regclass
                                        )
                                        #>> '{details,pending_writes}'
                                    )::integer
                                """
                            )
                            delta_records, pending_writes = cursor.fetchone()
                            last_states[database_name] = (
                                int(delta_records),
                                int(pending_writes),
                            )
                if all(state == (0, 0) for state in last_states.values()):
                    break
                time.sleep(0.1)
            else:
                raise AssertionError(
                    'maintenance did not converge across application '
                    f'databases without postgres: {last_states}'
                )
        finally:
            for database_name in database_names:
                dsn = (
                    f'host={socket_dir} port={port} '
                    f'dbname={database_name}'
                )
                try:
                    with psycopg.connect(
                        dsn,
                        autocommit=True,
                    ) as application:
                        with application.cursor() as cursor:
                            cursor.execute(
                                'DROP TABLE IF EXISTS '
                                'control_database_docs CASCADE'
                            )
                except psycopg.Error:
                    pass
            with control.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s
                      AND pid <> pg_backend_pid()
                    """,
                    (second_database,),
                )
                cursor.execute(
                    f'DROP DATABASE IF EXISTS {second_database}'
                )
                cursor.execute(
                    f'ALTER DATABASE {primary_database} RENAME TO postgres'
                )


def run_maintenance_lock_contention_probe(
    socket_dir: Path,
    port: int,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    with (
        psycopg.connect(dsn, autocommit=False) as lock_conn,
        psycopg.connect(dsn, autocommit=True) as worker_conn,
    ):
        acquired = False
        for _ in range(50):
            with lock_conn.cursor() as lock_cur:
                lock_cur.execute(
                    """
                    SELECT ii42_index_try_maintenance_lock(
                        'docs_body_idx'::regclass
                    )
                    """,
                )
                acquired = lock_cur.fetchone()[0] is True
            if acquired:
                break
            lock_conn.rollback()
            time.sleep(0.1)
        if not acquired:
            raise AssertionError('failed to acquire maintenance lock')

        try:
            with worker_conn.cursor() as worker_cur:
                worker_cur.execute(
                    "SET ii42.test_convergent_l0_rotation_records = '1'"
                )
                worker_cur.execute(
                    """
                    SELECT ii42_index_status('docs_body_idx'::regclass)
                    """,
                )
                before_status = worker_cur.fetchone()[0]
                worker_cur.execute(
                    """
                    INSERT INTO docs (id, body)
                    VALUES ('doc-d', 'lock protected semantic lifecycle')
                    """,
                )
                worker_cur.execute(
                    """
                    SELECT ii42_index_try_maintain(
                        'docs_body_idx'::regclass
                    )
                    """,
                )
                busy_result = str(worker_cur.fetchone()[0])
                worker_cur.execute(
                    """
                    SELECT ii42_index_status('docs_body_idx'::regclass)
                    """,
                )
                pending_status = worker_cur.fetchone()[0]

            if 'reason=lock_busy' not in busy_result:
                raise AssertionError(
                    f'maintenance lock did not block worker: {busy_result}',
                )
            if pending_status['generation']['generation_id'] != (
                before_status['generation']['generation_id']
            ):
                raise AssertionError(
                    'blocked maintenance published a partial generation',
                )
            if int(pending_status['generation']['posting']['record_count']) != 3:
                raise AssertionError(
                    f'blocked unified generation changed: {pending_status}',
                )
            if int(pending_status['details']['pending_writes']) < 1:
                raise AssertionError(
                    f'insert did not enter unified lifecycle debt: {pending_status}',
                )
        finally:
            with lock_conn.cursor() as lock_cur:
                lock_cur.execute(
                    """
                    SELECT ii42_index_maintenance_unlock(
                        'docs_body_idx'::regclass
                    )
                    """,
                )
            lock_conn.commit()

        with worker_conn.cursor() as worker_cur:
            maintenance_results = []
            converged_status = pending_status
            for _ in range(32):
                worker_cur.execute(
                    """
                    SELECT ii42_index_maintain('docs_body_idx'::regclass)
                    """,
                )
                maintenance_results.append(str(worker_cur.fetchone()[0]))
                worker_cur.execute(
                    """
                    SELECT ii42_index_status('docs_body_idx'::regclass)
                    """,
                )
                converged_status = worker_cur.fetchone()[0]
                generation = converged_status['generation']
                semantic_completion = generation['delta'][
                    'semantic_completion'
                ]
                if (
                    generation['generation_id']
                    != before_status['generation']['generation_id']
                    and int(
                        converged_status['details']['delta_records']
                    ) == 0
                    and int(
                        converged_status['details']['pending_writes']
                    ) == 0
                    and bool(semantic_completion['converged'])
                ):
                    break
            else:
                raise AssertionError(
                    'maintenance lock probe did not converge: '
                    f'results={maintenance_results}, '
                    f'status={converged_status}'
                )
            worker_cur.execute(
                """
                SELECT source.id
                FROM ii42_query(
                    'docs_body_idx'::regclass,
                    'lock protected semantic lifecycle',
                    4
                ) AS hit
                JOIN docs AS source ON source.ctid = hit.ctid
                """,
            )
            result_ids = {str(row[0]) for row in worker_cur.fetchall()}

        generation = converged_status['generation']
        if generation['generation_id'] == before_status['generation'][
            'generation_id'
        ]:
            raise AssertionError(
                'maintenance did not publish a generation: '
                f'results={maintenance_results}, '
                f'status={converged_status}'
            )
        if not generation['atomic'] or not generation['valid']:
            raise AssertionError(
                f'converged generation is not atomic: {converged_status}',
            )
        if (
            int(generation['posting']['record_count']) != 4
            or not bool(
                generation['delta']['semantic_completion']['converged']
            )
        ):
            raise AssertionError(
                f'semantic section did not converge: {converged_status}',
            )
        if 'doc-d' not in result_ids:
            raise AssertionError(
                f'converged document is not queryable: {result_ids}',
            )
        with worker_conn.cursor() as worker_cur:
            worker_cur.execute(
                'RESET ii42.test_convergent_l0_rotation_records'
            )


def run_semantic_generation_reuse_probe(
    socket_dir: Path,
    port: int,
) -> None:
    dsn = f'host={socket_dir} port={port} dbname=postgres'
    with (
        psycopg.connect(dsn, autocommit=True) as lock_conn,
        psycopg.connect(dsn, autocommit=True) as worker_conn,
    ):
        def encoded_texts() -> int:
            with worker_conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT (
                        ii42_runtime_service_status()->>'encoded_texts'
                    )::int8
                    """
                )
                return int(cursor.fetchone()[0])

        def wait_runtime_idle() -> int:
            deadline = time.monotonic() + 10.0
            previous_encoded: int | None = None

            while time.monotonic() < deadline:
                with worker_conn.cursor() as cursor:
                    cursor.execute('SELECT ii42_runtime_service_status()')
                    status = dict(cursor.fetchone()[0])
                current_encoded = int(status.get('encoded_texts', -1))
                idle = (
                    int(status.get('queue_depth', -1)) == 0
                    and int(status.get('worker_count_busy', -1)) == 0
                    and int(status.get('response_slots_in_use', -1)) == 0
                    and int(
                        status.get('document_response_slots_in_use', -1)
                    ) == 0
                )
                if idle and current_encoded == previous_encoded:
                    return current_encoded
                previous_encoded = current_encoded if idle else None
                time.sleep(0.05)
            raise AssertionError(
                f'runtime service did not become idle: {status}'
            )

        def maintain_until_compacted(
            index_name: str,
            results_out: list[str] | None = None,
        ) -> dict[str, object]:
            deadline = time.monotonic() + 10.0
            results: list[str] = []
            last_status: dict[str, object] = {}

            while time.monotonic() < deadline:
                with worker_conn.cursor() as cursor:
                    cursor.execute(
                        'SELECT ii42_index_maintain(%s::regclass)',
                        (index_name,),
                    )
                    current_result = str(cursor.fetchone()[0])
                    results.append(current_result)
                    if results_out is not None:
                        results_out.append(results[-1])
                    cursor.execute(
                        'SELECT ii42_index_status(%s::regclass)',
                        (index_name,),
                    )
                    last_status = dict(cursor.fetchone()[0])
                details = dict(last_status.get('details', {}))
                generation = dict(last_status.get('generation', {}))
                delta = dict(generation.get('delta', {}))
                semantic_completion = dict(
                    delta.get('semantic_completion', {})
                )
                if (
                    int(details.get('delta_records', -1)) == 0
                    and int(details.get('pending_writes', -1)) == 0
                    and int(details.get('pending_deletes', -1)) == 0
                    and bool(semantic_completion.get('converged', False))
                    and 'reason=no_pending' in current_result
                ):
                    return last_status
                time.sleep(0.05)
            raise AssertionError(
                f'index maintenance did not converge: index={index_name}, '
                f'results={results}, status={last_status}'
            )

        before_initial_build = wait_runtime_idle()
        with worker_conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE semantic_reuse_docs (
                    id integer PRIMARY KEY,
                    body text NOT NULL
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO semantic_reuse_docs
                SELECT value, 'semantic reuse base ' || value
                FROM pg_catalog.generate_series(1, 8) AS value
                """
            )
            cursor.execute(
                """
                CREATE INDEX semantic_reuse_docs_idx
                ON semantic_reuse_docs USING ii42 (body)
                WITH (
                    sae = true,
                    consistency = eventual
                )
                """
            )
        after_initial_build = wait_runtime_idle()
        if after_initial_build - before_initial_build != 8:
            raise AssertionError(
                'initial SAE build did not encode each source document once: '
                f'before={before_initial_build}, '
                f'after={after_initial_build}'
            )

        def acquire_lock(
            index_name: str = 'semantic_reuse_docs_idx',
        ) -> None:
            deadline = time.monotonic() + 30.0
            last_status: dict[str, object] = {}
            while time.monotonic() < deadline:
                with lock_conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT ii42_index_try_maintenance_lock(%s::regclass)
                        """,
                        (index_name,),
                    )
                    if cursor.fetchone()[0] is True:
                        return
                with worker_conn.cursor() as cursor:
                    cursor.execute(
                        'SELECT ii42_index_status(%s::regclass)',
                        (index_name,),
                    )
                    last_status = dict(cursor.fetchone()[0])
                time.sleep(0.05)
            raise AssertionError(
                f'failed to lock semantic reuse index {index_name}: '
                f'status={last_status}'
            )

        def release_lock(
            index_name: str = 'semantic_reuse_docs_idx',
        ) -> None:
            with lock_conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT ii42_index_maintenance_unlock(%s::regclass)
                    """,
                    (index_name,),
                )

        def status_and_pages() -> tuple[dict[str, object], int]:
            with worker_conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        ii42_index_status(
                            'semantic_reuse_docs_idx'::regclass
                        ) || jsonb_build_object(
                            'generation',
                            ii42_index_generation_audit_internal(
                                'semantic_reuse_docs_idx'::regclass
                            )
                        ),
                        pages
                    FROM ii42_index_details(
                        'semantic_reuse_docs_idx'::regclass
                    )
                    """
                )
                row = cursor.fetchone()
            return row[0], int(row[1])

        def search_ids(query: str) -> set[int]:
            with worker_conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT docs.id
                    FROM ii42_query(
                        'semantic_reuse_docs_idx'::regclass,
                        %s,
                        100
                    ) AS hit
                    JOIN semantic_reuse_docs AS docs
                      ON docs.ctid = hit.ctid
                    """,
                    (query,),
                )
                function_ids = {
                    int(row[0])
                    for row in cursor.fetchall()
                }
                cursor.execute(
                    """
                    SELECT id
                    FROM semantic_reuse_docs
                    ORDER BY ii42_query(
                        'semantic_reuse_docs_idx'::regclass,
                        %s
                    ) DESC
                    LIMIT 100
                    """,
                    (query,),
                )
                planner_ids = {
                    int(row[0])
                    for row in cursor.fetchall()
                }
                if planner_ids != function_ids:
                    raise AssertionError(
                        'planner-native lifecycle query diverged: '
                        f'query={query!r}, planner={planner_ids}, '
                        f'function={function_ids}'
                    )
                return function_ids

        with worker_conn.cursor() as cursor:
            cursor.execute(
                "SET ii42.test_convergent_l0_rotation_records = '1'"
            )
        saw_reuse = False
        saw_truncate = False

        for cycle in range(1, 5):
            acquire_lock()
            try:
                locked_status, _ = status_and_pages()
                cycle_generation = locked_status['generation'][
                    'generation_id'
                ]
                before_insert = wait_runtime_idle()
                with worker_conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO semantic_reuse_docs (id, body)
                        VALUES (%s, %s)
                        """,
                        (
                            8 + cycle,
                            f'semantic reuse committed cycle {cycle}',
                        ),
                    )
                    cursor.execute(
                        """
                        SELECT ii42_index_try_maintain(
                            'semantic_reuse_docs_idx'::regclass
                        )
                        """
                    )
                    busy_result = str(cursor.fetchone()[0])
                after_insert = wait_runtime_idle()
                if after_insert != before_insert:
                    raise AssertionError(
                        'eventual SAE insert performed foreground inference: '
                        f'before={before_insert}, '
                        f'after={after_insert}'
                    )
                pending_status, _ = status_and_pages()
                if 'reason=lock_busy' not in busy_result:
                    raise AssertionError(
                        'semantic reuse maintenance ignored its lock: '
                        f'{busy_result}'
                    )
                pending_generation = pending_status['generation']
                pending_delta = pending_generation.get('delta')
                expected_doc_id = 8 + cycle
                pending_result_ids = search_ids(
                    f'semantic reuse committed cycle {cycle}'
                )
                if (
                    pending_generation['generation_id'] != cycle_generation
                    or not pending_generation['atomic']
                    or not pending_generation['valid']
                    or int(pending_status['details']['pending_writes']) < 1
                    or int(pending_status['details']['delta_records']) != 1
                    or int(pending_status['details']['delta_bytes']) <= 0
                    or not isinstance(pending_delta, dict)
                    or int(pending_delta.get('records', -1)) != 1
                    or int(pending_delta.get('bytes', 0)) <= 0
                    or int(
                        pending_delta['semantic_completion']['pending']
                    ) != 1
                    or bool(
                        pending_delta['semantic_completion']['converged']
                    )
                    or pending_generation['posting']['role']
                    != 'unified_posting'
                    or expected_doc_id not in pending_result_ids
                ):
                    raise AssertionError(
                        'pending SAE mutation did not expose one unified '
                        f'delta: status={pending_status}, '
                        f'results={pending_result_ids}'
                    )
            finally:
                release_lock()

            before_completion = wait_runtime_idle()
            with worker_conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT ii42_index_try_maintain(
                        'semantic_reuse_docs_idx'::regclass
                    )
                    """
                )
                completion_result = str(cursor.fetchone()[0])
            after_completion = encoded_texts()
            compaction_results: list[str] = []
            converged_status = maintain_until_compacted(
                'semantic_reuse_docs_idx',
                compaction_results,
            )
            after_compaction = wait_runtime_idle()
            converged_status, relation_pages = status_and_pages()
            generation = converged_status['generation']
            primary = generation['primary']
            physical_blocks = int(primary['physical_blocks'])
            reachable_blocks = int(primary['reachable_blocks'])
            page_budget = max(reachable_blocks * 3 + 16, 32)
            expected_docs = 8 + cycle
            maintenance_results = [
                completion_result,
                *compaction_results,
            ]
            maintenance_summary = ' '.join(maintenance_results)
            if (
                'maintained=true' not in maintenance_summary
                or after_compaction - before_completion != 1
                or not generation['atomic']
                or not generation['valid']
                or generation['generation_id'] == cycle_generation
                or int(generation['docs']) != expected_docs
                or int(generation['posting']['record_count']) != expected_docs
                or not bool(
                    generation['delta']['semantic_completion']['converged']
                )
                or relation_pages != physical_blocks
                or relation_pages > page_budget
            ):
                raise AssertionError(
                    'unified generation did not converge within its storage '
                    f'bound: results={maintenance_results}, '
                    f'status={converged_status}, pages={relation_pages}, '
                    f'budget={page_budget}, '
                    f'encoded_before={before_completion}, '
                    f'encoded_after_completion={after_completion}, '
                    f'encoded_after_compaction={after_compaction}'
                )
            saw_reuse = (
                saw_reuse
                or 'segment_reused=true' in maintenance_summary
                or re.search(
                    r'reused_blocks=[1-9][0-9]*', maintenance_summary
                ) is not None
            )
            saw_truncate = (
                saw_truncate
                or 'retired_tail_truncated=true' in maintenance_summary
                or 'reason=unpublished_tail_truncated'
                    in maintenance_summary
            )
        if not saw_reuse:
            raise AssertionError(
                'unified generations did not reuse retired storage: '
                f'reuse={saw_reuse}, truncate={saw_truncate}'
            )

        acquire_lock()
        try:
            before_update = wait_runtime_idle()
            with worker_conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE semantic_reuse_docs
                    SET body = 'semantic reuse updated boundary document'
                    WHERE id = 2
                    """
                )
            after_update = wait_runtime_idle()
            pending_update_status, _ = status_and_pages()
            update_result_ids = search_ids(
                'semantic reuse updated boundary document'
            )
            if (
                after_update != before_update
                or int(
                    pending_update_status['details']['pending_writes']
                ) < 1
                or int(
                    pending_update_status['details']['delta_records']
                ) != 1
                or int(
                    pending_update_status['generation']['delta'][
                        'semantic_completion'
                    ]['pending']
                ) != 1
                or 2 not in update_result_ids
            ):
                raise AssertionError(
                    'eventual SAE update did not expose one lexical pending '
                    'record without foreground inference: '
                    f'before={before_update}, after={after_update}, '
                    f'status={pending_update_status}, '
                    f'results={update_result_ids}'
                )
        finally:
            release_lock()

        before_update_completion = wait_runtime_idle()
        with worker_conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT ii42_index_maintain(
                    'semantic_reuse_docs_idx'::regclass
                )
                """
            )
            update_completion_result = str(cursor.fetchone()[0])
        after_update_completion = encoded_texts()
        update_compaction_results: list[str] = []
        updated_status = maintain_until_compacted(
            'semantic_reuse_docs_idx',
            update_compaction_results,
        )
        after_update_compaction = wait_runtime_idle()
        update_maintenance_results = [
            update_completion_result,
            *update_compaction_results,
        ]
        update_maintenance_summary = ' '.join(update_maintenance_results)
        if (
            after_update_compaction - before_update_completion != 1
            or not bool(
                updated_status['generation']['delta'][
                    'semantic_completion'
                ]['converged']
            )
            # The pre-VACUUM generation retains both UPDATE TIDs so an older
            # snapshot can still resolve the prior tuple version.
            or int(updated_status['generation']['docs']) != 13
            or int(
                updated_status['generation']['posting']['record_count']
            ) != 13
            or int(updated_status['details']['delta_records']) != 0
        ):
            raise AssertionError(
                'eventual SAE update did not complete exactly once before '
                'compaction: '
                f'results={update_maintenance_results}, '
                f'status={updated_status}, '
                f'encoded_before={before_update_completion}, '
                f'encoded_after_completion={after_update_completion}, '
                f'encoded_after_compaction={after_update_compaction}'
            )

        before_delete = wait_runtime_idle()
        with worker_conn.cursor() as cursor:
            cursor.execute('DELETE FROM semantic_reuse_docs WHERE id = 1')
        after_delete = wait_runtime_idle()
        if after_delete != before_delete:
            raise AssertionError(
                'SAE delete unexpectedly invoked document inference: '
                f'before={before_delete}, after={after_delete}'
            )

        acquire_lock()
        try:
            vacuum_with_session_maintenance_lock(
                lock_conn,
                'semantic_reuse_docs',
            )
            pending_delete_status, _ = status_and_pages()
            pending_delete_generation = pending_delete_status['generation']
            pending_delete_delta = pending_delete_generation.get('delta')
            deleted_result_ids = search_ids('semantic reuse base 1')
            pending_delete_records = int(
                pending_delete_status['details']['delta_records']
            )
            pending_delete_bytes = int(
                pending_delete_status['details']['delta_bytes']
            )
            if (
                int(
                    pending_delete_status['details']['pending_deletes']
                ) < 1
                or pending_delete_records < 1
                or pending_delete_bytes < 1
                or not isinstance(pending_delete_delta, dict)
                or int(pending_delete_delta.get('records', -1))
                != pending_delete_records
                or int(pending_delete_delta.get('bytes', -1))
                != pending_delete_bytes
                or 1 in deleted_result_ids
            ):
                raise AssertionError(
                    'SAE delete/vacuum did not expose the relation-owned '
                    f'tombstone delta: status={pending_delete_status}, '
                    f'results={deleted_result_ids}, '
                    f'encoded_before={before_delete}, '
                    f'encoded_after={after_delete}'
                )
        finally:
            release_lock()

        before_delete_maintain = wait_runtime_idle()
        with worker_conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT ii42_index_try_maintain(
                    'semantic_reuse_docs_idx'::regclass
                )
                """
            )
            delete_maintain_result = str(cursor.fetchone()[0])
        delete_compaction_results: list[str] = []
        final_status = maintain_until_compacted(
            'semantic_reuse_docs_idx',
            delete_compaction_results,
        )
        after_delete_maintain = wait_runtime_idle()
        delete_maintenance_summary = ' '.join(
            [delete_maintain_result, *delete_compaction_results]
        )
        final_status, final_pages = status_and_pages()
        final_generation = final_status['generation']
        final_primary = final_generation['primary']
        final_physical_blocks = int(final_primary['physical_blocks'])
        final_reachable_blocks = int(final_primary['reachable_blocks'])
        if (
            'maintained=true' not in delete_maintenance_summary
            or after_delete_maintain != before_delete_maintain
            or int(final_generation['docs']) != 11
            or int(final_generation['posting']['record_count']) != 11
            or int(final_status['details']['pending_deletes']) != 0
            or int(final_status['details']['delta_records']) != 0
            or int(final_status['details']['delta_bytes']) != 0
            or final_pages != final_physical_blocks
            or final_pages > max(final_reachable_blocks * 3 + 16, 32)
        ):
            raise AssertionError(
                'SAE delete/vacuum did not converge atomically: '
                f'results={delete_maintenance_summary}, '
                f'status={final_status}, '
                f'pages={final_pages}, '
                f'encoded_before={before_delete_maintain}, '
                f'encoded_after={after_delete_maintain}'
            )

        before_reindex = wait_runtime_idle()
        with worker_conn.cursor() as cursor:
            cursor.execute('REINDEX INDEX semantic_reuse_docs_idx')
        after_reindex = wait_runtime_idle()
        if after_reindex - before_reindex != 11:
            raise AssertionError(
                'explicit SAE REINDEX did not globally re-encode the live '
                f'corpus: before={before_reindex}, after={after_reindex}'
            )

        with worker_conn.cursor() as cursor:
            cursor.execute('DROP TABLE semantic_reuse_docs')
            cursor.execute(
                'RESET ii42.test_convergent_l0_rotation_records'
            )



def run_planner_native_scale_probe(
    socket_dir: Path,
    port: int,
    document_count: int,
) -> None:
    if document_count < 1_000:
        raise ValueError('--planner-native-scale-docs must be at least 1000')
    dsn = (
        f'host={socket_dir} port={port} dbname=postgres '
        'user=' + os.environ.get('USER', '')
    )
    query = 'graph neural retrieval optimization'
    limit = 20
    natural_sql = '''
        SELECT id,
               ii42_query(
                   'planner_scale_docs_idx'::regclass,
                   %s,
                   ARRAY['title', 'body']::text[],
                   ARRAY[0.75, 1.25]::real[]
               ) AS score
        FROM planner_scale_docs
        WHERE bucket < %s::int4
        ORDER BY score DESC
        LIMIT %s
    '''
    oracle_sql = '''
        SELECT source.id, hit.score
        FROM ii42_query(
            'planner_scale_docs_idx'::regclass,
            %s,
            ARRAY['title', 'body']::text[],
            ARRAY[0.75, 1.25]::real[],
            ARRAY(
                SELECT ctid
                FROM planner_scale_docs
                WHERE bucket < %s::int4
            ),
            %s
        ) WITH ORDINALITY AS hit(ctid, doc_id, score, rank)
        JOIN planner_scale_docs AS source ON source.ctid = hit.ctid
        ORDER BY hit.rank
    '''
    results: list[dict[str, object]] = []

    def find_ii42_node(node: dict[str, object]) -> dict[str, object]:
        if node.get('Custom Plan Provider') == 'II42 Search':
            return node
        for child in node.get('Plans', []):
            if isinstance(child, dict):
                match = find_ii42_node(child)
                if match:
                    return match
        return {}

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute('DROP TABLE IF EXISTS planner_scale_docs CASCADE')
            cursor.execute('''
                CREATE TABLE planner_scale_docs (
                    id bigint PRIMARY KEY,
                    title text NOT NULL,
                    body text NOT NULL,
                    bucket int NOT NULL
                )
            ''')
            cursor.execute(
                '''
                INSERT INTO planner_scale_docs (id, title, body, bucket)
                SELECT value,
                       'graph neural document ' || value,
                       CASE value %% 5
                           WHEN 0 THEN 'semantic retrieval optimization'
                           WHEN 1 THEN 'database indexing systems'
                           WHEN 2 THEN 'neural information retrieval'
                           WHEN 3 THEN 'graph learning methods'
                           ELSE 'scientific document ranking'
                       END || ' sample ' || value,
                       value %% 1000
                FROM generate_series(1, %s) AS value
                ''',
                (document_count,),
            )
            cursor.execute(
                'CREATE INDEX planner_scale_docs_bucket_idx '
                'ON planner_scale_docs (bucket)'
            )
            cursor.execute('''
                CREATE INDEX planner_scale_docs_idx
                ON planner_scale_docs
                USING ii42 (title, body)
                INCLUDE (bucket)
                WITH (
                    sae = true,
                    field_aware = true,
                    consistency = eventual
                )
            ''')
            cursor.execute('ANALYZE planner_scale_docs')
            cursor.execute('SET ii42.enable_planner_native = on')

            maintenance_results: list[str] = []
            status: dict[str, object] = {}
            for attempt in range(240):
                cursor.execute(
                    "SELECT ii42_index_status("
                    "'planner_scale_docs_idx'::regclass)"
                )
                status = dict(cursor.fetchone()[0])
                generation = dict(status.get('generation', {}))
                semantic = dict(
                    generation.get('semantic_accelerator', {})
                )
                if bool(semantic.get('scope_present', False)):
                    break
                if attempt == 239:
                    raise AssertionError(
                        'scale scope baseline did not become available: '
                        f'maintenance={maintenance_results}, '
                        f'status={status}'
                    )
                cursor.execute(
                    "SELECT ii42_index_maintain("
                    "'planner_scale_docs_idx'::regclass)"
                )
                maintenance_results.append(str(cursor.fetchone()[0]))
                time.sleep(0.05)

            cursor.execute('SET enable_seqscan = off')
            cursor.execute('SET enable_indexscan = off')
            cursor.execute('SET enable_bitmapscan = on')
            cursor.execute(
                '''
                EXPLAIN (FORMAT JSON, COSTS false)
                SELECT id
                FROM planner_scale_docs
                WHERE bucket < 500::int4
                ORDER BY ii42_query(
                    'planner_scale_docs_idx'::regclass,
                    %s,
                    ARRAY['title', 'body']::text[],
                    ARRAY[0.75, 1.25]::real[]
                ) DESC
                LIMIT 20
                ''',
                (query,),
            )
            bitmap_plan = cursor.fetchone()[0]
            if 'II42 Search' not in json.dumps(bitmap_plan):
                raise AssertionError(
                    'bitmap filter did not use planner-native path: '
                    f'{bitmap_plan}'
                )
            cursor.execute(
                '''
                EXPLAIN (FORMAT JSON, COSTS false)
                SELECT id
                FROM planner_scale_docs
                WHERE id > %s
                ORDER BY ii42_query(
                    'planner_scale_docs_idx'::regclass,
                    %s,
                    ARRAY['title', 'body']::text[],
                    ARRAY[0.75, 1.25]::real[]
                ) DESC
                LIMIT 20
                ''',
                (document_count // 2, query),
            )
            indexed_filter_plan = cursor.fetchone()[0]
            if 'II42 Search' not in json.dumps(indexed_filter_plan):
                raise AssertionError(
                    'indexed filter did not use planner-native path: '
                    f'{indexed_filter_plan}'
                )
            cursor.execute('RESET enable_seqscan')
            cursor.execute('RESET enable_indexscan')
            cursor.execute('RESET enable_bitmapscan')

            thresholds = (0, 1, 10, 100, 500, 1000)
            for threshold in thresholds:
                cursor.execute(
                    'SELECT count(*) FROM planner_scale_docs '
                    'WHERE bucket < %s::int4',
                    (threshold,),
                )
                allowed_rows = int(cursor.fetchone()[0])
                cursor.execute(
                    oracle_sql,
                    (query, threshold, limit),
                )
                oracle_rows = cursor.fetchall()
                cursor.execute(
                    natural_sql,
                    (query, threshold, limit),
                )
                natural_rows = cursor.fetchall()
                if natural_rows != oracle_rows:
                    raise AssertionError(
                        'planner-native scale exactness mismatch: '
                        f'threshold={threshold}, natural={natural_rows}, '
                        f'oracle={oracle_rows}'
                    )
                natural_timings_ms: list[float] = []
                for _ in range(6):
                    started = time.perf_counter()
                    cursor.execute(
                        natural_sql,
                        (query, threshold, limit),
                    )
                    cursor.fetchall()
                    natural_timings_ms.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                cursor.execute('SELECT ii42_query_trace_internal()')
                trace = cursor.fetchone()[0]
                oracle_timings_ms: list[float] = []
                predicate_timings_ms: list[float] = []
                for _ in range(6):
                    started = time.perf_counter()
                    cursor.execute(
                        oracle_sql,
                        (query, threshold, limit),
                    )
                    cursor.fetchall()
                    oracle_timings_ms.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    started = time.perf_counter()
                    cursor.execute(
                        'SELECT count(*) FROM planner_scale_docs '
                        'WHERE bucket < %s::int4',
                        (threshold,),
                    )
                    cursor.fetchone()
                    predicate_timings_ms.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                cursor.execute(
                    '''
                    EXPLAIN (
                        ANALYZE true,
                        FORMAT JSON,
                        COSTS true,
                        TIMING false,
                        SUMMARY false
                    )
                    SELECT id
                    FROM planner_scale_docs
                    WHERE bucket < %s::int4
                    ORDER BY ii42_query(
                        'planner_scale_docs_idx'::regclass,
                        %s,
                        ARRAY['title', 'body']::text[],
                        ARRAY[0.75, 1.25]::real[]
                    ) DESC
                    LIMIT 20
                    ''',
                    (threshold, query),
                )
                plan = cursor.fetchone()[0]
                if 'II42 Search' not in json.dumps(plan):
                    raise AssertionError(
                        'scale query did not use planner-native path: '
                        f'{plan}'
                    )
                ii42_plan = find_ii42_node(plan[0]['Plan'])
                expected_fallback = allowed_rows < limit
                scope_contract = {
                    'eligible': ii42_plan.get('Scope Filter Eligible'),
                    'probes': ii42_plan.get('Scope Filter Probes'),
                    'complete': ii42_plan.get('Scope Filter Complete'),
                    'fallback': ii42_plan.get('Scope Filter Fallback'),
                }
                if scope_contract != {
                    'eligible': True,
                    'probes': 1,
                    'complete': not expected_fallback,
                    'fallback': expected_fallback,
                }:
                    raise AssertionError(
                        'scale scope route did not satisfy its contract: '
                        f'threshold={threshold}, allowed_rows={allowed_rows}, '
                        f'observed={scope_contract}, plan={plan}'
                    )
                results.append({
                    'threshold': threshold,
                    'allowed_rows': allowed_rows,
                    'selectivity': allowed_rows / document_count,
                    'natural_p50_ms': statistics.median(
                        natural_timings_ms[1:]
                    ),
                    'natural_max_ms': max(natural_timings_ms[1:]),
                    'oracle_p50_ms': statistics.median(
                        oracle_timings_ms[1:]
                    ),
                    'predicate_p50_ms': statistics.median(
                        predicate_timings_ms[1:]
                    ),
                    'scope_filter_eligible': ii42_plan.get(
                        'Scope Filter Eligible'
                    ),
                    'scope_filter_probes': ii42_plan.get(
                        'Scope Filter Probes'
                    ),
                    'scope_filter_candidates': ii42_plan.get(
                        'Scope Filter Candidates'
                    ),
                    'scope_filter_matches': ii42_plan.get(
                        'Scope Filter Matches'
                    ),
                    'scope_filter_complete': ii42_plan.get(
                        'Scope Filter Complete'
                    ),
                    'scope_filter_fallback': ii42_plan.get(
                        'Scope Filter Fallback'
                    ),
                    'plan_startup_cost': ii42_plan.get('Startup Cost'),
                    'plan_total_cost': ii42_plan.get('Total Cost'),
                    'trace': trace,
                })
            cursor.execute('RESET ii42.enable_planner_native')
            cursor.execute('DROP TABLE planner_scale_docs CASCADE')
    print(json.dumps({
        'planner_native_scale_docs': document_count,
        'results': results,
    }, sort_keys=True, default=str))


def run_planner_scope_filter_probe(
    socket_dir: Path,
    port: int,
) -> None:
    dsn = (
        f'host={socket_dir} port={port} dbname=postgres '
        'user=' + os.environ.get('USER', '')
    )
    query = 'cancer immunotherapy survival'
    natural_sql = '''
        SELECT id,
               ii42_query(
                   'planner_scope_docs_idx'::regclass,
                   %s,
                   ARRAY['title', 'abstract']::text[],
                   ARRAY[2.0, 1.0]::real[]
               ) AS score
        FROM planner_scope_docs
        WHERE publish_day BETWEEN %s AND %s
          AND categories && %s::text[]
          AND source_name ILIKE %s
        ORDER BY score DESC
        LIMIT 5
    '''
    oracle_sql = '''
        SELECT source.id, hit.score
        FROM ii42_query(
            'planner_scope_docs_idx'::regclass,
            %s,
            ARRAY['title', 'abstract']::text[],
            ARRAY[2.0, 1.0]::real[],
            ARRAY(
                SELECT ctid
                FROM planner_scope_docs
                WHERE publish_day BETWEEN %s AND %s
                  AND categories && %s::text[]
                  AND source_name ILIKE %s
            ),
            5
        ) WITH ORDINALITY AS hit(ctid, doc_id, score, rank)
        JOIN planner_scope_docs AS source ON source.ctid = hit.ctid
        ORDER BY hit.rank
    '''
    structured_sql = '''
        SELECT source.id, hit.score
        FROM ii42_query(
            'planner_scope_docs_idx'::regclass,
            %s,
            ARRAY['title', 'abstract']::text[],
            ARRAY[2.0, 1.0]::real[],
            %s::jsonb,
            5
        ) AS hit
        JOIN planner_scope_docs AS source ON source.ctid = hit.ctid
        ORDER BY hit.score DESC, source.id
    '''
    structured_filters = json.dumps({
        'publish_day': {
            'range': {'gte': 20240101, 'lte': 20241231},
        },
        'categories': {'overlap': ['Article']},
        'source_name': {'ilike': '%research%'},
    })
    parameters = (
        query,
        20240101,
        20241231,
        ['Article'],
        '%research%',
    )

    def find_ii42_node(node: dict[str, object]) -> dict[str, object]:
        if node.get('Custom Plan Provider') == 'II42 Search':
            return node
        for child in node.get('Plans', []):
            if isinstance(child, dict):
                match = find_ii42_node(child)
                if match:
                    return match
        return {}

    def assert_scope_plan(
        cursor: psycopg.Cursor[object],
        expected_fallback: bool = False,
    ) -> dict[str, object]:
        cursor.execute(
            'EXPLAIN (ANALYZE, FORMAT JSON, COSTS false, '
            'TIMING false, SUMMARY false) ' + natural_sql,
            parameters,
        )
        plan = cursor.fetchone()[0]
        ii42_node = find_ii42_node(plan[0]['Plan'])
        expected = {
            'Scope Filter Eligible': True,
            'Scope Filter Probes': 1,
            'Scope Filter Complete': not expected_fallback,
            'Scope Filter Fallback': expected_fallback,
        }
        observed = {
            key: ii42_node.get(key)
            for key in expected
        }
        if observed != expected:
            raise AssertionError(
                'planner scope route did not satisfy its contract: '
                f'observed={observed}, plan={plan}'
            )
        return ii42_node

    with psycopg.connect(dsn, autocommit=True) as conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'DROP TABLE IF EXISTS planner_scope_docs CASCADE'
                )
                cursor.execute('''
                    CREATE TABLE planner_scope_docs (
                        id bigint PRIMARY KEY,
                        title text NOT NULL,
                        abstract text NOT NULL,
                        publish_day int NOT NULL,
                        categories text[] NOT NULL,
                        source_name text NOT NULL
                    )
                ''')
                cursor.execute(
                    '''
                    INSERT INTO planner_scope_docs (
                        id,
                        title,
                        abstract,
                        publish_day,
                        categories,
                        source_name
                    )
                    SELECT value,
                           CASE mod(value, 4)
                               WHEN 0 THEN
                                   'cancer immunotherapy survival outcome'
                               WHEN 1 THEN
                                   'cancer therapy clinical survival'
                               WHEN 2 THEN
                                   'immunotherapy biomarkers oncology'
                               ELSE
                                   'database indexing systems'
                           END,
                           'sample abstract ' || value ||
                               ' with clinical evidence and outcomes',
                           20240100 + value,
                           CASE
                               WHEN value <= 30
                               THEN ARRAY['Article']::text[]
                               ELSE ARRAY['Review']::text[]
                           END,
                           CASE
                               WHEN value <= 35 THEN 'Research Archive'
                               ELSE 'General Feed'
                           END
                    FROM generate_series(1, 40) AS value
                    ''',
                )
                cursor.execute('''
                    CREATE INDEX planner_scope_docs_idx
                    ON planner_scope_docs
                    USING ii42 (title, abstract)
                    INCLUDE (publish_day, categories, source_name)
                    WITH (
                        sae = true,
                        field_aware = true,
                        consistency = eventual
                    )
                ''')
                cursor.execute('ANALYZE planner_scope_docs')
                cursor.execute('SET ii42.enable_planner_native = on')

                maintenance_results: list[str] = []
                status: dict[str, object] = {}
                for attempt in range(80):
                    cursor.execute(
                        "SELECT ii42_index_status("
                        "'planner_scope_docs_idx'::regclass)"
                    )
                    status = dict(cursor.fetchone()[0])
                    generation = dict(status.get('generation', {}))
                    semantic = dict(
                        generation.get('semantic_accelerator', {})
                    )
                    if bool(semantic.get('scope_present', False)):
                        break
                    if attempt == 79:
                        raise AssertionError(
                            'scope baseline did not become available: '
                            f'maintenance={maintenance_results}, '
                            f'status={status}'
                        )
                    cursor.execute(
                        "SELECT ii42_index_maintain("
                        "'planner_scope_docs_idx'::regclass)"
                    )
                    maintenance_results.append(str(cursor.fetchone()[0]))
                    time.sleep(0.05)

                cursor.execute(oracle_sql, parameters)
                oracle_rows = cursor.fetchall()
                cursor.execute(natural_sql, parameters)
                natural_rows = cursor.fetchall()
                if natural_rows != oracle_rows:
                    raise AssertionError(
                        'planner scope result differs from exact TID oracle: '
                        f'natural={natural_rows}, oracle={oracle_rows}'
                    )
                scope_plan = assert_scope_plan(cursor)
                if int(scope_plan.get('Scope Filter Matches', -1)) < 5:
                    raise AssertionError(
                        'planner scope route did not produce enough matches: '
                        f'{scope_plan}'
                    )

                cursor.execute('SET plan_cache_mode = force_generic_plan')
                cursor.execute('''
                    PREPARE planner_scope_generic(
                        text,
                        int,
                        int,
                        text[],
                        text
                    ) AS
                    SELECT id,
                           ii42_query(
                               'planner_scope_docs_idx'::regclass,
                               $1,
                               ARRAY['title', 'abstract']::text[],
                               ARRAY[2.0, 1.0]::real[]
                           ) AS score
                    FROM planner_scope_docs
                    WHERE publish_day BETWEEN $2 AND $3
                      AND categories && $4
                      AND source_name ILIKE $5
                    ORDER BY score DESC
                    LIMIT 5
                ''')
                cursor.execute('''
                    EXECUTE planner_scope_generic(
                        'cancer immunotherapy survival',
                        20240101,
                        20241231,
                        ARRAY['Article']::text[],
                        '%research%'
                    )
                ''')
                generic_rows = cursor.fetchall()
                cursor.execute('DEALLOCATE planner_scope_generic')
                cursor.execute('RESET plan_cache_mode')
                if generic_rows != oracle_rows:
                    raise AssertionError(
                        'generic planner scope result differs from oracle: '
                        f'generic={generic_rows}, oracle={oracle_rows}'
                    )

                stale_id = int(oracle_rows[0][0])
                cursor.execute('BEGIN')
                try:
                    cursor.execute(
                        'UPDATE planner_scope_docs '
                        "SET categories = ARRAY['Review']::text[] "
                        'WHERE id = %s',
                        (stale_id,),
                    )
                    cursor.execute(
                        '''
                        INSERT INTO planner_scope_docs (
                            id,
                            title,
                            abstract,
                            publish_day,
                            categories,
                            source_name
                        ) VALUES (
                            1001,
                            'cancer immunotherapy survival perfect match',
                            'new post-baseline clinical evidence',
                            20240601,
                            ARRAY['Article']::text[],
                            'Research Archive'
                        )
                        ''',
                    )
                    cursor.execute(natural_sql, parameters)
                    stale_rows = cursor.fetchall()
                    stale_ids = {int(row[0]) for row in stale_rows}
                    if (
                        len(stale_rows) != 5
                        or stale_id in stale_ids
                        or 1001 in stale_ids
                    ):
                        raise AssertionError(
                            'stale scope baseline did not preserve exact '
                            'membership with bounded delta omission: '
                            f'stale_id={stale_id}, rows={stale_rows}'
                        )
                    assert_scope_plan(cursor)
                    cursor.execute(
                        structured_sql,
                        (query, structured_filters),
                    )
                    structured_rows = cursor.fetchall()
                    structured_ids = {
                        int(row[0]) for row in structured_rows
                    }
                    if (
                        len(structured_rows) != 5
                        or stale_id in structured_ids
                        or 1001 in structured_ids
                    ):
                        raise AssertionError(
                            'structured stale scope did not preserve current '
                            'candidate membership with bounded delta omission: '
                            f'stale_id={stale_id}, rows={structured_rows}'
                        )
                    cursor.execute('SELECT ii42_query_trace_internal()')
                    structured_trace = dict(cursor.fetchone()[0])
                    if structured_trace.get('query_route') != 'scope_filter':
                        raise AssertionError(
                            'structured query did not use the serving scope: '
                            f'{structured_trace}'
                        )
                    cursor.execute('SAVEPOINT bounded_scope_shortfall')
                    cursor.execute('''
                        UPDATE planner_scope_docs
                        SET source_name = 'General Feed'
                        WHERE id <= 30
                          AND id NOT IN (1, 2, 3, 4)
                    ''')
                    cursor.execute(
                        structured_sql,
                        (query, structured_filters),
                    )
                    shortfall_rows = cursor.fetchall()
                    if len(shortfall_rows) >= 5:
                        raise AssertionError(
                            'structured serving scope unexpectedly filled the '
                            f'bounded shortfall: rows={shortfall_rows}'
                        )
                    shortfall_ids = [int(row[0]) for row in shortfall_rows]
                    if 1001 in shortfall_ids:
                        raise AssertionError(
                            'structured serving scope admitted a post-baseline '
                            f'row: rows={shortfall_rows}'
                        )
                    if shortfall_ids:
                        cursor.execute(
                            '''
                            SELECT count(*)
                            FROM planner_scope_docs
                            WHERE id = ANY(%s::int[])
                              AND publish_day BETWEEN 20240101 AND 20241231
                              AND categories && ARRAY['Article']::text[]
                              AND source_name ILIKE '%%research%%'
                            ''',
                            (shortfall_ids,),
                        )
                        if int(cursor.fetchone()[0]) != len(shortfall_ids):
                            raise AssertionError(
                                'structured serving scope returned a stale '
                                f'candidate: rows={shortfall_rows}'
                            )
                    cursor.execute('SELECT ii42_query_trace_internal()')
                    shortfall_trace = dict(cursor.fetchone()[0])
                    if (
                        shortfall_trace.get('query_route') != 'scope_filter'
                        or shortfall_trace.get('sql_filter_residual') is True
                        or shortfall_trace.get('filter_probe_attempted') is True
                    ):
                        raise AssertionError(
                            'structured serving scope escaped to the current '
                            f'predicate universe: {shortfall_trace}'
                        )
                    cursor.execute('ROLLBACK TO SAVEPOINT bounded_scope_shortfall')
                finally:
                    cursor.execute('ROLLBACK')

                cursor.execute('RESET ii42.enable_planner_native')
                cursor.execute('DROP TABLE planner_scope_docs CASCADE')
        finally:
            with conn.cursor() as cursor:
                cursor.execute(
                    'DROP TABLE IF EXISTS planner_scope_docs CASCADE'
                )


def run_app_role_product_query(
    psql_bin: Path,
    socket_dir: Path,
    port: int,
) -> None:
    output = psql(
        psql_bin,
        socket_dir,
        port,
        """
        SET ROLE ii42_app_user;
        SELECT count(*)
        FROM ii42_query(
            'docs_body_idx'::regclass,
            'alpha semantic optimization ordinary app role query',
            3
        );
        SELECT count(*)
        FROM (
            SELECT id
            FROM docs
            ORDER BY ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization ordinary app role query'
            ) DESC
            LIMIT 3
        ) AS ranked;
        SELECT
            ii42_index_status('docs_body_idx'::regclass)->>'query_ready',
            ii42_index_status('docs_body_idx'::regclass)
                #>>'{generation,atomic}';
        RESET ROLE;
        """,
    ).splitlines()

    if not output or output[0] != '3':
        raise AssertionError(f'app role product query mismatch: {output}')
    if len(output) < 2 or output[1] != '3':
        raise AssertionError(
            f'app role planner-native query mismatch: {output}'
        )
    if len(output) < 3 or output[2] != 'true|true':
        raise AssertionError(f'app role lifecycle status mismatch: {output}')


def main() -> None:
    args = parse_args()
    if args.runtime_liveness_timeout_ms <= 0:
        raise ValueError('--runtime-liveness-timeout-ms must be positive')
    if args.failover_backpressure_only and args.runtime_server_binary is None:
        raise ValueError(
            '--failover-backpressure-only requires '
            '--runtime-server-binary'
        )
    if (args.extension_libdir is None) != (
        args.extension_control_dir is None
    ):
        raise ValueError(
            '--extension-libdir and --extension-control-dir must be '
            'provided together'
        )
    if args.extension_libdir is not None:
        args.extension_libdir = args.extension_libdir.resolve()
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )
    pg_bin = Path(args.pg_bin)
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'
    psql_bin = pg_bin / 'psql'

    with tempfile.TemporaryDirectory(prefix='ii42_runtime_service_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        port = free_tcp_port()
        socket_dir.mkdir()
        model_path = args.model_path.expanduser().resolve()
        if not (model_path / 'manifest.json').is_file():
            raise FileNotFoundError(
                f'model manifest was not found: {model_path}'
            )
        escaped_model_path = str(model_path).replace("'", "''")

        run([str(initdb), '-D', str(data_dir), '-A', 'trust'])
        with (data_dir / 'postgresql.conf').open('a', encoding='utf-8') as f:
            f.write("\nshared_preload_libraries = 'ii42'\n")
            if args.extension_libdir is not None:
                libdir = str(args.extension_libdir).replace("'", "''")
                f.write(
                    "dynamic_library_path = '"
                    f'{libdir}:$libdir'
                    "'\n"
                )
            if args.extension_control_dir is not None:
                control_dir = str(
                    args.extension_control_dir
                ).replace("'", "''")
                f.write(
                    "extension_control_path = '"
                    f'{control_dir}:$system'
                    "'\n"
                )
            f.write("ii42.shared_runtime_size = '64MB'\n")
            if not args.use_packaged_model:
                f.write(f"ii42.sae_model_path = '{escaped_model_path}'\n")
            f.write("ii42.control_database = 'template1'\n")
            f.write(
                f'ii42.runtime_worker_count = {args.worker_count}\n'
            )
            f.write(
                'ii42.runtime_liveness_timeout = '
                f"'{args.runtime_liveness_timeout_ms}ms'\n"
            )
            f.write("listen_addresses = ''\n")
            f.write(f'max_worker_processes = {args.worker_count + 8}\n')
            f.write("ii42.maintenance_timer_interval_ms = '1000ms'\n")
            f.write(
                "ii42.maintenance_low_debt_interval_ms = '1000ms'\n"
            )

        started = False
        try:
            try:
                run([
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    '-l',
                    str(log_path),
                    '-o',
                    f'-k {socket_dir} -p {port}',
                    'start',
                    '-w',
                ])
            except subprocess.CalledProcessError:
                if log_path.is_file():
                    print(log_path.read_text(), file=sys.stderr)
                raise
            started = True
            try:
                run_database_lifecycle_probe(
                    socket_dir,
                    port,
                    args.worker_count,
                )
                psql(
                    psql_bin,
                    socket_dir,
                    port,
                    runtime_service_sql(
                        model_path,
                        args.worker_count,
                        'package'
                        if args.use_packaged_model
                        else 'environment',
                        args.runtime_liveness_timeout_ms,
                    ),
                )
                if args.failover_backpressure_only:
                    run_cpp_runtime_server_probe(
                        args.runtime_server_binary.resolve(),
                        socket_dir,
                        port,
                        model_path,
                        failover_backpressure_only=True,
                        runtime_liveness_timeout_ms=
                            args.runtime_liveness_timeout_ms,
                    )
                    print(
                        'Runtime failover backpressure temp PostgreSQL '
                        'smoke passed'
                    )
                    return
                if args.runtime_server_binary is not None:
                    run_cpp_runtime_server_probe(
                        args.runtime_server_binary.resolve(),
                        socket_dir,
                        port,
                        model_path,
                    )
                run_batch_request_validation_probe(
                    socket_dir,
                    port,
                    model_path,
                )
                run_semantic_product_contract_probe(socket_dir, port)
                run_low_memory_semantic_completion_probe(socket_dir, port)
                run_maintenance_lock_contention_probe(socket_dir, port)
                run_semantic_generation_reuse_probe(socket_dir, port)
                run_runtime_batch_cap_reload_probe(
                    socket_dir,
                    port,
                    model_path,
                )
                run_concurrent_runtime_queries(socket_dir, port)
                run_planner_native_cancellation_probe(socket_dir, port)
                run_parallel_worker_probe(
                    socket_dir,
                    port,
                    model_path,
                    args.worker_count,
                )
                run_model_affinity_probe(
                    socket_dir,
                    port,
                    model_path,
                    root,
                )
                run_same_path_checkout_reload_probe(
                    socket_dir,
                    port,
                    model_path,
                    root,
                )
                run_mixed_query_document_probe(
                    socket_dir,
                    port,
                    model_path,
                    args.worker_count,
                )
                if args.worker_count == 2:
                    run_completed_response_reservation_probe(
                        socket_dir,
                        port,
                        model_path,
                    )
                if args.worker_count >= 3:
                    run_single_build_pipeline_probe(
                        socket_dir,
                        port,
                        model_path,
                    )
                run_runtime_owner_cancellation_probe(
                    socket_dir,
                    port,
                    model_path,
                )
                run_app_role_product_query(psql_bin, socket_dir, port)
                run_control_database_independence_probe(
                    socket_dir,
                    port,
                    model_path,
                )
                run_planner_scope_filter_probe(socket_dir, port)
                if args.planner_native_scale_docs > 0:
                    run_planner_native_scale_probe(
                        socket_dir,
                        port,
                        args.planner_native_scale_docs,
                    )
                log_text = log_path.read_text(encoding='utf-8')
                if "you don't own a lock of type ExclusiveLock" in log_text:
                    raise AssertionError(
                        'mutation lifecycle released an unowned advisory lock'
                    )
            except Exception:
                if log_path.exists():
                    print(log_path.read_text(encoding='utf-8'), file=sys.stderr)
                raise
        finally:
            if started:
                subprocess.run(
                    [str(pg_ctl), '-D', str(data_dir), 'stop', '-m', 'fast'],
                    text=True,
                    cwd=REPO_ROOT,
                    check=False,
                    capture_output=True,
                )

    print('Runtime service temp PostgreSQL smoke passed')


if __name__ == '__main__':
    main()
