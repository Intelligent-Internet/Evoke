#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psycopg

from ii42_test_support import extension_control_root
from test_onnxruntime_smoke import REPO_ROOT
from test_runtime_service_temp_pg import runtime_service_sql, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the Evoke shared runtime-service restart smoke.',
    )
    parser.add_argument(
        '--pg-bin',
        default='/opt/homebrew/opt/postgresql@18/bin',
        help='Directory containing initdb, pg_ctl, and psql.',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=55435,
        help='Temporary PostgreSQL port.',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Production model checkout using the current runtime contract.',
    )
    parser.add_argument(
        '--restart-wait-seconds',
        type=float,
        default=15.0,
        help=(
            'Maximum time to wait for PostgreSQL to restart the worker '
            'after its five-second restart interval.'
        ),
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
    return parser.parse_args()


def fetch_status(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute('SELECT ii42_runtime_service_status()')
        row = cur.fetchone()
    if row is None:
        raise AssertionError('runtime status returned no row')
    return dict(row[0])


def setup_runtime_cluster(
    pg_bin: Path,
    data_dir: Path,
    socket_dir: Path,
    port: int,
    log_path: Path,
    model_path: Path,
    extension_libdir: Path | None,
    extension_control_dir: Path | None,
) -> None:
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'

    run([str(initdb), '-D', str(data_dir), '-A', 'trust'])
    escaped_model_path = str(model_path).replace("'", "''")
    with (data_dir / 'postgresql.conf').open('a', encoding='utf-8') as handle:
        handle.write("\nshared_preload_libraries = 'ii42'\n")
        if extension_libdir is not None:
            libdir = str(extension_libdir).replace("'", "''")
            handle.write(
                "dynamic_library_path = '"
                f'{libdir}:$libdir'
                "'\n"
            )
        if extension_control_dir is not None:
            control_dir = str(extension_control_dir).replace("'", "''")
            handle.write(
                "extension_control_path = '"
                f'{control_dir}:$system'
                "'\n"
            )
        handle.write("ii42.shared_runtime_size = '64MB'\n")
        handle.write(f"ii42.sae_model_path = '{escaped_model_path}'\n")
        handle.write("ii42.control_database = 'template1'\n")
        handle.write('ii42.runtime_worker_count = 2\n')
        handle.write("listen_addresses = ''\n")
        handle.write('max_worker_processes = 16\n')
        handle.write("ii42.maintenance_timer_interval_ms = '1000ms'\n")

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


def terminate_worker(conn: psycopg.Connection[Any], worker_pid: int) -> bool:
    with conn.cursor() as cur:
        cur.execute('SELECT pg_terminate_backend(%s)', (worker_pid,))
        row = cur.fetchone()
    if row is None:
        raise AssertionError('pg_terminate_backend returned no row')
    return bool(row[0])


def wait_for_restarted_worker(
    conn: psycopg.Connection[Any],
    worker_slot: int,
    old_pid: int,
    wait_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        time.sleep(1.0)
        last_status = fetch_status(conn)
        workers = list(last_status.get('workers', []))
        worker = next(
            (
                item
                for item in workers
                if int(item.get('slot', -1)) == worker_slot
            ),
            None,
        )
        new_pid = int(worker.get('pid', 0)) if worker is not None else 0
        if (
            new_pid > 0
            and new_pid != old_pid
            and worker is not None
            and worker.get('ready') is True
            and int(last_status.get('worker_count_ready', 0)) == 2
            and last_status.get('ready_for_text_encoding') is True
        ):
            return last_status
    raise AssertionError(
        'runtime worker did not restart before timeout: '
        f'old_pid={old_pid}, last_status={last_status}',
    )


def wait_for_degraded_pool(
    conn: psycopg.Connection[Any],
    old_pid: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + 4.0
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        time.sleep(0.1)
        last_status = fetch_status(conn)
        worker_pids = {
            int(item.get('pid', 0))
            for item in list(last_status.get('workers', []))
            if item.get('ready') is True
        }
        if (
            old_pid not in worker_pids
            and int(last_status.get('worker_count_ready', 0)) == 1
            and last_status.get('worker_ready') is True
        ):
            return last_status
    raise AssertionError(
        'terminated runtime worker did not leave a healthy degraded pool: '
        f'old_pid={old_pid}, last_status={last_status}',
    )


def product_query_rows(
    conn: psycopg.Connection[Any],
) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id
            FROM ii42_query(
                'docs_body_idx'::regclass,
                'alpha semantic optimization runtime restart',
                3
            ) WITH ORDINALITY AS hit(ctid, doc_id, score, ordinality)
            ORDER BY ordinality
            """,
        )
        rows = cur.fetchall()
    actual = [int(row[0]) for row in rows]
    return actual


def run_long_runtime_batch(dsn: str, model_path: Path) -> None:
    texts = [
        (
            f'worker failure isolation row {row} '
            + ('long semantic runtime document sentinel ' * 46)
        )
        for row in range(32)
    ]
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT ii42_runtime_service_document_atoms_batch(%s, %s)',
                (str(model_path), texts),
            )
            cur.fetchone()


def wait_for_processing_worker(
    conn: psycopg.Connection[Any],
    future: concurrent.futures.Future[None],
) -> dict[str, Any]:
    deadline = time.monotonic() + 5.0
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_status = fetch_status(conn)
        for worker in list(last_status.get('workers', [])):
            if (
                worker.get('processing') is True
                and worker.get('processing_document') is True
            ):
                return worker
        if future.done():
            break
        time.sleep(0.002)
    raise AssertionError(
        'long runtime batch did not expose an in-flight worker: '
        f'last_status={last_status}'
    )


def main() -> None:
    args = parse_args()
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
    pg_ctl = pg_bin / 'pg_ctl'

    with tempfile.TemporaryDirectory(prefix='ii42_runtime_restart_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        socket_dir.mkdir()
        model_path = args.model_path.expanduser().resolve()
        if not (model_path / 'manifest.json').is_file():
            raise FileNotFoundError(
                f'model manifest was not found: {model_path}'
            )
        manifest = json.loads(
            (model_path / 'manifest.json').read_text(encoding='utf-8')
        )
        if (
            manifest.get('schema_version') != 1
            or manifest.get('api_version') != 'evoke_model_v1'
            or manifest.get('runtime_abi')
            != 'evoke_p2_unified_text_atoms_v2'
        ):
            raise ValueError(
                '--model-path must use the current Evoke model contract'
            )

        started = False
        try:
            setup_runtime_cluster(
                pg_bin,
                data_dir,
                socket_dir,
                args.port,
                log_path,
                model_path,
                args.extension_libdir,
                args.extension_control_dir,
            )
            started = True
            dsn = (
                f'host={socket_dir} port={args.port} dbname=postgres'
            )
            with psycopg.connect(dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(runtime_service_sql(model_path, 2))

                before = fetch_status(conn)
                workers_before = list(before.get('workers', []))
                if (
                    int(before.get('worker_count_ready', 0)) != 2
                    or len(workers_before) != 2
                ):
                    raise AssertionError(
                        f'runtime worker pool is incomplete: {before}',
                    )
                query_rows_before = product_query_rows(conn)
                if len(query_rows_before) != 3:
                    raise AssertionError(
                        'product query returned an incomplete baseline: '
                        f'{query_rows_before}'
                    )
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                ) as executor:
                    interrupted = executor.submit(
                        run_long_runtime_batch,
                        dsn,
                        model_path,
                    )
                    old_worker = wait_for_processing_worker(conn, interrupted)
                    old_pid = int(old_worker.get('pid', 0))
                    old_slot = int(old_worker.get('slot', -1))
                    if old_pid <= 0 or old_slot < 0:
                        raise AssertionError(
                            f'runtime worker identity is invalid: {old_worker}',
                        )
                    if not terminate_worker(conn, old_pid):
                        raise AssertionError(
                            f'pg_terminate_backend({old_pid}) returned false',
                        )
                    degraded = wait_for_degraded_pool(conn, old_pid)
                    degraded_query_rows = product_query_rows(conn)
                    if degraded_query_rows != query_rows_before:
                        raise AssertionError(
                            'product query changed while one worker was down: '
                            f'before={query_rows_before}, '
                            f'degraded={degraded_query_rows}, status={degraded}'
                        )
                    try:
                        interrupted.result(timeout=10)
                    except psycopg.Error:
                        pass
                    else:
                        raise AssertionError(
                            'request owned by terminated worker did not fail'
                        )

                after_restart = wait_for_restarted_worker(
                    conn,
                    old_slot,
                    old_pid,
                    args.restart_wait_seconds,
                )
                if int(after_restart.get('worker_recoveries', 0)) <= int(
                    before.get('worker_recoveries', 0)
                ):
                    raise AssertionError(
                        'runtime worker restart did not increment recovery '
                        f'count: before={before}, after={after_restart}'
                    )
                query_rows = product_query_rows(conn)
                if query_rows != query_rows_before:
                    raise AssertionError(
                        'product query changed across runtime restart: '
                        f'before={query_rows_before}, after={query_rows}'
                    )
                after_query = fetch_status(conn)
                if after_query.get('backend_model_loading_allowed') is not False:
                    raise AssertionError(
                        'caller backend model loading became allowed after '
                        f'restart: {after_query}',
                    )
                if int(after_query.get('successes', 0)) <= int(
                    before.get('successes', 0)
                ):
                    raise AssertionError(
                        'post-restart product query did not reach runtime '
                        f'service: before={before}, after={after_query}',
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

    print(json.dumps({
        'old_worker_pid': old_pid,
        'new_worker_pid': int(next(
            item['pid']
            for item in after_restart['workers']
            if int(item['slot']) == old_slot
        )),
        'degraded_query_rows': degraded_query_rows,
        'query_rows': query_rows,
        'result': 'runtime service restart smoke passed',
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
