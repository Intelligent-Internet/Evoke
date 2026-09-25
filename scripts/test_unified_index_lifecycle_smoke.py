#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import psycopg

from evoke_test_support import (
    extension_control_root,
    vacuum_with_session_maintenance_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
QUERY = 'obsidian hummingbird quantum relay exact sentinel'
SCORE_PARITY_ABS_TOLERANCE = 1e-4
SCORE_PARITY_REL_TOLERANCE = 1e-6
T = TypeVar('T')


def make_owner_writable(path: Path) -> None:
    path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate the unified Evoke index through CREATE, CRUD, VACUUM, '
            'REINDEX, restart, and DROP.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share directory containing extension/evoke.control, '
            'or the extension directory itself.'
        ),
    )
    parser.add_argument('--port', type=int, default=55437)
    parser.add_argument('--soak-queries', type=int, default=0)
    parser.add_argument('--mixed-soak-cycles', type=int, default=0)
    parser.add_argument('--concurrent-crud-cycles', type=int, default=0)
    parser.add_argument('--concurrent-readers', type=int, default=4)
    parser.add_argument('--concurrent-writers', type=int, default=2)
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--allow-known-failures',
        action='store_true',
        help='Write the report without returning a failing exit code.',
    )
    return parser.parse_args()


def run(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result


def timed(metrics: dict[str, float], name: str, action: Callable[[], T]) -> T:
    started = time.perf_counter()
    try:
        return action()
    finally:
        metrics[name] = round((time.perf_counter() - started) * 1000.0, 3)


def load_manifest(model_path: Path) -> dict[str, Any]:
    manifest_path = model_path / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'model manifest does not exist: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (
        manifest.get('schema_version') != 1
        or manifest.get('api_version') != 'evoke_model_v1'
        or manifest.get('runtime_abi') != 'evoke_p2_unified_text_atoms_v2'
    ):
        raise ValueError(
            'model is not a current Evoke model contract'
        )
    return manifest


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def connect(socket_dir: Path, port: int) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname='postgres',
        user='postgres',
        host=str(socket_dir),
        port=port,
        autocommit=True,
    )


def acquire_maintenance_lock(
    socket_dir: Path,
    port: int,
    index_name: str,
) -> psycopg.Connection[Any]:
    connection = connect(socket_dir, port)
    connection.autocommit = False
    deadline = time.monotonic() + 5.0
    while True:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT evoke_index_try_maintenance_lock(%s::regclass)',
                (index_name,),
            )
            row = cursor.fetchone()
        if row is not None and row[0] is True:
            break
        if time.monotonic() >= deadline:
            connection.close()
            raise AssertionError('could not acquire maintenance test gate')
        time.sleep(0.01)
    # The C lock is session-scoped. End the acquisition transaction so this
    # test gate does not become an artificial VACUUM visibility horizon.
    connection.commit()
    return connection


def release_maintenance_lock(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> None:
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT evoke_index_maintenance_unlock(%s::regclass)',
                (index_name,),
            )
        connection.commit()
    finally:
        connection.close()


def configure_maintenance_worker_limit(
    connection: psycopg.Connection[Any],
    worker_limit: int,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            'ALTER SYSTEM SET evoke.maintenance_worker_limit = '
            f"'{int(worker_limit)}'",
        )
        cursor.execute('SELECT pg_reload_conf()')
    deadline = time.monotonic() + 10.0
    while True:
        with connection.cursor() as cursor:
            cursor.execute('SHOW evoke.maintenance_worker_limit')
            observed = int(cursor.fetchone()[0])
        if observed == worker_limit:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                'maintenance worker limit did not reload: '
                f'expected={worker_limit}, observed={observed}'
            )
        time.sleep(0.05)


def wait_for_maintenance_worker_quiescence(
    connection: psycopg.Connection[Any],
) -> None:
    deadline = time.monotonic() + 30.0
    while True:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*)
                FROM pg_stat_activity
                WHERE backend_type = 'evoke background'
                  AND datname = current_database()
                """
            )
            active_workers = int(cursor.fetchone()[0])
        if active_workers == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                'maintenance workers did not quiesce: '
                f'active={active_workers}'
            )
        time.sleep(0.05)


def fetch_json(
    connection: psycopg.Connection[Any],
    query: str,
    params: tuple[Any, ...] = (),
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError('expected one JSON result row')
    value = row[0]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f'expected JSON object, got {type(value).__name__}')
    return value


def p2_runtime_result(
    connection: psycopg.Connection[Any],
    model_path: Path,
    text: str,
) -> dict[str, Any]:
    return fetch_json(
        connection,
        'SELECT evoke_runtime_service_query_atoms(%s, %s)::jsonb',
        (str(model_path), text),
    )


def p2_runtime_batch_result(
    connection: psycopg.Connection[Any],
    model_path: Path,
    texts: list[str],
) -> dict[str, Any]:
    return fetch_json(
        connection,
        'SELECT evoke_runtime_service_query_atoms_batch(%s, %s)::jsonb',
        (str(model_path), texts),
    )


def p2_runtime_comparable(result: dict[str, Any]) -> dict[str, Any]:
    tokenizer = result.get('tokenizer')
    if not isinstance(tokenizer, dict):
        tokenizer = result
    return {
        'atoms': result.get('atoms'),
        'weights': result.get('weights'),
        'compiler': result.get('compiler'),
        'token_count': tokenizer.get('token_count'),
        'window_count': tokenizer.get('window_count'),
        'window_stride': tokenizer.get('window_stride'),
        'aggregation': tokenizer.get('aggregation'),
        'truncated': tokenizer.get('truncated'),
    }


def p2_runtime_max_weight_delta(
    left: dict[str, Any],
    right: dict[str, Any],
) -> float | None:
    left_weights = left.get('weights')
    right_weights = right.get('weights')
    if not isinstance(left_weights, list) or not isinstance(right_weights, list):
        return None
    if len(left_weights) != len(right_weights):
        return None
    return max(
        (
            abs(float(left_value) - float(right_value))
            for left_value, right_value in zip(
                left_weights,
                right_weights,
                strict=True,
            )
        ),
        default=0.0,
    )


def p2_runtime_views_match(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    exact_keys = (
        'atoms',
        'token_count',
        'window_count',
        'window_stride',
        'aggregation',
        'truncated',
    )
    if any(left.get(key) != right.get(key) for key in exact_keys):
        return False
    left_weights = left.get('weights')
    right_weights = right.get('weights')
    if not isinstance(left_weights, list) or not isinstance(right_weights, list):
        return False
    if len(left_weights) != len(right_weights):
        return False
    if not all(
        math.isclose(
            float(left_value),
            float(right_value),
            rel_tol=SCORE_PARITY_REL_TOLERANCE,
            abs_tol=SCORE_PARITY_ABS_TOLERANCE,
        )
        for left_value, right_value in zip(
            left_weights,
            right_weights,
            strict=True,
        )
    ):
        return False
    left_compiler = left.get('compiler')
    right_compiler = right.get('compiler')
    if not isinstance(left_compiler, dict) or not isinstance(
        right_compiler,
        dict,
    ):
        return False
    for key in ('lexical_atoms', 'semantic_atoms'):
        if left_compiler.get(key) != right_compiler.get(key):
            return False
    for key in ('lexical_proxy', 'semantic_proxy', 'query_scale'):
        if not math.isclose(
            float(left_compiler[key]),
            float(right_compiler[key]),
            rel_tol=SCORE_PARITY_REL_TOLERANCE,
            abs_tol=SCORE_PARITY_ABS_TOLERANCE,
        ):
            return False
    return True


def p2_semantic_atoms(
    result: dict[str, Any],
    lexical_dims: int,
) -> dict[int, float]:
    atoms = result.get('atoms')
    weights = result.get('weights')
    if not isinstance(atoms, list) or not isinstance(weights, list):
        raise TypeError('P2 runtime result is missing atoms or weights')
    return {
        int(atom): float(weight)
        for atom, weight in zip(atoms, weights, strict=True)
        if int(atom) >= lexical_dims
    }


def p2_full_text_window_audit(
    connection: psycopg.Connection[Any],
    model_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    atom_space_path = (
        model_path / manifest['artifacts']['atom_space']['path']
    )
    runtime_path = (
        model_path / manifest['artifacts']['semantic_runtime']['path']
    )
    atom_space = json.loads(atom_space_path.read_text(encoding='utf-8'))
    semantic_runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    lexical_dims = int(atom_space['lexical']['end_exclusive'])
    max_length = int(semantic_runtime['max_length'])
    short_texts = [
        'short full-text window smoke',
        'single and batch runtime parity',
    ]
    short_results = [
        p2_runtime_result(connection, model_path, text)
        for text in short_texts
    ]
    short_batch = p2_runtime_batch_result(
        connection,
        model_path,
        short_texts,
    )
    short_batch_results = short_batch.get('results')
    if not isinstance(short_batch_results, list):
        raise TypeError('P2 batch runtime result is missing results')

    shared_prefix = 'neutral shared background evidence ' * 220
    long_texts = [
        shared_prefix + 'oxidative mitochondria cellular stress ' * 90,
        shared_prefix + 'galactic astronomy telescope nebula ' * 90,
    ]
    long_results = [
        p2_runtime_result(connection, model_path, text)
        for text in long_texts
    ]
    long_batch = p2_runtime_batch_result(
        connection,
        model_path,
        long_texts,
    )
    long_batch_results = long_batch.get('results')
    if not isinstance(long_batch_results, list):
        raise TypeError('P2 long batch runtime result is missing results')

    limit_error: dict[str, str] = {}
    try:
        p2_runtime_result(connection, model_path, 'x ' * 600_000)
    except psycopg.errors.ProgramLimitExceeded as error:
        limit_error = {
            'sqlstate': str(error.sqlstate),
            'message': str(error).splitlines()[0],
        }

    short_views = [p2_runtime_comparable(row) for row in short_results]
    short_batch_views = [
        p2_runtime_comparable(row)
        for row in short_batch_results
    ]
    long_views = [p2_runtime_comparable(row) for row in long_results]
    long_batch_views = [
        p2_runtime_comparable(row)
        for row in long_batch_results
    ]
    gates = {
        'short_text_single_batch_runtime_parity': (
            all(
                p2_runtime_views_match(single, batched)
                for single, batched in zip(
                    short_views,
                    short_batch_views,
                    strict=True,
                )
            )
            and all(view['window_count'] == 1 for view in short_views)
            and all(view['truncated'] is False for view in short_views)
        ),
        'long_text_uses_complete_windowed_input': all(
            int(view['token_count']) > max_length
            and int(view['window_count']) > 1
            and view['aggregation'] == 'dimension_max_top_k'
            and view['truncated'] is False
            for view in long_views
        ),
        'long_tail_changes_semantic_atoms': (
            p2_semantic_atoms(long_results[0], lexical_dims)
            != p2_semantic_atoms(long_results[1], lexical_dims)
        ),
        'long_text_single_batch_runtime_parity': (
            all(
                p2_runtime_views_match(single, batched)
                for single, batched in zip(
                    long_views,
                    long_batch_views,
                    strict=True,
                )
            )
        ),
        'oversized_input_fails_explicitly': (
            limit_error.get('sqlstate') == '54000'
            and 'transport limit' in limit_error.get('message', '')
        ),
    }
    return {
        'gates': gates,
        'evidence': {
            'model_id': manifest['model_id'],
            'max_length': max_length,
            'short_rows': [
                {
                    'token_count': view['token_count'],
                    'window_count': view['window_count'],
                    'single_batch_max_weight_delta': (
                        p2_runtime_max_weight_delta(single, batched)
                    ),
                }
                for view, single, batched in zip(
                    short_views,
                    short_results,
                    short_batch_results,
                    strict=True,
                )
            ],
            'long_rows': [
                {
                    'token_count': view['token_count'],
                    'window_count': view['window_count'],
                    'semantic_atoms': len(
                        p2_semantic_atoms(row, lexical_dims)
                    ),
                    'single_batch_max_weight_delta': (
                        p2_runtime_max_weight_delta(row, batched)
                    ),
                }
                for row, view, batched in zip(
                    long_results,
                    long_views,
                    long_batch_results,
                    strict=True,
                )
            ],
            'single_batch_score_tolerance': {
                'absolute': SCORE_PARITY_ABS_TOLERANCE,
                'relative': SCORE_PARITY_REL_TOLERANCE,
            },
            'batch_window_count': long_batch.get('window_count'),
            'batch_sequence_length': long_batch.get('sequence_length'),
            'limit_error': limit_error,
        },
    }


def index_status(connection: psycopg.Connection[Any]) -> dict[str, Any]:
    return fetch_json(
        connection,
        "SELECT evoke_index_status('p2_mutable.docs_idx'::regclass)",
    )


def index_audit(connection: psycopg.Connection[Any]) -> dict[str, Any]:
    return fetch_json(
        connection,
        "SELECT evoke_index_audit('p2_mutable.docs_idx'::regclass)",
    )


def cache_state(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    return fetch_json(
        connection,
        'SELECT evoke_index_runtime_state_json(%s::regclass)',
        (index_name,),
    )


def page_native_backend_memory_state(
    connection: psycopg.Connection[Any],
) -> dict[str, int]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                count(*) FILTER (
                    WHERE name = 'evoke transaction-local SAE delta cache'
                )::int8,
                COALESCE(
                    sum(total_bytes) FILTER (
                        WHERE name =
                            'evoke transaction-local SAE delta cache'
                    ),
                    0
                )::int8,
                COALESCE(sum(total_bytes), 0)::int8
            FROM pg_backend_memory_contexts
            """
        )
        row = cursor.fetchone()
    if row is None or any(value is None for value in row):
        raise AssertionError('backend memory diagnostics are incomplete')
    return {
        'legacy_cache_contexts': int(row[0]),
        'legacy_cache_bytes': int(row[1]),
        'backend_total_bytes': int(row[2]),
    }


def backend_rss_kib(connection: psycopg.Connection[Any]) -> int:
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_backend_pid()')
        backend_pid = int(cursor.fetchone()[0])
    result = subprocess.run(
        ['ps', '-o', 'rss=', '-p', str(backend_pid)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f'could not read RSS for PostgreSQL backend {backend_pid}: '
            f'{result.stderr.strip()}'
        )
    return int(result.stdout.strip())


def run_page_native_backend_state_audit(
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    connection = connect(socket_dir, port)
    connection.autocommit = False
    first_rows: list[tuple[str, float]] = []
    second_rows: list[tuple[str, float]] = []
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ'
            )
            cursor.execute(
                'INSERT INTO p2_mutable.docs (id, body) VALUES (%s, %s)',
                ('local-cache-a', QUERY),
            )
        first_rows = query_hits(connection)
        for _ in range(7):
            if not query_rows_match(first_rows, query_hits(connection)):
                raise AssertionError(
                    'page-native L0 changed during baseline warmup'
                )
        first_status = index_status(connection)
        first_memory = page_native_backend_memory_state(connection)
        first_rss_kib = backend_rss_kib(connection)
        for _ in range(99):
            if not query_rows_match(first_rows, query_hits(connection)):
                raise AssertionError(
                    'page-native L0 changed unchanged query rows'
                )
        repeated_status = index_status(connection)
        repeated_memory = page_native_backend_memory_state(connection)
        repeated_rss_kib = backend_rss_kib(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO p2_mutable.docs (id, body) VALUES (%s, %s)',
                ('local-cache-b', QUERY),
            )
        second_rows = query_hits(connection)
        for _ in range(99):
            if not query_rows_match(second_rows, query_hits(connection)):
                raise AssertionError(
                    'page-native L0 changed post-DML query rows'
                )
        second_status = index_status(connection)
        second_memory = page_native_backend_memory_state(connection)
        connection.rollback()
        after_rollback_memory = page_native_backend_memory_state(connection)
        after_rollback_pending = backend_memory_state(connection)
    finally:
        connection.close()

    repeated_memory_growth = max(
        0,
        repeated_memory['backend_total_bytes']
            - first_memory['backend_total_bytes'],
    )
    repeated_rss_growth_kib = max(0, repeated_rss_kib - first_rss_kib)
    return {
        'first_status': first_status,
        'repeated_status': repeated_status,
        'second_status': second_status,
        'first_memory': first_memory,
        'repeated_memory': repeated_memory,
        'second_memory': second_memory,
        'after_rollback_memory': after_rollback_memory,
        'after_rollback_pending': after_rollback_pending,
        'first_rss_kib': first_rss_kib,
        'repeated_rss_kib': repeated_rss_kib,
        'repeated_memory_growth_bytes': repeated_memory_growth,
        'repeated_rss_growth_kib': repeated_rss_growth_kib,
        'first_rows': query_digest(first_rows),
        'second_rows': query_digest(second_rows),
        'gates': {
            'unchanged_transaction_reads_page_native_l0': (
                bool(first_rows)
                and first_rows[0][0] == 'local-cache-a'
                and int(first_status['details']['delta_records']) > 0
                and first_memory['legacy_cache_contexts'] == 0
                and first_memory['legacy_cache_bytes'] == 0
            ),
            'repeated_page_native_queries_keep_root_and_memory_bounded': (
                generation_identity(first_status)
                    == generation_identity(repeated_status)
                and repeated_memory['legacy_cache_contexts'] == 0
                and repeated_memory['legacy_cache_bytes'] == 0
                and repeated_memory_growth <= 8 * 1024 * 1024
                and repeated_rss_growth_kib <= 16 * 1024
            ),
            'intervening_dml_updates_page_native_l0': (
                {row[0] for row in second_rows}
                    >= {'local-cache-a', 'local-cache-b'}
                and int(second_status['details']['delta_records'])
                    >= int(first_status['details']['delta_records'])
                and second_memory['legacy_cache_contexts'] == 0
                and second_memory['legacy_cache_bytes'] == 0
            ),
            'transaction_end_retains_no_index_sized_backend_state': (
                after_rollback_memory['legacy_cache_contexts'] == 0
                and after_rollback_memory['legacy_cache_bytes'] == 0
                and after_rollback_pending['pending_contexts'] == 0
                and after_rollback_pending['pending_context_bytes'] == 0
            ),
        },
    }


def generation_identity(status: dict[str, Any]) -> str:
    return str(status['generation']['generation_id'])


def posting_records(status: dict[str, Any]) -> int:
    return int(status['generation']['posting']['record_count'])


def generation_storage_ready(
    generation: dict[str, Any],
    primary: dict[str, Any],
    posting: dict[str, Any],
    *,
    unified_posting: bool,
) -> bool:
    if int(primary.get('physical_blocks') or 0) <= 0:
        return False
    if generation.get('diagnostics_complete') is True:
        storage_ready = (
            int(primary.get('payload_bytes') or 0) > 0
            and int(primary.get('reachable_blocks') or 0) > 0
        )
        if not unified_posting:
            return storage_ready
        return (
            storage_ready
            and int(posting.get('bytes') or -1)
                == int(primary.get('payload_bytes') or -2)
        )
    return (
        generation.get('diagnostics_scope') == 'readiness'
        and primary.get('payload_bytes') is None
        and primary.get('reachable_blocks') is None
        and posting.get('bytes') is None
    )


def unified_generation_ready(
    status: dict[str, Any],
    *,
    expected_records: int,
) -> bool:
    generation = status.get('generation')
    if not isinstance(generation, dict):
        return False
    primary = generation.get('primary')
    posting = generation.get('posting')
    if not isinstance(primary, dict) or not isinstance(posting, dict):
        return False
    contract_signature = generation.get('contract_signature')
    return (
        status.get('query_ready') is True
        and status.get('runtime_signature_matches') is True
        and isinstance(contract_signature, str)
        and len(contract_signature) == 32
        and status.get('runtime_signature') == contract_signature
        and generation.get('atomic') is True
        and generation.get('valid') is True
        and bool(generation.get('generation_id'))
        and int(generation.get('generation', 0)) > 0
        and primary.get('role') == 'unified_posting'
        and primary.get('identity') == 'segment_versions'
        and primary.get('storage') == 'convergent_segments'
        and generation_storage_ready(
            generation,
            primary,
            posting,
            unified_posting=True,
        )
        and posting.get('role') == 'unified_posting'
        and posting.get('present') is True
        and posting.get('valid') is True
        and posting.get('active') is True
        and posting.get('signature') == contract_signature
        and int(posting.get('record_count', -1)) == expected_records
    )


def bm25_generation_ready(
    status: dict[str, Any],
    *,
    expected_docs: int,
) -> bool:
    generation = status.get('generation')
    if not isinstance(generation, dict):
        return False
    primary = generation.get('primary')
    posting = generation.get('posting')
    if not isinstance(primary, dict) or not isinstance(posting, dict):
        return False
    contract_signature = generation.get('contract_signature')
    return (
        status.get('query_ready') is True
        and status.get('sae_enabled') is False
        and status.get('runtime_signature_matches') is True
        and isinstance(contract_signature, str)
        and len(contract_signature) == 32
        and status.get('runtime_signature') == contract_signature
        and generation.get('atomic') is True
        and generation.get('valid') is True
        and bool(generation.get('generation_id'))
        and int(generation.get('docs', -1)) == expected_docs
        and primary.get('role') == 'bm25'
        and primary.get('identity') == 'segment_versions'
        and primary.get('storage') == 'convergent_segments'
        and generation_storage_ready(
            generation,
            primary,
            posting,
            unified_posting=False,
        )
        and posting.get('role') == 'none'
        and posting.get('present') is False
        and posting.get('active') is False
        and posting.get('signature') == ''
        and int(posting.get('record_count', -1)) == 0
    )


def query_hits(
    connection: psycopg.Connection[Any],
    query_text: str = QUERY,
    *,
    oracle: bool = False,
) -> list[tuple[str, float]]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_config("
            "'evoke.test_unified_overlay_oracle', %s, false)",
            ('on' if oracle else 'off',),
        )
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'p2_mutable.docs_idx'::regclass,
                %s,
                20
            ) AS hit
            JOIN p2_mutable.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """,
            (query_text,),
        )
        return [(str(row[0]), float(row[1])) for row in cursor.fetchall()]


def query_trace(
    connection: psycopg.Connection[Any],
    query_text: str = QUERY,
    *,
    oracle: bool = False,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_config("
            "'evoke.test_unified_overlay_oracle', %s, false)",
            ('on' if oracle else 'off',),
        )
        cursor.execute(
            """
            SELECT to_jsonb(hit)
            FROM evoke_query_semantic_internal(
                'p2_mutable.docs_idx'::regclass,
                %s,
                20
            ) AS hit
            ORDER BY hit.rank
            LIMIT 1
            """,
            (query_text,),
        )
        row = cursor.fetchone()
        return {} if row is None else dict(row[0])


def stale_query_encoding_audit(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    encoded = fetch_json(
        connection,
        """
        SELECT evoke_encode_text_internal(
            'p2_mutable.docs_idx'::regclass,
            %s
        )
        """,
        (QUERY,),
    )
    signature = str(encoded.get('runtime_signature') or '')
    if len(signature) != 32:
        raise AssertionError(
            f'query encoding lacks a generation signature: {encoded}'
        )
    stale_signature = (
        ('1' if signature[0] == '0' else '0') + signature[1:]
    )
    error: dict[str, str] = {}
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*)
                FROM evoke_index_semantic_query_native_internal(
                    'p2_mutable.docs_idx'::regclass,
                    %s::int4[],
                    %s::real[],
                    NULL::text[],
                    NULL::real[],
                    20,
                    %s
                )
                """,
                (
                    [int(value) for value in encoded['atoms']],
                    [float(value) for value in encoded['weights']],
                    stale_signature,
                ),
            )
    except psycopg.Error as caught:
        error = {
            'sqlstate': str(caught.sqlstate),
            'message': str(caught).splitlines()[0],
        }
    return {
        'encoded_signature': signature,
        'stale_signature': stale_signature,
        'error': error,
        'passed': (
            error.get('sqlstate') == '55000'
            and 'evoke semantic query generation changed'
                in error.get('message', '')
        ),
    }


def query_digest(hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return [(doc_id, round(score, 5)) for doc_id, score in hits]


def query_rows_match(
    left: list[tuple[str, float]],
    right: list[tuple[str, float]],
) -> bool:
    return len(left) == len(right) and all(
        left_id == right_id
        and math.isclose(
            left_score,
            right_score,
            rel_tol=SCORE_PARITY_REL_TOLERANCE,
            abs_tol=SCORE_PARITY_ABS_TOLERANCE,
        )
        for (left_id, left_score), (right_id, right_score) in zip(left, right)
    )


def query_matrix_matches(
    left: dict[str, list[tuple[str, float]]],
    right: dict[str, list[tuple[str, float]]],
) -> bool:
    return left.keys() == right.keys() and all(
        query_rows_match(left[query_text], right[query_text])
        for query_text in left
    )


def query_matrix_max_score_delta(
    left: dict[str, list[tuple[str, float]]],
    right: dict[str, list[tuple[str, float]]],
) -> float | None:
    deltas = [
        abs(left_score - right_score)
        for query_text, left_rows in left.items()
        for (left_id, left_score), (right_id, right_score) in zip(
            left_rows,
            right.get(query_text, []),
        )
        if left_id == right_id
    ]
    return max(deltas) if deltas else None


def semantic_mvcc_query_rows(
    connection: psycopg.Connection[Any],
) -> list[tuple[str, str, float]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id, hit.ctid::text, hit.score::float8
            FROM evoke_query(
                'p2_semantic_mvcc.docs_idx'::regclass,
                'database semantic retrieval',
                20
            ) AS hit
            JOIN p2_semantic_mvcc.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        return [
            (str(row[0]), str(row[1]), float(row[2]))
            for row in cursor.fetchall()
        ]


def semantic_mvcc_target_tid(
    rows: list[tuple[str, str, float]],
) -> str | None:
    for doc_id, heap_tid, _score in rows:
        if doc_id == 'mvcc-target':
            return heap_tid
    return None


def run_online_maintenance_rounds(
    connection: psycopg.Connection[Any],
    index_name: str,
    *,
    max_rounds: int = 4,
) -> tuple[list[str], dict[str, Any]]:
    results: list[str] = []
    status = fetch_json(
        connection,
        'SELECT evoke_index_status(%s::regclass)',
        (index_name,),
    )
    for _ in range(max_rounds):
        results.append(
            run_tiny_fixture_maintenance(
                connection,
                index_name,
                try_only=True,
            )
        )
        status = fetch_json(
            connection,
            'SELECT evoke_index_status(%s::regclass)',
            (index_name,),
        )
        time.sleep(0.01)
    return results, status


def semantic_completion_state(status: dict[str, Any]) -> dict[str, Any]:
    value = status['generation']['delta']['semantic_completion']
    if not isinstance(value, dict):
        raise TypeError('semantic completion status is not an object')
    return value


def semantic_accelerator_converged(status: dict[str, Any]) -> bool:
    value = status['generation'].get('semantic_accelerator')
    if not isinstance(value, dict) or not bool(value.get('eligible')):
        return True
    return (
        value.get('state') == 'ready'
        and value.get('baseline_current') is True
        and value.get('scope_current') is True
    )


def run_tiny_fixture_maintenance(
    connection: psycopg.Connection[Any],
    index_name: str,
    *,
    try_only: bool = False,
) -> str:
    setting = 'SET' if connection.autocommit else 'SET LOCAL'
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f'{setting} '
                "evoke.test_convergent_l0_rotation_records = '1'"
            )
            function_name = (
                'evoke_index_try_maintain'
                if try_only
                else 'evoke_index_maintain'
            )
            cursor.execute(
                f'SELECT {function_name}(%s::regclass)',
                (index_name,),
            )
            row = cursor.fetchone()
        if not connection.autocommit:
            connection.commit()
    finally:
        if connection.autocommit:
            with connection.cursor() as cursor:
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )
    if row is None:
        raise AssertionError('tiny-fixture maintenance returned no row')
    return str(row[0])


def run_eventual_maintenance_until_converged(
    connection: psycopg.Connection[Any],
    index_name: str,
    initial_status: dict[str, Any],
    *,
    max_rounds: int = 16,
    require_generation_change: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    status = fetch_json(
        connection,
        'SELECT evoke_index_status(%s::regclass)',
        (index_name,),
    )
    if not connection.autocommit:
        connection.commit()

    completed_rounds = 0
    attempts = 0
    max_attempts = max_rounds * 20
    while completed_rounds < max_rounds and attempts < max_attempts:
        attempts += 1
        completion = semantic_completion_state(status)
        if (
            bool(completion['converged'])
            and int(status['details']['delta_records']) == 0
            and semantic_accelerator_converged(status)
            and (
                not require_generation_change
                or generation_identity(status)
                    != generation_identity(initial_status)
            )
            and phases
            and 'reason=no_pending' in phases[-1]['result']
        ):
            return phases, status

        # Tiny fixtures cannot cross the production soft frontier naturally.
        # The helper restores the production setting after each transaction.
        result = run_tiny_fixture_maintenance(connection, index_name)
        status = fetch_json(
            connection,
            'SELECT evoke_index_status(%s::regclass)',
            (index_name,),
        )
        if not connection.autocommit:
            connection.commit()
        phases.append({'result': result, 'status': status})
        if any(
            reason in result
            for reason in (
                'xid_horizon',
                'lock_busy',
                'reader_fence_busy',
                'root_changed',
                'accelerator_build_busy',
                'accelerator_root_checkpoint',
            )
        ):
            # These are transient PostgreSQL ownership barriers, not useful
            # maintenance work. Avoid exhausting the bounded phase budget
            # while an older snapshot or concurrent worker releases it.
            time.sleep(0.05)
            continue
        completed_rounds += 1

    raise AssertionError(
        f'{index_name} did not complete core and derived convergence after '
        f'{completed_rounds} phases and {attempts} attempts: {phases}'
    )


def run_eventual_maintenance_until_reason(
    connection: psycopg.Connection[Any],
    index_name: str,
    reason: str,
    *,
    max_attempts: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    status = fetch_json(
        connection,
        'SELECT evoke_index_status(%s::regclass)',
        (index_name,),
    )
    if not connection.autocommit:
        connection.commit()

    for _ in range(max_attempts):
        result = run_tiny_fixture_maintenance(connection, index_name)
        status = fetch_json(
            connection,
            'SELECT evoke_index_status(%s::regclass)',
            (index_name,),
        )
        if not connection.autocommit:
            connection.commit()
        phases.append({'result': result, 'status': status})
        if f'reason={reason}' in result:
            return phases, status
        if any(
            transient in result
            for transient in (
                'xid_horizon',
                'lock_busy',
                'reader_fence_busy',
                'root_changed',
                'accelerator_build_busy',
                'accelerator_root_checkpoint',
            )
        ):
            time.sleep(0.05)

    raise AssertionError(
        f'{index_name} did not reach reason={reason} after '
        f'{max_attempts} attempts: {phases}'
    )


def run_semantic_long_snapshot_compaction_audit(
    connection: psycopg.Connection[Any],
    socket_dir: Path,
    port: int,
    model_path: Path,
) -> dict[str, Any]:
    index_name = 'p2_semantic_mvcc.docs_idx'
    old_connection: psycopg.Connection[Any] | None = None
    guard_connection: psycopg.Connection[Any] | None = None
    evidence: dict[str, Any] = {}

    with connection.cursor() as cursor:
        cursor.execute('DROP SCHEMA IF EXISTS p2_semantic_mvcc CASCADE')
        cursor.execute('CREATE SCHEMA p2_semantic_mvcc')
        cursor.execute(
            """
            CREATE TABLE p2_semantic_mvcc.docs (
                id text PRIMARY KEY,
                body text NOT NULL
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO p2_semantic_mvcc.docs VALUES
                (
                    'mvcc-target',
                    'database semantic retrieval legacy version'
                ),
                (
                    'mvcc-companion',
                    'database semantic retrieval stable companion'
                )
            """
        )
        cursor.execute(
            'CREATE INDEX docs_idx '
            'ON p2_semantic_mvcc.docs '
            'USING evoke (body) '
            'WITH ('
            'sae = true, '
            f'model_path = {sql_literal(str(model_path))}, '
            'consistency = eventual'
            ')'
        )
        cursor.execute(
            """
            SELECT ctid::text
            FROM p2_semantic_mvcc.docs
            WHERE id = 'mvcc-target'
            """
        )
        old_tid = str(cursor.fetchone()[0])

    try:
        initial_status = fetch_json(
            connection,
            'SELECT evoke_index_status(%s::regclass)',
            (index_name,),
        )
        guard_connection = acquire_maintenance_lock(
            socket_dir,
            port,
            index_name,
        )
        old_connection = connect(socket_dir, port)
        old_connection.autocommit = False
        with old_connection.cursor() as cursor:
            cursor.execute(
                'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ'
            )
        old_rows_before = semantic_mvcc_query_rows(old_connection)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE p2_semantic_mvcc.docs
                SET body = 'database semantic retrieval fresh version'
                WHERE id = 'mvcc-target'
                RETURNING ctid::text
                """
            )
            new_tid = str(cursor.fetchone()[0])
        pending_status = fetch_json(
            connection,
            'SELECT evoke_index_status(%s::regclass)',
            (index_name,),
        )
        fresh_rows_before = semantic_mvcc_query_rows(connection)
        old_rows_with_delta = semantic_mvcc_query_rows(old_connection)

        release_maintenance_lock(guard_connection, index_name)
        guard_connection = None
        horizon_maintain, horizon_status = run_online_maintenance_rounds(
            connection,
            index_name,
        )
        old_rows_after = semantic_mvcc_query_rows(old_connection)
        fresh_rows_after = semantic_mvcc_query_rows(connection)

        old_connection.commit()
        old_connection.close()
        old_connection = None
        guard_connection = acquire_maintenance_lock(
            socket_dir,
            port,
            index_name,
        )
        vacuum_with_session_maintenance_lock(
            guard_connection,
            'p2_semantic_mvcc.docs',
            index_cleanup=True,
        )
        vacuum_status = fetch_json(
            connection,
            'SELECT evoke_index_status(%s::regclass)',
            (index_name,),
        )
        final_maintain, final_status = run_eventual_maintenance_until_converged(
            guard_connection,
            index_name,
            initial_status,
            max_rounds=64,
        )
        release_maintenance_lock(guard_connection, index_name)
        guard_connection = None
        final_rows = semantic_mvcc_query_rows(connection)

        evidence = {
            'old_tid': old_tid,
            'new_tid': new_tid,
            'initial_status': initial_status,
            'pending_status': pending_status,
            'horizon_maintain': horizon_maintain,
            'horizon_status': horizon_status,
            'vacuum_status': vacuum_status,
            'final_maintain': final_maintain,
            'final_status': final_status,
            'old_rows_before': old_rows_before,
            'fresh_rows_before': fresh_rows_before,
            'old_rows_with_delta': old_rows_with_delta,
            'old_rows_after': old_rows_after,
            'fresh_rows_after': fresh_rows_after,
            'final_rows': final_rows,
        }
        evidence['passed'] = (
            old_tid != new_tid
            and semantic_mvcc_target_tid(old_rows_before) == old_tid
            and semantic_mvcc_target_tid(fresh_rows_before) == new_tid
            and semantic_mvcc_target_tid(old_rows_with_delta) == old_tid
            and semantic_mvcc_target_tid(old_rows_after) == old_tid
            and semantic_mvcc_target_tid(fresh_rows_after) == new_tid
            # Accelerator publication and active-L0 rotation may advance the
            # checked root without changing sealed query authority. The long
            # snapshot must keep that authority and the pending frontier
            # intact until its XID horizon is released.
            and posting_records(horizon_status)
                == posting_records(pending_status)
            and int(horizon_status['generation']['sealed_docs'])
                == int(pending_status['generation']['sealed_docs'])
            and int(
                horizon_status['generation']['delta']['pending']['records']
            ) > 0
            and any('xid_horizon' in result for result in horizon_maintain)
            and int(horizon_status['details']['delta_records']) > 0
            and generation_identity(final_status)
                != generation_identity(initial_status)
            and unified_generation_ready(final_status, expected_records=2)
            and int(final_status['details']['delta_records']) == 0
            and int(final_status['details']['pending_writes']) == 0
            and int(final_status['details']['pending_deletes']) == 0
            and semantic_completion_state(final_status)['converged'] is True
            and semantic_mvcc_target_tid(final_rows) == new_tid
        )
        return evidence
    finally:
        if old_connection is not None:
            old_connection.rollback()
            old_connection.close()
        if guard_connection is not None:
            release_maintenance_lock(guard_connection, index_name)
        with connection.cursor() as cursor:
            cursor.execute('DROP SCHEMA IF EXISTS p2_semantic_mvcc CASCADE')


def eventual_status(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    return fetch_json(
        connection,
        "SELECT evoke_index_status('p2_eventual.docs_idx'::regclass)",
    )


def eventual_query_hits(
    connection: psycopg.Connection[Any],
) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id
            FROM evoke_query(
                'p2_eventual.docs_idx'::regclass,
                %s,
                20
            ) AS hit
            JOIN p2_eventual.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """,
            (QUERY,),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def eventual_deferred_status(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    return fetch_json(
        connection,
        "SELECT evoke_index_status('p2_eventual.deferred_idx'::regclass)",
    )


def eventual_deferred_query_rows(
    connection: psycopg.Connection[Any],
    *,
    exact: bool = False,
) -> list[tuple[str, float]]:
    with connection.cursor() as cursor:
        if exact:
            cursor.execute(
                "SELECT set_config("
                "'evoke.test_disable_semantic_accelerator', 'on', false)"
            )
        try:
            cursor.execute(
                """
                SELECT source.id, hit.score::float8
                FROM evoke_query(
                    'p2_eventual.deferred_idx'::regclass,
                    %s,
                    20
                ) AS hit
                JOIN p2_eventual.deferred_docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """,
                (QUERY,),
            )
            return [
                (str(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        finally:
            if exact:
                cursor.execute(
                    "SELECT set_config("
                    "'evoke.test_disable_semantic_accelerator', 'off', false)"
                )


def eventual_deferred_query_hits(
    connection: psycopg.Connection[Any],
) -> list[str]:
    return [
        doc_id
        for doc_id, _score in eventual_deferred_query_rows(connection)
    ]


def eventual_deferred_filtered_background_hit(
    connection: psycopg.Connection[Any],
) -> str | None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id
            FROM p2_eventual.deferred_docs AS source
            WHERE source.id = 'deferred-background'
            ORDER BY evoke_query(
                'p2_eventual.deferred_idx'::regclass,
                'unrelated archival background record'
            ) DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        return None if row is None else str(row[0])


def run_semantic_root_snapshot_race_audit(
    connection: psycopg.Connection[Any],
    socket_dir: Path,
    port: int,
    expected_rows: list[tuple[str, float]],
    guard_connection: psycopg.Connection[Any],
    expected_pending_records: int,
) -> dict[str, Any]:
    index_name = 'p2_eventual.deferred_idx'
    preparation_phases: list[dict[str, Any]] = []
    for _ in range(4):
        prepared_status = eventual_deferred_status(connection)
        prepared_completion = semantic_completion_state(prepared_status)
        if (
            bool(prepared_completion['actionable'])
            and int(prepared_completion['sealed_pending'])
                == expected_pending_records
        ):
            break
        result = run_tiny_fixture_maintenance(
            guard_connection,
            index_name,
            try_only=True,
        )
        preparation_phases.append({
            'result': result,
            'status': eventual_deferred_status(connection),
        })
    else:
        raise AssertionError(
            'semantic root race could not prepare the expected sealed rows'
        )

    before_status = eventual_deferred_status(connection)
    actor_started = threading.Event()
    actor_pid: list[int] = []
    maintain_result = ''
    query_rows: list[tuple[str, float]] = []
    after_commit_status: dict[str, Any] | None = None

    def query_actor() -> list[tuple[str, float]]:
        actor_connection = connect(socket_dir, port)
        try:
            with actor_connection.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_convergent_root_snapshot_pause_ms = '1000'"
                )
                cursor.execute('SELECT pg_backend_pid()')
                actor_pid.append(int(cursor.fetchone()[0]))
            actor_started.set()
            return eventual_deferred_query_rows(actor_connection)
        finally:
            actor_connection.close()

    def wait_for_snapshot_pause(
        future: concurrent.futures.Future[list[tuple[str, float]]],
    ) -> None:
        deadline = time.monotonic() + 15.0
        while True:
            if actor_pid:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_stat_activity
                            WHERE pid = %s
                              AND wait_event_type = 'Extension'
                        )
                        """,
                        (actor_pid[0],),
                    )
                    paused = bool(cursor.fetchone()[0])
                if paused:
                    return
            if future.done():
                raise AssertionError(
                    'semantic query finished before its root snapshot pause: '
                    f'{future.result()}'
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    'semantic query did not reach its root snapshot pause'
                )
            time.sleep(0.002)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        query_future = executor.submit(query_actor)
        if not actor_started.wait(timeout=10.0):
            raise TimeoutError('semantic query actor did not start')
        wait_for_snapshot_pause(query_future)
        with guard_connection.cursor() as cursor:
            cursor.execute(
                "SELECT evoke_index_try_maintain("
                "'p2_eventual.deferred_idx'::regclass)"
            )
            row = cursor.fetchone()
        if row is None:
            raise AssertionError('semantic maintenance returned no row')
        maintain_result = str(row[0])
        query_rows = query_future.result(timeout=30.0)
    after_status = eventual_deferred_status(connection)
    guard_connection.commit()
    after_commit_status = eventual_deferred_status(connection)

    passed = (
        actor_pid
        and any(
            'reason=active_l0_rotated' in phase['result']
            for phase in preparation_phases
        )
        and any(
            'reason=pending_l0_sealed' in phase['result']
            for phase in preparation_phases
        )
        and 'reason=semantic_completed' in maintain_result
        and 'mode=semantic_completion' in maintain_result
        and generation_identity(after_status)
            == generation_identity(before_status)
        and semantic_completion_state(before_status)['pending']
            == expected_pending_records
        and semantic_completion_state(before_status)['pending_exact'] is True
        and semantic_completion_state(before_status)['actionable'] is True
        and semantic_completion_state(after_status)['pending']
            == expected_pending_records
        and after_commit_status is not None
        and int(
            semantic_completion_state(after_commit_status)['telemetry'][
                'completed'
            ]
        ) >= 1
        and generation_identity(after_commit_status)
            == generation_identity(before_status)
        and query_digest(query_rows) == query_digest(expected_rows)
    )
    return {
        'query_pid': actor_pid[0],
        'preparation_phases': preparation_phases,
        'before_status': before_status,
        'maintain_result': maintain_result,
        'after_status': after_status,
        'after_commit_status': after_commit_status,
        'expected_digest': query_digest(expected_rows),
        'query_digest': query_digest(query_rows),
        'passed': passed,
    }


def bm25_status(connection: psycopg.Connection[Any]) -> dict[str, Any]:
    return fetch_json(
        connection,
        "SELECT evoke_index_status('p2_bm25.docs_idx'::regclass)",
    )


def bm25_query_hits(
    connection: psycopg.Connection[Any],
) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id
            FROM evoke_query(
                'p2_bm25.docs_idx'::regclass,
                %s,
                20
            ) AS hit
            JOIN p2_bm25.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """,
            (QUERY,),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def relation_names(connection: psycopg.Connection[Any]) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT relation.relname
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'p2_mutable'
            ORDER BY relation.relname
            """
        )
        return [str(row[0]) for row in cursor.fetchall()]


def configure_cluster(
    data_dir: Path,
    socket_dir: Path,
    port: int,
    extension_libdir: Path | None,
    extension_control_dir: Path | None,
) -> None:
    config = data_dir / 'postgresql.conf'
    with config.open('a', encoding='utf-8') as handle:
        handle.write("\nshared_preload_libraries = 'evoke'\n")
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
        handle.write("listen_addresses = ''\n")
        handle.write(f"unix_socket_directories = '{socket_dir}'\n")
        handle.write(f'port = {port}\n')
        handle.write('max_worker_processes = 16\n')
        handle.write('max_prepared_transactions = 10\n')
        handle.write("evoke.maintenance_timer_interval_ms = '100ms'\n")
        handle.write("evoke.shared_runtime_size = '256MB'\n")


def start_cluster(pg_ctl: Path, data_dir: Path, log_path: Path) -> None:
    run(
        [
            str(pg_ctl),
            '-D',
            str(data_dir),
            '-l',
            str(log_path),
            'start',
            '-w',
        ]
    )


def restart_cluster(pg_ctl: Path, data_dir: Path, log_path: Path) -> None:
    run(
        [
            str(pg_ctl),
            '-D',
            str(data_dir),
            '-l',
            str(log_path),
            'restart',
            '-m',
            'fast',
            '-w',
        ]
    )


def crash_restart_cluster(
    pg_ctl: Path,
    data_dir: Path,
    log_path: Path,
) -> None:
    run(
        [
            str(pg_ctl),
            '-D',
            str(data_dir),
            'stop',
            '-m',
            'immediate',
            '-w',
        ]
    )
    start_cluster(pg_ctl, data_dir, log_path)


def stop_cluster(pg_ctl: Path, data_dir: Path) -> None:
    run(
        [
            str(pg_ctl),
            '-D',
            str(data_dir),
            'stop',
            '-m',
            'fast',
        ],
        check=False,
    )


def setup(
    connection: psycopg.Connection[Any],
    model_path: Path,
) -> None:
    ddl = f"""
        CREATE EXTENSION evoke;
        CREATE SCHEMA p2_mutable;
        CREATE TABLE p2_mutable.docs (
            id text PRIMARY KEY,
            body text NOT NULL,
            metadata text NOT NULL DEFAULT 'initial'
        );
        INSERT INTO p2_mutable.docs (id, body) VALUES
            (
                'base-target',
                'rare zebra quantum flux capacitor semantic retrieval'
            ),
            (
                'base-medical',
                'cardiovascular treatment and clinical trial evidence'
            ),
            (
                'base-db',
                'postgresql inverted index maintenance and recovery'
            ),
            (
                'base-space',
                'astronomy galaxy telescope observation'
            );
        CREATE INDEX docs_idx
        ON p2_mutable.docs
        USING evoke (body)
        WITH (
            sae = true,
            model_path = {sql_literal(str(model_path))}
        );
    """
    with connection.cursor() as cursor:
        cursor.execute(ddl)


def setup_eventual(
    connection: psycopg.Connection[Any],
    model_path: Path,
) -> None:
    ddl = f"""
        CREATE SCHEMA p2_eventual;
        CREATE TABLE p2_eventual.docs (
            id text PRIMARY KEY,
            body text NOT NULL,
            metadata text NOT NULL DEFAULT 'initial'
        );
        INSERT INTO p2_eventual.docs (id, body) VALUES
            ('eventual-base-a', 'database index maintenance'),
            ('eventual-base-b', 'semantic retrieval baseline');
        CREATE INDEX docs_idx
        ON p2_eventual.docs
        USING evoke (body)
        WITH (
            sae = true,
            model_path = {sql_literal(str(model_path))},
            consistency = eventual
        );
        CREATE TABLE p2_eventual.deferred_docs (
            id text PRIMARY KEY,
            body text NOT NULL,
            scope text NOT NULL DEFAULT repeat('scope-', 1024)
        );
        ALTER TABLE p2_eventual.deferred_docs
        ALTER COLUMN scope SET STORAGE EXTERNAL;
        INSERT INTO p2_eventual.deferred_docs (id, body) VALUES
            ('deferred-base-a', 'database index maintenance'),
            ('deferred-base-b', 'semantic retrieval baseline');
        CREATE INDEX deferred_idx
        ON p2_eventual.deferred_docs
        USING evoke (body)
        INCLUDE (id, scope)
        WITH (
            sae = true,
            model_path = {sql_literal(str(model_path))},
            consistency = eventual
        );
    """
    with connection.cursor() as cursor:
        cursor.execute(ddl)


def run_temporary_relation_policy_audit(
    connection: psycopg.Connection[Any],
    model_path: Path,
) -> dict[str, Any]:
    automatic_rejected = False
    automatic_error = ''
    sae_rejected = False
    sae_error = ''
    bm25_hits = 0
    maintenance_results: list[str] = []

    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TEMP TABLE evoke_temp_bm25_docs (
                id text PRIMARY KEY,
                body text NOT NULL
            );
            INSERT INTO evoke_temp_bm25_docs VALUES
                ('bm25-a', 'temporary lexical sentinel'),
                ('bm25-b', 'other temporary text');
            CREATE INDEX evoke_temp_bm25_idx
            ON evoke_temp_bm25_docs
            USING evoke (body)
            WITH (sae = false, consistency = manual);
            INSERT INTO evoke_temp_bm25_docs VALUES
                ('bm25-c', 'temporary lexical sentinel updated');
            UPDATE evoke_temp_bm25_docs
            SET body = 'temporary lexical sentinel replacement'
            WHERE id = 'bm25-a';
            DELETE FROM evoke_temp_bm25_docs WHERE id = 'bm25-b';
            """
        )
        cursor.execute(
            "SELECT evoke_index_maintain("
            "'evoke_temp_bm25_idx'::regclass)"
        )
        maintenance_results.append(str(cursor.fetchone()[0]))
        cursor.execute(
            """
            SELECT count(*)
            FROM evoke_query(
                'evoke_temp_bm25_idx'::regclass,
                'lexical sentinel',
                10
            ) AS hit
            JOIN evoke_temp_bm25_docs AS source
              ON source.ctid = hit.ctid
            """
        )
        bm25_hits = int(cursor.fetchone()[0])
        cursor.execute(
            """
            CREATE TEMP TABLE evoke_temp_automatic_docs (
                id text PRIMARY KEY,
                body text NOT NULL
            );
            CREATE TEMP TABLE evoke_temp_sae_docs (
                id text PRIMARY KEY,
                body text NOT NULL
            );
            """
        )

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE INDEX evoke_temp_automatic_idx
                ON evoke_temp_automatic_docs
                USING evoke (body)
                WITH (sae = false, consistency = eventual)
                """
            )
    except psycopg.errors.FeatureNotSupported as error:
        automatic_error = str(error)
        automatic_rejected = (
            'temporary evoke indexes require manual consistency'
            in automatic_error
        )

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE INDEX evoke_temp_sae_idx
                ON evoke_temp_sae_docs
                USING evoke (body)
                WITH (
                    sae = true,
                    model_path = {sql_literal(str(model_path))}
                )
                """
            )
    except psycopg.errors.FeatureNotSupported as error:
        sae_error = str(error)
        sae_rejected = (
            'temporary evoke indexes do not support SSR' in sae_error
        )

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*)
            FROM pg_catalog.pg_class
            WHERE relnamespace = pg_my_temp_schema()
              AND relname IN (
                  'evoke_temp_automatic_idx',
                  'evoke_temp_sae_idx'
              )
            """
        )
        failed_index_count = int(cursor.fetchone()[0])
        cursor.execute(
            """
            DROP TABLE evoke_temp_automatic_docs;
            DROP TABLE evoke_temp_sae_docs;
            DROP TABLE evoke_temp_bm25_docs;
            """
        )
        cursor.execute(
            """
            SELECT count(*)
            FROM pg_catalog.pg_class
            WHERE relnamespace = pg_my_temp_schema()
              AND relname LIKE 'evoke_temp_%'
            """
        )
        residue_count = int(cursor.fetchone()[0])

    return {
        'bm25_hits': bm25_hits,
        'maintenance_results': maintenance_results,
        'automatic_rejected': automatic_rejected,
        'automatic_error': automatic_error,
        'sae_rejected': sae_rejected,
        'sae_error': sae_error,
        'failed_index_count': failed_index_count,
        'residue_count': residue_count,
        'passed': (
            bm25_hits >= 2
            and automatic_rejected
            and sae_rejected
            and failed_index_count == 0
            and residue_count == 0
        ),
    }


def setup_bm25(connection: psycopg.Connection[Any]) -> None:
    ddl = """
        CREATE SCHEMA p2_bm25;
        CREATE TABLE p2_bm25.docs (
            id text PRIMARY KEY,
            body text NOT NULL
        );
        INSERT INTO p2_bm25.docs VALUES
            ('bm25-base-a', 'database index maintenance'),
            ('bm25-base-b', 'semantic retrieval baseline');
        CREATE INDEX docs_idx
        ON p2_bm25.docs
        USING evoke (body)
        WITH (
            sae = false,
            consistency = realtime
        );
        CREATE TABLE p2_bm25.contract_docs (
            id text PRIMARY KEY,
            body text NOT NULL
        );
        INSERT INTO p2_bm25.contract_docs VALUES
            ('contract-base-a', 'database index maintenance'),
            ('contract-base-b', 'semantic retrieval baseline');
        CREATE INDEX contract_idx
        ON p2_bm25.contract_docs
        USING evoke (body)
        WITH (
            sae = false,
            consistency = eventual
        );
    """
    with connection.cursor() as cursor:
        cursor.execute(ddl)


def bm25_fold_status(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    return fetch_json(
        connection,
        "SELECT evoke_index_status('p2_bm25_fold.docs_idx'::regclass)",
    )


def bm25_fold_query_hits(
    connection: psycopg.Connection[Any],
    query_text: str,
) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id
            FROM evoke_query(
                'p2_bm25_fold.docs_idx'::regclass,
                %s,
                20
            ) AS hit
            JOIN p2_bm25_fold.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """,
            (query_text,),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def acquire_bm25_fold_maintenance_guard(
    connection: psycopg.Connection[Any],
    *,
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT evoke_index_try_maintenance_lock("
                "'p2_bm25_fold.docs_idx'::regclass)"
            )
            acquired = bool(cursor.fetchone()[0])
        if acquired:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                'could not acquire the BM25 fold maintenance guard'
            )
        time.sleep(0.002)


def release_bm25_fold_maintenance_guard(
    connection: psycopg.Connection[Any],
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT evoke_index_maintenance_unlock("
            "'p2_bm25_fold.docs_idx'::regclass)"
        )


def run_bm25_page_native_seal_race_audit(
    connection: psycopg.Connection[Any],
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    ddl = """
        CREATE SCHEMA p2_bm25_fold;
        CREATE TABLE p2_bm25_fold.docs (
            id text PRIMARY KEY,
            body text NOT NULL
        );
        INSERT INTO p2_bm25_fold.docs VALUES
            ('fold-base-a', 'fold baseline database index'),
            ('fold-base-b', 'fold baseline semantic retrieval');
        CREATE INDEX docs_idx
        ON p2_bm25_fold.docs
        USING evoke (body)
        WITH (
            sae = false,
            consistency = eventual
        );
    """
    with connection.cursor() as cursor:
        cursor.execute(ddl)

    acquire_bm25_fold_maintenance_guard(connection)
    initial_status = bm25_fold_status(connection)

    same_xact_errors: dict[str, str] = {}
    same_xact_connection = connect(socket_dir, port)
    same_xact_connection.autocommit = False
    try:
        with same_xact_connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO p2_bm25_fold.docs VALUES (%s, %s)',
                (
                    'fold-same-xact-abort',
                    'foldsamexactabortunique rejected maintenance',
                ),
            )
            for ordinal, function_name in enumerate(
                (
                    'evoke_index_try_maintain',
                    'evoke_index_maintain',
                    'evoke_index_refresh',
                ),
                start=1,
            ):
                savepoint = f'evoke_same_xact_{ordinal}'
                cursor.execute(f'SAVEPOINT {savepoint}')
                try:
                    cursor.execute(
                        f"SELECT {function_name}("
                        "'p2_bm25_fold.docs_idx'::regclass)"
                    )
                    row = cursor.fetchone()
                    same_xact_errors[function_name] = (
                        f'unexpected success: {row}'
                    )
                except psycopg.Error as error:
                    same_xact_errors[function_name] = str(error)
                    cursor.execute(f'ROLLBACK TO SAVEPOINT {savepoint}')
                cursor.execute(f'RELEASE SAVEPOINT {savepoint}')
        same_xact_connection.rollback()
    finally:
        same_xact_connection.close()

    savepoint_connection = connect(socket_dir, port)
    savepoint_connection.autocommit = False
    try:
        with savepoint_connection.cursor() as cursor:
            cursor.execute('SAVEPOINT evoke_bm25_fold_abort')
            cursor.execute(
                'INSERT INTO p2_bm25_fold.docs VALUES (%s, %s)',
                (
                    'fold-savepoint-abort',
                    'foldsavepointabortunique rejected row',
                ),
            )
            cursor.execute('ROLLBACK TO SAVEPOINT evoke_bm25_fold_abort')
            cursor.execute('RELEASE SAVEPOINT evoke_bm25_fold_abort')
        savepoint_connection.commit()
    finally:
        savepoint_connection.close()

    writer_connection = connect(socket_dir, port)
    writer_connection.autocommit = False
    try:
        with writer_connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO p2_bm25_fold.docs VALUES (%s, %s)',
                (
                    'fold-open-abort',
                    'foldopenabortunique uncommitted writer',
                ),
            )
        before_blocked_maintain = bm25_fold_status(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT evoke_index_try_maintain("
                "'p2_bm25_fold.docs_idx'::regclass)"
            )
            blocked_maintain_result = str(cursor.fetchone()[0])
        after_blocked_maintain = bm25_fold_status(connection)
        writer_connection.rollback()
    finally:
        writer_connection.close()

    with connection.cursor() as cursor:
        cursor.execute(
            'INSERT INTO p2_bm25_fold.docs VALUES (%s, %s)',
            (
                'fold-seed',
                'foldseedunique committed snapshot member',
            ),
        )
    race_preparation_result = run_tiny_fixture_maintenance(
        connection,
        'p2_bm25_fold.docs_idx',
        try_only=True,
    )
    before_race_status = bm25_fold_status(connection)

    actor_started = threading.Event()
    actor_guard_acquired = threading.Event()
    actor_pid: list[int] = []

    def maintenance_actor() -> str:
        actor_connection = connect(socket_dir, port)
        actor_guard_held = False
        try:
            with actor_connection.cursor() as cursor:
                # This dedicated actor must force the tiny fixture through the
                # pending-seal pause exercised below.
                cursor.execute(
                    'SET evoke.test_convergent_l0_rotation_records = 1'
                )
                cursor.execute(
                    "SET evoke.test_online_maintenance_pause_ms = '1000'"
                )
                cursor.execute('SELECT pg_backend_pid()')
                actor_pid.append(int(cursor.fetchone()[0]))
            actor_started.set()
            acquire_bm25_fold_maintenance_guard(actor_connection)
            actor_guard_held = True
            actor_guard_acquired.set()
            with actor_connection.cursor() as cursor:
                cursor.execute(
                    "SELECT evoke_index_try_maintain("
                    "'p2_bm25_fold.docs_idx'::regclass)"
                )
                row = cursor.fetchone()
            if row is None:
                raise AssertionError('BM25 fold maintenance returned no row')
            return str(row[0])
        finally:
            if actor_guard_held:
                release_bm25_fold_maintenance_guard(actor_connection)
            actor_connection.close()

    def wait_for_actor_pause(
        future: concurrent.futures.Future[str],
    ) -> None:
        deadline = time.monotonic() + 15.0
        while True:
            if actor_pid:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_stat_activity
                            WHERE pid = %s
                              AND wait_event_type = 'Extension'
                        )
                        """,
                        (actor_pid[0],),
                    )
                    paused = bool(cursor.fetchone()[0])
                if paused:
                    return
            if future.done():
                raise AssertionError(
                    'BM25 pending seal finished before its test pause: '
                    f'{future.result()}'
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    'BM25 pending seal did not reach its test pause'
                )
            time.sleep(0.002)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        maintenance_future = executor.submit(maintenance_actor)
        if not actor_started.wait(timeout=10.0):
            raise TimeoutError('BM25 fold maintenance actor did not start')
        release_bm25_fold_maintenance_guard(connection)
        if not actor_guard_acquired.wait(timeout=10.0):
            raise TimeoutError(
                'BM25 fold maintenance actor did not acquire its guard'
            )
        wait_for_actor_pause(maintenance_future)
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO p2_bm25_fold.docs VALUES (%s, %s)',
                (
                    'fold-tail',
                    'foldtailunique concurrent committed writer',
                ),
            )
        race_result = maintenance_future.result(timeout=30.0)

    after_race_status = bm25_fold_status(connection)
    race_seed_hits = bm25_fold_query_hits(connection, 'foldseedunique')
    race_tail_hits = bm25_fold_query_hits(connection, 'foldtailunique')
    followup_results: list[str] = []
    acquire_bm25_fold_maintenance_guard(connection)
    try:
        for _ in range(8):
            followup_results.append(
                run_tiny_fixture_maintenance(
                    connection,
                    'p2_bm25_fold.docs_idx',
                    try_only=True,
                )
            )
            current_status = bm25_fold_status(connection)
            if (
                int(current_status['details']['delta_records']) == 0
                and int(current_status['details']['pending_writes']) == 0
            ):
                break
    finally:
        release_bm25_fold_maintenance_guard(connection)

    compacted_status = bm25_fold_status(connection)
    seed_hits = bm25_fold_query_hits(connection, 'foldseedunique')
    tail_hits = bm25_fold_query_hits(connection, 'foldtailunique')
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT id FROM p2_bm25_fold.docs ORDER BY id'
        )
        table_ids = [str(row[0]) for row in cursor.fetchall()]

    expected_ids = [
        'fold-base-a',
        'fold-base-b',
        'fold-seed',
        'fold-tail',
    ]
    # A scheduler may rotate the concurrent active L0 before the follow-up
    # actor acquires its guard. Final convergence and both ranked hits are the
    # ownership invariant; the exact actor that reports the rotation is not.
    passed = (
        bm25_generation_ready(initial_status, expected_docs=2)
        and all(
            'modified by this transaction' in message
            for message in same_xact_errors.values()
        )
        and len(same_xact_errors) == 3
        and 'maintained=false' in blocked_maintain_result
        and 'reason=no_pending' in blocked_maintain_result
        and generation_identity(before_blocked_maintain)
            == generation_identity(after_blocked_maintain)
        and int(before_blocked_maintain['details']['delta_records'])
            == int(after_blocked_maintain['details']['delta_records'])
        and int(before_race_status['details']['delta_records']) > 0
        and int(before_race_status['details']['pending_writes']) > 0
        and 'reason=active_l0_rotated' in race_preparation_result
        and 'reason=pending_l0_sealed' in race_result
        and 'mode=segment_seal' in race_result
        and generation_identity(after_race_status)
            != generation_identity(before_race_status)
        and int(after_race_status['details']['delta_records']) > 0
        and int(after_race_status['details']['pending_writes']) > 0
        and race_seed_hits
        and race_seed_hits[0] == 'fold-seed'
        and race_tail_hits
        and race_tail_hits[0] == 'fold-tail'
        and any(
            'reason=pending_l0_sealed' in result
            for result in followup_results
        )
        and bm25_generation_ready(
            compacted_status,
            expected_docs=len(expected_ids),
        )
        and int(compacted_status['details']['delta_records']) == 0
        and int(compacted_status['details']['pending_writes']) == 0
        and table_ids == expected_ids
        and seed_hits
        and seed_hits[0] == 'fold-seed'
        and tail_hits
        and tail_hits[0] == 'fold-tail'
    )
    return {
        'initial_status': initial_status,
        'same_transaction_errors': same_xact_errors,
        'before_blocked_maintain': before_blocked_maintain,
        'blocked_maintain_result': blocked_maintain_result,
        'after_blocked_maintain': after_blocked_maintain,
        'race_preparation_result': race_preparation_result,
        'before_race_status': before_race_status,
        'maintenance_pid': actor_pid[0],
        'race_result': race_result,
        'after_race_status': after_race_status,
        'race_seed_hits': race_seed_hits,
        'race_tail_hits': race_tail_hits,
        'followup_results': followup_results,
        'compacted_status': compacted_status,
        'table_ids': table_ids,
        'seed_hits': seed_hits,
        'tail_hits': tail_hits,
        'passed': passed,
    }


def backend_memory_state(
    connection: psycopg.Connection[Any],
) -> dict[str, int]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                count(*) FILTER (
                    WHERE name = 'evoke pending SAE mutation'
                )::int8,
                COALESCE(
                    sum(total_bytes) FILTER (
                        WHERE name = 'evoke pending SAE mutation'
                    ),
                    0
                )::int8,
                COALESCE(sum(total_bytes), 0)::int8
            FROM pg_backend_memory_contexts
            """
        )
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError('backend memory context query returned no row')
    return {
        'pending_contexts': int(row[0]),
        'pending_context_bytes': int(row[1]),
        'backend_total_bytes': int(row[2]),
    }


def relation_page_lsn(
    connection: psycopg.Connection[Any],
    relation_name: str,
    block_number: int,
) -> str:
    if block_number < 0 or block_number >= 131_072:
        raise AssertionError('test relation page is outside its first segment')
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                current_setting('data_directory'),
                pg_relation_filepath(%s::regclass)
            """,
            (relation_name,),
        )
        row = cursor.fetchone()
    if row is None or row[1] is None:
        raise RuntimeError('relation file path query returned no row')
    relation_path = Path(str(row[0])) / str(row[1])
    with relation_path.open('rb') as relation_file:
        relation_file.seek(block_number * 8192)
        page_lsn = relation_file.read(8)
    if len(page_lsn) != 8:
        raise RuntimeError('could not read the relation page LSN')
    return page_lsn.hex()


def run_failed_subtransaction_memory_audit(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    warmup_cycles = 4
    measured_cycles = 64
    max_backend_growth_bytes = 2 * 1024 * 1024
    body = 'subtransaction mutation payload ' * 2048
    errors = 0

    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE FUNCTION p2_mutable.reject_memory_sentinel()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.id LIKE 'memory-fail-%' THEN
                    RAISE EXCEPTION 'intentional mutation batch failure';
                END IF;
                RETURN NEW;
            END
            $$;
            CREATE TRIGGER reject_memory_sentinel
            BEFORE INSERT ON p2_mutable.docs
            FOR EACH ROW
            EXECUTE FUNCTION p2_mutable.reject_memory_sentinel();
            """
        )

    status_before = index_status(connection)
    connection.autocommit = False
    try:
        with connection.cursor() as cursor:
            for cycle in range(warmup_cycles + measured_cycles):
                cursor.execute('SAVEPOINT evoke_failed_mutation')
                try:
                    cursor.execute(
                        """
                        INSERT INTO p2_mutable.docs (id, body)
                        VALUES
                            (%s, %s),
                            (%s, %s)
                        """,
                        (
                            f'memory-good-{cycle}',
                            body,
                            f'memory-fail-{cycle}',
                            body,
                        ),
                    )
                except psycopg.errors.RaiseException:
                    errors += 1
                    cursor.execute(
                        'ROLLBACK TO SAVEPOINT evoke_failed_mutation'
                    )
                else:
                    raise AssertionError(
                        'memory-sentinel insert unexpectedly succeeded'
                    )
                cursor.execute(
                    'RELEASE SAVEPOINT evoke_failed_mutation'
                )
                if cycle + 1 == warmup_cycles:
                    baseline = backend_memory_state(connection)
            after = backend_memory_state(connection)
            cursor.execute(
                "SELECT count(*) FROM p2_mutable.docs "
                "WHERE id LIKE 'memory-%'"
            )
            leaked_rows = int(cursor.fetchone()[0])
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                'DROP TRIGGER reject_memory_sentinel '
                'ON p2_mutable.docs'
            )
            cursor.execute(
                'DROP FUNCTION p2_mutable.reject_memory_sentinel()'
            )

    status_after = index_status(connection)
    backend_growth_bytes = (
        after['backend_total_bytes'] - baseline['backend_total_bytes']
    )
    delta_record_growth = (
        int(status_after['details']['delta_records'])
        - int(status_before['details']['delta_records'])
    )
    return {
        'warmup_cycles': warmup_cycles,
        'measured_cycles': measured_cycles,
        'caught_errors': errors,
        'payload_bytes_per_row': len(body.encode('utf-8')),
        'baseline_memory': baseline,
        'after_memory': after,
        'backend_growth_bytes': backend_growth_bytes,
        'delta_record_growth': delta_record_growth,
        'max_backend_growth_bytes': max_backend_growth_bytes,
        'leaked_rows': leaked_rows,
        'status_before': status_before,
        'status_after': status_after,
        'passed': (
            errors == warmup_cycles + measured_cycles
            and leaked_rows == 0
            and baseline['pending_contexts'] == 0
            and after['pending_contexts'] == 0
            and after['pending_context_bytes'] == 0
            and backend_growth_bytes <= max_backend_growth_bytes
            and unified_generation_ready(
                status_after,
                expected_records=posting_records(status_before),
            )
            and posting_records(status_after)
                == posting_records(status_before)
            and 0 <= delta_record_growth <= errors
        ),
    }


def run_page_native_partial_batch_audit(
    connection: psycopg.Connection[Any],
    socket_dir: Path,
    port: int,
    model_path: Path,
) -> dict[str, Any]:
    index_name = 'p2_page_native_batch.docs_idx'
    batch_rows = 16

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE SCHEMA p2_page_native_batch;
            CREATE TABLE p2_page_native_batch.docs (
                id text PRIMARY KEY,
                body text NOT NULL
            );
            INSERT INTO p2_page_native_batch.docs VALUES (
                'baseline',
                'page native partial batch baseline'
            );
            CREATE INDEX docs_idx
            ON p2_page_native_batch.docs
            USING evoke (body)
            WITH (
                sae = true,
                model_path = {sql_literal(str(model_path))}
            );
            CREATE FUNCTION p2_page_native_batch.reject_partial_row()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.id = 'rollback-fail' THEN
                    RAISE EXCEPTION 'intentional page-native batch failure';
                END IF;
                RETURN NEW;
            END
            $$;
            CREATE TRIGGER reject_partial_row
            BEFORE INSERT ON p2_page_native_batch.docs
            FOR EACH ROW
            EXECUTE FUNCTION p2_page_native_batch.reject_partial_row();
            """
        )

    status_before = fetch_json(
        connection,
        'SELECT evoke_index_status(%s::regclass)',
        (index_name,),
    )
    error_message = ''
    connection.autocommit = False
    try:
        with connection.cursor() as cursor:
            cursor.execute('SAVEPOINT evoke_page_native_partial')
            try:
                cursor.execute(
                    """
                    INSERT INTO p2_page_native_batch.docs (id, body)
                    VALUES
                        ('rollback-first', 'page native rollback first'),
                        ('rollback-fail', 'page native rollback second')
                    """
                )
            except psycopg.errors.RaiseException as error:
                error_message = str(error).splitlines()[0]
                cursor.execute(
                    'ROLLBACK TO SAVEPOINT evoke_page_native_partial'
                )
            else:
                raise AssertionError(
                    'page-native partial mutation unexpectedly succeeded'
                )
            cursor.execute('RELEASE SAVEPOINT evoke_page_native_partial')
            rollback_memory = backend_memory_state(connection)
            cursor.execute(
                "SELECT count(*) FROM p2_page_native_batch.docs "
                "WHERE id LIKE 'rollback-%'"
            )
            leaked_rows = int(cursor.fetchone()[0])
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.autocommit = True

    status_after_rollback = fetch_json(
        connection,
        'SELECT evoke_index_status(%s::regclass)',
        (index_name,),
    )
    guard = acquire_maintenance_lock(socket_dir, port, index_name)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO p2_page_native_batch.docs (id, body)
                SELECT
                    'batch-' || value,
                    'page native wal marker ' || value
                FROM generate_series(1, %s) AS value
                """,
                (batch_rows,),
            )
        pending_status = fetch_json(
            connection,
            'SELECT evoke_index_status(%s::regclass)',
            (index_name,),
        )
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM evoke_query(
                        'p2_page_native_batch.docs_idx'::regclass,
                        'page native wal marker 16',
                        20
                    ) AS hit
                    JOIN p2_page_native_batch.docs AS source
                      ON source.ctid = hit.ctid
                    WHERE source.id = 'batch-16'
                )
                """
            )
            lexical_visible = bool(cursor.fetchone()[0])
            cursor.execute('CHECKPOINT')

        delta = pending_status['generation']['delta']
        active_pages = int(delta['active']['pages'])
        delta_start_block = int(delta['start_block'])
        if active_pages <= 0 or delta_start_block <= 0:
            raise AssertionError(
                f'page-native active L0 is missing: {pending_status}'
            )
        delta_last_block = delta_start_block + active_pages - 1
        meta_page_lsn = relation_page_lsn(connection, index_name, 0)
        delta_page_lsn = relation_page_lsn(
            connection,
            index_name,
            delta_last_block,
        )
    finally:
        release_maintenance_lock(guard, index_name)

    phases, completed_status = run_eventual_maintenance_until_converged(
        connection,
        index_name,
        status_after_rollback,
        max_rounds=64,
    )
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_maintain(%s::regclass)',
            (index_name,),
        )
        post_convergence_maintenance = str(cursor.fetchone()[0])
    accelerator_stable = (
        'reason=semantic_accelerator_published' not in
        post_convergence_maintenance
    )
    with connection.cursor() as cursor:
        cursor.execute('DROP SCHEMA p2_page_native_batch CASCADE')

    rollback_physical_records = (
        int(status_after_rollback['details']['delta_records'])
        - int(status_before['details']['delta_records'])
    )
    rollback_exact = (
        'intentional page-native batch failure' in error_message
        and leaked_rows == 0
        and rollback_memory['pending_contexts'] == 0
        and rollback_memory['pending_context_bytes'] == 0
        and generation_identity(status_after_rollback)
            == generation_identity(status_before)
        and 0 <= rollback_physical_records <= 1
    )
    pending_completion = semantic_completion_state(pending_status)
    pending_exact = (
        generation_identity(pending_status)
            == generation_identity(status_after_rollback)
        and int(pending_status['details']['delta_records'])
            == batch_rows + rollback_physical_records
        and int(pending_status['details']['pending_writes'])
            == batch_rows + rollback_physical_records
        and batch_rows <= int(delta['active']['records'])
            <= batch_rows + rollback_physical_records
        and int(pending_completion['pending'])
            == batch_rows + rollback_physical_records
        and lexical_visible
    )
    wal_record_atomic = (
        meta_page_lsn != '0000000000000000'
        and meta_page_lsn == delta_page_lsn
    )
    converged = (
        unified_generation_ready(
            completed_status,
            expected_records=batch_rows + 1,
        )
        and int(completed_status['details']['delta_records']) == 0
        and int(completed_status['details']['pending_writes']) == 0
        and bool(semantic_completion_state(completed_status)['converged'])
    )
    return {
        'batch_rows': batch_rows,
        'error': error_message,
        'leaked_rows': leaked_rows,
        'rollback_memory': rollback_memory,
        'rollback_physical_records': rollback_physical_records,
        'status_before': status_before,
        'status_after_rollback': status_after_rollback,
        'pending_status': pending_status,
        'pending_active_records': int(delta['active']['records']),
        'completed_status': completed_status,
        'maintenance_phases': phases,
        'post_convergence_maintenance': post_convergence_maintenance,
        'accelerator_stable': accelerator_stable,
        'meta_page_lsn': meta_page_lsn,
        'delta_page_lsn': delta_page_lsn,
        'rollback_exact': rollback_exact,
        'pending_exact': pending_exact,
        'lexical_visible': lexical_visible,
        'wal_record_atomic': wal_record_atomic,
        'converged': converged,
        'passed': (
            rollback_exact
            and pending_exact
            and wal_record_atomic
            and converged
            and accelerator_stable
        ),
    }


def run_bulk_dml_batch_audit(
    connection: psycopg.Connection[Any],
    socket_dir: Path,
    port: int,
    model_path: Path,
) -> dict[str, Any]:
    insert_select_rows = 96
    copy_rows = 64
    index_name = 'p2_batch.docs_idx'

    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE SCHEMA p2_batch;
            CREATE TABLE p2_batch.docs (
                id int PRIMARY KEY,
                body text NOT NULL
            );
            INSERT INTO p2_batch.docs VALUES (
                0,
                'batch encoder baseline document'
            );
            """
        )
        cursor.execute(
            f"""
            CREATE INDEX docs_idx
            ON p2_batch.docs
            USING evoke (body)
            WITH (
                sae = true,
                model_path = {sql_literal(str(model_path))}
            )
            """
        )

    initial_status = fetch_json(
        connection,
        'SELECT evoke_index_status(%s::regclass)',
        (index_name,),
    )
    guard = acquire_maintenance_lock(socket_dir, port, index_name)
    runtime_before = fetch_json(
        connection,
        'SELECT evoke_runtime_service_status()',
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO p2_batch.docs
                SELECT value,
                       'insert select batch semantic marker ' || value::text
                FROM generate_series(1, %s) AS value
                """,
                (insert_select_rows,),
            )
        with connection.cursor() as cursor:
            with cursor.copy(
                'COPY p2_batch.docs (id, body) FROM STDIN'
            ) as copy:
                for offset in range(copy_rows):
                    copy.write_row((
                        insert_select_rows + offset + 1,
                        f'copy batch semantic marker {offset}',
                    ))
        runtime_after_foreground = fetch_json(
            connection,
            'SELECT evoke_runtime_service_status()',
        )
        pending_status = fetch_json(
            connection,
            'SELECT evoke_index_status(%s::regclass)',
            (index_name,),
        )
        with connection.cursor() as cursor:
            cursor.execute('SELECT count(*) FROM p2_batch.docs')
            row_count = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT source.id
                FROM evoke_query(
                    'p2_batch.docs_idx'::regclass,
                    'copy batch semantic marker 63',
                    20
                ) AS hit
                JOIN p2_batch.docs AS source ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """
            )
            copy_hits = [int(row[0]) for row in cursor.fetchall()]
        runtime_after_lexical_query = fetch_json(
            connection,
            'SELECT evoke_runtime_service_status()',
        )
    finally:
        release_maintenance_lock(guard, index_name)

    phases, completed_status = run_eventual_maintenance_until_converged(
        connection,
        index_name,
        initial_status,
        max_rounds=32,
    )
    runtime_after_completion = fetch_json(
        connection,
        'SELECT evoke_runtime_service_status()',
    )

    max_batch_size = int(runtime_before['max_supported_batch_size'])
    expected_rows = 1 + insert_select_rows + copy_rows
    expected_worker_requests = math.ceil(
        (insert_select_rows + copy_rows) / max_batch_size
    )
    foreground_request_delta = (
        int(runtime_after_foreground['requests'])
        - int(runtime_before['requests'])
    )
    foreground_encoded_delta = (
        int(runtime_after_foreground['encoded_texts'])
        - int(runtime_before['encoded_texts'])
    )
    worker_request_delta = (
        int(runtime_after_completion['requests'])
        - int(runtime_after_lexical_query['requests'])
    )
    worker_encoded_delta = (
        int(runtime_after_completion['encoded_texts'])
        - int(runtime_after_lexical_query['encoded_texts'])
    )
    worker_batch_success_delta = (
        int(runtime_after_completion['batch_successes'])
        - int(runtime_after_lexical_query['batch_successes'])
    )
    # Foreground active-L0 publication may advance the root generation. It
    # must not change the sealed posting authority or trigger a rebuild.
    passed = (
        int(pending_status['generation']['sealed_docs'])
            == int(initial_status['generation']['sealed_docs'])
        and posting_records(pending_status) == posting_records(initial_status)
        and int(pending_status['details']['rebuilds'])
            == int(initial_status['details']['rebuilds'])
        and row_count == expected_rows
        and int(pending_status['details']['delta_records'])
            == insert_select_rows + copy_rows
        and int(pending_status['details']['pending_writes'])
            == insert_select_rows + copy_rows
        and foreground_request_delta == 0
        and foreground_encoded_delta == 0
        and bool(copy_hits)
        and insert_select_rows + copy_rows in copy_hits
        and worker_request_delta == expected_worker_requests
        and worker_encoded_delta == insert_select_rows + copy_rows
        and worker_batch_success_delta == expected_worker_requests
        and int(runtime_after_completion['max_observed_batch_size'])
            >= max_batch_size
        and unified_generation_ready(
            completed_status,
            expected_records=expected_rows,
        )
        and int(completed_status['details']['delta_records']) == 0
        and int(completed_status['details']['pending_writes']) == 0
    )
    return {
        'insert_select_rows': insert_select_rows,
        'copy_rows': copy_rows,
        'expected_rows': expected_rows,
        'row_count': row_count,
        'max_batch_size': max_batch_size,
        'expected_worker_requests': expected_worker_requests,
        'foreground_request_delta': foreground_request_delta,
        'foreground_encoded_delta': foreground_encoded_delta,
        'worker_request_delta': worker_request_delta,
        'worker_encoded_delta': worker_encoded_delta,
        'worker_batch_success_delta': worker_batch_success_delta,
        'initial_status': initial_status,
        'pending_status': pending_status,
        'completed_status': completed_status,
        'maintenance_phases': phases,
        'copy_hits': copy_hits,
        'runtime_before': runtime_before,
        'runtime_after_foreground': runtime_after_foreground,
        'runtime_after_lexical_query': runtime_after_lexical_query,
        'runtime_after_completion': runtime_after_completion,
        'passed': passed,
    }
def run_mixed_soak(
    connection: psycopg.Connection[Any],
    cycles: int,
) -> dict[str, Any]:
    for cycle in range(cycles):
        doc_id = f'soak-{cycle}'
        insert_text = f'{QUERY} cycle {cycle}'
        update_text = f'{QUERY} updated cycle {cycle}'
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO p2_mutable.docs (id, body) VALUES (%s, %s)',
                (doc_id, insert_text),
            )
        if doc_id not in {item[0] for item in query_hits(connection)}:
            raise RuntimeError(f'inserted soak document is not visible: {doc_id}')
        with connection.cursor() as cursor:
            cursor.execute(
                'UPDATE p2_mutable.docs SET body = %s WHERE id = %s',
                (update_text, doc_id),
            )
        if doc_id not in {item[0] for item in query_hits(connection)}:
            raise RuntimeError(f'updated soak document is not visible: {doc_id}')
        with connection.cursor() as cursor:
            cursor.execute(
                'DELETE FROM p2_mutable.docs WHERE id = %s',
                (doc_id,),
            )
            cursor.execute(
                'VACUUM (INDEX_CLEANUP ON) p2_mutable.docs'
            )
        if doc_id in {item[0] for item in query_hits(connection)}:
            raise RuntimeError(f'deleted soak document remains visible: {doc_id}')

    initial_status = index_status(connection)
    maintenance_phases, status = run_eventual_maintenance_until_converged(
        connection,
        'p2_mutable.docs_idx',
        initial_status,
        max_rounds=32,
        require_generation_change=False,
    )
    with connection.cursor() as cursor:
        cursor.execute('SELECT count(*) FROM p2_mutable.docs')
        table_rows = int(cursor.fetchone()[0])
    return {
        'cycles': cycles,
        'table_rows': table_rows,
        'posting_records': posting_records(status),
        'query_ready': bool(status['query_ready']),
        'maintenance_phases': maintenance_phases,
    }


def summarize_latency_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            'count': 0,
            'p50': 0.0,
            'p95': 0.0,
            'p99': 0.0,
            'max': 0.0,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(len(ordered) * fraction) - 1)
        return round(ordered[min(index, len(ordered) - 1)], 3)

    return {
        'count': len(ordered),
        'p50': percentile(0.50),
        'p95': percentile(0.95),
        'p99': percentile(0.99),
        'max': round(ordered[-1], 3),
    }


def run_concurrent_crud_soak(
    connection: psycopg.Connection[Any],
    socket_dir: Path,
    port: int,
    *,
    cycles: int,
    readers: int,
    writers: int,
) -> dict[str, Any]:
    if cycles <= 0 or readers <= 0 or writers <= 0:
        raise ValueError(
            'concurrent CRUD cycles, readers, and writers must be positive'
        )

    runtime_before = fetch_json(
        connection,
        'SELECT evoke_runtime_service_status()',
    )
    cache_before = cache_state(connection, 'p2_mutable.docs_idx')
    stop_event = threading.Event()
    actor_count = readers + writers + 1
    start_barrier = threading.Barrier(actor_count, timeout=30.0)

    def configure_actor(
        actor_connection: psycopg.Connection[Any],
    ) -> None:
        with actor_connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = '30s'")
            cursor.execute("SET lock_timeout = '10s'")

    def probe_visibility(
        actor_connection: psycopg.Connection[Any],
        doc_id: str,
        *,
        expected: bool,
        phase: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        first_ids: list[str] = []
        first_status: dict[str, Any] | None = None
        for attempt in range(1, 51):
            ids = [item[0] for item in query_hits(actor_connection)]
            visible = doc_id in ids
            if attempt == 1:
                first_ids = ids
            if visible is expected:
                return {
                    'phase': phase,
                    'doc_id': doc_id,
                    'expected': expected,
                    'first_attempt_visible': attempt == 1,
                    'attempts': attempt,
                    'latency_ms': round(
                        (time.perf_counter() - started) * 1000.0,
                        3,
                    ),
                    'first_ids': first_ids,
                    'first_status': first_status,
                }
            if attempt == 1:
                status = index_status(actor_connection)
                first_status = {
                    'query_ready': status.get('query_ready'),
                    'blocker': status.get('blocker'),
                    'details': status.get('details'),
                    'generation': status.get('generation'),
                }
            time.sleep(0.01)
        raise RuntimeError(
            f'{phase} visibility did not converge for {doc_id}: '
            f'expected={expected}, first_ids={first_ids}, '
            f'first_status={first_status}'
        )

    def reader_actor(reader_id: int) -> dict[str, Any]:
        latencies: list[float] = []
        actor_connection: psycopg.Connection[Any] | None = None
        try:
            actor_connection = connect(socket_dir, port)
            configure_actor(actor_connection)
            if not query_hits(actor_connection):
                raise RuntimeError('reader warmup returned no rows')
            start_barrier.wait()
            while not stop_event.is_set():
                started = time.perf_counter()
                hits = query_hits(actor_connection)
                latencies.append(
                    (time.perf_counter() - started) * 1000.0
                )
                if not hits:
                    raise RuntimeError('concurrent query returned no rows')
            return {
                'actor': f'reader-{reader_id}',
                'ok': True,
                'latency_ms': summarize_latency_ms(latencies),
            }
        except Exception as error:
            stop_event.set()
            start_barrier.abort()
            return {
                'actor': f'reader-{reader_id}',
                'ok': False,
                'error': str(error),
                'latency_ms': summarize_latency_ms(latencies),
            }
        finally:
            if actor_connection is not None:
                actor_connection.close()

    def writer_actor(writer_id: int) -> dict[str, Any]:
        dml_latencies: list[float] = []
        visibility_latencies: list[float] = []
        visibility_events: list[dict[str, Any]] = []
        actor_connection: psycopg.Connection[Any] | None = None
        operations = 0
        try:
            actor_connection = connect(socket_dir, port)
            configure_actor(actor_connection)
            start_barrier.wait()
            for cycle in range(cycles):
                doc_id = f'concurrent-{writer_id}-{cycle}'
                insert_text = (
                    f'{QUERY} concurrent writer {writer_id} cycle {cycle}'
                )
                update_text = (
                    f'{QUERY} updated concurrent writer {writer_id} '
                    f'cycle {cycle}'
                )
                started = time.perf_counter()
                with actor_connection.cursor() as cursor:
                    cursor.execute(
                        '''
                        INSERT INTO p2_mutable.docs (id, body)
                        VALUES (%s, %s)
                        ''',
                        (doc_id, insert_text),
                    )
                dml_latencies.append(
                    (time.perf_counter() - started) * 1000.0
                )
                operations += 1

                event = probe_visibility(
                    actor_connection,
                    doc_id,
                    expected=True,
                    phase='insert',
                )
                visibility_events.append(event)
                visibility_latencies.append(float(event['latency_ms']))

                started = time.perf_counter()
                with actor_connection.cursor() as cursor:
                    cursor.execute(
                        '''
                        UPDATE p2_mutable.docs
                        SET body = %s
                        WHERE id = %s
                        ''',
                        (update_text, doc_id),
                    )
                dml_latencies.append(
                    (time.perf_counter() - started) * 1000.0
                )
                operations += 1

                event = probe_visibility(
                    actor_connection,
                    doc_id,
                    expected=True,
                    phase='update',
                )
                visibility_events.append(event)
                visibility_latencies.append(float(event['latency_ms']))

                started = time.perf_counter()
                with actor_connection.cursor() as cursor:
                    cursor.execute(
                        'DELETE FROM p2_mutable.docs WHERE id = %s',
                        (doc_id,),
                    )
                dml_latencies.append(
                    (time.perf_counter() - started) * 1000.0
                )
                operations += 1

                event = probe_visibility(
                    actor_connection,
                    doc_id,
                    expected=False,
                    phase='delete',
                )
                visibility_events.append(event)
                visibility_latencies.append(float(event['latency_ms']))

            return {
                'actor': f'writer-{writer_id}',
                'ok': True,
                'operations': operations,
                'visibility_events': visibility_events,
                'first_attempt_misses': sum(
                    not bool(event['first_attempt_visible'])
                    for event in visibility_events
                ),
                'dml_latency_ms': summarize_latency_ms(dml_latencies),
                'visibility_latency_ms': summarize_latency_ms(
                    visibility_latencies
                ),
            }
        except Exception as error:
            stop_event.set()
            start_barrier.abort()
            return {
                'actor': f'writer-{writer_id}',
                'ok': False,
                'error': str(error),
                'operations': operations,
                'visibility_events': visibility_events,
                'first_attempt_misses': sum(
                    not bool(event['first_attempt_visible'])
                    for event in visibility_events
                ),
                'dml_latency_ms': summarize_latency_ms(dml_latencies),
                'visibility_latency_ms': summarize_latency_ms(
                    visibility_latencies
                ),
            }
        finally:
            if actor_connection is not None:
                actor_connection.close()

    def maintenance_actor() -> dict[str, Any]:
        latencies: list[float] = []
        actor_connection: psycopg.Connection[Any] | None = None
        try:
            actor_connection = connect(socket_dir, port)
            configure_actor(actor_connection)
            start_barrier.wait()
            while not stop_event.is_set():
                started = time.perf_counter()
                with actor_connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT evoke_index_maintain("
                        "'p2_mutable.docs_idx'::regclass)"
                    )
                    cursor.fetchone()
                latencies.append(
                    (time.perf_counter() - started) * 1000.0
                )
                time.sleep(0.01)
            return {
                'actor': 'maintenance',
                'ok': True,
                'latency_ms': summarize_latency_ms(latencies),
            }
        except Exception as error:
            stop_event.set()
            start_barrier.abort()
            return {
                'actor': 'maintenance',
                'ok': False,
                'error': str(error),
                'latency_ms': summarize_latency_ms(latencies),
            }
        finally:
            if actor_connection is not None:
                actor_connection.close()

    worker_count = readers + writers + 1
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=worker_count,
    ) as executor:
        reader_futures = [
            executor.submit(reader_actor, reader_id)
            for reader_id in range(readers)
        ]
        writer_futures = [
            executor.submit(writer_actor, writer_id)
            for writer_id in range(writers)
        ]
        maintenance_future = executor.submit(maintenance_actor)
        writer_results = [future.result() for future in writer_futures]
        stop_event.set()
        reader_results = [future.result() for future in reader_futures]
        maintenance_result = maintenance_future.result()

    with connection.cursor() as cursor:
        cursor.execute('VACUUM p2_mutable.docs')
        cursor.execute('SELECT count(*) FROM p2_mutable.docs')
        table_rows = int(cursor.fetchone()[0])

    final_initial_status = index_status(connection)
    final_phases, final_status = run_eventual_maintenance_until_converged(
        connection,
        'p2_mutable.docs_idx',
        final_initial_status,
        max_rounds=32,
        require_generation_change=False,
    )
    final_maintenance_rounds = [
        str(phase['result'])
        for phase in final_phases
    ]

    final_hits = query_hits(connection)
    final_oracle_hits = query_hits(connection, oracle=True)
    final_cache = cache_state(connection, 'p2_mutable.docs_idx')
    for _ in range(40):
        maintenance = final_cache['maintenance']
        if (
            int(maintenance['active_background_workers']) == 0
            and int(maintenance['active_index_maintenance_workers']) == 0
        ):
            break
        time.sleep(0.05)
        final_cache = cache_state(connection, 'p2_mutable.docs_idx')
    runtime_after = fetch_json(
        connection,
        'SELECT evoke_runtime_service_status()',
    )
    actors = reader_results + writer_results + [maintenance_result]
    expected_operations = cycles * writers * 3
    observed_operations = sum(
        int(actor.get('operations', 0))
        for actor in writer_results
    )
    first_attempt_misses = sum(
        int(actor.get('first_attempt_misses', 0))
        for actor in writer_results
    )
    runtime_counter_delta = {
        key: int(runtime_after.get(key, 0))
        - int(runtime_before.get(key, 0))
        for key in (
            'requests',
            'successes',
            'failures',
            'busy_rejections',
            'canceled_requests',
            'orphan_responses',
            'worker_recoveries',
            'queue_waits',
            'queue_total_wait_ms',
        )
    }
    shared = final_cache['shared_preload']
    maintenance = final_cache['maintenance']
    passed = (
        all(bool(actor.get('ok')) for actor in actors)
        and observed_operations == expected_operations
        and first_attempt_misses == 0
        and sum(
            int(actor['latency_ms']['count'])
            for actor in reader_results
        ) > 0
        and int(maintenance_result['latency_ms']['count']) > 0
        and table_rows == 4
        and unified_generation_ready(final_status, expected_records=4)
        and int(final_status['details']['delta_records']) == 0
        and int(final_status['details']['pending_writes']) == 0
        and int(final_status['details']['pending_deletes']) == 0
        and query_rows_match(final_hits, final_oracle_hits)
        and runtime_after.get('worker_ready') is True
        and int(runtime_after.get('queue_depth', -1)) == 0
        and int(runtime_after.get('processing_batch_count', -1)) == 0
        and runtime_after.get('request_pending') is False
        and runtime_after.get('request_processing') is False
        and runtime_after.get('response_ready') is False
        and runtime_counter_delta['failures'] == 0
        and runtime_counter_delta['busy_rejections'] == 0
        and runtime_counter_delta['canceled_requests'] == 0
        and runtime_counter_delta['orphan_responses'] == 0
        and runtime_counter_delta['worker_recoveries'] == 0
        and shared['loading'] is False
        and int(shared['refcounted_entries']) == 0
        and int(maintenance['active_background_workers']) == 0
        and int(maintenance['active_index_maintenance_workers']) == 0
    )
    return {
        'cycles_per_writer': cycles,
        'readers': readers,
        'writers': writers,
        'actors': actors,
        'expected_writer_operations': expected_operations,
        'observed_writer_operations': observed_operations,
        'first_attempt_visibility_misses': first_attempt_misses,
        'table_rows': table_rows,
        'posting_records': posting_records(final_status),
        'final_maintenance': final_maintenance_rounds[-1],
        'final_maintenance_rounds': final_maintenance_rounds,
        'final_hits': final_hits,
        'final_oracle_hits': final_oracle_hits,
        'runtime_before': runtime_before,
        'runtime_after': runtime_after,
        'runtime_counter_delta': runtime_counter_delta,
        'cache_before': cache_before,
        'cache_after': final_cache,
        'status_after': final_status,
        'passed': passed,
    }


def delta_trace_query_rows(
    connection: psycopg.Connection[Any],
    query_text: str,
    *,
    oracle: bool = False,
) -> list[tuple[str, float]]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_config("
            "'evoke.test_unified_overlay_oracle', %s, false)",
            ('on' if oracle else 'off',),
        )
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'p2_delta_trace.docs_idx'::regclass,
                %s,
                100
            ) AS hit
            JOIN p2_delta_trace.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """,
            (query_text,),
        )
        return [(str(row[0]), float(row[1])) for row in cursor.fetchall()]


def tid_reuse_query_rows(
    connection: psycopg.Connection[Any],
    query_text: str,
    *,
    oracle: bool = False,
) -> list[tuple[str, float]]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_config("
            "'evoke.test_disable_semantic_accelerator', 'on', false)"
        )
        cursor.execute(
            "SELECT set_config("
            "'evoke.test_unified_overlay_oracle', %s, false)",
            ('on' if oracle else 'off',),
        )
        try:
            cursor.execute(
                """
                SELECT source.id, hit.score::float8
                FROM evoke_query(
                    'p2_tid_reuse.docs_idx'::regclass,
                    %s,
                    100
                ) AS hit
                JOIN p2_tid_reuse.docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """,
                (query_text,),
            )
            return [
                (str(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        finally:
            cursor.execute(
                "SELECT set_config("
                "'evoke.test_disable_semantic_accelerator', 'off', false)"
            )


def _run_tid_reuse_after_vacuum_audit(
    connection: psycopg.Connection[Any],
    model_path: Path,
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    old_query = 'obsolete narwhal amber archive'
    new_query = 'replacement zephyr cobalt ledger'
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE SCHEMA p2_tid_reuse;
            CREATE TABLE p2_tid_reuse.docs (
                id text PRIMARY KEY,
                body text NOT NULL
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO p2_tid_reuse.docs
            VALUES ('old-row', %s)
            RETURNING ctid::text
            """,
            (old_query,),
        )
        old_tid = str(cursor.fetchone()[0])
        cursor.execute(
            'CREATE INDEX docs_idx ON p2_tid_reuse.docs '
            'USING evoke (body) WITH ('
            'sae = true, '
            f'model_path = {sql_literal(str(model_path))}, '
            'consistency = eventual)'
        )

    initial_status = fetch_json(
        connection,
        "SELECT evoke_index_status('p2_tid_reuse.docs_idx'::regclass)",
    )
    guard_connection = acquire_maintenance_lock(
        socket_dir,
        port,
        'p2_tid_reuse.docs_idx',
    )
    initial_generation = generation_identity(initial_status)
    with connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM p2_tid_reuse.docs WHERE id = 'old-row'"
        )
    vacuum_with_session_maintenance_lock(
        guard_connection,
        'p2_tid_reuse.docs',
        index_cleanup=True,
    )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO p2_tid_reuse.docs
            VALUES ('replacement-row', %s)
            RETURNING ctid::text
            """,
            (new_query,),
        )
        replacement_tid = str(cursor.fetchone()[0])

    tid_reused = replacement_tid == old_tid
    overlay_status = fetch_json(
        connection,
        "SELECT evoke_index_status('p2_tid_reuse.docs_idx'::regclass)",
    )
    vacuum_attempts = 1
    while (
        not tid_reused
        and int(overlay_status['details']['delta_records']) < 2
        and vacuum_attempts < 4
    ):
        time.sleep(0.05)
        vacuum_with_session_maintenance_lock(
            guard_connection,
            'p2_tid_reuse.docs',
            index_cleanup=True,
        )
        vacuum_attempts += 1
        overlay_status = fetch_json(
            connection,
            "SELECT evoke_index_status('p2_tid_reuse.docs_idx'::regclass)",
        )
    queries = [old_query, new_query]
    overlay_rows = {
        query_text: tid_reuse_query_rows(connection, query_text)
        for query_text in queries
    }
    oracle_rows = {
        query_text: tid_reuse_query_rows(
            connection,
            query_text,
            oracle=True,
        )
        for query_text in queries
    }
    maintenance_phases, compacted_status = (
        run_eventual_maintenance_until_converged(
            guard_connection,
            'p2_tid_reuse.docs_idx',
            initial_status,
        )
    )
    release_maintenance_lock(
        guard_connection,
        'p2_tid_reuse.docs_idx',
    )
    compacted_rows = {
        query_text: tid_reuse_query_rows(connection, query_text)
        for query_text in queries
    }
    with connection.cursor() as cursor:
        cursor.execute('REINDEX INDEX p2_tid_reuse.docs_idx')
    reindexed_status = fetch_json(
        connection,
        "SELECT evoke_index_status('p2_tid_reuse.docs_idx'::regclass)",
    )
    reindexed_rows = {
        query_text: tid_reuse_query_rows(connection, query_text)
        for query_text in queries
    }
    overlay_ids = {
        doc_id
        for rows in overlay_rows.values()
        for doc_id, _score in rows
    }
    overlay_generation = generation_identity(overlay_status)
    overlay_delta_records = int(
        overlay_status['details']['delta_records']
    )
    transition_state_valid = (
        not tid_reused
        or
        (
            overlay_generation == initial_generation
            and overlay_delta_records >= 2
        )
        or (
            overlay_generation != initial_generation
            and overlay_delta_records >= 1
            and posting_records(overlay_status) == 0
        )
    )
    passed = (
        transition_state_valid
        and query_matrix_matches(overlay_rows, oracle_rows)
        and query_matrix_matches(compacted_rows, reindexed_rows)
        and int(compacted_status['details']['delta_records']) == 0
        and int(reindexed_status['details']['delta_records']) == 0
        and posting_records(compacted_status) == 1
        and posting_records(reindexed_status) == 1
        and 'old-row' not in overlay_ids
        and 'replacement-row' in overlay_ids
    )
    return {
        'old_tid': old_tid,
        'replacement_tid': replacement_tid,
        'vacuum_attempts': vacuum_attempts,
        'tid_reused': tid_reused,
        'tid_reuse_observed': tid_reused,
        'transition_state': (
            'exact_delta'
            if overlay_generation == initial_generation
            else 'reconciled_generation'
        ),
        'transition_state_valid': transition_state_valid,
        'initial_status': initial_status,
        'overlay_status': overlay_status,
        'overlay_digests': {
            query_text: query_digest(rows)
            for query_text, rows in overlay_rows.items()
        },
        'oracle_digests': {
            query_text: query_digest(rows)
            for query_text, rows in oracle_rows.items()
        },
        'maintenance_phases': maintenance_phases,
        'compacted_status': compacted_status,
        'compacted_digests': {
            query_text: query_digest(rows)
            for query_text, rows in compacted_rows.items()
        },
        'reindexed_status': reindexed_status,
        'reindexed_digests': {
            query_text: query_digest(rows)
            for query_text, rows in reindexed_rows.items()
        },
        'overlay_to_oracle_max_score_delta': (
            query_matrix_max_score_delta(overlay_rows, oracle_rows)
        ),
        'overlay_to_compaction_max_score_delta': (
            query_matrix_max_score_delta(overlay_rows, compacted_rows)
        ),
        'compaction_to_reindex_max_score_delta': (
            query_matrix_max_score_delta(compacted_rows, reindexed_rows)
        ),
        'passed': passed,
    }


def run_tid_reuse_after_vacuum_audit(
    connection: psycopg.Connection[Any],
    model_path: Path,
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute('SHOW evoke.maintenance_worker_limit')
        original_worker_limit = int(cursor.fetchone()[0])

    configure_maintenance_worker_limit(connection, 0)
    wait_for_maintenance_worker_quiescence(connection)
    try:
        return _run_tid_reuse_after_vacuum_audit(
            connection,
            model_path,
            socket_dir,
            port,
        )
    finally:
        configure_maintenance_worker_limit(
            connection,
            original_worker_limit,
        )


def run_unified_delta_crud_differential(
    connection: psycopg.Connection[Any],
    model_path: Path,
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    queries = [
        QUERY,
        'cardiovascular clinical trial evidence',
        'postgresql inverted index recovery',
        'astronomy telescope galaxy observation',
    ]
    initial_documents = {
        f'trace-base-{index}': f'{queries[index % len(queries)]} base {index}'
        for index in range(12)
    }
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE SCHEMA p2_delta_trace;
            CREATE TABLE p2_delta_trace.docs (
                id text PRIMARY KEY,
                body text NOT NULL
            )
            """
        )
        cursor.executemany(
            'INSERT INTO p2_delta_trace.docs VALUES (%s, %s)',
            list(initial_documents.items()),
        )
        cursor.execute(
            'CREATE INDEX docs_idx ON p2_delta_trace.docs '
            'USING evoke (body) WITH ('
            'sae = true, '
            f'model_path = {sql_literal(str(model_path))}, '
            'consistency = eventual)'
        )

    initial_status = fetch_json(
        connection,
        "SELECT evoke_index_status('p2_delta_trace.docs_idx'::regclass)",
    )
    guard_connection = acquire_maintenance_lock(
        socket_dir,
        port,
        'p2_delta_trace.docs_idx',
    )
    initial_generation = generation_identity(initial_status)
    live_documents = dict(initial_documents)
    operations: list[dict[str, str]] = []
    rng = random.Random(2202)

    with connection.cursor() as cursor:
        cursor.execute('BEGIN')
        cursor.execute(
            'INSERT INTO p2_delta_trace.docs VALUES (%s, %s)',
            ('trace-rolled-back', f'{QUERY} rolled back'),
        )
        cursor.execute('ROLLBACK')
        cursor.execute('BEGIN')
        cursor.execute('SAVEPOINT p2_delta_trace_savepoint')
        cursor.execute(
            'INSERT INTO p2_delta_trace.docs VALUES (%s, %s)',
            ('trace-savepoint-rolled-back', f'{QUERY} savepoint'),
        )
        cursor.execute('ROLLBACK TO SAVEPOINT p2_delta_trace_savepoint')
        cursor.execute('COMMIT')

    for operation_index in range(36):
        action = rng.choices(
            ['insert', 'update', 'delete'],
            weights=[4, 5, 3],
            k=1,
        )[0]
        if action == 'delete' and len(live_documents) <= 6:
            action = 'insert'
        if action == 'insert':
            doc_id = f'trace-new-{operation_index}'
            body = (
                f'{queries[rng.randrange(len(queries))]} '
                f'insert marker {operation_index}'
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO p2_delta_trace.docs VALUES (%s, %s)',
                    (doc_id, body),
                )
            live_documents[doc_id] = body
        elif action == 'update':
            doc_id = rng.choice(sorted(live_documents))
            body = (
                f'{queries[rng.randrange(len(queries))]} '
                f'update marker {operation_index}'
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    'UPDATE p2_delta_trace.docs SET body = %s WHERE id = %s',
                    (body, doc_id),
                )
            live_documents[doc_id] = body
        else:
            doc_id = rng.choice(sorted(live_documents))
            with connection.cursor() as cursor:
                cursor.execute(
                    'DELETE FROM p2_delta_trace.docs WHERE id = %s',
                    (doc_id,),
                )
            del live_documents[doc_id]
            body = ''
        operations.append({'action': action, 'id': doc_id, 'body': body})

    with connection.cursor() as cursor:
        cursor.execute(
            'INSERT INTO p2_delta_trace.docs VALUES (%s, %s)',
            ('trace-deleted-target', QUERY),
        )
        cursor.execute(
            'DELETE FROM p2_delta_trace.docs WHERE id = %s',
            ('trace-deleted-target',),
        )
    vacuum_with_session_maintenance_lock(
        guard_connection,
        'p2_delta_trace.docs',
        index_cleanup=True,
    )

    overlay_status = fetch_json(
        connection,
        "SELECT evoke_index_status('p2_delta_trace.docs_idx'::regclass)",
    )
    overlay_rows = {
        query_text: delta_trace_query_rows(connection, query_text)
        for query_text in queries
    }
    oracle_rows = {
        query_text: delta_trace_query_rows(
            connection,
            query_text,
            oracle=True,
        )
        for query_text in queries
    }
    overlay_ids = {
        doc_id
        for rows in overlay_rows.values()
        for doc_id, _score in rows
    }
    maintenance_phases, compacted_status = (
        run_eventual_maintenance_until_converged(
            guard_connection,
            'p2_delta_trace.docs_idx',
            initial_status,
        )
    )
    release_maintenance_lock(
        guard_connection,
        'p2_delta_trace.docs_idx',
    )
    compacted_rows = {
        query_text: delta_trace_query_rows(connection, query_text)
        for query_text in queries
    }
    with connection.cursor() as cursor:
        cursor.execute('REINDEX INDEX p2_delta_trace.docs_idx')
    reindexed_status = fetch_json(
        connection,
        "SELECT evoke_index_status('p2_delta_trace.docs_idx'::regclass)",
    )
    reindexed_rows = {
        query_text: delta_trace_query_rows(connection, query_text)
        for query_text in queries
    }

    return {
        'seed': 2202,
        'operations': operations,
        'initial_generation': initial_generation,
        'initial_status': initial_status,
        'overlay_status': overlay_status,
        'score_parity_tolerance': {
            'absolute': SCORE_PARITY_ABS_TOLERANCE,
            'relative': SCORE_PARITY_REL_TOLERANCE,
        },
        'overlay_digests': {
            query_text: query_digest(rows)
            for query_text, rows in overlay_rows.items()
        },
        'oracle_digests': {
            query_text: query_digest(rows)
            for query_text, rows in oracle_rows.items()
        },
        'maintenance_phases': maintenance_phases,
        'compacted_status': compacted_status,
        'compacted_digests': {
            query_text: query_digest(rows)
            for query_text, rows in compacted_rows.items()
        },
        'reindexed_status': reindexed_status,
        'reindexed_digests': {
            query_text: query_digest(rows)
            for query_text, rows in reindexed_rows.items()
        },
        'pending_to_compaction_max_score_delta': (
            query_matrix_max_score_delta(overlay_rows, compacted_rows)
        ),
        'overlay_to_oracle_max_score_delta': (
            query_matrix_max_score_delta(overlay_rows, oracle_rows)
        ),
        'oracle_to_compaction_max_score_delta': (
            query_matrix_max_score_delta(oracle_rows, compacted_rows)
        ),
        'compaction_to_reindex_max_score_delta': (
            query_matrix_max_score_delta(compacted_rows, reindexed_rows)
        ),
        'live_document_count': len(live_documents),
        'rolled_back_documents_hidden': (
            'trace-rolled-back' not in overlay_ids
            and 'trace-savepoint-rolled-back' not in overlay_ids
            and 'trace-deleted-target' not in overlay_ids
        ),
        'passed': (
            int(overlay_status['details']['delta_records']) > 0
            and int(overlay_status['details']['delta_bytes']) > 0
            and semantic_completion_state(overlay_status)['pending'] > 0
            and query_matrix_matches(overlay_rows, oracle_rows)
            and query_matrix_matches(compacted_rows, reindexed_rows)
            and int(compacted_status['details']['delta_records']) == 0
            and int(reindexed_status['details']['delta_records']) == 0
            and posting_records(compacted_status) == len(live_documents)
            and posting_records(reindexed_status) == len(live_documents)
            and 'trace-rolled-back' not in overlay_ids
            and 'trace-savepoint-rolled-back' not in overlay_ids
            and 'trace-deleted-target' not in overlay_ids
        ),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.model_path)
    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    metrics: dict[str, float] = {}
    gates: dict[str, bool] = {}
    evidence: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix='evoke_unified_lifecycle_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        alternate_model_path = root / 'alternate-model'
        corrupt_model_path = root / 'corrupt-model'
        shutil.copytree(args.model_path, alternate_model_path)
        alternate_manifest_path = alternate_model_path / 'manifest.json'
        make_owner_writable(alternate_manifest_path)
        alternate_manifest = json.loads(
            alternate_manifest_path.read_text(encoding='utf-8')
        )
        alternate_manifest['model_id'] = (
            f"{alternate_manifest['model_id']}_contract_change"
        )
        alternate_manifest_path.write_text(
            json.dumps(alternate_manifest, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        shutil.copytree(args.model_path, corrupt_model_path)
        corrupt_manifest = load_manifest(corrupt_model_path)
        corrupt_artifact_path = (
            corrupt_model_path
            / corrupt_manifest['artifacts']['scoring_profile']['path']
        )
        make_owner_writable(corrupt_artifact_path)
        socket_dir.mkdir()

        run([str(initdb), '-D', str(data_dir), '-A', 'trust', '-U', 'postgres'])
        configure_cluster(
            data_dir,
            socket_dir,
            args.port,
            args.extension_libdir,
            args.extension_control_dir,
        )
        started = False
        connection: psycopg.Connection[Any] | None = None
        delta_guard_connection: psycopg.Connection[Any] | None = None
        eventual_guard_connection: psycopg.Connection[Any] | None = None
        deferred_guard_connection: psycopg.Connection[Any] | None = None
        try:
            timed(
                metrics,
                'cluster_start_ms',
                lambda: start_cluster(pg_ctl, data_dir, log_path),
            )
            started = True
            connection = connect(socket_dir, args.port)
            timed(
                metrics,
                'create_index_ms',
                lambda: setup(connection, args.model_path),
            )
            temporary_relation_policy_audit = timed(
                metrics,
                'temporary_relation_policy_audit_ms',
                lambda: run_temporary_relation_policy_audit(
                    connection,
                    args.model_path,
                ),
            )
            evidence['temporary_relation_policy_audit'] = (
                temporary_relation_policy_audit
            )
            gates[
                'temporary_relation_policy_is_explicit_and_exact'
            ] = temporary_relation_policy_audit['passed']
            failed_subtransaction_memory_audit = timed(
                metrics,
                'failed_subtransaction_memory_audit_ms',
                lambda: run_failed_subtransaction_memory_audit(
                    connection,
                ),
            )
            evidence['failed_subtransaction_memory_audit'] = (
                failed_subtransaction_memory_audit
            )
            gates[
                'failed_subtransactions_release_pending_mutation_memory'
            ] = failed_subtransaction_memory_audit['passed']
            page_native_partial_batch_audit = timed(
                metrics,
                'page_native_partial_batch_audit_ms',
                lambda: run_page_native_partial_batch_audit(
                    connection,
                    socket_dir,
                    args.port,
                    args.model_path,
                ),
            )
            evidence['page_native_partial_batch_audit'] = (
                page_native_partial_batch_audit
            )
            gates[
                'page_native_partial_batch_is_atomic_and_bounded'
            ] = page_native_partial_batch_audit['passed']
            gates[
                'l0_payload_and_root_share_one_wal_record'
            ] = page_native_partial_batch_audit['wal_record_atomic']
            full_text_audit = timed(
                metrics,
                'full_text_window_audit_ms',
                lambda: p2_full_text_window_audit(
                    connection,
                    args.model_path,
                    manifest,
                ),
            )
            gates.update(full_text_audit['gates'])
            evidence['full_text_window_audit'] = full_text_audit['evidence']
            bulk_dml_batch_audit = timed(
                metrics,
                'bulk_dml_batch_audit_ms',
                lambda: run_bulk_dml_batch_audit(
                    connection,
                    socket_dir,
                    args.port,
                    args.model_path,
                ),
            )
            evidence['bulk_dml_batch_audit'] = bulk_dml_batch_audit
            gates['bulk_dml_is_lexical_first_and_worker_batched'] = (
                bulk_dml_batch_audit['passed']
            )
            semantic_long_snapshot_audit = timed(
                metrics,
                'semantic_long_snapshot_compaction_audit_ms',
                lambda: run_semantic_long_snapshot_compaction_audit(
                    connection,
                    socket_dir,
                    args.port,
                    args.model_path,
                ),
            )
            evidence['semantic_long_snapshot_compaction_audit'] = (
                semantic_long_snapshot_audit
            )
            gates[
                'semantic_compaction_preserves_long_snapshot_until_vacuum'
            ] = semantic_long_snapshot_audit['passed']

            initial_status = index_status(connection)
            initial_relations = relation_names(connection)
            evidence['initial_status'] = initial_status
            evidence['initial_relations'] = initial_relations
            gates['create_index_builds_unified_generation'] = (
                unified_generation_ready(
                    initial_status,
                    expected_records=4,
                )
            )
            stale_encoding_audit = stale_query_encoding_audit(connection)
            evidence['stale_query_encoding_audit'] = stale_encoding_audit
            gates['stale_query_encoding_cannot_cross_generation'] = (
                stale_encoding_audit['passed']
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) FROM evoke_query("
                    "'p2_mutable.docs_idx'::regclass, %s, 10)",
                    (QUERY,),
                )
                public_query_count = int(cursor.fetchone()[0])
            exact_bm25_errors: list[str] = []
            for weight_mask in (None, [1.0]):
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT count(*) FROM evoke_query_tokens("
                            "'p2_mutable.docs_idx'::regclass, "
                            "ARRAY[%s]::text[], 10, %s::real[])",
                            (QUERY, weight_mask),
                        )
                except psycopg.errors.FeatureNotSupported as error:
                    exact_bm25_errors.append(str(error))
            evidence['sae_public_query_count'] = public_query_count
            evidence['sae_exact_bm25_errors'] = exact_bm25_errors
            gates['sae_uses_public_query_and_rejects_exact_diagnostics'] = (
                public_query_count >= 0
                and len(exact_bm25_errors) == 2
                and all(
                    'exact BM25 access is unavailable' in error
                    and 'Use evoke_query' in error
                    for error in exact_bm25_errors
                )
            )
            initial_runtime_signature = initial_status.get(
                'runtime_signature'
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_mutable.docs_idx '
                    f'SET (model_path = {sql_literal(str(alternate_model_path))})'
                )
            mismatch_status = index_status(connection)
            mismatch_error = ''
            try:
                query_hits(connection)
            except psycopg.Error as error:
                mismatch_error = str(error)
            evidence['runtime_generation_mismatch_status'] = mismatch_status
            evidence['runtime_generation_mismatch_error'] = mismatch_error
            gates['model_change_fails_closed_until_reindex'] = (
                initial_runtime_signature
                and mismatch_status.get('query_ready') is False
                and mismatch_status.get('blocker')
                    == 'runtime_generation_mismatch'
                and mismatch_status.get('runtime_signature_matches') is False
                and 'does not match the configured index contract'
                    in mismatch_error
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_mutable.docs_idx '
                    f'SET (model_path = {sql_literal(str(args.model_path))})'
                )
            restored_status = index_status(connection)
            gates['restoring_generation_model_restores_readiness'] = (
                restored_status.get('query_ready') is True
                and restored_status.get('runtime_signature_matches') is True
                and restored_status.get('runtime_signature')
                    == initial_runtime_signature
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_mutable.docs_idx '
                    f'SET (model_path = {sql_literal(str(corrupt_model_path))})'
                )
            pre_corruption_status = index_status(connection)
            pre_corruption_audit = index_audit(connection)
            with corrupt_artifact_path.open('ab') as handle:
                handle.write(b'\n')
            corrupt_status = index_status(connection)
            corrupt_audit = index_audit(connection)
            corrupt_query_error = ''
            try:
                query_hits(connection)
            except psycopg.Error as error:
                corrupt_query_error = str(error)
            evidence['pre_corruption_status'] = pre_corruption_status
            evidence['pre_corruption_audit'] = pre_corruption_audit
            evidence['corrupt_artifact_status'] = corrupt_status
            evidence['corrupt_artifact_audit'] = corrupt_audit
            evidence['corrupt_artifact_query_error'] = corrupt_query_error
            gates['corrupt_artifact_fails_closed'] = (
                pre_corruption_status.get('query_ready') is True
                and pre_corruption_audit.get('model_artifacts_valid') is True
                and corrupt_status.get('query_ready') is True
                and corrupt_audit.get('passed') is False
                and corrupt_audit.get('model_artifacts_valid') is False
                and 'invalid Evoke runtime model artifact' in (
                    corrupt_audit.get('model_artifact_error') or ''
                )
                and 'sha256 mismatch' in corrupt_query_error
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_mutable.docs_idx '
                    f'SET (model_path = {sql_literal(str(args.model_path))})'
                )
            artifact_restored_status = index_status(connection)
            artifact_restored_audit = index_audit(connection)
            gates['restoring_valid_artifacts_restores_readiness'] = (
                artifact_restored_status.get('query_ready') is True
                and artifact_restored_audit.get('model_artifacts_valid')
                    is True
                and artifact_restored_audit.get('passed') is True
            )
            gates['no_generation_or_sidecar_relations'] = not any(
                marker in relation
                for relation in initial_relations
                for marker in (
                    'generation',
                    'overlay',
                    'debt',
                    'identity',
                    'model_registry',
                )
            )

            with connection.cursor() as cursor:
                cursor.execute('CREATE ROLE evoke_product_reader')
                cursor.execute(
                    'GRANT USAGE ON SCHEMA p2_mutable '
                    'TO evoke_product_reader'
                )
                cursor.execute(
                    'GRANT SELECT ON p2_mutable.docs '
                    'TO evoke_product_reader'
                )
                cursor.execute('SET ROLE evoke_product_reader')
                cursor.execute(
                    "SELECT evoke_index_options("
                    "'p2_mutable.docs_idx'::regclass)"
                )
                reader_options = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT evoke_index_status("
                    "'p2_mutable.docs_idx'::regclass)"
                )
                reader_status = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT count(*) FROM evoke_query("
                    "'p2_mutable.docs_idx'::regclass, %s, 5)",
                    (QUERY,),
                )
                reader_hits = int(cursor.fetchone()[0])
                cursor.execute('RESET ROLE')
            evidence['product_reader_hits'] = reader_hits
            evidence['product_reader_options'] = reader_options
            evidence['product_reader_status'] = reader_status
            gates['unified_search_is_available_to_product_role'] = (
                reader_hits > 0
            )
            gates['unified_inspection_is_available_to_product_role'] = (
                reader_options.get('index_type') == 'semantic'
                and reader_status.get('query_ready') is True
                and reader_status.get('runtime_signature_matches') is True
            )

            hot_status_before = index_status(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT ctid::text FROM p2_mutable.docs "
                    "WHERE id = 'base-target'"
                )
                hot_tid_before = str(cursor.fetchone()[0])
                cursor.execute(
                    "UPDATE p2_mutable.docs SET metadata = 'hot-updated' "
                    "WHERE id = 'base-target'"
                )
                cursor.execute(
                    "SELECT ctid::text FROM p2_mutable.docs "
                    "WHERE id = 'base-target'"
                )
                hot_tid_after = str(cursor.fetchone()[0])
                cursor.execute(
                    """
                    SELECT hit.ctid::text
                    FROM evoke_query(
                        'p2_mutable.docs_idx'::regclass,
                        'rare zebra quantum flux capacitor semantic retrieval',
                        20
                    ) AS hit
                    JOIN p2_mutable.docs AS source
                      ON source.ctid = hit.ctid
                    WHERE source.id = 'base-target'
                    """
                )
                hot_hit = cursor.fetchone()
            hot_status_after = index_status(connection)
            evidence['hot_update_tid_before'] = hot_tid_before
            evidence['hot_update_tid_after'] = hot_tid_after
            evidence['hot_update_hit_tid'] = (
                None if hot_hit is None else str(hot_hit[0])
            )
            evidence['hot_update_status_before'] = hot_status_before
            evidence['hot_update_status_after'] = hot_status_after
            gates['nonindexed_hot_update_preserves_generation'] = (
                hot_tid_after != hot_tid_before
                and hot_hit is not None
                and str(hot_hit[0]) == hot_tid_after
                and generation_identity(hot_status_after)
                    == generation_identity(hot_status_before)
                and unified_generation_ready(
                    hot_status_after,
                    expected_records=4,
                )
            )

            # Keep this section focused on durable delta behavior. Without the
            # gate, the background worker may legally compact between two
            # observations and turn a delta assertion into a timing race.
            delta_guard_connection = acquire_maintenance_lock(
                socket_dir,
                args.port,
                'p2_mutable.docs_idx',
            )
            page_native_backend_state_audit = timed(
                metrics,
                'page_native_backend_state_audit_ms',
                lambda: run_page_native_backend_state_audit(
                    socket_dir,
                    args.port,
                ),
            )
            evidence['page_native_backend_state_audit'] = (
                page_native_backend_state_audit
            )
            gates.update(page_native_backend_state_audit['gates'])
            rollback_before = index_status(connection)
            rollback_connection = connect(socket_dir, args.port)
            rollback_connection.autocommit = False
            rollback_tid = ''
            try:
                with rollback_connection.cursor() as cursor:
                    cursor.execute(
                        'INSERT INTO p2_mutable.docs (id, body) '
                        'VALUES (%s, %s) RETURNING ctid::text',
                        ('rollback-target', QUERY),
                    )
                    rollback_tid = str(cursor.fetchone()[0])
                rollback_connection.rollback()
            finally:
                rollback_connection.close()
            rollback_after = index_status(connection)
            rollback_hits = query_hits(connection)
            evidence['rollback_status_before'] = rollback_before
            evidence['rollback_status_after'] = rollback_after
            evidence['rollback_hits'] = rollback_hits
            evidence['rollback_tid'] = rollback_tid
            gates['aborted_dml_is_hidden_with_reclaimable_delta_debt'] = (
                generation_identity(rollback_after)
                == generation_identity(rollback_before)
                and unified_generation_ready(
                    rollback_after,
                    expected_records=4,
                )
                and int(
                    rollback_after.get('details', {}).get(
                        'pending_writes',
                        -1,
                    )
                ) == int(
                    rollback_before.get('details', {}).get(
                        'pending_writes',
                        -1,
                    )
                ) + 1
                and int(
                    rollback_after.get('details', {}).get(
                        'delta_records',
                        -1,
                    )
                ) == int(
                    rollback_before.get('details', {}).get(
                        'delta_records',
                        -1,
                    )
                ) + 1
                and 'rollback-target'
                    not in {item[0] for item in rollback_hits}
            )

            savepoint_before = index_status(connection)
            savepoint_connection = connect(socket_dir, args.port)
            savepoint_connection.autocommit = False
            savepoint_tid = ''
            try:
                with savepoint_connection.cursor() as cursor:
                    cursor.execute('SAVEPOINT evoke_semantic_write')
                    cursor.execute(
                        'INSERT INTO p2_mutable.docs (id, body) '
                        'VALUES (%s, %s) RETURNING ctid::text',
                        ('savepoint-target', QUERY),
                    )
                    savepoint_tid = str(cursor.fetchone()[0])
                    cursor.execute(
                        'ROLLBACK TO SAVEPOINT evoke_semantic_write'
                    )
                    cursor.execute('RELEASE SAVEPOINT evoke_semantic_write')
                savepoint_connection.commit()
            finally:
                savepoint_connection.close()
            savepoint_after = index_status(connection)
            savepoint_hits = query_hits(connection)
            evidence['savepoint_status_before'] = savepoint_before
            evidence['savepoint_status_after'] = savepoint_after
            evidence['savepoint_hits'] = savepoint_hits
            evidence['savepoint_tid'] = savepoint_tid
            gates['savepoint_rollback_is_hidden_with_reclaimable_delta_debt'] = (
                generation_identity(savepoint_after)
                == generation_identity(savepoint_before)
                and unified_generation_ready(
                    savepoint_after,
                    expected_records=4,
                )
                and int(
                    savepoint_after.get('details', {}).get(
                        'pending_writes',
                        -1,
                    )
                ) == int(
                    savepoint_before.get('details', {}).get(
                        'pending_writes',
                        -1,
                    )
                ) + 1
                and int(
                    savepoint_after.get('details', {}).get(
                        'delta_records',
                        -1,
                    )
                ) == int(
                    savepoint_before.get('details', {}).get(
                        'delta_records',
                        -1,
                    )
                ) + 1
                and 'savepoint-target'
                    not in {item[0] for item in savepoint_hits}
            )

            committed_savepoint_connection = connect(socket_dir, args.port)
            committed_savepoint_connection.autocommit = False
            insert_tid = ''
            try:
                with committed_savepoint_connection.cursor() as cursor:
                    cursor.execute('SAVEPOINT evoke_semantic_commit')
                    cursor.execute(
                        'INSERT INTO p2_mutable.docs (id, body) '
                        'VALUES (%s, %s) RETURNING ctid::text',
                        ('insert-target', QUERY),
                    )
                    insert_tid = str(cursor.fetchone()[0])
                    cursor.execute('RELEASE SAVEPOINT evoke_semantic_commit')
                committed_savepoint_connection.commit()
            finally:
                committed_savepoint_connection.close()
            inserted_status = index_status(connection)
            inserted_hits = query_hits(connection)
            evidence['insert_status'] = inserted_status
            evidence['insert_hits'] = inserted_hits
            evidence['insert_tid'] = insert_tid
            gates['insert_uses_exact_delta_without_generation_rewrite'] = (
                generation_identity(inserted_status)
                == generation_identity(initial_status)
                and unified_generation_ready(
                    inserted_status,
                    expected_records=4,
                )
                and int(inserted_status['details']['delta_records'])
                    == int(savepoint_after['details']['delta_records']) + 1
                and int(inserted_status['details']['pending_writes'])
                    == int(savepoint_after['details']['pending_writes']) + 1
            )
            gates['insert_is_semantically_visible'] = (
                bool(inserted_hits)
                and inserted_hits[0][0] == 'insert-target'
            )
            gates['savepoint_commit_is_immediately_visible_via_delta'] = (
                gates['insert_uses_exact_delta_without_generation_rewrite']
                and gates['insert_is_semantically_visible']
            )

            update_query = 'ultraviolet narwhal database sentinel exact phrase'
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE p2_mutable.docs
                    SET body = %s
                    WHERE id = 'base-db'
                    RETURNING ctid::text
                    """,
                    (update_query,),
                )
                update_tid = str(cursor.fetchone()[0])
            updated_status = index_status(connection)
            updated_hits = query_hits(connection, update_query)
            evidence['update_status'] = updated_status
            evidence['update_hits'] = updated_hits
            evidence['update_tid'] = update_tid
            gates['update_uses_exact_delta_without_generation_rewrite'] = (
                generation_identity(updated_status)
                == generation_identity(inserted_status)
                and unified_generation_ready(
                    updated_status,
                    expected_records=4,
                )
                and int(updated_status['details']['delta_records'])
                    == int(inserted_status['details']['delta_records']) + 1
                and int(updated_status['details']['pending_writes'])
                    == int(inserted_status['details']['pending_writes']) + 1
            )
            gates['update_is_semantically_visible'] = (
                bool(updated_hits)
                and updated_hits[0][0] == 'base-db'
            )

            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM p2_mutable.docs WHERE id = 'insert-target'"
                )
            deleted_hits = query_hits(connection)
            pre_vacuum_status = index_status(connection)
            evidence['delete_hits_before_vacuum'] = deleted_hits
            evidence['pre_vacuum_status'] = pre_vacuum_status
            gates['delete_is_immediately_hidden'] = (
                'insert-target' not in {item[0] for item in deleted_hits}
            )
            gates['delete_is_mvcc_hidden_before_physical_vacuum'] = (
                generation_identity(pre_vacuum_status)
                == generation_identity(updated_status)
                and unified_generation_ready(
                    pre_vacuum_status,
                    expected_records=4,
                )
                and int(pre_vacuum_status['details']['delta_records'])
                    == int(updated_status['details']['delta_records'])
                and gates['delete_is_immediately_hidden']
            )

            hits_before_skipped_index_cleanup = query_hits(
                connection,
                update_query,
            )
            status_before_skipped_index_cleanup = index_status(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    'VACUUM (INDEX_CLEANUP OFF) p2_mutable.docs'
                )
            hits_after_skipped_index_cleanup = query_hits(
                connection,
                update_query,
            )
            status_after_skipped_index_cleanup = index_status(connection)
            evidence['skipped_index_cleanup'] = {
                'before_hits': hits_before_skipped_index_cleanup,
                'after_hits': hits_after_skipped_index_cleanup,
                'before_status': status_before_skipped_index_cleanup,
                'after_status': status_after_skipped_index_cleanup,
            }
            gates['skipped_index_cleanup_preserves_mvcc_results'] = (
                query_digest(hits_after_skipped_index_cleanup)
                == query_digest(hits_before_skipped_index_cleanup)
                and generation_identity(status_after_skipped_index_cleanup)
                == generation_identity(status_before_skipped_index_cleanup)
            )

            vacuum_with_session_maintenance_lock(
                delta_guard_connection,
                'p2_mutable.docs',
                index_cleanup=True,
            )
            vacuum_status = index_status(connection)
            evidence['vacuum_status'] = vacuum_status
            gates['vacuum_records_generation_pinned_tombstones'] = (
                generation_identity(vacuum_status)
                != generation_identity(updated_status)
                and unified_generation_ready(
                    vacuum_status,
                    expected_records=3,
                )
                and int(vacuum_status['details']['delta_records'])
                    > int(pre_vacuum_status['details']['delta_records'])
                and int(vacuum_status['details']['pending_deletes']) > 0
            )
            active_l0_mvcc_hits = query_hits(connection, update_query)
            evidence['active_l0_mvcc_hits'] = active_l0_mvcc_hits
            gates['active_l0_mvcc_xids_remain_queryable_until_seal'] = (
                bool(active_l0_mvcc_hits)
                and active_l0_mvcc_hits[0][0] == 'base-db'
                and int(vacuum_status['details']['delta_records']) > 0
            )
            release_maintenance_lock(
                delta_guard_connection,
                'p2_mutable.docs_idx',
            )
            delta_guard_connection = None
            hits_before_compaction = query_hits(connection, update_query)
            trace_before_compaction = query_trace(connection, update_query)
            oracle_hits_before_compaction = query_hits(
                connection,
                update_query,
                oracle=True,
            )
            oracle_trace_before_compaction = query_trace(
                connection,
                update_query,
                oracle=True,
            )
            digest_before_compaction = query_digest(
                hits_before_compaction
            )
            oracle_digest_before_compaction = query_digest(
                oracle_hits_before_compaction
            )
            (
                lifecycle_maintain_phases,
                compacted_lifecycle_status,
            ) = run_eventual_maintenance_until_converged(
                connection,
                'p2_mutable.docs_idx',
                vacuum_status,
                max_rounds=16,
            )
            lifecycle_maintain_result = lifecycle_maintain_phases[-1][
                'result'
            ]
            hits_after_compaction = query_hits(connection, update_query)
            digest_after_compaction = query_digest(
                hits_after_compaction
            )
            evidence['lifecycle_maintain_result'] = (
                lifecycle_maintain_result
            )
            evidence['lifecycle_maintain_phases'] = (
                lifecycle_maintain_phases
            )
            evidence['compacted_lifecycle_status'] = (
                compacted_lifecycle_status
            )
            evidence['digest_before_compaction'] = digest_before_compaction
            evidence['trace_before_compaction'] = trace_before_compaction
            evidence['oracle_digest_before_compaction'] = (
                oracle_digest_before_compaction
            )
            evidence['oracle_trace_before_compaction'] = (
                oracle_trace_before_compaction
            )
            evidence['digest_after_compaction'] = digest_after_compaction
            gates['aborted_savepoint_and_vacuum_debt_is_reclaimed'] = (
                generation_identity(compacted_lifecycle_status)
                != generation_identity(vacuum_status)
                and unified_generation_ready(
                    compacted_lifecycle_status,
                    expected_records=4,
                )
                and int(
                    compacted_lifecycle_status['details']['delta_records']
                ) == 0
                and int(
                    compacted_lifecycle_status['details']['pending_writes']
                ) == 0
                and int(
                    compacted_lifecycle_status['details']['pending_deletes']
                ) == 0
                and query_rows_match(
                    hits_before_compaction,
                    oracle_hits_before_compaction,
                )
                and {
                    doc_id for doc_id, _score in hits_after_compaction
                } == {
                    doc_id for doc_id, _score in hits_before_compaction
                }
            )

            digest_before_reindex = query_digest(
                query_hits(connection, update_query)
            )
            with connection.cursor() as cursor:
                cursor.execute('REINDEX INDEX p2_mutable.docs_idx')
            reindex_status = index_status(connection)
            digest_after_reindex = query_digest(
                query_hits(connection, update_query)
            )
            evidence['reindex_status'] = reindex_status
            evidence['digest_before_reindex'] = digest_before_reindex
            evidence['digest_after_reindex'] = digest_after_reindex
            gates['reindex_preserves_results'] = (
                digest_after_reindex == digest_before_reindex
                and unified_generation_ready(
                    reindex_status,
                    expected_records=4,
                )
            )

            before_failed_rebuild = index_status(connection)
            before_failed_digest = query_digest(
                query_hits(connection, update_query)
            )
            failed_rebuild_error: str | None = None
            missing_model = root / 'missing-model-checkout'
            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_mutable.docs_idx SET '
                    f'(model_path = {sql_literal(str(missing_model))})'
                )
            try:
                with connection.cursor() as cursor:
                    cursor.execute('REINDEX INDEX p2_mutable.docs_idx')
            except psycopg.Error as error:
                failed_rebuild_error = str(error)
            finally:
                with connection.cursor() as cursor:
                    cursor.execute(
                        'ALTER INDEX p2_mutable.docs_idx SET '
                        f'(model_path = {sql_literal(str(args.model_path))})'
                    )
            after_failed_rebuild = index_status(connection)
            after_failed_digest = query_digest(
                query_hits(connection, update_query)
            )
            evidence['failed_rebuild_error'] = failed_rebuild_error
            evidence['status_before_failed_rebuild'] = (
                before_failed_rebuild
            )
            evidence['status_after_failed_rebuild'] = after_failed_rebuild
            gates['failed_rebuild_rolls_back_complete_generation'] = (
                failed_rebuild_error is not None
                and int(after_failed_rebuild['details']['rebuilds'])
                    == int(before_failed_rebuild['details']['rebuilds'])
                and int(after_failed_rebuild['generation']['sealed_docs'])
                    == int(
                        before_failed_rebuild['generation']['sealed_docs']
                    )
                and posting_records(after_failed_rebuild)
                    == posting_records(before_failed_rebuild)
                and after_failed_rebuild['runtime_signature']
                    == before_failed_rebuild['runtime_signature']
                and after_failed_digest == before_failed_digest
                and unified_generation_ready(
                    after_failed_rebuild,
                    expected_records=4,
                )
            )

            setup_eventual(connection, args.model_path)
            eventual_initial = eventual_status(connection)
            eventual_guard_connection = acquire_maintenance_lock(
                socket_dir,
                args.port,
                'p2_eventual.docs_idx',
            )
            eventual_runtime_before = fetch_json(
                connection,
                'SELECT evoke_runtime_service_status()',
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO p2_eventual.docs (id, body) VALUES (%s, %s) '
                    'RETURNING ctid::text',
                    ('eventual-target', QUERY),
                )
                eventual_hot_tids = [str(cursor.fetchone()[0])]
                for metadata in ('hot-a', 'hot-b', 'hot-c'):
                    cursor.execute(
                        """
                        UPDATE p2_eventual.docs
                        SET metadata = %s
                        WHERE id = 'eventual-target'
                        RETURNING ctid::text
                        """,
                        (metadata,),
                    )
                    eventual_hot_tids.append(str(cursor.fetchone()[0]))
            vacuum_with_session_maintenance_lock(
                eventual_guard_connection,
                'p2_eventual.docs',
                index_cleanup=True,
            )
            eventual_runtime_after_write = fetch_json(
                connection,
                'SELECT evoke_runtime_service_status()',
            )
            eventual_before_maintain = eventual_status(connection)
            eventual_pending_hits = eventual_query_hits(connection)
            eventual_phases, eventual_after_maintain = (
                run_eventual_maintenance_until_converged(
                    eventual_guard_connection,
                    'p2_eventual.docs_idx',
                    eventual_initial,
                )
            )
            release_maintenance_lock(
                eventual_guard_connection,
                'p2_eventual.docs_idx',
            )
            eventual_guard_connection = None
            eventual_hits = eventual_query_hits(connection)
            evidence['eventual_initial_status'] = eventual_initial
            evidence['eventual_runtime_before_write'] = (
                eventual_runtime_before
            )
            evidence['eventual_runtime_after_write'] = (
                eventual_runtime_after_write
            )
            evidence['eventual_pending_hot_tids'] = eventual_hot_tids
            evidence['eventual_before_maintain'] = (
                eventual_before_maintain
            )
            evidence['eventual_pending_hits'] = eventual_pending_hits
            evidence['eventual_maintenance_phases'] = eventual_phases
            evidence['eventual_after_maintain'] = eventual_after_maintain
            evidence['eventual_hits'] = eventual_hits
            eventual_semantic_phase_index = next(
                index
                for index, phase in enumerate(eventual_phases)
                if 'mode=semantic_completion' in phase['result']
            )
            eventual_semantic_phase = eventual_phases[
                eventual_semantic_phase_index
            ]
            eventual_final_completion = semantic_completion_state(
                eventual_after_maintain
            )
            gates['eventual_lexical_first_write_is_nonblocking'] = (
                unified_generation_ready(
                    eventual_initial,
                    expected_records=2,
                )
                and int(
                    eventual_before_maintain['generation']['sealed_docs']
                ) == int(eventual_initial['generation']['sealed_docs'])
                and posting_records(eventual_before_maintain)
                    == posting_records(eventual_initial)
                and int(eventual_before_maintain['details']['rebuilds'])
                    == int(eventual_initial['details']['rebuilds'])
                and int(
                    eventual_before_maintain['generation']['delta'][
                        'records'
                    ]
                ) == 1
                and int(
                    eventual_before_maintain['details']['pending_writes']
                ) == 1
                and semantic_completion_state(
                    eventual_before_maintain
                )['pending'] == 1
                and int(eventual_runtime_after_write['requests'])
                    == int(eventual_runtime_before['requests'])
                and int(eventual_runtime_after_write['encoded_texts'])
                    == int(eventual_runtime_before['encoded_texts'])
                and eventual_pending_hits
                and eventual_pending_hits[0] == 'eventual-target'
            )
            gates['eventual_completion_precedes_compaction'] = (
                eventual_semantic_phase_index > 0
                and eventual_semantic_phase_index < len(eventual_phases) - 1
                and generation_identity(eventual_semantic_phase['status'])
                    == generation_identity(
                        eventual_phases[
                            eventual_semantic_phase_index - 1
                        ]['status']
                    )
                and int(
                    semantic_completion_state(
                        eventual_semantic_phase['status']
                    )['telemetry']['completed']
                ) >= 1
                and eventual_final_completion['converged'] is True
            )
            pending_completion = semantic_completion_state(
                eventual_before_maintain
            )
            completed_telemetry = semantic_completion_state(
                eventual_semantic_phase['status']
            )['telemetry']
            gates['eventual_completion_observability_is_bounded'] = (
                int(pending_completion['pending']) == 1
                and (
                    int(pending_completion['sealed_pending'])
                    + int(pending_completion['unsealed_upserts'])
                ) == int(pending_completion['pending'])
                and int(
                    eventual_before_maintain['generation']['delta']['bytes']
                ) > 0
                and int(
                    eventual_before_maintain['generation']['delta']['records']
                ) >= int(pending_completion['pending'])
                and bool(completed_telemetry['available'])
                and bool(completed_telemetry['observed'])
                and int(completed_telemetry['attempts']) >= 1
                and int(completed_telemetry['completed']) >= 1
                and int(completed_telemetry['failures']) == 0
                and float(
                    completed_telemetry['last_rows_per_second']
                ) >= 0.0
                and bool(eventual_final_completion['pending_exact'])
                and int(eventual_final_completion['pending']) == 0
            )
            gates['eventual_pending_hot_chain_completes_semantic'] = (
                len(set(eventual_hot_tids)) >= 2
                and int(completed_telemetry['completed']) >= 1
                and eventual_final_completion['pending'] == 0
                and eventual_hits
                and eventual_hits[0] == 'eventual-target'
            )
            gates['eventual_maintain_converges_unified_generation'] = (
                unified_generation_ready(
                    eventual_after_maintain,
                    expected_records=3,
                )
                and generation_identity(eventual_after_maintain)
                    != generation_identity(eventual_before_maintain)
                and int(
                    eventual_after_maintain['generation']['delta']['records']
                ) == 0
                and semantic_completion_state(
                    eventual_after_maintain
                )['converged'] is True
                and bool(eventual_hits)
                and eventual_hits[0] == 'eventual-target'
            )

            deferred_initial_before_baseline = eventual_deferred_status(
                connection
            )
            (
                deferred_baseline_phases,
                deferred_initial,
            ) = run_eventual_maintenance_until_converged(
                connection,
                'p2_eventual.deferred_idx',
                deferred_initial_before_baseline,
                require_generation_change=False,
            )
            deferred_guard_connection = acquire_maintenance_lock(
                socket_dir,
                args.port,
                'p2_eventual.deferred_idx',
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO p2_eventual.deferred_docs '
                    '(id, body) VALUES (%s, %s)',
                    ('deferred-target', QUERY),
                )
            deferred_pending = eventual_deferred_status(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT evoke_index_preload("
                    "'p2_eventual.deferred_idx'::regclass)"
                )
                deferred_preload_result = str(cursor.fetchone()[0])
            deferred_after_preload = eventual_deferred_status(connection)
            deferred_cache_after_preload = cache_state(
                connection,
                'p2_eventual.deferred_idx',
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO p2_eventual.deferred_docs '
                    '(id, body) VALUES (%s, %s)',
                    (
                        'deferred-background',
                        'unrelated archival background record',
                    ),
                )
            deferred_after_ingress = eventual_deferred_status(connection)
            deferred_cache_after_ingress = cache_state(
                connection,
                'p2_eventual.deferred_idx',
            )
            deferred_rows_before = eventual_deferred_query_rows(connection)
            deferred_hits_before = [
                doc_id for doc_id, _score in deferred_rows_before
            ]
            evidence['eventual_deferred_initial_before_baseline'] = (
                deferred_initial_before_baseline
            )
            evidence['eventual_deferred_baseline_phases'] = (
                deferred_baseline_phases
            )
            evidence['eventual_deferred_initial'] = deferred_initial
            evidence['eventual_deferred_pending'] = deferred_pending
            evidence['eventual_deferred_preload_result'] = (
                deferred_preload_result
            )
            evidence['eventual_deferred_after_preload'] = (
                deferred_after_preload
            )
            evidence['eventual_deferred_cache_after_preload'] = (
                deferred_cache_after_preload
            )
            evidence['eventual_deferred_after_ingress'] = (
                deferred_after_ingress
            )
            evidence['eventual_deferred_cache_after_ingress'] = (
                deferred_cache_after_ingress
            )
            evidence['eventual_deferred_hits_before_maintain'] = (
                deferred_hits_before
            )
            evidence['eventual_deferred_digest_before_maintain'] = (
                query_digest(deferred_rows_before)
            )
            gates['eventual_sae_reads_page_native_l0_before_completion'] = (
                unified_generation_ready(
                    deferred_initial,
                    expected_records=2,
                )
                and generation_identity(deferred_pending)
                == generation_identity(deferred_initial)
                and generation_identity(deferred_after_preload)
                == generation_identity(deferred_initial)
                and int(
                    deferred_after_preload['details']['pending_writes']
                ) == 1
                and int(
                    deferred_after_preload['details']['delta_records']
                ) == 1
                and int(
                    deferred_after_preload['details']['delta_bytes']
                ) > 0
                and 'storage=convergent_segments' in deferred_preload_result
                and 'tier=postgres_buffer_cache' in deferred_preload_result
                and 'pages_warmed=' in deferred_preload_result
                and deferred_hits_before
                and deferred_hits_before[0] == 'deferred-target'
            )
            gates['sae_preload_warms_checked_root_pages'] = (
                deferred_cache_after_preload['shared_preload']['resident']
                is True
                and deferred_cache_after_preload['shared_preload'][
                    'unified_warm_entries'
                ] >= 1
                and 'tier=postgres_buffer_cache' in deferred_preload_result
                and 'pages_warmed=' in deferred_preload_result
                and 'pages_warmed=0' not in deferred_preload_result
            )
            after_preload_shared = deferred_cache_after_preload[
                'shared_preload'
            ]
            after_ingress_shared = deferred_cache_after_ingress[
                'shared_preload'
            ]
            gates['sae_preload_survives_active_l0_ingress'] = (
                generation_identity(deferred_after_ingress)
                    == generation_identity(deferred_after_preload)
                and int(
                    deferred_after_ingress['details']['pending_writes']
                ) == 2
                and int(
                    deferred_after_ingress['details']['delta_records']
                ) == 2
                and after_ingress_shared['resident'] is True
                and after_ingress_shared['query_metadata_warm'] is True
                and after_ingress_shared['query_warm_marker_valid'] is True
                and after_ingress_shared['unified_warm_entries']
                    == after_preload_shared['unified_warm_entries']
                    == 1
                and after_ingress_shared['document_length_entries']
                    == after_preload_shared['document_length_entries']
                    == 1
                and after_ingress_shared['document_tid_lookup_entries']
                    == after_preload_shared['document_tid_lookup_entries']
                    == 1
            )
            deferred_shared_preload = deferred_cache_after_preload[
                'shared_preload'
            ]
            gates['shared_arena_accounts_exact_root_marker_only'] = (
                deferred_shared_preload['entries']
                    == (
                        deferred_shared_preload['unified_warm_entries']
                        + deferred_shared_preload['hot_fold_entries']
                        + deferred_shared_preload[
                            'resident_fold_entries'
                        ]
                        + deferred_shared_preload[
                            'document_length_entries'
                        ]
                        + deferred_shared_preload[
                            'document_tid_lookup_entries'
                        ]
                        + deferred_shared_preload[
                            'validated_page_entries'
                        ]
                    )
                and deferred_shared_preload['ready_entries']
                    == deferred_shared_preload['entries']
                and deferred_shared_preload['unified_warm_entries'] == 1
                and deferred_shared_preload['resident_fold_entries'] == 0
                and deferred_shared_preload['loading'] is False
                and deferred_shared_preload['used'] > 0
            )
            semantic_root_snapshot_race = timed(
                metrics,
                'semantic_root_snapshot_race_audit_ms',
                lambda: run_semantic_root_snapshot_race_audit(
                    connection,
                    socket_dir,
                    args.port,
                    deferred_rows_before,
                    deferred_guard_connection,
                    2,
                ),
            )
            evidence['semantic_root_snapshot_race_audit'] = (
                semantic_root_snapshot_race
            )
            gates['semantic_query_pins_checked_root'] = bool(
                semantic_root_snapshot_race['passed']
            )
            deferred_seal_phases, deferred_after_seal = (
                run_eventual_maintenance_until_reason(
                    deferred_guard_connection,
                    'p2_eventual.deferred_idx',
                    'pending_l0_sealed',
                )
            )
            deferred_cache_after_seal = cache_state(
                connection,
                'p2_eventual.deferred_idx',
            )
            deferred_accelerator_after_seal = deferred_after_seal[
                'generation'
            ]['semantic_accelerator']
            evidence['eventual_deferred_seal_phases'] = (
                deferred_seal_phases
            )
            evidence['eventual_deferred_after_seal'] = deferred_after_seal
            evidence['eventual_deferred_cache_after_seal'] = (
                deferred_cache_after_seal
            )
            gates['accelerator_warm_marker_survives_manifest_seal'] = (
                deferred_accelerator_after_seal['state']
                    == 'ready_baseline_delta'
                and deferred_accelerator_after_seal['baseline_current']
                    is False
                and deferred_cache_after_seal['shared_preload'][
                    'query_metadata_warm'
                ] is True
                and deferred_cache_after_seal['shared_preload'][
                    'query_warm_marker_valid'
                ] is True
                and deferred_cache_after_seal['shared_preload'][
                    'unified_warm_entries'
                ] == 1
            )
            deferred_filtered_hit_after_seal = (
                eventual_deferred_filtered_background_hit(connection)
            )
            evidence['eventual_deferred_filtered_hit_after_seal'] = (
                deferred_filtered_hit_after_seal
            )
            deferred_rows_after_seal = eventual_deferred_query_rows(
                connection
            )
            deferred_exact_rows_after_seal = (
                eventual_deferred_query_rows(connection, exact=True)
            )
            evidence['eventual_deferred_rows_after_seal'] = (
                deferred_rows_after_seal
            )
            evidence['eventual_deferred_exact_rows_after_seal'] = (
                deferred_exact_rows_after_seal
            )
            deferred_after_seal_ids = {
                doc_id for doc_id, _score in deferred_rows_after_seal
            }
            deferred_exact_after_seal_ids = {
                doc_id for doc_id, _score in deferred_exact_rows_after_seal
            }
            gates['stale_accelerator_serves_bounded_approximation'] = (
                deferred_after_seal_ids
                    <= deferred_exact_after_seal_ids
                and {
                    'deferred-target',
                    'deferred-background',
                    'deferred-base-a',
                    'deferred-base-b',
                } <= deferred_exact_after_seal_ids
                and deferred_accelerator_after_seal['refresh_due'] is False
                and deferred_accelerator_after_seal['refresh_max_age_ms']
                    is None
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT evoke_index_preload("
                    "'p2_eventual.deferred_idx'::regclass)"
                )
                deferred_seal_preload_result = str(cursor.fetchone()[0])
            deferred_cache_after_seal_preload = cache_state(
                connection,
                'p2_eventual.deferred_idx',
            )
            evidence['eventual_deferred_seal_preload_result'] = (
                deferred_seal_preload_result
            )
            evidence['eventual_deferred_cache_after_seal_preload'] = (
                deferred_cache_after_seal_preload
            )
            seal_preload_shared = deferred_cache_after_seal_preload[
                'shared_preload'
            ]
            deferred_filtered_hit_after_seal_preload = (
                eventual_deferred_filtered_background_hit(connection)
            )
            evidence[
                'eventual_deferred_filtered_hit_after_seal_preload'
            ] = deferred_filtered_hit_after_seal_preload
            deferred_rows_after_seal_preload = (
                eventual_deferred_query_rows(connection)
            )
            deferred_exact_rows_after_seal_preload = (
                eventual_deferred_query_rows(connection, exact=True)
            )
            evidence['eventual_deferred_rows_after_seal_preload'] = (
                deferred_rows_after_seal_preload
            )
            evidence[
                'eventual_deferred_exact_rows_after_seal_preload'
            ] = deferred_exact_rows_after_seal_preload
            gates[
                'accelerator_preload_preserves_serving_generation'
            ] = (
                seal_preload_shared['query_metadata_warm'] is True
                and seal_preload_shared['query_warm_marker_valid'] is True
                and seal_preload_shared['unified_warm_entries'] == 1
                and seal_preload_shared['document_length_entries'] >= 1
                and seal_preload_shared['document_tid_lookup_entries'] >= 1
                and deferred_filtered_hit_after_seal_preload
                    == deferred_filtered_hit_after_seal
                and query_digest(deferred_rows_after_seal_preload)
                    == query_digest(deferred_rows_after_seal)
                and query_digest(deferred_exact_rows_after_seal_preload)
                    == query_digest(deferred_exact_rows_after_seal)
            )
            deferred_remaining_phases, deferred_after_maintain = (
                run_eventual_maintenance_until_converged(
                    deferred_guard_connection,
                    'p2_eventual.deferred_idx',
                    deferred_initial,
                )
            )
            deferred_phases = (
                deferred_seal_phases + deferred_remaining_phases
            )
            release_maintenance_lock(
                deferred_guard_connection,
                'p2_eventual.deferred_idx',
            )
            deferred_guard_connection = None
            deferred_cache_after_maintain = cache_state(
                connection,
                'p2_eventual.deferred_idx',
            )
            deferred_exact_rows_before_repreload = (
                eventual_deferred_query_rows(connection, exact=True)
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT evoke_index_preload("
                    "'p2_eventual.deferred_idx'::regclass)"
                )
                deferred_repreload_result = str(cursor.fetchone()[0])
            deferred_cache_after_repreload = cache_state(
                connection,
                'p2_eventual.deferred_idx',
            )
            deferred_rows_after = eventual_deferred_query_rows(connection)
            deferred_exact_rows_after_repreload = (
                eventual_deferred_query_rows(connection, exact=True)
            )
            deferred_hits_after = [
                doc_id for doc_id, _score in deferred_rows_after
            ]
            evidence['eventual_deferred_maintenance_phases'] = (
                deferred_phases
            )
            evidence['eventual_deferred_after_maintain'] = (
                deferred_after_maintain
            )
            evidence['eventual_deferred_cache_after_maintain'] = (
                deferred_cache_after_maintain
            )
            evidence['eventual_deferred_repreload_result'] = (
                deferred_repreload_result
            )
            evidence['eventual_deferred_cache_after_repreload'] = (
                deferred_cache_after_repreload
            )
            evidence['eventual_deferred_hits_after_maintain'] = (
                deferred_hits_after
            )
            evidence['eventual_deferred_exact_digest_before_repreload'] = (
                query_digest(deferred_exact_rows_before_repreload)
            )
            evidence['eventual_deferred_exact_digest_after_repreload'] = (
                query_digest(deferred_exact_rows_after_repreload)
            )
            evidence['eventual_deferred_digest_after_maintain'] = (
                query_digest(deferred_rows_after)
            )
            gates['eventual_deferred_maintain_converges_generation'] = (
                generation_identity(deferred_after_maintain)
                != generation_identity(deferred_after_preload)
                and unified_generation_ready(
                    deferred_after_maintain,
                    expected_records=4,
                )
                and int(
                    deferred_after_maintain['details']['delta_records']
                ) == 0
                and int(
                    deferred_after_maintain['details']['delta_bytes']
                ) == 0
                and semantic_completion_state(
                    deferred_after_maintain
                )['converged'] is True
                and bool(deferred_hits_after)
                and deferred_hits_after[0] == 'deferred-target'
            )
            gates['sae_preload_tracks_generation_replacement'] = (
                deferred_cache_after_maintain['shared_preload']['resident']
                is False
                and deferred_cache_after_repreload['shared_preload']['resident']
                is True
                and deferred_cache_after_repreload['shared_preload'][
                    'query_metadata_warm'
                ] is True
                and deferred_cache_after_repreload['shared_preload'][
                    'query_warm_marker_valid'
                ] is True
                and deferred_cache_after_repreload['shared_preload'][
                    'document_tid_lookup_entries'
                ] >= 1
                and deferred_cache_after_repreload['shared_preload'][
                    'resident_fold_current'
                ] is False
                and deferred_cache_after_repreload['shared_preload'][
                    'resident_fold_entries'
                ] == 0
                and 'tier=postgres_buffer_cache' in deferred_repreload_result
            )
            gates['resident_fold_matches_page_native_exact_root'] = (
                query_rows_match(
                    deferred_exact_rows_before_repreload,
                    deferred_exact_rows_after_repreload,
                )
            )
            with connection.cursor() as cursor:
                cursor.execute('DROP INDEX p2_eventual.deferred_idx')
                cursor.execute('DROP INDEX p2_eventual.docs_idx')

            delta_trace = timed(
                metrics,
                'unified_delta_crud_differential_ms',
                lambda: run_unified_delta_crud_differential(
                    connection,
                    args.model_path,
                    socket_dir,
                    args.port,
                ),
            )
            evidence['unified_delta_crud_differential'] = delta_trace
            gates['unified_delta_matches_compaction_and_reindex'] = bool(
                delta_trace['passed']
            )
            with connection.cursor() as cursor:
                cursor.execute('DROP INDEX p2_delta_trace.docs_idx')

            tid_reuse_audit = timed(
                metrics,
                'tid_reuse_after_vacuum_audit_ms',
                lambda: run_tid_reuse_after_vacuum_audit(
                    connection,
                    args.model_path,
                    socket_dir,
                    args.port,
                ),
            )
            evidence['tid_reuse_after_vacuum_audit'] = tid_reuse_audit
            gates['vacuum_tombstone_prevents_tid_reuse_resurrection'] = (
                tid_reuse_audit['passed']
            )
            with connection.cursor() as cursor:
                cursor.execute('DROP INDEX p2_tid_reuse.docs_idx')

            setup_bm25(connection)
            bm25_fold_audit = timed(
                metrics,
                'bm25_page_native_seal_race_audit_ms',
                lambda: run_bm25_page_native_seal_race_audit(
                    connection,
                    socket_dir,
                    args.port,
                ),
            )
            evidence['bm25_page_native_seal_race_audit'] = bm25_fold_audit
            gates[
                'bm25_pending_seal_preserves_concurrent_active_l0_append'
            ] = bm25_fold_audit['passed']
            bm25_initial = bm25_status(connection)
            bm25_delta_lock_connection = acquire_maintenance_lock(
                socket_dir,
                args.port,
                'p2_bm25.docs_idx',
            )

            bm25_rollback_connection = connect(socket_dir, args.port)
            bm25_rollback_connection.autocommit = False
            try:
                with bm25_rollback_connection.cursor() as cursor:
                    cursor.execute(
                        'INSERT INTO p2_bm25.docs VALUES (%s, %s)',
                        ('bm25-rollback-target', QUERY),
                    )
                bm25_rollback_connection.rollback()
            finally:
                bm25_rollback_connection.close()
            bm25_after_rollback = bm25_status(connection)
            bm25_rollback_hits = bm25_query_hits(connection)
            bm25_after_rollback_query = bm25_status(connection)
            evidence['bm25_after_rollback_status'] = bm25_after_rollback
            evidence['bm25_after_rollback_query_status'] = (
                bm25_after_rollback_query
            )
            evidence['bm25_rollback_hits'] = bm25_rollback_hits
            gates['bm25_aborted_dml_is_hidden_and_converges'] = (
                generation_identity(bm25_after_rollback)
                    == generation_identity(bm25_initial)
                and bm25_generation_ready(
                    bm25_after_rollback,
                    expected_docs=2,
                )
                and 'bm25-rollback-target' not in bm25_rollback_hits
                and bm25_generation_ready(
                    bm25_after_rollback_query,
                    expected_docs=2,
                )
                and int(
                    bm25_after_rollback_query.get('details', {}).get(
                        'pending_writes',
                        -1,
                    )
                ) == int(
                    bm25_initial.get('details', {}).get(
                        'pending_writes',
                        -1,
                    )
                ) + 1
                and int(
                    bm25_after_rollback_query.get('details', {}).get(
                        'delta_records',
                        -1,
                    )
                ) == int(
                    bm25_initial.get('details', {}).get(
                        'delta_records',
                        -1,
                    )
                ) + 1
            )

            bm25_savepoint_before = bm25_after_rollback_query
            bm25_savepoint_connection = connect(socket_dir, args.port)
            bm25_savepoint_connection.autocommit = False
            try:
                with bm25_savepoint_connection.cursor() as cursor:
                    cursor.execute('SAVEPOINT evoke_bm25_write')
                    cursor.execute(
                        'INSERT INTO p2_bm25.docs VALUES (%s, %s)',
                        ('bm25-savepoint-target', QUERY),
                    )
                    cursor.execute('ROLLBACK TO SAVEPOINT evoke_bm25_write')
                    cursor.execute('RELEASE SAVEPOINT evoke_bm25_write')
                bm25_savepoint_connection.commit()
            finally:
                bm25_savepoint_connection.close()
            bm25_after_savepoint = bm25_status(connection)
            bm25_savepoint_hits = bm25_query_hits(connection)
            bm25_after_savepoint_query = bm25_status(connection)
            evidence['bm25_after_savepoint_status'] = bm25_after_savepoint
            evidence['bm25_after_savepoint_query_status'] = (
                bm25_after_savepoint_query
            )
            evidence['bm25_savepoint_hits'] = bm25_savepoint_hits
            gates['bm25_savepoint_rollback_is_hidden_and_converges'] = (
                generation_identity(bm25_after_savepoint)
                    == generation_identity(bm25_savepoint_before)
                and bm25_generation_ready(
                    bm25_after_savepoint,
                    expected_docs=2,
                )
                and 'bm25-savepoint-target' not in bm25_savepoint_hits
                and bm25_generation_ready(
                    bm25_after_savepoint_query,
                    expected_docs=2,
                )
                and int(
                    bm25_after_savepoint_query.get('details', {}).get(
                        'pending_writes',
                        -1,
                    )
                ) == int(
                    bm25_savepoint_before.get('details', {}).get(
                        'pending_writes',
                        -1,
                    )
                ) + 1
                and int(
                    bm25_after_savepoint_query.get('details', {}).get(
                        'delta_records',
                        -1,
                    )
                ) == int(
                    bm25_savepoint_before.get('details', {}).get(
                        'delta_records',
                        -1,
                    )
                ) + 1
            )

            with connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO p2_bm25.docs VALUES (%s, %s)',
                    ('bm25-target', QUERY),
                )
            bm25_after_insert = bm25_status(connection)
            bm25_hits = bm25_query_hits(connection)
            evidence['bm25_initial_status'] = bm25_initial
            evidence['bm25_after_insert_status'] = bm25_after_insert
            evidence['bm25_hits'] = bm25_hits
            gates['bm25_uses_primary_generation_without_posting_section'] = (
                bm25_generation_ready(bm25_initial, expected_docs=2)
                and bm25_generation_ready(
                    bm25_after_insert,
                    expected_docs=2,
                )
                and generation_identity(bm25_after_insert)
                    == generation_identity(bm25_initial)
                and int(
                    bm25_after_insert['details']['delta_records']
                ) == int(
                    bm25_after_savepoint_query['details']['delta_records']
                ) + 1
                and int(
                    bm25_after_insert['details']['pending_writes']
                ) == int(
                    bm25_after_savepoint_query['details']['pending_writes']
                ) + 1
                and bool(bm25_hits)
                and bm25_hits[0] == 'bm25-target'
            )

            bm25_initial_signature = bm25_initial.get('runtime_signature')
            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_bm25.docs_idx '
                    'SET (text_lowercase = false)'
                )
            bm25_text_option_pending = bm25_status(connection)
            bm25_text_option_error = ''
            try:
                bm25_query_hits(connection)
            except psycopg.Error as error:
                bm25_text_option_error = str(error)
            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_bm25.docs_idx '
                    'SET (text_lowercase = true)'
                )
            bm25_text_option_restored = bm25_status(connection)

            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_bm25.docs_idx SET (k1 = 2.0)'
                )
            bm25_scoring_option_pending = bm25_status(connection)
            bm25_scoring_option_error = ''
            try:
                bm25_query_hits(connection)
            except psycopg.Error as error:
                bm25_scoring_option_error = str(error)
            with connection.cursor() as cursor:
                cursor.execute('REINDEX INDEX p2_bm25.docs_idx')
            bm25_scoring_option_ready = bm25_status(connection)
            bm25_scoring_option_signature = (
                bm25_scoring_option_ready.get('runtime_signature')
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_bm25.docs_idx SET (k1 = 1.5)'
                )
                cursor.execute('REINDEX INDEX p2_bm25.docs_idx')
            bm25_contract_restored = bm25_status(connection)

            evidence['bm25_text_option_pending_status'] = (
                bm25_text_option_pending
            )
            evidence['bm25_text_option_pending_error'] = (
                bm25_text_option_error
            )
            evidence['bm25_text_option_restored_status'] = (
                bm25_text_option_restored
            )
            evidence['bm25_scoring_option_pending_status'] = (
                bm25_scoring_option_pending
            )
            evidence['bm25_scoring_option_pending_error'] = (
                bm25_scoring_option_error
            )
            evidence['bm25_scoring_option_ready_status'] = (
                bm25_scoring_option_ready
            )
            evidence['bm25_contract_restored_status'] = (
                bm25_contract_restored
            )
            gates['bm25_build_options_are_generation_bound'] = (
                bool(bm25_initial_signature)
                and bm25_text_option_pending.get('query_ready') is False
                and bm25_text_option_pending.get('blocker')
                    == 'runtime_generation_mismatch'
                and 'does not match the configured index options'
                    in bm25_text_option_error
                and bm25_generation_ready(
                    bm25_text_option_restored,
                    expected_docs=2,
                )
                and int(
                    bm25_text_option_restored['details']['delta_records']
                ) == 3
                and bm25_text_option_restored.get('runtime_signature')
                    == bm25_initial_signature
                and bm25_scoring_option_pending.get('query_ready') is False
                and bm25_scoring_option_pending.get('blocker')
                    == 'runtime_generation_mismatch'
                and 'does not match the configured index options'
                    in bm25_scoring_option_error
                and bm25_generation_ready(
                    bm25_scoring_option_ready,
                    expected_docs=3,
                )
                and bm25_scoring_option_signature
                    != bm25_initial_signature
                and bm25_generation_ready(
                    bm25_contract_restored,
                    expected_docs=3,
                )
                and bm25_contract_restored.get('runtime_signature')
                    == bm25_initial_signature
            )
            release_maintenance_lock(
                bm25_delta_lock_connection,
                'p2_bm25.docs_idx',
            )

            contract_initial = fetch_json(
                connection,
                "SELECT evoke_index_status("
                "'p2_bm25.contract_idx'::regclass)",
            )
            contract_lock_connection = connect(socket_dir, args.port)
            contract_lock_connection.autocommit = False
            contract_lock_acquired = False
            with contract_lock_connection.cursor() as cursor:
                cursor.execute(
                    "SELECT evoke_index_try_maintenance_lock("
                    "'p2_bm25.contract_idx'::regclass)"
                )
                contract_lock_acquired = bool(cursor.fetchone()[0])
            contract_pending: dict[str, Any] = {}
            contract_pending_error = ''
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        'ALTER INDEX p2_bm25.contract_idx '
                        'SET (text_lowercase = false)'
                    )
                    cursor.execute(
                        'INSERT INTO p2_bm25.contract_docs VALUES (%s, %s)',
                        ('contract-target', QUERY),
                    )
                contract_pending = fetch_json(
                    connection,
                    "SELECT evoke_index_status("
                    "'p2_bm25.contract_idx'::regclass)",
                )
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT * FROM evoke_query("
                            "'p2_bm25.contract_idx'::regclass, %s, 20)",
                            (QUERY,),
                        )
                except psycopg.Error as error:
                    contract_pending_error = str(error)
            finally:
                try:
                    if contract_lock_acquired:
                        with contract_lock_connection.cursor() as cursor:
                            cursor.execute(
                                "SELECT evoke_index_maintenance_unlock("
                                "'p2_bm25.contract_idx'::regclass)"
                            )
                    contract_lock_connection.commit()
                finally:
                    contract_lock_connection.close()

            with connection.cursor() as cursor:
                cursor.execute('REINDEX INDEX p2_bm25.contract_idx')
            contract_rebuild_action = 'REINDEX INDEX p2_bm25.contract_idx'
            contract_ready = fetch_json(
                connection,
                "SELECT evoke_index_status("
                "'p2_bm25.contract_idx'::regclass)",
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT source.id
                    FROM evoke_query(
                        'p2_bm25.contract_idx'::regclass,
                        %s,
                        20
                    ) AS hit
                    JOIN p2_bm25.contract_docs AS source
                      ON source.ctid = hit.ctid
                    ORDER BY hit.score DESC, source.id
                    """,
                    (QUERY,),
                )
                contract_hits = [str(row[0]) for row in cursor.fetchall()]
            evidence['bm25_eventual_contract_initial'] = contract_initial
            evidence['bm25_eventual_contract_pending'] = contract_pending
            evidence['bm25_eventual_contract_pending_error'] = (
                contract_pending_error
            )
            evidence['bm25_eventual_contract_rebuild_action'] = (
                contract_rebuild_action
            )
            evidence['bm25_eventual_contract_ready'] = contract_ready
            evidence['bm25_eventual_contract_hits'] = contract_hits
            gates['bm25_contract_drift_requires_explicit_reindex'] = (
                contract_lock_acquired
                and contract_pending.get('query_ready') is False
                and contract_pending.get('blocker')
                    == 'runtime_generation_mismatch'
                and int(
                    contract_pending.get('details', {}).get(
                        'delta_records',
                        -1,
                    )
                ) == 1
                and int(
                    contract_pending.get('details', {}).get(
                        'pending_writes',
                        -1,
                    )
                ) == 1
                and contract_pending.get('details', {}).get('stale') is False
                and 'does not match the configured index options'
                    in contract_pending_error
                and bm25_generation_ready(
                    contract_ready,
                    expected_docs=3,
                )
                and contract_ready.get('runtime_signature')
                    != contract_initial.get('runtime_signature')
                and bool(contract_hits)
                and contract_hits[0] == 'contract-target'
            )
            with connection.cursor() as cursor:
                cursor.execute('DROP INDEX p2_bm25.contract_idx')

            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_bm25.docs_idx '
                    'RESET (k1, text_lowercase)'
                )
                cursor.execute(
                    'ALTER INDEX p2_bm25.docs_idx SET ('
                    'sae = true, '
                    'consistency = eventual, '
                    f'model_path = {sql_literal(str(args.model_path))}'
                    ')'
                )
            bm25_to_sae_pending = bm25_status(connection)
            bm25_to_sae_error = ''
            try:
                bm25_query_hits(connection)
            except psycopg.Error as error:
                bm25_to_sae_error = str(error)
            with connection.cursor() as cursor:
                cursor.execute('REINDEX INDEX p2_bm25.docs_idx')
            bm25_to_sae_ready = bm25_status(connection)
            bm25_to_sae_hits = bm25_query_hits(connection)

            with connection.cursor() as cursor:
                cursor.execute(
                    'ALTER INDEX p2_bm25.docs_idx RESET (model_path)'
                )
                cursor.execute(
                    'ALTER INDEX p2_bm25.docs_idx SET ('
                    'sae = false, consistency = realtime)'
                )
            sae_to_bm25_pending = bm25_status(connection)
            sae_to_bm25_error = ''
            try:
                bm25_query_hits(connection)
            except psycopg.Error as error:
                sae_to_bm25_error = str(error)
            with connection.cursor() as cursor:
                cursor.execute('REINDEX INDEX p2_bm25.docs_idx')
            sae_to_bm25_ready = bm25_status(connection)
            sae_to_bm25_hits = bm25_query_hits(connection)

            evidence['bm25_to_sae_pending_status'] = bm25_to_sae_pending
            evidence['bm25_to_sae_pending_error'] = bm25_to_sae_error
            evidence['bm25_to_sae_ready_status'] = bm25_to_sae_ready
            evidence['bm25_to_sae_hits'] = bm25_to_sae_hits
            evidence['sae_to_bm25_pending_status'] = sae_to_bm25_pending
            evidence['sae_to_bm25_pending_error'] = sae_to_bm25_error
            evidence['sae_to_bm25_ready_status'] = sae_to_bm25_ready
            evidence['sae_to_bm25_hits'] = sae_to_bm25_hits
            gates['index_type_change_requires_atomic_reindex'] = (
                bm25_to_sae_pending.get('query_ready') is False
                and bm25_to_sae_pending['generation']['layout']['matches']
                    is False
                and bm25_to_sae_pending['generation']['layout']['physical']
                    == 'bm25'
                and bm25_to_sae_pending['generation']['layout']['requested']
                    == 'semantic'
                and bool(bm25_to_sae_error)
                and unified_generation_ready(
                    bm25_to_sae_ready,
                    expected_records=3,
                )
                and bool(bm25_to_sae_hits)
                and sae_to_bm25_pending.get('query_ready') is False
                and sae_to_bm25_pending['generation']['layout']['matches']
                    is False
                and sae_to_bm25_pending['generation']['layout']['physical']
                    == 'semantic'
                and sae_to_bm25_pending['generation']['layout']['requested']
                    == 'bm25'
                and bool(sae_to_bm25_error)
                and bm25_generation_ready(
                    sae_to_bm25_ready,
                    expected_docs=3,
                )
                and sae_to_bm25_hits == bm25_hits
            )

            if args.mixed_soak_cycles > 0:
                evidence['mixed_soak'] = run_mixed_soak(
                    connection,
                    args.mixed_soak_cycles,
                )
                gates['mixed_crud_soak_converges'] = (
                    evidence['mixed_soak']['table_rows']
                    == evidence['mixed_soak']['posting_records']
                    and evidence['mixed_soak']['query_ready']
                )

            if args.concurrent_crud_cycles > 0:
                evidence['concurrent_crud_soak'] = run_concurrent_crud_soak(
                    connection,
                    socket_dir,
                    args.port,
                    cycles=args.concurrent_crud_cycles,
                    readers=args.concurrent_readers,
                    writers=args.concurrent_writers,
                )
                gates['concurrent_crud_query_and_maintenance_converge'] = (
                    evidence['concurrent_crud_soak']['passed']
                )

            if args.soak_queries > 0:
                soak_initial = index_status(connection)
                soak_phases, soak_status = (
                    run_eventual_maintenance_until_converged(
                        connection,
                        'p2_mutable.docs_idx',
                        soak_initial,
                        max_rounds=32,
                        require_generation_change=False,
                    )
                )
                baseline = query_digest(query_hits(connection, update_query))
                soak_stable = True
                for _ in range(args.soak_queries):
                    if query_digest(
                        query_hits(connection, update_query)
                    ) != baseline:
                        soak_stable = False
                        break
                evidence['soak_queries'] = args.soak_queries
                evidence['soak_convergence_phases'] = soak_phases
                evidence['soak_converged_status'] = soak_status
                gates['query_soak_is_stable'] = soak_stable

            connection.close()
            connection = None
            timed(
                metrics,
                'cold_restart_ms',
                lambda: restart_cluster(pg_ctl, data_dir, log_path),
            )
            connection = connect(socket_dir, args.port)
            restart_status = index_status(connection)
            digest_after_restart = query_digest(
                query_hits(connection, update_query)
            )
            evidence['restart_status'] = restart_status
            evidence['digest_after_restart'] = digest_after_restart
            bm25_restart_status = bm25_status(connection)
            bm25_restart_hits = bm25_query_hits(connection)
            bm25_fold_restart_status = bm25_fold_status(connection)
            bm25_fold_restart_seed_hits = bm25_fold_query_hits(
                connection,
                'foldseedunique',
            )
            bm25_fold_restart_tail_hits = bm25_fold_query_hits(
                connection,
                'foldtailunique',
            )
            evidence['bm25_restart_status'] = bm25_restart_status
            evidence['bm25_restart_hits'] = bm25_restart_hits
            evidence['bm25_fold_restart_status'] = bm25_fold_restart_status
            evidence['bm25_fold_restart_seed_hits'] = (
                bm25_fold_restart_seed_hits
            )
            evidence['bm25_fold_restart_tail_hits'] = (
                bm25_fold_restart_tail_hits
            )
            gates['cold_restart_preserves_results'] = (
                digest_after_restart == digest_after_reindex
                and unified_generation_ready(
                    restart_status,
                    expected_records=4,
                )
            )
            gates['bm25_cold_restart_preserves_results'] = (
                bm25_generation_ready(
                    bm25_restart_status,
                    expected_docs=3,
                )
                and bm25_restart_hits == bm25_hits
            )
            gates['bm25_lightweight_fold_survives_cold_restart'] = (
                bm25_generation_ready(
                    bm25_fold_restart_status,
                    expected_docs=len(bm25_fold_audit['table_ids']),
                )
                and int(
                    bm25_fold_restart_status['details']['delta_records']
                ) == 0
                and bm25_fold_restart_seed_hits
                    == bm25_fold_audit['seed_hits']
                and bm25_fold_restart_tail_hits
                    == bm25_fold_audit['tail_hits']
            )

            connection.close()
            connection = None
            timed(
                metrics,
                'crash_restart_ms',
                lambda: crash_restart_cluster(pg_ctl, data_dir, log_path),
            )
            connection = connect(socket_dir, args.port)
            crash_status = index_status(connection)
            digest_after_crash = query_digest(
                query_hits(connection, update_query)
            )
            bm25_crash_status = bm25_status(connection)
            bm25_crash_hits = bm25_query_hits(connection)
            bm25_fold_crash_status = bm25_fold_status(connection)
            bm25_fold_crash_seed_hits = bm25_fold_query_hits(
                connection,
                'foldseedunique',
            )
            bm25_fold_crash_tail_hits = bm25_fold_query_hits(
                connection,
                'foldtailunique',
            )
            evidence['crash_status'] = crash_status
            evidence['digest_after_crash'] = digest_after_crash
            evidence['bm25_crash_status'] = bm25_crash_status
            evidence['bm25_crash_hits'] = bm25_crash_hits
            evidence['bm25_fold_crash_status'] = bm25_fold_crash_status
            evidence['bm25_fold_crash_seed_hits'] = (
                bm25_fold_crash_seed_hits
            )
            evidence['bm25_fold_crash_tail_hits'] = (
                bm25_fold_crash_tail_hits
            )
            gates['crash_restart_preserves_results'] = (
                digest_after_crash == digest_after_restart
                and unified_generation_ready(
                    crash_status,
                    expected_records=4,
                )
            )
            gates['bm25_crash_restart_preserves_results'] = (
                bm25_generation_ready(
                    bm25_crash_status,
                    expected_docs=3,
                )
                and bm25_crash_hits == bm25_restart_hits
            )
            gates['bm25_lightweight_fold_survives_crash_restart'] = (
                bm25_generation_ready(
                    bm25_fold_crash_status,
                    expected_docs=len(bm25_fold_audit['table_ids']),
                )
                and int(
                    bm25_fold_crash_status['details']['delta_records']
                ) == 0
                and bm25_fold_crash_seed_hits
                    == bm25_fold_restart_seed_hits
                and bm25_fold_crash_tail_hits
                    == bm25_fold_restart_tail_hits
            )

            bm25_fold_before_reindex = bm25_fold_crash_status
            with connection.cursor() as cursor:
                cursor.execute('REINDEX INDEX p2_bm25_fold.docs_idx')
            bm25_fold_after_reindex = bm25_fold_status(connection)
            bm25_fold_reindex_seed_hits = bm25_fold_query_hits(
                connection,
                'foldseedunique',
            )
            bm25_fold_reindex_tail_hits = bm25_fold_query_hits(
                connection,
                'foldtailunique',
            )
            evidence['bm25_fold_before_reindex'] = (
                bm25_fold_before_reindex
            )
            evidence['bm25_fold_after_reindex'] = bm25_fold_after_reindex
            evidence['bm25_fold_reindex_seed_hits'] = (
                bm25_fold_reindex_seed_hits
            )
            evidence['bm25_fold_reindex_tail_hits'] = (
                bm25_fold_reindex_tail_hits
            )
            gates['bm25_lightweight_fold_survives_full_reindex'] = (
                bm25_generation_ready(
                    bm25_fold_after_reindex,
                    expected_docs=4,
                )
                and generation_identity(bm25_fold_after_reindex)
                    != generation_identity(bm25_fold_before_reindex)
                and int(
                    bm25_fold_after_reindex['details']['delta_records']
                ) == 0
                and bm25_fold_reindex_seed_hits
                    == bm25_fold_crash_seed_hits
                and bm25_fold_reindex_tail_hits
                    == bm25_fold_crash_tail_hits
            )

            with connection.cursor() as cursor:
                cursor.execute('DROP INDEX p2_bm25_fold.docs_idx')
                cursor.execute('DROP INDEX p2_bm25.docs_idx')
                cursor.execute('DROP INDEX p2_mutable.docs_idx')
                cursor.execute(
                    """
                    SELECT to_regclass('p2_bm25_fold.docs_idx'),
                           to_regclass('p2_bm25.docs_idx'),
                           to_regclass('p2_mutable.docs_idx'),
                           count(*)
                    FROM p2_mutable.docs
                    """
                )
                (
                    dropped_fold_index,
                    dropped_bm25_index,
                    dropped_semantic_index,
                    source_rows,
                ) = cursor.fetchone()
            evidence['dropped_indexes'] = {
                'bm25_fold': dropped_fold_index,
                'bm25': dropped_bm25_index,
                'semantic': dropped_semantic_index,
            }
            evidence['relations_after_drop'] = relation_names(connection)
            gates['drop_removes_relation_and_preserves_source'] = (
                dropped_fold_index is None
                and dropped_bm25_index is None
                and dropped_semantic_index is None
                and int(source_rows) == 4
            )
        except Exception:
            if log_path.exists():
                print(
                    log_path.read_text(encoding='utf-8', errors='replace'),
                    file=sys.stderr,
                )
            raise
        finally:
            if delta_guard_connection is not None:
                try:
                    release_maintenance_lock(
                        delta_guard_connection,
                        'p2_mutable.docs_idx',
                    )
                except Exception:
                    delta_guard_connection.close()
            if eventual_guard_connection is not None:
                try:
                    release_maintenance_lock(
                        eventual_guard_connection,
                        'p2_eventual.docs_idx',
                    )
                except Exception:
                    eventual_guard_connection.close()
            if deferred_guard_connection is not None:
                try:
                    release_maintenance_lock(
                        deferred_guard_connection,
                        'p2_eventual.deferred_idx',
                    )
                except Exception:
                    deferred_guard_connection.close()
            if connection is not None:
                connection.close()
            if started:
                stop_cluster(pg_ctl, data_dir)

    return {
        'api_version': 'evoke_index_v1',
        'route': 'unified relation-owned P2 SSR index lifecycle',
        'extension_binding': {
            'mode': (
                'staged'
                if args.extension_libdir is not None
                else 'installed'
            ),
            'library_dir': (
                str(args.extension_libdir)
                if args.extension_libdir is not None
                else None
            ),
            'control_root': (
                str(args.extension_control_dir)
                if args.extension_control_dir is not None
                else None
            ),
        },
        'model_path': str(args.model_path),
        'model_id': manifest['model_id'],
        'runtime_abi': manifest['runtime_abi'],
        'gates': gates,
        'passed_gates': sum(gates.values()),
        'total_gates': len(gates),
        'all_gates_passed': all(gates.values()),
        'metrics_ms': metrics,
        'evidence': evidence,
    }


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    if args.concurrent_crud_cycles < 0:
        raise ValueError('--concurrent-crud-cycles cannot be negative')
    if args.concurrent_readers <= 0:
        raise ValueError('--concurrent-readers must be positive')
    if args.concurrent_writers <= 0:
        raise ValueError('--concurrent-writers must be positive')
    if (args.extension_libdir is None) != (
        args.extension_control_dir is None
    ):
        raise ValueError(
            '--extension-libdir and --extension-control-dir must be '
            'provided together'
        )
    if args.extension_libdir is not None:
        args.extension_libdir = args.extension_libdir.resolve()
        extension_libraries = [
            args.extension_libdir / name
            for name in ('evoke.so', 'evoke.dylib')
        ]
        if not any(path.is_file() for path in extension_libraries):
            raise FileNotFoundError(
                'evoke extension library is missing from '
                f'{args.extension_libdir}'
            )
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )
    report = run_audit(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    if not report['all_gates_passed'] and not args.allow_known_failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
