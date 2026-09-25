#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

import psycopg

from evoke_test_support import (
    create_short_socket_root,
    extension_control_root,
    vacuum_with_session_maintenance_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')
INITIAL_ROWS = (
    (1, 'database semantic retrieval baseline'),
    (2, 'lexical indexing and semantic search companion'),
    (3, 'obsolete archival semantic document'),
)
INSERT_TEXT = 'new database maintenance indexing document'
UPDATE_TEXT = 'updated semantic lifecycle maintenance query'
RACE_SOURCE_TEXT = 'semantic completion stale source version'
RACE_REPLACEMENT_TEXT = 'semantic completion current replacement version'
HOT_SOURCE_TEXT = 'semantic completion valid hot successor'
CRASH_SOURCE_TEXT = 'semantic completion crash recovery source'
SAME_TRANSACTION_TEXT = 'same transaction lexical visibility sentinel'
RECLAIM_RETIRED_RANGE_THRESHOLD = 8
RECLAIM_RETIRED_BLOCK_THRESHOLD = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate convergent SAE lexical-first CRUD and worker '
            'completion in isolated PostgreSQL 18.'
        ),
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument('--extension-libdir', type=Path, required=True)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        required=True,
        help=(
            'PostgreSQL share root containing extension/evoke.control, '
            'or the extension directory itself.'
        ),
    )
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument(
        '--semantic-alpha-mass',
        type=float,
        default=1.0,
        help='Per-index semantic alpha-mass lifecycle profile.',
    )
    parser.add_argument(
        '--semantic-impact-precision',
        choices=('f32', 'fp16', 'u8'),
        default='f32',
        help='Per-index semantic posting impact precision.',
    )
    parser.add_argument('--output', type=Path)
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


def load_manifest(model_path: Path) -> dict[str, Any]:
    manifest_path = model_path / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'model manifest does not exist: {manifest_path}'
        )
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (
        manifest.get('schema_version') != 1
        or manifest.get('api_version') != 'evoke_model_v1'
        or manifest.get('runtime_abi')
        != 'evoke_p2_unified_text_atoms_v2'
    ):
        raise ValueError('model is not a current Evoke model contract')
    return manifest


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        return int(listener.getsockname()[1])


def pg_config_value(pg_bin: Path, option: str) -> Path:
    result = run([str(pg_bin / 'pg_config'), option])
    return Path(result.stdout.strip()).resolve()


def quote_config(value: Path) -> str:
    return str(value).replace("'", "''")


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def configure_cluster(
    data_dir: Path,
    socket_dir: Path,
    port: int,
    *,
    extension_libdir: Path,
    system_libdir: Path,
    extension_control_dir: Path,
    system_sharedir: Path,
) -> None:
    config = data_dir / 'postgresql.conf'
    with config.open('a', encoding='utf-8') as handle:
        handle.write("\nshared_preload_libraries = 'evoke'\n")
        handle.write("evoke.shared_runtime_size = '256MB'\n")
        handle.write(
            "evoke.maintenance_timer_interval_ms = '3600000ms'\n"
        )
        handle.write(
            "evoke.maintenance_low_debt_interval_ms = '1000ms'\n"
        )
        handle.write("listen_addresses = ''\n")
        handle.write(
            f"unix_socket_directories = '{quote_config(socket_dir)}'\n"
        )
        handle.write(f'port = {port}\n')
        handle.write('max_worker_processes = 16\n')
        handle.write(
            "dynamic_library_path = '"
            f'{quote_config(extension_libdir)}:'
            f"{quote_config(system_libdir)}'\n"
        )
        handle.write(
            "extension_control_path = '"
            f'{quote_config(extension_control_dir)}:'
            f"{quote_config(system_sharedir)}'\n"
        )


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


def stop_cluster(pg_ctl: Path, data_dir: Path) -> None:
    run(
        [
            str(pg_ctl),
            '-D',
            str(data_dir),
            'stop',
            '-m',
            'fast',
            '-w',
        ],
        check=False,
    )


def crash_cluster(pg_ctl: Path, data_dir: Path) -> None:
    run(
        [
            str(pg_ctl),
            '-D',
            str(data_dir),
            'stop',
            '-m',
            'immediate',
            '-w',
        ],
    )


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
) -> psycopg.Connection[Any]:
    connection = connect(socket_dir, port)
    connection.autocommit = False
    deadline = time.monotonic() + 5.0
    while True:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT evoke_index_try_maintenance_lock('
                "'convergent_sae.docs_idx'::regclass)"
            )
            row = cursor.fetchone()
        if row is not None and row[0] is True:
            connection.commit()
            return connection
        if time.monotonic() >= deadline:
            connection.close()
            raise AssertionError('could not acquire maintenance test gate')
        time.sleep(0.01)


def release_maintenance_lock(
    connection: psycopg.Connection[Any],
) -> None:
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT evoke_index_maintenance_unlock('
                "'convergent_sae.docs_idx'::regclass)"
            )
        connection.commit()
    finally:
        connection.close()


def fetch_status(
    connection: psycopg.Connection[Any],
    index_name: str = 'convergent_sae.docs_idx',
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_status(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid status for {index_name}')
    return row[0]


def fetch_runtime_status(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute('SELECT evoke_runtime_service_status()')
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError('invalid runtime service status')
    return row[0]


def wait_for_accelerator_preload_state(
    connection: psycopg.Connection[Any],
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    runtime_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT evoke_index_runtime_state_json('
                "'convergent_sae.docs_idx'::regclass)"
            )
            row = cursor.fetchone()
        if row is None or not isinstance(row[0], dict):
            raise AssertionError('invalid index runtime state')
        runtime_state = row[0]
        shared_preload = runtime_state.get('shared_preload', {})
        if (
            shared_preload.get('query_metadata_warm') is True
            and shared_preload.get('resident_fold_current') is False
            and shared_preload.get('resident_fold_loading') is False
        ):
            return runtime_state
        time.sleep(0.05)
    raise AssertionError(
        'semantic accelerator preload did not publish compact query '
        f'metadata without a resident fold: {runtime_state}'
    )


def fetch_maintenance_status(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT evoke_index_runtime_state_json("
            "'convergent_sae.docs_idx'::regclass)->'maintenance'"
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError('invalid maintenance scheduler status')
    return row[0]


def wait_for_worker_limit(
    connection: psycopg.Connection[Any],
    expected: int,
) -> None:
    deadline = time.monotonic() + 5.0
    while True:
        with connection.cursor() as cursor:
            cursor.execute('SHOW evoke.maintenance_worker_limit')
            observed = int(cursor.fetchone()[0])
        if observed == expected:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                'maintenance worker limit did not reload: '
                f'expected={expected}, observed={observed}'
            )
        time.sleep(0.01)


def configure_maintenance_worker_limit(
    connection: psycopg.Connection[Any],
    value: int,
) -> None:
    if value < 0:
        raise AssertionError(f'invalid maintenance worker limit: {value}')
    with connection.cursor() as cursor:
        cursor.execute(
            f'ALTER SYSTEM SET evoke.maintenance_worker_limit = {value}'
        )
        cursor.execute('SELECT pg_reload_conf()')
    wait_for_worker_limit(connection, value)


def run_failed_touch_retry_audit(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    baseline = fetch_maintenance_status(connection)
    baseline_reconciles = int(baseline['maintenance_reconcile_count'])
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'ALTER SYSTEM SET evoke.maintenance_worker_limit = 0'
            )
            cursor.execute('SELECT pg_reload_conf()')
        wait_for_worker_limit(connection, 0)
        with connection.cursor() as cursor:
            cursor.execute('SELECT evoke_index_touch_maintenance()')
            blocked_touch = str(cursor.fetchone()[0])

        with connection.cursor() as cursor:
            cursor.execute(
                'ALTER SYSTEM SET evoke.maintenance_worker_limit = 1'
            )
            cursor.execute('SELECT pg_reload_conf()')
        wait_for_worker_limit(connection, 1)
        with connection.cursor() as cursor:
            cursor.execute('SELECT evoke_index_touch_maintenance()')
            retry_touch = str(cursor.fetchone()[0])

        deadline = time.monotonic() + 5.0
        while True:
            completed = fetch_maintenance_status(connection)
            if (
                int(completed['maintenance_reconcile_count'])
                > baseline_reconciles
            ):
                return {
                    'baseline': baseline,
                    'blocked_touch': blocked_touch,
                    'retry_touch': retry_touch,
                    'completed': completed,
                }
            if time.monotonic() >= deadline:
                raise AssertionError(
                    'failed maintenance launch consumed the retry cooldown: '
                    f'baseline={baseline}, completed={completed}'
                )
            time.sleep(0.01)
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                'ALTER SYSTEM SET evoke.maintenance_worker_limit = 1'
            )
            cursor.execute('SELECT pg_reload_conf()')
        wait_for_worker_limit(connection, 1)


def fetch_hits(
    connection: psycopg.Connection[Any],
    query: str,
    *,
    limit: int = 100,
    exact: bool = False,
) -> list[tuple[int, float]]:
    with connection.cursor() as cursor:
        if exact:
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_disable_semantic_accelerator', 'on', false)"
            )
        try:
            cursor.execute(
                """
                SELECT source.id, hit.score::float8
                FROM evoke_query(
                    'convergent_sae.docs_idx'::regclass,
                    %s,
                    %s
                ) AS hit
                JOIN convergent_sae.docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """,
                (query, limit),
            )
            return [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        finally:
            if exact:
                cursor.execute(
                    'RESET evoke.test_disable_semantic_accelerator'
                )


def encode_query_once(
    connection: psycopg.Connection[Any],
    query: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT evoke_encode_text_internal("
            "'convergent_sae.docs_idx'::regclass, %s)",
            (query,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError('query encoder returned an invalid payload')
    encoded = row[0]
    if (
        not isinstance(encoded.get('atoms'), list)
        or not isinstance(encoded.get('weights'), list)
        or not isinstance(encoded.get('runtime_signature'), str)
    ):
        raise AssertionError(f'invalid encoded query contract: {encoded}')
    encoded['atoms'] = [int(value) for value in encoded['atoms']]
    encoded['weights'] = [float(value) for value in encoded['weights']]
    return encoded


def fetch_encoded_hits(
    connection: psycopg.Connection[Any],
    encoded: dict[str, Any],
    *,
    limit: int = 100,
    filters: dict[str, Any] | None = None,
    exact: bool = False,
    require_scope: bool = False,
) -> list[tuple[int, float]]:
    with connection.cursor() as cursor:
        if exact:
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_disable_semantic_accelerator', 'on', false)"
            )
        if require_scope:
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_require_scope_filter', 'on', false)"
            )
        try:
            cursor.execute(
                """
                SELECT source.id, hit.score::float8
                FROM evoke_index_semantic_query_native_internal(
                    'convergent_sae.docs_idx'::regclass,
                    %s::int4[], %s::real[], NULL, NULL, %s,
                    %s, NULL, NULL, %s::jsonb
                ) AS hit
                JOIN convergent_sae.docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.rank
                """,
                (
                    encoded['atoms'],
                    encoded['weights'],
                    limit,
                    encoded['runtime_signature'],
                    None if filters is None else json.dumps(filters),
                ),
            )
            return [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        finally:
            if exact:
                cursor.execute(
                    'RESET evoke.test_disable_semantic_accelerator'
                )
            if require_scope:
                cursor.execute('RESET evoke.test_require_scope_filter')


def fetch_query_trace(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute('SELECT evoke_query_trace_internal()')
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError('query trace is unavailable')
    return row[0]


def fetch_filtered_hits(
    connection: psycopg.Connection[Any],
    query: str,
    allowed_ids: list[int],
    *,
    limit: int = 100,
    exact: bool = False,
    forward_route: str = 'auto',
) -> list[tuple[int, float]]:
    with connection.cursor() as cursor:
        if exact:
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_disable_semantic_accelerator', 'on', false)"
            )
        if forward_route != 'auto':
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_filtered_forward_route', %s, false)",
                (forward_route,),
            )
        try:
            cursor.execute(
                """
                WITH allowed AS MATERIALIZED (
                    SELECT array_agg(ctid ORDER BY id)::tid[] AS tids
                    FROM convergent_sae.docs
                    WHERE id = ANY (%s::int4[])
                )
                SELECT source.id, hit.score::float8
                FROM allowed
                CROSS JOIN LATERAL evoke_query(
                    'convergent_sae.docs_idx'::regclass,
                    %s,
                    allowed.tids,
                    %s
                ) AS hit
                JOIN convergent_sae.docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """,
                (allowed_ids, query, limit),
            )
            return [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        finally:
            if exact:
                cursor.execute(
                    'RESET evoke.test_disable_semantic_accelerator'
                )
            if forward_route != 'auto':
                cursor.execute('RESET evoke.test_filtered_forward_route')


def fetch_tid_filtered_hits(
    connection: psycopg.Connection[Any],
    query: str,
    allowed_tids: list[str],
    *,
    limit: int = 100,
    exact: bool = False,
) -> list[tuple[int, float]]:
    with connection.cursor() as cursor:
        if exact:
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_disable_semantic_accelerator', 'on', false)"
            )
        try:
            cursor.execute(
                """
                SELECT source.id, hit.score::float8
                FROM evoke_query(
                    'convergent_sae.docs_idx'::regclass,
                    %s,
                    %s::tid[],
                    %s
                ) AS hit
                JOIN convergent_sae.docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """,
                (query, allowed_tids, limit),
            )
            return [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        finally:
            if exact:
                cursor.execute(
                    'RESET evoke.test_disable_semantic_accelerator'
                )


def fetch_structured_filtered_hits(
    connection: psycopg.Connection[Any],
    query: str,
    filters: dict[str, Any],
    *,
    limit: int = 100,
    exact: bool = False,
    require_scope: bool = False,
) -> list[tuple[int, float]]:
    with connection.cursor() as cursor:
        if exact:
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_disable_semantic_accelerator', 'on', false)"
            )
        if require_scope:
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_require_scope_filter', 'on', false)"
            )
        try:
            cursor.execute(
                """
                SELECT source.id, hit.score::float8
                FROM evoke_query(
                    'convergent_sae.docs_idx'::regclass,
                    %s,
                    %s::jsonb,
                    %s::integer
                ) AS hit
                JOIN convergent_sae.docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """,
                (query, json.dumps(filters), limit),
            )
            return [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        finally:
            if exact:
                cursor.execute(
                    'RESET evoke.test_disable_semantic_accelerator'
                )
            if require_scope:
                cursor.execute('RESET evoke.test_require_scope_filter')


def fetch_page_native_exactness_probe(
    connection: psycopg.Connection[Any],
    query: str,
    *,
    disable_semantic_bmp: bool,
    disable_fused_semantic_taat: bool = False,
    error_budget_ratio: float = 0.0,
) -> dict[str, Any]:
    setting = 'on' if disable_semantic_bmp else 'off'
    fused_setting = 'on' if disable_fused_semantic_taat else 'off'
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_disable_semantic_accelerator', 'on', false)"
        )
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_force_semantic_bmp', 'on', false)"
        )
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_disable_semantic_bmp', %s, false)",
            (setting,),
        )
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_disable_fused_semantic_taat', %s, false)",
            (fused_setting,),
        )
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_query_semantic_error_budget_ratio', %s, false)",
            (str(error_budget_ratio),),
        )
        try:
            cursor.execute(
                """
                WITH encoded AS (
                    SELECT evoke_encode_text_internal(
                        'convergent_sae.docs_idx'::regclass,
                        %s
                    ) AS value
                ), atoms AS (
                    SELECT
                        array_agg(atom.value::int4 ORDER BY atom.ordinality)
                            AS ids,
                        array_agg(weight.value::real ORDER BY weight.ordinality)
                            AS weights
                    FROM encoded
                    CROSS JOIN LATERAL jsonb_array_elements_text(
                        encoded.value->'atoms'
                    ) WITH ORDINALITY AS atom(value, ordinality)
                    JOIN LATERAL jsonb_array_elements_text(
                        encoded.value->'weights'
                    ) WITH ORDINALITY AS weight(value, ordinality)
                      ON weight.ordinality = atom.ordinality
                ), requested AS (
                    SELECT LEAST(10, count(*))::int AS k
                    FROM convergent_sae.docs
                )
                SELECT convergent_sae.test_query_page_native_topk(
                    'convergent_sae.docs_idx'::regclass,
                    atoms.ids,
                    atoms.weights,
                    requested.k
                )
                FROM atoms, requested
                """,
                (query,),
            )
            row = cursor.fetchone()
        finally:
            cursor.execute(
                'RESET evoke.test_query_semantic_error_budget_ratio'
            )
            cursor.execute('RESET evoke.test_disable_fused_semantic_taat')
            cursor.execute('RESET evoke.test_disable_semantic_bmp')
            cursor.execute('RESET evoke.test_force_semantic_bmp')
            cursor.execute('RESET evoke.test_disable_semantic_accelerator')
    if row is None or not isinstance(row[0], dict):
        raise AssertionError('invalid semantic BMP exactness probe')
    return row[0]


def fetch_filtered_bmp_exactness_probe(
    connection: psycopg.Connection[Any],
    query: str,
    allowed_document_slots: list[int],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_disable_semantic_accelerator', 'on', false)"
        )
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_force_semantic_bmp', 'on', false)"
        )
        try:
            cursor.execute(
                """
                WITH encoded AS (
                    SELECT evoke_encode_text_internal(
                        'convergent_sae.docs_idx'::regclass,
                        %s
                    ) AS value
                ), atoms AS (
                    SELECT
                        array_agg(atom.value::int4 ORDER BY atom.ordinality)
                            AS ids,
                        array_agg(weight.value::real ORDER BY weight.ordinality)
                            AS weights
                    FROM encoded
                    CROSS JOIN LATERAL jsonb_array_elements_text(
                        encoded.value->'atoms'
                    ) WITH ORDINALITY AS atom(value, ordinality)
                    JOIN LATERAL jsonb_array_elements_text(
                        encoded.value->'weights'
                    ) WITH ORDINALITY AS weight(value, ordinality)
                      ON weight.ordinality = atom.ordinality
                )
                SELECT convergent_sae.test_query_page_native_topk_filtered(
                    'convergent_sae.docs_idx'::regclass,
                    atoms.ids,
                    atoms.weights,
                    %s,
                    true,
                    %s::int4[]
                )
                FROM atoms
                """,
                (query, len(allowed_document_slots), allowed_document_slots),
            )
            row = cursor.fetchone()
        finally:
            cursor.execute('RESET evoke.test_force_semantic_bmp')
            cursor.execute('RESET evoke.test_disable_semantic_accelerator')
    if row is None or not isinstance(row[0], dict):
        raise AssertionError('invalid filtered semantic BMP probe')
    return row[0]


def fetch_semantic_accelerator_probe(
    connection: psycopg.Connection[Any],
    query: str,
    *,
    heap_factor: float,
    bound_residual_candidates: bool = False,
    seed_bmp: bool = False,
    error_budget_ratio: float = 0.0,
    use_defaults: bool = False,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        if not use_defaults:
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_disable_semantic_accelerator', 'off', false)"
            )
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_semantic_accelerator_heap_factor', %s, false)",
                (str(heap_factor),),
            )
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_semantic_accelerator_bound_residual_candidates', "
                "%s, false)",
                ('on' if bound_residual_candidates else 'off',),
            )
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_semantic_accelerator_seed_bmp', %s, false)",
                ('on' if seed_bmp else 'off',),
            )
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_force_semantic_bmp', %s, false)",
                ('on' if seed_bmp else 'off',),
            )
            cursor.execute(
                "SELECT pg_catalog.set_config("
                "'evoke.test_query_semantic_error_budget_ratio', %s, false)",
                (str(error_budget_ratio),),
            )
        try:
            cursor.execute(
                """
                WITH encoded AS (
                    SELECT evoke_encode_text_internal(
                        'convergent_sae.docs_idx'::regclass,
                        %s
                    ) AS value
                ), atoms AS (
                    SELECT
                        array_agg(atom.value::int4 ORDER BY atom.ordinality)
                            AS ids,
                        array_agg(weight.value::real ORDER BY weight.ordinality)
                            AS weights
                    FROM encoded
                    CROSS JOIN LATERAL jsonb_array_elements_text(
                        encoded.value->'atoms'
                    ) WITH ORDINALITY AS atom(value, ordinality)
                    JOIN LATERAL jsonb_array_elements_text(
                        encoded.value->'weights'
                    ) WITH ORDINALITY AS weight(value, ordinality)
                      ON weight.ordinality = atom.ordinality
                ), requested AS (
                    SELECT LEAST(10, count(*))::int AS k
                    FROM convergent_sae.docs
                )
                SELECT convergent_sae.test_query_page_native_topk(
                    'convergent_sae.docs_idx'::regclass,
                    atoms.ids,
                    atoms.weights,
                    requested.k
                )
                FROM atoms, requested
                """,
                (query,),
            )
            row = cursor.fetchone()
        finally:
            if not use_defaults:
                cursor.execute(
                    'RESET evoke.test_query_semantic_error_budget_ratio'
                )
                cursor.execute('RESET evoke.test_force_semantic_bmp')
                cursor.execute(
                    'RESET evoke.test_semantic_accelerator_seed_bmp'
                )
                cursor.execute(
                    'RESET '
                    'evoke.test_semantic_accelerator_bound_residual_candidates'
                )
                cursor.execute(
                    'RESET evoke.test_semantic_accelerator_heap_factor'
                )
                cursor.execute(
                    'RESET evoke.test_disable_semantic_accelerator'
                )
    if row is None or not isinstance(row[0], dict):
        raise AssertionError('invalid semantic accelerator probe')
    return row[0]


def fetch_single_atom_exactness_probes(
    connection: psycopg.Connection[Any],
    query: str,
    *,
    disable_semantic_bmp: bool,
) -> list[dict[str, Any]]:
    setting = 'on' if disable_semantic_bmp else 'off'
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_disable_semantic_accelerator', 'on', false)"
        )
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_force_semantic_bmp', 'on', false)"
        )
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_disable_semantic_bmp', %s, false)",
            (setting,),
        )
        try:
            cursor.execute(
                """
                WITH encoded AS (
                    SELECT evoke_encode_text_internal(
                        'convergent_sae.docs_idx'::regclass,
                        %s
                    ) AS value
                ), atoms AS (
                    SELECT
                        atom.ordinality,
                        atom.value::int4 AS id,
                        weight.value::real AS weight
                    FROM encoded
                    CROSS JOIN LATERAL jsonb_array_elements_text(
                        encoded.value->'atoms'
                    ) WITH ORDINALITY AS atom(value, ordinality)
                    JOIN LATERAL jsonb_array_elements_text(
                        encoded.value->'weights'
                    ) WITH ORDINALITY AS weight(value, ordinality)
                      ON weight.ordinality = atom.ordinality
                ), requested AS (
                    SELECT LEAST(10, count(*))::int AS k
                    FROM convergent_sae.docs
                )
                SELECT
                    atoms.id,
                    atoms.weight,
                    convergent_sae.test_query_page_native_topk(
                        'convergent_sae.docs_idx'::regclass,
                        ARRAY[atoms.id]::int4[],
                        ARRAY[atoms.weight]::real[],
                        requested.k
                    )
                FROM atoms, requested
                ORDER BY atoms.ordinality
                """,
                (query,),
            )
            rows = cursor.fetchall()
        finally:
            cursor.execute('RESET evoke.test_disable_semantic_bmp')
            cursor.execute('RESET evoke.test_force_semantic_bmp')
            cursor.execute('RESET evoke.test_disable_semantic_accelerator')
    probes = [
        {
            'atom': int(row[0]),
            'weight': float(row[1]),
            'probe': row[2],
        }
        for row in rows
    ]
    if not probes or not all(
        isinstance(item['probe'], dict) for item in probes
    ):
        raise AssertionError('invalid single-atom exactness probes')
    return probes


def assert_single_atom_exactness_probes(
    bmp_probes: list[dict[str, Any]],
    fallback_probes: list[dict[str, Any]],
) -> None:
    if len(bmp_probes) != len(fallback_probes):
        raise AssertionError(
            'single-atom exactness route count mismatch: '
            f'{len(bmp_probes)} != {len(fallback_probes)}'
        )
    zero_fill_seen = False
    for bmp, fallback in zip(bmp_probes, fallback_probes):
        if bmp['atom'] != fallback['atom']:
            raise AssertionError(
                'single-atom exactness route order mismatch: '
                f'{bmp["atom"]} != {fallback["atom"]}'
            )
        for route in (bmp, fallback):
            probe = route['probe']
            if (
                probe['matched'] is not True
                or probe['topk_complete'] is not True
            ):
                raise AssertionError(
                    'single-atom exactness probe failed: '
                    f'{route}'
                )
            zero_fill_seen = zero_fill_seen or (
                int(probe['zero_score_documents_added']) > 0
            )
    if not zero_fill_seen:
        raise AssertionError(
            'single-atom probes did not exercise zero-score completion'
        )


def assert_page_native_exactness_probe(
    bmp_probe: dict[str, Any],
    fallback_probe: dict[str, Any],
    phase: str,
) -> None:
    if not (
        bmp_probe['matched'] is True
        and bmp_probe['semantic_bmp_attempted'] is True
        and bmp_probe['semantic_bmp_fallback'] is False
        and bmp_probe['semantic_bmp_query_path'] is True
        and int(bmp_probe['semantic_bmp_query_bytes']) > 0
        and int(bmp_probe['semantic_bmp_super_ref_reads']) > 0
        and int(bmp_probe['semantic_bmp_ref_reads']) > 0
        and fallback_probe['matched'] is True
        and fallback_probe['semantic_bmp_attempted'] is False
        and fallback_probe['semantic_bmp_fallback'] is False
        and fallback_probe['semantic_bmp_query_path'] is False
    ):
        raise AssertionError(
            f'{phase} compact and streaming scorer A/B failed: '
            f'bmp={bmp_probe}, fallback={fallback_probe}'
        )


def assert_page_native_error_budget_probe(
    probe: dict[str, Any],
    ratio: float,
) -> None:
    total_bound = float(probe['query_semantic_total_absolute_bound'])
    omitted_bound = float(probe['query_semantic_omitted_absolute_bound'])
    tolerance = max(1e-12, abs(total_bound) * 1e-12)

    if not (
        probe['topk_complete'] is True
        and int(probe['query_error_budget_pruned_term_count']) > 0
        and int(probe['query_error_budget_pruned_postings']) > 0
        and total_bound > 0.0
        and omitted_bound > 0.0
        and omitted_bound <= ratio * total_bound + tolerance
    ):
        raise AssertionError(
            'semantic error-budget contract failed: '
            f'ratio={ratio}, probe={probe}'
        )


def assert_restart_exactness_probe(
    before_bmp: dict[str, Any],
    before_fallback: dict[str, Any],
    after_bmp: dict[str, Any],
    after_fallback: dict[str, Any],
) -> None:
    probes = (before_bmp, before_fallback, after_bmp, after_fallback)
    if not all(probe['matched'] is True for probe in probes):
        raise AssertionError(
            'cold restart exact scorer oracle mismatch: '
            f'before_bmp={before_bmp}, '
            f'before_fallback={before_fallback}, '
            f'after_bmp={after_bmp}, '
            f'after_fallback={after_fallback}'
        )

    route_keys = (
        'semantic_bmp_attempted',
        'semantic_bmp_query_path',
        'ordered_block_query_path',
        'term_at_a_time_query_path',
        'materialized_query_path',
        'query_run_count',
    )
    for before, after, route in (
        (before_bmp, after_bmp, 'BMP-enabled'),
        (before_fallback, after_fallback, 'BMP-disabled'),
    ):
        changed = {
            key: (before[key], after[key])
            for key in route_keys
            if before[key] != after[key]
        }
        if changed:
            raise AssertionError(
                f'cold restart changed the {route} physical query route: '
                f'{changed}'
            )


def same_hits(
    left: list[tuple[int, float]],
    right: list[tuple[int, float]],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        left_id == right_id
        and math.isclose(
            left_score,
            right_score,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        for (left_id, left_score), (right_id, right_score)
        in zip(left, right)
    )


def same_hit_ids(
    left: list[tuple[int, float]],
    right: list[tuple[int, float]],
) -> bool:
    return [row_id for row_id, _score in left] == [
        row_id for row_id, _score in right
    ]


def score_for_id(
    rows: list[tuple[int, float]],
    document_id: int,
) -> float:
    for row_id, score in rows:
        if row_id == document_id:
            if not math.isfinite(score):
                raise AssertionError(
                    f'non-finite score for document {document_id}: {score}'
                )
            return score
    raise AssertionError(
        f'document {document_id} is absent from search rows: {rows}'
    )


def semantic_completion(status: dict[str, Any]) -> dict[str, Any]:
    completion = status['generation']['delta']['semantic_completion']
    if not isinstance(completion, dict):
        raise AssertionError(
            f'invalid semantic completion status: {completion}'
        )
    return completion


def semantic_accelerator_is_ready(status: dict[str, Any]) -> bool:
    accelerator = status['generation']['semantic_accelerator']
    return (
        accelerator['present'] is True
        and accelerator['eligible'] is True
        and accelerator['state'] == 'ready'
        and int(accelerator['source_manifest_id']) > 0
        and accelerator['forward_complete'] is True
        and int(accelerator['term_count']) > 0
        and int(accelerator['forward_chunk_count']) > 0
        and accelerator['tid_lookup_present'] is True
        and int(accelerator['tid_lookup_bytes']) > 0
    )


def fetch_scope_plateau_footprint(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_relation_size("
            "'convergent_sae.docs_idx'::regclass), pg_backend_pid()"
        )
        row = cursor.fetchone()
    if row is None:
        raise AssertionError('scope plateau footprint returned no row')

    status = fetch_status(connection)
    if not is_converged(status):
        raise AssertionError(
            f'scope plateau observed a non-converged root: {status}'
        )
    rss_result = run(
        ['ps', '-o', 'rss=', '-p', str(int(row[1]))],
        check=False,
    )
    rss_kib = int(rss_result.stdout.strip() or '0')
    primary = status['generation']['primary']
    accelerator = status['generation']['semantic_accelerator']
    return {
        'relation_bytes': int(row[0]),
        'physical_blocks': int(primary['physical_blocks']),
        'retired_hint_ranges': int(primary['retired_hint_ranges']),
        'retired_hint_blocks': int(primary['retired_hint_blocks']),
        'backend_rss_bytes': rss_kib * 1024,
        'generation': int(status['generation']['generation']),
        'source_manifest_id': int(accelerator['source_manifest_id']),
    }


def run_scope_plateau_audit(
    connection: psycopg.Connection[Any],
    *,
    cycles: int = 12,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    categories = ('scope-plateau-even', 'scope-plateau-odd')
    with connection.cursor() as cursor:
        cursor.execute('VACUUM (INDEX_CLEANUP ON) convergent_sae.docs')
    maintain_until_converged(connection, expected_docs=3)
    for cycle in range(cycles):
        category = categories[cycle % len(categories)]
        with connection.cursor() as cursor:
            cursor.execute(
                'UPDATE convergent_sae.docs '
                'SET categories = %s, source_name = %s, '
                "published_on = DATE '2024-01-01' + %s "
                'WHERE id = 2',
                ([category], f'scope-source-{cycle % 3}', cycle),
            )

        immediate_hits = fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[1][1],
            {'categories': {'overlap': [category]}},
            limit=1,
            exact=True,
        )
        score_for_id(immediate_hits, 2)
        update_phases, _update_status = maintain_until_converged(
            connection,
            expected_docs=None,
        )

        vacuum_phases: list[dict[str, Any]] = []
        for _vacuum_attempt in range(4):
            with connection.cursor() as cursor:
                cursor.execute(
                    'VACUUM (INDEX_CLEANUP ON) convergent_sae.docs'
                )
            phases, vacuum_status = maintain_until_converged(
                connection,
                expected_docs=None,
            )
            vacuum_phases.extend(phases)
            if int(vacuum_status['generation']['docs']) == 3:
                break
            time.sleep(0.05)
        else:
            raise AssertionError(
                'scope metadata retirement did not return to the fixed '
                f'live set: status={vacuum_status}'
            )
        settled_hits = fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[1][1],
            {'categories': {'overlap': [category]}},
            limit=1,
            exact=True,
            require_scope=True,
        )
        score_for_id(settled_hits, 2)
        footprint = fetch_scope_plateau_footprint(connection)
        footprint.update(
            {
                'cycle': cycle,
                'category': category,
                'update_maintenance_rounds': len(update_phases),
                'vacuum_maintenance_rounds': len(vacuum_phases),
                'maintenance_results': [
                    phase['result']
                    for phase in update_phases + vacuum_phases
                    if 'maintained=true' in phase['result']
                ],
            }
        )
        observations.append(footprint)

    middle = observations[cycles // 3 : 2 * cycles // 3]
    tail = observations[2 * cycles // 3 :]
    relation_ceiling = max(
        item['relation_bytes'] for item in middle
    ) + 2 * 1024 * 1024
    block_ceiling = max(
        item['physical_blocks'] for item in middle
    ) + 256
    rss_ceiling = max(
        item['backend_rss_bytes'] for item in middle
    ) + 16 * 1024 * 1024
    if max(item['relation_bytes'] for item in tail) > relation_ceiling:
        raise AssertionError(
            'scope metadata updates did not reach a relation-size plateau: '
            f'observations={observations}'
        )
    if max(item['physical_blocks'] for item in tail) > block_ceiling:
        raise AssertionError(
            'scope metadata updates did not reach a block-count plateau: '
            f'observations={observations}'
        )
    if max(item['backend_rss_bytes'] for item in tail) > rss_ceiling:
        raise AssertionError(
            'scope metadata updates did not reach a backend-RSS plateau: '
            f'observations={observations}'
        )
    source_manifest_ids = [
        item['source_manifest_id'] for item in observations
    ]
    if any(
        current <= previous
        for previous, current in zip(
            source_manifest_ids,
            source_manifest_ids[1:],
        )
    ):
        raise AssertionError(
            'scope metadata updates did not advance source identity: '
            f'observations={observations}'
        )
    return {
        'observations': observations,
        'relation_ceiling_bytes': relation_ceiling,
        'block_ceiling': block_ceiling,
        'backend_rss_ceiling_bytes': rss_ceiling,
    }


def is_converged(status: dict[str, Any]) -> bool:
    delta = status['generation']['delta']
    completion = semantic_completion(status)
    return (
        status['query_ready'] is True
        and status['generation']['valid'] is True
        and completion['converged'] is True
        and completion['pending_exact'] is True
        and int(completion['pending']) == 0
        and int(delta['records']) == 0
        and int(delta['active']['records']) == 0
        and int(delta['pending']['records']) == 0
        and semantic_accelerator_is_ready(status)
    )


def maintain_until_converged(
    connection: psycopg.Connection[Any],
    *,
    expected_docs: int | None,
    index_name: str = 'convergent_sae.docs_idx',
    max_rounds: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for _ in range(max_rounds):
        status = fetch_status(connection, index_name)
        if (
            is_converged(status)
            and (
                expected_docs is None
                or int(status['generation']['docs']) == expected_docs
            )
        ):
            return phases, status

        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT evoke_index_maintain(%s::regclass)',
                (index_name,),
            )
            row = cursor.fetchone()
        if row is None:
            raise AssertionError('maintenance returned no result')
        result = str(row[0])
        phases.append(
            {
                'result': result,
                'status': fetch_status(connection, index_name),
            }
        )
        if (
            'reason=lock_busy' in result
            or 'reason=xid_horizon' in result
            or 'reason=accelerator_build_busy' in result
            or 'reason=accelerator_root_checkpoint' in result
        ):
            time.sleep(0.05)
        else:
            time.sleep(0.01)

    raise AssertionError(
        'convergent SAE maintenance did not drain after '
        f'{max_rounds} rounds: {phases}'
    )


def maintain_once(
    connection: psycopg.Connection[Any],
    index_name: str = 'convergent_sae.docs_idx',
) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_maintain(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    connection.commit()
    if row is None:
        raise AssertionError('maintenance returned no result')
    return str(row[0])


def maintain_due_once(
    connection: psycopg.Connection[Any],
    index_name: str = 'convergent_sae.docs_idx',
) -> str | None:
    with connection.cursor() as cursor:
        cursor.execute('SELECT %s::regclass::oid', (index_name,))
        target_row = cursor.fetchone()
        if target_row is None:
            raise AssertionError('maintenance target returned no OID')
        target_oid = int(target_row[0])
        cursor.execute(
            'SELECT index_oid::oid, result '
            'FROM evoke_index_maintain_due(256)'
        )
        rows = cursor.fetchall()
    connection.commit()
    for index_oid, result in rows:
        if int(index_oid) == target_oid:
            return str(result)
    return None


def maintain_due_until_accelerator_published_with_semantic_debt(
    connection: psycopg.Connection[Any],
    *,
    max_rounds: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initial_status = fetch_status(connection)
    initial_source_manifest = int(
        initial_status['generation']['semantic_accelerator'][
            'source_manifest_id'
        ]
    )
    phases: list[dict[str, Any]] = []

    for _ in range(max_rounds):
        result = maintain_due_once(connection)
        status = fetch_status(connection)
        completion = semantic_completion(status)
        accelerator = status['generation']['semantic_accelerator']
        phases.append({'result': result, 'status': status})
        if (
            result is not None
            and 'reason=semantic_accelerator_published' in result
            and int(accelerator['source_manifest_id'])
                > initial_source_manifest
            and int(completion['pending']) > 0
        ):
            return phases, status
        time.sleep(0.01)

    raise AssertionError(
        'background accelerator refresh was starved by semantic debt: '
        f'{phases}'
    )


def prepare_sealed_semantic_debt(
    maintenance_connection: psycopg.Connection[Any],
    status_connection: psycopg.Connection[Any],
    *,
    source_name: str,
    timeout_seconds: float = 5.0,
) -> tuple[list[str], dict[str, Any]]:
    results: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = fetch_status(status_connection)
        delta = status['generation']['delta']
        completion = semantic_completion(status)
        if (
            int(delta['active']['records']) == 0
            and int(delta['pending']['records']) == 0
            and int(completion['sealed_pending']) == 1
            and completion['pending_exact'] is True
        ):
            return results, status

        if time.monotonic() >= deadline:
            raise AssertionError(
                f'{source_name} did not reach sealed semantic debt: '
                f'{results}, {status}'
            )

        result = maintain_once(maintenance_connection)
        results.append(result)
        if 'reason=xid_horizon' in result or 'reason=lock_busy' in result:
            time.sleep(0.05)
        else:
            time.sleep(0.01)


def configure_maintenance_budget(
    connection: psycopg.Connection[Any],
    value: str | None,
) -> str:
    with connection.cursor() as cursor:
        if value is None:
            cursor.execute(
                'ALTER SYSTEM RESET evoke.maintenance_rebuild_memory_budget'
            )
        else:
            cursor.execute(
                'ALTER SYSTEM SET evoke.maintenance_rebuild_memory_budget '
                f'= {sql_literal(value)}'
            )
        cursor.execute('SELECT pg_reload_conf()')
        if cursor.fetchone() != (True,):
            raise AssertionError('PostgreSQL did not reload maintenance budget')

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with connection.cursor() as cursor:
            cursor.execute(
                'SHOW evoke.maintenance_rebuild_memory_budget'
            )
            row = cursor.fetchone()
        if row is not None:
            current = str(row[0])
            if (value is None and current != '1MB') or current == value:
                return current
        time.sleep(0.05)
    raise AssertionError(
        'maintenance budget did not reload: '
        f'expected={value}, observed={row}'
    )


def configure_test_l0_rotation_records(
    connection: psycopg.Connection[Any],
    value: int | None,
) -> str:
    with connection.cursor() as cursor:
        if value is None:
            cursor.execute(
                'ALTER SYSTEM RESET '
                'evoke.test_convergent_l0_rotation_records'
            )
        else:
            cursor.execute(
                'ALTER SYSTEM SET '
                'evoke.test_convergent_l0_rotation_records '
                f'= {sql_literal(str(value))}'
            )
        cursor.execute('SELECT pg_reload_conf()')
        if cursor.fetchone() != (True,):
            raise AssertionError(
                'PostgreSQL did not reload the convergent-L0 test threshold'
            )
        cursor.execute('RESET evoke.test_convergent_l0_rotation_records')

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_setting("
                "'evoke.test_convergent_l0_rotation_records', true)"
            )
            row = cursor.fetchone()
        current = '' if row is None or row[0] is None else str(row[0])
        if (value is None and current == '') or current == str(value):
            return current
        time.sleep(0.05)
    raise AssertionError(
        'convergent-L0 test threshold did not reload: '
        f'expected={value}, observed={current}'
    )


def maintain_until_accelerator_budget_blocked(
    connection: psycopg.Connection[Any],
    *,
    max_rounds: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phases: list[dict[str, Any]] = []

    for _ in range(max_rounds):
        result = maintain_once(connection)
        status = fetch_status(connection)
        phases.append({'result': result, 'status': status})
        if 'reason=accelerator_memory_budget' in result:
            return phases, status
        time.sleep(0.01)
    raise AssertionError(
        'semantic accelerator did not reach its memory admission gate: '
        f'{phases}'
    )


def wait_for_extension_pause(
    connection: psycopg.Connection[Any],
    backend_pid: int,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT state, wait_event_type, wait_event, query '
                'FROM pg_stat_activity WHERE pid = %s',
                (backend_pid,),
            )
            row = cursor.fetchone()
        if row is not None:
            last = {
                'state': row[0],
                'wait_event_type': row[1],
                'wait_event': row[2],
                'query': row[3],
            }
            if (
                row[0] == 'active'
                and row[1] == 'Extension'
                and 'evoke_index_maintain' in str(row[3])
            ):
                return last
        time.sleep(0.01)
    raise AssertionError(
        'semantic maintenance did not reach publication pause: '
        f'{last}'
    )


def maintain_during_publish_pause(
    maintenance: psycopg.Connection[Any],
    observer: psycopg.Connection[Any],
    mutate: Callable[[], Any],
) -> tuple[str, dict[str, Any], Any]:
    with maintenance.cursor() as cursor:
        cursor.execute('SELECT pg_backend_pid()')
        backend_pid = int(cursor.fetchone()[0])
    maintenance.commit()
    result: dict[str, Any] = {}

    def run_maintenance() -> None:
        try:
            with maintenance.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_semantic_publish_pause_ms = '2000'"
                )
            maintenance.commit()
            result['result'] = maintain_once(maintenance)
        except Exception as error:
            maintenance.rollback()
            result['error'] = repr(error)

    thread = threading.Thread(
        target=run_maintenance,
        daemon=True,
    )
    thread.start()
    try:
        pause = wait_for_extension_pause(observer, backend_pid)
        mutation_result = mutate()
        thread.join(timeout=8.0)
        if thread.is_alive():
            raise AssertionError(
                'semantic publication race did not finish'
            )
        if result.get('error') is not None:
            raise AssertionError(
                'semantic publication race failed: '
                f"{result['error']}"
            )
        return str(result.get('result') or ''), pause, mutation_result
    finally:
        if thread.is_alive():
            thread.join(timeout=12.0)


def start_maintenance_during_post_append_pause(
    maintenance: psycopg.Connection[Any],
    observer: psycopg.Connection[Any],
) -> tuple[
    threading.Thread,
    dict[str, Any],
    dict[str, Any],
]:
    with maintenance.cursor() as cursor:
        cursor.execute('SELECT pg_backend_pid()')
        backend_pid = int(cursor.fetchone()[0])
    maintenance.commit()
    result: dict[str, Any] = {}

    def run_maintenance() -> None:
        try:
            with maintenance.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_semantic_post_append_pause_ms = '10000'"
                )
            maintenance.commit()
            result['result'] = maintain_once(maintenance)
        except Exception as error:
            try:
                maintenance.rollback()
            except Exception:
                pass
            result['error'] = repr(error)

    thread = threading.Thread(
        target=run_maintenance,
        daemon=True,
    )
    thread.start()
    pause = wait_for_extension_pause(observer, backend_pid)
    return thread, result, pause


def wait_for_background_convergence(
    connection: psycopg.Connection[Any],
    *,
    expected_docs: int,
    timeout_seconds: float = 30.0,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute('SELECT evoke_index_touch_maintenance()')
        row = cursor.fetchone()
    if row is None:
        raise AssertionError('background maintenance touch returned no result')
    touch_result = str(row[0])

    observations: list[dict[str, Any]] = []
    last_identity: tuple[int, int, int, bool] | None = None
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = fetch_status(connection)
        completion = semantic_completion(status)
        delta = status['generation']['delta']
        identity = (
            int(status['generation']['generation']),
            int(delta['records']),
            int(completion['pending']),
            bool(completion['pending_exact']),
        )
        if identity != last_identity:
            observations.append(status)
            last_identity = identity
        if (
            is_converged(status)
            and int(status['generation']['docs']) == expected_docs
        ):
            return touch_result, observations, status
        if time.monotonic() >= deadline:
            raise AssertionError(
                'background semantic maintenance did not converge: '
                f'touch={touch_result}, observations={observations}, '
                f'last_status={status}'
            )
        time.sleep(0.05)


def wait_for_background_structural_convergence(
    connection: psycopg.Connection[Any],
    before_status: dict[str, Any],
    timeout_seconds: float = 30.0,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    before_primary = before_status['generation']['primary']
    before_segments = int(before_primary['segment_count'])
    before_retired_ranges = int(before_primary['retired_hint_ranges'])
    before_retired_blocks = int(before_primary['retired_hint_blocks'])

    def reclamation_due(primary: dict[str, Any]) -> bool:
        return (
            int(primary['retired_hint_ranges'])
            >= RECLAIM_RETIRED_RANGE_THRESHOLD
            or int(primary['retired_hint_blocks'])
            >= RECLAIM_RETIRED_BLOCK_THRESHOLD
        )

    if not reclamation_due(before_primary) and is_converged(before_status):
        return 'below reclamation threshold', [before_status], before_status

    with connection.cursor() as cursor:
        cursor.execute('SELECT evoke_index_touch_maintenance()')
        row = cursor.fetchone()
    if row is None:
        raise AssertionError('structural maintenance touch returned no result')
    touch_result = str(row[0])

    observations: list[dict[str, Any]] = []
    last_identity: tuple[int, int, int, int] | None = None
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = fetch_status(connection)
        primary = status['generation']['primary']
        identity = (
            int(status['generation']['generation']),
            int(primary['segment_count']),
            int(primary['retired_hint_ranges']),
            int(primary['retired_hint_blocks']),
        )
        if identity != last_identity:
            observations.append(status)
            last_identity = identity
        if (
            is_converged(status)
            and not reclamation_due(primary)
            and (
                int(primary['segment_count']) < before_segments
                or int(primary['retired_hint_ranges'])
                < before_retired_ranges
                or int(primary['retired_hint_blocks'])
                < before_retired_blocks
            )
        ):
            return touch_result, observations, status
        if time.monotonic() >= deadline:
            maintenance_status = fetch_maintenance_status(connection)
            raise AssertionError(
                'background structural maintenance did not converge: '
                f'touch={touch_result}, observations={observations}, '
                f'maintenance={maintenance_status}, '
                f'last_status={status}'
            )
        time.sleep(0.05)


def assert_initial_status(status: dict[str, Any]) -> None:
    generation = status['generation']
    completion = semantic_completion(status)
    if not (
        status['index_type'] == 'semantic'
        and status['sae_enabled'] is True
        and status['options']['consistency'] == 'eventual'
        and generation['layout']['storage'] == 'convergent_segments'
        and generation['valid'] is True
        and generation['docs'] == len(INITIAL_ROWS)
        and generation['primary']['role'] == 'unified_posting'
        and generation['posting']['present'] is True
        and completion['enabled'] is True
        and is_converged(status)
    ):
        raise AssertionError(f'invalid initial SAE v3 status: {status}')


def assert_pending_lexical_status(status: dict[str, Any]) -> None:
    delta = status['generation']['delta']
    completion = semantic_completion(status)
    if not (
        status['query_ready'] is True
        and int(delta['records']) > 0
        and int(delta['upserts']) > 0
        and int(completion['unsealed_upserts']) > 0
        and int(completion['pending']) > 0
        and completion['pending_exact'] is False
        and completion['converged'] is False
    ):
        raise AssertionError(
            f'lexical-first semantic debt is not visible: {status}'
        )


def setup(
    connection: psycopg.Connection[Any],
    model_path: Path,
    semantic_alpha_mass: float,
    semantic_impact_precision: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute('CREATE EXTENSION evoke')
        cursor.execute('CREATE EXTENSION pg_trgm')
        cursor.execute('CREATE SCHEMA convergent_sae')
        cursor.execute(
            'CREATE FUNCTION convergent_sae.array_search_text(text[]) '
            'RETURNS text LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ '
            "SELECT array_to_string(COALESCE($1, ARRAY[]::text[]), ' ') "
            '$$'
        )
        cursor.execute(
            'CREATE TABLE convergent_sae.docs ('
            'id int PRIMARY KEY, '
            'body text NOT NULL, '
            'revision int NOT NULL DEFAULT 0, '
            "categories text[] NOT NULL DEFAULT ARRAY['general']::text[], "
            'tags text[], '
            "source_name text NOT NULL DEFAULT '', "
            "published_on date NOT NULL DEFAULT DATE '2024-01-01'"
            ') WITH (fillfactor = 50)'
        )
        cursor.executemany(
            'INSERT INTO convergent_sae.docs '
            '(id, body, categories, tags, source_name, published_on) '
            'VALUES (%s, %s, %s, %s, %s, %s)',
            (
                (
                    *INITIAL_ROWS[0],
                    ['database', 'systems'],
                    ['data', 'base', '数据库'],
                    'NVIDIA Research',
                    '2024-01-15',
                ),
                (
                    *INITIAL_ROWS[1],
                    ['retrieval'],
                    None,
                    'PostgreSQL Journal',
                    '2024-02-15',
                ),
                (
                    *INITIAL_ROWS[2],
                    ['archive', 'Kelvin'],
                    ['historical', 'archive'],
                    'Archive Proceedings',
                    '2024-03-15',
                ),
            ),
        )
        cursor.execute(
            'CREATE INDEX docs_idx ON convergent_sae.docs '
            'USING evoke (body) '
            'INCLUDE (id, categories, source_name, published_on) WITH ('
            'sae = true, '
            'auto_preload = 100, '
            'semantic_impact_precision = '
            f'{sql_literal(semantic_impact_precision)}, '
            f'semantic_alpha_mass = {semantic_alpha_mass}, '
            f'model_path = {sql_literal(str(model_path))}'
            ')'
        )
        cursor.execute(
            'CREATE INDEX docs_categories_idx '
            'ON convergent_sae.docs USING gin (categories)'
        )
        cursor.execute(
            'CREATE INDEX docs_tags_text_idx ON convergent_sae.docs '
            'USING gin (convergent_sae.array_search_text(tags) '
            'gin_trgm_ops) WHERE cardinality(tags) > 0'
        )
        cursor.execute(
            'CREATE FUNCTION convergent_sae.test_query_page_native_topk('
            'regclass, int4[], real[], int4) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_page_native_topk' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION '
            'convergent_sae.test_query_page_native_topk_filtered('
            'regclass, int4[], real[], int4, boolean, int4[]) '
            'RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_page_native_topk' "
            'LANGUAGE C STRICT'
        )

def run_owner_bm25_diagnostic_contract_audit(
    connection: psycopg.Connection[Any],
) -> dict[str, str]:
    rejected: dict[str, str] = {}
    for case_name, weight_mask in (
        ('unweighted', None),
        ('weighted', [1.0]),
    ):
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM evoke_query_tokens(
                        'convergent_sae.docs_idx'::regclass,
                        ARRAY['database']::text[],
                        10,
                        %s::real[]
                    )
                    """,
                    (weight_mask,),
                )
        except psycopg.Error as error:
            message = str(error)
            if (
                error.sqlstate != '0A000'
                or 'exact BM25 access is unavailable' not in message
                or 'Use evoke_query' not in message
            ):
                raise AssertionError(
                    f'unexpected {case_name} SAE diagnostic error: '
                    f'{message}'
                ) from error
            rejected[case_name] = message
        else:
            raise AssertionError(
                f'{case_name} exact BM25 diagnostic accepted an SAE index'
            )
    return rejected


def run_consistency_contract_audit(
    connection: psycopg.Connection[Any],
    model_path: Path,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE SCHEMA convergent_contract; '
            'CREATE TABLE convergent_contract.docs ('
            'id int PRIMARY KEY, body text NOT NULL)'
        )
        cursor.execute(
            'CREATE INDEX sae_eventual_idx '
            'ON convergent_contract.docs USING evoke (body) WITH ('
            'sae = true, '
            f'model_path = {sql_literal(str(model_path))}, '
            'consistency = eventual)'
        )
        cursor.execute(
            'CREATE INDEX bm25_default_idx '
            'ON convergent_contract.docs USING evoke (body)'
        )
        cursor.execute(
            'CREATE INDEX bm25_realtime_idx '
            'ON convergent_contract.docs USING evoke (body) '
            'WITH (consistency = realtime)'
        )

    rejected: dict[str, str] = {}
    for consistency in ('realtime', 'manual'):
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f'CREATE INDEX sae_{consistency}_idx '
                    'ON convergent_contract.docs USING evoke (body) WITH ('
                    'sae = true, '
                    f'model_path = {sql_literal(str(model_path))}, '
                    f'consistency = {consistency})'
                )
        except psycopg.Error as error:
            message = str(error)
            if (
                error.sqlstate != '22023'
                or "SAE indexes require consistency = 'eventual'"
                not in message
            ):
                raise AssertionError(
                    f'unexpected SAE {consistency} error: {message}'
                ) from error
            rejected[consistency] = message
        else:
            raise AssertionError(
                f'SAE consistency={consistency} was not rejected'
            )

    statuses = {
        name: fetch_status(
            connection,
            f'convergent_contract.{name}',
        )
        for name in (
            'sae_eventual_idx',
            'bm25_default_idx',
            'bm25_realtime_idx',
        )
    }
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM pg_catalog.pg_class "
            "WHERE relnamespace = 'convergent_contract'::regnamespace "
            "AND relname IN ('sae_realtime_idx', 'sae_manual_idx')"
        )
        rejected_index_count = int(cursor.fetchone()[0])
        cursor.execute('DROP SCHEMA convergent_contract CASCADE')

    consistency = {
        name: status['options']['consistency']
        for name, status in statuses.items()
    }
    passed = (
        consistency['sae_eventual_idx'] == 'eventual'
        and consistency['bm25_default_idx'] == 'realtime'
        and consistency['bm25_realtime_idx'] == 'realtime'
        and set(rejected) == {'realtime', 'manual'}
        and rejected_index_count == 0
    )
    if not passed:
        raise AssertionError(
            'SAE/BM25 consistency contract mismatch: '
            f'{consistency}, rejected={rejected}, '
            f'residue={rejected_index_count}'
        )
    return {
        'consistency': consistency,
        'rejected': rejected,
        'rejected_index_count': rejected_index_count,
    }


def run_multicolumn_sae_contract_audit(
    connection: psycopg.Connection[Any],
    model_path: Path,
    semantic_alpha_mass: float,
    semantic_impact_precision: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_catalog.set_config("
            "'evoke.test_disable_semantic_accelerator', 'on', false)"
        )
        cursor.execute(
            'CREATE SCHEMA convergent_multicol; '
            'CREATE TABLE convergent_multicol.docs ('
            'id int PRIMARY KEY, '
            'title text NOT NULL, '
            'body text NOT NULL)'
        )
        cursor.executemany(
            'INSERT INTO convergent_multicol.docs '
            '(id, title, body) VALUES (%s, %s, %s)',
            (
                (1, 'database search', 'reliable index maintenance'),
                (2, 'semantic retrieval', 'unified sparse postings'),
            ),
        )
        cursor.execute(
            'CREATE INDEX docs_idx ON convergent_multicol.docs '
            'USING evoke (title, body) WITH ('
            'sae = true, '
            'semantic_impact_precision = '
            f'{sql_literal(semantic_impact_precision)}, '
            f'semantic_alpha_mass = {semantic_alpha_mass}, '
            f'model_path = {sql_literal(str(model_path))}'
            ')'
        )

    status = fetch_status(
        connection,
        'convergent_multicol.docs_idx',
    )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_idx'::regclass,
                'database maintenance',
                10
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]

    score_for_id(hits, 1)
    if not (
        status['index_type'] == 'semantic'
        and status['sae_enabled'] is True
        and status['options']['consistency'] == 'eventual'
        and status['query_ready'] is True
    ):
        raise AssertionError(
            f'invalid multicolumn SAE status: {status}'
        )

    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE INDEX docs_field_idx '
            'ON convergent_multicol.docs USING evoke (title, body) '
            'WITH ('
            'sae = true, '
            'field_aware = true, '
            'semantic_impact_precision = '
            f'{sql_literal(semantic_impact_precision)}, '
            f'semantic_alpha_mass = {semantic_alpha_mass}, '
            f'model_path = {sql_literal(str(model_path))}'
            ')'
        )
    field_status = fetch_status(
        connection,
        'convergent_multicol.docs_field_idx',
    )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'database maintenance',
                10
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        field_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
    score_for_id(field_hits, 1)
    weighted_field_hits: dict[str, list[tuple[int, float]]] = {}
    field_cases = {
        'equal': (['title', 'body'], [1.0, 1.0]),
        'title': (['title'], [1.0]),
        'body': (['body'], [1.0]),
        'weighted': (['title', 'body'], [2.0, 3.0]),
    }
    with connection.cursor() as cursor:
        for case_name, (field_names, field_weights) in field_cases.items():
            cursor.execute(
                """
                SELECT source.id, hit.score::float8
                FROM evoke_query(
                    'convergent_multicol.docs_field_idx'::regclass,
                    'database maintenance',
                    %s::text[],
                    %s::real[],
                    10
                ) AS hit
                JOIN convergent_multicol.docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """,
                (field_names, field_weights),
            )
            weighted_field_hits[case_name] = [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
    if not same_hits(field_hits, weighted_field_hits['equal']):
        raise AssertionError(
            'default field-aware SAE scores differ from explicit equal '
            f'weights: {field_hits} != {weighted_field_hits["equal"]}'
        )
    score_maps = {
        case_name: dict(rows)
        for case_name, rows in weighted_field_hits.items()
    }
    with connection.cursor() as cursor:
        cursor.execute(
            """
            WITH allowed AS MATERIALIZED (
                SELECT array_agg(ctid)::tid[] AS tids
                FROM convergent_multicol.docs
                WHERE id = 2
            )
            SELECT source.id, hit.score::float8
            FROM allowed
            CROSS JOIN LATERAL evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'database maintenance',
                ARRAY['title', 'body']::text[],
                ARRAY[2.0, 3.0]::real[],
                allowed.tids,
                1
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        filtered_field_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
    expected_filtered_field_hits = [
        hit for hit in weighted_field_hits['weighted'] if hit[0] == 2
    ]
    if not same_hits(
        expected_filtered_field_hits,
        filtered_field_hits,
    ):
        raise AssertionError(
            'field-aware TID-filtered top-k changed weighted scores: '
            f'{expected_filtered_field_hits} != {filtered_field_hits}'
        )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'database maintenance',
                ARRAY['title', 'body']::text[],
                ARRAY[2.0, 3.0]::real[],
                '{"id":{"eq":2}}'::jsonb,
                1
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        structured_field_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
    if not same_hits(
        expected_filtered_field_hits,
        structured_field_hits,
    ):
        raise AssertionError(
            'field-aware structured filter changed weighted scores: '
            f'{expected_filtered_field_hits} != {structured_field_hits}'
        )
    for document_id in set(score_maps['weighted']):
        expected_score = (
            2.0 * score_maps['title'].get(document_id, 0.0)
            + 3.0 * score_maps['body'].get(document_id, 0.0)
        )
        if not math.isclose(
            score_maps['weighted'][document_id],
            expected_score,
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise AssertionError(
                'field-aware SAE score is not linearly field weighted: '
                f'doc={document_id} actual='
                f'{score_maps["weighted"][document_id]} '
                f'expected={expected_score}'
            )
    with connection.cursor() as cursor:
        for field_names, field_weights, expected_error in (
            (['title'], [-1.0], 'finite and non-negative'),
            (['title', 'title'], [1.0, 1.0], 'duplicates'),
        ):
            try:
                cursor.execute(
                    """
                    SELECT *
                    FROM evoke_query(
                        'convergent_multicol.docs_field_idx'::regclass,
                        'database maintenance',
                        %s::text[],
                        %s::real[],
                        10
                    )
                    """,
                    (field_names, field_weights),
                )
            except psycopg.Error as error:
                if expected_error not in str(error):
                    raise AssertionError(
                        f'unexpected field validation error: {error}'
                    ) from error
            else:
                raise AssertionError(
                    f'invalid field specification was accepted: '
                    f'{field_names}, {field_weights}'
                )
    with connection.cursor() as cursor:
        cursor.execute('REINDEX INDEX convergent_multicol.docs_field_idx')
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'database maintenance',
                10
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        field_reindexed_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
        cursor.execute(
            'INSERT INTO convergent_multicol.docs '
            '(id, title, body) VALUES '
            "(3, 'transaction visibility', 'semantic maintenance')"
        )
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'transaction visibility',
                10
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        field_insert_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
    field_insert_phases, field_insert_status = maintain_until_converged(
        connection,
        expected_docs=3,
        index_name='convergent_multicol.docs_field_idx',
    )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'transaction visibility',
                ARRAY['title', 'body']::text[],
                ARRAY[4.0, 0.5]::real[],
                10
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        field_completed_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
        cursor.execute(
            'UPDATE convergent_multicol.docs '
            "SET title = '', "
            "body = 'transaction visibility semantic maintenance' "
            'WHERE id = 3'
        )
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'transaction visibility',
                ARRAY['body']::text[],
                ARRAY[1.0]::real[],
                10
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        field_update_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
    field_update_phases, field_update_status = maintain_until_converged(
        connection,
        expected_docs=None,
        index_name='convergent_multicol.docs_field_idx',
    )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'transaction visibility',
                ARRAY['title', 'body']::text[],
                ARRAY[4.0, 0.5]::real[],
                10
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        field_updated_completed_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
        field_updated_completed_components = {}
        for field_name in ('title', 'body'):
            cursor.execute(
                """
                SELECT source.id, hit.score::float8
                FROM evoke_query(
                    'convergent_multicol.docs_field_idx'::regclass,
                    'transaction visibility',
                    ARRAY[%s]::text[],
                    ARRAY[1.0]::real[],
                    10
                ) AS hit
                JOIN convergent_multicol.docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """,
                (field_name,),
            )
            field_updated_completed_components[field_name] = [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        cursor.execute('VACUUM convergent_multicol.docs')
    field_update_vacuum_phases, field_update_vacuum_status = (
        maintain_until_converged(
            connection,
            expected_docs=3,
            index_name='convergent_multicol.docs_field_idx',
        )
    )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'transaction visibility',
                ARRAY['title', 'body']::text[],
                ARRAY[4.0, 0.5]::real[],
                10
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        field_updated_settled_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
        field_updated_settled_components = {}
        for field_name in ('title', 'body'):
            cursor.execute(
                """
                SELECT source.id, hit.score::float8
                FROM evoke_query(
                    'convergent_multicol.docs_field_idx'::regclass,
                    'transaction visibility',
                    ARRAY[%s]::text[],
                    ARRAY[1.0]::real[],
                    10
                ) AS hit
                JOIN convergent_multicol.docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """,
                (field_name,),
            )
            field_updated_settled_components[field_name] = [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        cursor.execute('REINDEX INDEX convergent_multicol.docs_field_idx')
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'transaction visibility',
                ARRAY['title', 'body']::text[],
                ARRAY[4.0, 0.5]::real[],
                10
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        field_updated_reindexed_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
        field_updated_reindexed_components = {}
        for field_name in ('title', 'body'):
            cursor.execute(
                """
                SELECT source.id, hit.score::float8
                FROM evoke_query(
                    'convergent_multicol.docs_field_idx'::regclass,
                    'transaction visibility',
                    ARRAY[%s]::text[],
                    ARRAY[1.0]::real[],
                    10
                ) AS hit
                JOIN convergent_multicol.docs AS source
                  ON source.ctid = hit.ctid
                ORDER BY hit.score DESC, source.id
                """,
                (field_name,),
            )
            field_updated_reindexed_components[field_name] = [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        cursor.execute(
            'DELETE FROM convergent_multicol.docs WHERE id = 3'
        )
        cursor.execute(
            """
            SELECT source.id, hit.score::float8
            FROM evoke_query(
                'convergent_multicol.docs_field_idx'::regclass,
                'transaction visibility',
                10
            ) AS hit
            JOIN convergent_multicol.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """
        )
        field_deleted_hits = [
            (int(row[0]), float(row[1]))
            for row in cursor.fetchall()
        ]
    if not same_hits(field_hits, field_reindexed_hits):
        raise AssertionError(
            'field-aware SAE REINDEX changed rows or scores: '
            f'{field_hits} != {field_reindexed_hits}'
        )
    score_for_id(field_insert_hits, 3)
    score_for_id(field_completed_hits, 3)
    score_for_id(field_update_hits, 3)
    score_for_id(field_updated_completed_hits, 3)
    if not same_hits(
        field_updated_settled_hits,
        field_updated_reindexed_hits,
    ):
        raise AssertionError(
            'field-aware SAE settled UPDATE and REINDEX diverged: '
            f'{field_updated_settled_hits} != '
            f'{field_updated_reindexed_hits}; components '
            f'{field_updated_settled_components} != '
            f'{field_updated_reindexed_components}'
        )
    if any(row_id == 3 for row_id, _score in field_deleted_hits):
        raise AssertionError(
            'deleted field-aware SAE row remained query-visible'
        )
    if not (
        field_status['index_type'] == 'semantic'
        and field_status['sae_enabled'] is True
        and field_status['options']['field_aware'] is True
        and field_status['query_ready'] is True
    ):
        raise AssertionError(
            f'invalid field-aware SAE status: {field_status}'
        )

    with connection.cursor() as cursor:
        cursor.execute('RESET evoke.test_disable_semantic_accelerator')
        cursor.execute('DROP SCHEMA convergent_multicol CASCADE')

    return {
        'status': status,
        'hits': hits,
        'field_status': field_status,
        'field_hits': field_hits,
        'weighted_field_hits': weighted_field_hits,
        'structured_field_hits': structured_field_hits,
        'field_reindexed_hits': field_reindexed_hits,
        'field_insert_hits': field_insert_hits,
        'field_insert_phases': field_insert_phases,
        'field_insert_status': field_insert_status,
        'field_completed_hits': field_completed_hits,
        'field_update_hits': field_update_hits,
        'field_update_phases': field_update_phases,
        'field_update_status': field_update_status,
        'field_updated_completed_hits': field_updated_completed_hits,
        'field_updated_completed_components': (
            field_updated_completed_components
        ),
        'field_update_vacuum_phases': field_update_vacuum_phases,
        'field_update_vacuum_status': field_update_vacuum_status,
        'field_updated_settled_hits': field_updated_settled_hits,
        'field_updated_settled_components': (
            field_updated_settled_components
        ),
        'field_updated_reindexed_hits': field_updated_reindexed_hits,
        'field_updated_reindexed_components': (
            field_updated_reindexed_components
        ),
        'field_deleted_hits': field_deleted_hits,
    }


def run_fold_compaction_preload_audit(
    connection: psycopg.Connection[Any],
    model_path: Path,
    semantic_alpha_mass: float,
    semantic_impact_precision: str,
) -> dict[str, Any]:
    index_name = 'convergent_fold_compaction.docs_idx'
    maintenance: list[str] = []
    structural_folds: list[str] = []
    compactions: list[str] = []
    preload_results: list[str] = []
    lock_held = False

    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE SCHEMA convergent_fold_compaction; '
            'CREATE TABLE convergent_fold_compaction.docs ('
            'id int PRIMARY KEY, body text NOT NULL)'
        )
        cursor.executemany(
            'INSERT INTO convergent_fold_compaction.docs '
            'VALUES (%s, %s)',
            INITIAL_ROWS,
        )
        cursor.execute(
            'CREATE INDEX docs_idx '
            'ON convergent_fold_compaction.docs USING evoke (body) '
            'WITH ('
            'sae = true, '
            'semantic_impact_precision = '
            f'{sql_literal(semantic_impact_precision)}, '
            f'semantic_alpha_mass = {semantic_alpha_mass}, '
            f'model_path = {sql_literal(str(model_path))}'
            ')'
        )
        cursor.execute(
            'SELECT evoke_index_try_maintenance_lock(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
        lock_held = row is not None and row[0] is True
        if not lock_held:
            raise AssertionError(
                'could not lock the fold-compaction regression index'
            )
        cursor.execute('SET evoke.test_force_structural_term_fold = true')

    try:
        for row_offset in range(9):
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_convergent_l0_rotation_records = '1'"
                )
                cursor.execute(
                    'INSERT INTO convergent_fold_compaction.docs '
                    'VALUES (%s, %s)',
                    (
                        100 + row_offset,
                        'database semantic fold shared term '
                        f'{row_offset}',
                    ),
                )
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )

            for _ in range(20):
                result = maintain_once(connection, index_name)
                maintenance.append(result)
                if 'mode=term_structural_fold' in result:
                    structural_folds.append(result)
                status = fetch_status(connection, index_name)
                delta = status['generation']['delta']
                if (
                    int(delta['active']['records']) == 0
                    and int(delta['pending']['records']) == 0
                    and int(
                        delta['semantic_completion']['pending']
                    )
                    == 0
                    and (
                        'maintained=false' in result
                        or structural_folds
                    )
                ):
                    break
            else:
                raise AssertionError(
                    'fold-compaction setup did not drain: '
                    f'{maintenance[-20:]}'
                )

        if not structural_folds:
            raise AssertionError(
                'fold-compaction setup produced no structural fold'
            )
        with connection.cursor() as cursor:
            cursor.execute('RESET evoke.test_force_structural_term_fold')

        for _ in range(48):
            result = maintain_once(connection, index_name)
            maintenance.append(result)
            if 'mode=segment_compaction' in result:
                compactions.append(result)
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT evoke_index_preload(%s::regclass)',
                    (index_name,),
                )
                preload_row = cursor.fetchone()
                if preload_row is None:
                    raise AssertionError(
                        'fold-compaction preload returned no result'
                    )
                preload_results.append(str(preload_row[0]))
                cursor.execute(
                    'SELECT count(*) FROM evoke_query('
                    '%s::regclass, %s, 20)',
                    (index_name, 'database semantic fold'),
                )
                hit_count = int(cursor.fetchone()[0])
            if hit_count != 12:
                raise AssertionError(
                    'fold-compaction query lost documents: '
                    f'count={hit_count}, maintenance={result}'
                )
            if 'maintained=false' in result:
                break
        else:
            raise AssertionError(
                'fold-compaction maintenance did not become idle'
            )

        if len(compactions) < 3:
            raise AssertionError(
                'fold-compaction regression did not cross enough '
                f'boundaries: {compactions}'
            )
        return {
            'structural_fold_count': len(structural_folds),
            'compactions': compactions,
            'preload_count': len(preload_results),
            'final_status': fetch_status(connection, index_name),
        }
    finally:
        with connection.cursor() as cursor:
            cursor.execute('RESET evoke.test_force_structural_term_fold')
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
            if lock_held:
                cursor.execute(
                    'SELECT evoke_index_maintenance_unlock('
                    '%s::regclass)',
                    (index_name,),
                )
            cursor.execute(
                'DROP SCHEMA IF EXISTS '
                'convergent_fold_compaction CASCADE'
            )


def run_semantic_alpha_contract_audit(
    connection: psycopg.Connection[Any],
    model_path: Path,
    semantic_alpha_mass: float,
    semantic_impact_precision: str,
) -> dict[str, Any]:
    alternate_alpha = 0.75 if semantic_alpha_mass != 0.75 else 0.50

    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE INDEX docs_alpha_contract_idx '
            'ON convergent_sae.docs USING evoke (body) WITH ('
            'sae = true, '
            'semantic_impact_precision = '
            f'{sql_literal(semantic_impact_precision)}, '
            f'semantic_alpha_mass = {semantic_alpha_mass}, '
            f'model_path = {sql_literal(str(model_path))}'
            ')'
        )
    initial_status = fetch_status(
        connection,
        'convergent_sae.docs_alpha_contract_idx',
    )
    if not initial_status['query_ready']:
        raise AssertionError(
            f'alpha contract index is not query-ready: {initial_status}'
        )

    with connection.cursor() as cursor:
        cursor.execute(
            'ALTER INDEX convergent_sae.docs_alpha_contract_idx SET '
            f'(semantic_alpha_mass = {alternate_alpha})'
        )
    altered_status = fetch_status(
        connection,
        'convergent_sae.docs_alpha_contract_idx',
    )
    if altered_status['query_ready']:
        raise AssertionError(
            'changed semantic alpha mass accepted the prior generation'
        )

    query_error = None
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM evoke_query("
                "'convergent_sae.docs_alpha_contract_idx'::regclass, "
                "'database search', 2)"
            )
    except psycopg.Error as error:
        query_error = str(error)
    if query_error is None or 'does not match' not in query_error:
        raise AssertionError(
            f'alpha contract mismatch did not fail closed: {query_error}'
        )

    with connection.cursor() as cursor:
        cursor.execute(
            'REINDEX INDEX convergent_sae.docs_alpha_contract_idx'
        )
    rebuilt_status = fetch_status(
        connection,
        'convergent_sae.docs_alpha_contract_idx',
    )
    if not (
        rebuilt_status['query_ready']
        and rebuilt_status['options']['semantic_alpha_mass']
        == alternate_alpha
    ):
        raise AssertionError(
            f'alpha contract did not recover after REINDEX: {rebuilt_status}'
        )

    with connection.cursor() as cursor:
        cursor.execute(
            'DROP INDEX convergent_sae.docs_alpha_contract_idx'
        )
    return {
        'initial_status': initial_status,
        'altered_status': altered_status,
        'query_error': query_error,
        'rebuilt_status': rebuilt_status,
    }


def run_semantic_impact_precision_contract_audit(
    connection: psycopg.Connection[Any],
    model_path: Path,
    semantic_impact_precision: str,
    semantic_alpha_mass: float,
) -> dict[str, Any]:
    alternate_precision = (
        'fp16' if semantic_impact_precision == 'u8' else 'u8'
    )

    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE INDEX docs_precision_contract_idx '
            'ON convergent_sae.docs USING evoke (body) WITH ('
            'sae = true, '
            'semantic_impact_precision = '
            f'{sql_literal(semantic_impact_precision)}, '
            f'semantic_alpha_mass = {semantic_alpha_mass}, '
            f'model_path = {sql_literal(str(model_path))}'
            ')'
        )
    initial_status = fetch_status(
        connection,
        'convergent_sae.docs_precision_contract_idx',
    )
    if not (
        initial_status['query_ready']
        and initial_status['options']['semantic_impact_precision']
        == semantic_impact_precision
    ):
        raise AssertionError(
            'impact precision contract index is not query-ready: '
            f'{initial_status}'
        )

    with connection.cursor() as cursor:
        cursor.execute(
            'ALTER INDEX convergent_sae.docs_precision_contract_idx SET '
            '(semantic_impact_precision = '
            f'{sql_literal(alternate_precision)})'
        )
    altered_status = fetch_status(
        connection,
        'convergent_sae.docs_precision_contract_idx',
    )
    if altered_status['query_ready']:
        raise AssertionError(
            'changed semantic impact precision accepted the prior generation'
        )

    query_error = None
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM evoke_query("
                "'convergent_sae.docs_precision_contract_idx'::regclass, "
                "'database search', 2)"
            )
    except psycopg.Error as error:
        query_error = str(error)
    if query_error is None or 'does not match' not in query_error:
        raise AssertionError(
            'impact precision mismatch did not fail closed: '
            f'{query_error}'
        )

    with connection.cursor() as cursor:
        cursor.execute(
            'REINDEX INDEX convergent_sae.docs_precision_contract_idx'
        )
    rebuilt_status = fetch_status(
        connection,
        'convergent_sae.docs_precision_contract_idx',
    )
    if not (
        rebuilt_status['query_ready']
        and rebuilt_status['options']['semantic_impact_precision']
        == alternate_precision
    ):
        raise AssertionError(
            'impact precision contract did not recover after REINDEX: '
            f'{rebuilt_status}'
        )

    with connection.cursor() as cursor:
        cursor.execute(
            'DROP INDEX convergent_sae.docs_precision_contract_idx'
        )
    return {
        'initial_status': initial_status,
        'altered_status': altered_status,
        'query_error': query_error,
        'rebuilt_status': rebuilt_status,
    }


def run_same_transaction_audit(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    connection.autocommit = False
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO convergent_sae.docs '
                '(id, body, categories) VALUES (900, %s, %s)',
                (SAME_TRANSACTION_TEXT, ['transaction-visible']),
            )
        visible_hits = fetch_hits(connection, SAME_TRANSACTION_TEXT)
        score_for_id(visible_hits, 900)
        visible_filtered_hits = fetch_structured_filtered_hits(
            connection,
            SAME_TRANSACTION_TEXT,
            {'categories': {'overlap': ['transaction-visible']}},
            limit=1,
            exact=True,
        )
        score_for_id(visible_filtered_hits, 900)

        with connection.cursor() as cursor:
            cursor.execute('SAVEPOINT evoke_filter_scope_savepoint')
            cursor.execute(
                'UPDATE convergent_sae.docs '
                'SET categories = %s WHERE id = 900',
                (['savepoint-visible'],),
            )
        savepoint_hits = fetch_structured_filtered_hits(
            connection,
            SAME_TRANSACTION_TEXT,
            {'categories': {'overlap': ['savepoint-visible']}},
            limit=1,
            exact=True,
        )
        score_for_id(savepoint_hits, 900)
        with connection.cursor() as cursor:
            cursor.execute('ROLLBACK TO SAVEPOINT evoke_filter_scope_savepoint')
            cursor.execute('RELEASE SAVEPOINT evoke_filter_scope_savepoint')
        restored_hits = fetch_structured_filtered_hits(
            connection,
            SAME_TRANSACTION_TEXT,
            {'categories': {'overlap': ['transaction-visible']}},
            limit=1,
            exact=True,
        )
        score_for_id(restored_hits, 900)
        rolled_back_savepoint_hits = fetch_structured_filtered_hits(
            connection,
            SAME_TRANSACTION_TEXT,
            {'categories': {'overlap': ['savepoint-visible']}},
            limit=1,
            exact=True,
        )
        if rolled_back_savepoint_hits:
            raise AssertionError(
                'rolled-back savepoint metadata remained query-visible'
            )
        connection.rollback()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.autocommit = True

    rolled_back_hits = fetch_hits(connection, SAME_TRANSACTION_TEXT)
    rolled_back_filtered_hits = fetch_structured_filtered_hits(
        connection,
        SAME_TRANSACTION_TEXT,
        {'categories': {'overlap': ['transaction-visible']}},
        limit=1,
        exact=True,
    )
    if any(row_id == 900 for row_id, _score in rolled_back_hits):
        raise AssertionError(
            'rolled-back same-transaction row remained query-visible'
        )
    if rolled_back_filtered_hits:
        raise AssertionError(
            'rolled-back same-transaction scope remained query-visible'
        )
    return {
        'visible_hits': visible_hits,
        'visible_filtered_hits': visible_filtered_hits,
        'savepoint_hits': savepoint_hits,
        'restored_hits': restored_hits,
        'rolled_back_savepoint_hits': rolled_back_savepoint_hits,
        'rolled_back_hits': rolled_back_hits,
        'rolled_back_filtered_hits': rolled_back_filtered_hits,
    }


def run_structured_filter_repeatable_read_audit(
    connection: psycopg.Connection[Any],
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    reader = connect(socket_dir, port)
    reader.autocommit = False
    old_filter = {'categories': {'overlap': ['database']}}
    new_filter = {'categories': {'overlap': ['snapshot-new']}}

    try:
        with reader.cursor() as cursor:
            cursor.execute(
                'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ'
            )
        before_update = fetch_structured_filtered_hits(
            reader,
            INITIAL_ROWS[0][1],
            old_filter,
            limit=3,
            exact=True,
        )
        score_for_id(before_update, 1)

        with connection.cursor() as cursor:
            cursor.execute(
                'UPDATE convergent_sae.docs '
                'SET categories = %s WHERE id = 1',
                (['snapshot-new'],),
            )

        old_snapshot_old_filter = fetch_structured_filtered_hits(
            reader,
            INITIAL_ROWS[0][1],
            old_filter,
            limit=3,
            exact=True,
        )
        old_snapshot_new_filter = fetch_structured_filtered_hits(
            reader,
            INITIAL_ROWS[0][1],
            new_filter,
            limit=3,
            exact=True,
        )
        fresh_old_filter = fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            old_filter,
            limit=3,
            exact=True,
        )
        fresh_new_filter = fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            new_filter,
            limit=3,
            exact=True,
        )

        score_for_id(old_snapshot_old_filter, 1)
        score_for_id(fresh_new_filter, 1)
        if any(row_id == 1 for row_id, _score in old_snapshot_new_filter):
            raise AssertionError(
                'REPEATABLE READ structured filter observed new metadata'
            )
        if any(row_id == 1 for row_id, _score in fresh_old_filter):
            raise AssertionError(
                'fresh structured filter retained old metadata'
            )
    finally:
        reader.rollback()
        reader.close()

    with connection.cursor() as cursor:
        cursor.execute(
            'UPDATE convergent_sae.docs '
            'SET categories = %s WHERE id = 1',
            (['database', 'systems'],),
        )
    maintenance, status = maintain_until_converged(
        connection,
        expected_docs=None,
    )
    restored_old_filter = fetch_structured_filtered_hits(
        connection,
        INITIAL_ROWS[0][1],
        old_filter,
        limit=3,
    )
    restored_new_filter = fetch_structured_filtered_hits(
        connection,
        INITIAL_ROWS[0][1],
        new_filter,
        limit=3,
    )
    score_for_id(restored_old_filter, 1)
    if any(row_id == 1 for row_id, _score in restored_new_filter):
        raise AssertionError(
            'restored structured filter retained transient metadata'
        )

    return {
        'before_update': before_update,
        'old_snapshot_old_filter': old_snapshot_old_filter,
        'old_snapshot_new_filter': old_snapshot_new_filter,
        'fresh_old_filter': fresh_old_filter,
        'fresh_new_filter': fresh_new_filter,
        'restored_old_filter': restored_old_filter,
        'restored_new_filter': restored_new_filter,
        'restore_maintenance': maintenance,
        'restored_status': status,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    extension_libdir = args.extension_libdir.expanduser().resolve()
    extension_control_dir = extension_control_root(
        args.extension_control_dir
    )
    model_path = args.model_path.expanduser().resolve()
    manifest = load_manifest(model_path)
    if not 0.01 <= args.semantic_alpha_mass <= 1.0:
        raise ValueError('semantic alpha mass must be between 0.01 and 1.0')
    extension_libraries = [
        extension_libdir / name for name in ('evoke.so', 'evoke.dylib')
    ]
    if not any(path.is_file() for path in extension_libraries):
        raise FileNotFoundError(
            f'evoke library is missing from {extension_libdir}'
        )

    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    system_libdir = pg_config_value(args.pg_bin, '--pkglibdir')
    system_sharedir = pg_config_value(args.pg_bin, '--sharedir')
    root = create_short_socket_root('evoke-v3-sae-')
    data_dir = root / 'data'
    socket_dir = root / 's'
    log_path = root / 'postgres.log'
    port = reserve_port()
    socket_dir.mkdir()
    started = False
    connection: psycopg.Connection[Any] | None = None
    maintenance_guard: psycopg.Connection[Any] | None = None
    gates: dict[str, bool] = {}
    evidence: dict[str, Any] = {}

    try:
        run(
            [
                str(initdb),
                '-D',
                str(data_dir),
                '-A',
                'trust',
                '-U',
                'postgres',
            ]
        )
        configure_cluster(
            data_dir,
            socket_dir,
            port,
            extension_libdir=extension_libdir,
            system_libdir=system_libdir,
            extension_control_dir=extension_control_dir,
            system_sharedir=system_sharedir,
        )
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        setup(
            connection,
            model_path,
            args.semantic_alpha_mass,
            args.semantic_impact_precision,
        )

        (
            initial_touch,
            initial_observations,
            initial_status,
        ) = wait_for_background_convergence(
            connection,
            expected_docs=len(INITIAL_ROWS),
        )
        assert_initial_status(initial_status)
        evidence['initial_background_publication'] = {
            'touch': initial_touch,
            'observations': initial_observations,
        }
        gates['fresh_v3_sae_build_is_complete'] = True
        gates['default_sae_resolves_to_eventual'] = True

        failed_touch_retry = run_failed_touch_retry_audit(connection)
        gates['failed_worker_launch_does_not_consume_retry_cooldown'] = True

        owner_bm25_diagnostic_contract = (
            run_owner_bm25_diagnostic_contract_audit(connection)
        )
        gates['owner_bm25_diagnostics_reject_unified_sae'] = True

        bmp_probe = fetch_page_native_exactness_probe(
            connection,
            INITIAL_ROWS[0][1],
            disable_semantic_bmp=False,
        )
        fallback_probe = fetch_page_native_exactness_probe(
            connection,
            INITIAL_ROWS[0][1],
            disable_semantic_bmp=True,
        )
        legacy_stream_probe = fetch_page_native_exactness_probe(
            connection,
            INITIAL_ROWS[0][1],
            disable_semantic_bmp=True,
            disable_fused_semantic_taat=True,
        )
        assert_page_native_exactness_probe(
            bmp_probe,
            fallback_probe,
            'initial publication',
        )
        gates['compact_and_streaming_scorers_match_exact_oracle'] = True
        if not (
            fallback_probe['term_at_a_time_query_path'] is True
            and legacy_stream_probe['term_at_a_time_query_path'] is True
            and fallback_probe['matched'] is True
            and legacy_stream_probe['matched'] is True
            and fallback_probe['page_native_doc_ids'] ==
                legacy_stream_probe['page_native_doc_ids']
            and fallback_probe['postings_examined'] ==
                legacy_stream_probe['postings_examined']
        ):
            raise AssertionError(
                'fused semantic TAAT parity failed: '
                f'fused={fallback_probe}, legacy={legacy_stream_probe}'
            )
        gates['fused_semantic_taat_matches_materialized_stream'] = True

        allowed_document_slots = bmp_probe['page_native_doc_ids'][::2]
        filtered_bmp_probe = fetch_filtered_bmp_exactness_probe(
            connection,
            INITIAL_ROWS[0][1],
            allowed_document_slots,
        )
        expected_filtered_slots = [
            document_slot
            for document_slot in bmp_probe['page_native_doc_ids']
            if document_slot in allowed_document_slots
        ]
        if not (
            filtered_bmp_probe['filtered'] is True
            and filtered_bmp_probe['semantic_bmp_query_path'] is True
            and filtered_bmp_probe[
                'semantic_bmp_direct_flat_filtered'
            ] is True
            and filtered_bmp_probe['topk_complete'] is True
            and int(filtered_bmp_probe['semantic_bmp_ref_reads']) > 0
            and filtered_bmp_probe['semantic_bmp_ref_reads'] ==
                filtered_bmp_probe['filtered_bmp_matching_ref_count']
            and filtered_bmp_probe['page_native_doc_ids'] ==
                expected_filtered_slots
        ):
            raise AssertionError(
                'filtered semantic BMP membership path failed: '
                f'{filtered_bmp_probe}'
            )
        evidence['filtered_bmp_membership_probe'] = filtered_bmp_probe
        gates['filtered_semantic_bmp_uses_exact_membership'] = True

        error_budget_ratio = 1.0
        error_budget_probe = fetch_page_native_exactness_probe(
            connection,
            INITIAL_ROWS[0][1],
            disable_semantic_bmp=False,
            error_budget_ratio=error_budget_ratio,
        )
        assert_page_native_error_budget_probe(
            error_budget_probe,
            error_budget_ratio,
        )
        post_budget_exact_probe = fetch_page_native_exactness_probe(
            connection,
            INITIAL_ROWS[0][1],
            disable_semantic_bmp=False,
        )
        if post_budget_exact_probe['matched'] is not True:
            raise AssertionError(
                'resetting the semantic error budget did not restore exact '
                f'parity: {post_budget_exact_probe}'
            )
        gates['query_semantic_error_budget_is_bounded'] = True

        filtered_allowed_ids = [1, 3]
        filtered_full_hits = fetch_hits(
            connection,
            INITIAL_ROWS[0][1],
            exact=True,
        )
        expected_filtered_hits = [
            hit
            for hit in filtered_full_hits
            if hit[0] in filtered_allowed_ids
        ]
        filtered_hits = fetch_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            filtered_allowed_ids,
            limit=len(filtered_allowed_ids),
            exact=True,
        )
        empty_filtered_hits = fetch_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            [],
            limit=2,
            exact=True,
        )
        if (
            not same_hits(expected_filtered_hits, filtered_hits)
            or empty_filtered_hits
        ):
            raise AssertionError(
                'TID-filtered top-k differs from the exact subset ranking: '
                f'expected={expected_filtered_hits}, got={filtered_hits}, '
                f'empty={empty_filtered_hits}'
            )
        gates['tid_filter_is_exact_within_the_allowed_snapshot'] = True
        accelerated_filtered_hits = fetch_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            filtered_allowed_ids,
            limit=len(filtered_allowed_ids),
        )
        if not same_hit_ids(expected_filtered_hits, accelerated_filtered_hits):
            raise AssertionError(
                'TID-filtered accelerator changed subset ranking IDs: '
                f'exact={expected_filtered_hits}, '
                f'accelerated={accelerated_filtered_hits}'
            )
        gates['tid_filter_uses_the_default_bounded_accelerator'] = True
        with connection.cursor() as cursor:
            cursor.execute('SELECT evoke_query_trace_internal()')
            tid_filter_trace = cursor.fetchone()[0]
        if (
            tid_filter_trace is None
            or tid_filter_trace.get('query_route')
                not in {'forward_rows', 'forward_transpose'}
            or int(tid_filter_trace.get('ranked_prefix_probe_attempts', -1))
                != 0
            or int(tid_filter_trace.get('allowed_documents', 0))
                != len(filtered_allowed_ids)
            or int(tid_filter_trace.get('documents_examined', 0))
                > len(filtered_allowed_ids)
            or int(
                tid_filter_trace.get(
                    'accelerator_forward_postings_examined',
                    0,
                )
            ) <= 0
            or int(tid_filter_trace.get('accelerator_forward_bytes', 0)) <= 0
        ):
            raise AssertionError(
                'TID filter did not reject the more expensive ranked prefix: '
                f'{tid_filter_trace}'
            )
        evidence['tid_filter_trace'] = tid_filter_trace
        gates['tid_filter_avoids_more_expensive_prefix'] = True

        forward_route_evidence: dict[str, Any] = {}
        forward_route_hits: dict[str, list[tuple[int, float]]] = {}
        for forward_route in ('direct', 'transpose'):
            route_hits = fetch_filtered_hits(
                connection,
                INITIAL_ROWS[0][1],
                filtered_allowed_ids,
                limit=len(filtered_allowed_ids),
                forward_route=forward_route,
            )
            with connection.cursor() as cursor:
                cursor.execute('SELECT evoke_query_trace_internal()')
                route_trace = cursor.fetchone()[0]
            forward_route_hits[forward_route] = route_hits
            forward_route_evidence[forward_route] = route_trace
        if (
            not same_hits(
                forward_route_hits['direct'],
                forward_route_hits['transpose'],
            )
            or not same_hit_ids(
                expected_filtered_hits,
                forward_route_hits['direct'],
            )
            or forward_route_evidence['direct'].get('query_route') !=
                'forward_rows'
            or forward_route_evidence['transpose'].get('query_route') !=
                'forward_transpose'
        ):
            raise AssertionError(
                'filtered forward route diagnostics disagree: '
                f'hits={forward_route_hits}, '
                f'traces={forward_route_evidence}'
            )
        evidence['filtered_forward_route_probes'] = forward_route_evidence
        gates['filtered_forward_route_controls_are_equivalent'] = True

        structured_cases = {
            'eq': ({'id': {'eq': 3}}, [3]),
            'in': ({'id': {'in': [1, 3]}}, [1, 3]),
            'range': (
                {'id': {'range': {'gte': 1, 'lt': 3}}},
                [1, 2],
            ),
            'date_range': (
                {
                    'published_on': {
                        'range': {
                            'gte': '2024-02-01',
                            'lt': '2024-04-01',
                        },
                    },
                },
                [2, 3],
            ),
            'and': (
                {
                    'id': {'in': [1, 2, 3]},
                    'revision': {'eq': 0},
                },
                [1, 2, 3],
            ),
            'overlap': (
                {'categories': {'overlap': ['database']}},
                [1],
            ),
            'ilike': (
                {'source_name': {'ilike': '%research%'}},
                [1],
            ),
            'ilike_any': (
                {
                    'source_name': {
                        'ilike_any': ['%journal%', '%archive%'],
                    },
                },
                [2, 3],
            ),
            'array_ilike_any': (
                {'categories': {'ilike_any': ['%base%', '%archive%']}},
                [1, 3],
            ),
            'array_ilike_unicode_ascii_fold': (
                {'categories': {'ilike': '%kelvin%'}},
                [3],
            ),
            'array_ilike_residual': (
                {'tags': {'ilike_any': ['%base%', '%archive%']}},
                [1, 3],
            ),
            'array_ilike_residual_empty': (
                {'tags': {'ilike_any': []}},
                [],
            ),
            'array_ilike_residual_unicode': (
                {'tags': {'ilike_any': ['%数据%']}},
                [1],
            ),
            'range_and_overlap': (
                {
                    'id': {'range': {'gte': 1, 'lte': 2}},
                    'categories': {'overlap': ['retrieval']}},
                [2],
            ),
            'two_range_intersection': (
                {
                    'id': {'range': {'gte': 2}},
                    'published_on': {
                        'range': {'lte': '2024-02-29'},
                    },
                },
                [2],
            ),
            'range_and_sql_residual': (
                {
                    'published_on': {
                        'range': {'gte': '2024-02-01'},
                    },
                    'revision': {'eq': 0},
                },
                [2, 3],
            ),
            'date_range_and_ilike': (
                {
                    'published_on': {
                        'range': {
                            'gte': '2024-02-01',
                            'lt': '2024-04-01',
                        },
                    },
                    'source_name': {'ilike': '%archive%'},
                },
                [3],
            ),
        }
        for case_name, (filters, allowed_ids) in structured_cases.items():
            expected = [
                hit for hit in filtered_full_hits if hit[0] in allowed_ids
            ]
            observed = fetch_structured_filtered_hits(
                connection,
                INITIAL_ROWS[0][1],
                filters,
                limit=max(len(allowed_ids), 1),
                exact=True,
            )
            if not same_hits(expected, observed):
                raise AssertionError(
                    f'structured {case_name} filter changed subset ranking: '
                    f'expected={expected}, observed={observed}'
                )
        scope_cases = {
            name: value
            for name, value in structured_cases.items()
            if name in {
                'eq',
                'in',
                'range',
                'date_range',
                'overlap',
                'ilike',
                'ilike_any',
                'array_ilike_any',
                'array_ilike_unicode_ascii_fold',
                'range_and_overlap',
                'two_range_intersection',
                'range_and_sql_residual',
                'date_range_and_ilike',
            }
        }
        for case_name, (filters, allowed_ids) in scope_cases.items():
            expected = [
                hit for hit in filtered_full_hits if hit[0] in allowed_ids
            ]
            observed = fetch_structured_filtered_hits(
                connection,
                INITIAL_ROWS[0][1],
                filters,
                limit=len(allowed_ids),
                exact=True,
                require_scope=True,
            )
            if not same_hits(expected, observed):
                raise AssertionError(
                    f'scope {case_name} filter changed subset ranking: '
                    f'expected={expected}, observed={observed}'
                )
        ordered_filters = structured_cases['range_and_overlap'][0]
        reversed_filters = dict(reversed(tuple(ordered_filters.items())))
        ordered_hits = fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            ordered_filters,
            limit=1,
            exact=True,
            require_scope=True,
        )
        reversed_hits = fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            reversed_filters,
            limit=1,
            exact=True,
            require_scope=True,
        )
        if not same_hits(ordered_hits, reversed_hits):
            raise AssertionError(
                'structured filter result depends on JSON key order: '
                f'ordered={ordered_hits}, reversed={reversed_hits}'
            )
        gates['structured_filter_order_is_irrelevant'] = True
        fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            structured_cases['range_and_sql_residual'][0],
            limit=3,
            exact=True,
            require_scope=True,
        )
        with connection.cursor() as cursor:
            cursor.execute('SELECT evoke_query_trace_internal()')
            partial_scope_trace = cursor.fetchone()[0]
        if (
            partial_scope_trace is None
            or partial_scope_trace.get('scope_predicates') != 2
            or partial_scope_trace.get('scope_resolved_predicates') != 1
            or partial_scope_trace.get('sql_filter_residual') is not True
            or int(partial_scope_trace.get('allowed_documents', -1)) != 2
            or int(
                partial_scope_trace.get('ranked_prefix_probe_attempts', 0)
            ) != 0
        ):
            raise AssertionError(
                'SQL residual discarded or bypassed the native scope: '
                f'{partial_scope_trace}'
            )
        evidence['partial_scope_trace'] = partial_scope_trace
        gates['structured_filter_preserves_partial_native_scope'] = True
        gates['partial_scope_uses_final_allowed_set_for_route_selection'] = (
            True
        )
        fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            structured_cases['array_ilike_residual'][0],
            limit=2,
            exact=True,
        )
        with connection.cursor() as cursor:
            cursor.execute('SELECT evoke_query_trace_internal()')
            no_scope_trace = cursor.fetchone()[0]
        if (
            no_scope_trace is None
            or no_scope_trace.get('scope_predicates') != 1
            or no_scope_trace.get('scope_resolved_predicates') != 0
            or no_scope_trace.get('filter_probe_attempted') is not True
            or no_scope_trace.get('filter_probe_complete') is not True
            or int(no_scope_trace.get('filter_probe_rows', -1)) != 2
            or int(
                no_scope_trace.get('ranked_prefix_probe_attempts', 0)
            ) != 0
        ):
            raise AssertionError(
                'small-root no-scope filter performed an uneconomic prefix: '
                f'{no_scope_trace}'
            )
        evidence['no_scope_prefix_trace'] = no_scope_trace
        gates['small_root_skips_no_scope_ranked_prefix'] = True
        gates['small_root_no_scope_filter_uses_bounded_probe'] = True
        gates['structured_filter_intersects_independent_ranges'] = True
        cross_element_hits = fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            {'tags': {'ilike': '%a b%'}},
            limit=3,
            exact=True,
        )
        if cross_element_hits:
            raise AssertionError(
                'array trigram candidate bypassed element-level exactness: '
                f'{cross_element_hits}'
            )
        gates['array_trigram_candidate_preserves_element_exactness'] = True
        observed = fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            structured_cases['date_range_and_ilike'][0],
            limit=2,
            exact=True,
            require_scope=True,
        )
        if not same_hits(
            [hit for hit in filtered_full_hits if hit[0] == 3],
            observed,
        ):
            raise AssertionError(
                'scope trace probe changed exact subset ranking: '
                f'observed={observed}'
            )
        with connection.cursor() as cursor:
            cursor.execute('SELECT evoke_query_trace_internal()')
            structured_query_trace = cursor.fetchone()[0]
        if (
            structured_query_trace is None
            or structured_query_trace.get('scope_predicates') != 2
            or structured_query_trace.get('scope_resolved_predicates') != 2
            or structured_query_trace.get('sql_filter_residual') is not False
            or int(structured_query_trace.get('allowed_documents', -1)) != 1
            or int(
                structured_query_trace.get('visibility_rank_attempts', 0)
            ) < 1
        ):
            raise AssertionError(
                'structured query telemetry does not describe the native '
                f'scope intersection: {structured_query_trace}'
            )
        evidence['structured_query_trace'] = structured_query_trace
        gates['structured_filter_route_is_observable'] = True
        gates['visibility_rank_attempts_are_observable'] = True
        if (
            structured_query_trace.get('ranked_prefix_probe_attempts') != 0
            or structured_query_trace.get('ranked_prefix_query_path') is True
        ):
            raise AssertionError(
                'fully resolved native scope performed ranked-prefix work: '
                f'{structured_query_trace}'
            )
        gates['structured_filter_uses_scope_before_prefix'] = True
        empty_structured_hits = fetch_structured_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            {'id': {'in': []}},
            limit=2,
            exact=True,
        )
        if empty_structured_hits:
            raise AssertionError(
                'empty structured IN filter returned search hits: '
                f'{empty_structured_hits}'
            )
        invalid_structured_cases = (
            ({}, 'non-empty JSON object'),
            ({'missing': {'eq': 1}}, 'does not exist'),
            ({'id': {'contains': 1}}, 'unsupported evoke filter operation'),
            ({'id': {'overlap': [1]}}, 'must have an array type'),
            ({'id': {'ilike': '%1%'}}, 'must have a string type'),
            (
                {'source_name': {'ilike_any': ['%journal%', 3]}},
                'array of JSON strings',
            ),
            (
                {'id': {'in': list(range(4097))}},
                'accepts at most 4096 values',
            ),
            (
                {'id': {'eq': 1, 'range': {'gte': 1}}},
                'exactly one operation',
            ),
        )
        for filters, expected_error in invalid_structured_cases:
            try:
                fetch_structured_filtered_hits(
                    connection,
                    INITIAL_ROWS[0][1],
                    filters,
                    limit=2,
                )
            except psycopg.Error as error:
                if expected_error not in str(error):
                    raise AssertionError(
                        'unexpected structured filter validation error: '
                        f'filters={filters}, error={error}'
                    ) from error
            else:
                raise AssertionError(
                    f'invalid structured filter was accepted: {filters}'
                )
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT *
                    FROM evoke_query(
                        'convergent_sae.docs_idx'::regclass,
                        %s,
                        NULL::jsonb,
                        2
                    )
                    """,
                    (INITIAL_ROWS[0][1],),
                )
        except psycopg.Error as error:
            if 'non-empty JSON object' not in str(error):
                raise AssertionError(
                    'unexpected null structured filter error: '
                    f'{error}'
                ) from error
        else:
            raise AssertionError('null structured filter was accepted')
        gates['structured_filters_are_exact_within_the_allowed_snapshot'] = (
            True
        )
        gates['included_scope_filters_use_same_root_postings'] = True
        gates['structured_filters_reject_unsafe_or_ambiguous_shapes'] = True

        configure_test_l0_rotation_records(connection, 1)
        consistency_contract = run_consistency_contract_audit(
            connection,
            model_path,
        )
        gates['sae_and_bm25_consistency_contract_is_exact'] = True

        multicolumn_contract = run_multicolumn_sae_contract_audit(
            connection,
            model_path,
            args.semantic_alpha_mass,
            args.semantic_impact_precision,
        )
        gates[
            'multicolumn_sae_supports_fused_and_field_aware_modes'
        ] = True

        fold_compaction_preload = run_fold_compaction_preload_audit(
            connection,
            model_path,
            args.semantic_alpha_mass,
            args.semantic_impact_precision,
        )
        gates[
            'sae_fold_compaction_preserves_preload_and_query'
        ] = True

        alpha_contract = run_semantic_alpha_contract_audit(
            connection,
            model_path,
            args.semantic_alpha_mass,
            args.semantic_impact_precision,
        )
        gates['semantic_alpha_mass_is_generation_bound'] = True

        impact_precision_contract = (
            run_semantic_impact_precision_contract_audit(
                connection,
                model_path,
                args.semantic_impact_precision,
                args.semantic_alpha_mass,
            )
        )
        gates['semantic_impact_precision_is_generation_bound'] = True

        pre_reindex_hits = fetch_hits(
            connection,
            INITIAL_ROWS[0][1],
            limit=2,
            exact=True,
        )
        with connection.cursor() as cursor:
            cursor.execute('REINDEX INDEX convergent_sae.docs_idx')
        reindexed_hits = fetch_hits(
            connection,
            INITIAL_ROWS[0][1],
            limit=2,
            exact=True,
        )
        reindex_maintenance, reindexed_status = maintain_until_converged(
            connection,
            expected_docs=len(INITIAL_ROWS),
        )
        assert_initial_status(reindexed_status)
        if not same_hits(pre_reindex_hits, reindexed_hits):
            raise AssertionError(
                'default SAE REINDEX changed rows or scores: '
                f'{pre_reindex_hits} != {reindexed_hits}'
            )
        gates['default_sae_reindex_preserves_rows_and_scores'] = True

        accelerator_status = fetch_status(connection)
        accelerator_hits = fetch_hits(
            connection,
            INITIAL_ROWS[0][1],
            limit=2,
        )
        accelerator_exact_hits = fetch_hits(
            connection,
            INITIAL_ROWS[0][1],
            limit=2,
            exact=True,
        )
        accelerator_state = accelerator_status['generation'][
            'semantic_accelerator'
        ]
        if not (
            accelerator_state['present'] is True
            and accelerator_state['eligible'] is True
            and accelerator_state['state'] == 'ready'
            and int(accelerator_state['term_count']) > 0
            and accelerator_state['forward_complete'] is True
            and int(accelerator_state['forward_chunk_count']) > 0
            and int(accelerator_state['directory_bytes']) > 0
            and accelerator_state['bytes'] is None
            and same_hits(reindexed_hits, accelerator_exact_hits)
        ):
            raise AssertionError(
                'semantic accelerator publication changed authority: '
                f'status={accelerator_state}, '
                f'hits={accelerator_hits}'
            )
        gates['semantic_accelerator_publication_is_derived_only'] = True

        accelerator_default_probe = fetch_semantic_accelerator_probe(
            connection,
            INITIAL_ROWS[0][1],
            heap_factor=0.7,
            use_defaults=True,
        )
        if not (
            accelerator_default_probe[
                'semantic_accelerator_attempted'
            ] is True
            and accelerator_default_probe[
                'semantic_accelerator_fallback'
            ] is False
            and accelerator_default_probe[
                'semantic_accelerator_query_path'
            ] is True
            and accelerator_default_probe[
                'semantic_accelerator_residual_candidates_bounded'
            ] is True
            and accelerator_default_probe['positive_topk_complete'] is True
            and same_hit_ids(accelerator_exact_hits, accelerator_hits)
        ):
            raise AssertionError(
                'default semantic accelerator route is not active: '
                f'probe={accelerator_default_probe}, '
                f'exact={accelerator_exact_hits}, default={accelerator_hits}'
            )
        gates['semantic_accelerator_is_the_default_query_route'] = True

        accelerator_exact_probe = fetch_semantic_accelerator_probe(
            connection,
            INITIAL_ROWS[0][1],
            heap_factor=0.0,
        )
        if not (
            accelerator_exact_probe['matched'] is True
            and accelerator_exact_probe[
                'semantic_accelerator_attempted'
            ] is True
            and accelerator_exact_probe[
                'semantic_accelerator_fallback'
            ] is True
            and accelerator_exact_probe[
                'semantic_accelerator_query_path'
            ] is False
        ):
            raise AssertionError(
                'exact query did not bypass the lossy accelerator: '
                f'{accelerator_exact_probe}'
            )
        gates['exact_query_bypasses_lossy_semantic_accelerator'] = True

        accelerator_bounded_probe = fetch_semantic_accelerator_probe(
            connection,
            INITIAL_ROWS[0][1],
            heap_factor=0.7,
            bound_residual_candidates=True,
        )
        if not (
            accelerator_bounded_probe[
                'semantic_accelerator_attempted'
            ] is True
            and accelerator_bounded_probe[
                'semantic_accelerator_fallback'
            ] is False
            and accelerator_bounded_probe[
                'semantic_accelerator_query_path'
            ] is True
            and accelerator_bounded_probe[
                'semantic_accelerator_residual_candidates_bounded'
            ] is True
            and accelerator_bounded_probe['positive_topk_complete'] is True
        ):
            raise AssertionError(
                'bounded accelerator residual candidate canary is invalid: '
                f'{accelerator_bounded_probe}'
            )
        gates['semantic_accelerator_residual_candidate_canary_is_bounded'] = (
            True
        )
        accelerator_memory_parts = (
            'accelerator_owned_index_bytes',
            'accelerator_membership_bytes',
            'accelerator_candidate_scratch_bytes',
            'accelerator_forward_scratch_bytes',
            'accelerator_residual_scratch_bytes',
        )
        if any(
            field not in accelerator_bounded_probe
            or int(accelerator_bounded_probe[field]) < 0
            for field in accelerator_memory_parts
        ) or (
            'semantic_accelerator_residual_dense_accumulation'
            not in accelerator_bounded_probe
        ):
            raise AssertionError(
                'accelerator ownership telemetry is incomplete'
            )
        accelerator_query_bytes = int(
            accelerator_bounded_probe.get(
                'semantic_accelerator_query_bytes',
                -1,
            )
        )
        if accelerator_query_bytes <= 0 or accelerator_query_bytes != sum(
            int(accelerator_bounded_probe[field])
            for field in accelerator_memory_parts
        ):
            raise AssertionError(
                'accelerator ownership telemetry does not reconcile'
            )
        gates['accelerator_query_memory_is_observable'] = True
        residual_scratch_bytes = int(
            accelerator_bounded_probe['accelerator_residual_scratch_bytes']
        )
        if residual_scratch_bytes > 64 * 1024 * 1024:
            raise AssertionError(
                'residual candidate workspace exceeded its query-local '
                f'budget: {residual_scratch_bytes}'
            )
        gates['residual_candidate_workspace_is_query_bounded'] = True
        gates['residual_candidate_strategy_is_observable'] = True

        accelerator_budget_probe = fetch_semantic_accelerator_probe(
            connection,
            INITIAL_ROWS[0][1],
            heap_factor=0.7,
            bound_residual_candidates=True,
            error_budget_ratio=1.0,
        )
        if not (
            accelerator_budget_probe[
                'semantic_accelerator_attempted'
            ] is True
            and accelerator_budget_probe[
                'semantic_accelerator_fallback'
            ] is False
            and accelerator_budget_probe[
                'semantic_accelerator_query_path'
            ] is True
            and int(
                accelerator_budget_probe[
                    'query_error_budget_pruned_term_count'
                ]
            ) > 0
            and int(
                accelerator_budget_probe[
                    'query_error_budget_pruned_postings'
                ]
            ) > 0
            and int(
                accelerator_budget_probe[
                    'accelerator_residual_postings'
                ]
            ) < int(
                accelerator_bounded_probe[
                    'accelerator_residual_postings'
                ]
            )
            and accelerator_budget_probe['positive_topk_complete'] is True
        ):
            raise AssertionError(
                'accelerator error budget did not reduce residual work: '
                f'baseline={accelerator_bounded_probe}, '
                f'budget={accelerator_budget_probe}'
            )
        gates['semantic_accelerator_error_budget_reduces_residual_work'] = (
            True
        )

        accelerator_seed_probe = fetch_semantic_accelerator_probe(
            connection,
            INITIAL_ROWS[0][1],
            heap_factor=0.7,
            bound_residual_candidates=True,
            seed_bmp=True,
        )
        if not (
            accelerator_seed_probe['matched'] is True
            and accelerator_seed_probe[
                'semantic_accelerator_attempted'
            ] is True
            and accelerator_seed_probe[
                'semantic_accelerator_fallback'
            ] is False
            and accelerator_seed_probe[
                'semantic_accelerator_query_path'
            ] is False
            and accelerator_seed_probe['semantic_bmp_attempted'] is True
            and accelerator_seed_probe['semantic_bmp_query_path'] is True
        ):
            raise AssertionError(
                'accelerator-seeded BMP did not preserve exact results: '
                f'{accelerator_seed_probe}'
            )
        gates['semantic_accelerator_can_seed_exact_bmp'] = True

        configure_test_l0_rotation_records(connection, None)
        configure_maintenance_worker_limit(connection, 0)
        maintenance_guard = acquire_maintenance_lock(socket_dir, port)
        release_maintenance_lock(maintenance_guard)
        maintenance_guard = None
        preinsert_due_result = maintain_due_once(connection)
        if preinsert_due_result is not None:
            raise AssertionError(
                'maintenance selector was not idle before low-debt audit: '
                f'{preinsert_due_result}'
            )
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO convergent_sae.docs (id, body) VALUES (4, %s)',
                (INSERT_TEXT,),
            )
        insert_pending_status = fetch_status(connection)
        assert_pending_lexical_status(insert_pending_status)
        insert_accelerator_state = insert_pending_status['generation'][
            'semantic_accelerator'
        ]
        if not (
            insert_accelerator_state['present'] is True
            and insert_accelerator_state['eligible'] is True
            and insert_accelerator_state['compatible'] is True
            and insert_accelerator_state['state'] == 'ready_baseline_delta'
            and insert_accelerator_state['baseline_current'] is False
            and int(insert_accelerator_state['baseline_sequence']) > 0
            and int(insert_accelerator_state['refresh_delta_records']) == 0
            and int(insert_accelerator_state['refresh_delta_bytes']) == 0
            and insert_accelerator_state['refresh_max_age_ms'] is None
            and insert_accelerator_state['refresh_due'] is False
            and insert_accelerator_state[
                'periodic_refresh_eligible'
            ] is False
            and int(
                insert_accelerator_state[
                    'periodic_refresh_interval_ms'
                ]
            ) == 1000
        ):
            raise AssertionError(
                'semantic accelerator did not retain its baseline for L0: '
                f'{insert_accelerator_state}'
            )
        premature_due_results = [
            maintain_due_once(connection) for _ in range(4)
        ]
        stale_noop_status = fetch_status(connection)
        stale_noop_accelerator = stale_noop_status['generation'][
            'semantic_accelerator'
        ]
        if not (
            all(result is None for result in premature_due_results)
            and int(
                stale_noop_status['generation']['delta']['active']['records']
            ) == 1
            and int(
                stale_noop_accelerator['baseline_sequence']
            ) == int(insert_accelerator_state['baseline_sequence'])
            and stale_noop_accelerator['refresh_due'] is False
            and stale_noop_accelerator['refresh_max_age_ms'] is None
        ):
            raise AssertionError(
                'sub-threshold delta changed before its periodic deadline: '
                f'results={premature_due_results}, status={stale_noop_status}'
            )
        gates['semantic_accelerator_stale_baseline_is_bounded'] = True
        gates['sub_threshold_delta_waits_for_periodic_checkpoint'] = True
        insert_stale_hits = fetch_hits(
            connection,
            INITIAL_ROWS[0][1],
        )
        insert_stale_trace = fetch_query_trace(connection)
        if not (
            insert_stale_trace['accelerator_query_path'] is True
            and insert_stale_trace['accelerator_stale_baseline'] is True
            and int(
                insert_stale_trace['accelerator_baseline_sequence']
            ) == int(insert_accelerator_state['baseline_sequence'])
            and int(
                insert_stale_trace[
                    'accelerator_baseline_document_slots'
                ]
            ) == len(INITIAL_ROWS)
        ):
            raise AssertionError(
                'semantic accelerator did not retain the stale baseline: '
                f'{insert_stale_trace}'
            )
        score_for_id(insert_stale_hits, 1)
        insert_immediate_hits = fetch_hits(
            connection,
            INSERT_TEXT,
            exact=True,
        )
        insert_immediate_score = score_for_id(insert_immediate_hits, 4)
        insert_filtered_hits = fetch_structured_filtered_hits(
            connection,
            INSERT_TEXT,
            {'categories': {'overlap': ['general']}},
            limit=1,
            exact=True,
        )
        score_for_id(insert_filtered_hits, 4)
        insert_l0_bmp_probe = fetch_page_native_exactness_probe(
            connection,
            INSERT_TEXT,
            disable_semantic_bmp=False,
        )
        insert_l0_fallback_probe = fetch_page_native_exactness_probe(
            connection,
            INSERT_TEXT,
            disable_semantic_bmp=True,
        )
        assert_page_native_exactness_probe(
            insert_l0_bmp_probe,
            insert_l0_fallback_probe,
            'linked-L0 exact BMP',
        )
        gates['exact_bmp_merges_linked_l0_projection'] = True
        gates['insert_is_lexically_visible_before_inference'] = True
        gates['insert_is_immediately_visible_to_structured_filters'] = True
        time.sleep(1.2)
        runtime_before_active_refresh = fetch_runtime_status(connection)
        active_rotation_result = maintain_due_once(connection)
        active_rotation_status = fetch_status(connection)
        active_rotation_delta = active_rotation_status['generation']['delta']
        active_rotation_accelerator = active_rotation_status['generation'][
            'semantic_accelerator'
        ]
        runtime_after_active_rotation = fetch_runtime_status(connection)
        if not (
            active_rotation_result is not None
            and 'reason=active_l0_rotated' in active_rotation_result
            and int(active_rotation_delta['active']['records']) == 0
            and int(active_rotation_delta['pending']['records']) == 1
            and active_rotation_accelerator['eligible'] is True
            and active_rotation_accelerator['state'] == 'ready_baseline_delta'
            and int(
                active_rotation_accelerator['forward_document_shift']
            ) == int(insert_accelerator_state['forward_document_shift'])
            and int(runtime_after_active_rotation['runtime_runs']) == int(
                runtime_before_active_refresh['runtime_runs']
            )
        ):
            raise AssertionError(
                'accelerator refresh did not yield to active ingress: '
                f'result={active_rotation_result}, '
                f'status={active_rotation_status}, '
                f'runtime_before={runtime_before_active_refresh}, '
                f'runtime_after={runtime_after_active_rotation}'
            )
        score_for_id(fetch_hits(connection, INSERT_TEXT, exact=True), 4)
        gates['periodic_low_debt_checkpoint_runs_without_threshold'] = True
        gates['accelerator_refresh_yields_to_active_ingress'] = True
        configure_test_l0_rotation_records(connection, 1)
        (
            accelerator_fairness_phases,
            accelerator_fairness_status,
        ) = maintain_due_until_accelerator_published_with_semantic_debt(
            connection
        )
        accelerator_fairness_completion = semantic_completion(
            accelerator_fairness_status
        )
        if not (
            int(accelerator_fairness_completion['pending']) > 0
            and accelerator_fairness_status['generation'][
                'semantic_accelerator'
            ]['state'] == 'ready'
        ):
            raise AssertionError(
                'accelerator refresh did not publish independently of '
                'semantic completion: '
                f'{accelerator_fairness_status}'
            )
        gates[
            'accelerator_refresh_progresses_with_continuous_semantic_debt'
        ] = True
        configure_maintenance_worker_limit(connection, 1)
        configured_budget = configure_maintenance_budget(connection, '1MB')
        if configured_budget != '1MB':
            raise AssertionError(
                f'invalid accelerator test budget: {configured_budget}'
            )

        (
            accelerator_budget_phases,
            accelerator_budget_status,
        ) = maintain_until_accelerator_budget_blocked(connection)
        if not any(
            'reason=pending_l0_sealed' in phase['result']
            for phase in accelerator_budget_phases
        ):
            raise AssertionError(
                'blocked accelerator refresh starved pending L0 sealing: '
                f'{accelerator_budget_phases}'
            )
        gates['blocked_accelerator_does_not_starve_l0_sealing'] = True
        blocked_completion = semantic_completion(accelerator_budget_status)
        blocked_accelerator = accelerator_budget_status['generation'][
            'semantic_accelerator'
        ]
        if not (
            blocked_completion['converged'] is True
            and int(blocked_completion['pending']) == 0
            and blocked_accelerator['state'] != 'ready'
        ):
            raise AssertionError(
                'accelerator budget gate did not preserve a converged exact '
                f'root: {accelerator_budget_status}'
            )
        blocked_exact_probe = fetch_page_native_exactness_probe(
            connection,
            INITIAL_ROWS[0][1],
            disable_semantic_bmp=False,
        )
        if blocked_exact_probe['matched'] is not True:
            raise AssertionError(
                'accelerator budget gate changed exact query results: '
                f'{blocked_exact_probe}'
            )
        runtime_after_accelerator_block = fetch_runtime_status(connection)
        gates['semantic_accelerator_memory_budget_preserves_exact_root'] = True
        configure_maintenance_budget(connection, None)

        insert_touch = 'manual_threshold_drain'
        insert_phases, insert_status = maintain_until_converged(
            connection,
            expected_docs=4,
        )
        insert_converged_accelerator = insert_status['generation'][
            'semantic_accelerator'
        ]
        runtime_after_accelerator_rebuild = fetch_runtime_status(connection)
        if int(runtime_after_accelerator_rebuild['runtime_runs']) != int(
            runtime_after_accelerator_block['runtime_runs']
        ):
            raise AssertionError(
                'derived accelerator rebuild reran semantic inference: '
                f'before={runtime_after_accelerator_block}, '
                f'after={runtime_after_accelerator_rebuild}'
            )
        if not semantic_accelerator_is_ready(insert_status):
            raise AssertionError(
                'semantic accelerator did not rebuild after convergence: '
                f'{insert_converged_accelerator}'
            )
        accelerator_rebuilt_probe = fetch_semantic_accelerator_probe(
            connection,
            INITIAL_ROWS[0][1],
            heap_factor=0.0,
        )
        if not (
            accelerator_rebuilt_probe['matched'] is True
            and accelerator_rebuilt_probe[
                'semantic_accelerator_attempted'
            ] is True
            and accelerator_rebuilt_probe[
                'semantic_accelerator_fallback'
            ] is True
            and accelerator_rebuilt_probe[
                'semantic_accelerator_query_path'
            ] is False
        ):
            raise AssertionError(
                'exact query did not bypass the rebuilt lossy accelerator: '
                f'{accelerator_rebuilt_probe}'
            )
        gates[
            'semantic_accelerator_rebuilds_after_convergence_without_inference'
        ] = True
        insert_complete_hits = fetch_hits(connection, INSERT_TEXT)
        insert_complete_score = score_for_id(insert_complete_hits, 4)
        if math.isclose(
            insert_immediate_score,
            insert_complete_score,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                'semantic completion did not change the inserted document '
                f'score: {insert_immediate_score}'
            )
        gates['background_candidate_drains_sealed_semantic_debt'] = True

        single_atom_bmp_probes = fetch_single_atom_exactness_probes(
            connection,
            INSERT_TEXT,
            disable_semantic_bmp=False,
        )
        single_atom_fallback_probes = fetch_single_atom_exactness_probes(
            connection,
            INSERT_TEXT,
            disable_semantic_bmp=True,
        )
        assert_single_atom_exactness_probes(
            single_atom_bmp_probes,
            single_atom_fallback_probes,
        )
        gates[
            'single_atom_zero_fill_survives_incremental_publication'
        ] = True

        race_maintenance = acquire_maintenance_lock(socket_dir, port)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    'UPDATE convergent_sae.docs SET body = %s WHERE id = 4',
                    (RACE_SOURCE_TEXT,),
                )
            race_prepare, race_pending_status = prepare_sealed_semantic_debt(
                race_maintenance,
                connection,
                source_name='race source',
            )

            def replace_race_source() -> None:
                with connection.cursor() as cursor:
                    cursor.execute(
                        'UPDATE convergent_sae.docs '
                        'SET body = %s WHERE id = 4',
                        (RACE_REPLACEMENT_TEXT,),
                    )

            (
                race_publish_result,
                race_pause,
                _race_mutation,
            ) = maintain_during_publish_pause(
                race_maintenance,
                connection,
                replace_race_source,
            )
            if (
                'semantic_completed=0' not in race_publish_result
                or 'semantic_retired=1' not in race_publish_result
            ):
                raise AssertionError(
                    'stale semantic output was not rejected: '
                    f'{race_publish_result}'
                )
        finally:
            release_maintenance_lock(race_maintenance)

        race_phases, race_status = maintain_until_converged(
            connection,
            expected_docs=None,
        )
        race_current_hits = fetch_hits(connection, RACE_REPLACEMENT_TEXT)
        score_for_id(race_current_hits, 4)
        race_stale_hits = fetch_hits(connection, RACE_SOURCE_TEXT)
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT body FROM convergent_sae.docs WHERE id = 4'
            )
            race_current_body = str(cursor.fetchone()[0])
        if race_current_body != RACE_REPLACEMENT_TEXT:
            raise AssertionError(
                'semantic race changed the current heap version: '
                f'{race_current_body}'
            )
        gates['semantic_completion_rejects_stale_tuple_version'] = True

        hot_maintenance = acquire_maintenance_lock(socket_dir, port)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    'UPDATE convergent_sae.docs '
                    'SET body = %s, revision = 0 WHERE id = 4',
                    (HOT_SOURCE_TEXT,),
                )
            hot_prepare, hot_pending_status = prepare_sealed_semantic_debt(
                hot_maintenance,
                connection,
                source_name='HOT source',
            )
            hot_immediate_hits = fetch_hits(connection, HOT_SOURCE_TEXT)
            hot_immediate_score = score_for_id(hot_immediate_hits, 4)
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT ctid::text '
                    'FROM convergent_sae.docs WHERE id = 4'
                )
                hot_source_tid = str(cursor.fetchone()[0])

            def advance_hot_successor() -> dict[str, Any]:
                with connection.cursor() as cursor:
                    cursor.execute(
                        'UPDATE convergent_sae.docs '
                        'SET revision = revision + 1 WHERE id = 4'
                    )
                    cursor.execute(
                        'SELECT ctid::text, revision '
                        'FROM convergent_sae.docs WHERE id = 4'
                    )
                    row = cursor.fetchone()
                return {
                    'tid': str(row[0]),
                    'revision': int(row[1]),
                    'status': fetch_status(connection),
                }

            (
                hot_publish_result,
                hot_pause,
                hot_mutation,
            ) = maintain_during_publish_pause(
                hot_maintenance,
                connection,
                advance_hot_successor,
            )
            hot_mid_delta = hot_mutation['status']['generation']['delta']
            if (
                hot_mutation['tid'] == hot_source_tid
                or int(hot_mutation['revision']) != 1
                or int(hot_mid_delta['active']['records']) != 0
                or 'semantic_completed=1' not in hot_publish_result
                or 'semantic_retired=0' not in hot_publish_result
            ):
                raise AssertionError(
                    'valid HOT successor did not retain semantic completion: '
                    f'tid={hot_source_tid}, mutation={hot_mutation}, '
                    f'result={hot_publish_result}'
                )
        finally:
            release_maintenance_lock(hot_maintenance)

        hot_phases, hot_status = maintain_until_converged(
            connection,
            expected_docs=None,
        )
        hot_complete_hits = fetch_hits(connection, HOT_SOURCE_TEXT)
        hot_complete_score = score_for_id(hot_complete_hits, 4)
        if math.isclose(
            hot_immediate_score,
            hot_complete_score,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                'valid HOT successor lost semantic completion: '
                f'{hot_immediate_score}'
            )
        gates['semantic_completion_accepts_valid_hot_successor'] = True

        crash_maintenance = acquire_maintenance_lock(socket_dir, port)
        with connection.cursor() as cursor:
            cursor.execute(
                'UPDATE convergent_sae.docs '
                'SET body = %s WHERE id = 4',
                (CRASH_SOURCE_TEXT,),
            )
        crash_prepare, crash_pending_status = prepare_sealed_semantic_debt(
            crash_maintenance,
            connection,
            source_name='crash source',
        )
        crash_immediate_hits = fetch_hits(connection, CRASH_SOURCE_TEXT)
        crash_immediate_score = score_for_id(crash_immediate_hits, 4)
        (
            crash_thread,
            crash_thread_result,
            crash_pause,
        ) = start_maintenance_during_post_append_pause(
            crash_maintenance,
            connection,
        )
        crash_cluster(pg_ctl, data_dir)
        started = False
        crash_thread.join(timeout=12.0)
        if crash_thread.is_alive():
            raise AssertionError(
                'semantic maintenance backend survived immediate stop'
            )
        if (
            crash_thread_result.get('error') is None
            or crash_thread_result.get('result') is not None
        ):
            raise AssertionError(
                'semantic maintenance committed before immediate stop: '
                f'{crash_thread_result}'
            )
        crash_maintenance.close()
        connection.close()
        connection = None

        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        crash_restarted_status = fetch_status(connection)
        crash_restarted_completion = semantic_completion(
            crash_restarted_status
        )
        if not (
            int(crash_restarted_completion['sealed_pending']) == 1
            and crash_restarted_completion['pending_exact'] is True
            and crash_restarted_completion['converged'] is False
        ):
            raise AssertionError(
                'uncommitted semantic completion survived crash: '
                f'{crash_restarted_status}, {crash_thread_result}'
            )
        crash_phases, crash_status = maintain_until_converged(
            connection,
            expected_docs=None,
        )
        crash_complete_hits = fetch_hits(connection, CRASH_SOURCE_TEXT)
        crash_complete_score = score_for_id(crash_complete_hits, 4)
        if math.isclose(
            crash_immediate_score,
            crash_complete_score,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                'crash retry did not publish semantic completion: '
                f'{crash_immediate_score}'
            )
        gates[
            'semantic_completion_recovers_after_uncommitted_crash'
        ] = True

        crash_bmp_probe = fetch_page_native_exactness_probe(
            connection,
            CRASH_SOURCE_TEXT,
            disable_semantic_bmp=False,
        )
        crash_fallback_probe = fetch_page_native_exactness_probe(
            connection,
            CRASH_SOURCE_TEXT,
            disable_semantic_bmp=True,
        )
        assert_page_native_exactness_probe(
            crash_bmp_probe,
            crash_fallback_probe,
            'immediate crash recovery',
        )
        gates['semantic_bmp_survives_immediate_crash_recovery'] = True

        maintenance_guard = acquire_maintenance_lock(socket_dir, port)
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT ctid::text FROM convergent_sae.docs WHERE id = 4'
            )
            old_tid = str(cursor.fetchone()[0])
            cursor.execute(
                'SELECT ctid::text FROM convergent_sae.docs WHERE id = 1'
            )
            immutable_tid = str(cursor.fetchone()[0])
            cursor.execute(
                'UPDATE convergent_sae.docs '
                "SET body = %s, categories = ARRAY['database']::text[] "
                'WHERE id = 4',
                (UPDATE_TEXT,),
            )
            cursor.execute(
                'SELECT ctid::text FROM convergent_sae.docs WHERE id = 4'
            )
            new_tid = str(cursor.fetchone()[0])
        if old_tid == new_tid:
            raise AssertionError('indexed UPDATE did not create a new TID')

        update_pending_status = fetch_status(connection)
        assert_pending_lexical_status(update_pending_status)
        update_immediate_hits = fetch_hits(
            connection,
            UPDATE_TEXT,
            exact=True,
        )
        score_for_id(update_immediate_hits, 4)
        update_filtered_hits = fetch_structured_filtered_hits(
            connection,
            UPDATE_TEXT,
            {'categories': {'overlap': ['database']}},
            limit=2,
            exact=True,
        )
        score_for_id(update_filtered_hits, 4)
        stale_metadata_hits = fetch_structured_filtered_hits(
            connection,
            UPDATE_TEXT,
            {'categories': {'overlap': ['general']}},
            limit=2,
            exact=True,
        )
        if any(row_id == 4 for row_id, _score in stale_metadata_hits):
            raise AssertionError(
                'structured filter retained the pre-update metadata tuple'
            )
        shadowed_tid_hits = fetch_tid_filtered_hits(
            connection,
            INSERT_TEXT,
            [old_tid],
            limit=2,
        )
        replacement_tid_hits = fetch_tid_filtered_hits(
            connection,
            UPDATE_TEXT,
            [new_tid],
            limit=2,
            exact=True,
        )
        immutable_tid_hits = fetch_tid_filtered_hits(
            connection,
            INITIAL_ROWS[0][1],
            [immutable_tid],
            limit=2,
        )
        if shadowed_tid_hits:
            raise AssertionError(
                'linked L0 retained a shadowed immutable-root TID: '
                f'{shadowed_tid_hits}'
            )
        score_for_id(replacement_tid_hits, 4)
        score_for_id(immutable_tid_hits, 1)
        gates['update_replaces_tid_and_is_immediately_visible'] = True
        gates['structured_filters_follow_updated_mvcc_metadata'] = True
        gates['linked_l0_reuses_immutable_tid_directory_safely'] = True
        release_maintenance_lock(maintenance_guard)
        maintenance_guard = None

        update_phases, update_status = maintain_until_converged(
            connection,
            expected_docs=None,
        )
        update_complete_hits = fetch_hits(connection, UPDATE_TEXT)
        score_for_id(update_complete_hits, 4)
        update_settled_filtered_hits = fetch_structured_filtered_hits(
            connection,
            UPDATE_TEXT,
            {'categories': {'overlap': ['database']}},
            limit=2,
            exact=True,
            require_scope=True,
        )
        score_for_id(update_settled_filtered_hits, 4)
        gates['updated_version_semantics_converge'] = True
        gates['updated_scope_postings_republish_after_convergence'] = True

        maintenance_guard = acquire_maintenance_lock(socket_dir, port)
        with connection.cursor() as cursor:
            cursor.execute(
                'DELETE FROM convergent_sae.docs WHERE id = 3'
            )
        delete_immediate_hits = fetch_hits(
            connection,
            'obsolete archival semantic document',
        )
        if any(row_id == 3 for row_id, _score in delete_immediate_hits):
            raise AssertionError(
                'deleted document remained visible before VACUUM'
            )
        delete_filtered_hits = fetch_structured_filtered_hits(
            connection,
            'obsolete archival semantic document',
            {'id': {'eq': 3}},
            limit=1,
        )
        if delete_filtered_hits:
            raise AssertionError(
                'structured filter returned a deleted MVCC tuple'
            )
        gates['delete_is_hidden_by_mvcc_immediately'] = True
        gates['structured_filters_hide_deleted_mvcc_tuples'] = True

        vacuum_with_session_maintenance_lock(
            maintenance_guard,
            'convergent_sae.docs',
            index_cleanup=True,
        )
        delete_pending_status = fetch_status(connection)
        if int(delete_pending_status['generation']['delta']['records']) == 0:
            raise AssertionError(
                'VACUUM did not publish retirement debt: '
                f'{delete_pending_status}'
            )
        release_maintenance_lock(maintenance_guard)
        maintenance_guard = None
        delete_phases, delete_status = maintain_until_converged(
            connection,
            expected_docs=3,
        )
        if int(delete_status['generation']['sealed_docs']) != 3:
            raise AssertionError(
                f'retirement did not converge to 3 docs: {delete_status}'
            )
        delete_settled_filtered_hits = fetch_structured_filtered_hits(
            connection,
            'obsolete archival semantic document',
            {'id': {'eq': 3}},
            limit=1,
        )
        if delete_settled_filtered_hits:
            raise AssertionError(
                'structured filter returned a retired document after VACUUM'
            )
        fetch_structured_filtered_hits(
            connection,
            UPDATE_TEXT,
            {'categories': {'overlap': ['database']}},
            limit=2,
            exact=True,
            require_scope=True,
        )
        gates['vacuum_retirements_converge_without_rebuild'] = True
        gates['retired_scope_postings_republish_after_convergence'] = True

        (
            structural_touch,
            structural_observations,
            structural_status,
        ) = wait_for_background_structural_convergence(
            connection,
            delete_status,
        )
        gates['background_reclaims_and_compacts_structural_debt'] = True

        retired_slot_bmp_probe = fetch_page_native_exactness_probe(
            connection,
            UPDATE_TEXT,
            disable_semantic_bmp=False,
        )
        retired_slot_fallback_probe = fetch_page_native_exactness_probe(
            connection,
            UPDATE_TEXT,
            disable_semantic_bmp=True,
        )
        assert_page_native_exactness_probe(
            retired_slot_bmp_probe,
            retired_slot_fallback_probe,
            'interior document retirement',
        )
        gates['semantic_bmp_survives_interior_document_retirement'] = True

        completion_before_idle = semantic_completion(structural_status)
        telemetry_before_idle = completion_before_idle['telemetry']
        runtime_before_idle = fetch_runtime_status(connection)
        idle_maintenance = [maintain_once(connection) for _ in range(4)]
        idle_status = fetch_status(connection)
        telemetry_after_idle = semantic_completion(idle_status)['telemetry']
        runtime_after_idle = fetch_runtime_status(connection)
        if int(telemetry_after_idle['completed']) != int(
            telemetry_before_idle['completed']
        ):
            raise AssertionError(
                'ordinary maintenance published another semantic completion: '
                f"before={telemetry_before_idle['completed']}, "
                f"after={telemetry_after_idle['completed']}, "
                f'maintenance={idle_maintenance}'
            )
        for counter in ('requests', 'document_dispatches'):
            after_value = int(runtime_after_idle[counter])
            before_value = int(runtime_before_idle[counter])
            if after_value != before_value:
                raise AssertionError(
                    'ordinary maintenance re-encoded completed semantic '
                    f'documents: counter={counter}, '
                    f'before={before_value}, after={after_value}, '
                    f'maintenance={idle_maintenance}'
                )
        idle_maintenance_evidence = {
            'results': idle_maintenance,
            'semantic_attempts_before': telemetry_before_idle['attempts'],
            'semantic_attempts_after': telemetry_after_idle['attempts'],
            'runtime_requests_before': runtime_before_idle['requests'],
            'runtime_requests_after': runtime_after_idle['requests'],
            'document_dispatches_before': (
                runtime_before_idle['document_dispatches']
            ),
            'document_dispatches_after': (
                runtime_after_idle['document_dispatches']
            ),
        }
        gates['ordinary_maintenance_reuses_completed_semantic_postings'] = True

        restart_query = encode_query_once(connection, UPDATE_TEXT)
        before_restart_hot_hits = fetch_encoded_hits(
            connection,
            restart_query,
        )
        before_restart_trace = fetch_query_trace(connection)
        before_restart_hits = fetch_encoded_hits(
            connection,
            restart_query,
            exact=True,
        )
        before_restart_filtered_hits = fetch_encoded_hits(
            connection,
            restart_query,
            limit=2,
            filters={'categories': {'overlap': ['database']}},
            exact=True,
            require_scope=True,
        )
        before_restart_bmp_probe = fetch_page_native_exactness_probe(
            connection,
            UPDATE_TEXT,
            disable_semantic_bmp=False,
        )
        before_restart_fallback_probe = fetch_page_native_exactness_probe(
            connection,
            UPDATE_TEXT,
            disable_semantic_bmp=True,
        )
        connection.close()
        connection = None
        stop_cluster(pg_ctl, data_dir)
        started = False
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)

        restarted_status = fetch_status(connection)
        restarted_cold_hits = fetch_encoded_hits(connection, restart_query)
        restarted_cold_trace = fetch_query_trace(connection)
        restarted_hits = fetch_encoded_hits(
            connection,
            restart_query,
            exact=True,
        )
        restarted_filtered_hits = fetch_encoded_hits(
            connection,
            restart_query,
            limit=2,
            filters={'categories': {'overlap': ['database']}},
            exact=True,
            require_scope=True,
        )
        if not is_converged(restarted_status):
            raise AssertionError(
                f'restarted index is not converged: {restarted_status}'
            )
        if int(restarted_status['generation']['docs']) != 3:
            raise AssertionError(
                f'restarted index has wrong live count: {restarted_status}'
            )
        if len(before_restart_hits) != len(restarted_hits):
            raise AssertionError(
                'restart changed hit count: '
                f'{before_restart_hits} != {restarted_hits}'
            )
        for before, after in zip(before_restart_hits, restarted_hits):
            if (
                before[0] != after[0]
                or not math.isclose(
                    before[1],
                    after[1],
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                raise AssertionError(
                    'restart changed rows or scores: '
                    f'{before_restart_hits} != {restarted_hits}; '
                    f'before_trace={before_restart_trace}; '
                    f'cold_trace={restarted_cold_trace}'
                )
        gates['cold_restart_preserves_converged_rows_and_scores'] = True
        if not same_hit_ids(
            before_restart_hot_hits,
            restarted_cold_hits,
        ):
            raise AssertionError(
                'cold restart changed product ranking IDs: '
                f'{before_restart_hot_hits} != {restarted_cold_hits}; '
                f'before_trace={before_restart_trace}; '
                f'cold_trace={restarted_cold_trace}'
            )
        gates['cold_restart_is_immediately_queryable'] = True

        restarted_hot_hits = fetch_encoded_hits(connection, restart_query)
        restarted_hot_trace = fetch_query_trace(connection)
        restarted_route = restarted_hot_trace.get('query_route')
        restarted_runtime_state = wait_for_accelerator_preload_state(
            connection
        )
        if (
            restarted_route != 'accelerator_block_major'
            or not same_hit_ids(
                before_restart_hot_hits,
                restarted_hot_hits,
            )
        ):
            raise AssertionError(
                'restart did not restore the accelerator product route: '
                f'before={before_restart_hot_hits}, '
                f'after={restarted_hot_hits}; '
                f'before_trace={before_restart_trace}; '
                f'after_trace={restarted_hot_trace}'
            )
        gates['restart_restores_a_current_warm_product_route'] = True
        gates[
            'restart_publishes_accelerator_metadata_without_resident_fold'
        ] = True
        if before_restart_filtered_hits != restarted_filtered_hits:
            raise AssertionError(
                'restart changed filtered rows or scores: '
                f'{before_restart_filtered_hits} != '
                f'{restarted_filtered_hits}'
            )
        gates['durable_tid_lookup_survives_cold_restart'] = True

        restarted_bmp_probe = fetch_page_native_exactness_probe(
            connection,
            UPDATE_TEXT,
            disable_semantic_bmp=False,
        )
        restarted_fallback_probe = fetch_page_native_exactness_probe(
            connection,
            UPDATE_TEXT,
            disable_semantic_bmp=True,
        )
        assert_restart_exactness_probe(
            before_restart_bmp_probe,
            before_restart_fallback_probe,
            restarted_bmp_probe,
            restarted_fallback_probe,
        )
        gates['page_native_exact_route_survives_cold_restart'] = True

        structured_filter_repeatable_read = (
            run_structured_filter_repeatable_read_audit(
                connection,
                socket_dir,
                port,
            )
        )
        gates[
            'structured_filters_preserve_repeatable_read_metadata'
        ] = True

        same_transaction = run_same_transaction_audit(connection)
        gates['default_sae_same_transaction_visibility_is_exact'] = True
        gates['filter_scope_savepoint_and_abort_visibility_is_exact'] = True

        scope_plateau = run_scope_plateau_audit(connection)
        gates['filter_scope_tracks_the_current_source_manifest'] = True
        gates['filter_scope_storage_and_backend_rss_reach_a_plateau'] = True

        evidence = {
            'initial_status': initial_status,
            'semantic_bmp_probe': bmp_probe,
            'semantic_bmp_fallback_probe': fallback_probe,
            'filtered_bmp_membership_probe': filtered_bmp_probe,
            'tid_filter_trace': tid_filter_trace,
            'semantic_error_budget_probe': error_budget_probe,
            'post_semantic_error_budget_exact_probe': (
                post_budget_exact_probe
            ),
            'consistency_contract': consistency_contract,
            'multicolumn_contract': multicolumn_contract,
            'fold_compaction_preload': fold_compaction_preload,
            'semantic_alpha_contract': alpha_contract,
            'structured_query_trace': structured_query_trace,
            'reindex_maintenance': reindex_maintenance,
            'reindexed_status': reindexed_status,
            'reindexed_hits': reindexed_hits,
            'accelerator_status': accelerator_status,
            'accelerator_hits': accelerator_hits,
            'accelerator_exact_probe': accelerator_exact_probe,
            'accelerator_bounded_probe': accelerator_bounded_probe,
            'accelerator_error_budget_probe': accelerator_budget_probe,
            'insert_pending_status': insert_pending_status,
            'insert_immediate_hits': insert_immediate_hits,
            'insert_l0_bmp_probe': insert_l0_bmp_probe,
            'insert_l0_fallback_probe': insert_l0_fallback_probe,
            'accelerator_refresh_yields_to_active_ingress': {
                'result': active_rotation_result,
                'status': active_rotation_status,
                'runtime_before': runtime_before_active_refresh,
                'runtime_after': runtime_after_active_rotation,
            },
            'accelerator_fairness_phases': accelerator_fairness_phases,
            'accelerator_fairness_status': accelerator_fairness_status,
            'accelerator_budget_phases': accelerator_budget_phases,
            'accelerator_budget_status': accelerator_budget_status,
            'accelerator_budget_exact_probe': blocked_exact_probe,
            'insert_complete_hits': insert_complete_hits,
            'insert_touch': insert_touch,
            'insert_maintenance': insert_phases,
            'insert_status': insert_status,
            'single_atom_bmp_probes': single_atom_bmp_probes,
            'single_atom_fallback_probes': (
                single_atom_fallback_probes
            ),
            'race_prepare': race_prepare,
            'race_pending_status': race_pending_status,
            'race_pause': race_pause,
            'race_publish_result': race_publish_result,
            'race_maintenance': race_phases,
            'race_status': race_status,
            'race_current_hits': race_current_hits,
            'race_stale_hits': race_stale_hits,
            'race_current_body': race_current_body,
            'hot_prepare': hot_prepare,
            'hot_pending_status': hot_pending_status,
            'hot_pause': hot_pause,
            'hot_publish_result': hot_publish_result,
            'hot_source_tid': hot_source_tid,
            'hot_mutation': hot_mutation,
            'hot_maintenance': hot_phases,
            'hot_status': hot_status,
            'hot_immediate_hits': hot_immediate_hits,
            'hot_complete_hits': hot_complete_hits,
            'crash_prepare': crash_prepare,
            'crash_pending_status': crash_pending_status,
            'crash_pause': crash_pause,
            'crash_thread_result': crash_thread_result,
            'crash_restarted_status': crash_restarted_status,
            'crash_maintenance': crash_phases,
            'crash_status': crash_status,
            'crash_immediate_hits': crash_immediate_hits,
            'crash_complete_hits': crash_complete_hits,
            'crash_semantic_bmp_probe': crash_bmp_probe,
            'crash_semantic_bmp_fallback_probe': crash_fallback_probe,
            'update_pending_status': update_pending_status,
            'update_immediate_hits': update_immediate_hits,
            'update_complete_hits': update_complete_hits,
            'update_maintenance': update_phases,
            'update_status': update_status,
            'old_tid': old_tid,
            'new_tid': new_tid,
            'delete_pending_status': delete_pending_status,
            'delete_maintenance': delete_phases,
            'delete_status': delete_status,
            'retired_slot_semantic_bmp_probe': retired_slot_bmp_probe,
            'retired_slot_semantic_bmp_fallback_probe': (
                retired_slot_fallback_probe
            ),
            'structural_touch': structural_touch,
            'structural_observations': structural_observations,
            'structural_status': structural_status,
            'idle_maintenance': idle_maintenance_evidence,
            'before_restart_semantic_bmp_probe': (
                before_restart_bmp_probe
            ),
            'before_restart_query_trace': before_restart_trace,
            'restarted_cold_query_trace': restarted_cold_trace,
            'restarted_hot_query_trace': restarted_hot_trace,
            'restarted_runtime_state': restarted_runtime_state,
            'before_restart_semantic_bmp_fallback_probe': (
                before_restart_fallback_probe
            ),
            'restarted_status': restarted_status,
            'restarted_hits': restarted_hits,
            'restarted_filtered_hits': restarted_filtered_hits,
            'restarted_semantic_bmp_probe': restarted_bmp_probe,
            'restarted_semantic_bmp_fallback_probe': (
                restarted_fallback_probe
            ),
            'structured_filter_repeatable_read': (
                structured_filter_repeatable_read
            ),
            'same_transaction': same_transaction,
            'scope_plateau': scope_plateau,
            'failed_touch_retry': failed_touch_retry,
            'owner_bm25_diagnostic_contract': (
                owner_bm25_diagnostic_contract
            ),
            'semantic_impact_precision_contract': (
                impact_precision_contract
            ),
        }
    except Exception:
        if log_path.is_file():
            print(log_path.read_text(encoding='utf-8'), file=sys.stderr)
        raise
    finally:
        if maintenance_guard is not None:
            release_maintenance_lock(maintenance_guard)
        if connection is not None:
            connection.close()
        if started:
            stop_cluster(pg_ctl, data_dir)
        shutil.rmtree(root, ignore_errors=True)

    return {
        'api_version': 'evoke_index_v1',
        'route': 'convergent SAE lexical-first lifecycle',
        'model_path': str(model_path),
        'model_id': manifest['model_id'],
        'runtime_abi': manifest['runtime_abi'],
        'semantic_impact_precision': args.semantic_impact_precision,
        'semantic_alpha_mass': args.semantic_alpha_mass,
        'gates': gates,
        'passed_gates': sum(gates.values()),
        'total_gates': len(gates),
        'all_gates_passed': all(gates.values()),
        'evidence': evidence,
    }


def main() -> None:
    args = parse_args()
    report = run_smoke(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    if not report['all_gates_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
