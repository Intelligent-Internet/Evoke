#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import psycopg

from evoke_test_support import (
    create_short_socket_root,
    extension_control_root,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')
TEXT_ROWS = (
    (1, 'blue bird flies over the river'),
    (2, 'red fox sleeps beside a tree'),
    (3, 'blue fox and blue bird share a forest'),
    (4, 'postgresql inverted index supports fast lexical search'),
    (5, 'semantic retrieval and lexical retrieval cooperate'),
    (6, 'bird migration follows the river valley'),
)
PREDICATE_ROWS = (
    (1, ['blue', 'bird', 'flies', 'river']),
    (2, ['red', 'fox', 'tree']),
    (3, ['blue', 'fox', 'bird', 'forest']),
    (4, ['postgresql', 'inverted', 'index', 'lexical', 'search']),
    (5, ['semantic', 'retrieval', 'lexical', 'retrieval']),
    (6, ['bird', 'migration', 'river', 'valley']),
)
ID_ROWS = (
    (1, [0, 0, 1]),
    (2, [1, 2]),
    (3, [0, 2, 2]),
    (4, [3]),
)
HIGH_DF_ROWS = 513
TEXT_QUERIES = (
    'blue bird river',
    'lexical retrieval',
    'fox forest',
)
RAW_TEXT_QUERIES = (
    'bl*',
    'blue bl*',
    '"blue bird"',
    '+blue -fox',
    'blue AND NOT fox',
    'blue OR lexical',
    '(blue AND bird) OR semantic',
)
TOKEN_QUERIES = (
    ['blue', 'bird', 'river'],
    ['lexical', 'retrieval'],
)
ID_QUERIES = ([0, 2], [1, 3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate default v3 sealed-segment parity in isolated PG18.'
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
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        return int(listener.getsockname()[1])


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
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


def pg_config_value(pg_bin: Path, option: str) -> Path:
    result = run([str(pg_bin / 'pg_config'), option])
    return Path(result.stdout.strip()).resolve()


def quote_config(value: Path) -> str:
    return str(value).replace("'", "''")


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
        handle.write("evoke.shared_runtime_size = '64MB'\n")
        handle.write("evoke.workspace_cache_bytes = '0'\n")
        handle.write("listen_addresses = ''\n")
        handle.write(
            f"unix_socket_directories = '{quote_config(socket_dir)}'\n"
        )
        handle.write(f'port = {port}\n')
        handle.write('max_worker_processes = 16\n')
        handle.write('max_prepared_transactions = 8\n')
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
    command = [
        str(pg_ctl),
        '-D',
        str(data_dir),
        'stop',
        '-m',
        'fast',
        '-w',
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        result = None
    if result is not None and result.returncode == 0:
        return

    # A failed test backend must not leave an orphan cluster or block the gate.
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
        check=False,
    )


def connect(socket_dir: Path, port: int) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname='postgres',
        user='postgres',
        host=str(socket_dir),
        port=port,
        autocommit=True,
    )


def quiesce_background_maintenance(
    connection: psycopg.Connection[Any],
) -> list[dict[str, int]]:
    with connection.cursor() as cursor:
        cursor.execute(
            "ALTER SYSTEM SET evoke.maintenance_worker_limit = '0'"
        )
        cursor.execute('SELECT pg_reload_conf()')
        if cursor.fetchone()[0] is not True:
            raise AssertionError('could not reload maintenance worker limit')

    observations: list[dict[str, int]] = []
    deadline = time.monotonic() + 15.0
    while True:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT '
                "current_setting('evoke.maintenance_worker_limit')::int, "
                'count(*) FILTER ('
                "WHERE backend_type = 'evoke background') "
                'FROM pg_stat_activity'
            )
            worker_limit, active_workers = cursor.fetchone()
        observation = {
            'worker_limit': int(worker_limit),
            'active_workers': int(active_workers),
        }
        observations.append(observation)
        if worker_limit == 0 and active_workers == 0:
            return observations
        if time.monotonic() >= deadline:
            raise AssertionError(
                'background maintenance did not quiesce: '
                f'{observations}'
            )
        time.sleep(0.05)


def create_page_native_reference_index(cursor: Any, statement: str) -> None:
    cursor.execute(statement)


def restart_preserves_l0_frontier(
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    before_delta = before['generation']['delta']
    after_delta = after['generation']['delta']
    before_active = before_delta['active']
    before_pending = before_delta['pending']
    after_active = after_delta['active']
    after_pending = after_delta['pending']
    aggregate_fields = (
        'bytes',
        'pages',
        'records',
        'start_block',
    )
    segment_fields = (
        'bytes',
        'pages',
        'records',
        'segment_id',
    )

    aggregate_unchanged = all(
        after_delta[field] == before_delta[field]
        for field in aggregate_fields
    )
    frontiers_unchanged = (
        all(
            after_active[field] == before_active[field]
            for field in segment_fields
        )
        and all(
            after_pending[field] == before_pending[field]
            for field in segment_fields
        )
    )
    active_rotated_to_pending = (
        before_pending['records'] == 0
        and all(
            after_pending[field] == before_active[field]
            for field in segment_fields
        )
        and after_active['bytes'] == 0
        and after_active['pages'] == 0
        and after_active['records'] == 0
        and after_active['segment_id'] != 0
        and after_active['segment_id'] != before_active['segment_id']
    )

    return (
        aggregate_unchanged
        and (frontiers_unchanged or active_rotated_to_pending)
    )


def l0_accepts_records_after_optional_pending_seal(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    new_upserts: int,
    new_retirements: int,
) -> bool:
    before_generation = before['generation']
    after_generation = after['generation']
    before_delta = before_generation['delta']
    after_delta = after_generation['delta']
    new_records = new_upserts + new_retirements
    expected_records = before_delta['records'] + new_records
    expected_upserts = before_delta['upserts'] + new_upserts
    expected_retirements = (
        before_delta['retirements'] + new_retirements
    )
    consumed_records = expected_records - after_delta['records']
    consumed_upserts = expected_upserts - after_delta['upserts']
    consumed_retirements = (
        expected_retirements - after_delta['retirements']
    )
    frontier_counts_are_exact = (
        after_delta['records']
        == (
            after_delta['active']['records']
            + after_delta['pending']['records']
        )
        and after_delta['records']
        == after_delta['upserts'] + after_delta['retirements']
    )
    no_publication = (
        consumed_records == 0
        and consumed_upserts == 0
        and consumed_retirements == 0
    )
    records_were_published = (
        consumed_records > 0
        and consumed_upserts >= 0
        and consumed_retirements >= 0
        and consumed_records
        == consumed_upserts + consumed_retirements
        and after_generation['primary']['published_block_high_watermark']
        > before_generation['primary']['published_block_high_watermark']
    )

    return (
        frontier_counts_are_exact
        and (no_publication or records_were_published)
    )


def setup(connection: psycopg.Connection[Any]) -> None:
    with connection.cursor() as cursor:
        cursor.execute('CREATE EXTENSION evoke')
        cursor.execute('CREATE SCHEMA parity')
        cursor.execute(
            'CREATE FUNCTION parity.test_term_cow_pages_write(regclass) '
            'RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_term_cow_pages_write' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_term_cow_pages_read('
            'regclass, int4, int4, text, text, text, text, int4, int4) '
            'RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_term_cow_pages_read' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_term_fold_publish(regclass) '
            'RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_term_fold_publish' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_term_fold_state(regclass, int4) '
            'RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_term_fold_state' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_unpublished_tail_append(regclass) '
            'RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_unpublished_tail_append' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_term_plan_stats('
            'regclass, int4) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_term_plan_stats' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_term_page_native('
            'regclass, int4) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_term_page_native' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_term_page_native_topk('
            'regclass, int4, int4) RETURNS jsonb '
            "AS '$libdir/evoke', "
            "'evoke_test_query_term_page_native_topk' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_page_native_topk('
            'regclass, int4[], int4) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_page_native_topk' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_page_native_topk('
            'regclass, int4[], real[], int4) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_page_native_topk' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_page_native_topk_filtered('
            'regclass, int4[], real[], int4, boolean, int4[]) '
            'RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_page_native_topk' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_block_cost('
            'regclass, int4[], real[], int4) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_block_cost' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_forward_bound_cost('
            'regclass, int4[], real[], int4[], int4) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_forward_bound_cost' "
            'LANGUAGE C CALLED ON NULL INPUT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_term_suffix_ceiling('
            'regclass, int4[], real[]) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_term_suffix_ceiling' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_page_native_l0_topk('
            'regclass, int4[], int4) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_page_native_l0_topk' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_page_native_l0_topk('
            'regclass, text[], int4) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_page_native_l0_topk' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_context_page_native('
            'regclass) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_context_page_native' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_query_lexicon_page_native('
            'regclass, text) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_query_lexicon_page_native' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE FUNCTION parity.test_l0_stream_page_native('
            'regclass) RETURNS jsonb '
            "AS '$libdir/evoke', 'evoke_test_l0_stream_page_native' "
            'LANGUAGE C STRICT'
        )
        cursor.execute(
            'CREATE TABLE parity.docs_reference ('
            'id int PRIMARY KEY, body text NOT NULL)'
        )
        cursor.execute(
            'CREATE TABLE parity.docs_v3 ('
            'id int PRIMARY KEY, body text NOT NULL)'
        )
        cursor.executemany(
            'INSERT INTO parity.docs_reference VALUES (%s, %s)',
            TEXT_ROWS,
        )
        cursor.executemany(
            'INSERT INTO parity.docs_v3 VALUES (%s, %s)',
            TEXT_ROWS,
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX docs_reference_idx ON parity.docs_reference '
            'USING evoke (body) WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX docs_v3_idx ON parity.docs_v3 '
            'USING evoke (body) WITH (sae=false, consistency=realtime)'
        )

        cursor.execute(
            'CREATE TABLE parity.predicate_reference ('
            'id int PRIMARY KEY, body text[] NOT NULL)'
        )
        cursor.execute(
            'CREATE TABLE parity.predicate_v3 ('
            'id int PRIMARY KEY, body text[] NOT NULL)'
        )
        cursor.executemany(
            'INSERT INTO parity.predicate_reference VALUES (%s, %s)',
            PREDICATE_ROWS,
        )
        cursor.executemany(
            'INSERT INTO parity.predicate_v3 VALUES (%s, %s)',
            PREDICATE_ROWS,
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX predicate_reference_idx ON parity.predicate_reference '
            'USING evoke (body) WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX predicate_v3_idx ON parity.predicate_v3 '
            'USING evoke (body) WITH (sae=false, consistency=realtime)'
        )

        cursor.execute(
            'CREATE TABLE parity.ids_reference ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'CREATE TABLE parity.ids_v3 ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.executemany(
            'INSERT INTO parity.ids_reference VALUES (%s, %s)',
            ID_ROWS,
        )
        cursor.executemany(
            'INSERT INTO parity.ids_v3 VALUES (%s, %s)',
            ID_ROWS,
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX ids_reference_idx ON parity.ids_reference '
            'USING evoke (tokens) WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX ids_v3_idx ON parity.ids_v3 '
            'USING evoke (tokens) WITH (sae=false, consistency=realtime)'
        )

        cursor.execute(
            'CREATE TABLE parity.high_df_v3 ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.high_df_v3 '
            'SELECT gs, ARRAY[0, 1000 + (gs %% 4), gs + 2000]::int4[] '
            'FROM generate_series(1, %s) gs',
            (HIGH_DF_ROWS,),
        )
        cursor.execute(
            'CREATE INDEX high_df_v3_idx ON parity.high_df_v3 '
            'USING evoke (tokens) WITH (sae=false, consistency=realtime)'
        )

        cursor.execute(
            'CREATE TABLE parity.filtered_v3 ('
            'id int PRIMARY KEY, tokens text[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.filtered_v3 '
            "SELECT gs, CASE WHEN gs IN (257, 513) "
            "THEN ARRAY['rank', 'accept']::text[] "
            "ELSE ARRAY['rank']::text[] END "
            'FROM generate_series(1, %s) gs',
            (HIGH_DF_ROWS,),
        )
        cursor.execute(
            'CREATE INDEX filtered_v3_idx ON parity.filtered_v3 '
            'USING evoke (tokens) WITH (sae=false, consistency=realtime)'
        )

        cursor.execute(
            'CREATE TABLE parity.empty_v3 ('
            'id int PRIMARY KEY, body text NOT NULL)'
        )
        cursor.execute(
            'CREATE INDEX empty_v3_idx ON parity.empty_v3 '
            'USING evoke (body) WITH (sae=false, consistency=realtime)'
        )

        cursor.execute(
            'CREATE TABLE parity.tail_v3 ('
            'id int PRIMARY KEY, body text NOT NULL)'
        )
        cursor.executemany(
            'INSERT INTO parity.tail_v3 VALUES (%s, %s)',
            TEXT_ROWS,
        )
        cursor.execute(
            'CREATE INDEX tail_v3_idx ON parity.tail_v3 '
            'USING evoke (body) WITH (sae=false, consistency=realtime)'
        )


def fetch_search(
    connection: psycopg.Connection[Any],
    index_name: str,
    query: str,
) -> list[tuple[int, float]]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT doc_id, score::float8 '
            'FROM evoke_query(%s::regclass, %s, 100)',
            (index_name, query),
        )
        return [(int(row[0]), float(row[1])) for row in cursor.fetchall()]


def fetch_tokens(
    connection: psycopg.Connection[Any],
    index_name: str,
    query: list[str],
) -> list[tuple[int, float]]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT doc_id, score::float8 '
            'FROM evoke_query_tokens(%s::regclass, %s::text[], 100)',
            (index_name, query),
        )
        return [(int(row[0]), float(row[1])) for row in cursor.fetchall()]


def fetch_ids(
    connection: psycopg.Connection[Any],
    index_name: str,
    query: list[int],
) -> list[tuple[int, float]]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT doc_id, score::float8 '
            'FROM evoke_query_ids(%s::regclass, %s::int4[], 100)',
            (index_name, query),
        )
        return [(int(row[0]), float(row[1])) for row in cursor.fetchall()]


def fetch_ids_by_tid(
    connection: psycopg.Connection[Any],
    index_name: str,
    query: list[int],
) -> list[tuple[str, float]]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT ctid::text, score::float8 '
            'FROM evoke_query_ids(%s::regclass, %s::int4[], 100)',
            (index_name, query),
        )
        return [(str(row[0]), float(row[1])) for row in cursor.fetchall()]


def fetch_ordered(
    connection: psycopg.Connection[Any],
    table_name: str,
    index_name: str,
    query: str,
) -> list[tuple[int, float]]:
    statement = (
        f'SELECT id, body <=> '
        f'evoke_order_tokens(%s::regclass, %s) AS distance '
        f'FROM {table_name} '
        f'ORDER BY body <=> evoke_order_tokens(%s::regclass, %s) '
        f'ASC LIMIT 100'
    )
    with connection.cursor() as cursor:
        cursor.execute(
            statement,
            (index_name, query, index_name, query),
        )
        return [(int(row[0]), float(row[1])) for row in cursor.fetchall()]


def fetch_ids_ordered(
    connection: psycopg.Connection[Any],
    table_name: str,
    query: list[int],
    *,
    limit: int = 100,
    force_index: bool = False,
) -> list[tuple[int, float]]:
    statement = (
        f'SELECT id, tokens <=> %s::int4[] AS distance '
        f'FROM {table_name} '
        f'ORDER BY tokens <=> %s::int4[] ASC LIMIT %s'
    )
    with connection.cursor() as cursor:
        if force_index:
            cursor.execute('SET enable_seqscan = false')
        try:
            cursor.execute(statement, (query, query, limit))
            return [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        finally:
            if force_index:
                cursor.execute('RESET enable_seqscan')


def fetch_filtered_ordered(
    connection: psycopg.Connection[Any],
    table_name: str,
    index_name: str,
    filter_query: str,
    order_query: str,
    *,
    limit: int = 100,
    force_index: bool = False,
) -> list[tuple[int, float]]:
    statement = (
        f'SELECT id, body <=> '
        f'evoke_order_tokens(%s::regclass, %s) AS distance '
        f'FROM {table_name} WHERE body @@ %s '
        f'ORDER BY body <=> evoke_order_tokens(%s::regclass, %s) '
        f'ASC LIMIT %s'
    )
    with connection.cursor() as cursor:
        if force_index:
            cursor.execute('SET enable_seqscan = false')
            cursor.execute('SET enable_bitmapscan = false')
        try:
            cursor.execute(
                statement,
                (
                    index_name,
                    order_query,
                    filter_query,
                    index_name,
                    order_query,
                    limit,
                ),
            )
            return [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        finally:
            if force_index:
                cursor.execute('RESET enable_bitmapscan')
                cursor.execute('RESET enable_seqscan')


def fetch_filtered_tokens_ordered(
    connection: psycopg.Connection[Any],
    table_name: str,
    filter_query: str,
    order_query: list[str],
    *,
    limit: int,
) -> list[tuple[int, float]]:
    statement = (
        f'SELECT id, tokens <=> %s::text[] AS distance '
        f'FROM {table_name} WHERE tokens @@ %s '
        f'ORDER BY tokens <=> %s::text[] ASC LIMIT %s'
    )
    with connection.cursor() as cursor:
        cursor.execute('SET enable_seqscan = false')
        cursor.execute('SET enable_bitmapscan = false')
        try:
            cursor.execute(
                statement,
                (order_query, filter_query, order_query, limit),
            )
            return [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]
        finally:
            cursor.execute('RESET enable_bitmapscan')
            cursor.execute('RESET enable_seqscan')


def fetch_predicate_ids(
    connection: psycopg.Connection[Any],
    table_name: str,
    query: str,
    *,
    force_index: bool = False,
) -> list[int]:
    statement = f'SELECT id FROM {table_name} WHERE body @@ %s'
    with connection.cursor() as cursor:
        if force_index:
            cursor.execute('SET enable_seqscan = false')
            cursor.execute('SET enable_bitmapscan = false')
        try:
            cursor.execute(statement, (query,))
            return sorted(int(row[0]) for row in cursor.fetchall())
        finally:
            if force_index:
                cursor.execute('RESET enable_bitmapscan')
                cursor.execute('RESET enable_seqscan')


def fetch_heap_predicate_ids(
    connection: psycopg.Connection[Any],
    table_name: str,
    query: str,
) -> list[int]:
    statement = (
        f'SELECT id FROM {table_name} WHERE body @@ %s ORDER BY id'
    )
    with connection.cursor() as cursor:
        cursor.execute('SET enable_indexscan = false')
        cursor.execute('SET enable_bitmapscan = false')
        try:
            cursor.execute(statement, (query,))
            return [int(row[0]) for row in cursor.fetchall()]
        finally:
            cursor.execute('RESET enable_bitmapscan')
            cursor.execute('RESET enable_indexscan')


def fetch_status(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_status(%s::regclass), '
            'evoke_index_generation_audit_internal(%s::regclass)',
            (index_name, index_name),
        )
        row = cursor.fetchone()
    if (
        row is None
        or not isinstance(row[0], dict)
        or not isinstance(row[1], dict)
    ):
        raise AssertionError(f'invalid status for {index_name}')
    status = row[0]
    readiness_generation = status.get('generation')
    if not isinstance(readiness_generation, dict):
        raise AssertionError(f'invalid readiness generation for {index_name}')
    status['generation'] = readiness_generation | row[1]
    return status


def write_term_cow_probe(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_term_cow_pages_write(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid COW page probe for {index_name}')
    return row[0]


def read_term_cow_probe(
    connection: psycopg.Connection[Any],
    index_name: str,
    root: dict[str, Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_term_cow_pages_read('
            '%s::regclass, %s, %s, %s, %s, %s, %s, %s, %s)',
            (
                index_name,
                root['start_block'],
                root['page_count'],
                str(root['object_id']),
                str(root['owner_manifest_id']),
                str(root['object_bytes']),
                str(root['blob_checksum']),
                root['vocab_size'],
                root['term_id'],
            ),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid COW page lookup for {index_name}')
    return row[0]


def publish_term_fold_probe(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_term_fold_publish(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid term fold probe for {index_name}')
    return row[0]


def fetch_term_fold_state(
    connection: psycopg.Connection[Any],
    index_name: str,
    term_id: int,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_term_fold_state(%s::regclass, %s)',
            (index_name, term_id),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid term fold state for {index_name}')
    return row[0]


def append_unpublished_tail_probe(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_unpublished_tail_append(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid unpublished-tail probe for {index_name}'
        )
    return row[0]


def fetch_page_native_term(
    connection: psycopg.Connection[Any],
    index_name: str,
    term_id: int,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_query_term_page_native('
            '%s::regclass, %s)',
            (index_name, term_id),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid page-native term probe for {index_name}'
        )
    return row[0]


def fetch_query_term_plan_stats(
    connection: psycopg.Connection[Any],
    index_name: str,
    term_id: int,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_query_term_plan_stats('
            '%s::regclass, %s)',
            (index_name, term_id),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid query-term plan stats for {index_name}'
        )
    return row[0]


def fetch_page_native_context(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_query_context_page_native('
            '%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid page-native query context for {index_name}'
        )
    return row[0]


def fetch_page_native_lexicon(
    connection: psycopg.Connection[Any],
    index_name: str,
    term: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_query_lexicon_page_native('
            '%s::regclass, %s)',
            (index_name, term),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid page-native lexicon probe for {index_name}:{term}'
        )
    return row[0]


def fetch_page_native_l0_stream(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_l0_stream_page_native('
            '%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid page-native L0 stream for {index_name}'
        )
    return row[0]


def fetch_page_native_topk(
    connection: psycopg.Connection[Any],
    index_name: str,
    term_id: int,
    k: int,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_query_term_page_native_topk('
            '%s::regclass, %s, %s)',
            (index_name, term_id, k),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid page-native top-k probe for {index_name}'
        )
    return row[0]


def fetch_page_native_query_topk(
    connection: psycopg.Connection[Any],
    index_name: str,
    query_ids: list[int],
    k: int,
    query_weights: list[float] | None = None,
    allowed_doc_ids: list[int] | None = None,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        if allowed_doc_ids is not None:
            weights = query_weights or [1.0] * len(query_ids)
            cursor.execute(
                'SELECT parity.test_query_page_native_topk_filtered('
                '%s::regclass, %s::int4[], %s::real[], %s, true, '
                '%s::int4[])',
                (index_name, query_ids, weights, k, allowed_doc_ids),
            )
        elif query_weights is None:
            cursor.execute(
                'SELECT parity.test_query_page_native_topk('
                '%s::regclass, %s::int4[], %s)',
                (index_name, query_ids, k),
            )
        else:
            cursor.execute(
                'SELECT parity.test_query_page_native_topk('
                '%s::regclass, %s::int4[], %s::real[], %s)',
                (index_name, query_ids, query_weights, k),
            )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid page-native query probe for {index_name}'
        )
    return row[0]


def fetch_page_native_block_cost(
    connection: psycopg.Connection[Any],
    index_name: str,
    query_ids: list[int],
    query_weights: list[float],
    k: int,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_query_block_cost('
            '%s::regclass, %s::int4[], %s::real[], %s)',
            (index_name, query_ids, query_weights, k),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid page-native block-cost probe for {index_name}'
        )
    return row[0]


def fetch_page_native_l0_query_topk(
    connection: psycopg.Connection[Any],
    index_name: str,
    query_ids: list[int],
    k: int,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_query_page_native_l0_topk('
            '%s::regclass, %s::int4[], %s)',
            (index_name, query_ids, k),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid page-native L0 query probe for {index_name}'
        )
    return row[0]


def fetch_page_native_text_l0_query_topk(
    connection: psycopg.Connection[Any],
    index_name: str,
    query_tokens: list[str],
    k: int,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT parity.test_query_page_native_l0_topk('
            '%s::regclass, %s::text[], %s)',
            (index_name, query_tokens, k),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(
            f'invalid page-native text L0 query probe for {index_name}'
        )
    return row[0]


def term_cow_query_contract_identity(
    probe: dict[str, Any],
) -> tuple[int, int, str, str, str, str]:
    return (
        int(probe['query_contract_start_block']),
        int(probe['query_contract_page_count']),
        str(probe['query_contract_object_id']),
        str(probe['query_contract_owner_manifest_id']),
        str(probe['query_contract_object_bytes']),
        str(probe['query_contract_blob_checksum']),
    )


def term_cow_catalog_summary(
    probe: dict[str, Any],
) -> tuple[int, str, int, int, int, str, str, str, str]:
    return (
        int(probe['lexical_catalog_count']),
        str(probe['lexical_catalog_total_bytes']),
        int(probe['latest_lexical_catalog_term_count']),
        int(probe['latest_lexical_catalog_start_block']),
        int(probe['latest_lexical_catalog_page_count']),
        str(probe['latest_lexical_catalog_object_id']),
        str(probe['latest_lexical_catalog_owner_manifest_id']),
        str(probe['latest_lexical_catalog_object_bytes']),
        str(probe['latest_lexical_catalog_blob_checksum']),
    )


def fetch_generation_cache_state(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_runtime_state(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], str):
        raise AssertionError(f'invalid cache state for {index_name}')
    return row[0]


def try_maintain(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_try_maintain(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], str):
        raise AssertionError(f'invalid maintenance result for {index_name}')
    return row[0]


def try_maintain_after_lock_contention(
    connection: psycopg.Connection[Any],
    index_name: str,
    *,
    timeout_seconds: float = 2.0,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    deadline = time.monotonic() + timeout_seconds
    attempts: list[dict[str, str]] = []
    while True:
        fields = maintenance_result_fields(
            try_maintain(connection, index_name)
        )
        attempts.append(fields)
        if fields.get('reason') != 'lock_busy':
            return fields, attempts
        if time.monotonic() >= deadline:
            return fields, attempts
        time.sleep(0.01)


def acquire_maintenance_guard(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_try_maintenance_lock(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None or row[0] is not True:
        raise AssertionError(
            f'could not acquire maintenance guard for {index_name}'
        )


def maintenance_guard_is_held(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_maintenance_lock_held(%s::regclass)',
            (index_name,),
        )
        row = cursor.fetchone()
    return row is not None and row[0] is True


def release_maintenance_guard(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_maintenance_unlock(%s::regclass)',
            (index_name,),
        )


def maintenance_result_fields(result: str) -> dict[str, str]:
    match = re.fullmatch(r'evoke_maintenance_result\((.*)\)', result)
    if match is None:
        raise AssertionError(f'invalid maintenance result: {result}')
    fields: dict[str, str] = {}
    for item in match.group(1).split(', '):
        name, separator, value = item.partition('=')
        if not separator or not name or not value:
            raise AssertionError(f'invalid maintenance field: {item}')
        fields[name] = value
    return fields


def exercise_unpublished_tail_cleanup(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    index_name = 'parity.tail_v3_idx'
    baseline_hits = fetch_search(
        connection,
        index_name,
        TEXT_QUERIES[0],
    )
    baseline_ordered_hits = fetch_ordered(
        connection,
        'parity.tail_v3',
        index_name,
        TEXT_QUERIES[0],
    )
    baseline_status = fetch_status(connection, index_name)
    baseline_primary = baseline_status['generation']['primary']
    if (
        baseline_primary['trailing_unpublished_blocks'] != 0
        or baseline_primary['physical_blocks']
        != baseline_primary['published_block_high_watermark']
        or baseline_primary['bytes']
        != baseline_primary['reachable_blocks'] * 8192
        or baseline_primary['payload_bytes'] != baseline_primary['bytes']
    ):
        raise AssertionError(
            'fresh tail-cleanup index has invalid page accounting: '
            f'{baseline_status}'
        )

    probe = append_unpublished_tail_probe(connection, index_name)
    if (
        not probe.get('range_verified')
        or probe.get('range_bytes') != 512
    ):
        raise AssertionError(
            f'checked page-range probe failed: {probe}'
        )
    dirty_status = fetch_status(connection, index_name)
    dirty_primary = dirty_status['generation']['primary']
    if (
        dirty_primary['trailing_unpublished_blocks'] <= 0
        or dirty_primary['physical_blocks']
        != (
            dirty_primary['published_block_high_watermark']
            + dirty_primary['trailing_unpublished_blocks']
        )
        or dirty_primary['reachable_blocks']
        + dirty_primary['interior_unreachable_blocks']
        != dirty_primary['published_block_high_watermark']
    ):
        raise AssertionError(
            'unpublished tail was not classified exactly: '
            f'{dirty_status}, probe={probe}'
        )

    reader = psycopg.connect(connection.info.dsn, autocommit=False)
    reader_pid = reader.info.backend_pid
    try:
        with reader.cursor() as cursor:
            cursor.execute('SET LOCAL enable_seqscan = false')
        reader_hits = fetch_ordered(
            reader,
            'parity.tail_v3',
            index_name,
            TEXT_QUERIES[0],
        )
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT mode '
                'FROM pg_locks '
                'WHERE pid = %s '
                'AND relation = %s::regclass '
                'AND granted '
                'ORDER BY mode',
                (reader_pid, index_name),
            )
            reader_locks = [str(row[0]) for row in cursor.fetchall()]
        if 'AccessShareLock' not in reader_locks:
            raise AssertionError(
                'native reader did not retain an index lock: '
                f'{reader_locks}'
            )

        busy_result = try_maintain(connection, index_name)
        busy_fields = maintenance_result_fields(busy_result)
        if (
            busy_fields.get('maintained') != 'false'
            or busy_fields.get('reason') != 'reader_fence_busy'
            or busy_fields.get('mode') != 'segment_tail_cleanup'
            or int(busy_fields.get('truncated_blocks', '-1')) != 0
        ):
            raise AssertionError(
                'tail cleanup did not yield to an old reader: '
                f'{busy_result}'
            )
        busy_status = fetch_status(connection, index_name)
        if (
            busy_status['generation']['primary'][
                'trailing_unpublished_blocks'
            ]
            != dirty_primary['trailing_unpublished_blocks']
        ):
            raise AssertionError(
                'busy cleanup changed the unpublished tail: '
                f'{busy_status}'
            )
    finally:
        reader.rollback()
        reader.close()

    cleanup_result = try_maintain(connection, index_name)
    cleanup_fields = maintenance_result_fields(cleanup_result)
    cleaned_status = fetch_status(connection, index_name)
    cleaned_primary = cleaned_status['generation']['primary']
    cleaned_hits = fetch_search(
        connection,
        index_name,
        TEXT_QUERIES[0],
    )
    assert_rows_close(
        baseline_hits,
        cleaned_hits,
        label='unpublished tail cleanup',
    )
    if (
        cleanup_fields.get('maintained') != 'true'
        or cleanup_fields.get('reason')
        != 'unpublished_tail_truncated'
        or cleanup_fields.get('mode') != 'segment_tail_cleanup'
        or int(cleanup_fields.get('truncated_blocks', '0'))
        != dirty_primary['trailing_unpublished_blocks']
        or cleaned_primary['trailing_unpublished_blocks'] != 0
        or cleaned_primary['physical_blocks']
        != cleaned_primary['published_block_high_watermark']
        or cleaned_primary['reachable_blocks']
        != dirty_primary['reachable_blocks']
        or cleaned_primary['interior_unreachable_blocks']
        != dirty_primary['interior_unreachable_blocks']
    ):
        raise AssertionError(
            'unpublished tail cleanup was not exact: '
            f'result={cleanup_result}, dirty={dirty_status}, '
            f'cleaned={cleaned_status}'
        )
    assert_rows_close(
        sorted(baseline_ordered_hits),
        sorted(reader_hits),
        label='old reader during unpublished tail cleanup',
    )
    return {
        'probe': probe,
        'baseline_status': baseline_status,
        'dirty_status': dirty_status,
        'busy_result': busy_result,
        'busy_status': busy_status,
        'cleanup_result': cleanup_result,
        'cleaned_status': cleaned_status,
        'hits': cleaned_hits,
    }


def exercise_query_safe_retirement(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    index_name = 'parity.reuse_v3_idx'
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.reuse_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.reuse_docs VALUES (1, ARRAY[1, 2])'
        )
        cursor.execute(
            'CREATE INDEX reuse_v3_idx '
            'ON parity.reuse_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )

    def append_row(row_id: int) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                "SET evoke.test_convergent_l0_rotation_records = '1'"
            )
            cursor.execute(
                'INSERT INTO parity.reuse_docs VALUES (%s, ARRAY[1, %s])',
                (row_id, row_id + 10),
            )
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )

    def rotate_and_seal(
        row_id: int,
    ) -> tuple[list[str], dict[str, str]]:
        maintenance: list[str] = []
        for _ in range(12):
            result = try_maintain(connection, index_name)
            maintenance.append(result)
            fields = maintenance_result_fields(result)
            if (
                fields.get('mode') == 'segment_seal'
                and fields.get('maintained') == 'true'
            ):
                return maintenance, fields
            if fields.get('maintained') == 'true':
                continue
            if fields.get('reason') in {'lock_busy', 'xid_horizon'}:
                continue
            raise AssertionError(
                'page-reuse setup stopped before sealing: '
                f'{maintenance}'
            )
        raise AssertionError(
            f'page-reuse setup did not seal row {row_id}: {maintenance}'
        )

    def compact_until_published(
    ) -> tuple[list[str], dict[str, str]]:
        maintenance: list[str] = []

        for _ in range(16):
            result = try_maintain(connection, index_name)
            maintenance.append(result)
            fields = maintenance_result_fields(result)
            if (
                fields.get('mode') == 'segment_compaction'
                and fields.get('maintained') == 'true'
            ):
                return maintenance, fields
            if fields.get('maintained') == 'true':
                continue
            if fields.get('reason') in {'lock_busy', 'xid_horizon'}:
                continue
            raise AssertionError(
                'page-reuse setup stopped before compaction: '
                f'{maintenance}'
            )
        raise AssertionError(
            f'page-reuse setup did not compact: {maintenance}'
        )

    append_row(2)
    first_maintenance, first_seal = rotate_and_seal(2)
    first_status = fetch_status(connection, index_name)
    first_primary = first_status['generation']['primary']
    if (
        first_primary['retired_hint_blocks']
        != first_primary['interior_unreachable_blocks']
    ):
        raise AssertionError(
            'first successful COW transition did not retire every '
            f'unreachable block: {first_status}'
        )
    append_row(3)
    second_maintenance, second_seal = rotate_and_seal(3)
    append_row(4)
    third_maintenance, third_seal = rotate_and_seal(4)
    append_row(5)
    fourth_maintenance, fourth_seal = rotate_and_seal(5)

    reader = psycopg.connect(connection.info.dsn, autocommit=False)
    reader_pid = reader.info.backend_pid
    try:
        with reader.cursor() as cursor:
            cursor.execute('SET LOCAL enable_seqscan = false')
        reader_hits = fetch_ids_ordered(
            reader,
            'parity.reuse_docs',
            [1],
        )
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT mode FROM pg_locks '
                'WHERE pid = %s AND relation = %s::regclass '
                'AND granted ORDER BY mode',
                (reader_pid, index_name),
            )
            reader_locks = [str(row[0]) for row in cursor.fetchall()]
        if 'AccessShareLock' not in reader_locks:
            raise AssertionError(
                'page-reuse reader did not retain its index lock: '
                f'{reader_locks}'
            )

        reader_compaction_maintenance, reader_compaction = (
            compact_until_published()
        )
        reader_compaction_status = fetch_status(connection, index_name)
        if (
            int(second_seal.get('reused_blocks', '-1')) != 0
            or int(third_seal.get('reused_blocks', '-1')) != 0
            or int(fourth_seal.get('reused_blocks', '-1')) != 0
            or int(reader_compaction.get('reused_blocks', '-1')) != 0
            or first_status['generation']['primary'][
                'interior_unreachable_blocks'
            ]
            <= 0
            or reader_compaction_status['generation']['primary'][
                'interior_unreachable_blocks'
            ]
            <= 0
        ):
            raise AssertionError(
                'page reuse crossed an old-reader fence: '
                f'first={first_seal}, second={second_seal}, '
                f'third={third_seal}, '
                f'fourth={fourth_seal}, '
                f'compaction={reader_compaction}, '
                f'first_status={first_status}, '
                f'compaction_status={reader_compaction_status}'
            )
    finally:
        reader.rollback()
        reader.close()

    post_reader_maintenance: list[list[str]] = []
    for row_id in range(6, 9):
        append_row(row_id)
        maintenance, _ = rotate_and_seal(row_id)
        post_reader_maintenance.append(maintenance)
    final_compaction_maintenance, final_compaction = (
        compact_until_published()
    )
    final_status = fetch_status(connection, index_name)
    final_primary = final_status['generation']['primary']
    final_accounted_blocks = (
        final_primary['retired_hint_blocks']
        + final_primary['recyclable_marker_blocks']
    )
    final_hits = fetch_ids(connection, index_name, [1])
    if (
        int(final_compaction.get('reused_blocks', '-1')) != 0
        or not 0 <= final_primary['retired_hint_ranges'] <= 64
        or final_primary['interior_unreachable_blocks'] <= 0
        or final_primary['recyclable_marker_blocks'] != 0
        or final_accounted_blocks
        != final_primary['interior_unreachable_blocks']
        or len(positive_result_ids(final_hits)) != 8
        or final_primary['trailing_unpublished_blocks'] != 0
        or final_primary['reachable_blocks']
        + final_primary['interior_unreachable_blocks']
        != final_primary['published_block_high_watermark']
    ):
        raise AssertionError(
            'query-safe retirement did not remain exactly accounted: '
            f'compaction={final_compaction}, status={final_status}, '
            f'hits={final_hits}'
        )
    return {
        'reader_hits': reader_hits,
        'reader_locks': reader_locks,
        'first_maintenance': first_maintenance,
        'first_seal': first_seal,
        'first_status': first_status,
        'second_maintenance': second_maintenance,
        'second_seal': second_seal,
        'third_maintenance': third_maintenance,
        'third_seal': third_seal,
        'fourth_maintenance': fourth_maintenance,
        'fourth_seal': fourth_seal,
        'reader_compaction_maintenance': reader_compaction_maintenance,
        'reader_compaction': reader_compaction,
        'reader_compaction_status': reader_compaction_status,
        'post_reader_maintenance': post_reader_maintenance,
        'final_compaction_maintenance': final_compaction_maintenance,
        'final_compaction': final_compaction,
        'final_status': final_status,
        'final_hits': final_hits,
    }


def exercise_fixed_live_set_churn(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    table_name = 'parity.fixed_churn_docs'
    reference_index = 'parity.fixed_churn_reference_idx'
    v3_index = 'parity.fixed_churn_v3_idx'
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.fixed_churn_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.fixed_churn_docs VALUES '
            '(1, ARRAY[1, 2]), (2, ARRAY[1, 3]), '
            '(3, ARRAY[1, 2]), (4, ARRAY[1, 3])'
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX fixed_churn_reference_idx '
            'ON parity.fixed_churn_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX fixed_churn_v3_idx '
            'ON parity.fixed_churn_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )

    timeline: list[dict[str, Any]] = []
    maintenance: list[str] = []
    compactions: list[dict[str, str]] = []
    try:
        for cycle in range(48):
            row_id = cycle % 4 + 1
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_convergent_l0_rotation_records = '1'"
                )
                cursor.execute(
                    'UPDATE parity.fixed_churn_docs '
                    'SET tokens = CASE WHEN tokens = ARRAY[1, 2] '
                    'THEN ARRAY[1, 3] ELSE ARRAY[1, 2] END '
                    'WHERE id = %s',
                    (row_id,),
                )
                cursor.execute('VACUUM parity.fixed_churn_docs')

            converged = False
            maintenance_deadline = time.monotonic() + 15.0
            for _ in range(320):
                result = try_maintain(connection, v3_index)
                maintenance.append(result)
                fields = maintenance_result_fields(result)
                status = fetch_status(connection, v3_index)
                if (
                    fields.get('mode') == 'segment_compaction'
                    and fields.get('maintained') == 'true'
                ):
                    compactions.append(fields)
                delta = status['generation']['delta']
                if (
                    delta['active']['records'] > 0
                    or delta['pending']['records'] > 0
                ):
                    continue
                if fields.get('maintained') == 'true':
                    continue
                if fields.get('reason') in {'lock_busy', 'xid_horizon'}:
                    if time.monotonic() >= maintenance_deadline:
                        break
                    time.sleep(0.05)
                    continue
                converged = True
                break
            if not converged:
                raise AssertionError(
                    'fixed-live-set maintenance did not converge: '
                    f'cycle={cycle}, results={maintenance[-320:]}'
                )

            reference_hits = fetch_ids_by_tid(connection, reference_index, [1, 2])
            try:
                v3_hits = fetch_ids_by_tid(
                    connection,
                    v3_index,
                    [1, 2],
                )
            except psycopg.Error as exc:
                raise AssertionError(
                    'fixed-live-set generation failed to attach: '
                    f'cycle={cycle}, status={status}, '
                    f'maintenance={maintenance[-8:]}'
                ) from exc
            assert_rows_close(
                reference_hits,
                v3_hits,
                label=f'fixed-live-set cycle {cycle}',
            )
            primary = status['generation']['primary']
            timeline.append(
                {
                    'cycle': cycle,
                    'document_slot_high_watermark': status['generation'][
                        'document_slot_high_watermark'
                    ],
                    'reusable_document_slot_cursor': status['generation'][
                        'reusable_document_slot_cursor'
                    ],
                    'published_blocks': primary[
                        'published_block_high_watermark'
                    ],
                    'reachable_blocks': primary['reachable_blocks'],
                    'unreachable_blocks': primary[
                        'interior_unreachable_blocks'
                    ],
                    'retired_hint_blocks': primary[
                        'retired_hint_blocks'
                    ],
                    'retired_hint_ranges': primary[
                        'retired_hint_ranges'
                    ],
                    'recyclable_marker_blocks': primary[
                        'recyclable_marker_blocks'
                    ],
                    'segment_count': primary['segment_count'],
                }
            )
    finally:
        with connection.cursor() as cursor:
            cursor.execute('RESET evoke.test_convergent_l0_rotation_records')

    slot_high_watermarks = [
        point['document_slot_high_watermark'] for point in timeline
    ]
    if any(
        current < previous
        for previous, current in zip(
            slot_high_watermarks,
            slot_high_watermarks[1:],
        )
    ):
        raise AssertionError(
            'fixed-live-set document slot high watermark regressed: '
            f'{slot_high_watermarks}'
        )
    plateau_window = slot_high_watermarks[len(slot_high_watermarks) // 2 :]
    if len(set(plateau_window)) != 1:
        raise AssertionError(
            'fixed-live-set document slots did not plateau: '
            f'{slot_high_watermarks}'
        )
    cursor_values = [
        point['reusable_document_slot_cursor'] for point in timeline
    ]
    if any(cursor != 0 for cursor in cursor_values):
        raise AssertionError(
            'fixed-live-set reusable cursor did not reset after convergence: '
            f'{cursor_values}'
        )

    return {
        'timeline': timeline,
        'document_slot_plateau': plateau_window[0],
        'maintenance': maintenance,
        'compactions': compactions,
        'final_status': fetch_status(connection, v3_index),
        'final_reference_hits': fetch_ids_by_tid(connection, reference_index, [1, 2]),
        'final_v3_hits': fetch_ids_by_tid(connection, v3_index, [1, 2]),
    }


def exercise_history_barrier_compaction(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    table_name = 'parity.history_barrier_docs'
    reference_index = 'parity.history_barrier_reference_idx'
    v3_index = 'parity.history_barrier_v3_idx'
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.history_barrier_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.history_barrier_docs VALUES '
            '(1, ARRAY[1, 11])'
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX history_barrier_reference_idx '
            'ON parity.history_barrier_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX history_barrier_v3_idx '
            'ON parity.history_barrier_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )

    maintenance: list[str] = []
    compaction: dict[str, str] | None = None
    try:
        for row_id in range(2, 5):
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_convergent_l0_rotation_records = '1'"
                )
                cursor.execute(
                    'INSERT INTO parity.history_barrier_docs '
                    'VALUES (%s, ARRAY[1, %s])',
                    (row_id, row_id + 10),
                )
            for _ in range(16):
                result = try_maintain(connection, v3_index)
                maintenance.append(result)
                fields = maintenance_result_fields(result)
                if (
                    fields.get('mode') == 'segment_seal'
                    and fields.get('maintained') == 'true'
                ):
                    break
                if fields.get('maintained') == 'true':
                    continue
                if fields.get('reason') in {'lock_busy', 'xid_horizon'}:
                    continue
                raise AssertionError(
                    'history-barrier setup stopped before sealing: '
                    f'{maintenance[-16:]}'
                )
            else:
                raise AssertionError(
                    'history-barrier setup did not seal row '
                    f'{row_id}: {maintenance[-16:]}'
                )

        with connection.cursor() as cursor:
            cursor.execute('DELETE FROM parity.history_barrier_docs')
            cursor.execute('VACUUM parity.history_barrier_docs')

        for _ in range(64):
            result = try_maintain(connection, v3_index)
            maintenance.append(result)
            fields = maintenance_result_fields(result)
            if (
                fields.get('mode') == 'segment_compaction'
                and fields.get('maintained') == 'true'
                and int(fields.get('reclaimed_documents', '0')) > 0
            ):
                compaction = fields
                break
            if fields.get('maintained') == 'true':
                continue
            if fields.get('reason') in {'lock_busy', 'xid_horizon'}:
                continue
        if compaction is None:
            raise AssertionError(
                'history-barrier compaction did not reclaim an empty '
                f'range: {maintenance[-32:]}'
            )

        empty_reference_hits = fetch_ids_by_tid(connection, reference_index, [1])
        empty_v3_hits = fetch_ids_by_tid(connection, v3_index, [1])
        empty_status = fetch_status(connection, v3_index)
        assert_rows_close(
            empty_reference_hits,
            empty_v3_hits,
            label='history-barrier empty result',
        )
        if empty_v3_hits or empty_status['query_ready'] is not True:
            raise AssertionError(
                'history-barrier replacement is not query ready: '
                f'status={empty_status}, hits={empty_v3_hits}'
            )

        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO parity.history_barrier_docs VALUES '
                '(5, ARRAY[1, 15])'
            )
        active_reference_hits = fetch_ids_by_tid(connection, reference_index, [1])
        active_v3_hits = fetch_ids_by_tid(connection, v3_index, [1])
        assert_rows_close(
            active_reference_hits,
            active_v3_hits,
            label='history-barrier active L0 insert',
        )
        for _ in range(32):
            result = try_maintain(connection, v3_index)
            maintenance.append(result)
            fields = maintenance_result_fields(result)
            status = fetch_status(connection, v3_index)
            delta = status['generation']['delta']
            if (
                delta['active']['records'] == 0
                and delta['pending']['records'] == 0
                and fields.get('maintained') != 'true'
                and fields.get('reason') not in {'lock_busy', 'xid_horizon'}
            ):
                break
        else:
            raise AssertionError(
                'history-barrier post-insert maintenance did not '
                f'converge: {maintenance[-32:]}'
            )
        converged_reference_hits = fetch_ids_by_tid(connection, reference_index, [1])
        converged_v3_hits = fetch_ids_by_tid(connection, v3_index, [1])
        assert_rows_close(
            converged_reference_hits,
            converged_v3_hits,
            label='history-barrier converged insert',
        )
    finally:
        with connection.cursor() as cursor:
            cursor.execute('RESET evoke.test_convergent_l0_rotation_records')

    return {
        'maintenance': maintenance,
        'compaction': compaction,
        'empty_status': empty_status,
        'empty_hits': empty_v3_hits,
        'active_hits': active_v3_hits,
        'converged_hits': converged_v3_hits,
        'final_status': fetch_status(connection, v3_index),
    }


def exercise_extent_pressure(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.extent_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.extent_docs VALUES (1, ARRAY[1, 2])'
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX extent_reference_idx '
            'ON parity.extent_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX extent_v3_idx '
            'ON parity.extent_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )

    acquire_maintenance_guard(connection, 'parity.extent_v3_idx')
    atom_counts = (16, 5000, 12000, 25000, 50000, 100000)
    next_term = 1000
    maintenance_results: list[str] = []
    compactions: list[dict[str, str]] = []
    max_segments = 1
    for row_offset, atom_count in enumerate(atom_counts):
        first_term = next_term
        last_term = first_term + atom_count - 1
        next_term = last_term + 1
        row_id = 100 + row_offset
        with connection.cursor() as cursor:
            cursor.execute(
                "SET evoke.test_convergent_l0_rotation_records = '1'"
            )
            cursor.execute(
                'INSERT INTO parity.extent_docs VALUES ('
                '%s, ARRAY[1] || ARRAY('
                'SELECT term_id FROM generate_series('
                '%s::int4, %s::int4) '
                'AS terms(term_id)))',
                (row_id, first_term, last_term),
            )

        try:
            for _ in range(16):
                result = try_maintain(
                    connection,
                    'parity.extent_v3_idx',
                )
                maintenance_results.append(result)
                fields = maintenance_result_fields(result)
                status = fetch_status(connection, 'parity.extent_v3_idx')
                max_segments = max(
                    max_segments,
                    status['generation']['primary']['segment_count'],
                )
                if fields.get('mode') == 'segment_compaction':
                    if fields.get('maintained') != 'true':
                        raise AssertionError(
                            'extent-pressure compaction did not progress: '
                            f'{result}'
                        )
                    if fields.get('reason') != 'extent_pressure':
                        raise AssertionError(
                            'non-uniform segments compacted for the wrong '
                            f'reason: {result}'
                        )
                    if (
                        int(fields['segment_count']) > 8
                        or int(fields['input_bytes']) > 64 * 1024 * 1024
                    ):
                        raise AssertionError(
                            'extent-pressure compaction exceeded budget: '
                            f'{result}'
                        )
                    compactions.append(fields)
                    continue
                if fields.get('reason') == 'lock_busy':
                    time.sleep(0.05)
                    continue
                if (
                    fields.get('mode') == 'segment_reclamation'
                    and fields.get('reason')
                    == 'retired_pages_reclaimed'
                    and fields.get('maintained') == 'true'
                ):
                    continue
                if (
                    status['generation']['delta']['pending']['records'] > 0
                ):
                    continue
                if (
                    status['generation']['delta']['active']['records'] > 0
                ):
                    continue
                if (
                    fields.get('mode') == 'segment_seal'
                    and fields.get('maintained') == 'true'
                ):
                    continue
                break
            else:
                raise AssertionError(
                    'extent-pressure maintenance did not drain after row '
                    f'{row_id}: {maintenance_results[-16:]}'
                )
        finally:
            with connection.cursor() as cursor:
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )

    release_maintenance_guard(connection, 'parity.extent_v3_idx')
    status = fetch_status(connection, 'parity.extent_v3_idx')
    reference_hits = fetch_ids(connection, 'parity.extent_reference_idx', [1])
    v3_hits = fetch_ids(connection, 'parity.extent_v3_idx', [1])
    assert_rows_close(reference_hits, v3_hits, label='extent-pressure reference-v3')
    checks = {
        'all_rows_searchable': len(positive_result_ids(v3_hits)) == 7,
        'extent_compaction_exercised': len(compactions) >= 1,
        'active_debt_bounded': (
            0 <= status['details']['pending_writes'] <= 1
            and status['details']['pending_deletes'] == 0
            and status['details']['delta_records']
            == status['details']['pending_writes']
            and status['details']['delta_bytes']
            == status['generation']['delta']['bytes']
            and status['generation']['delta']['upserts']
            == status['details']['pending_writes']
            and status['generation']['delta']['retirements'] == 0
            and status['generation']['delta']['active']['records'] <= 1
        ),
        'pending_drained': (
            status['generation']['delta']['pending']['records'] == 0
        ),
        'fanout_bounded': (
            status['generation']['primary']['segment_count'] <= 5
            and max_segments <= 6
        ),
    }
    if not all(checks.values()):
        raise AssertionError(
            'extent-pressure convergence failed: '
            f'{checks}, status={status}, compactions={compactions}, '
            f'maintenance_tail={maintenance_results[-8:]}'
        )
    return {
        'atom_counts': atom_counts,
        'maintenance': maintenance_results,
        'compactions': compactions,
        'max_segments': max_segments,
        'status': status,
        'reference_hits': reference_hits,
        'v3_hits': v3_hits,
    }


def exercise_structural_term_fold(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.structural_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.structural_docs VALUES (1, ARRAY[1, 2])'
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX structural_reference_idx '
            'ON parity.structural_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX structural_v3_idx '
            'ON parity.structural_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )
        cursor.execute('SET evoke.test_force_structural_term_fold = true')

    maintenance: list[str] = []
    structural_folds: list[dict[str, str]] = []
    try:
        for row_offset in range(7):
            row_id = 200 + row_offset
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_convergent_l0_rotation_records = '1'"
                )
                cursor.execute(
                    'INSERT INTO parity.structural_docs VALUES '
                    '(%s, ARRAY[1, %s])',
                    (row_id, 1000 + row_offset),
                )
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )

            for _ in range(12):
                result = try_maintain(
                    connection,
                    'parity.structural_v3_idx',
                )
                maintenance.append(result)
                fields = maintenance_result_fields(result)
                status = fetch_status(
                    connection,
                    'parity.structural_v3_idx',
                )
                if fields.get('mode') == 'segment_compaction':
                    raise AssertionError(
                        'forced structural-fold setup compacted segments: '
                        f'{result}'
                    )
                if fields.get('mode') == 'term_structural_fold':
                    if fields.get('maintained') != 'true':
                        raise AssertionError(
                            'structural fold did not progress: '
                            f'{result}'
                        )
                    structural_folds.append(fields)
                    continue
                if (
                    status['generation']['delta']['pending']['records'] > 0
                ):
                    continue
                if (
                    fields.get('mode') == 'segment_seal'
                    and fields.get('maintained') == 'true'
                ):
                    continue
                break
            else:
                raise AssertionError(
                    'structural-fold setup did not drain: '
                    f'{maintenance[-12:]}'
                )
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                'RESET evoke.test_force_structural_term_fold'
            )

    if not structural_folds:
        raise AssertionError(
            'forced structural-fold setup produced no structural fold'
        )
    converged_multi_extent_fold = False
    for fold in structural_folds:
        if fold.get('reason') == 'geometric_fold_promotion':
            if (
                int(fold['segment_count']) != 0
                or int(fold['consumed_extents']) != 0
                or int(fold['input_bytes']) > 64 * 1024 * 1024
            ):
                raise AssertionError(
                    'structural fold promotion was not a bounded '
                    f'major/minor merge: {fold}'
                )
            continue
        if (
            int(fold['segment_count']) > 8
            or int(fold['input_bytes']) > 64 * 1024 * 1024
            or int(fold['consumed_extents']) < 1
            or int(fold['remaining_extents']) > 3
        ):
            raise AssertionError(
                f'structural fold violated its hard budget: {fold}'
            )
        if (
            int(fold['consumed_extents']) >= 3
            and int(fold['remaining_extents']) <= 3
        ):
            converged_multi_extent_fold = True
    if not converged_multi_extent_fold:
        raise AssertionError(
            'structural fold did not consume multiple extents in one '
            f'bounded action: {structural_folds}'
        )

    with connection.cursor() as cursor:
        cursor.execute(
            "SET evoke.test_convergent_l0_rotation_records = '1'"
        )
        cursor.execute(
            'INSERT INTO parity.structural_docs VALUES '
            '(999, ARRAY[1, 2000])'
        )
        cursor.execute('RESET evoke.test_convergent_l0_rotation_records')
    tail_maintenance: list[str] = []
    for _ in range(12):
        result = try_maintain(
            connection,
            'parity.structural_v3_idx',
        )
        tail_maintenance.append(result)
        fields = maintenance_result_fields(result)
        status = fetch_status(connection, 'parity.structural_v3_idx')
        if (
            status['generation']['delta']['pending']['records'] == 0
            and fields.get('mode') == 'segment_seal'
            and fields.get('maintained') == 'true'
        ):
            break
        if status['generation']['delta']['pending']['records'] > 0:
            continue
        if fields.get('maintained') != 'true':
            break
    else:
        raise AssertionError(
            f'post-fold tail did not drain: {tail_maintenance}'
        )

    reference_hits = fetch_ids(connection, 'parity.structural_reference_idx', [1])
    v3_hits = fetch_ids(connection, 'parity.structural_v3_idx', [1])
    assert_rows_close(reference_hits, v3_hits, label='structural-fold reference-v3')
    status = fetch_status(connection, 'parity.structural_v3_idx')
    if (
        len(positive_result_ids(v3_hits)) != 9
        or status['generation']['delta']['pending']['records'] != 0
    ):
        raise AssertionError(
            'structural fold or its post-fold tail is incomplete: '
            f'{status}, hits={v3_hits}'
        )
    page_native_term = fetch_page_native_term(
        connection,
        'parity.structural_v3_idx',
        1,
    )
    term_plan_stats = fetch_query_term_plan_stats(
        connection,
        'parity.structural_v3_idx',
        1,
    )
    page_native_topk = fetch_page_native_query_topk(
        connection,
        'parity.structural_v3_idx',
        [1],
        5,
    )
    page_native_all = fetch_page_native_query_topk(
        connection,
        'parity.structural_v3_idx',
        [1],
        9,
    )
    allowed_doc_ids = [
        page_native_all['page_native_doc_ids'][0],
        page_native_all['page_native_doc_ids'][4],
        page_native_all['page_native_doc_ids'][7],
    ]
    filtered_topk = fetch_page_native_query_topk(
        connection,
        'parity.structural_v3_idx',
        [1],
        2,
        allowed_doc_ids=allowed_doc_ids,
    )
    expected_filtered = [
        document_id
        for document_id in page_native_all['page_native_doc_ids']
        if document_id in allowed_doc_ids
    ][:2]
    if (
        int(page_native_term['plan_run_count']) <= 1
        or page_native_topk['matched'] is not True
        or page_native_topk['positive_topk_complete'] is not True
        or page_native_topk['topk_complete'] is not True
        or int(page_native_topk['query_term_count']) != 1
        or int(page_native_topk['query_run_count'])
        != int(page_native_term['plan_run_count'])
        or int(term_plan_stats['run_count'])
        != int(page_native_term['plan_run_count'])
        or int(term_plan_stats['posting_count'])
        != int(page_native_term['posting_count'])
        or int(term_plan_stats['block_count'])
        != int(page_native_term['block_count'])
        or int(term_plan_stats['fold_runs'])
        + int(term_plan_stats['payload_runs'])
        != int(term_plan_stats['run_count'])
        or int(page_native_topk['max_document_block_records']) > 128
        or filtered_topk['filtered'] is not True
        or int(filtered_topk['allowed_document_count']) != 3
        or filtered_topk['page_native_doc_ids'] != expected_filtered
    ):
        raise AssertionError(
            'page-native multi-run top-k differs from snapshot: '
            f'term={page_native_term}, topk={page_native_topk}, '
            f'filtered={filtered_topk}, expected={expected_filtered}'
        )
    return {
        'maintenance': maintenance,
        'structural_folds': structural_folds,
        'tail_maintenance': tail_maintenance,
        'status': status,
        'reference_hits': reference_hits,
        'v3_hits': v3_hits,
        'page_native_term': page_native_term,
        'term_plan_stats': term_plan_stats,
        'page_native_topk': page_native_topk,
        'filtered_topk': filtered_topk,
    }


def exercise_filtered_block_pruning(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    document_count = 512
    allowed_doc_ids = [0, 1, 2]

    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.filtered_block_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.filtered_block_docs '
            'SELECT value, ARRAY[1, value + 2] '
            'FROM generate_series(0, %s) AS value',
            (document_count - 1,),
        )
        cursor.execute(
            'CREATE INDEX filtered_block_idx '
            'ON parity.filtered_block_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )

    full_topk = fetch_page_native_query_topk(
        connection,
        'parity.filtered_block_idx',
        [1],
        document_count,
    )
    filtered_topk = fetch_page_native_query_topk(
        connection,
        'parity.filtered_block_idx',
        [1],
        len(allowed_doc_ids),
        allowed_doc_ids=allowed_doc_ids,
    )
    expected_doc_ids = [
        document_id
        for document_id in full_topk['page_native_doc_ids']
        if document_id in allowed_doc_ids
    ]
    if (
        filtered_topk['page_native_doc_ids'] != expected_doc_ids
        or int(filtered_topk['blocks_skipped']) < 1
        or int(filtered_topk['blocks_scored']) > 1
        or int(filtered_topk['postings_examined']) > 128
    ):
        raise AssertionError(
            'filtered page-native query did not prune empty candidate '
            f'blocks exactly: expected={expected_doc_ids}, '
            f'actual={filtered_topk}'
        )
    return {
        'allowed_doc_ids': allowed_doc_ids,
        'expected_doc_ids': expected_doc_ids,
        'filtered_topk': filtered_topk,
    }


def exercise_unified_maintenance_scheduler(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    optional_index = 'parity.scheduler_optional_v3_idx'
    mandatory_index = 'parity.scheduler_mandatory_v3_idx'
    optional_guard_released = True
    mandatory_guard_released = True
    setup_maintenance: list[str] = []
    mandatory_maintenance: list[str] = []
    mandatory_policy_pending_total = 0

    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.scheduler_optional_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.scheduler_optional_docs '
            'VALUES (1, ARRAY[1, 2])'
        )
        cursor.execute(
            'CREATE INDEX scheduler_optional_v3_idx '
            'ON parity.scheduler_optional_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=eventual)'
        )

        cursor.execute(
            'CREATE TABLE parity.scheduler_mandatory_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'CREATE INDEX scheduler_mandatory_v3_idx '
            'ON parity.scheduler_mandatory_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=eventual)'
        )

    acquire_maintenance_guard(connection, optional_index)
    optional_guard_released = False
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SET evoke.test_force_structural_term_fold = true'
            )
        for row_offset in range(3):
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_convergent_l0_rotation_records = '1'"
                )
                cursor.execute(
                    'INSERT INTO parity.scheduler_optional_docs VALUES '
                    '(%s, ARRAY[1, %s])',
                    (100 + row_offset, 1000 + row_offset),
                )
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )
            for _ in range(8):
                setup_maintenance.append(
                    try_maintain(connection, optional_index)
                )
                status = fetch_status(connection, optional_index)
                delta = status['generation']['delta']
                if (
                    delta['active']['records'] == 0
                    and delta['pending']['records'] == 0
                ):
                    break
            else:
                raise AssertionError(
                    'scheduler optional surface did not drain: '
                    f'{setup_maintenance}'
                )
        with connection.cursor() as cursor:
            cursor.execute(
                'RESET evoke.test_force_structural_term_fold'
            )

        optional_before_hits = fetch_ids(
            connection,
            optional_index,
            [1],
        )
        optional_candidate_status = fetch_status(
            connection,
            optional_index,
        )
        for _batch_index in range(4):
            for _ in range(32):
                assert_rows_close(
                    optional_before_hits,
                    fetch_ids(connection, optional_index, [1]),
                    label='unified scheduler heat admission',
                )
            optional_candidate_status = fetch_status(
                connection,
                optional_index,
            )
            if optional_candidate_status['generation'][
                'workload_fold'
            ]['candidate'] is True:
                break
        optional_candidate = optional_candidate_status['generation'][
            'workload_fold'
        ]
        if (
            optional_candidate['candidate'] is not True
            or optional_candidate['extents'] < 2
        ):
            raise AssertionError(
                'scheduler optional heat was not admitted: '
                f'{optional_candidate}'
            )

        acquire_maintenance_guard(connection, mandatory_index)
        mandatory_guard_released = False
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO parity.scheduler_mandatory_docs '
                'VALUES (1, ARRAY[10, 20])'
            )
            cursor.execute(
                'SELECT pending_total::bigint '
                'FROM evoke_index_policy_recommend('
                '%s::regclass, \'balanced\')',
                (mandatory_index,),
            )
            policy_row = cursor.fetchone()
            if policy_row is None:
                raise AssertionError(
                    'v3 policy recommendation returned no row'
                )
            mandatory_policy_pending_total = int(policy_row[0])
            cursor.execute(
                'SELECT index_oid::oid, result '
                'FROM evoke_index_maintain_due(1)'
            )
            first_due = cursor.fetchone()
            cursor.execute(
                'SELECT %s::regclass::oid',
                (mandatory_index,),
            )
            mandatory_oid = int(cursor.fetchone()[0])
        if mandatory_policy_pending_total != 1:
            raise AssertionError(
                'v3 policy recommendation did not expose linked-L0 debt: '
                f'{mandatory_policy_pending_total}'
            )
        if first_due is None or int(first_due[0]) != mandatory_oid:
            raise AssertionError(
                'public scheduler did not prioritize mandatory work: '
                f'row={first_due}, expected_oid={mandatory_oid}'
            )
        first_fields = maintenance_result_fields(str(first_due[1]))
        optional_still_due = fetch_status(connection, optional_index)
        if (
            first_fields.get('mode') != 'segment_rotation'
            or optional_still_due['generation']['workload_fold'][
                'candidate'
            ]
            is not True
        ):
            raise AssertionError(
                'mandatory scheduling consumed or bypassed optional work: '
                f'first={first_due[1]}, optional={optional_still_due}'
            )

        for _ in range(8):
            mandatory_status = fetch_status(connection, mandatory_index)
            mandatory_delta = mandatory_status['generation']['delta']
            if (
                mandatory_delta['active']['records'] == 0
                and mandatory_delta['pending']['records'] == 0
            ):
                break
            mandatory_maintenance.append(
                try_maintain(connection, mandatory_index)
            )
        else:
            raise AssertionError(
                'mandatory scheduler surface did not converge: '
                f'{mandatory_maintenance}'
            )

        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT index_oid::oid, result '
                'FROM evoke_index_maintain_due(1)'
            )
            second_due = cursor.fetchone()
            cursor.execute(
                'SELECT %s::regclass::oid',
                (optional_index,),
            )
            optional_oid = int(cursor.fetchone()[0])
        if second_due is None or int(second_due[0]) != optional_oid:
            raise AssertionError(
                'public scheduler did not discover optional heat work: '
                f'row={second_due}, expected_oid={optional_oid}'
            )
        second_fields = maintenance_result_fields(str(second_due[1]))
        optional_after_status = fetch_status(connection, optional_index)
        optional_after_hits = fetch_ids(connection, optional_index, [1])
        assert_rows_close(
            optional_before_hits,
            optional_after_hits,
            label='public scheduler workload fold',
        )
        if (
            second_fields.get('mode') != 'term_workload_fold'
            or optional_after_status['generation']['generation']
            == optional_candidate_status['generation']['generation']
            or optional_after_status['generation']['workload_fold'][
                'candidate'
            ]
            is True
        ):
            raise AssertionError(
                'public scheduler did not publish exact optional fold: '
                f'second={second_due[1]}, status={optional_after_status}'
            )

        release_maintenance_guard(connection, mandatory_index)
        mandatory_guard_released = True
        release_maintenance_guard(connection, optional_index)
        optional_guard_released = True
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                'RESET evoke.test_force_structural_term_fold'
            )
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
        if (
            not mandatory_guard_released
            and maintenance_guard_is_held(connection, mandatory_index)
        ):
            release_maintenance_guard(connection, mandatory_index)
        if (
            not optional_guard_released
            and maintenance_guard_is_held(connection, optional_index)
        ):
            release_maintenance_guard(connection, optional_index)
        with connection.cursor() as cursor:
            cursor.execute(
                'DROP TABLE IF EXISTS parity.scheduler_mandatory_docs'
            )
            cursor.execute(
                'DROP TABLE IF EXISTS parity.scheduler_optional_docs'
            )

    return {
        'setup_maintenance': setup_maintenance,
        'optional_candidate_status': optional_candidate_status,
        'mandatory_policy_pending_total': mandatory_policy_pending_total,
        'first_due_result': str(first_due[1]),
        'mandatory_maintenance': mandatory_maintenance,
        'second_due_result': str(second_due[1]),
        'optional_after_status': optional_after_status,
        'before_hits': optional_before_hits,
        'after_hits': optional_after_hits,
    }


def exercise_manual_v3_lifecycle(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    table_name = 'parity.manual_v3_docs'
    index_name = 'parity.manual_v3_idx'

    def fetch_table_ids() -> list[int]:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT d.id '
                'FROM evoke_query_ids(%s::regclass, ARRAY[1], 100) h '
                f'JOIN {table_name} d ON d.ctid = h.ctid '
                'ORDER BY h.score DESC, d.id',
                (index_name,),
            )
            return [int(row[0]) for row in cursor.fetchall()]

    with connection.cursor() as cursor:
        cursor.execute(
            f'CREATE TABLE {table_name} ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            f'INSERT INTO {table_name} VALUES '
            '(1, ARRAY[1, 2]), (2, ARRAY[1, 3]), (3, ARRAY[4])'
        )
        cursor.execute(
            f'CREATE INDEX manual_v3_idx ON {table_name} '
            'USING evoke (tokens) '
            'WITH (sae=false, consistency=manual)'
        )
        cursor.execute(
            f'INSERT INTO {table_name} VALUES (4, ARRAY[1, 9])'
        )

    after_insert_status = fetch_status(connection, index_name)
    after_insert_ids = fetch_table_ids()
    if (
        after_insert_status['details']['stale'] is not True
        or after_insert_status['details']['pending_writes'] != 1
        or after_insert_status['details']['delta_records'] != 0
        or 4 in after_insert_ids
    ):
        raise AssertionError(
            'manual v3 insert did not retain the published-root contract: '
            f'status={after_insert_status}, ids={after_insert_ids}'
        )

    with connection.cursor() as cursor:
        cursor.execute(f'DELETE FROM {table_name} WHERE id = 1')
        cursor.execute(f'VACUUM (INDEX_CLEANUP ON) {table_name}')

    before_refresh_status = fetch_status(connection, index_name)
    before_refresh_ids = fetch_table_ids()
    if (
        before_refresh_status['details']['stale'] is not True
        or before_refresh_status['details']['pending_writes'] != 1
        or before_refresh_status['details']['delta_records'] != 0
        or 1 in before_refresh_ids
        or 4 in before_refresh_ids
    ):
        raise AssertionError(
            'manual v3 MVCC or debt accounting is incorrect: '
            f'status={before_refresh_status}, ids={before_refresh_ids}'
        )

    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_refresh(%s::regclass)',
            (index_name,),
        )
        refresh_result = str(cursor.fetchone()[0])

    after_refresh_status = fetch_status(connection, index_name)
    after_refresh_ids = fetch_table_ids()
    if (
        after_refresh_status['details']['stale'] is not False
        or after_refresh_status['details']['pending_writes'] != 0
        or after_refresh_status['details']['pending_deletes'] != 0
        or after_refresh_status['generation']['delta']['records'] != 0
        or 1 in after_refresh_ids
        or 4 not in after_refresh_ids
    ):
        raise AssertionError(
            'manual v3 refresh did not converge exactly: '
            f'status={after_refresh_status}, ids={after_refresh_ids}'
        )

    with connection.cursor() as cursor:
        cursor.execute(f'DROP TABLE {table_name}')

    return {
        'after_insert_status': after_insert_status,
        'after_insert_ids': after_insert_ids,
        'before_refresh_status': before_refresh_status,
        'before_refresh_ids': before_refresh_ids,
        'refresh_result': refresh_result,
        'after_refresh_status': after_refresh_status,
        'after_refresh_ids': after_refresh_ids,
    }


def exercise_vacuum_retirement_statistics(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    live_table = 'parity.retirement_live'
    live_index = 'parity.retirement_live_idx'
    oracle_table = 'parity.retirement_oracle'
    oracle_index = 'parity.retirement_oracle_idx'
    rows = (
        (1, 'alpha alpha river anchor'),
        (2, 'alpha beta river'),
        (3, 'alpha gamma'),
        (4, 'delta river'),
    )

    def fetch_ranked(
        table_name: str,
        index_name: str,
        query: str,
    ) -> list[tuple[int, float]]:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT d.id, h.score::float8 '
                'FROM evoke_query(%s::regclass, %s, 100) h '
                f'JOIN {table_name} d ON d.ctid = h.ctid '
                'ORDER BY h.score DESC, d.id',
                (index_name, query),
            )
            return [
                (int(row[0]), float(row[1]))
                for row in cursor.fetchall()
            ]

    with connection.cursor() as cursor:
        cursor.execute(
            f'CREATE TABLE {live_table} ('
            'id int PRIMARY KEY, body text NOT NULL)'
        )
        cursor.execute(
            f'CREATE TABLE {oracle_table} ('
            'id int PRIMARY KEY, body text NOT NULL)'
        )
        cursor.executemany(f'INSERT INTO {live_table} VALUES (%s, %s)', rows)
        cursor.executemany(
            f'INSERT INTO {oracle_table} VALUES (%s, %s)',
            rows,
        )
        cursor.execute(
            f'CREATE INDEX retirement_live_idx ON {live_table} '
            'USING evoke (body) WITH ('
            "sae=false, consistency=realtime, method='bm25+', "
            "idf_method='bm25+')"
        )
        cursor.execute(
            f'CREATE INDEX retirement_oracle_idx ON {oracle_table} '
            'USING evoke (body) WITH ('
            "sae=false, consistency=realtime, method='bm25+', "
            "idf_method='bm25+')"
        )
        cursor.execute(
            f"UPDATE {live_table} SET body = 'omega replacement' "
            'WHERE id = 1'
        )
        cursor.execute(
            f"UPDATE {oracle_table} SET body = 'omega replacement' "
            'WHERE id = 1'
        )
        cursor.execute(f'DELETE FROM {live_table} WHERE id = 4')
        cursor.execute(f'DELETE FROM {oracle_table} WHERE id = 4')
        cursor.execute('REINDEX INDEX parity.retirement_oracle_idx')

    queries = ('alpha', 'omega')
    before_vacuum: dict[str, list[tuple[int, float]]] = {}
    oracle: dict[str, list[tuple[int, float]]] = {}
    score_drift: dict[str, float] = {}
    for query in queries:
        live_rows = fetch_ranked(
            live_table,
            live_index,
            query,
        )
        oracle_rows = fetch_ranked(
            oracle_table,
            oracle_index,
            query,
        )
        live_scores = dict(live_rows)
        oracle_scores = dict(oracle_rows)
        if live_scores.keys() != oracle_scores.keys():
            raise AssertionError(
                f'pre-VACUUM live row set differs for {query}: '
                f'live={live_rows}, oracle={oracle_rows}'
            )
        before_vacuum[query] = live_rows
        oracle[query] = oracle_rows
        score_drift[query] = max(
            (
                abs(live_scores[document_id] - oracle_scores[document_id])
                for document_id in live_scores
            ),
            default=0.0,
        )

    before_status = fetch_status(connection, live_index)
    if (
        before_status['generation'].get('retirement_statistics')
        != 'vacuum_convergent'
    ):
        raise AssertionError(
            'v3 status did not expose the VACUUM retirement contract: '
            f'{before_status}'
        )

    with connection.cursor() as cursor:
        cursor.execute(f'VACUUM (INDEX_CLEANUP ON) {live_table}')

    lexicon = {
        query: fetch_page_native_lexicon(connection, live_index, query)
        for query in queries
    }
    if lexicon['alpha']['found'] is not True:
        raise AssertionError(
            f'immutable alpha term disappeared after VACUUM: {lexicon}'
        )
    if (
        lexicon['omega']['found'] is True
        and lexicon['omega']['term_id'] == lexicon['alpha']['term_id']
    ):
        raise AssertionError(
            f'appended omega term aliases immutable alpha: {lexicon}'
        )
    if (
        lexicon['omega']['found'] is False
        and lexicon['omega']['term_id'] != 2**32 - 1
    ):
        raise AssertionError(
            f'page-native lexicon miss exposed a usable term id: {lexicon}'
        )

    after_vacuum: dict[str, list[tuple[int, float]]] = {}
    for query in queries:
        live_rows = fetch_ranked(
            live_table,
            live_index,
            query,
        )
        assert_rows_close(
            live_rows,
            oracle[query],
            label=(
                f'VACUUM retirement statistics:{query}:'
                f'{lexicon[query]}'
            ),
        )
        after_vacuum[query] = live_rows

    return {
        'contract': before_status['generation']['retirement_statistics'],
        'before_vacuum': before_vacuum,
        'oracle': oracle,
        'score_drift_before_vacuum': score_drift,
        'after_vacuum': after_vacuum,
        'lexicon': lexicon,
        'after_vacuum_status': fetch_status(connection, live_index),
    }


def exercise_workload_term_fold(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    index_name = 'parity.workload_v3_idx'
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.workload_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.workload_docs VALUES (1, ARRAY[1, 2])'
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX workload_reference_idx '
            'ON parity.workload_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX workload_v3_idx '
            'ON parity.workload_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )

    acquire_maintenance_guard(connection, index_name)
    setup_maintenance: list[str] = []
    guard_released = False
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SET evoke.test_force_structural_term_fold = true'
            )
        for row_offset in range(3):
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_convergent_l0_rotation_records = '1'"
                )
                cursor.execute(
                    'INSERT INTO parity.workload_docs VALUES '
                    '(%s, ARRAY[1, %s])',
                    (300 + row_offset, 3000 + row_offset),
                )
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )

            for _ in range(8):
                result = try_maintain(connection, index_name)
                setup_maintenance.append(result)
                fields = maintenance_result_fields(result)
                status = fetch_status(connection, index_name)
                if fields.get('mode') == 'segment_compaction':
                    raise AssertionError(
                        'workload-fold setup compacted segments: '
                        f'{result}'
                    )
                if (
                    status['generation']['delta']['active']['records'] == 0
                    and status['generation']['delta']['pending']['records']
                    == 0
                ):
                    break
            else:
                raise AssertionError(
                    'workload-fold setup did not drain: '
                    f'{setup_maintenance[-8:]}'
                )
        with connection.cursor() as cursor:
            cursor.execute(
                'RESET evoke.test_force_structural_term_fold'
            )

        setup_status = fetch_status(connection, index_name)
        if (
            setup_status['generation']['primary']['segment_count'] < 3
            or not maintenance_guard_is_held(connection, index_name)
        ):
            raise AssertionError(
                'workload-fold setup did not preserve fragmented surfaces: '
                f'{setup_status}'
            )

        before_hits = fetch_ids(connection, index_name, [1])
        for _ in range(31):
            assert_rows_close(
                before_hits,
                fetch_ids(connection, index_name, [1]),
                label='workload heat first batch',
            )
        first_batch_status = fetch_status(connection, index_name)
        first_heat = first_batch_status['generation']['workload_fold']
        if first_heat['candidate'] is not False:
            raise AssertionError(
                'workload fold admitted before its heat threshold: '
                f'{first_heat}'
            )

        candidate_status = first_batch_status
        candidate = first_heat
        for batch_index in range(3):
            for _ in range(32):
                assert_rows_close(
                    before_hits,
                    fetch_ids(connection, index_name, [1]),
                    label='workload heat admission batch',
                )
            candidate_status = fetch_status(connection, index_name)
            candidate = candidate_status['generation']['workload_fold']
            if candidate['candidate'] is True:
                break
            if candidate['query_heat'] >= 64:
                raise AssertionError(
                    'eligible workload heat was not admitted: '
                    f'batch={batch_index}, candidate={candidate}'
                )
        if (
            candidate['candidate'] is not True
            or candidate['query_heat'] < 64
            or candidate['root_heat'] < 32
            or candidate['extents'] < 2
        ):
            raise AssertionError(
                'workload fold did not admit a hot fragmented term: '
                f'{candidate}'
            )

        release_maintenance_guard(connection, index_name)
        guard_released = True
        if maintenance_guard_is_held(connection, index_name):
            raise AssertionError(
                'workload-fold maintenance guard remained held after '
                f'release: {index_name}'
            )
        with connection.cursor() as cursor:
            cursor.execute('SELECT evoke_index_touch_maintenance()')
            touch_result = str(cursor.fetchone()[0])

        fold_statuses: list[dict[str, Any]] = []
        fold_scheduler: list[dict[str, Any]] = []
        deadline = time.monotonic() + 30.0
        while True:
            observed = fetch_status(connection, index_name)
            fold_statuses.append(observed)
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT '
                    "evoke_index_runtime_state_json(%s::regclass)"
                    "->'maintenance'",
                    (index_name,),
                )
                scheduler_row = cursor.fetchone()
            if scheduler_row is not None:
                fold_scheduler.append(scheduler_row[0])
            if (
                observed['generation']['generation']
                != candidate_status['generation']['generation']
                and observed['generation']['workload_fold']['candidate']
                is False
            ):
                break
            if time.monotonic() >= deadline:
                raise AssertionError(
                    'background worker did not publish workload fold: '
                    f'touch={touch_result}, '
                    f'scheduler={fold_scheduler[-10:]}, '
                    f'statuses={fold_statuses[-10:]}'
                )
            time.sleep(0.05)

        after_hits = fetch_ids(connection, index_name, [1])
        assert_rows_close(
            before_hits,
            after_hits,
            label='workload fold publication',
        )
        after_status = fetch_status(connection, index_name)
        if (
            after_status['generation']['generation']
            == candidate_status['generation']['generation']
            or after_status['generation']['workload_fold']['candidate']
            is True
        ):
            raise AssertionError(
                'workload fold did not advance and clear root-local heat: '
                f'{after_status}'
            )

        repeat_result = try_maintain(connection, index_name)
        repeat_fields = maintenance_result_fields(repeat_result)
        if repeat_fields.get('mode') == 'term_workload_fold':
            raise AssertionError(
                'workload fold repeated without new-root observations: '
                f'{repeat_result}'
            )
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                'RESET evoke.test_force_structural_term_fold'
            )
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
        if not guard_released and maintenance_guard_is_held(
            connection,
            index_name,
        ):
            release_maintenance_guard(connection, index_name)

    geometric_maintenance: list[str] = []
    acquire_maintenance_guard(connection, index_name)
    geometric_guard_released = False
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SET evoke.test_force_structural_term_fold = true'
            )

        def append_and_seal(
            rows: list[tuple[int, list[int]]],
        ) -> None:
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT set_config('
                    "'evoke.test_convergent_l0_rotation_records', "
                    '%s, false)',
                    (str(len(rows)),),
                )
                cursor.executemany(
                    'INSERT INTO parity.workload_docs '
                    'VALUES (%s, %s::int4[])',
                    rows,
                )
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )
            for _ in range(12):
                result = try_maintain(connection, index_name)
                geometric_maintenance.append(result)
                status = fetch_status(connection, index_name)
                delta = status['generation']['delta']
                if (
                    delta['active']['records'] == 0
                    and delta['pending']['records'] == 0
                ):
                    return
            raise AssertionError(
                'geometric workload setup did not drain: '
                f'{geometric_maintenance[-12:]}'
            )

        def observe_workload_candidate(
            label: str,
        ) -> dict[str, Any]:
            expected = fetch_ids(connection, index_name, [1])
            for _ in range(96):
                assert_rows_close(
                    expected,
                    fetch_ids(connection, index_name, [1]),
                    label=label,
                )
            return fetch_status(
                connection,
                index_name,
            )['generation']['workload_fold']

        def publish_workload_fold(
            label: str,
        ) -> dict[str, str]:
            for _ in range(4):
                result = try_maintain(connection, index_name)
                geometric_maintenance.append(result)
                fields = maintenance_result_fields(result)
                if (
                    fields.get('mode') == 'term_workload_fold'
                    and fields.get('maintained') == 'true'
                ):
                    return fields
                if not (
                    fields.get('mode') == 'segment_reclamation'
                    and fields.get('maintained') == 'true'
                ):
                    break
                observe_workload_candidate(
                    f'{label} after retired-page reclamation',
                )
            state = fetch_term_fold_state(
                connection,
                index_name,
                1,
            )
            raise AssertionError(
                f'{label} did not publish a workload fold: '
                f'result={result}, state={state}'
            )

        append_and_seal([(400, [1, 4000])])
        append_and_seal([(401, [1, 4001])])
        bootstrap_promotion_candidate = observe_workload_candidate(
            'workload geometric bootstrap promotion',
        )
        if bootstrap_promotion_candidate['candidate'] is not True:
            raise AssertionError(
                'initial minor did not become promotable: '
                f'{bootstrap_promotion_candidate}'
            )
        bootstrap_promotion_action = publish_workload_fold(
            'workload geometric bootstrap promotion',
        )
        bootstrap_promotion_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        if (
            bootstrap_promotion_action.get('reason')
            != 'workload_fold_promotion'
            or bootstrap_promotion_state['minor'] is not False
        ):
            raise AssertionError(
                'initial major/minor promotion used the wrong action: '
                f'action={bootstrap_promotion_action}, '
                f'state={bootstrap_promotion_state}'
            )

        minor_candidate = observe_workload_candidate(
            'workload geometric initial minor',
        )
        if minor_candidate['candidate'] is not True:
            raise AssertionError(
                'workload geometric setup did not admit its next minor: '
                f'{minor_candidate}'
            )
        minor_action = publish_workload_fold(
            'workload geometric initial minor',
        )
        initial_minor_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        if (
            minor_action.get('reason') != 'workload_heat'
            or initial_minor_state['minor'] is not True
        ):
            raise AssertionError(
                'workload geometric setup used the wrong minor action: '
                f'action={minor_action}, state={initial_minor_state}'
            )

        initial_minor_promotion_candidate = observe_workload_candidate(
            'workload geometric initial-minor promotion',
        )
        if initial_minor_promotion_candidate['candidate'] is not True:
            raise AssertionError(
                'the next minor did not become promotable: '
                f'{initial_minor_promotion_candidate}'
            )
        initial_minor_promotion_action = publish_workload_fold(
            'workload geometric initial-minor promotion',
        )
        initial_minor_promotion_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        if (
            initial_minor_promotion_action.get('reason')
            != 'workload_fold_promotion'
            or initial_minor_promotion_state['minor'] is not False
        ):
            raise AssertionError(
                'the next major/minor promotion used the wrong action: '
                f'action={initial_minor_promotion_action}, '
                f'state={initial_minor_promotion_state}'
            )

        bootstrap_rows = [
            (500 + offset, [1, 5000 + offset])
            for offset in range(40)
        ]
        append_and_seal([(450, [1, 4500])])
        append_and_seal(bootstrap_rows)
        large_tail_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        large_minor_candidate = observe_workload_candidate(
            'workload geometric large minor',
        )
        if large_minor_candidate['candidate'] is not True:
            raise AssertionError(
                'large bootstrap tail did not admit a minor: '
                f'{large_minor_candidate}'
            )
        large_minor_action = publish_workload_fold(
            'workload geometric large minor',
        )
        large_minor_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        if (
            large_minor_action.get('reason') != 'workload_heat'
            or int(large_minor_action.get('posting_bytes', '0')) < 320
            or large_minor_state['minor'] is not True
        ):
            raise AssertionError(
                'large bootstrap tail did not become a minor: '
                f'action={large_minor_action}, '
                f'before={large_tail_state}, after={large_minor_state}'
            )
        large_promotion_candidate = observe_workload_candidate(
            'workload geometric large promotion',
        )
        if large_promotion_candidate['candidate'] is not True:
            raise AssertionError(
                'large bootstrap minor did not become promotable: '
                f'{large_promotion_candidate}'
            )
        large_promotion_action = publish_workload_fold(
            'workload geometric large promotion',
        )
        large_promotion_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        if (
            large_promotion_action.get('reason')
            != 'workload_fold_promotion'
            or large_promotion_state['minor'] is not False
        ):
            raise AssertionError(
                'large bootstrap promotion used the wrong action: '
                f'action={large_promotion_action}, '
                f'state={large_promotion_state}'
            )

        append_and_seal([(600, [1, 6000])])
        append_and_seal([(601, [1, 6001])])
        small_minor_candidate = observe_workload_candidate(
            'workload geometric small minor',
        )
        if small_minor_candidate['candidate'] is not True:
            raise AssertionError(
                'two small raw surfaces did not admit a minor: '
                f'{small_minor_candidate}'
            )
        small_minor_action = publish_workload_fold(
            'workload geometric small minor',
        )
        small_minor_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        if (
            small_minor_action.get('reason') != 'workload_heat'
            or small_minor_state['minor'] is not True
        ):
            raise AssertionError(
                'small minor used the wrong action: '
                f'action={small_minor_action}, '
                f'state={small_minor_state}'
            )

        append_and_seal([(602, [1, 6002])])
        deferred_candidate = observe_workload_candidate(
            'workload geometric small tail',
        )
        deferred_before = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        deferred_result = try_maintain(connection, index_name)
        geometric_maintenance.append(deferred_result)
        deferred_fields = maintenance_result_fields(deferred_result)
        deferred_after = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        deferred_geometry_before = {
            key: value
            for key, value in deferred_before.items()
            if key != 'root_id'
        }
        deferred_geometry_after = {
            key: value
            for key, value in deferred_after.items()
            if key != 'root_id'
        }
        if (
            deferred_fields.get('mode') == 'term_workload_fold'
            or deferred_geometry_before != deferred_geometry_after
        ):
            raise AssertionError(
                'small tail rewrote an existing minor before geometric '
                'parity: '
                f'candidate={deferred_candidate}, '
                f'result={deferred_result}, '
                f'before={deferred_before}, after={deferred_after}'
            )

        large_rows = [
            (700 + offset, [1, 7000 + offset])
            for offset in range(40)
        ]
        append_and_seal(large_rows)
        carry_candidate = observe_workload_candidate(
            'workload geometric carry',
        )
        if carry_candidate['candidate'] is not True:
            raise AssertionError(
                'comparable tail did not admit geometric minor carry: '
                f'{carry_candidate}'
            )
        carry_action = publish_workload_fold(
            'workload geometric carry',
        )
        carry_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        carry_posting_bytes = int(
            carry_action.get('posting_bytes', '0')
        )
        if (
            carry_action.get('reason') != 'workload_heat'
            or carry_posting_bytes < 320
            or carry_posting_bytes < deferred_before['minor_bytes']
            or carry_state['minor'] is not True
            or carry_state['minor_bytes'] <= deferred_before['minor_bytes']
        ):
            raise AssertionError(
                'geometric minor carry did not consume the comparable '
                f'tail: action={carry_action}, '
                f'before={deferred_before}, after={carry_state}'
            )

        promotion_candidate = observe_workload_candidate(
            'workload geometric promotion',
        )
        if promotion_candidate['candidate'] is not True:
            raise AssertionError(
                'geometric minor did not become promotable: '
                f'{promotion_candidate}'
            )
        promotion_action = publish_workload_fold(
            'workload geometric promotion',
        )
        promotion_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        if (
            promotion_action.get('reason')
            != 'workload_fold_promotion'
            or int(promotion_action.get('consumed_extents', '-1')) != 0
            or int(promotion_action.get('segment_count', '-1')) != 0
            or carry_state['minor_bytes'] * 2
            < carry_state['major_bytes']
            or promotion_state['minor'] is not False
            or promotion_state['extent_count'] != 0
            or promotion_state['major_coverage']
            != carry_state['minor_coverage']
        ):
            raise AssertionError(
                'major/minor promotion was not an exact metadata-only '
                f'carry: action={promotion_action}, '
                f'before={carry_state}, after={promotion_state}'
            )
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                'RESET evoke.test_force_structural_term_fold'
            )
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
        if (
            not geometric_guard_released
            and maintenance_guard_is_held(connection, index_name)
        ):
            release_maintenance_guard(connection, index_name)
            geometric_guard_released = True

    reference_hits = fetch_ids(
        connection,
        'parity.workload_reference_idx',
        [1],
    )
    v3_hits = fetch_ids(connection, index_name, [1])
    assert_rows_close(reference_hits, v3_hits, label='workload-fold reference-v3')
    final_status = fetch_status(connection, index_name)
    return {
        'setup_maintenance': setup_maintenance,
        'setup_status': setup_status,
        'first_batch_status': first_batch_status,
        'candidate_status': candidate_status,
        'touch_result': touch_result,
        'fold_statuses': fold_statuses,
        'repeat_result': repeat_result,
        'after_status': after_status,
        'geometric_maintenance': geometric_maintenance,
        'minor_action': minor_action,
        'initial_minor_state': initial_minor_state,
        'initial_minor_promotion_action': initial_minor_promotion_action,
        'initial_minor_promotion_state': initial_minor_promotion_state,
        'bootstrap_promotion_action': bootstrap_promotion_action,
        'bootstrap_promotion_state': bootstrap_promotion_state,
        'large_minor_action': large_minor_action,
        'large_minor_state': large_minor_state,
        'large_promotion_action': large_promotion_action,
        'large_promotion_state': large_promotion_state,
        'small_minor_action': small_minor_action,
        'small_minor_state': small_minor_state,
        'deferred_candidate': deferred_candidate,
        'deferred_result': deferred_result,
        'deferred_before': deferred_before,
        'deferred_after': deferred_after,
        'carry_action': carry_action,
        'carry_state': carry_state,
        'promotion_action': promotion_action,
        'promotion_state': promotion_state,
        'write_amplification': {
            'deferred_raw_postings': deferred_before[
                'raw_posting_count'
            ],
            'minor_bytes_before_carry': deferred_before['minor_bytes'],
            'new_posting_bytes_in_carry': carry_posting_bytes,
            'minor_bytes_after_carry': carry_state['minor_bytes'],
            'major_bytes_before_promotion': carry_state['major_bytes'],
            'minor_bytes_before_promotion': carry_state['minor_bytes'],
            'major_bytes_after_promotion': promotion_state[
                'major_bytes'
            ],
            'promotion_consumed_extents': int(
                promotion_action['consumed_extents']
            ),
        },
        'final_status': final_status,
        'reference_hits': reference_hits,
        'v3_hits': v3_hits,
    }


def exercise_impact_specialization(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    index_name = 'parity.workload_v3_idx'
    reference_index_name = 'parity.workload_reference_idx'
    maintenance: list[str] = []
    guard_released = False
    reference_guard_released = False

    def assert_query_parity(label: str) -> list[tuple[int, float]]:
        expected_by_tid = fetch_ids_by_tid(
            connection,
            reference_index_name,
            [1],
        )
        actual_by_tid = fetch_ids_by_tid(
            connection,
            index_name,
            [1],
        )
        assert_rows_close(expected_by_tid, actual_by_tid, label=label)
        return fetch_ids(connection, index_name, [1])

    def observe_candidate(label: str) -> dict[str, Any]:
        expected = fetch_ids_by_tid(connection, reference_index_name, [1])
        for _ in range(96):
            assert_rows_close(
                expected,
                fetch_ids_by_tid(connection, index_name, [1]),
                label=label,
            )
        candidate = fetch_status(
            connection,
            index_name,
        )['generation']['workload_fold']
        if candidate['candidate'] is not True:
            raise AssertionError(
                f'{label} did not admit its hot term: {candidate}'
            )
        return candidate

    def drain_l0(label: str) -> None:
        for _ in range(12):
            result = try_maintain(connection, index_name)
            maintenance.append(result)
            status = fetch_status(connection, index_name)
            delta = status['generation']['delta']
            if (
                delta['active']['records'] == 0
                and delta['pending']['records'] == 0
            ):
                return
        raise AssertionError(
            f'{label} did not drain its L0: {maintenance[-12:]}'
        )

    def converge_to_impact(
        label: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for action_index in range(12):
            candidate = observe_candidate(
                f'{label} heat {action_index}'
            )
            result = try_maintain(connection, index_name)
            maintenance.append(result)
            fields = maintenance_result_fields(result)
            state = fetch_term_fold_state(
                connection,
                index_name,
                1,
            )
            actions.append(
                {
                    'candidate': candidate,
                    'result': result,
                    'state': state,
                }
            )
            assert_query_parity(f'{label} action {action_index}')
            if (
                fields.get('mode') == 'term_impact_specialization'
                and fields.get('maintained') == 'true'
            ):
                return actions, state
            if (
                fields.get('mode') == 'segment_reclamation'
                and fields.get('reason') == 'retired_pages_reclaimed'
                and fields.get('maintained') == 'true'
            ):
                continue
            if (
                fields.get('mode') != 'term_workload_fold'
                or fields.get('maintained') != 'true'
            ):
                raise AssertionError(
                    f'{label} used an unexpected maintenance action: '
                    f'{result}, state={state}'
                )
        raise AssertionError(
            f'{label} did not converge to impact specialization: '
            f'{actions}'
        )

    acquire_maintenance_guard(connection, reference_index_name)
    acquire_maintenance_guard(connection, index_name)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SET evoke.test_force_structural_term_fold = true'
            )

        initial_actions, initial_state = converge_to_impact(
            'initial impact specialization',
        )
        initial_cache_state = fetch_generation_cache_state(
            connection,
            index_name,
        )
        if 'shared_hot_fold_current=true' not in initial_cache_state:
            raise AssertionError(
                'initial impact specialization did not publish its shared '
                f'hot fold: {initial_cache_state}'
            )
        initial_coverage = (
            initial_state['minor_coverage']
            if initial_state['minor']
            else initial_state['major_coverage']
        )
        if (
            initial_state['impact'] is not True
            or initial_state['impact_selected'] is not True
            or initial_state['impact_sparse_eligible'] is not True
            or initial_state['impact_coverage'] != initial_coverage
            or initial_state['impact_statistics_epoch']
            != initial_state['statistics_epoch']
            or initial_state['active_l0_records'] != 0
            or initial_state['pending_l0_records'] != 0
        ):
            raise AssertionError(
                'impact specialization was not exact and selectable: '
                f'{initial_state}'
            )

        with connection.cursor() as cursor:
            cursor.execute('SELECT evoke_runtime_cache_clear()')
            cache_clear_count = int(cursor.fetchone()[0])
        cache_cleared_state = fetch_generation_cache_state(
            connection,
            index_name,
        )
        if 'shared_hot_fold_current=false' not in cache_cleared_state:
            raise AssertionError(
                'cache clear retained the shared hot fold: '
                f'{cache_cleared_state}'
            )
        cache_cold_hits = assert_query_parity(
            'impact specialization cache clear',
        )
        cache_cold_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        if (
            cache_cold_state['impact_selected'] is not True
            or cache_cold_state['impact_sparse_eligible'] is not True
        ):
            raise AssertionError(
                'cache clear did not reload the impact specialization: '
                f'{cache_cold_state}'
            )
        cache_republish_candidate = observe_candidate(
            'impact specialization cache republish',
        )
        cache_republish_results: list[str] = []
        for _ in range(8):
            cache_republish_result = try_maintain(connection, index_name)
            maintenance.append(cache_republish_result)
            cache_republish_results.append(cache_republish_result)
            cache_republish_fields = maintenance_result_fields(
                cache_republish_result
            )
            if (
                cache_republish_fields.get('maintained') == 'true'
                and cache_republish_fields.get('mode')
                == 'term_impact_specialization'
                and cache_republish_fields.get('reason')
                == 'impact_specialization_hot_cache'
            ):
                break
            if (
                cache_republish_fields.get('maintained') == 'true'
                and cache_republish_fields.get('mode')
                == 'segment_reclamation'
                and cache_republish_fields.get('reason')
                == 'retired_pages_reclaimed'
            ):
                cache_republish_candidate = observe_candidate(
                    'impact specialization cache republish after '
                    'reclamation',
                )
                continue
            raise AssertionError(
                'impact cache republish used an unexpected action: '
                f'{cache_republish_result}'
            )
        else:
            raise AssertionError(
                'impact cache republish did not finish after reclamation: '
                f'{cache_republish_results}'
            )
        cache_republished_state = fetch_generation_cache_state(
            connection,
            index_name,
        )
        if (
            cache_republish_fields.get('maintained') != 'true'
            or cache_republish_fields.get('mode')
            != 'term_impact_specialization'
            or cache_republish_fields.get('reason')
            != 'impact_specialization_hot_cache'
            or 'shared_hot_fold_current=true'
            not in cache_republished_state
        ):
            raise AssertionError(
                'maintenance did not republish the persisted impact fold: '
                f'{cache_republish_results}, {cache_republished_state}'
            )

        with connection.cursor() as cursor:
            cursor.execute(
                "SET evoke.test_convergent_l0_rotation_records = '1'"
            )
            cursor.execute(
                'INSERT INTO parity.workload_docs '
                'VALUES (900, ARRAY[1, 9000])'
            )
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
        invalidated_hits = assert_query_parity(
            'impact specialization active L0 invalidation',
        )
        invalidated_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        invalidated_cache_state = fetch_generation_cache_state(
            connection,
            index_name,
        )
        if (
            invalidated_state['impact'] is not True
            or invalidated_state['impact_selected'] is not False
            or invalidated_state['impact_sparse_eligible'] is not False
            or invalidated_state['active_l0_records'] == 0
            or 'shared_hot_fold_current=false'
            not in invalidated_cache_state
        ):
            raise AssertionError(
                'active L0 did not invalidate the impact read image: '
                f'{invalidated_state}'
            )
        drain_l0('first impact invalidation')

        with connection.cursor() as cursor:
            cursor.execute(
                "SET evoke.test_convergent_l0_rotation_records = '1'"
            )
            cursor.execute(
                'INSERT INTO parity.workload_docs '
                'VALUES (901, ARRAY[1, 9001])'
            )
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
        assert_query_parity('second impact invalidation')
        drain_l0('second impact invalidation')
        stale_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        if (
            stale_state['impact'] is not True
            or stale_state['impact_selected'] is not False
            or stale_state['impact_sparse_eligible'] is not False
            or stale_state['active_l0_records'] != 0
            or stale_state['pending_l0_records'] != 0
        ):
            raise AssertionError(
                'sealed lexical changes selected a stale impact image: '
                f'{stale_state}'
            )

        rebuild_actions, rebuilt_state = converge_to_impact(
            'rebuilt impact specialization',
        )
        rebuilt_coverage = (
            rebuilt_state['minor_coverage']
            if rebuilt_state['minor']
            else rebuilt_state['major_coverage']
        )
        if (
            rebuilt_state['impact_selected'] is not True
            or rebuilt_state['impact_sparse_eligible'] is not True
            or rebuilt_state['impact_coverage'] != rebuilt_coverage
            or rebuilt_state['impact_statistics_epoch']
            != rebuilt_state['statistics_epoch']
            or rebuilt_state['impact_statistics_epoch']
            == initial_state['impact_statistics_epoch']
        ):
            raise AssertionError(
                'impact specialization did not rebuild for the new epoch: '
                f'initial={initial_state}, rebuilt={rebuilt_state}'
            )
        rebuilt_hits = assert_query_parity(
            'rebuilt impact specialization',
        )
        rebuilt_cache_state = fetch_generation_cache_state(
            connection,
            index_name,
        )
        if 'shared_hot_fold_current=true' not in rebuilt_cache_state:
            raise AssertionError(
                'rebuilt impact specialization did not restore its shared '
                f'hot fold: {rebuilt_cache_state}'
            )

        with connection.cursor() as cursor:
            cursor.execute(
                "SET evoke.test_convergent_l0_rotation_records = '1'"
            )
            cursor.execute(
                'DELETE FROM parity.workload_docs WHERE id = 1'
            )
            cursor.execute(
                'VACUUM (INDEX_CLEANUP ON) parity.workload_docs'
            )
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
        retirement_invalidated_hits = assert_query_parity(
            'impact specialization retirement invalidation',
        )
        retirement_invalidated_state = fetch_term_fold_state(
            connection,
            index_name,
            1,
        )
        retirement_invalidated_cache_state = (
            fetch_generation_cache_state(connection, index_name)
        )
        if (
            retirement_invalidated_state['impact'] is not True
            or retirement_invalidated_state['impact_selected'] is not False
            or retirement_invalidated_state[
                'impact_sparse_eligible'
            ] is not False
            or retirement_invalidated_state['retired_documents'] < 1
            or retirement_invalidated_state['impact_statistics_epoch']
            == retirement_invalidated_state['statistics_epoch']
            or 'shared_hot_fold_current=false'
            not in retirement_invalidated_cache_state
        ):
            raise AssertionError(
                'retirement did not invalidate the impact read image: '
                f'{retirement_invalidated_state}'
            )
        drain_l0('impact retirement invalidation')
        retirement_actions, retirement_state = converge_to_impact(
            'retirement-aware impact specialization',
        )
        retirement_coverage = (
            retirement_state['minor_coverage']
            if retirement_state['minor']
            else retirement_state['major_coverage']
        )
        if (
            retirement_state['impact_selected'] is not True
            or retirement_state['impact_sparse_eligible'] is not True
            or retirement_state['retired_documents'] < 1
            or retirement_state['impact_coverage'] != retirement_coverage
            or retirement_state['impact_statistics_epoch']
            != retirement_state['statistics_epoch']
        ):
            raise AssertionError(
                'retirement-aware impact specialization was not exact: '
                f'{retirement_state}'
            )
        final_hits = assert_query_parity(
            'retirement-aware impact specialization',
        )
        final_cache_state = fetch_generation_cache_state(
            connection,
            index_name,
        )
        if 'shared_hot_fold_current=true' not in final_cache_state:
            raise AssertionError(
                'retirement-aware impact specialization did not publish '
                f'its shared hot fold: {final_cache_state}'
            )
        final_status = fetch_status(connection, index_name)
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                'RESET evoke.test_force_structural_term_fold'
            )
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
        if (
            not guard_released
            and maintenance_guard_is_held(connection, index_name)
        ):
            release_maintenance_guard(connection, index_name)
            guard_released = True
        if (
            not reference_guard_released
            and maintenance_guard_is_held(connection, reference_index_name)
        ):
            release_maintenance_guard(connection, reference_index_name)
            reference_guard_released = True

    return {
        'initial_actions': initial_actions,
        'initial_state': initial_state,
        'initial_cache_state': initial_cache_state,
        'cache_clear_count': cache_clear_count,
        'cache_cleared_state': cache_cleared_state,
        'cache_cold_hits': cache_cold_hits,
        'cache_cold_state': cache_cold_state,
        'cache_republish_candidate': cache_republish_candidate,
        'cache_republish_results': cache_republish_results,
        'cache_republished_state': cache_republished_state,
        'invalidated_hits': invalidated_hits,
        'invalidated_state': invalidated_state,
        'invalidated_cache_state': invalidated_cache_state,
        'stale_state': stale_state,
        'rebuild_actions': rebuild_actions,
        'rebuilt_state': rebuilt_state,
        'rebuilt_hits': rebuilt_hits,
        'rebuilt_cache_state': rebuilt_cache_state,
        'retirement_invalidated_hits': retirement_invalidated_hits,
        'retirement_invalidated_state': retirement_invalidated_state,
        'retirement_invalidated_cache_state': (
            retirement_invalidated_cache_state
        ),
        'retirement_actions': retirement_actions,
        'retirement_state': retirement_state,
        'maintenance': maintenance,
        'final_hits': final_hits,
        'final_cache_state': final_cache_state,
        'final_status': final_status,
    }


def exercise_pending_l0_headroom(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    index_name = 'parity.headroom_v3_idx'
    rows = [
        (row_id, [1, 1000 + row_id])
        for row_id in range(1, 6)
    ]
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.headroom_reference '
            '(id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'CREATE TABLE parity.headroom_v3 '
            '(id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX headroom_reference_idx '
            'ON parity.headroom_reference USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX headroom_v3_idx '
            'ON parity.headroom_v3 USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )
        cursor.executemany(
            'INSERT INTO parity.headroom_reference VALUES (%s, %s::int4[])',
            rows,
        )

    acquire_maintenance_guard(connection, index_name)
    guard_released = False
    maintenance: list[str] = []
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SET evoke.test_convergent_l0_rotation_records = '2'"
            )
            cursor.executemany(
                'INSERT INTO parity.headroom_v3 VALUES (%s, %s::int4[])',
                rows[:2],
            )

        rotation = try_maintain(connection, index_name)
        rotation_fields = maintenance_result_fields(rotation)
        rotated_status = fetch_status(connection, index_name)
        rotated_delta = rotated_status['generation']['delta']
        if (
            rotation_fields.get('mode') != 'segment_rotation'
            or rotated_delta['pending']['records'] != 2
            or rotated_delta['active']['records'] != 0
        ):
            raise AssertionError(
                'headroom setup did not preserve one pending L0: '
                f'result={rotation}, status={rotated_status}'
            )

        with connection.cursor() as cursor:
            cursor.executemany(
                'INSERT INTO parity.headroom_v3 VALUES (%s, %s::int4[])',
                rows[2:],
            )
        headroom_status = fetch_status(connection, index_name)
        headroom_delta = headroom_status['generation']['delta']
        if (
            headroom_delta['pending']['records'] != 2
            or headroom_delta['active']['records'] != 3
            or headroom_delta['records'] != 5
        ):
            raise AssertionError(
                'active L0 did not use reserved headroom: '
                f'{headroom_status}'
            )

        reference_hits = fetch_ids(
            connection,
            'parity.headroom_reference_idx',
            [1],
        )
        before_hits = fetch_ids(connection, index_name, [1])
        assert_rows_close(
            reference_hits,
            before_hits,
            label='pending-L0 headroom reference-v3',
        )

        for _ in range(16):
            status = fetch_status(connection, index_name)
            delta = status['generation']['delta']
            if (
                delta['active']['records'] == 0
                and delta['pending']['records'] == 0
            ):
                break
            maintenance.append(try_maintain(connection, index_name))
        else:
            raise AssertionError(
                'reserved active headroom did not converge: '
                f'status={status}, maintenance={maintenance}'
            )

        converged_status = fetch_status(connection, index_name)
        after_hits = fetch_ids(connection, index_name, [1])
        assert_rows_close(
            reference_hits,
            after_hits,
            label='pending-L0 headroom convergence',
        )
        release_maintenance_guard(connection, index_name)
        guard_released = True
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
        if not guard_released and maintenance_guard_is_held(
            connection,
            index_name,
        ):
            release_maintenance_guard(connection, index_name)

    return {
        'rotation': rotation,
        'rotated_status': rotated_status,
        'headroom_status': headroom_status,
        'maintenance': maintenance,
        'converged_status': converged_status,
        'reference_hits': reference_hits,
        'v3_hits': after_hits,
    }


def exercise_posting_heat_query_batching(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    index_name = 'parity.heat_batch_v3_idx'
    tokens = list(range(128))
    query = list(range(70))
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.heat_batch_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.heat_batch_docs VALUES (1, %s::int4[])',
            (tokens,),
        )
        cursor.execute(
            'CREATE INDEX heat_batch_v3_idx '
            'ON parity.heat_batch_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )

    before = fetch_status(connection, index_name)
    before_heat = before['generation']['workload_fold']
    fresh = psycopg.connect(connection.info.dsn, autocommit=True)
    try:
        expected = fetch_ids(fresh, index_name, query)
        after_one = fetch_status(connection, index_name)
        for _ in range(30):
            assert_rows_close(
                expected,
                fetch_ids(fresh, index_name, query),
                label='posting heat local batching',
            )
        after_thirty_one = fetch_status(connection, index_name)
        assert_rows_close(
            expected,
            fetch_ids(fresh, index_name, query),
            label='posting heat batch flush',
        )
        after_thirty_two = fetch_status(connection, index_name)
    finally:
        fresh.close()

    short_lived = psycopg.connect(connection.info.dsn, autocommit=True)
    try:
        assert_rows_close(
            expected,
            fetch_ids(short_lived, index_name, query),
            label='posting heat short-lived backend',
        )
    finally:
        short_lived.close()
    after_short_lived = fetch_status(connection, index_name)

    after_one_heat = after_one['generation']['workload_fold']
    after_thirty_one_heat = (
        after_thirty_one['generation']['workload_fold']
    )
    after_thirty_two_heat = (
        after_thirty_two['generation']['workload_fold']
    )
    after_short_lived_heat = (
        after_short_lived['generation']['workload_fold']
    )
    initial_flushes = int(before_heat['shared_flushes'])
    if (
        int(after_one_heat['shared_flushes']) != initial_flushes
        or int(after_thirty_one_heat['shared_flushes']) != initial_flushes
    ):
        raise AssertionError(
            'wide query flushed posting heat before the query batch: '
            f'before={before_heat}, one={after_one_heat}, '
            f'thirty_one={after_thirty_one_heat}'
        )
    if (
        int(after_thirty_two_heat['shared_flushes'])
        != initial_flushes + 1
    ):
        raise AssertionError(
            'posting heat did not perform one batched shared flush: '
            f'before={before_heat}, after={after_thirty_two_heat}'
        )
    observation_delta = (
        int(after_thirty_two_heat['shared_query_observations'])
        - int(before_heat['shared_query_observations'])
    )
    if observation_delta != 64 * 32:
        raise AssertionError(
            'wide-query posting heat was not bounded to the local capacity: '
            f'observation_delta={observation_delta}, '
            f'after={after_thirty_two_heat}'
        )
    if (
        int(after_short_lived_heat['shared_flushes'])
        != initial_flushes + 2
        or int(after_short_lived_heat['shared_query_observations'])
        - int(after_thirty_two_heat['shared_query_observations'])
        != 64
    ):
        raise AssertionError(
            'short-lived backend discarded its local posting heat: '
            f'before_close={after_thirty_two_heat}, '
            f'after_close={after_short_lived_heat}'
        )
    eviction_delta = (
        int(after_thirty_two_heat['shared_evictions'])
        - int(before_heat['shared_evictions'])
    )
    entry_delta = (
        int(after_thirty_two_heat['entries'])
        - int(before_heat['entries'])
    )
    if (
        eviction_delta < 0
        or entry_delta != 64
    ):
        raise AssertionError(
            'posting-heat set admission was not bounded to 64 keys: '
            f'before={before_heat}, after={after_thirty_two_heat}'
        )
    if (
        int(after_thirty_two_heat['blocks_considered']) != 0
        or int(after_thirty_two_heat['blocks_scored']) != 0
        or int(after_thirty_two_heat['postings_scored']) != 0
    ):
        raise AssertionError(
            'query-wide scorer work was attributed to individual terms: '
            f'after={after_thirty_two_heat}'
        )
    return {
        'before': before,
        'after_one': after_one,
        'after_thirty_one': after_thirty_one,
        'after_thirty_two': after_thirty_two,
        'after_short_lived': after_short_lived,
        'query_width': len(query),
        'retained_width': 64,
        'observation_delta': observation_delta,
        'entry_delta': entry_delta,
        'eviction_delta': eviction_delta,
        'hits': expected,
    }


def exercise_fold_safe_compaction(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.fold_compaction_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.fold_compaction_docs '
            'VALUES (1, ARRAY[1, 2])'
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX fold_compaction_reference_idx '
            'ON parity.fold_compaction_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX fold_compaction_v3_idx '
            'ON parity.fold_compaction_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )

    fold_probe = fetch_term_fold_state(
        connection,
        'parity.fold_compaction_v3_idx',
        1,
    )
    if fold_probe['major'] is not True:
        raise AssertionError(
            f'initial folded build has no major fold: {fold_probe}'
        )
    maintenance: list[str] = []
    compaction: dict[str, str] | None = None
    for row_offset in range(4):
        with connection.cursor() as cursor:
            cursor.execute(
                "SET evoke.test_convergent_l0_rotation_records = '1'"
            )
            cursor.execute(
                'INSERT INTO parity.fold_compaction_docs VALUES '
                '(%s, ARRAY[1, 2])',
                (10 + row_offset,),
            )
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
        for _ in range(12):
            result = try_maintain(
                connection,
                'parity.fold_compaction_v3_idx',
            )
            maintenance.append(result)
            fields = maintenance_result_fields(result)
            status = fetch_status(
                connection,
                'parity.fold_compaction_v3_idx',
            )
            if fields.get('mode') == 'segment_compaction':
                compaction = fields
                break
            if (
                status['generation']['delta']['pending']['records'] > 0
            ):
                continue
            if (
                fields.get('mode') == 'segment_seal'
                and fields.get('maintained') == 'true'
            ):
                continue
            break
        else:
            raise AssertionError(
                'fold-safe compaction maintenance did not drain: '
                f'{maintenance[-12:]}'
            )

    if compaction is None:
        for _ in range(8):
            result = try_maintain(
                connection,
                'parity.fold_compaction_v3_idx',
            )
            maintenance.append(result)
            fields = maintenance_result_fields(result)
            if fields.get('mode') == 'segment_compaction':
                compaction = fields
                break
            if fields.get('maintained') != 'true':
                break
    if (
        compaction is None
        or compaction.get('maintained') != 'true'
        or compaction.get('reason') != 'tier_pressure'
        or int(compaction['segment_count']) > 4
        or int(compaction['segment_count']) < 2
    ):
        raise AssertionError(
            'post-fold tail range was not compacted safely: '
            f'{compaction}, maintenance={maintenance}'
        )

    reference_hits = fetch_ids(
        connection,
        'parity.fold_compaction_reference_idx',
        [1],
    )
    v3_hits = fetch_ids(
        connection,
        'parity.fold_compaction_v3_idx',
        [1],
    )
    assert_rows_close(reference_hits, v3_hits, label='fold-safe compaction')
    status = fetch_status(
        connection,
        'parity.fold_compaction_v3_idx',
    )
    if (
        len(positive_result_ids(v3_hits)) != 5
        or status['generation']['delta']['pending']['records'] != 0
    ):
        raise AssertionError(
            'fold-safe compaction left incomplete state: '
            f'{status}, hits={v3_hits}'
        )
    return {
        'fold_probe': fold_probe,
        'maintenance': maintenance,
        'compaction': compaction,
        'status': status,
        'reference_hits': reference_hits,
        'v3_hits': v3_hits,
    }


def exercise_fold_forward_compaction(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE parity.fold_forward_docs ('
            'id int PRIMARY KEY, tokens int4[] NOT NULL)'
        )
        cursor.execute(
            'INSERT INTO parity.fold_forward_docs '
            'VALUES (1, ARRAY[1, 2])'
        )
        create_page_native_reference_index(
            cursor,
            'CREATE INDEX fold_forward_reference_idx '
            'ON parity.fold_forward_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)',
        )
        cursor.execute(
            'CREATE INDEX fold_forward_v3_idx '
            'ON parity.fold_forward_docs USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )

    fold_probe = fetch_term_fold_state(
        connection,
        'parity.fold_forward_v3_idx',
        1,
    )
    if fold_probe['major'] is not True:
        raise AssertionError(
            f'initial folded build has no major fold: {fold_probe}'
        )
    with connection.cursor() as cursor:
        cursor.execute(
            "SET evoke.test_convergent_l0_rotation_records = '1'"
        )
        cursor.execute(
            'INSERT INTO parity.fold_forward_docs '
            'VALUES (2, ARRAY[1, 2]), (3, ARRAY[1, 2])'
        )
        cursor.execute('RESET evoke.test_convergent_l0_rotation_records')
        cursor.execute(
            'SET evoke.test_force_fold_conflict_compaction = true'
        )

    maintenance: list[str] = []
    fold_forward: dict[str, str] | None = None
    compaction: dict[str, str] | None = None
    try:
        for _ in range(16):
            result = try_maintain(
                connection,
                'parity.fold_forward_v3_idx',
            )
            maintenance.append(result)
            fields = maintenance_result_fields(result)
            if fields.get('mode') == 'term_fold_forward':
                fold_forward = fields
            if fields.get('mode') == 'segment_compaction':
                compaction = fields
            status = fetch_status(
                connection,
                'parity.fold_forward_v3_idx',
            )
            if (
                compaction is not None
                and compaction.get('maintained') == 'true'
                and status['generation']['primary']['segment_count'] == 1
                and status['generation']['delta']['pending']['records'] == 0
                and status['generation']['delta']['active']['records'] <= 1
            ):
                break
        else:
            raise AssertionError(
                'fold-forward compaction did not converge: '
                f'{maintenance}'
            )
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                'RESET evoke.test_force_fold_conflict_compaction'
            )

    if fold_forward is not None and (
        fold_forward.get('maintained') != 'true'
        or fold_forward.get('reason') != 'fold_watermark'
        or int(fold_forward['segment_count']) > 8
        or int(fold_forward['input_bytes']) > 64 * 1024 * 1024
        or int(fold_forward['consumed_extents']) < 1
        or fold_forward.get('compaction_segment_count') != '2'
    ):
        raise AssertionError(
            'fold watermark did not trigger bounded fold-forward: '
            f'{fold_forward}, maintenance={maintenance}'
        )
    if (
        compaction is None
        or compaction.get('maintained') != 'true'
        or compaction.get('reason') != 'tier_pressure'
        or compaction.get('segment_count') != '2'
    ):
        raise AssertionError(
            'compaction did not resume after fold-forward: '
            f'{compaction}, maintenance={maintenance}'
        )

    reference_hits = fetch_ids(
        connection,
        'parity.fold_forward_reference_idx',
        [1],
    )
    v3_hits = fetch_ids(
        connection,
        'parity.fold_forward_v3_idx',
        [1],
    )
    assert_rows_close(reference_hits, v3_hits, label='fold-forward compaction')
    status = fetch_status(
        connection,
        'parity.fold_forward_v3_idx',
    )
    if (
        len(positive_result_ids(v3_hits)) != 3
        or status['generation']['delta']['pending']['records'] != 0
        or status['generation']['delta']['active']['records'] > 1
        or status['generation']['primary']['segment_count'] != 1
    ):
        raise AssertionError(
            'fold-forward compaction left incomplete state: '
            f'{status}, hits={v3_hits}, maintenance={maintenance}'
        )
    return {
        'fold_probe': fold_probe,
        'maintenance': maintenance,
        'fold_forward': fold_forward,
        'compaction': compaction,
        'status': status,
        'reference_hits': reference_hits,
        'v3_hits': v3_hits,
    }


def plan_uses_index(
    connection: psycopg.Connection[Any],
    table_name: str,
    index_name: str,
    query: str,
) -> bool:
    statement = (
        f'EXPLAIN (FORMAT JSON, COSTS OFF) '
        f'SELECT id FROM {table_name} '
        f'ORDER BY body <=> evoke_order_tokens(%s::regclass, %s) '
        f'ASC LIMIT 3'
    )
    with connection.cursor() as cursor:
        cursor.execute('SET enable_seqscan = false')
        cursor.execute(statement, (index_name, query))
        plan = cursor.fetchone()
        cursor.execute('RESET enable_seqscan')
    return plan is not None and index_name.split('.')[-1] in json.dumps(plan[0])


def ids_plan_uses_index(
    connection: psycopg.Connection[Any],
    table_name: str,
    index_name: str,
    query: list[int],
) -> bool:
    statement = (
        f'EXPLAIN (FORMAT JSON, COSTS OFF) '
        f'SELECT id FROM {table_name} '
        f'ORDER BY tokens <=> %s::int4[] ASC LIMIT 513'
    )
    with connection.cursor() as cursor:
        cursor.execute('SET enable_seqscan = false')
        cursor.execute(statement, (query,))
        plan = cursor.fetchone()
        cursor.execute('RESET enable_seqscan')
    return plan is not None and index_name.split('.')[-1] in json.dumps(plan[0])


def filtered_tokens_plan_uses_index(
    connection: psycopg.Connection[Any],
    table_name: str,
    index_name: str,
    filter_query: str,
    order_query: list[str],
) -> bool:
    statement = (
        f'EXPLAIN (FORMAT JSON, COSTS OFF) '
        f'SELECT id FROM {table_name} WHERE tokens @@ %s '
        f'ORDER BY tokens <=> %s::text[] ASC LIMIT 2'
    )
    with connection.cursor() as cursor:
        cursor.execute('SET enable_seqscan = false')
        cursor.execute('SET enable_bitmapscan = false')
        cursor.execute(statement, (filter_query, order_query))
        plan = cursor.fetchone()
        cursor.execute('RESET enable_bitmapscan')
        cursor.execute('RESET enable_seqscan')
    return plan is not None and index_name.split('.')[-1] in json.dumps(plan[0])


def predicate_plan_uses_index(
    connection: psycopg.Connection[Any],
    table_name: str,
    index_name: str,
    query: str,
) -> bool:
    statement = (
        f'EXPLAIN (FORMAT JSON, COSTS OFF) '
        f'SELECT id FROM {table_name} WHERE body @@ %s'
    )
    with connection.cursor() as cursor:
        cursor.execute('SET enable_seqscan = false')
        cursor.execute('SET enable_bitmapscan = false')
        try:
            cursor.execute(statement, (query,))
            plan = cursor.fetchone()
        finally:
            cursor.execute('RESET enable_bitmapscan')
            cursor.execute('RESET enable_seqscan')
    return plan is not None and index_name.split('.')[-1] in json.dumps(plan[0])


def assert_rows_close(
    left: list[tuple[int | str, float]],
    right: list[tuple[int | str, float]],
    *,
    label: str,
) -> None:
    if len(left) != len(right):
        raise AssertionError(
            f'{label}: row count differs: {len(left)} != {len(right)}'
        )
    for rank, (left_row, right_row) in enumerate(zip(left, right)):
        if left_row[0] != right_row[0]:
            raise AssertionError(
                f'{label}: rank {rank} id differs: '
                f'{left_row[0]} != {right_row[0]}; '
                f'left={left}, right={right}'
            )
        if not math.isclose(
            left_row[1],
            right_row[1],
            rel_tol=2e-6,
            abs_tol=2e-6,
        ):
            raise AssertionError(
                f'{label}: rank {rank} score differs: '
                f'{left_row[1]} != {right_row[1]}; '
                f'left={left}, right={right}'
            )


def collect_surfaces(
    connection: psycopg.Connection[Any],
    *,
    version: str,
) -> dict[str, list[list[float | int]]]:
    text_index = f'parity.docs_{version}_idx'
    id_index = f'parity.ids_{version}_idx'
    table_name = f'parity.docs_{version}'
    surfaces: dict[str, list[list[float | int]]] = {}
    for query in TEXT_QUERIES:
        surfaces[f'search:{query}'] = [
            [doc_id, score]
            for doc_id, score in fetch_search(
                connection,
                text_index,
                query,
            )
        ]
        surfaces[f'ordered:{query}'] = [
            [doc_id, score]
            for doc_id, score in fetch_ordered(
                connection,
                table_name,
                text_index,
                query,
            )
        ]
    for query in RAW_TEXT_QUERIES:
        surfaces[f'raw:{query}'] = [
            [doc_id, score]
            for doc_id, score in fetch_search(
                connection,
                text_index,
                query,
            )
        ]
    for query in TOKEN_QUERIES:
        surfaces[f'tokens:{" ".join(query)}'] = [
            [doc_id, score]
            for doc_id, score in fetch_tokens(
                connection,
                text_index,
                query,
            )
        ]
    for query in ID_QUERIES:
        surfaces[f'ids:{query}'] = [
            [doc_id, score]
            for doc_id, score in fetch_ids(
                connection,
                id_index,
                list(query),
            )
        ]
    return surfaces


def assert_surface_maps_close(
    left: dict[str, list[list[float | int]]],
    right: dict[str, list[list[float | int]]],
    *,
    label: str,
) -> None:
    if left.keys() != right.keys():
        raise AssertionError(f'{label}: surface names differ')
    for name in left:
        assert_rows_close(
            [(int(row[0]), float(row[1])) for row in left[name]],
            [(int(row[0]), float(row[1])) for row in right[name]],
            label=f'{label}:{name}',
        )


def positive_result_ids(
    rows: list[tuple[int, float]],
) -> set[int]:
    return {document_id for document_id, score in rows if score > 0.0}


def expect_error(
    action: Callable[[], None],
    expected_text: str,
) -> str:
    try:
        action()
    except psycopg.Error as exc:
        message = str(exc)
        if expected_text not in message:
            raise AssertionError(
                f'expected error containing {expected_text!r}, got {message!r}'
            ) from exc
        return message.splitlines()[0]
    raise AssertionError(f'expected error containing {expected_text!r}')


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    extension_libdir = args.extension_libdir.expanduser().resolve()
    extension_control_dir = extension_control_root(
        args.extension_control_dir
    )
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
    root = create_short_socket_root('evoke-v3-read-')
    data_dir = root / 'data'
    socket_dir = root / 's'
    log_path = root / 'postgres.log'
    port = reserve_port()
    socket_dir.mkdir()
    gates: dict[str, bool] = {}
    evidence: dict[str, Any] = {}
    started = False
    connection: psycopg.Connection[Any] | None = None

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
        setup(connection)

        initial_reference_status = fetch_status(connection, 'parity.docs_reference_idx')
        initial_v3_status = fetch_status(connection, 'parity.docs_v3_idx')
        gates['reference_and_subject_use_page_native_storage'] = (
            initial_reference_status['generation']['layout'].get('storage')
            == 'convergent_segments'
            and initial_v3_status['generation']['layout'].get('storage')
            == 'convergent_segments'
        )
        if not gates['reference_and_subject_use_page_native_storage']:
            raise AssertionError(
                'reference and subject builds are not page-native: '
                f'reference={initial_reference_status}, v3={initial_v3_status}'
            )
        evidence['initial_storage_layouts'] = {
            'reference': initial_reference_status['generation']['layout'],
            'subject': initial_v3_status['generation']['layout'],
        }

        reference_surfaces = collect_surfaces(connection, version='reference')
        v3_surfaces = collect_surfaces(connection, version='v3')
        assert_surface_maps_close(
            reference_surfaces,
            v3_surfaces,
            label='page-native reference parity',
        )
        gates['page_native_reference_rank_and_score_parity'] = True
        evidence['v3_surfaces'] = v3_surfaces

        status = fetch_status(connection, 'parity.docs_v3_idx')
        generation = status['generation']
        gates['v3_status_is_query_ready'] = (
            status['query_ready'] is True
            and status['runtime_signature_matches'] is True
            and generation['valid'] is True
            and generation['layout']['storage'] == 'convergent_segments'
            and generation['primary']['segment_count'] == 0
            and generation['primary']['fragmented'] is True
            and generation['primary']['start_block'] == 0
            and generation['primary']['pages'] == 0
            and generation['primary']['manifest_start_block'] > 0
            and generation['primary']['manifest_pages'] > 0
            and generation['primary']['published_block_high_watermark'] > 1
            and status['details']['index_bytes'] > 0
        )
        if not gates['v3_status_is_query_ready']:
            raise AssertionError(f'invalid v3 status: {status}')
        gates['native_ordered_scan_uses_v3_index'] = plan_uses_index(
            connection,
            'parity.docs_v3',
            'parity.docs_v3_idx',
            TEXT_QUERIES[0],
        )
        if not gates['native_ordered_scan_uses_v3_index']:
            raise AssertionError('v3 ORDER BY did not use the evoke index')

        high_df_prefix = fetch_ids_ordered(
            connection,
            'parity.high_df_v3',
            [0],
            limit=255,
            force_index=True,
        )
        high_df_expanded = fetch_ids_ordered(
            connection,
            'parity.high_df_v3',
            [0],
            limit=HIGH_DF_ROWS,
            force_index=True,
        )
        assert_rows_close(
            high_df_prefix,
            high_df_expanded[:len(high_df_prefix)],
            label='page-native ordered rank-window prefix',
        )
        gates['high_df_ordered_scan_uses_v3_index'] = ids_plan_uses_index(
            connection,
            'parity.high_df_v3',
            'parity.high_df_v3_idx',
            [0],
        )
        gates['page_native_ordered_rank_window_expands_exactly'] = (
            len(high_df_prefix) == 255
            and len(high_df_expanded) == HIGH_DF_ROWS
            and {row[0] for row in high_df_expanded}
            == set(range(1, HIGH_DF_ROWS + 1))
            and all(math.isfinite(row[1]) for row in high_df_expanded)
        )
        if (
            not gates['high_df_ordered_scan_uses_v3_index']
            or not gates['page_native_ordered_rank_window_expands_exactly']
        ):
            raise AssertionError(
                'v3 ordered rank-window expansion failed: '
                f'prefix={len(high_df_prefix)}, '
                f'expanded={len(high_df_expanded)}'
            )
        evidence['page_native_ordered_rank_window'] = {
            'initial_prefix_rows': len(high_df_prefix),
            'expanded_rows': len(high_df_expanded),
            'unique_document_ids': len(
                {row[0] for row in high_df_expanded}
            ),
            'first_row': high_df_expanded[0],
            'last_row': high_df_expanded[-1],
        }

        filtered_parity: dict[str, list[tuple[int, float]]] = {}
        for filter_query in (
            'blue OR fox',
            '"blue bird" OR (fox AND forest)',
        ):
            reference_filtered = fetch_filtered_ordered(
                connection,
                'parity.docs_reference',
                'parity.docs_reference_idx',
                filter_query,
                'blue bird river',
                limit=6,
                force_index=True,
            )
            v3_filtered = fetch_filtered_ordered(
                connection,
                'parity.docs_v3',
                'parity.docs_v3_idx',
                filter_query,
                'blue bird river',
                limit=6,
                force_index=True,
            )
            assert_rows_close(
                reference_filtered,
                v3_filtered,
                label=f'page-native filtered ordered:{filter_query}',
            )
            filtered_parity[filter_query] = v3_filtered
        gates['page_native_filtered_ordered_matches_reference'] = True

        late_filtered_hits = fetch_filtered_tokens_ordered(
            connection,
            'parity.filtered_v3',
            'accept',
            ['rank'],
            limit=2,
        )
        gates['filtered_ordered_scan_uses_v3_index'] = (
            filtered_tokens_plan_uses_index(
                connection,
                'parity.filtered_v3',
                'parity.filtered_v3_idx',
                'accept',
                ['rank'],
            )
        )
        gates['page_native_filtered_ordered_expands_to_late_matches'] = (
            [row[0] for row in late_filtered_hits] == [257, 513]
            and all(math.isfinite(row[1]) for row in late_filtered_hits)
        )
        if (
            not gates['filtered_ordered_scan_uses_v3_index']
            or not gates[
                'page_native_filtered_ordered_expands_to_late_matches'
            ]
        ):
            raise AssertionError(
                'v3 filtered ordered expansion failed: '
                f'hits={late_filtered_hits}'
            )
        evidence['page_native_filtered_ordered'] = {
            'reference_v3_parity': filtered_parity,
            'late_matches': late_filtered_hits,
        }

        predicate_parity: dict[str, list[int]] = {}
        for predicate_query in (
            'blue',
            '"blue bird"',
            '(blue OR fox)',
            '(blue OR fox) AND NOT red',
            'retriev*',
            'NOT blue',
        ):
            reference_predicate = fetch_predicate_ids(
                connection,
                'parity.predicate_reference',
                predicate_query,
                force_index=True,
            )
            v3_predicate = fetch_predicate_ids(
                connection,
                'parity.predicate_v3',
                predicate_query,
                force_index=True,
            )
            heap_predicate = fetch_heap_predicate_ids(
                connection,
                'parity.predicate_v3',
                predicate_query,
            )
            if (
                reference_predicate != v3_predicate or
                v3_predicate != heap_predicate
            ):
                lexicon_probe = {
                    term: fetch_page_native_lexicon(
                        connection,
                        'parity.predicate_v3_idx',
                        term,
                    )
                    for term in ('blue', 'fox', 'red')
                }
                positive_ids = [
                    int(lexicon_probe[term]['term_id'])
                    for term in ('blue', 'fox')
                ]
                topk_probe = fetch_page_native_query_topk(
                    connection,
                    'parity.predicate_v3_idx',
                    positive_ids,
                    len(PREDICATE_ROWS),
                )
                raise AssertionError(
                    'page-native predicate differs from page-native reference: '
                    f'query={predicate_query!r}, '
                    f'reference={reference_predicate}, v3={v3_predicate}, '
                    f'heap={heap_predicate}, '
                    f'lexicon={lexicon_probe}, topk={topk_probe}'
                )
            if not predicate_plan_uses_index(
                connection,
                'parity.predicate_v3',
                'parity.predicate_v3_idx',
                predicate_query,
            ):
                raise AssertionError(
                    'v3 predicate did not use the evoke index: '
                    f'{predicate_query!r}'
                )
            predicate_parity[predicate_query] = v3_predicate
        gates['page_native_predicate_matches_reference'] = True
        gates['page_native_predicate_uses_v3_index'] = True
        evidence['page_native_predicate'] = predicate_parity

        empty_status = fetch_status(connection, 'parity.empty_v3_idx')
        empty_hits = fetch_search(
            connection,
            'parity.empty_v3_idx',
            'anything',
        )
        gates['empty_v3_index_is_query_ready'] = (
            empty_status['query_ready'] is True
            and empty_status['generation']['docs'] == 0
            and empty_hits == []
        )
        if not gates['empty_v3_index_is_query_ready']:
            raise AssertionError(
                f'empty v3 index failed: {empty_status}, {empty_hits}'
            )

        tail_cleanup = exercise_unpublished_tail_cleanup(connection)
        gates['unpublished_tail_cleanup_respects_old_readers'] = True
        evidence['unpublished_tail_cleanup'] = tail_cleanup

        safe_retirement = exercise_query_safe_retirement(connection)
        gates['online_retirement_respects_old_readers'] = True
        gates['bounded_retirement_hints_remain_exactly_accounted'] = True
        gates['successful_transition_retirement_is_exact'] = True
        evidence['query_safe_retirement'] = safe_retirement

        cow_probe = write_term_cow_probe(
            connection,
            'parity.docs_v3_idx',
        )
        cow_lookup = read_term_cow_probe(
            connection,
            'parity.docs_v3_idx',
            cow_probe,
        )
        gates['physical_cow_term_lookup_matches_flat_directory'] = (
            cow_probe['start_block'] > 0
            and cow_probe['page_count'] > 0
            and cow_probe['object_count'] > 1
            and cow_lookup['term_id'] == cow_probe['term_id']
            and cow_lookup['raw_document_frequency']
            == cow_probe['raw_document_frequency']
            and cow_lookup['extent_count'] == cow_probe['extent_count']
        )
        if not gates['physical_cow_term_lookup_matches_flat_directory']:
            raise AssertionError(
                f'physical COW lookup differs: {cow_probe}, {cow_lookup}'
            )
        evidence['physical_cow_probe'] = cow_probe
        page_native_context = fetch_page_native_context(
            connection,
            'parity.docs_v3_idx',
        )
        gates['page_native_query_context_is_bounded'] = (
            page_native_context['bounded'] is True
            and page_native_context['vocabulary_materialized'] is False
            and page_native_context['doc_frequencies_materialized'] is False
            and page_native_context['cow_term_directory'] is True
            and page_native_context['cow_document_directory'] is True
            and page_native_context['lexicon_lookup'] is True
            and page_native_context['prefix_lookup'] is True
            and page_native_context['segment_count'] == 0
            and int(page_native_context['document_slot_count']) > 0
        )
        if not gates['page_native_query_context_is_bounded']:
            raise AssertionError(
                'page-native query context is not bounded: '
                f'{page_native_context}'
            )
        evidence['page_native_query_context'] = page_native_context
        page_native_before_fold = fetch_page_native_term(
            connection,
            'parity.docs_v3_idx',
            int(cow_probe['term_id']),
        )
        gates['page_native_term_matches_initial_folded_snapshot'] = (
            page_native_before_fold['matched'] is True
            and page_native_before_fold['cursor_matched'] is True
            and page_native_before_fold['plan_run_count']
            == page_native_before_fold['extent_count']
            and page_native_before_fold['max_block_postings'] <= 128
            and page_native_before_fold['extent_count'] > 0
            and cow_lookup['extent_count'] == 0
        )
        if not gates['page_native_term_matches_initial_folded_snapshot']:
            raise AssertionError(
                'page-native term differs from the initial folded snapshot: '
                f'{page_native_before_fold}'
            )
        evidence['page_native_before_fold'] = page_native_before_fold

        page_native_high_df = fetch_page_native_term(
            connection,
            'parity.high_df_v3_idx',
            0,
        )
        gates['page_native_high_df_cursor_is_block_bounded'] = (
            page_native_high_df['matched'] is True
            and page_native_high_df['cursor_matched'] is True
            and int(page_native_high_df['raw_document_frequency'])
            == HIGH_DF_ROWS
            and int(page_native_high_df['posting_count']) == HIGH_DF_ROWS
            and int(page_native_high_df['block_count']) > 1
            and int(page_native_high_df['max_block_postings']) <= 128
        )
        if not gates['page_native_high_df_cursor_is_block_bounded']:
            raise AssertionError(
                'high-DF page-native cursor is not block bounded: '
                f'{page_native_high_df}'
            )
        evidence['page_native_high_df'] = page_native_high_df

        page_native_high_df_topk = fetch_page_native_topk(
            connection,
            'parity.high_df_v3_idx',
            0,
            10,
        )
        gates['page_native_single_term_topk_matches_snapshot'] = (
            page_native_high_df_topk['matched'] is True
            and int(page_native_high_df_topk['live_df']) == HIGH_DF_ROWS
            and int(page_native_high_df_topk['heap_capacity']) == 10
            and int(page_native_high_df_topk['block_reads']) == 10
            and int(page_native_high_df_topk['document_block_reads']) == 10
            and int(
                page_native_high_df_topk['max_document_block_records']
            ) <= 128
        )
        if not gates['page_native_single_term_topk_matches_snapshot']:
            raise AssertionError(
                'page-native one-term top-k differs from snapshot: '
                f'{page_native_high_df_topk}'
            )
        evidence['page_native_high_df_topk'] = page_native_high_df_topk

        page_native_multi_term_topk = fetch_page_native_query_topk(
            connection,
            'parity.high_df_v3_idx',
            [0, 1000],
            10,
            [1.0, 0.5],
        )
        gates['page_native_multi_term_topk_matches_snapshot'] = (
            page_native_multi_term_topk['matched'] is True
            and page_native_multi_term_topk['positive_topk_complete'] is True
            and page_native_multi_term_topk['topk_complete'] is True
            and int(page_native_multi_term_topk['query_term_count']) == 2
            and int(page_native_multi_term_topk['query_run_count']) == 2
            and int(
                page_native_multi_term_topk['max_document_block_records']
            ) <= 128
            and int(page_native_multi_term_topk['blocks_considered']) > 0
            and int(page_native_multi_term_topk['blocks_skipped']) > 0
            and int(page_native_multi_term_topk['blocks_scored'])
            + int(page_native_multi_term_topk['blocks_skipped'])
            == int(page_native_multi_term_topk['blocks_considered'])
            and int(page_native_multi_term_topk['document_block_reads']) < 10
            and int(page_native_multi_term_topk['postings_skipped']) > 0
            and int(page_native_multi_term_topk['documents_examined'])
            < HIGH_DF_ROWS * 2
        )
        if not gates['page_native_multi_term_topk_matches_snapshot']:
            raise AssertionError(
                'page-native multi-term top-k differs from snapshot: '
                f'{page_native_multi_term_topk}'
            )
        evidence['page_native_multi_term_topk'] = (
            page_native_multi_term_topk
        )

        page_native_block_cost = fetch_page_native_block_cost(
            connection,
            'parity.high_df_v3_idx',
            [0, 1000],
            [1.0, 0.5],
            10,
        )
        block_cost_levels = page_native_block_cost['levels']
        hierarchy = page_native_block_cost['hierarchy']
        hierarchy_levels = page_native_block_cost['hierarchy_levels']
        essential_projection = page_native_block_cost[
            'essential_term_projection'
        ]
        essential_points = essential_projection['points']
        optimal_sparse = hierarchy['optimal_sparse']
        optimal_dense = hierarchy['optimal_dense']
        gates['page_native_block_cost_oracle_is_bounded'] = (
            int(page_native_block_cost['query_terms']) == 2
            and int(page_native_block_cost['query_runs']) == 2
            and int(page_native_block_cost['query_postings']) > 0
            and [int(level['block_size']) for level in block_cost_levels]
            == [8, 16, 32, 64, 128]
            and all(
                int(level['competitive_postings'])
                <= int(level['total_postings'])
                == int(page_native_block_cost['query_postings'])
                and int(level['competitive_blocks'])
                <= int(level['block_count'])
                and 0.0 <= float(level['posting_fraction']) <= 1.0
                and 0.0 <= float(level['document_fraction']) <= 1.0
                for level in block_cost_levels
            )
            and len(hierarchy_levels) > len(block_cost_levels)
            and int(hierarchy['nodes_read']) > 0
            and int(hierarchy['sparse_term_hits']) > 0
            and int(hierarchy['dense_term_probes'])
            >= int(hierarchy['sparse_term_hits'])
            and int(hierarchy['leaf_blocks'])
            == int(block_cost_levels[0]['competitive_blocks'])
            and int(hierarchy['leaf_postings'])
            == int(block_cost_levels[0]['competitive_postings'])
            and int(optimal_sparse['term_hits'])
            <= int(block_cost_levels[0]['touched_term_blocks'])
            and (int(optimal_sparse['level_mask']) & 1) == 1
            and int(optimal_dense['term_probes'])
            <= int(hierarchy_levels[0]['block_count']) * 2
            and (int(optimal_dense['level_mask']) & 1) == 1
            and int(essential_projection['mandatory_terms']) == 2
            and int(essential_projection['projectable_semantic_terms']) == 0
            and len(essential_points) == 1
            and int(essential_points[0]['essential_semantic_terms']) == 0
            and int(essential_points[0]['residual_semantic_terms']) == 0
            and float(essential_points[0]['residual_global_cap']) == 0.0
            and int(
                essential_points[0]['missed_full_competitive_blocks']
            ) == 0
        )
        if not gates['page_native_block_cost_oracle_is_bounded']:
            raise AssertionError(
                'page-native block-cost oracle is invalid: '
                f'{page_native_block_cost}'
            )
        evidence['page_native_block_cost'] = page_native_block_cost

        ordered_query_ids = [0, 1000, 1001, 1002, 1003, 2001, 2002, 2003]
        page_native_ordered_topk = fetch_page_native_query_topk(
            connection,
            'parity.high_df_v3_idx',
            ordered_query_ids,
            10,
        )
        gates['page_native_ordered_block_topk_is_exact_and_bounded'] = (
            page_native_ordered_topk['matched'] is True
            and page_native_ordered_topk['materialized_query_path'] is False
            and page_native_ordered_topk['ordered_block_query_path'] is True
            and 0 < int(
                page_native_ordered_topk[
                    'posting_block_metadata_cache_bytes'
                ]
            ) <= 32 * 1024 * 1024
            and int(page_native_ordered_topk['blocks_considered']) > 0
            and int(page_native_ordered_topk['blocks_skipped']) > 0
            and int(page_native_ordered_topk['postings_skipped']) > 0
            and int(page_native_ordered_topk['query_term_count']) == 8
        )
        if not gates[
            'page_native_ordered_block_topk_is_exact_and_bounded'
        ]:
            raise AssertionError(
                'page-native ordered block top-k is not exact and bounded: '
                f'{page_native_ordered_topk}'
            )
        evidence['page_native_ordered_block_topk'] = (
            page_native_ordered_topk
        )

        materialized_query_ids = [0, 1000] * 5
        materialized_query_weights = [0.2, 0.1] * 5
        page_native_materialized_topk = fetch_page_native_query_topk(
            connection,
            'parity.high_df_v3_idx',
            materialized_query_ids,
            10,
            materialized_query_weights,
        )
        gates['page_native_materialized_topk_is_exact_and_bounded'] = (
            page_native_materialized_topk['matched'] is True
            and page_native_materialized_topk[
                'materialized_query_path'
            ] is True
            and 0 < int(
                page_native_materialized_topk[
                    'materialized_query_bytes'
                ]
            ) <= 8 * 1024 * 1024
            and int(
                page_native_materialized_topk[
                    'posting_block_metadata_reads'
                ]
            ) == 0
            and int(
                page_native_materialized_topk['posting_block_reads']
            ) > 0
            and int(
                page_native_materialized_topk['postings_examined']
            ) > 0
            and int(
                page_native_materialized_topk['document_block_reads']
            ) > 0
            and int(page_native_materialized_topk['query_term_count']) == 10
            and int(page_native_materialized_topk['blocks_considered'])
            == int(page_native_materialized_topk['blocks_scored'])
        )
        if not gates[
            'page_native_materialized_topk_is_exact_and_bounded'
        ]:
            raise AssertionError(
                'page-native materialized top-k is not exact and bounded: '
                f'{page_native_materialized_topk}'
            )
        evidence['page_native_materialized_topk'] = (
            page_native_materialized_topk
        )

        term_at_a_time_query_ids = [0] * 2_100
        page_native_term_at_a_time_topk = fetch_page_native_query_topk(
            connection,
            'parity.high_df_v3_idx',
            term_at_a_time_query_ids,
            10,
        )
        gates['page_native_term_at_a_time_topk_is_exact_and_bounded'] = (
            page_native_term_at_a_time_topk['matched'] is True
            and page_native_term_at_a_time_topk[
                'materialized_query_path'
            ] is False
            and page_native_term_at_a_time_topk[
                'term_at_a_time_query_path'
            ] is True
            and page_native_term_at_a_time_topk[
                'ordered_block_query_path'
            ] is False
            and 0 < int(
                page_native_term_at_a_time_topk[
                    'term_at_a_time_score_bytes'
                ]
            ) <= 64 * 1024 * 1024
            and int(
                page_native_term_at_a_time_topk['query_term_count']
            ) == len(term_at_a_time_query_ids)
            and int(
                page_native_term_at_a_time_topk['posting_block_reads']
            ) > 0
            and int(
                page_native_term_at_a_time_topk['blocks_considered']
            ) > 0
        )
        if not gates[
            'page_native_term_at_a_time_topk_is_exact_and_bounded'
        ]:
            raise AssertionError(
                'page-native TAAT top-k is not exact and bounded: '
                f'{page_native_term_at_a_time_topk}'
            )
        evidence['page_native_term_at_a_time_topk'] = (
            page_native_term_at_a_time_topk
        )

        page_native_zero_score_topk = fetch_page_native_query_topk(
            connection,
            'parity.high_df_v3_idx',
            [2_147_483_647],
            10,
        )
        gates['page_native_zero_score_topk_is_bounded'] = (
            page_native_zero_score_topk['matched'] is True
            and page_native_zero_score_topk['topk_complete'] is True
            and page_native_zero_score_topk[
                'positive_topk_complete'
            ] is False
            and int(
                page_native_zero_score_topk['zero_score_documents_added']
            ) == 10
            and 0 < int(
                page_native_zero_score_topk[
                    'zero_score_cow_records_examined'
                ]
            ) < HIGH_DF_ROWS
            and 0 < int(
                page_native_zero_score_topk['zero_score_heap_peak']
            ) < HIGH_DF_ROWS
        )
        if not gates['page_native_zero_score_topk_is_bounded']:
            raise AssertionError(
                'page-native zero-score top-k is not exact and bounded: '
                f'{page_native_zero_score_topk}'
            )
        evidence['page_native_zero_score_topk'] = (
            page_native_zero_score_topk
        )

        folded_surfaces = collect_surfaces(connection, version='v3')
        assert_surface_maps_close(
            v3_surfaces,
            folded_surfaces,
            label='initial folded publication',
        )
        gates['initial_folded_publication_is_exact'] = True
        evidence['page_native_after_fold'] = page_native_before_fold
        restart_cow_probe = write_term_cow_probe(
            connection,
            'parity.docs_v3_idx',
        )
        restart_cow_lookup = read_term_cow_probe(
            connection,
            'parity.docs_v3_idx',
            restart_cow_probe,
        )
        evidence['restart_physical_cow_probe'] = restart_cow_probe
        interior_status = fetch_status(
            connection,
            'parity.docs_v3_idx',
        )
        interior_primary = interior_status['generation']['primary']
        gates['physical_block_accounting_is_exact'] = (
            interior_primary['trailing_unpublished_blocks'] == 0
            and interior_primary['physical_blocks']
            == interior_primary['published_block_high_watermark']
            and interior_primary['reachable_blocks']
            + interior_primary['interior_unreachable_blocks']
            == interior_primary['published_block_high_watermark']
        )
        if not gates['physical_block_accounting_is_exact']:
            raise AssertionError(
                'physical block ownership was not classified exactly: '
                f'{interior_status}'
            )
        evidence['interior_orphan_status'] = interior_status
        posting_heat_batching = exercise_posting_heat_query_batching(
            connection,
        )
        gates['posting_heat_shared_lock_is_query_batched'] = True
        evidence['posting_heat_query_batching'] = posting_heat_batching
        scheduler = exercise_unified_maintenance_scheduler(connection)
        gates['public_and_worker_scheduler_share_action_priority'] = True
        evidence['unified_maintenance_scheduler'] = scheduler
        manual_v3 = exercise_manual_v3_lifecycle(connection)
        gates['manual_v3_debt_and_refresh_are_exact'] = True
        evidence['manual_v3_lifecycle'] = manual_v3
        retirement_statistics = exercise_vacuum_retirement_statistics(
            connection,
        )
        gates['v3_retirement_statistics_converge_after_vacuum'] = True
        evidence['vacuum_retirement_statistics'] = retirement_statistics
        structural_fold = exercise_structural_term_fold(connection)
        gates['hard_extent_pressure_uses_bounded_structural_fold'] = True
        gates['page_native_multi_run_topk_matches_snapshot'] = True
        evidence['structural_term_fold'] = structural_fold
        filtered_block_pruning = exercise_filtered_block_pruning(connection)
        gates['filtered_topk_prunes_empty_candidate_blocks'] = True
        evidence['filtered_block_pruning'] = filtered_block_pruning
        workload_fold = exercise_workload_term_fold(connection)
        gates['workload_heat_drives_background_cow_fold'] = True
        gates['geometric_term_folds_bound_write_amplification'] = True
        evidence['workload_term_fold'] = workload_fold
        impact_specialization = exercise_impact_specialization(connection)
        gates['epoch_bound_impact_specialization_is_exact'] = True
        gates['l0_invalidates_impact_specialization_immediately'] = True
        gates['maintenance_rebuilds_impact_after_convergence'] = True
        gates['cache_clear_reloads_impact_specialization'] = True
        gates['retirement_aware_impact_sparse_path_is_exact'] = True
        evidence['impact_specialization'] = impact_specialization
        pending_headroom = exercise_pending_l0_headroom(connection)
        gates['pending_l0_retains_reserved_active_headroom'] = True
        evidence['pending_l0_headroom'] = pending_headroom
        fold_safe_compaction = exercise_fold_safe_compaction(connection)
        gates['post_fold_tail_compaction_is_safe'] = True
        evidence['fold_safe_compaction'] = fold_safe_compaction
        fold_forward_compaction = exercise_fold_forward_compaction(
            connection,
        )
        gates['fold_boundary_compaction_converges'] = True
        evidence['fold_forward_compaction'] = fold_forward_compaction
        history_barrier = exercise_history_barrier_compaction(connection)
        gates['empty_compaction_history_barrier_is_query_ready'] = True
        evidence['history_barrier_compaction'] = history_barrier

        connection.close()
        connection = None
        crash_cluster(pg_ctl, data_dir)
        started = False
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        restarted_surfaces = collect_surfaces(connection, version='v3')
        assert_surface_maps_close(
            folded_surfaces,
            restarted_surfaces,
            label='v3 restart',
        )
        restarted_history_barrier_hits = fetch_ids_by_tid(
            connection,
            'parity.history_barrier_v3_idx',
            [1],
        )
        assert_rows_close(
            history_barrier['converged_hits'],
            restarted_history_barrier_hits,
            label='history-barrier restart',
        )
        gates['restart_preserves_history_barrier_replacement'] = True
        evidence['restarted_history_barrier_hits'] = (
            restarted_history_barrier_hits
        )
        gates['restart_preserves_v3_surfaces'] = True
        restarted_tail_status = fetch_status(
            connection,
            'parity.tail_v3_idx',
        )
        restarted_tail_hits = fetch_search(
            connection,
            'parity.tail_v3_idx',
            TEXT_QUERIES[0],
        )
        assert_rows_close(
            tail_cleanup['hits'],
            restarted_tail_hits,
            label='unpublished tail cleanup restart',
        )
        restarted_tail_primary = (
            restarted_tail_status['generation']['primary']
        )
        gates['restart_preserves_tail_cleanup'] = (
            restarted_tail_primary['trailing_unpublished_blocks'] == 0
            and restarted_tail_primary['physical_blocks']
            == restarted_tail_primary['published_block_high_watermark']
        )
        if not gates['restart_preserves_tail_cleanup']:
            raise AssertionError(
                'restart restored unpublished tail debt: '
                f'{restarted_tail_status}'
            )
        evidence['restarted_tail_cleanup_status'] = (
            restarted_tail_status
        )
        restarted_reuse_hits = fetch_ids(
            connection,
            'parity.reuse_v3_idx',
            [1],
        )
        restarted_reuse_status = fetch_status(
            connection,
            'parity.reuse_v3_idx',
        )
        restarted_reuse_primary = (
            restarted_reuse_status['generation']['primary']
        )
        assert_rows_close(
            safe_retirement['final_hits'],
            restarted_reuse_hits,
            label='query-safe retirement restart',
        )
        gates['restart_preserves_query_safe_retirement'] = (
            restarted_reuse_primary['trailing_unpublished_blocks'] == 0
            and restarted_reuse_primary['reachable_blocks']
            + restarted_reuse_primary['interior_unreachable_blocks']
            == restarted_reuse_primary['published_block_high_watermark']
        )
        if not gates['restart_preserves_query_safe_retirement']:
            raise AssertionError(
                'restart changed query-safe retirement closure: '
                f'{restarted_reuse_status}'
            )
        evidence['restarted_query_safe_retirement'] = {
            'status': restarted_reuse_status,
            'hits': restarted_reuse_hits,
        }
        restarted_cow_lookup = read_term_cow_probe(
            connection,
            'parity.docs_v3_idx',
            restart_cow_probe,
        )
        gates['restart_preserves_physical_cow_term_lookup'] = (
            restarted_cow_lookup == restart_cow_lookup
        )
        if not gates['restart_preserves_physical_cow_term_lookup']:
            raise AssertionError(
                'restart changed physical COW lookup: '
                f'{restart_cow_lookup} != {restarted_cow_lookup}'
            )
        restarted_structural_hits = fetch_ids(
            connection,
            'parity.structural_v3_idx',
            [1],
        )
        assert_rows_close(
            structural_fold['v3_hits'],
            restarted_structural_hits,
            label='structural-fold restart',
        )
        gates['restart_preserves_structural_fold_and_tail'] = True
        restarted_hot_cache_before = fetch_generation_cache_state(
            connection,
            'parity.workload_v3_idx',
        )
        if 'shared_hot_fold_current=false' not in restarted_hot_cache_before:
            raise AssertionError(
                'restart unexpectedly retained process-local hot-fold state: '
                f'{restarted_hot_cache_before}'
            )
        acquire_maintenance_guard(
            connection,
            'parity.workload_v3_idx',
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    'SET evoke.test_force_structural_term_fold = true'
                )
            expected_workload_hits = fetch_ids_by_tid(
                connection,
                'parity.workload_reference_idx',
                [1],
            )
            for _ in range(96):
                assert_rows_close(
                    expected_workload_hits,
                    fetch_ids_by_tid(
                        connection,
                        'parity.workload_v3_idx',
                        [1],
                    ),
                    label='impact-specialization restart heat',
                )
            restarted_hot_cache_candidate = fetch_status(
                connection,
                'parity.workload_v3_idx',
            )['generation']['workload_fold']
            if restarted_hot_cache_candidate['candidate'] is not True:
                raise AssertionError(
                    'restart did not re-observe the persisted hot term: '
                    f'{restarted_hot_cache_candidate}'
                )
            restarted_hot_cache_result = try_maintain(
                connection,
                'parity.workload_v3_idx',
            )
            restarted_hot_cache_fields = maintenance_result_fields(
                restarted_hot_cache_result
            )
            restarted_hot_cache_after = fetch_generation_cache_state(
                connection,
                'parity.workload_v3_idx',
            )
            if (
                restarted_hot_cache_fields.get('maintained') != 'true'
                or restarted_hot_cache_fields.get('mode')
                != 'term_impact_specialization'
                or restarted_hot_cache_fields.get('reason')
                != 'impact_specialization_hot_cache'
                or 'shared_hot_fold_current=true'
                not in restarted_hot_cache_after
            ):
                raise AssertionError(
                    'restart maintenance did not republish the hot fold: '
                    f'{restarted_hot_cache_result}, '
                    f'{restarted_hot_cache_after}'
                )
        finally:
            with connection.cursor() as cursor:
                cursor.execute(
                    'RESET evoke.test_force_structural_term_fold'
                )
            if maintenance_guard_is_held(
                connection,
                'parity.workload_v3_idx',
            ):
                release_maintenance_guard(
                    connection,
                    'parity.workload_v3_idx',
                )
        restarted_workload_hits = fetch_ids(
            connection,
            'parity.workload_v3_idx',
            [1],
        )
        assert_rows_close(
            impact_specialization['final_hits'],
            restarted_workload_hits,
            label='workload-fold restart',
        )
        restarted_workload_status = fetch_status(
            connection,
            'parity.workload_v3_idx',
        )
        if (
            restarted_workload_status['generation']['generation']
            != impact_specialization['final_status']['generation'][
                'generation'
            ]
            or restarted_workload_status['generation']['workload_fold'][
                'candidate'
            ]
            is True
        ):
            raise AssertionError(
                'restart changed workload fold or restored stale heat: '
                f'{restarted_workload_status}'
        )
        gates['restart_preserves_workload_fold'] = True
        evidence['restarted_workload_status'] = (
            restarted_workload_status
        )
        restarted_impact_hits = fetch_ids(
            connection,
            'parity.workload_v3_idx',
            [1],
        )
        assert_rows_close(
            impact_specialization['final_hits'],
            restarted_impact_hits,
            label='impact-specialization restart',
        )
        restarted_impact_state = fetch_term_fold_state(
            connection,
            'parity.workload_v3_idx',
            1,
        )
        if (
            restarted_impact_state['impact_selected'] is not True
            or restarted_impact_state['impact_sparse_eligible'] is not True
            or restarted_impact_state['retired_documents'] < 1
            or restarted_impact_state['impact_statistics_epoch']
            != restarted_impact_state['statistics_epoch']
        ):
            raise AssertionError(
                'restart did not restore the exact impact read image: '
                f'{restarted_impact_state}'
            )
        gates['restart_preserves_impact_specialization'] = True
        gates['restart_republishes_shared_hot_fold'] = True
        evidence['restarted_impact_specialization'] = {
            'hits': restarted_impact_hits,
            'state': restarted_impact_state,
            'hot_cache_before': restarted_hot_cache_before,
            'hot_cache_candidate': restarted_hot_cache_candidate,
            'hot_cache_result': restarted_hot_cache_result,
            'hot_cache_after': restarted_hot_cache_after,
        }
        restarted_fold_safe_hits = fetch_ids(
            connection,
            'parity.fold_compaction_v3_idx',
            [1],
        )
        assert_rows_close(
            fold_safe_compaction['v3_hits'],
            restarted_fold_safe_hits,
            label='fold-safe compaction restart',
        )
        gates['restart_preserves_fold_safe_compaction'] = True
        restarted_fold_forward_hits = fetch_ids(
            connection,
            'parity.fold_forward_v3_idx',
            [1],
        )
        assert_rows_close(
            fold_forward_compaction['v3_hits'],
            restarted_fold_forward_hits,
            label='fold-forward compaction restart',
        )
        gates['restart_preserves_fold_forward_compaction'] = True

        with connection.cursor() as cursor:
            cursor.execute('ALTER INDEX parity.docs_v3_idx SET (k1 = 2.0)')
        contract_error = expect_error(
            lambda: fetch_search(
                connection,
                'parity.docs_v3_idx',
                TEXT_QUERIES[0],
            ),
            'does not match the configured index options',
        )
        mismatch_status = fetch_status(connection, 'parity.docs_v3_idx')
        gates['contract_drift_fails_closed'] = (
            mismatch_status['query_ready'] is False
            and mismatch_status['runtime_signature_matches'] is False
            and mismatch_status['blocker'] == 'runtime_generation_mismatch'
            and mismatch_status['generation']['valid'] is True
            and mismatch_status['generation']['runtime_contract_matches']
            is False
        )
        if not gates['contract_drift_fails_closed']:
            raise AssertionError(
                f'contract drift was not reported: {mismatch_status}'
            )
        evidence['contract_error'] = contract_error

        with connection.cursor() as cursor:
            cursor.execute('ALTER INDEX parity.docs_v3_idx RESET (k1)')
            cursor.execute('REINDEX INDEX parity.docs_v3_idx')
        rebuilt_surfaces = collect_surfaces(connection, version='v3')
        assert_surface_maps_close(
            reference_surfaces,
            rebuilt_surfaces,
            label='v3 reindex',
        )
        gates['reindex_restores_v3_contract_and_parity'] = True

        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO parity.docs_v3 VALUES '
                "(7, 'new incremental lexical document')"
            )
            cursor.execute(
                'INSERT INTO parity.docs_v3 '
                "SELECT 8, string_agg('l0_unique_' || term_id::text, ' ') "
                'FROM generate_series(1, 2000) AS terms(term_id)'
            )
            docs_large_l0_status = fetch_status(
                connection,
                'parity.docs_v3_idx',
            )
            cursor.execute(
                'INSERT INTO parity.docs_v3 VALUES '
                "(9, 'small inline tail record')"
            )
            cursor.execute(
                'INSERT INTO parity.ids_v3 VALUES (5, ARRAY[4, 4, 8])'
            )
            cursor.execute('SELECT count(*) FROM parity.docs_v3')
            v3_count = int(cursor.fetchone()[0])
            cursor.execute('SELECT count(*) FROM parity.ids_v3')
            v3_id_count = int(cursor.fetchone()[0])
            cursor.execute(
                'INSERT INTO parity.docs_reference VALUES '
                "(7, 'new incremental lexical document')"
            )
            cursor.execute(
                'INSERT INTO parity.docs_reference '
                "SELECT 8, string_agg('l0_unique_' || term_id::text, ' ') "
                'FROM generate_series(1, 2000) AS terms(term_id)'
            )
            cursor.execute(
                'INSERT INTO parity.docs_reference VALUES '
                "(9, 'small inline tail record')"
            )
            cursor.execute(
                'INSERT INTO parity.ids_reference VALUES (5, ARRAY[4, 4, 8])'
            )
            cursor.execute('SELECT count(*) FROM parity.docs_reference')
            reference_count = int(cursor.fetchone()[0])
            cursor.execute('SELECT count(*) FROM parity.ids_reference')
            reference_id_count = int(cursor.fetchone()[0])
        reference_incremental_hits = fetch_search(
            connection,
            'parity.docs_reference_idx',
            'incremental lexical',
        )
        docs_l0_status = fetch_status(connection, 'parity.docs_v3_idx')
        ids_l0_status = fetch_status(connection, 'parity.ids_v3_idx')
        docs_l0_cache_state = fetch_generation_cache_state(
            connection,
            'parity.docs_v3_idx',
        )
        connection.close()
        connection = None
        stop_cluster(pg_ctl, data_dir)
        started = False
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        restarted_docs_l0_status = fetch_status(
            connection,
            'parity.docs_v3_idx',
        )
        restarted_ids_l0_status = fetch_status(
            connection,
            'parity.ids_v3_idx',
        )
        page_native_l0_stream = fetch_page_native_l0_stream(
            connection,
            'parity.docs_v3_idx',
        )
        page_native_numeric_l0_topk = fetch_page_native_l0_query_topk(
            connection,
            'parity.ids_v3_idx',
            [4, 8],
            1,
        )
        page_native_text_l0_topk = fetch_page_native_text_l0_query_topk(
            connection,
            'parity.docs_v3_idx',
            ['lexical', 'l0_unique_1999'],
            2,
        )
        with connection.cursor() as cursor:
            cursor.execute('SELECT count(*) FROM parity.docs_v3')
            dirty_visible_documents = int(cursor.fetchone()[0])
        page_native_zero_score_l0_topk = (
            fetch_page_native_text_l0_query_topk(
                connection,
                'parity.docs_v3_idx',
                ['evoke_zero_score_missing'],
                dirty_visible_documents,
            )
        )
        page_native_zero_score_l0_bounded_topk = (
            fetch_page_native_text_l0_query_topk(
                connection,
                'parity.docs_v3_idx',
                ['evoke_zero_score_missing'],
                3,
            )
        )
        v3_incremental_hits = fetch_search(
            connection,
            'parity.docs_v3_idx',
            'incremental lexical',
        )
        v3_large_hits = fetch_search(
            connection,
            'parity.docs_v3_idx',
            'l0_unique_1999',
        )
        v3_inline_hits = fetch_search(
            connection,
            'parity.docs_v3_idx',
            'small inline tail',
        )
        v3_id_hits = fetch_ids(
            connection,
            'parity.ids_v3_idx',
            [4, 8],
        )
        l0_raw_queries = (
            'l0_unique_19*',
            'l0_unique_1999 l0_unique_19*',
            '"small inline"',
            '+l0_unique_1999 -missing',
            'l0_unique_1999 OR lexical',
        )
        l0_raw_surfaces: dict[str, list[tuple[int, float]]] = {}
        for raw_query in l0_raw_queries:
            reference_raw_hits = fetch_search(
                connection,
                'parity.docs_reference_idx',
                raw_query,
            )
            v3_raw_hits = fetch_search(
                connection,
                'parity.docs_v3_idx',
                raw_query,
            )
            assert_rows_close(
                reference_raw_hits,
                v3_raw_hits,
                label=f'reference-v3 active-L0 raw query:{raw_query}',
            )
            l0_raw_surfaces[raw_query] = v3_raw_hits
        assert_rows_close(
            reference_incremental_hits,
            v3_incremental_hits,
            label='reference-v3 incremental text',
        )
        assert_rows_close(
            fetch_search(
                connection,
                'parity.docs_reference_idx',
                'l0_unique_1999',
            ),
            v3_large_hits,
            label='reference-v3 oversized text',
        )
        assert_rows_close(
            fetch_search(
                connection,
                'parity.docs_reference_idx',
                'small inline tail',
            ),
            v3_inline_hits,
            label='reference-v3 inline text',
        )
        assert_rows_close(
            fetch_ids(connection, 'parity.ids_reference_idx', [4, 8]),
            v3_id_hits,
            label='reference-v3 incremental ids',
        )
        l0_checks = {
            'document_count': v3_count == 9,
            'numeric_count': v3_id_count == 5,
            'text_records': (
                docs_l0_status['generation']['delta']['records'] == 3
            ),
            'cross_page_record': (
                docs_l0_status['generation']['delta']['pages'] > 3
            ),
            'large_record_frontier': (
                docs_large_l0_status['generation']['delta']['records'] == 2
            ),
            'tail_page_reused': (
                docs_l0_status['generation']['delta']['pages']
                == docs_large_l0_status['generation']['delta']['pages']
            ),
            'numeric_records': (
                ids_l0_status['generation']['delta']['records'] == 1
            ),
            'numeric_pages': (
                ids_l0_status['generation']['delta']['pages'] == 1
            ),
            'text_debt_exact': (
                docs_l0_status['details']['pending_writes'] == 3
                and docs_l0_status['details']['pending_deletes'] == 0
                and docs_l0_status['details']['delta_records'] == 3
                and docs_l0_status['details']['delta_bytes']
                == docs_l0_status['generation']['delta']['bytes']
                and docs_l0_status['generation']['delta']['upserts'] == 3
                and docs_l0_status['generation']['delta'][
                    'retirements'
                ]
                == 0
            ),
            'numeric_debt_exact': (
                ids_l0_status['details']['pending_writes'] == 1
                and ids_l0_status['details']['pending_deletes'] == 0
                and ids_l0_status['details']['delta_records'] == 1
                and ids_l0_status['details']['delta_bytes']
                == ids_l0_status['generation']['delta']['bytes']
                and ids_l0_status['generation']['delta']['upserts'] == 1
                and ids_l0_status['generation']['delta'][
                    'retirements'
                ]
                == 0
            ),
            'sealed_docs_are_explicit': (
                docs_l0_status['generation']['docs_scope']
                == 'sealed_generation'
                and docs_l0_status['generation']['sealed_docs']
                == docs_l0_status['generation']['docs']
            ),
            'raw_debt_exact': (
                'pending_writes=3' in docs_l0_cache_state
                and 'pending_deletes=0' in docs_l0_cache_state
                and 'delta_records=3' in docs_l0_cache_state
                and (
                    f"delta_bytes={docs_l0_status['generation']['delta']['bytes']}"
                    in docs_l0_cache_state
                )
            ),
            'query_ready': docs_l0_status['query_ready'] is True,
            'no_blocker': docs_l0_status['blocker'] == 'none',
            'healthy': (
                docs_l0_status['generation']['health_reason'] == 'ok'
            ),
            'restart_frontier': restart_preserves_l0_frontier(
                docs_l0_status,
                restarted_docs_l0_status,
            ),
            'text_query': len(v3_incremental_hits) > 0,
            'large_query': len(v3_large_hits) > 0,
            'inline_query': len(v3_inline_hits) > 0,
            'numeric_query': len(v3_id_hits) > 0,
        }
        gates['v3_mutation_publishes_linked_l0'] = all(
            l0_checks.values()
        )
        gates['page_native_raw_queries_match_reference_with_active_l0'] = True
        gates['page_native_l0_stream_matches_snapshot'] = (
            page_native_l0_stream['matched'] is True
            and int(page_native_l0_stream['record_count']) == 3
            and int(page_native_l0_stream['visited_count']) == 3
            and int(page_native_l0_stream['page_count'])
            == docs_l0_status['generation']['delta']['pages']
            and int(page_native_l0_stream['payload_bytes'])
            == docs_l0_status['generation']['delta']['bytes']
        )
        gates['page_native_l0_topk_matches_snapshot'] = (
            page_native_numeric_l0_topk['matched'] is True
            and page_native_numeric_l0_topk[
                'positive_topk_complete'
            ] is True
            and page_native_numeric_l0_topk['topk_complete'] is True
            and int(page_native_numeric_l0_topk['visited_records']) == 1
            and int(page_native_numeric_l0_topk['visible_records']) == 1
            and int(page_native_numeric_l0_topk['projected_documents']) == 1
            and int(
                page_native_numeric_l0_topk['projected_contributions']
            ) >= 1
            and int(
                page_native_numeric_l0_topk['max_document_block_records']
            ) <= 128
        )
        gates['page_native_text_l0_topk_matches_snapshot'] = (
            page_native_text_l0_topk['matched'] is True
            and page_native_text_l0_topk['positive_topk_complete'] is True
            and page_native_text_l0_topk['topk_complete'] is True
            and page_native_text_l0_topk['numeric_source'] is False
            and int(page_native_text_l0_topk['visited_records']) == 3
            and int(page_native_text_l0_topk['visible_records']) == 3
            and int(page_native_text_l0_topk['projected_documents']) == 3
            and int(page_native_text_l0_topk['projected_contributions']) >= 2
            and int(
                page_native_text_l0_topk['max_document_block_records']
            ) <= 128
        )
        gates['page_native_zero_score_l0_topk_matches_snapshot'] = (
            page_native_zero_score_l0_topk['matched'] is True
            and page_native_zero_score_l0_topk['topk_complete'] is True
            and page_native_zero_score_l0_topk[
                'positive_topk_complete'
            ] is False
            and int(
                page_native_zero_score_l0_topk[
                    'zero_score_documents_added'
                ]
            ) == dirty_visible_documents
            and int(
                page_native_zero_score_l0_topk['projected_documents']
            ) == 3
        )
        gates['page_native_zero_score_l0_topk_is_k_bounded'] = (
            page_native_zero_score_l0_bounded_topk['matched'] is True
            and page_native_zero_score_l0_bounded_topk['topk_complete']
            is True
            and int(
                page_native_zero_score_l0_bounded_topk[
                    'zero_score_documents_added'
                ]
            ) == 3
            and int(
                page_native_zero_score_l0_bounded_topk[
                    'zero_score_cow_records_examined'
                ]
            ) <= 128
            and int(
                page_native_zero_score_l0_bounded_topk[
                    'zero_score_heap_peak'
                ]
            ) <= 256
        )
        gates['reference_mutation_path_remains_available'] = (
            reference_count == 9
            and reference_id_count == 5
            and len(reference_incremental_hits) > 0
        )
        if not all(
            (
                gates['v3_mutation_publishes_linked_l0'],
                gates['page_native_l0_stream_matches_snapshot'],
                gates['page_native_l0_topk_matches_snapshot'],
                gates['page_native_text_l0_topk_matches_snapshot'],
                gates[
                    'page_native_zero_score_l0_topk_matches_snapshot'
                ],
                gates['page_native_zero_score_l0_topk_is_k_bounded'],
                gates[
                    'page_native_raw_queries_match_reference_with_active_l0'
                ],
                gates['reference_mutation_path_remains_available'],
            )
        ):
            raise AssertionError(
                'mutation isolation failed: '
                f'reference={reference_count}, v3={v3_count}, '
                f'reference_ids={reference_id_count}, v3_ids={v3_id_count}, '
                f'l0_checks={l0_checks}, '
                f'docs_status={docs_l0_status}, '
                f'restarted_docs_status={restarted_docs_l0_status}, '
                f'ids_status={ids_l0_status}'
            )

        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE parity.docs_reference SET body = "
                "'replacement lexical generation' WHERE id = 7"
            )
            cursor.execute(
                "UPDATE parity.docs_v3 SET body = "
                "'replacement lexical generation' WHERE id = 7"
            )
            cursor.execute('DELETE FROM parity.docs_reference WHERE id = 8')
            cursor.execute('DELETE FROM parity.docs_v3 WHERE id = 8')
            cursor.execute(
                'UPDATE parity.ids_reference SET tokens = ARRAY[7, 7, 8] '
                'WHERE id = 5'
            )
            cursor.execute(
                'UPDATE parity.ids_v3 SET tokens = ARRAY[7, 7, 8] '
                'WHERE id = 5'
            )
            cursor.execute('DELETE FROM parity.ids_reference WHERE id = 2')
            cursor.execute('DELETE FROM parity.ids_v3 WHERE id = 2')
            cursor.execute('VACUUM (INDEX_CLEANUP ON) parity.docs_reference')
            cursor.execute('VACUUM (INDEX_CLEANUP ON) parity.docs_v3')
            cursor.execute('VACUUM (INDEX_CLEANUP ON) parity.ids_reference')
            cursor.execute('VACUUM (INDEX_CLEANUP ON) parity.ids_v3')

        maintained_reference_status = fetch_status(
            connection,
            'parity.docs_reference_idx',
        )
        gates['reference_maintenance_preserves_page_native_storage'] = (
            maintained_reference_status['generation']['layout'].get('storage')
            == 'convergent_segments'
        )
        if not gates[
            'reference_maintenance_preserves_page_native_storage'
        ]:
            raise AssertionError(
                'ordinary maintenance changed page-native reference storage: '
                f'{maintained_reference_status}'
            )

        crud_text_queries = (
            'replacement lexical',
            'incremental lexical',
            'l0_unique_1999',
            'blue bird river',
        )
        crud_id_queries = ([7, 8], [1, 2], [0, 2])
        crud_text_hits: dict[str, list[tuple[int, float]]] = {}
        crud_id_hits: dict[str, list[tuple[int, float]]] = {}
        for query in crud_text_queries:
            reference_hits = fetch_ordered(
                connection,
                'parity.docs_reference',
                'parity.docs_reference_idx',
                query,
            )
            v3_hits = fetch_ordered(
                connection,
                'parity.docs_v3',
                'parity.docs_v3_idx',
                query,
            )
            assert_rows_close(
                reference_hits,
                v3_hits,
                label=f'reference-v3 CRUD text:{query}',
            )
            crud_text_hits[query] = v3_hits
        for query in crud_id_queries:
            reference_hits = fetch_ids_ordered(
                connection,
                'parity.ids_reference',
                query,
            )
            v3_hits = fetch_ids_ordered(
                connection,
                'parity.ids_v3',
                query,
            )
            assert_rows_close(
                reference_hits,
                v3_hits,
                label=f'reference-v3 CRUD ids:{query}',
            )
            crud_id_hits[str(query)] = v3_hits

        page_native_crud_text = fetch_page_native_text_l0_query_topk(
            connection,
            'parity.docs_v3_idx',
            ['replacement', 'lexical'],
            2,
        )
        page_native_crud_ids = fetch_page_native_l0_query_topk(
            connection,
            'parity.ids_v3_idx',
            [7, 8],
            1,
        )

        docs_crud_status = fetch_status(
            connection,
            'parity.docs_v3_idx',
        )
        ids_crud_status = fetch_status(
            connection,
            'parity.ids_v3_idx',
        )
        connection.close()
        connection = None
        stop_cluster(pg_ctl, data_dir)
        started = False
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        for query in crud_text_queries:
            assert_rows_close(
                crud_text_hits[query],
                fetch_ordered(
                    connection,
                    'parity.docs_v3',
                    'parity.docs_v3_idx',
                    query,
                ),
                label=f'v3 CRUD restart text:{query}',
            )
        for query in crud_id_queries:
            assert_rows_close(
                crud_id_hits[str(query)],
                fetch_ids_ordered(
                    connection,
                    'parity.ids_v3',
                    query,
                ),
                label=f'v3 CRUD restart ids:{query}',
            )
        restarted_page_native_crud_text = (
            fetch_page_native_text_l0_query_topk(
                connection,
                'parity.docs_v3_idx',
                ['replacement', 'lexical'],
                2,
            )
        )
        restarted_page_native_crud_ids = fetch_page_native_l0_query_topk(
            connection,
            'parity.ids_v3_idx',
            [7, 8],
            1,
        )
        restarted_docs_crud_status = fetch_status(
            connection,
            'parity.docs_v3_idx',
        )
        gates['v3_crud_retirement_matches_reference'] = (
            docs_crud_status['query_ready'] is True
            and ids_crud_status['query_ready'] is True
            and l0_accepts_records_after_optional_pending_seal(
                restarted_docs_l0_status,
                docs_crud_status,
                new_upserts=1,
                new_retirements=2,
            )
            and l0_accepts_records_after_optional_pending_seal(
                restarted_ids_l0_status,
                ids_crud_status,
                new_upserts=1,
                new_retirements=2,
            )
            and restarted_docs_crud_status['generation']['delta']
            == docs_crud_status['generation']['delta']
        )
        gates['page_native_crud_retirement_matches_snapshot'] = (
            page_native_crud_text['matched'] is True
            and page_native_crud_ids['matched'] is True
            and restarted_page_native_crud_text['matched'] is True
            and restarted_page_native_crud_ids['matched'] is True
        )
        if not (
            gates['v3_crud_retirement_matches_reference']
            and gates['page_native_crud_retirement_matches_snapshot']
        ):
            raise AssertionError(
                'v3 retirement closure failed: '
                f'docs={docs_crud_status}, ids={ids_crud_status}, '
                f'restarted={restarted_docs_crud_status}, '
                f'page_text={page_native_crud_text}, '
                f'page_ids={page_native_crud_ids}'
            )

        second_vacuum_records = docs_crud_status['generation']['delta'][
            'records'
        ]
        with connection.cursor() as cursor:
            cursor.execute('VACUUM (INDEX_CLEANUP ON) parity.docs_v3')
        second_vacuum_status = fetch_status(
            connection,
            'parity.docs_v3_idx',
        )
        if (
            second_vacuum_status['generation']['delta']['records']
            != second_vacuum_records
        ):
            raise AssertionError(
                'repeat VACUUM appended duplicate retirement records'
            )

        transaction_connection = connect(socket_dir, port)
        transaction_connection.autocommit = False
        observer_connection = connect(socket_dir, port)
        transaction_page_native: dict[str, dict[str, Any]] = {}
        try:
            with transaction_connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO parity.docs_v3 VALUES '
                    "(20, 'evoketxcurrentmarker')"
                )
            current_hits = fetch_search(
                transaction_connection,
                'parity.docs_v3_idx',
                'evoketxcurrentmarker',
            )
            unresolved_hits = fetch_search(
                observer_connection,
                'parity.docs_v3_idx',
                'evoketxcurrentmarker',
            )
            transaction_page_native['current'] = (
                fetch_page_native_text_l0_query_topk(
                    transaction_connection,
                    'parity.docs_v3_idx',
                    ['lexical', 'evoketxcurrentmarker'],
                    1,
                )
            )
            transaction_page_native['unresolved_observer'] = (
                fetch_page_native_text_l0_query_topk(
                    observer_connection,
                    'parity.docs_v3_idx',
                    ['lexical', 'evoketxcurrentmarker'],
                    1,
                )
            )
            current_ids = positive_result_ids(current_hits)
            if len(current_ids) != 1:
                raise AssertionError(
                    'current transaction L0 row was not searchable: '
                    f'{current_hits}'
                )
            if positive_result_ids(unresolved_hits):
                raise AssertionError(
                    'unresolved transaction L0 row leaked to observer: '
                    f'{unresolved_hits}'
                )
            transaction_connection.commit()
            committed_hits = fetch_search(
                observer_connection,
                'parity.docs_v3_idx',
                'evoketxcurrentmarker',
            )
            transaction_page_native['committed'] = (
                fetch_page_native_text_l0_query_topk(
                    observer_connection,
                    'parity.docs_v3_idx',
                    ['lexical', 'evoketxcurrentmarker'],
                    1,
                )
            )
            if positive_result_ids(committed_hits) != current_ids:
                raise AssertionError(
                    'committed transaction did not become searchable'
                )

            with transaction_connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO parity.docs_v3 VALUES '
                    "(21, 'evoketxabortedmarker')"
                )
            aborted_current_ids = positive_result_ids(
                fetch_search(
                    transaction_connection,
                    'parity.docs_v3_idx',
                    'evoketxabortedmarker',
                )
            )
            transaction_page_native['aborted_current'] = (
                fetch_page_native_text_l0_query_topk(
                    transaction_connection,
                    'parity.docs_v3_idx',
                    ['lexical', 'evoketxabortedmarker'],
                    1,
                )
            )
            if len(aborted_current_ids) != 1:
                raise AssertionError(
                    'aborted test row was not visible before rollback'
                )
            transaction_connection.rollback()
            transaction_page_native['aborted_observer'] = (
                fetch_page_native_text_l0_query_topk(
                    observer_connection,
                    'parity.docs_v3_idx',
                    ['lexical', 'evoketxabortedmarker'],
                    1,
                )
            )
            if positive_result_ids(
                fetch_search(
                    observer_connection,
                    'parity.docs_v3_idx',
                    'evoketxabortedmarker',
                )
            ):
                raise AssertionError('aborted transaction remained searchable')

            with transaction_connection.cursor() as cursor:
                cursor.execute('SAVEPOINT evoke_l0_savepoint')
                cursor.execute(
                    'INSERT INTO parity.docs_v3 VALUES '
                    "(22, 'evoketxsavepointmarker')"
                )
            savepoint_current_ids = positive_result_ids(
                fetch_search(
                    transaction_connection,
                    'parity.docs_v3_idx',
                    'evoketxsavepointmarker',
                )
            )
            transaction_page_native['savepoint_current'] = (
                fetch_page_native_text_l0_query_topk(
                    transaction_connection,
                    'parity.docs_v3_idx',
                    ['lexical', 'evoketxsavepointmarker'],
                    1,
                )
            )
            if len(savepoint_current_ids) != 1:
                raise AssertionError(
                    'savepoint row was not visible before rollback'
                )
            with transaction_connection.cursor() as cursor:
                cursor.execute('ROLLBACK TO SAVEPOINT evoke_l0_savepoint')
            transaction_page_native['savepoint_rolled_back'] = (
                fetch_page_native_text_l0_query_topk(
                    transaction_connection,
                    'parity.docs_v3_idx',
                    ['lexical', 'evoketxsavepointmarker'],
                    1,
                )
            )
            if positive_result_ids(
                fetch_search(
                    transaction_connection,
                    'parity.docs_v3_idx',
                    'evoketxsavepointmarker',
                )
            ):
                raise AssertionError(
                    'rolled-back savepoint row remained searchable'
                )
            transaction_connection.commit()

            with transaction_connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO parity.docs_v3 VALUES '
                    "(23, 'evoketxpreparedmarker')"
                )
                cursor.execute(
                    "PREPARE TRANSACTION 'evoke_v3_l0_prepared'"
                )
            if positive_result_ids(
                fetch_search(
                    observer_connection,
                    'parity.docs_v3_idx',
                    'evoketxpreparedmarker',
                )
            ):
                raise AssertionError(
                    'prepared transaction became visible before commit'
                )
            transaction_page_native['prepared_unresolved'] = (
                fetch_page_native_text_l0_query_topk(
                    observer_connection,
                    'parity.docs_v3_idx',
                    ['lexical', 'evoketxpreparedmarker'],
                    1,
                )
            )
            with observer_connection.cursor() as cursor:
                cursor.execute(
                    "COMMIT PREPARED 'evoke_v3_l0_prepared'"
                )
            if len(
                positive_result_ids(
                    fetch_search(
                        observer_connection,
                        'parity.docs_v3_idx',
                        'evoketxpreparedmarker',
                    )
                )
            ) != 1:
                raise AssertionError(
                    'committed prepared transaction was not searchable'
                )
            transaction_page_native['prepared_committed'] = (
                fetch_page_native_text_l0_query_topk(
                    observer_connection,
                    'parity.docs_v3_idx',
                    ['lexical', 'evoketxpreparedmarker'],
                    1,
                )
            )

            visibility_pairs = (
                ('current', 'unresolved_observer'),
                ('aborted_current', 'aborted_observer'),
                ('savepoint_current', 'savepoint_rolled_back'),
                ('prepared_committed', 'prepared_unresolved'),
            )
            transaction_projection_checks = {
                f'{visible_name}_includes_one_more_record': (
                    int(
                        transaction_page_native[visible_name][
                            'visible_records'
                        ]
                    )
                    == int(
                        transaction_page_native[hidden_name][
                            'visible_records'
                        ]
                    ) + 1
                )
                for visible_name, hidden_name in visibility_pairs
            }
            transaction_projection_checks[
                'commit_preserves_current_visibility'
            ] = (
                int(transaction_page_native['committed']['visible_records'])
                == int(transaction_page_native['current']['visible_records'])
            )
            transaction_projection_checks['all_oracles_match'] = all(
                probe['matched'] is True
                for probe in transaction_page_native.values()
            )
            gates['page_native_transaction_visibility_is_exact'] = all(
                transaction_projection_checks.values()
            )
            if not gates['page_native_transaction_visibility_is_exact']:
                raise AssertionError(
                    'page-native transaction visibility differs: '
                    f'{transaction_projection_checks}, '
                    f'{transaction_page_native}'
                )

            old_snapshot_connection = connect(socket_dir, port)
            old_snapshot_connection.autocommit = False
            try:
                with old_snapshot_connection.cursor() as cursor:
                    cursor.execute(
                        'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ'
                    )
                    cursor.execute('SELECT count(*) FROM parity.docs_v3')
                    cursor.fetchone()
                with observer_connection.cursor() as cursor:
                    cursor.execute(
                        'INSERT INTO parity.docs_v3 VALUES '
                        "(24, 'evokeoldsnapshotmarker')"
                    )
                old_snapshot_hits = fetch_search(
                    old_snapshot_connection,
                    'parity.docs_v3_idx',
                    'evokeoldsnapshotmarker',
                )
                observer_snapshot_hits = fetch_search(
                    observer_connection,
                    'parity.docs_v3_idx',
                    'evokeoldsnapshotmarker',
                )
                snapshot_rotation, snapshot_rotation_attempts = (
                    try_maintain_after_lock_contention(
                        observer_connection,
                        'parity.docs_v3_idx',
                    )
                )
                snapshot_deferred, snapshot_deferred_attempts = (
                    try_maintain_after_lock_contention(
                        observer_connection,
                        'parity.docs_v3_idx',
                    )
                )
                deferred_snapshot_status = fetch_status(
                    observer_connection,
                    'parity.docs_v3_idx',
                )
                old_snapshot_checks = {
                    'old_reader_hides_new_tid': (
                        len(positive_result_ids(old_snapshot_hits)) == 0
                    ),
                    'new_reader_sees_committed_tid': (
                        len(
                            positive_result_ids(observer_snapshot_hits)
                        )
                        == 1
                    ),
                    'committed_frontier_is_pending': (
                        deferred_snapshot_status['generation']['delta'][
                            'pending'
                        ]['records'] > 0
                        and (
                            (
                                snapshot_rotation.get('maintained')
                                == 'true'
                                and snapshot_rotation.get('mode')
                                == 'segment_rotation'
                            )
                            or (
                                snapshot_rotation.get('maintained')
                                == 'false'
                                and snapshot_rotation.get('reason')
                                == 'xid_horizon'
                            )
                        )
                    ),
                    'seal_waits_for_safe_xid_horizon': (
                        snapshot_deferred.get('maintained') == 'false'
                        and snapshot_deferred.get('reason')
                        == 'xid_horizon'
                        and deferred_snapshot_status['generation']['delta'][
                            'pending'
                        ]['records'] > 0
                    ),
                }
                if not all(old_snapshot_checks.values()):
                    raise AssertionError(
                        'old-snapshot seal fence failed: '
                        f'{old_snapshot_checks}, '
                        f'rotation={snapshot_rotation}, '
                        f'rotation_attempts={snapshot_rotation_attempts}, '
                        f'deferred={snapshot_deferred}, '
                        f'deferred_attempts={snapshot_deferred_attempts}, '
                        f'status={deferred_snapshot_status}'
                    )
                old_snapshot_connection.commit()
                snapshot_seal, snapshot_seal_attempts = (
                    try_maintain_after_lock_contention(
                        observer_connection,
                        'parity.docs_v3_idx',
                    )
                )
                converged_snapshot_status = fetch_status(
                    observer_connection,
                    'parity.docs_v3_idx',
                )
                old_snapshot_checks[
                    'seal_resumes_after_old_reader'
                ] = (
                    converged_snapshot_status['generation']['delta'][
                        'pending'
                    ]['records'] == 0
                    and converged_snapshot_status['generation']['primary'][
                        'segment_count'
                    ] > 0
                )
                if not old_snapshot_checks[
                    'seal_resumes_after_old_reader'
                ]:
                    raise AssertionError(
                        'safe-horizon seal did not resume: '
                        f'{snapshot_seal}, '
                        f'attempts={snapshot_seal_attempts}, '
                        f'{converged_snapshot_status}'
                    )
                with observer_connection.cursor() as cursor:
                    cursor.execute(
                        'INSERT INTO parity.docs_v3 VALUES '
                        "(25, 'evokerotationseed')"
                    )
            finally:
                old_snapshot_connection.close()
        finally:
            transaction_connection.close()
            observer_connection.close()

        pre_rotation_maintenance: list[dict[str, str]] = []
        for _ in range(8):
            transaction_status = fetch_status(
                connection,
                'parity.docs_v3_idx',
            )
            if (
                transaction_status['generation']['delta']['pending'][
                    'records'
                ]
                == 0
            ):
                break
            pre_rotation_maintenance.append(
                maintenance_result_fields(
                    try_maintain(connection, 'parity.docs_v3_idx')
                )
            )
        else:
            raise AssertionError(
                'pending L0 did not seal before rotation test: '
                f'{transaction_status}, '
                f'maintenance={pre_rotation_maintenance}'
            )
        pre_rotation_records = transaction_status['generation']['delta'][
            'active'
        ]['records']
        with connection.cursor() as cursor:
            cursor.execute(
                "SET evoke.test_convergent_l0_rotation_records = '1'"
            )
            cursor.execute(
                'INSERT INTO parity.docs_v3 VALUES '
                "(30, 'evokerotationactiveone')"
            )
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
            cursor.execute(
                'INSERT INTO parity.docs_v3 VALUES '
                "(31, 'evokerotationactivetwo')"
            )
        rotation_status = fetch_status(
            connection,
            'parity.docs_v3_idx',
        )
        rotation_delta = rotation_status['generation']['delta']
        pending_preserves_old_active = (
            rotation_delta['pending']['records']
            == pre_rotation_records
            and rotation_delta['pending']['segment_id']
            == transaction_status['generation']['delta']['active'][
                'segment_id'
            ]
        )
        rotated_frontier_was_sealed = (
            rotation_delta['pending']['records'] == 0
            and rotation_delta['pending']['segment_id'] == 0
            and rotation_status['generation']['primary'][
                'published_block_high_watermark'
            ]
            > transaction_status['generation']['primary'][
                'published_block_high_watermark'
            ]
        )
        rotation_checks = {
            'pending_preserves_old_active': (
                pending_preserves_old_active
                or rotated_frontier_was_sealed
            ),
            'new_active_accepts_writes': (
                rotation_delta['active']['records'] == 2
            ),
            'segment_identity_advanced': (
                rotation_delta['active']['segment_id']
                != transaction_status['generation']['delta']['active'][
                    'segment_id'
                ]
                and (
                    rotated_frontier_was_sealed
                    or rotation_delta['pending']['segment_id']
                    != rotation_delta['active']['segment_id']
                )
            ),
            'pending_query_visible': (
                len(
                    positive_result_ids(
                        fetch_search(
                            connection,
                            'parity.docs_v3_idx',
                            'evoketxpreparedmarker',
                        )
                    )
                )
                == 1
            ),
            'new_active_queries_visible': (
                len(
                    positive_result_ids(
                        fetch_search(
                            connection,
                            'parity.docs_v3_idx',
                            'evokerotationactiveone',
                        )
                    )
                )
                == 1
                and len(
                    positive_result_ids(
                        fetch_search(
                            connection,
                            'parity.docs_v3_idx',
                            'evokerotationactivetwo',
                        )
                    )
                )
                == 1
            ),
        }
        if not all(rotation_checks.values()):
            raise AssertionError(
                'active-to-pending L0 rotation failed: '
                f'{rotation_checks}, status={rotation_status}, '
                f'pre_maintenance={pre_rotation_maintenance}'
            )

        connection.close()
        connection = None
        stop_cluster(pg_ctl, data_dir)
        started = False
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        restarted_rotation_status = fetch_status(
            connection,
            'parity.docs_v3_idx',
        )
        restarted_delta = restarted_rotation_status['generation']['delta']
        pending_remains = restarted_delta == rotation_delta
        active_rotated_on_restart = restart_preserves_l0_frontier(
            rotation_status,
            restarted_rotation_status,
        )
        pending_was_sealed = (
            restarted_delta['pending']['segment_id'] == 0
            and restarted_delta['pending']['records'] == 0
            and restarted_delta['pending']['bytes'] == 0
            and restarted_delta['pending']['pages'] == 0
            and restarted_delta['active'] == rotation_delta['active']
            and (
                restarted_rotation_status['generation']['primary'][
                    'segment_count'
                ]
                == rotation_status['generation']['primary']['segment_count']
                + 1
            )
            and (
                restarted_rotation_status['generation']['primary'][
                    'published_block_high_watermark'
                ]
                >= rotation_status['generation']['primary'][
                    'published_block_high_watermark'
                ]
            )
        )
        pending_sealed_and_active_rotated = (
            restarted_delta['pending'] == rotation_delta['active']
            and restarted_delta['active']['records'] == 0
            and restarted_delta['active']['bytes'] == 0
            and restarted_delta['active']['pages'] == 0
            and (
                restarted_rotation_status['generation']['primary'][
                    'segment_count'
                ]
                == rotation_status['generation']['primary']['segment_count']
                + 1
            )
            and (
                restarted_rotation_status['generation']['primary'][
                    'published_block_high_watermark'
                ]
                >= rotation_status['generation']['primary'][
                    'published_block_high_watermark'
                ]
            )
        )
        restarted_rotation_checks = {
            'frontiers_or_sealed_manifest': (
                pending_remains
                or active_rotated_on_restart
                or pending_was_sealed
                or pending_sealed_and_active_rotated
            ),
            'pending_query': (
                len(
                    positive_result_ids(
                        fetch_search(
                            connection,
                            'parity.docs_v3_idx',
                            'evoketxpreparedmarker',
                        )
                    )
                )
                == 1
            ),
            'active_query': (
                len(
                    positive_result_ids(
                        fetch_search(
                            connection,
                            'parity.docs_v3_idx',
                            'evokerotationactivetwo',
                        )
                    )
                )
                == 1
            ),
        }
        if not all(restarted_rotation_checks.values()):
            raise AssertionError(
                'rotated L0 restart closure failed: '
                f'{restarted_rotation_checks}, '
                f'before={rotation_status}, '
                f'status={restarted_rotation_status}'
            )

        compaction_maintenance_results: list[str] = []
        compaction_status = restarted_rotation_status
        for _ in range(8):
            if (
                compaction_status['generation']['delta']['pending'][
                    'records'
                ]
                == 0
            ):
                break
            compaction_maintenance_results.append(
                try_maintain(connection, 'parity.docs_v3_idx')
            )
            compaction_status = fetch_status(
                connection,
                'parity.docs_v3_idx',
            )
        else:
            raise AssertionError(
                'pending L0 did not seal before compaction test: '
                f'{compaction_maintenance_results}'
            )
        post_seal_cow_probe = write_term_cow_probe(
            connection,
            'parity.docs_v3_idx',
        )
        gates['new_terms_append_catalog_without_contract_rewrite'] = (
            term_cow_query_contract_identity(post_seal_cow_probe)
            == term_cow_query_contract_identity(cow_probe)
            and post_seal_cow_probe['vocab_size']
            > cow_probe['vocab_size']
            and post_seal_cow_probe['lexical_catalog_count']
            > cow_probe['lexical_catalog_count']
            and post_seal_cow_probe[
                'latest_lexical_catalog_term_count'
            ]
            > 0
            and post_seal_cow_probe[
                'latest_lexical_catalog_term_count'
            ]
            <= (
                post_seal_cow_probe['vocab_size']
                - cow_probe['vocab_size']
            )
        )
        if not gates['new_terms_append_catalog_without_contract_rewrite']:
            raise AssertionError(
                'new-term seal rewrote stable metadata or catalog ranges: '
                f'before={cow_probe}, after={post_seal_cow_probe}'
            )
        evidence['post_seal_cow_probe'] = post_seal_cow_probe

        compact_marker = 0
        while compact_marker < 4:
            marker = f'evokecompactmarker{compact_marker}'
            previous_segment_count = compaction_status['generation'][
                'primary'
            ]['segment_count']
            previous_active_records = compaction_status['generation'][
                'delta'
            ]['active']['records']
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_convergent_l0_rotation_records = '1'"
                )
                cursor.execute(
                    'INSERT INTO parity.docs_v3 VALUES (%s, %s)',
                    (40 + compact_marker, marker),
                )
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )
            compaction_status = fetch_status(
                connection,
                'parity.docs_v3_idx',
            )
            pending_records = compaction_status['generation']['delta'][
                'pending'
            ]['records']
            current_segment_count = compaction_status['generation'][
                'primary'
            ]['segment_count']
            current_active_records = compaction_status['generation'][
                'delta'
            ]['active']['records']
            if (
                pending_records == 0
                and current_segment_count == previous_segment_count
                and current_active_records <= previous_active_records
            ):
                raise AssertionError(
                    'compaction setup made no observable progress: '
                    f'{compaction_status}'
                )
            if pending_records > 0:
                for _ in range(8):
                    compaction_maintenance_results.append(
                        try_maintain(connection, 'parity.docs_v3_idx')
                    )
                    compaction_status = fetch_status(
                        connection,
                        'parity.docs_v3_idx',
                    )
                    if (
                        compaction_status['generation']['delta']['pending'][
                            'records'
                        ]
                        == 0
                    ):
                        break
                else:
                    raise AssertionError(
                        'pending L0 did not seal during compaction setup: '
                        f'{compaction_maintenance_results}'
                    )
            compact_marker += 1

        if (
            compaction_status['generation']['primary']['segment_count']
            < 4
        ):
            raise AssertionError(
                'compaction setup did not reach four sealed segments: '
                f'{compaction_status}, '
                f'maintenance={compaction_maintenance_results}'
            )
        pre_compaction_segment_count = compaction_status['generation'][
            'primary'
        ]['segment_count']
        compaction_queries = (
            'blue bird river',
            'evokerotationactiveone',
            'evokecompactmarker0',
            f'evokecompactmarker{compact_marker - 1}',
        )
        for _ in range(16):
            compaction_status = fetch_status(
                connection,
                'parity.docs_v3_idx',
            )
            pre_compaction_segment_count = compaction_status['generation'][
                'primary'
            ]['segment_count']
            pre_compaction_hits = {
                query: fetch_search(
                    connection,
                    'parity.docs_v3_idx',
                    query,
                )
                for query in compaction_queries
            }
            pre_compaction_delta = compaction_status['generation']['delta']
            pre_compaction_docs = compaction_status['generation']['docs']
            pre_compaction_cow_probe = write_term_cow_probe(
                connection,
                'parity.docs_v3_idx',
            )
            compaction_result = try_maintain(
                connection,
                'parity.docs_v3_idx',
            )
            compacted_status = fetch_status(
                connection,
                'parity.docs_v3_idx',
            )
            post_compaction_hits = {
                query: fetch_search(
                    connection,
                    'parity.docs_v3_idx',
                    query,
                )
                for query in compaction_queries
            }
            post_compaction_cow_probe = write_term_cow_probe(
                connection,
                'parity.docs_v3_idx',
            )
            compaction_fields = maintenance_result_fields(
                compaction_result
            )
            if (
                compaction_fields.get('mode') == 'segment_compaction'
                and compaction_fields.get('maintained') == 'true'
            ):
                break
            compaction_maintenance_results.append(compaction_result)
        else:
            raise AssertionError(
                'maintenance did not reach selective compaction: '
                f'{compaction_maintenance_results}, '
                f'status={compacted_status}'
            )
        compaction_checks = {
            'bounded_tier_selected': (
                'maintained=true' in compaction_result
                and 'mode=segment_compaction' in compaction_result
                and 'reason=tier_pressure' in compaction_result
                and 'segment_count=4' in compaction_result
            ),
            'four_segments_replaced_by_one': (
                compacted_status['generation']['primary'][
                    'segment_count'
                ]
                == pre_compaction_segment_count - 3
            ),
            'frontiers_preserved': (
                compacted_status['generation']['delta']
                == pre_compaction_delta
            ),
            'logical_statistics_preserved': (
                compacted_status['generation']['docs']
                == pre_compaction_docs
            ),
            'query_results_exact': (
                post_compaction_hits == pre_compaction_hits
            ),
            'publication_advanced': (
                compacted_status['generation']['generation']
                > compaction_status['generation']['generation']
                and compacted_status['generation']['primary'][
                    'manifest_start_block'
                ]
                != compaction_status['generation']['primary'][
                    'manifest_start_block'
                ]
            ),
            'stable_contract_and_catalogs_preserved': (
                term_cow_query_contract_identity(
                    post_compaction_cow_probe
                )
                == term_cow_query_contract_identity(
                    pre_compaction_cow_probe
                )
                and term_cow_catalog_summary(
                    post_compaction_cow_probe
                )
                == term_cow_catalog_summary(
                    pre_compaction_cow_probe
                )
            ),
        }
        if not all(compaction_checks.values()):
            raise AssertionError(
                'selective segment compaction failed: '
                f'{compaction_checks}, result={compaction_result}, '
                f'before={compaction_status}, after={compacted_status}'
            )
        gates['v3_compaction_preserves_lexical_catalogs'] = True
        evidence['pre_compaction_cow_probe'] = pre_compaction_cow_probe
        evidence['post_compaction_cow_probe'] = post_compaction_cow_probe

        connection.close()
        connection = None
        stop_cluster(pg_ctl, data_dir)
        started = False
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        restarted_compaction_status = fetch_status(
            connection,
            'parity.docs_v3_idx',
        )
        restarted_compaction_hits = {
            query: fetch_search(
                connection,
                'parity.docs_v3_idx',
                query,
            )
            for query in compaction_queries
        }
        restarted_segment_count = restarted_compaction_status['generation'][
            'primary'
        ]['segment_count']
        if (
            restarted_segment_count > pre_compaction_segment_count - 3
            or restarted_compaction_status['generation']['delta']
            != pre_compaction_delta
            or restarted_compaction_status['generation']['docs']
            != pre_compaction_docs
            or restarted_compaction_hits != pre_compaction_hits
        ):
            raise AssertionError(
                'compacted logical state changed after restart: '
                f'{restarted_compaction_status}'
            )

        with connection.cursor() as cursor:
            cursor.execute(
                'CREATE TABLE parity.sustained_docs ('
                'id int PRIMARY KEY, body text NOT NULL)'
            )
            cursor.execute(
                "INSERT INTO parity.sustained_docs VALUES "
                "(1, 'evokesustainedcommon evokesustainedbase')"
            )
            create_page_native_reference_index(
                cursor,
                'CREATE INDEX sustained_reference_idx '
                'ON parity.sustained_docs USING evoke (body) '
                'WITH (sae=false, consistency=realtime)',
            )
            cursor.execute(
                'CREATE INDEX sustained_v3_idx '
                'ON parity.sustained_docs USING evoke (body) '
                'WITH (sae=false, consistency=realtime)'
            )

        sustained_maintenance: list[str] = []
        sustained_compactions: list[dict[str, str]] = []
        sustained_max_segments = 1
        for marker_index in range(32):
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET evoke.test_convergent_l0_rotation_records = '1'"
                )
                cursor.execute(
                    'INSERT INTO parity.sustained_docs VALUES (%s, %s)',
                    (
                        100 + marker_index,
                        (
                            'evokesustainedcommon '
                            f'evokesustainedmarker{marker_index:02d}'
                        ),
                    ),
                )
                cursor.execute(
                    'RESET evoke.test_convergent_l0_rotation_records'
                )

            for maintenance_pass in range(12):
                maintenance_result = try_maintain(
                    connection,
                    'parity.sustained_v3_idx',
                )
                sustained_maintenance.append(maintenance_result)
                fields = maintenance_result_fields(maintenance_result)
                sustained_status = fetch_status(
                    connection,
                    'parity.sustained_v3_idx',
                )
                segment_count = sustained_status['generation']['primary'][
                    'segment_count'
                ]
                sustained_max_segments = max(
                    sustained_max_segments,
                    segment_count,
                )
                if fields.get('mode') == 'segment_compaction':
                    if fields.get('maintained') != 'true':
                        raise AssertionError(
                            'sustained compaction did not make progress: '
                            f'{maintenance_result}'
                        )
                    if (
                        int(fields['segment_count']) > 8
                        or int(fields['input_bytes']) > 64 * 1024 * 1024
                    ):
                        raise AssertionError(
                            'sustained compaction exceeded its budget: '
                            f'{maintenance_result}'
                        )
                    sustained_compactions.append(fields)
                    continue
                if (
                    sustained_status['generation']['delta']['pending'][
                        'records'
                    ]
                    > 0
                ):
                    continue
                if (
                    fields.get('mode') == 'segment_seal'
                    and fields.get('maintained') == 'true'
                ):
                    continue
                break
            else:
                raise AssertionError(
                    'sustained maintenance did not drain after marker '
                    f'{marker_index}: {sustained_maintenance[-12:]}'
                )

        fixed_churn_background_quiescence = (
            quiesce_background_maintenance(connection)
        )
        sustained_status = fetch_status(
            connection,
            'parity.sustained_v3_idx',
        )
        sustained_reference_hits = fetch_search(
            connection,
            'parity.sustained_reference_idx',
            'evokesustainedcommon',
        )
        sustained_v3_hits = fetch_search(
            connection,
            'parity.sustained_v3_idx',
            'evokesustainedcommon',
        )
        assert_rows_close(
            sustained_reference_hits,
            sustained_v3_hits,
            label='sustained reference-v3 parity',
        )
        sustained_checks = {
            'all_rows_searchable': (
                len(positive_result_ids(sustained_v3_hits)) == 33
            ),
            'active_debt_bounded': (
                sustained_status['details']['pending_writes'] <= 1
                and sustained_status['details']['pending_deletes'] == 0
                and sustained_status['details']['delta_records']
                == sustained_status['details']['pending_writes']
                and sustained_status['details']['delta_bytes']
                == sustained_status['generation']['delta']['bytes']
                and sustained_status['generation']['delta']['upserts']
                == sustained_status['details']['pending_writes']
                and sustained_status['generation']['delta'][
                    'retirements'
                ]
                == 0
                and sustained_status['generation']['delta']['active'][
                    'records'
                ]
                <= 1
            ),
            'pending_drained': (
                sustained_status['generation']['delta']['pending'][
                    'records'
                ]
                == 0
            ),
            'geometric_segment_bound': (
                sustained_status['generation']['primary'][
                    'segment_count'
                ]
                <= 8
                and sustained_max_segments <= 8
            ),
            'compaction_exercised': len(sustained_compactions) >= 8,
        }
        if not all(sustained_checks.values()):
            raise AssertionError(
                'sustained segment convergence failed: '
                f'{sustained_checks}, status={sustained_status}, '
                f'compactions={sustained_compactions}'
            )
        extent_evidence = exercise_extent_pressure(connection)
        connection.close()
        connection = None
        stop_cluster(pg_ctl, data_dir)
        started = False
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        restarted_sustained_status = fetch_status(
            connection,
            'parity.sustained_v3_idx',
        )
        restarted_sustained_hits = fetch_search(
            connection,
            'parity.sustained_v3_idx',
            'evokesustainedcommon',
        )
        assert_rows_close(
            sustained_reference_hits,
            restarted_sustained_hits,
            label='sustained v3 restart',
        )
        restarted_extent_status = fetch_status(
            connection,
            'parity.extent_v3_idx',
        )
        restarted_extent_hits = fetch_ids(
            connection,
            'parity.extent_v3_idx',
            [1],
        )
        assert_rows_close(
            extent_evidence['reference_hits'],
            restarted_extent_hits,
            label='extent-pressure v3 restart',
        )
        if (
            restarted_sustained_status['generation']['primary'][
                'segment_count'
            ]
            != sustained_status['generation']['primary']['segment_count']
            or restarted_sustained_status['generation']['delta']
            != sustained_status['generation']['delta']
        ):
            raise AssertionError(
                'sustained generation changed after restart: '
                f'{restarted_sustained_status}'
            )
        if (
            restarted_extent_status['generation']['primary'][
                'segment_count'
            ]
            != extent_evidence['status']['generation']['primary'][
                'segment_count'
            ]
            or restarted_extent_status['generation']['delta']
            != extent_evidence['status']['generation']['delta']
        ):
            raise AssertionError(
                'extent-pressure generation changed after restart: '
                f'{restarted_extent_status}'
            )
        fixed_churn = exercise_fixed_live_set_churn(connection)
        gates['fixed_live_set_churn_remains_query_exact'] = True
        gates['fixed_live_set_document_slots_plateau'] = True
        gates['fixed_live_set_reusable_cursor_resets'] = True
        evidence['fixed_churn_background_quiescence'] = (
            fixed_churn_background_quiescence
        )
        evidence['fixed_live_set_churn'] = fixed_churn

        with connection.cursor() as cursor:
            cursor.execute('REINDEX INDEX parity.docs_reference_idx')
        reindexed_reference_status = fetch_status(
            connection,
            'parity.docs_reference_idx',
        )
        for query in crud_text_queries:
            assert_rows_close(
                crud_text_hits[query],
                fetch_ordered(
                    connection,
                    'parity.docs_reference',
                    'parity.docs_reference_idx',
                    query,
                ),
                label=f'explicit page-native reference REINDEX:{query}',
            )
        gates['explicit_reindex_preserves_page_native_reference'] = (
            reindexed_reference_status['generation']['layout'].get('storage')
            == 'convergent_segments'
        )
        if not gates['explicit_reindex_preserves_page_native_reference']:
            raise AssertionError(
                'explicit REINDEX changed page-native reference storage: '
                f'{reindexed_reference_status}'
            )
        evidence['page_native_reference_storage_boundary'] = {
            'after_maintenance': maintained_reference_status['generation']['layout'],
            'after_explicit_reindex': (
                reindexed_reference_status['generation']['layout']
            ),
        }

        gates['v3_l0_transaction_boundaries_are_exact'] = True
        gates['v3_seal_respects_old_snapshot_horizon'] = all(
            old_snapshot_checks.values()
        )
        gates['v3_repeat_vacuum_is_idempotent'] = True
        gates['v3_l0_rotation_keeps_reads_and_writes_live'] = True
        gates['v3_selective_compaction_is_exact'] = True
        gates['v3_sustained_writes_converge_geometrically'] = True
        gates['v3_extent_pressure_converges_without_tier_match'] = True
        evidence['v3_incremental_hits'] = v3_incremental_hits
        evidence['v3_large_hits'] = v3_large_hits
        evidence['v3_inline_hits'] = v3_inline_hits
        evidence['v3_active_l0_raw_query_surfaces'] = l0_raw_surfaces
        evidence['v3_id_hits'] = v3_id_hits
        evidence['crud_text_hits'] = crud_text_hits
        evidence['crud_id_hits'] = crud_id_hits
        evidence['page_native_crud_text'] = page_native_crud_text
        evidence['page_native_crud_ids'] = page_native_crud_ids
        evidence['restarted_page_native_crud_text'] = (
            restarted_page_native_crud_text
        )
        evidence['restarted_page_native_crud_ids'] = (
            restarted_page_native_crud_ids
        )
        evidence['docs_crud_status'] = docs_crud_status
        evidence['ids_crud_status'] = ids_crud_status
        evidence['restarted_docs_crud_status'] = (
            restarted_docs_crud_status
        )
        evidence['second_vacuum_status'] = second_vacuum_status
        evidence['transaction_status'] = transaction_status
        evidence['transaction_page_native'] = transaction_page_native
        evidence['transaction_projection_checks'] = (
            transaction_projection_checks
        )
        evidence['old_snapshot_checks'] = old_snapshot_checks
        evidence['old_snapshot_deferred_status'] = (
            deferred_snapshot_status
        )
        evidence['old_snapshot_converged_status'] = (
            converged_snapshot_status
        )
        evidence['rotation_status'] = rotation_status
        evidence['restarted_rotation_status'] = (
            restarted_rotation_status
        )
        evidence['pending_was_sealed_after_restart'] = pending_was_sealed
        evidence['compaction_maintenance_results'] = (
            compaction_maintenance_results
        )
        evidence['compaction_result'] = compaction_result
        evidence['pre_compaction_status'] = compaction_status
        evidence['compacted_status'] = compacted_status
        evidence['restarted_compaction_status'] = (
            restarted_compaction_status
        )
        evidence['sustained_maintenance'] = sustained_maintenance
        evidence['sustained_compactions'] = sustained_compactions
        evidence['sustained_max_segments'] = sustained_max_segments
        evidence['sustained_status'] = sustained_status
        evidence['restarted_sustained_status'] = (
            restarted_sustained_status
        )
        evidence['extent_pressure'] = extent_evidence
        evidence['restarted_extent_status'] = restarted_extent_status
        evidence['docs_l0_status'] = docs_l0_status
        evidence['docs_l0_cache_state'] = docs_l0_cache_state
        evidence['page_native_l0_stream'] = page_native_l0_stream
        evidence['page_native_numeric_l0_topk'] = (
            page_native_numeric_l0_topk
        )
        evidence['page_native_text_l0_topk'] = page_native_text_l0_topk
        evidence['page_native_zero_score_l0_topk'] = (
            page_native_zero_score_l0_topk
        )
        evidence['page_native_zero_score_l0_bounded_topk'] = (
            page_native_zero_score_l0_bounded_topk
        )
        evidence['docs_large_l0_status'] = docs_large_l0_status
        evidence['restarted_docs_l0_status'] = restarted_docs_l0_status
        evidence['ids_l0_status'] = ids_l0_status
        evidence['v3_status'] = status
    except Exception:
        if log_path.is_file():
            print(log_path.read_text(encoding='utf-8'), file=sys.stderr)
        raise
    finally:
        if connection is not None:
            connection.close()
        if started:
            stop_cluster(pg_ctl, data_dir)
        shutil.rmtree(root, ignore_errors=True)

    return {
        'api_version': 'evoke_index_v1',
        'route': 'default convergent segment v3 read parity',
        'gates': gates,
        'passed_gates': sum(gates.values()),
        'total_gates': len(gates),
        'all_gates_passed': all(gates.values()),
        'evidence': evidence,
    }


def main() -> None:
    args = parse_args()
    if args.output is not None:
        args.output.unlink(missing_ok=True)
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
