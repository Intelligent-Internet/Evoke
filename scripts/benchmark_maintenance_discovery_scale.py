#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import psycopg
from psycopg import sql

from evoke_test_support import create_short_socket_root


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')
INDEXES_PER_TABLE = 100
MAINTENANCE_WAKEUP_COOLDOWN_SECONDS = 1.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Measure authoritative evoke maintenance discovery at increasing '
            'catalog sizes.'
        ),
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument(
        '--levels',
        default='100,1000,10000',
        help='comma-separated cumulative evoke index counts',
    )
    parser.add_argument('--batch-size', type=int, default=250)
    parser.add_argument('--timeout-seconds', type=float, default=180.0)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def parse_levels(raw: str) -> list[int]:
    levels = sorted({int(value.strip()) for value in raw.split(',')})
    if not levels or levels[0] <= 0:
        raise ValueError('levels must contain positive integers')
    return levels


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        return int(listener.getsockname()[1])


def run(command: list[str], *, check: bool = True) -> None:
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


def configure_cluster(
    data_dir: Path,
    socket_dir: Path,
    port: int,
) -> None:
    with (data_dir / 'postgresql.conf').open('a', encoding='utf-8') as handle:
        handle.write("\nshared_preload_libraries = 'evoke'\n")
        handle.write("listen_addresses = ''\n")
        handle.write(f"unix_socket_directories = '{socket_dir}'\n")
        handle.write(f'port = {port}\n')
        handle.write('max_worker_processes = 24\n')
        handle.write('max_locks_per_transaction = 1024\n')
        handle.write("evoke.maintenance_timer_interval_ms = '1s'\n")
        handle.write("evoke.preload_timer_interval_ms = '1h'\n")
        handle.write("evoke.shared_runtime_size = '256MB'\n")
        handle.write('autovacuum = off\n')
        handle.write('fsync = off\n')
        handle.write('full_page_writes = off\n')
        handle.write('synchronous_commit = off\n')


def connect(
    socket_dir: Path,
    port: int,
) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname='postgres',
        user='postgres',
        host=str(socket_dir),
        port=port,
        autocommit=True,
    )


def wait_until(
    description: str,
    fn: Callable[[], Any],
    timeout: float,
) -> Any:
    deadline = time.monotonic() + timeout
    last_value: Any = None
    while time.monotonic() < deadline:
        last_value = fn()
        if last_value:
            return last_value
        time.sleep(0.005)
    raise TimeoutError(f'timed out waiting for {description}: {last_value!r}')


def setup_tables(
    connection: psycopg.Connection[Any],
    max_indexes: int,
) -> None:
    table_count = (
        max_indexes + INDEXES_PER_TABLE - 1
    ) // INDEXES_PER_TABLE
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            CREATE EXTENSION evoke;
            CREATE SCHEMA maintenance_scale;
            '''
        )
        cursor.execute(
            sql.SQL(
                '''
            DO $block$
            DECLARE
                table_ordinal integer;
            BEGIN
                FOR table_ordinal IN 0..{} LOOP
                    EXECUTE format(
                        'CREATE TABLE maintenance_scale.docs_%s '
                        '(id bigint, body text)',
                        table_ordinal
                    );
                END LOOP;
            END
            $block$;
            '''
            ).format(sql.Literal(table_count - 1))
        )


def create_indexes(
    connection: psycopg.Connection[Any],
    start: int,
    stop: int,
    batch_size: int,
) -> float:
    started_at = time.monotonic()
    with connection.cursor() as cursor:
        for batch_start in range(start, stop + 1, batch_size):
            batch_stop = min(stop, batch_start + batch_size - 1)
            cursor.execute(
                sql.SQL(
                    '''
                DO $block$
                DECLARE
                    index_ordinal integer;
                    table_ordinal integer;
                BEGIN
                    FOR index_ordinal IN {}..{} LOOP
                        table_ordinal :=
                            (index_ordinal - 1) / {};
                        EXECUTE format(
                            'CREATE INDEX evoke_scale_%s '
                            'ON maintenance_scale.docs_%s '
                            'USING evoke (body) '
                            'WITH (consistency = manual)',
                            index_ordinal,
                            table_ordinal
                        );
                    END LOOP;
                END
                $block$;
                '''
                ).format(
                    sql.Literal(batch_start),
                    sql.Literal(batch_stop),
                    sql.Literal(INDEXES_PER_TABLE),
                )
            )
    return (time.monotonic() - started_at) * 1000.0


def mark_indexes_eventual(
    connection: psycopg.Connection[Any],
    start: int,
    stop: int,
    batch_size: int,
) -> float:
    started_at = time.monotonic()
    with connection.cursor() as cursor:
        for batch_start in range(start, stop + 1, batch_size):
            batch_stop = min(stop, batch_start + batch_size - 1)
            cursor.execute(
                sql.SQL(
                    '''
                DO $block$
                DECLARE
                    index_ordinal integer;
                BEGIN
                    FOR index_ordinal IN {}..{} LOOP
                        EXECUTE format(
                            'ALTER INDEX maintenance_scale.evoke_scale_%s '
                            'SET (consistency = eventual)',
                            index_ordinal
                        );
                    END LOOP;
                END
                $block$;
                '''
                ).format(
                    sql.Literal(batch_start),
                    sql.Literal(batch_stop),
                )
            )
    return (time.monotonic() - started_at) * 1000.0


def cache_state(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT evoke_index_runtime_state_json("
            "'maintenance_scale.evoke_scale_1'::regclass)"
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid runtime state: {row}')
    return dict(row[0])


def maintenance_state(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    return dict(cache_state(connection).get('maintenance', {}))


def wait_for_idle(
    connection: psycopg.Connection[Any],
    timeout: float,
) -> dict[str, Any]:
    return wait_until(
        'maintenance workers to become idle',
        lambda: (
            state
            if (
                int(
                    (state := maintenance_state(connection)).get(
                        'active_background_workers',
                        -1,
                    )
                )
                == 0
                and int(
                    state.get('pending_maintenance_worker_launches', -1)
                )
                == 0
                and not bool(
                    state.get('maintenance_reconcile_running', True)
                )
            )
            else None
        ),
        timeout,
    )


def catalog_index_count(
    connection: psycopg.Connection[Any],
) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT count(*)
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_am AS access_method
                ON access_method.oid = relation.relam
            JOIN pg_catalog.pg_index AS index_catalog
                ON index_catalog.indexrelid = relation.oid
            WHERE relation.relkind = 'i'
              AND access_method.amname = 'evoke'
              AND index_catalog.indisvalid
              AND index_catalog.indisready
              AND COALESCE(relation.reloptions, ARRAY[]::text[])
                  @> ARRAY['consistency=eventual']::text[]
            '''
        )
        row = cursor.fetchone()
    if row is None:
        raise AssertionError('catalog count returned no row')
    return int(row[0])


def measure_reconciliation(
    connection: psycopg.Connection[Any],
    expected_rows: int,
    timeout: float,
) -> dict[str, Any]:
    before = wait_for_idle(connection, timeout)
    before_count = int(before.get('maintenance_reconcile_count', -1))
    before_overflows = int(before.get('work_hint_overflows', -1))
    time.sleep(MAINTENANCE_WAKEUP_COOLDOWN_SECONDS)

    started_at = time.monotonic()
    with connection.cursor() as cursor:
        cursor.execute('SELECT evoke_index_touch_maintenance()')
        touch_result = str(cursor.fetchone()[0])

    reconciled = wait_until(
        'authoritative maintenance catalog scan',
        lambda: (
            state
            if (
                int(
                    (state := maintenance_state(connection)).get(
                        'maintenance_reconcile_count',
                        -1,
                    )
                )
                > before_count
            )
            else None
        ),
        timeout,
    )
    catalog_completion_ms = (time.monotonic() - started_at) * 1000.0
    observed_rows = int(
        reconciled.get('maintenance_reconcile_last_rows', -1)
    )
    if observed_rows != expected_rows:
        raise AssertionError(
            'maintenance reconciliation scanned an unexpected number of '
            f'indexes: expected={expected_rows}, observed={observed_rows}, '
            f'state={reconciled}'
        )
    try:
        after = wait_for_idle(connection, timeout)
    except TimeoutError as exc:
        state = maintenance_state(connection)
        raise TimeoutError(
            'maintenance catalog scan completed but the worker did not '
            f'finish due checks: expected_rows={expected_rows}, '
            f'catalog_completion_ms={catalog_completion_ms:.3f}, '
            f'state={state}'
        ) from exc
    cycle_ms = (time.monotonic() - started_at) * 1000.0
    return {
        'touch_result': touch_result,
        'catalog_completion_ms': round(catalog_completion_ms, 3),
        'worker_cycle_ms': round(cycle_ms, 3),
        'catalog_scan_ms': round(
            int(after['maintenance_reconcile_last_duration_us']) / 1000.0,
            3,
        ),
        'catalog_rows': observed_rows,
        'reconcile_count': int(after['maintenance_reconcile_count']),
        'max_catalog_scan_ms': round(
            int(after['maintenance_reconcile_max_duration_us']) / 1000.0,
            3,
        ),
        'hint_overflow_delta': (
            int(after.get('work_hint_overflows', -1)) - before_overflows
        ),
        'pending_hint_entries': int(
            after.get('pending_maintenance_hint_entries', -1)
        ),
    }


def exercise(
    connection: psycopg.Connection[Any],
    levels: list[int],
    batch_size: int,
    timeout: float,
) -> dict[str, Any]:
    setup_tables(connection, levels[-1])
    results: list[dict[str, Any]] = []
    existing = 0
    for level in levels:
        create_ms = create_indexes(
            connection,
            existing + 1,
            level,
            batch_size,
        )
        alter_ms = mark_indexes_eventual(
            connection,
            existing + 1,
            level,
            batch_size,
        )
        count = catalog_index_count(connection)
        if count != level:
            raise AssertionError(
                f'expected {level} eventual indexes, found {count}'
            )
        measurement = measure_reconciliation(
            connection,
            level,
            timeout,
        )
        measurement.update(
            {
                'index_count': level,
                'setup_create_ms': round(create_ms, 3),
                'setup_alter_ms': round(alter_ms, 3),
            }
        )
        results.append(measurement)
        existing = level

    return {
        'levels': results,
        'indexes_per_table': INDEXES_PER_TABLE,
        'batch_size': batch_size,
        'maintenance_wakeup_cooldown_seconds': (
            MAINTENANCE_WAKEUP_COOLDOWN_SECONDS
        ),
        'durability_disabled_for_setup': True,
    }


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.levels)
    if args.batch_size <= 0:
        raise ValueError('batch size must be positive')
    pg_bin = args.pg_bin.expanduser().resolve()
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'
    for executable in (initdb, pg_ctl, pg_bin / 'postgres'):
        if not executable.is_file():
            raise FileNotFoundError(
                f'missing PostgreSQL executable: {executable}'
            )

    socket_root = create_short_socket_root('evoke-maint-scale-')
    evidence: dict[str, Any]
    try:
        with tempfile.TemporaryDirectory(
            prefix='evoke-maint-scale-data-',
        ) as temp:
            root = Path(temp)
            data_dir = root / 'data'
            socket_dir = socket_root / 's'
            log_path = root / 'postgres.log'
            socket_dir.mkdir()
            port = reserve_port()
            started = False
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
                configure_cluster(data_dir, socket_dir, port)
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
                started = True
                with connect(socket_dir, port) as connection:
                    evidence = exercise(
                        connection,
                        levels,
                        args.batch_size,
                        args.timeout_seconds,
                    )
            except Exception:
                if log_path.is_file():
                    print(
                        log_path.read_text(encoding='utf-8'),
                        file=sys.stderr,
                    )
                raise
            finally:
                if started:
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
    finally:
        shutil.rmtree(socket_root, ignore_errors=True)

    rendered = json.dumps(evidence, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
