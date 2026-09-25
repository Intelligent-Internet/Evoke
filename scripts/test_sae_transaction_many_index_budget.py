#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psycopg

from test_unified_index_lifecycle_smoke import (
    backend_memory_state,
    configure_cluster,
    connect,
    run,
    sql_literal,
    start_cluster,
    stop_cluster,
)


SCHEMA = 'sae_many_index_budget'
TABLE = f'{SCHEMA}.docs'
DEFAULT_BUDGET_BYTES = 16 * 1024 * 1024
DEFAULT_RSS_GROWTH_LIMIT_KIB = 96 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Prove the transaction-wide SSR mutation budget across indexes.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument('--port', type=int, default=55593)
    parser.add_argument('--index-count', type=int, default=24)
    parser.add_argument('--payload-bytes', type=int, default=768 * 1024)
    parser.add_argument(
        '--budget-bytes',
        type=int,
        default=DEFAULT_BUDGET_BYTES,
    )
    parser.add_argument(
        '--rss-growth-limit-kib',
        type=int,
        default=DEFAULT_RSS_GROWTH_LIMIT_KIB,
    )
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def index_name(ordinal: int) -> str:
    return f'{SCHEMA}.docs_idx_{ordinal:02d}'


def backend_rss_kib(pid: int) -> int:
    result = subprocess.run(
        ['ps', '-o', 'rss=', '-p', str(pid)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f'could not read RSS for PostgreSQL backend {pid}: '
            f'{result.stderr.strip()}'
        )
    return int(result.stdout.strip())


def backend_pid(connection: psycopg.Connection[Any]) -> int:
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_backend_pid()')
        return int(cursor.fetchone()[0])


def setup(
    connection: psycopg.Connection[Any],
    model_path: Path,
    index_count: int,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE EXTENSION evoke;
            CREATE SCHEMA {SCHEMA};
            CREATE TABLE {TABLE} (
                id text PRIMARY KEY,
                body text NOT NULL
            );
            INSERT INTO {TABLE} VALUES (
                'baseline',
                'aggregate memory baseline document'
            );
            """
        )
        for ordinal in range(index_count):
            cursor.execute(
                f"""
                CREATE INDEX docs_idx_{ordinal:02d}
                ON {TABLE}
                USING evoke (body)
                WITH (
                    sae = true,
                    model_path = {sql_literal(str(model_path))},
                    consistency = eventual,
                    auto_preload = 0
                )
                """
            )


def make_body(marker: str, payload_bytes: int) -> str:
    prefix = f'aggregate memory sentinel {marker} '
    if len(prefix.encode('utf-8')) >= payload_bytes:
        raise ValueError('payload-bytes must exceed the marker prefix')
    repeated = 'alpha '
    repeat_count = (
        payload_bytes - len(prefix.encode('utf-8'))
    ) // len(repeated)
    return prefix + repeated * repeat_count


def set_budget(
    connection: psycopg.Connection[Any],
    budget_bytes: int,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT set_config('
            "'evoke.sae_transaction_mutation_max_bytes', %s, false)",
            (str(budget_bytes),),
        )


def total_delta_records(
    connection: psycopg.Connection[Any],
    index_count: int,
) -> int:
    total = 0
    with connection.cursor() as cursor:
        for ordinal in range(index_count):
            cursor.execute(
                'SELECT ('
                'evoke_index_status(%s::regclass)'
                "->'details'->>'delta_records'"
                ')::int8',
                (index_name(ordinal),),
            )
            total += int(cursor.fetchone()[0])
    return total


def query_ids(
    connection: psycopg.Connection[Any],
    ordinal: int,
    marker: str,
    *,
    oracle: bool,
) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_config("
            "'evoke.test_unified_overlay_oracle', %s, false)",
            ('on' if oracle else 'off',),
        )
        cursor.execute(
            f"""
            SELECT source.id
            FROM evoke_query(
                %s::regclass,
                %s,
                10
            ) AS hit
            JOIN {TABLE} AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """,
            (index_name(ordinal), marker),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def query_matrix(
    connection: psycopg.Connection[Any],
    index_count: int,
    marker: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for ordinal in range(index_count):
        normal = query_ids(
            connection,
            ordinal,
            marker,
            oracle=False,
        )
        oracle = query_ids(
            connection,
            ordinal,
            marker,
            oracle=True,
        )
        rows[index_name(ordinal)] = {
            'normal': normal,
            'oracle': oracle,
            'parity': normal == oracle,
        }
    return rows


def sample_backend_rss(
    pid: int,
    stop_event: threading.Event,
    samples: list[int],
    errors: list[str],
) -> None:
    try:
        while not stop_event.is_set():
            samples.append(backend_rss_kib(pid))
            stop_event.wait(0.005)
        samples.append(backend_rss_kib(pid))
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')


def rollback_fixture(
    connection: psycopg.Connection[Any],
    body: str,
) -> dict[str, Any]:
    connection.autocommit = False
    try:
        with connection.cursor() as cursor:
            cursor.execute('SAVEPOINT evoke_many_index_rollback')
            cursor.execute(
                f'INSERT INTO {TABLE} (id, body) VALUES (%s, %s)',
                ('rolledback-row', body),
            )
            before_rollback = backend_memory_state(connection)
            cursor.execute(
                'ROLLBACK TO SAVEPOINT evoke_many_index_rollback'
            )
            cursor.execute(
                'RELEASE SAVEPOINT evoke_many_index_rollback'
            )
            after_rollback = backend_memory_state(connection)
            cursor.execute(
                f"SELECT count(*) FROM {TABLE} "
                "WHERE id = 'rolledback-row'"
            )
            visible_rows = int(cursor.fetchone()[0])
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.autocommit = True
    return {
        'before_rollback': before_rollback,
        'after_rollback': after_rollback,
        'visible_rows': visible_rows,
        'passed': (
            visible_rows == 0
            and after_rollback['pending_contexts'] == 0
            and after_rollback['pending_context_bytes'] == 0
        ),
    }


def commit_fixture(
    connection: psycopg.Connection[Any],
    body: str,
    index_count: int,
    budget_bytes: int,
    rss_growth_limit_kib: int,
) -> dict[str, Any]:
    pid = backend_pid(connection)
    baseline_memory = backend_memory_state(connection)
    baseline_rss_kib = backend_rss_kib(pid)
    rss_samples: list[int] = [baseline_rss_kib]
    sample_errors: list[str] = []
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=sample_backend_rss,
        args=(pid, stop_event, rss_samples, sample_errors),
        daemon=True,
    )

    connection.autocommit = False
    sampler.start()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f'INSERT INTO {TABLE} (id, body) VALUES (%s, %s)',
                ('committed-row', body),
            )
        retained_memory = backend_memory_state(connection)
        flushed_delta_records = total_delta_records(
            connection,
            index_count,
        )
        read_your_writes = query_matrix(
            connection,
            index_count,
            'commit marker',
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        stop_event.set()
        sampler.join(timeout=5.0)
        connection.autocommit = True
    after_commit = backend_memory_state(connection)
    final_rss_kib = backend_rss_kib(pid)
    rss_samples.append(final_rss_kib)
    max_rss_kib = max(rss_samples)
    rss_growth_kib = max_rss_kib - baseline_rss_kib
    target_visible = all(
        'committed-row' in row['normal']
        and row['parity']
        for row in read_your_writes.values()
    )
    return {
        'backend_pid': pid,
        'baseline_memory': baseline_memory,
        'retained_memory': retained_memory,
        'after_commit_memory': after_commit,
        'flushed_delta_records_before_commit': flushed_delta_records,
        'read_your_writes': read_your_writes,
        'target_visible': target_visible,
        'rss_samples': len(rss_samples),
        'baseline_rss_kib': baseline_rss_kib,
        'max_rss_kib': max_rss_kib,
        'final_rss_kib': final_rss_kib,
        'rss_growth_kib': rss_growth_kib,
        'rss_growth_limit_kib': rss_growth_limit_kib,
        'sample_errors': sample_errors,
        'passed': (
            retained_memory['pending_context_bytes'] <= budget_bytes
            and flushed_delta_records > 0
            and target_visible
            and after_commit['pending_contexts'] == 0
            and after_commit['pending_context_bytes'] == 0
            and not sample_errors
            and rss_growth_kib <= rss_growth_limit_kib
        ),
    }


def late_failure_fixture(
    connection: psycopg.Connection[Any],
    body: str,
) -> dict[str, Any]:
    error_message = ''
    connection.autocommit = False
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT set_config("
                "'evoke.test_precommit_error_after_flush', 'on', true)"
            )
            cursor.execute(
                f'INSERT INTO {TABLE} (id, body) VALUES (%s, %s)',
                ('late-failure-row', body),
            )
        try:
            connection.commit()
        except psycopg.Error as error:
            error_message = str(error).splitlines()[0]
            connection.rollback()
        else:
            raise AssertionError('late pre-commit fault did not fire')
    finally:
        connection.autocommit = True
    memory_after = backend_memory_state(connection)
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT count(*) FROM {TABLE} "
            "WHERE id = 'late-failure-row'"
        )
        visible_rows = int(cursor.fetchone()[0])
    return {
        'error': error_message,
        'memory_after': memory_after,
        'visible_rows': visible_rows,
        'passed': (
            'injected evoke error after pre-commit delta flush'
            in error_message
            and visible_rows == 0
            and memory_after['pending_contexts'] == 0
            and memory_after['pending_context_bytes'] == 0
        ),
    }


def prepared_fixture(
    connection: psycopg.Connection[Any],
    body: str,
    budget_bytes: int,
    index_count: int,
) -> dict[str, Any]:
    gid = 'evoke_many_index_budget'
    set_budget(connection, budget_bytes)
    with connection.cursor() as cursor:
        cursor.execute('BEGIN')
        cursor.execute(
            f'INSERT INTO {TABLE} (id, body) VALUES (%s, %s)',
            ('prepared-row', body),
        )
        cursor.execute(f"PREPARE TRANSACTION '{gid}'")
    memory_after_prepare = backend_memory_state(connection)
    with connection.cursor() as cursor:
        cursor.execute(f"COMMIT PREPARED '{gid}'")
    matrix = query_matrix(
        connection,
        index_count,
        'prepared marker',
    )
    target_visible = all(
        'prepared-row' in row['normal'] and row['parity']
        for row in matrix.values()
    )
    memory_after_commit = backend_memory_state(connection)
    return {
        'memory_after_prepare': memory_after_prepare,
        'memory_after_commit': memory_after_commit,
        'query_matrix': matrix,
        'target_visible': target_visible,
        'passed': (
            memory_after_prepare['pending_contexts'] == 0
            and memory_after_prepare['pending_context_bytes'] == 0
            and target_visible
            and memory_after_commit['pending_contexts'] == 0
            and memory_after_commit['pending_context_bytes'] == 0
        ),
    }


def main() -> None:
    args = parse_args()
    if not (args.model_path / 'manifest.json').is_file():
        raise FileNotFoundError(
            f'model checkout is missing: {args.model_path}'
        )
    if args.index_count < 2:
        raise ValueError('index-count must be at least two')
    if args.payload_bytes <= 0:
        raise ValueError('payload-bytes must be positive')
    if args.budget_bytes < DEFAULT_BUDGET_BYTES:
        raise ValueError('budget-bytes must be at least 16 MiB')

    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    temporary = tempfile.TemporaryDirectory(delete=not args.keep)
    root = Path(temporary.name)
    data_dir = root / 'data'
    socket_dir = root / 'socket'
    log_path = root / 'postgres.log'
    socket_dir.mkdir()
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
        args.port,
        args.extension_libdir,
        args.extension_control_dir,
    )
    with (data_dir / 'postgresql.conf').open(
        'a',
        encoding='utf-8',
    ) as handle:
        handle.write(
            "evoke.maintenance_timer_interval_ms = '3600000ms'\n"
        )
    start_cluster(pg_ctl, data_dir, log_path)

    connection = connect(socket_dir, args.port)
    try:
        setup(connection, args.model_path, args.index_count)
        set_budget(connection, args.budget_bytes)
        warmup_body = make_body('warmup marker', args.payload_bytes)
        warmup = rollback_fixture(connection, warmup_body)
        baseline = backend_memory_state(connection)

        rollback = rollback_fixture(
            connection,
            make_body('rollback marker', args.payload_bytes),
        )
        commit = commit_fixture(
            connection,
            make_body('commit marker', args.payload_bytes),
            args.index_count,
            args.budget_bytes,
            args.rss_growth_limit_kib,
        )
        late_failure = late_failure_fixture(
            connection,
            make_body('late failure marker', args.payload_bytes),
        )
        prepared = prepared_fixture(
            connection,
            make_body('prepared marker', args.payload_bytes),
            args.budget_bytes,
            args.index_count,
        )
        final_memory = backend_memory_state(connection)
        failures: list[str] = []
        fixtures = {
            'warmup': warmup,
            'rollback': rollback,
            'commit': commit,
            'late_failure': late_failure,
            'prepared': prepared,
        }
        for name, fixture in fixtures.items():
            if not fixture['passed']:
                failures.append(f'{name} fixture failed')
        if final_memory['pending_contexts'] != 0:
            failures.append('final pending mutation contexts remain')
        if final_memory['pending_context_bytes'] != 0:
            failures.append('final pending mutation bytes remain')

        report = {
            'index_count': args.index_count,
            'payload_bytes': args.payload_bytes,
            'transaction_budget_bytes': args.budget_bytes,
            'baseline_memory': baseline,
            'fixtures': fixtures,
            'final_memory': final_memory,
            'failures': failures,
            'passed': not failures,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + '\n', encoding='utf-8')
        print(rendered)
        if failures:
            raise SystemExit(1)
    finally:
        connection.close()
        stop_cluster(pg_ctl, data_dir)
        if args.keep:
            print(f'kept temporary cluster at {root}')
        else:
            temporary.cleanup()


if __name__ == '__main__':
    main()
