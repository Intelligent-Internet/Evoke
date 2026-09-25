#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from ii42_test_support import (
    create_short_socket_root,
    extension_control_root,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate writer, generation-barrier, and append-lock concurrency.'
        ),
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument('--statement-timeout-ms', type=int, default=2000)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--model-writers', type=int, default=8)
    parser.add_argument('--model-readers', type=int, default=8)
    parser.add_argument('--model-cycles', type=int, default=12)
    parser.add_argument('--staging-cycles', type=int, default=24)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


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
    extension_libdir: Path | None,
    extension_control_dir: Path | None,
) -> None:
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
        # Every maintenance transition in this harness is driven explicitly.
        # Keep the global supervisor outside the test window so it cannot
        # rotate an L0 chain between deterministic root-snapshot assertions.
        handle.write(
            "ii42.maintenance_timer_interval_ms = '3600000ms'\n"
        )
        handle.write('ii42.maintenance_worker_limit = 0\n')
        handle.write('log_lock_waits = on\n')
        handle.write("deadlock_timeout = '100ms'\n")
        handle.write("log_line_prefix = '%m [%p] %a '\n")
        handle.write("listen_addresses = ''\n")
        handle.write(f"unix_socket_directories = '{socket_dir}'\n")
        handle.write(f'port = {port}\n')
        handle.write('max_worker_processes = 16\n')


def connect(
    socket_dir: Path,
    port: int,
    *,
    autocommit: bool,
) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname='postgres',
        user='postgres',
        host=str(socket_dir),
        port=port,
        autocommit=autocommit,
    )


def setup(connection: psycopg.Connection[Any]) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE EXTENSION ii42;
            CREATE TABLE docs (
                id text PRIMARY KEY,
                body text NOT NULL
            );
            INSERT INTO docs VALUES ('base', 'base stable sentinel');
            CREATE INDEX docs_body_idx
            ON docs USING ii42 (body)
            WITH (
                consistency = eventual
            );
            """
        )


def setup_model_eventual(
    connection: psycopg.Connection[Any],
    model_path: Path,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE model_docs (
                id text PRIMARY KEY,
                body text NOT NULL
            );
            INSERT INTO model_docs VALUES
                ('model-base-a', 'semantic retrieval database index'),
                ('model-base-b', 'clinical evidence cardiovascular trial'),
                ('model-base-c', 'astronomy galaxy telescope observation'),
                ('model-base-d', 'rare zebra quantum flux capacitor');
            """
        )
        cursor.execute(
            sql.SQL(
                """
                CREATE INDEX model_docs_body_idx
                ON model_docs USING ii42 (body)
                WITH (
                    sae = true,
                    model_path = {},
                    consistency = eventual
                )
                """
            ).format(sql.Literal(str(model_path)))
        )


def maintain(
    socket_dir: Path,
    port: int,
) -> str:
    with connect(socket_dir, port, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT ii42_index_maintain('docs_body_idx'::regclass)"
            )
            row = cursor.fetchone()
    if row is None:
        raise AssertionError('maintenance returned no row')
    return str(row[0])


def model_search_ids(
    connection: psycopg.Connection[Any],
    query: str,
    *,
    oracle: bool = False,
) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_config("
            "'ii42.test_unified_overlay_oracle', %s, false)",
            ('on' if oracle else 'off',),
        )
        cursor.execute(
            """
            SELECT docs.id
            FROM ii42_query(
                'model_docs_body_idx'::regclass,
                %s,
                100
            ) AS hit
            JOIN model_docs AS docs ON docs.ctid = hit.ctid
            ORDER BY docs.id
            """,
            (query,),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def lexical_search_ids(
    connection: psycopg.Connection[Any],
    query: str,
) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT docs.id
            FROM ii42_query(
                'docs_body_idx'::regclass,
                %s,
                100
            ) AS hit
            JOIN docs ON docs.ctid = hit.ctid
            ORDER BY docs.id
            """,
            (query,),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def fetch_json(
    connection: psycopg.Connection[Any],
    query: str,
    params: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid JSON result: {row}')
    return dict(row[0])


def fetch_deep_status(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT ii42_index_status(%s::regclass), '
            'ii42_index_generation_audit_internal(%s::regclass)',
            (index_name, index_name),
        )
        row = cursor.fetchone()
    if (
        row is None
        or not isinstance(row[0], dict)
        or not isinstance(row[1], dict)
    ):
        raise AssertionError(f'invalid deep status for {index_name}')
    status = dict(row[0])
    generation = status.get('generation')
    if not isinstance(generation, dict):
        raise AssertionError(f'invalid generation for {index_name}')
    status['generation'] = generation | dict(row[1])
    return status


def model_status(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    return fetch_json(
        connection,
        "SELECT ii42_index_status('model_docs_body_idx'::regclass)",
    )


def lexical_status(
    connection: psycopg.Connection[Any],
) -> dict[str, Any]:
    return fetch_json(
        connection,
        "SELECT ii42_index_status('docs_body_idx'::regclass)",
    )


def wait_for_extension_pause(
    connection: psycopg.Connection[Any],
    pid: int,
    future: concurrent.futures.Future[Any],
    description: str,
) -> None:
    deadline = time.monotonic() + 15.0
    while True:
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
                (pid,),
            )
            paused = bool(cursor.fetchone()[0])
        if paused:
            return
        if future.done():
            raise AssertionError(
                f'{description} finished before its test pause: '
                f'{future.result()}'
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f'{description} did not reach its test pause')
        time.sleep(0.002)


def maintain_index_until_clean(
    connection: psycopg.Connection[Any],
    *,
    index_name: str,
    max_attempts: int = 64,
) -> dict[str, Any]:
    results: list[str] = []

    for _ in range(max_attempts + 1):
        status_value = fetch_json(
            connection,
            "SELECT ii42_index_status(%s::regclass)",
            (index_name,),
        )
        details = status_value.get('details', {})
        if len(results) == max_attempts:
            break
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT ii42_index_try_maintain(%s::regclass)",
                (index_name,),
            )
            row = cursor.fetchone()
        if row is None:
            raise AssertionError(
                f'{index_name} maintenance returned no row'
            )
        result = str(row[0])
        results.append(result)
        if 'maintained=false' in result and 'reason=no_pending' in result:
            status_value = fetch_json(
                connection,
                "SELECT ii42_index_status(%s::regclass)",
                (index_name,),
            )
            details = status_value.get('details', {})
            if (
                int(details.get('delta_records', -1)) == 0
                and int(details.get('pending_writes', -1)) == 0
                and int(details.get('pending_deletes', -1)) == 0
                and details.get('stale') is False
            ):
                return {
                    'results': results,
                    'status': status_value,
                }

    evidence = {
        'results': results,
        'status': status_value,
    }
    raise AssertionError(
        f'{index_name} maintenance did not converge within '
        f'{max_attempts} attempts: '
        f'{json.dumps(evidence, sort_keys=True)}'
    )


def maintenance_result_fields(result: str) -> dict[str, str]:
    prefix = 'ii42_maintenance_result('

    if not result.startswith(prefix) or not result.endswith(')'):
        raise AssertionError(f'invalid maintenance result: {result}')
    fields: dict[str, str] = {}
    for item in result[len(prefix):-1].split(', '):
        name, separator, value = item.partition('=')
        if not separator or not name or not value:
            raise AssertionError(f'invalid maintenance field: {item}')
        fields[name] = value
    return fields


def maintain_model_until_clean(
    connection: psycopg.Connection[Any],
    *,
    max_attempts: int = 64,
) -> dict[str, Any]:
    return maintain_index_until_clean(
        connection,
        index_name='model_docs_body_idx',
        max_attempts=max_attempts,
    )


def exercise_vacuum_maintenance_authority(
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    index_name = 'authority_docs_body_idx'
    maintenance_connection = connect(socket_dir, port, autocommit=True)
    session_lock_held = False
    try:
        with maintenance_connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE authority_docs (
                    id text PRIMARY KEY,
                    body text NOT NULL
                );
                INSERT INTO authority_docs VALUES
                    ('retired', 'retired document boundary'),
                    ('stable', 'stable document boundary');
                CREATE INDEX authority_docs_body_idx
                ON authority_docs USING ii42 (body)
                WITH (consistency = eventual);
                """
            )
            cursor.execute(
                "SELECT ii42_index_try_maintenance_lock(%s::regclass)",
                (index_name,),
            )
            lock_row = cursor.fetchone()
            if lock_row is None or lock_row[0] is not True:
                raise AssertionError(
                    'authority probe could not reserve maintenance'
                )
            session_lock_held = True
            cursor.execute(
                """
                DELETE FROM authority_docs WHERE id = 'retired';
                INSERT INTO authority_docs VALUES
                    ('pending', 'pending lexical boundary');
                """
            )
            cursor.execute(
                "SELECT ii42_index_try_maintain(%s::regclass)",
                (index_name,),
            )
            row = cursor.fetchone()
        rotation = str(row[0]) if row is not None else 'missing'
        if 'reason=active_l0_rotated' not in rotation:
            raise AssertionError(
                f'authority probe did not rotate active L0: {rotation}'
            )

        maintenance_pid: list[int] = []
        maintenance_ready = threading.Event()
        vacuum_pid: list[int] = []
        vacuum_ready = threading.Event()

        def maintenance_actor() -> str:
            nonlocal session_lock_held
            try:
                with maintenance_connection.cursor() as cursor:
                    cursor.execute(
                        "SET ii42.test_online_maintenance_pause_ms = '2000'"
                    )
                    cursor.execute('SELECT pg_backend_pid()')
                    maintenance_pid.append(int(cursor.fetchone()[0]))
                    maintenance_ready.set()
                    for _ in range(100):
                        cursor.execute(
                            "SELECT ii42_index_try_maintain(%s::regclass)",
                            (index_name,),
                        )
                        result = cursor.fetchone()
                        result_text = (
                            str(result[0])
                            if result is not None
                            else 'missing'
                        )
                        if 'reason=xid_horizon' not in result_text:
                            return result_text
                        time.sleep(0.01)
                return result_text
            finally:
                if session_lock_held:
                    with maintenance_connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT ii42_index_maintenance_unlock("
                            "%s::regclass)",
                            (index_name,),
                        )
                    session_lock_held = False

        def vacuum_actor() -> float:
            started = time.monotonic()
            with connect(socket_dir, port, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute('SELECT pg_backend_pid()')
                    vacuum_pid.append(int(cursor.fetchone()[0]))
                    vacuum_ready.set()
                    cursor.execute('VACUUM authority_docs')
            return (time.monotonic() - started) * 1000.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            maintenance_future = executor.submit(maintenance_actor)
            if not maintenance_ready.wait(timeout=15.0):
                raise TimeoutError('authority maintenance actor did not start')
            with connect(socket_dir, port, autocommit=True) as observer:
                wait_for_extension_pause(
                    observer,
                    maintenance_pid[0],
                    maintenance_future,
                    'authority pending seal',
                )
            vacuum_future = executor.submit(vacuum_actor)
            if not vacuum_ready.wait(timeout=15.0):
                raise TimeoutError('authority VACUUM actor did not start')
            lock_deadline = time.monotonic() + 15.0
            vacuum_wait_lock_tag = 0
            while vacuum_wait_lock_tag == 0:
                with connect(
                    socket_dir,
                    port,
                    autocommit=True,
                ) as observer:
                    with observer.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT classid::bigint
                            FROM pg_locks
                            WHERE pid = %s
                              AND locktype = 'advisory'
                              AND NOT granted
                            ORDER BY classid
                            LIMIT 1
                            """,
                            (vacuum_pid[0],),
                        )
                        lock_row = cursor.fetchone()
                if lock_row is not None:
                    vacuum_wait_lock_tag = int(lock_row[0])
                    break
                if vacuum_future.done():
                    break
                if time.monotonic() >= lock_deadline:
                    raise TimeoutError(
                        'authority VACUUM did not reach a lock boundary'
                    )
                time.sleep(0.002)
            vacuum_waited = not vacuum_future.done()
            maintenance_result = maintenance_future.result(timeout=15.0)
            vacuum_elapsed_ms = vacuum_future.result(timeout=15.0)
    finally:
        if session_lock_held:
            with maintenance_connection.cursor() as cursor:
                cursor.execute(
                    "SELECT ii42_index_maintenance_unlock(%s::regclass)",
                    (index_name,),
                )
        maintenance_connection.close()

    with connect(socket_dir, port, autocommit=True) as connection:
        convergence = maintain_index_until_clean(
            connection,
            index_name=index_name,
        )
        normal_rows = []
        oracle_rows = []
        with connection.cursor() as cursor:
            for oracle, target in (
                (False, normal_rows),
                (True, oracle_rows),
            ):
                cursor.execute(
                    "SELECT set_config("
                    "'ii42.test_unified_overlay_oracle', %s, false)",
                    ('on' if oracle else 'off',),
                )
                cursor.execute(
                    """
                    SELECT docs.id
                    FROM ii42_query(
                        'authority_docs_body_idx'::regclass,
                        'boundary',
                        100
                    ) AS hit
                    JOIN authority_docs AS docs ON docs.ctid = hit.ctid
                    ORDER BY docs.id
                    """
                )
                target.extend(str(value[0]) for value in cursor.fetchall())
        status_value = fetch_json(
            connection,
            "SELECT ii42_index_status(%s::regclass)",
            (index_name,),
        )

    details = status_value.get('details', {})
    expected_maintenance_lock_tag = 0x3253424D
    passed = (
        vacuum_waited
        and vacuum_wait_lock_tag == expected_maintenance_lock_tag
        and 'reason=pending_l0_sealed' in maintenance_result
        and normal_rows == oracle_rows
        and normal_rows == ['pending', 'stable']
        and int(details.get('delta_records', -1)) == 0
        and int(details.get('pending_writes', -1)) == 0
        and int(details.get('pending_deletes', -1)) == 0
        and status_value.get('query_ready') is True
    )
    evidence = {
        'rotation': rotation,
        'vacuum_waited_for_maintenance': vacuum_waited,
        'vacuum_wait_lock_tag': vacuum_wait_lock_tag,
        'expected_maintenance_lock_tag': expected_maintenance_lock_tag,
        'vacuum_elapsed_ms': round(vacuum_elapsed_ms, 3),
        'maintenance_result': maintenance_result,
        'convergence': convergence['results'],
        'normal_query_ids': normal_rows,
        'oracle_query_ids': oracle_rows,
        'final_status': status_value,
        'passed': passed,
    }
    if not passed:
        raise AssertionError(
            'VACUUM did not share the online-maintenance authority: '
            f'{json.dumps(evidence, sort_keys=True)}'
        )
    return evidence


def exercise_shared_retirement_sequence_guard(
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    index_name = 'shared_retirement_docs_body_idx'
    table_name = 'shared_retirement_docs'

    def query_ids(
        connection: psycopg.Connection[Any],
        *,
        oracle: bool = False,
    ) -> list[str]:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT set_config("
                "'ii42.test_unified_overlay_oracle', %s, false)",
                ('on' if oracle else 'off',),
            )
            cursor.execute(
                f"""
                SELECT docs.id
                FROM ii42_query(
                    '{index_name}'::regclass,
                    'shared retirement sentinel',
                    100
                ) AS hit
                JOIN {table_name} AS docs ON docs.ctid = hit.ctid
                ORDER BY docs.id
                """
            )
            return [str(row[0]) for row in cursor.fetchall()]

    def delta_records(connection: psycopg.Connection[Any]) -> int:
        status_value = fetch_json(
            connection,
            'SELECT ii42_index_status(%s::regclass)',
            (index_name,),
        )
        return int(
            status_value.get('details', {}).get('delta_records', -1)
        )

    with connect(socket_dir, port, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE {table_name} (
                    id text PRIMARY KEY,
                    body text NOT NULL
                ) WITH (autovacuum_enabled = false);
                INSERT INTO {table_name} VALUES
                    ('stable', 'shared retirement sentinel stable');
                CREATE INDEX {index_name}
                ON {table_name} USING ii42 (body)
                WITH (consistency = eventual);
                INSERT INTO {table_name} VALUES
                    ('dead-a', 'shared retirement sentinel alpha'),
                    ('dead-b', 'shared retirement sentinel beta');
                DELETE FROM {table_name} WHERE id LIKE 'dead-%';
                """
            )

        before_vacuum_records = delta_records(connection)
        with connection.cursor() as cursor:
            cursor.execute(f'VACUUM {table_name}')
        after_first_vacuum_records = delta_records(connection)
        before_seal_normal_ids = query_ids(connection)
        before_seal_oracle_ids = query_ids(connection, oracle=True)

        with connection.cursor() as cursor:
            cursor.execute(f'VACUUM {table_name}')
        after_second_vacuum_records = delta_records(connection)
        after_repeat_normal_ids = query_ids(connection)
        after_repeat_oracle_ids = query_ids(connection, oracle=True)

        convergence = maintain_index_until_clean(
            connection,
            index_name=index_name,
        )
        after_seal_normal_ids = query_ids(connection)
        after_seal_oracle_ids = query_ids(connection, oracle=True)

    passed = (
        before_vacuum_records == 2
        and after_first_vacuum_records == 3
        and after_second_vacuum_records == after_first_vacuum_records
        and before_seal_normal_ids == before_seal_oracle_ids == ['stable']
        and after_repeat_normal_ids == after_repeat_oracle_ids == ['stable']
        and after_seal_normal_ids == after_seal_oracle_ids == ['stable']
        and int(
            convergence['status'].get('details', {}).get(
                'delta_records',
                -1,
            )
        ) == 0
    )
    evidence = {
        'before_vacuum_records': before_vacuum_records,
        'after_first_vacuum_records': after_first_vacuum_records,
        'after_second_vacuum_records': after_second_vacuum_records,
        'before_seal_normal_ids': before_seal_normal_ids,
        'before_seal_oracle_ids': before_seal_oracle_ids,
        'after_repeat_normal_ids': after_repeat_normal_ids,
        'after_repeat_oracle_ids': after_repeat_oracle_ids,
        'convergence': convergence['results'],
        'after_seal_normal_ids': after_seal_normal_ids,
        'after_seal_oracle_ids': after_seal_oracle_ids,
        'passed': passed,
    }
    if not passed:
        raise AssertionError(
            'shared retirement sequence guard failed: '
            f'{json.dumps(evidence, sort_keys=True)}'
        )
    return evidence


def exercise_retired_page_reader_fence(
    socket_dir: Path,
    port: int,
    *,
    pg_ctl: Path,
    data_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    index_name = 'reuse_fence_docs_body_idx'
    table_name = 'reuse_fence_docs'
    history: list[dict[str, Any]] = []

    def query_ids(
        connection: psycopg.Connection[Any],
        *,
        oracle: bool = False,
    ) -> list[str]:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT set_config("
                "'ii42.test_unified_overlay_oracle', %s, false)",
                ('on' if oracle else 'off',),
            )
            cursor.execute(
                f"""
                SELECT docs.id
                FROM ii42_query(
                    '{index_name}'::regclass,
                    'reader fence stable',
                    100
                ) AS hit
                JOIN {table_name} AS docs ON docs.ctid = hit.ctid
                ORDER BY docs.id
                """
            )
            return [str(row[0]) for row in cursor.fetchall()]

    maintenance_connection = connect(socket_dir, port, autocommit=True)
    maintenance_locked = False
    blocker: psycopg.Connection[Any] | None = None
    reader_pid: list[int] = []
    reader_ready = threading.Event()
    maintenance_pid = 0
    prepared_reader_ids: list[str] = []
    busy_result = ''
    pressure_status: dict[str, Any] = {}
    blocker_ids: list[str] = []

    def maintenance_actor() -> str:
        with maintenance_connection.cursor() as cursor:
            cursor.execute(
                'SELECT ii42_index_try_maintain(%s::regclass)',
                (index_name,),
            )
            row = cursor.fetchone()
        return str(row[0]) if row is not None else 'missing'

    def reader_actor() -> list[str]:
        with connect(socket_dir, port, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SET ii42.test_convergent_root_snapshot_pause_ms "
                    "= '1000'"
                )
                cursor.execute('SELECT pg_backend_pid()')
                reader_pid.append(int(cursor.fetchone()[0]))
                reader_ready.set()
            return query_ids(connection)

    def prepared_reader_actor() -> list[str]:
        with connect(socket_dir, port, autocommit=True) as connection:
            return query_ids(connection)

    try:
        with maintenance_connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE {table_name} (
                    id text PRIMARY KEY,
                    body text NOT NULL
                ) WITH (autovacuum_enabled = false);
                INSERT INTO {table_name} VALUES
                    ('stable', 'reader fence stable initial');
                CREATE INDEX {index_name}
                ON {table_name} USING ii42 (body)
                WITH (consistency = eventual);
                """
            )
            cursor.execute(
                'SELECT ii42_index_try_maintenance_lock(%s::regclass)',
                (index_name,),
            )
            row = cursor.fetchone()
        maintenance_locked = row is not None and bool(row[0])
        if not maintenance_locked:
            raise AssertionError(
                'reader-fence fixture could not isolate maintenance'
            )

        blocker = connect(socket_dir, port, autocommit=False)
        blocker_ids = query_ids(blocker)
        with maintenance_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_locks
                    WHERE pid = %s
                      AND relation = %s::regclass
                      AND mode = 'AccessShareLock'
                      AND granted
                )
                """,
                (blocker.info.backend_pid, index_name),
            )
            blocker_holds_fence = bool(cursor.fetchone()[0])
        if not blocker_holds_fence:
            raise AssertionError(
                'retirement-pressure reader did not retain its index lock'
            )

        for cycle in range(144):
            repeated_initial = ' '.join(
                ['initial'] * (1 + cycle % 2)
            )
            with maintenance_connection.cursor() as cursor:
                cursor.execute(
                    f'UPDATE {table_name} SET body = %s WHERE id = %s',
                    (
                        f'reader fence stable {repeated_initial}',
                        'stable',
                    ),
                )
                cursor.execute(f'VACUUM {table_name}')

            cycle_results: list[str] = []
            for _ in range(16):
                with maintenance_connection.cursor() as cursor:
                    cursor.execute(
                        'SELECT ii42_index_try_maintain(%s::regclass)',
                        (index_name,),
                    )
                    row = cursor.fetchone()
                result = str(row[0]) if row is not None else 'missing'
                fields = maintenance_result_fields(result)
                cycle_results.append(result)
                if fields.get('reason') == 'reader_fence_busy':
                    busy_result = result
                    break
                status_value = fetch_json(
                    maintenance_connection,
                    'SELECT ii42_index_status(%s::regclass)',
                    (index_name,),
                )
                if int(
                    status_value.get('details', {}).get(
                        'delta_records',
                        -1,
                    )
                ) == 0:
                    break
            pressure_status = fetch_deep_status(
                maintenance_connection,
                index_name,
            )
            pressure_primary = pressure_status.get('generation', {}).get(
                'primary',
                {},
            )
            history.append({
                'cycle': cycle,
                'retired_hint_ranges': int(
                    pressure_primary.get('retired_hint_ranges', -1)
                ),
                'retired_hint_blocks': int(
                    pressure_primary.get('retired_hint_blocks', -1)
                ),
                'interior_unreachable_blocks': int(
                    pressure_primary.get('interior_unreachable_blocks', -1)
                ),
                'recyclable_marker_blocks': int(
                    pressure_primary.get('recyclable_marker_blocks', -1)
                ),
                'results': cycle_results,
            })
            if busy_result:
                break

        pressure_primary = pressure_status.get('generation', {}).get(
            'primary',
            {},
        )
        pressure_ranges = int(
            pressure_primary.get('retired_hint_ranges', -1)
        )
        pressure_accounted = (
            int(pressure_primary.get('retired_hint_blocks', -1))
            + int(pressure_primary.get('recyclable_marker_blocks', -1))
        )
        pressure_interior = int(
            pressure_primary.get('interior_unreachable_blocks', -1)
        )
        if (
            not busy_result
            or not 56 <= pressure_ranges <= 64
            or pressure_accounted != pressure_interior
        ):
            raise AssertionError(
                'reader-fence fixture did not stop losslessly at the '
                'retirement-range boundary: '
                f'busy={busy_result}, status={pressure_status}, '
                f'history={json.dumps(history, sort_keys=True)}'
            )

        blocker.rollback()
        blocker.close()
        blocker = None

        with maintenance_connection.cursor() as cursor:
            cursor.execute(
                "SET ii42.test_prepared_cow_publication_pause_ms = '2000'"
            )
            cursor.execute(
                "SET ii42.test_reuse_reader_fence_pause_ms = '2000'"
            )
            cursor.execute('SELECT pg_backend_pid()')
            maintenance_pid = int(cursor.fetchone()[0])

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=3,
        ) as executor:
            maintenance_future = executor.submit(maintenance_actor)
            with connect(
                socket_dir,
                port,
                autocommit=True,
            ) as observer:
                wait_for_extension_pause(
                    observer,
                    maintenance_pid,
                    maintenance_future,
                    'retired-page overflow handoff',
                )

            prepared_reader_future = executor.submit(
                prepared_reader_actor
            )
            prepared_reader_ids = prepared_reader_future.result(
                timeout=1.5
            )

            publication_fence_ready = False
            fence_deadline = time.monotonic() + 15.0
            while not publication_fence_ready:
                with connect(
                    socket_dir,
                    port,
                    autocommit=True,
                ) as observer:
                    with observer.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT EXISTS (
                                SELECT 1
                                FROM pg_locks
                                WHERE pid = %s
                                  AND locktype = 'relation'
                                  AND relation = %s::regclass
                                  AND mode = 'AccessExclusiveLock'
                                  AND granted
                            )
                            """,
                            (maintenance_pid, index_name),
                        )
                        publication_fence_ready = bool(
                            cursor.fetchone()[0]
                        )
                if publication_fence_ready:
                    break
                if maintenance_future.done():
                    raise AssertionError(
                        'retired-page publication finished before its fence '
                        f'pause: {maintenance_future.result()}'
                    )
                if time.monotonic() >= fence_deadline:
                    raise TimeoutError(
                        'retired-page publication did not acquire its fence'
                    )
                time.sleep(0.002)

            reader_future = executor.submit(reader_actor)
            if not reader_ready.wait(timeout=15.0):
                raise TimeoutError('reader-fence query actor did not start')

            relation_waited = False
            wait_deadline = time.monotonic() + 15.0
            while not relation_waited:
                with connect(
                    socket_dir,
                    port,
                    autocommit=True,
                ) as observer:
                    with observer.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT EXISTS (
                                SELECT 1
                                FROM pg_locks
                                WHERE pid = %s
                                  AND locktype = 'relation'
                                  AND relation = %s::regclass
                                  AND mode = 'AccessShareLock'
                                  AND NOT granted
                            )
                            """,
                            (reader_pid[0], index_name),
                        )
                        relation_waited = bool(cursor.fetchone()[0])
                if relation_waited or reader_future.done():
                    break
                if time.monotonic() >= wait_deadline:
                    raise TimeoutError(
                        'query did not wait on the retirement handoff fence'
                    )
                time.sleep(0.002)

            maintenance_result = maintenance_future.result(timeout=15.0)
            paused_reader_ids = reader_future.result(timeout=15.0)

        with maintenance_connection.cursor() as cursor:
            cursor.execute(
                'RESET ii42.test_prepared_cow_publication_pause_ms'
            )
            cursor.execute('RESET ii42.test_reuse_reader_fence_pause_ms')
        convergence = maintain_index_until_clean(
            maintenance_connection,
            index_name=index_name,
        )
        normal_ids = query_ids(maintenance_connection)
        oracle_ids = query_ids(maintenance_connection, oracle=True)
        final_status = fetch_deep_status(
            maintenance_connection,
            index_name,
        )
        final_primary = final_status.get('generation', {}).get(
            'primary',
            {},
        )
        final_accounted = (
            int(final_primary.get('retired_hint_blocks', -1))
            + int(final_primary.get('recyclable_marker_blocks', -1))
        )
        final_interior = int(
            final_primary.get('interior_unreachable_blocks', -1)
        )
        handoff_fields = maintenance_result_fields(maintenance_result)
        handoff_reused_blocks = int(
            handoff_fields.get('reused_blocks', '0')
        )
        passed = (
            relation_waited
            and prepared_reader_ids == ['stable']
            and handoff_fields.get('maintained') == 'true'
            and handoff_reused_blocks > 0
            and paused_reader_ids == ['stable']
            and normal_ids == oracle_ids == ['stable']
            and final_status.get('query_ready') is True
            and int(
                final_status.get('details', {}).get('delta_records', -1)
            ) == 0
            and final_accounted == final_interior
        )
        if not passed:
            raise AssertionError(
                'retired-page overflow handoff failed: '
                f'result={maintenance_result}, status={final_status}'
            )
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        if maintenance_locked:
            with maintenance_connection.cursor() as cursor:
                cursor.execute(
                    'SELECT ii42_index_maintenance_unlock(%s::regclass)',
                    (index_name,),
                )
        maintenance_connection.close()

    run([
        str(pg_ctl),
        '-D',
        str(data_dir),
        '-l',
        str(log_path),
        'restart',
        '-m',
        'fast',
        '-w',
    ])
    with connect(socket_dir, port, autocommit=True) as connection:
        restarted_ids = query_ids(connection)
        restarted_oracle_ids = query_ids(connection, oracle=True)
        restarted_status = fetch_deep_status(
            connection,
            index_name,
        )
        restarted_primary = restarted_status.get('generation', {}).get(
            'primary',
            {},
        )
        restarted_accounted = (
            int(restarted_primary.get('retired_hint_blocks', -1))
            + int(restarted_primary.get('recyclable_marker_blocks', -1))
        )
        restarted_interior = int(
            restarted_primary.get('interior_unreachable_blocks', -1)
        )
        before_reuse_blocks = int(
            restarted_primary.get('physical_blocks', -1)
        )
        with connection.cursor() as cursor:
            cursor.execute(
                f'UPDATE {table_name} SET body = %s WHERE id = %s',
                ('reader fence stable reuse proof', 'stable'),
            )
            cursor.execute(f'VACUUM {table_name}')
        reuse_convergence = maintain_index_until_clean(
            connection,
            index_name=index_name,
        )
        reuse_fields = [
            maintenance_result_fields(result)
            for result in reuse_convergence['results']
        ]
        reused_blocks = max(
            (int(fields.get('reused_blocks', '0')) for fields in reuse_fields),
            default=0,
        )
        reuse_status = fetch_deep_status(
            connection,
            index_name,
        )
        reuse_ids = query_ids(connection)
        reuse_oracle_ids = query_ids(connection, oracle=True)

    evidence = {
        'history': history,
        'blocking_reader_ids': blocker_ids,
        'overflow_busy_result': busy_result,
        'pressure_status': pressure_status,
        'prepared_closure_reader_ids': prepared_reader_ids,
        'query_waited_for_relation_fence': relation_waited,
        'maintenance_result': maintenance_result,
        'paused_reader_ids': paused_reader_ids,
        'convergence': convergence['results'],
        'normal_query_ids': normal_ids,
        'oracle_query_ids': oracle_ids,
        'final_status': final_status,
        'restarted_query_ids': restarted_ids,
        'restarted_oracle_query_ids': restarted_oracle_ids,
        'restarted_status': restarted_status,
        'restart_accounted_blocks': restarted_accounted,
        'restart_interior_blocks': restarted_interior,
        'before_reuse_physical_blocks': before_reuse_blocks,
        'reuse_maintenance': reuse_convergence['results'],
        'reused_blocks': reused_blocks,
        'reuse_status': reuse_status,
        'reuse_query_ids': reuse_ids,
        'reuse_oracle_query_ids': reuse_oracle_ids,
        'passed': (
            passed
            and restarted_ids == restarted_oracle_ids == ['stable']
            and restarted_accounted == restarted_interior
            and reused_blocks > 0
            and reuse_ids == reuse_oracle_ids == ['stable']
        ),
    }
    if not evidence['passed']:
        raise AssertionError(
            'retired-page restart/reuse closure failed: '
            f'{json.dumps(evidence, sort_keys=True)}'
        )
    return evidence


def exercise_linked_l0_logical_prefix(
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    observer = connect(socket_dir, port, autocommit=True)
    maintenance_locked = False

    with observer.cursor() as cursor:
        cursor.execute(
            "SELECT ii42_index_try_maintenance_lock("
            "'docs_body_idx'::regclass)"
        )
        row = cursor.fetchone()
    maintenance_locked = row is not None and bool(row[0])
    if not maintenance_locked:
        observer.close()
        raise AssertionError(
            'linked-L0 prefix fixture could not isolate maintenance'
        )

    baseline_rows = lexical_search_ids(observer, 'base stable sentinel')

    def frontier_page_count(status_value: dict[str, Any]) -> int:
        delta = status_value.get('generation', {}).get('delta', {})
        return sum(
            int(delta.get(role, {}).get('pages', 0))
            for role in ('active', 'pending')
        )

    def active_page_count(status_value: dict[str, Any]) -> int:
        return int(
            status_value.get('generation', {})
            .get('delta', {})
            .get('active', {})
            .get('pages', 0)
        )

    def frontier_bytes(status_value: dict[str, Any]) -> int:
        return int(
            status_value.get('generation', {})
            .get('delta', {})
            .get('bytes', 0)
        )

    def paused_query_with_append(
        *,
        new_id: str,
        new_body: str,
        expected_page_growth: bool,
    ) -> dict[str, Any]:
        before_rows = lexical_search_ids(observer, 'prefixrace')
        before_status = lexical_status(observer)
        actor_ready = threading.Event()
        actor_pid: list[int] = []

        def query_actor() -> list[str]:
            connection = connect(socket_dir, port, autocommit=True)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SET ii42.test_convergent_root_snapshot_pause_ms = "
                        "'1000'"
                    )
                    cursor.execute('SELECT pg_backend_pid()')
                    actor_pid.append(int(cursor.fetchone()[0]))
                actor_ready.set()
                return lexical_search_ids(connection, 'prefixrace')
            finally:
                connection.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            query_future = executor.submit(query_actor)
            if not actor_ready.wait(timeout=10.0):
                raise TimeoutError('linked-L0 prefix reader did not start')
            wait_for_extension_pause(
                observer,
                actor_pid[0],
                query_future,
                'linked-L0 prefix reader',
            )
            with observer.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO docs VALUES (%s, %s)',
                    (new_id, new_body),
                )
            paused_rows = query_future.result(timeout=15.0)

        after_rows = lexical_search_ids(observer, 'prefixrace')
        after_status = lexical_status(observer)
        before_pages = frontier_page_count(before_status)
        after_pages = frontier_page_count(after_status)
        before_delta = before_status.get('generation', {}).get('delta', {})
        after_delta = after_status.get('generation', {}).get('delta', {})
        if paused_rows != before_rows:
            raise AssertionError(
                'old linked-L0 root exposed a physical suffix: '
                f'before={before_rows}, paused={paused_rows}'
            )
        if new_id in before_rows or new_id in paused_rows or new_id not in after_rows:
            raise AssertionError(
                'linked-L0 prefix visibility was not snapshot exact: '
                f'id={new_id}, before={before_rows}, '
                f'paused={paused_rows}, after={after_rows}'
            )
        if expected_page_growth != (after_pages > before_pages):
            raise AssertionError(
                'linked-L0 prefix fixture did not exercise the expected '
                f'physical transition: before_pages={before_pages}, '
                f'after_pages={after_pages}, '
                f'expected_growth={expected_page_growth}, '
                f'before_bytes={frontier_bytes(before_status)}, '
                f'after_bytes={frontier_bytes(after_status)}, '
                f'before_delta={json.dumps(before_delta, sort_keys=True)}, '
                f'after_delta={json.dumps(after_delta, sort_keys=True)}'
            )
        return {
            'new_id': new_id,
            'before_rows': before_rows,
            'paused_rows': paused_rows,
            'after_rows': after_rows,
            'before_frontier_pages': before_pages,
            'after_frontier_pages': after_pages,
            'before_frontier_bytes': frontier_bytes(before_status),
            'after_frontier_bytes': frontier_bytes(after_status),
            'record_bytes': len(new_body.encode('utf-8')),
            'passed': True,
        }

    try:
        with observer.cursor() as cursor:
            cursor.execute(
                'INSERT INTO docs VALUES (%s, %s)',
                ('prefix-anchor', 'prefixrace anchor sentinel'),
            )
        anchor_status = lexical_status(observer)
        seed_count = 0
        while active_page_count(anchor_status) == 0 and seed_count < 8:
            seed_count += 1
            with observer.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO docs VALUES (%s, %s)',
                    (
                        f'prefix-seed-{seed_count}',
                        f'prefixrace root seed {seed_count}',
                    ),
                )
            anchor_status = lexical_status(observer)
        if active_page_count(anchor_status) == 0:
            raise AssertionError(
                'linked-L0 prefix fixture could not seed a physical root'
            )
        inline = paused_query_with_append(
            new_id='prefix-inline',
            new_body='prefixrace inline sentinel',
            expected_page_growth=False,
        )
        linked = paused_query_with_append(
            new_id='prefix-linked',
            new_body=(
                'prefixrace linked sentinel ' +
                ' '.join(
                    f'linkedprefixterm{term_id}'
                    for term_id in range(2000)
                )
            ),
            expected_page_growth=True,
        )
        with observer.cursor() as cursor:
            cursor.execute(
                "DELETE FROM docs WHERE id LIKE 'prefix-%'"
            )
            cursor.execute('VACUUM docs')
        cleanup = maintain_index_until_clean(
            observer,
            index_name='docs_body_idx',
        )
        normal_rows = lexical_search_ids(observer, 'base stable sentinel')
        final_status = lexical_status(observer)
        passed = (
            inline['passed']
            and linked['passed']
            and normal_rows == baseline_rows
            and int(final_status['details']['delta_records']) == 0
        )
        evidence = {
            'anchor_frontier_pages': frontier_page_count(anchor_status),
            'root_seed_records': seed_count,
            'inline_suffix': inline,
            'linked_successor': linked,
            'cleanup': cleanup,
            'baseline_query_ids': baseline_rows,
            'normal_query_ids': normal_rows,
            'passed': passed,
        }
        if not passed:
            raise AssertionError(
                'linked-L0 logical-prefix gate failed: '
                f'{json.dumps(evidence, sort_keys=True)}'
            )
        return evidence
    finally:
        if maintenance_locked:
            with observer.cursor() as cursor:
                cursor.execute(
                    "SELECT ii42_index_maintenance_unlock("
                    "'docs_body_idx'::regclass)"
                )
        observer.close()


def active_generation_pages(status_value: dict[str, Any]) -> int:
    generation = status_value.get('generation', {})
    return (
        1
        + int(generation.get('primary', {}).get('pages', 0))
        + int(generation.get('posting', {}).get('pages', 0))
        + int(generation.get('delta', {}).get('pages', 0))
    )


def exercise_generation_staging_reuse(
    connection: psycopg.Connection[Any],
    *,
    cycles: int = 24,
) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []

    def capture(phase: str, cycle: int, result: str) -> None:
        status_value = fetch_json(
            connection,
            "SELECT ii42_index_status("
            "'model_docs_body_idx'::regclass)",
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_relation_size("
                "'model_docs_body_idx'::regclass)"
            )
            relation_bytes = int(cursor.fetchone()[0])
        snapshots.append({
            'cycle': cycle,
            'phase': phase,
            'maintain_result': result,
            'physical_pages': int(
                status_value.get('details', {}).get('pages', -1)
            ),
            'relation_bytes': relation_bytes,
            'active_generation_pages': active_generation_pages(status_value),
            'generation': int(
                status_value.get('generation', {}).get('generation', -1)
            ),
            'delta_records': int(
                status_value.get('details', {}).get('delta_records', -1)
            ),
            'storage': (
                status_value.get('generation', {})
                .get('layout', {})
                .get('storage')
            ),
        })

    for cycle in range(cycles):
        doc_id = f'staging-reuse-{cycle}'
        with connection.cursor() as cursor:
            cursor.execute(
                'INSERT INTO model_docs VALUES (%s, %s)',
                (
                    doc_id,
                    'staging replacement semantic generation',
                ),
            )
        insert_maintenance = maintain_model_until_clean(connection)
        insert_result = ' -> '.join(insert_maintenance['results'])
        capture('insert', cycle, insert_result)
        with connection.cursor() as cursor:
            cursor.execute(
                'DELETE FROM model_docs WHERE id = %s',
                (doc_id,),
            )
            cursor.execute('VACUUM model_docs')
        delete_maintenance = maintain_model_until_clean(connection)
        delete_result = ' -> '.join(delete_maintenance['results'])
        capture('delete', cycle, delete_result)

    physical_pages = [
        int(snapshot['physical_pages'])
        for snapshot in snapshots
    ]
    active_pages = [
        int(snapshot['active_generation_pages'])
        for snapshot in snapshots
    ]
    steady_offset = len(physical_pages) // 2
    warmup_pages = physical_pages[:steady_offset]
    steady_pages = physical_pages[steady_offset:]
    steady_bytes = [
        int(snapshot['relation_bytes'])
        for snapshot in snapshots[steady_offset:]
    ]
    footprint_limit = (2 * max(active_pages)) + 2
    plateau_limit = max(warmup_pages) + 2
    convergent_storage = all(
        snapshot['storage'] == 'convergent_segments'
        for snapshot in snapshots
    )
    if convergent_storage:
        footprint_passed = (
            len(set(steady_pages)) == 1
            and len(set(steady_bytes)) == 1
        )
        footprint_gate = 'linked_l0_high_watermark_plateau'
    else:
        footprint_passed = (
            max(physical_pages) <= footprint_limit
            and max(steady_pages) <= plateau_limit
        )
        footprint_gate = 'dual_generation_staging_slots'
    normal_rows = model_search_ids(
        connection,
        'semantic retrieval database',
    )
    oracle_rows = model_search_ids(
        connection,
        'semantic retrieval database',
        oracle=True,
    )
    passed = (
        len(snapshots) == cycles * 2
        and all(
            int(snapshot['delta_records']) == 0
            for snapshot in snapshots
        )
        and footprint_passed
        and normal_rows == oracle_rows
    )
    return {
        'cycles': cycles,
        'snapshots': snapshots,
        'footprint_limit_pages': footprint_limit,
        'plateau_limit_pages': plateau_limit,
        'footprint_gate': footprint_gate,
        'footprint_passed': footprint_passed,
        'maximum_physical_pages': max(physical_pages),
        'maximum_steady_pages': max(steady_pages),
        'normal_query_ids': normal_rows,
        'oracle_query_ids': oracle_rows,
        'passed': passed,
    }


def exercise_online_tail_handoff(
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    observer = connect(socket_dir, port, autocommit=True)
    initial_status = fetch_json(
        observer,
        "SELECT ii42_index_status("
        "'model_docs_body_idx'::regclass)",
    )
    if (
        initial_status.get('generation', {})
        .get('layout', {})
        .get('storage') == 'convergent_segments'
    ):
        observer.close()
        return {
            'applicable': False,
            'reason': 'linked_l0_replaces_v2_tail_handoff',
            'passed': True,
        }
    maintenance_ready = threading.Event()
    maintenance_run = threading.Event()
    maintenance_pid: list[int] = []

    def maintenance_actor() -> str:
        with connect(socket_dir, port, autocommit=True) as connection:
            lock_held = False
            results: list[str] = []
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SET ii42.test_online_maintenance_pause_ms = '1000'"
                    )
                    cursor.execute('SELECT pg_backend_pid()')
                    maintenance_pid.append(int(cursor.fetchone()[0]))
                    cursor.execute(
                        "SELECT ii42_index_try_maintenance_lock("
                        "'model_docs_body_idx'::regclass)"
                    )
                    lock_row = cursor.fetchone()
                    if lock_row is None or lock_row[0] is not True:
                        raise AssertionError(
                            'tail-handoff actor could not acquire its '
                            'maintenance lock'
                        )
                    lock_held = True
                    maintenance_ready.set()
                    if not maintenance_run.wait(timeout=15.0):
                        raise TimeoutError(
                            'tail-handoff actor was not released to run'
                        )
                    for _ in range(2):
                        cursor.execute(
                            "SELECT ii42_index_try_maintain("
                            "'model_docs_body_idx'::regclass)"
                        )
                        row = cursor.fetchone()
                        if row is None:
                            raise AssertionError(
                                'tail-handoff maintenance returned no row'
                            )
                        results.append(str(row[0]))
                        if 'reason=shared_unified_delta' not in results[-1]:
                            break
            finally:
                if lock_held:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT ii42_index_maintenance_unlock("
                            "'model_docs_body_idx'::regclass)"
                        )
        return ' -> '.join(results)

    def wait_for_test_pause(
        pid: int,
        maintenance_future: concurrent.futures.Future[str],
    ) -> None:
        deadline = time.monotonic() + 15.0
        while True:
            with observer.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_stat_activity
                        WHERE pid = %s
                          AND wait_event_type = 'Extension'
                    )
                    """,
                    (pid,),
                )
                found = bool(cursor.fetchone()[0])
            if found:
                return
            if maintenance_future.done():
                raise AssertionError(
                    'tail-handoff maintenance finished before the test '
                    f'pause: {maintenance_future.result()}'
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f'backend {pid} did not reach the maintenance test pause'
                )
            time.sleep(0.002)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            maintenance_future = executor.submit(maintenance_actor)
            if not maintenance_ready.wait(timeout=10.0):
                raise TimeoutError('tail-handoff maintenance did not start')
            with observer.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO model_docs VALUES (%s, %s)',
                    (
                        'tail-seed',
                        'seedhandoffunique maintenance snapshot delta',
                    ),
                )
            baseline = fetch_json(
                observer,
                "SELECT ii42_index_status("
                "'model_docs_body_idx'::regclass)",
            )
            if int(
                baseline.get(
                    'generation',
                    {},
                ).get('primary', {}).get('start_block', -1)
            ) != 1:
                raise AssertionError(
                    'tail-handoff audit requires a first-generation layout: '
                    f'{baseline}'
                )
            maintenance_run.set()
            wait_for_test_pause(
                maintenance_pid[0],
                maintenance_future,
            )
            with observer.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO model_docs VALUES (%s, %s)',
                    (
                        'tail-handoff',
                        'tailhandoffunique concurrent semantic delta',
                    ),
                )
            maintenance_result = maintenance_future.result(timeout=30.0)

        tail_status = fetch_json(
            observer,
            "SELECT ii42_index_status("
            "'model_docs_body_idx'::regclass)",
        )
        details = tail_status.get('details', {})
        normal_rows = model_search_ids(
            observer,
            'tailhandoffunique concurrent semantic delta',
        )
        oracle_rows = model_search_ids(
            observer,
            'tailhandoffunique concurrent semantic delta',
            oracle=True,
        )
        passed = (
            'tail_carried=true' in maintenance_result
            and 'tail_records=1' in maintenance_result
            and 'segment_reused=false' in maintenance_result
            and int(details.get('delta_records', -1)) == 1
            and int(details.get('pending_writes', -1)) == 1
            and int(details.get('pending_deletes', -1)) == 0
            and 'tail-handoff' in normal_rows
            and normal_rows == oracle_rows
        )
        evidence = {
            'maintenance_pid': maintenance_pid[0],
            'maintenance_result': maintenance_result,
            'tail_status': tail_status,
            'normal_query_ids': normal_rows,
            'oracle_query_ids': oracle_rows,
            'passed': passed,
        }
        if not passed:
            raise AssertionError(
                'online fallback tail handoff failed: '
                f'{json.dumps(evidence, sort_keys=True)}'
            )
        return evidence
    finally:
        maintenance_run.set()
        observer.close()


def verify_online_tail_handoff_after_restart(
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    with connect(socket_dir, port, autocommit=True) as connection:
        before = fetch_json(
            connection,
            "SELECT ii42_index_status("
            "'model_docs_body_idx'::regclass)",
        )
        normal_rows = model_search_ids(
            connection,
            'tailhandoffunique concurrent semantic delta',
        )
        oracle_rows = model_search_ids(
            connection,
            'tailhandoffunique concurrent semantic delta',
            oracle=True,
        )
        compact_maintenance = maintain_model_until_clean(connection)
        compacted = compact_maintenance['status']
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM model_docs "
                "WHERE id IN ('tail-seed', 'tail-handoff')"
            )
            cursor.execute('VACUUM model_docs')
        cleanup_maintenance = maintain_model_until_clean(connection)
        cleaned = cleanup_maintenance['status']
    before_details = before.get('details', {})
    compacted_details = compacted.get('details', {})
    cleaned_details = cleaned.get('details', {})
    passed = (
        int(before_details.get('delta_records', -1)) == 1
        and int(before_details.get('pending_writes', -1)) == 1
        and 'tail-handoff' in normal_rows
        and normal_rows == oracle_rows
        and int(compacted_details.get('delta_records', -1)) == 0
        and int(compacted_details.get('pending_writes', -1)) == 0
        and int(cleaned_details.get('delta_records', -1)) == 0
        and int(cleaned_details.get('pending_writes', -1)) == 0
        and int(cleaned.get('generation', {}).get('docs', -1)) == 4
    )
    evidence = {
        'before_compaction': before,
        'normal_query_ids': normal_rows,
        'oracle_query_ids': oracle_rows,
        'compact_results': compact_maintenance['results'],
        'compacted': compacted,
        'cleanup_results': cleanup_maintenance['results'],
        'cleaned': cleaned,
        'passed': passed,
    }
    if not passed:
        raise AssertionError(
            'online fallback tail restart failed: '
            f'{json.dumps(evidence, sort_keys=True)}'
        )
    return evidence


def search_ids(
    connection: psycopg.Connection[Any],
    query: str,
) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT docs.id
            FROM ii42_query('docs_body_idx'::regclass, %s, 10) AS hit
            JOIN docs ON docs.ctid = hit.ctid
            ORDER BY docs.id
            """,
            (query,),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def status(connection: psycopg.Connection[Any]) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT ii42_index_status('docs_body_idx'::regclass)")
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid ii42 status: {row}')
    return dict(row[0])


def cache_status(connection: psycopg.Connection[Any]) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT ii42_index_runtime_state_json("
            "'docs_body_idx'::regclass)"
        )
        row = cursor.fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError(f'invalid ii42 cache status: {row}')
    return dict(row[0])


def exercise_writer_protocol(
    socket_dir: Path,
    port: int,
    statement_timeout_ms: int,
) -> dict[str, Any]:
    writer_a = connect(socket_dir, port, autocommit=False)
    writer_b = connect(socket_dir, port, autocommit=False)
    observer = connect(socket_dir, port, autocommit=True)
    maintenance_result = ''
    maintenance_results: list[str] = []

    try:
        with writer_a.cursor() as cursor:
            cursor.execute(
                f"SET statement_timeout = '{statement_timeout_ms}ms'"
            )
            cursor.execute(
                "INSERT INTO docs VALUES "
                "('abort-a', 'writer abortaunique rollback sentinel')"
            )

        started_at = time.monotonic()
        with writer_b.cursor() as cursor:
            cursor.execute(
                f"SET statement_timeout = '{statement_timeout_ms}ms'"
            )
            cursor.execute(
                "INSERT INTO docs VALUES "
                "('commit-b', 'writer commitbunique committed sentinel')"
            )
            second_writer_ms = (time.monotonic() - started_at) * 1000.0
            cursor.execute('SAVEPOINT writer_b_savepoint')
            cursor.execute(
                "INSERT INTO docs VALUES "
                "('savepoint-b', 'writer savepointunique rollback sentinel')"
            )
            cursor.execute('ROLLBACK TO SAVEPOINT writer_b_savepoint')
            cursor.execute('RELEASE SAVEPOINT writer_b_savepoint')
        writer_b.commit()

        if second_writer_ms >= statement_timeout_ms * 0.75:
            raise AssertionError(
                'second same-index writer was transaction-serialized: '
                f'{second_writer_ms:.1f}ms'
            )

        with observer.cursor() as cursor:
            cursor.execute('SELECT id FROM docs ORDER BY id')
            visible_while_a_open = [str(row[0]) for row in cursor.fetchall()]
        if visible_while_a_open != ['base', 'commit-b']:
            raise AssertionError(
                'transaction visibility changed while writer A was open: '
                f'{visible_while_a_open}'
            )
        with observer.cursor() as cursor:
            cursor.execute(
                f"SET statement_timeout = '{statement_timeout_ms}ms'"
            )
        query_started_at = time.monotonic()
        visible_query_ids = search_ids(observer, 'stable')
        concurrent_query_ms = (
            time.monotonic() - query_started_at
        ) * 1000.0
        if sorted(visible_query_ids) != visible_while_a_open:
            raise AssertionError(
                'query visibility diverged from the heap while writer A '
                f'was open: query={visible_query_ids}, '
                f'heap={visible_while_a_open}'
            )
        if concurrent_query_ms >= statement_timeout_ms * 0.75:
            raise AssertionError(
                'query admission waited for an open same-index writer: '
                f'{concurrent_query_ms:.1f}ms'
            )
        status_while_a_open = status(observer)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            maintenance_future = executor.submit(maintain, socket_dir, port)
            time.sleep(0.25)
            maintenance_waited = not maintenance_future.done()
            writer_a.rollback()
            maintenance_result = maintenance_future.result(timeout=15)
            maintenance_results.append(maintenance_result)

        first_maintenance_status = status(observer)
        maintenance_deferred = (
            'maintained=false' in maintenance_result
            and (
                'reason=lock_busy' in maintenance_result
                or 'reason=xid_horizon' in maintenance_result
            )
        )
        maintenance_rotated_only = (
            'maintained=true' in maintenance_result
            and 'mode=segment_rotation' in maintenance_result
        )
        if (
            not maintenance_waited
            and not maintenance_deferred
            and not maintenance_rotated_only
        ):
            raise AssertionError(
                'maintenance crossed an active writer barrier: '
                f'{maintenance_result}'
            )
        if (
            maintenance_deferred
            and status_while_a_open['generation']['generation_id']
                != first_maintenance_status['generation']['generation_id']
        ):
            raise AssertionError(
                'deferred maintenance changed the immutable generation: '
                f'before={status_while_a_open}, '
                f'after={first_maintenance_status}'
            )
        deadline = time.monotonic() + 10.0
        while True:
            current_status = status(observer)
            details = current_status.get('details', {})
            if (
                int(details.get('pending_writes', -1)) == 0
                and int(details.get('pending_deletes', -1)) == 0
                and int(details.get('delta_records', -1)) == 0
            ):
                break
            if time.monotonic() >= deadline:
                raise AssertionError(
                    'writer debt did not converge: '
                    f'results={maintenance_results}, status={current_status}'
                )
            maintenance_result = maintain(socket_dir, port)
            maintenance_results.append(maintenance_result)
            time.sleep(0.05)

        details = current_status.get('details', {})
        maintenance_status = cache_status(observer).get('maintenance', {})
        if (
            int(details.get('pending_writes', -1)) != 0
            or int(details.get('pending_deletes', -1)) != 0
            or int(details.get('delta_records', -1)) != 0
        ):
            raise AssertionError(
                f'writer debt remained after maintenance: {current_status}'
            )
        required_hint_keys = {
            'work_hint_sequence',
            'work_hint_overflows',
            'work_hints_consumed',
            'pending_maintenance_hint_entries',
            'pending_preload_hint_entries',
            'maintenance_reconcile_age_ms',
            'preload_reconcile_age_ms',
            'maintenance_reconcile_count',
            'maintenance_reconcile_last_rows',
            'maintenance_reconcile_last_duration_us',
            'maintenance_reconcile_max_duration_us',
            'preload_reconcile_count',
            'preload_reconcile_last_rows',
            'preload_reconcile_last_duration_us',
            'preload_reconcile_max_duration_us',
        }
        if not required_hint_keys.issubset(maintenance_status):
            raise AssertionError(
                f'maintenance hint telemetry is incomplete: {current_status}'
            )

        committed = search_ids(observer, 'commitbunique')
        rolled_back = search_ids(observer, 'abortaunique')
        savepoint = search_ids(observer, 'savepointunique')
        if 'commit-b' not in committed:
            raise AssertionError(f'committed writer is missing: {committed}')
        if 'abort-a' in rolled_back:
            raise AssertionError(
                f'aborted writer remained searchable: {rolled_back}'
            )
        if 'savepoint-b' in savepoint:
            raise AssertionError(
                f'rolled-back savepoint remained searchable: {savepoint}'
            )
        if int(current_status.get('generation', {}).get('docs', -1)) != 2:
            raise AssertionError(
                'aborted writer rows changed the rebuilt document count: '
                f'{current_status}'
            )

        return {
            'second_writer_ms': round(second_writer_ms, 3),
            'concurrent_query_ms': round(concurrent_query_ms, 3),
            'visible_query_ids_while_writer_a_open': visible_query_ids,
            'maintenance_waited_for_writer': maintenance_waited,
            'maintenance_deferred_for_writer': maintenance_deferred,
            'maintenance_rotated_only': maintenance_rotated_only,
            'maintenance_results': maintenance_results,
            'visible_while_writer_a_open': visible_while_a_open,
            'generation_while_writer_a_open': (
                status_while_a_open['generation']['generation_id']
            ),
            'generation_after_first_maintenance': (
                first_maintenance_status['generation']['generation_id']
            ),
            'committed_query_ids': committed,
            'rollback_query_ids': rolled_back,
            'savepoint_query_ids': savepoint,
            'final_docs': int(
                current_status.get('generation', {}).get('docs', -1)
            ),
            'maintenance': maintenance_status,
        }
    finally:
        writer_a.close()
        writer_b.close()
        observer.close()


def exercise_model_eventual_concurrency(
    socket_dir: Path,
    port: int,
    *,
    readers: int,
    writers: int,
    cycles: int,
    staging_cycles: int,
) -> dict[str, Any]:
    if (readers <= 0 or writers <= 0 or cycles <= 0 or
        staging_cycles <= 0):
        raise ValueError('model concurrency dimensions must be positive')

    stop_event = threading.Event()
    start_barrier = threading.Barrier(readers + writers + 2, timeout=30.0)

    def configure(
        connection: psycopg.Connection[Any],
        application_name: str,
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = '30s'")
            cursor.execute("SET lock_timeout = '10s'")
            cursor.execute(
                "SELECT set_config('application_name', %s, false)",
                (application_name,),
            )

    def summarize_latency_ms(samples: list[float]) -> dict[str, float | int]:
        if not samples:
            return {'count': 0}
        ordered = sorted(samples)

        def percentile(fraction: float) -> float:
            position = max(
                0,
                min(len(ordered) - 1, int(len(ordered) * fraction)),
            )
            return ordered[position]

        return {
            'count': len(ordered),
            'mean': sum(ordered) / len(ordered),
            'p50': percentile(0.50),
            'p95': percentile(0.95),
            'p99': percentile(0.99),
            'max': ordered[-1],
        }

    def measure_query_latency(
        connection: psycopg.Connection[Any],
        *,
        samples: int = 12,
    ) -> dict[str, float | int]:
        for _ in range(2):
            model_search_ids(connection, 'semantic retrieval database')
        latencies_ms: list[float] = []
        for _ in range(samples):
            started = time.perf_counter()
            rows = model_search_ids(
                connection,
                'semantic retrieval database',
            )
            latencies_ms.append(
                (time.perf_counter() - started) * 1000.0
            )
            if not rows:
                raise AssertionError('latency probe returned no rows')
        return summarize_latency_ms(latencies_ms)

    def measure_parallel_query_latency(
        *,
        actor_count: int,
        samples: int = 12,
    ) -> list[dict[str, float | int]]:
        barrier = threading.Barrier(actor_count, timeout=30.0)

        def actor(actor_id: int) -> dict[str, float | int]:
            with connect(socket_dir, port, autocommit=True) as connection:
                configure(connection, f'ii42-control-reader-{actor_id}')
                for _ in range(2):
                    model_search_ids(
                        connection,
                        'semantic retrieval database',
                    )
                barrier.wait()
                latencies_ms: list[float] = []
                for _ in range(samples):
                    started = time.perf_counter()
                    rows = model_search_ids(
                        connection,
                        'semantic retrieval database',
                    )
                    latencies_ms.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    if not rows:
                        raise AssertionError(
                            'parallel latency probe returned no rows'
                        )
                return summarize_latency_ms(latencies_ms)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=actor_count
        ) as executor:
            return list(executor.map(actor, range(actor_count)))

    with connect(socket_dir, port, autocommit=True) as connection:
        configure(connection, 'ii42-baseline-latency')
        baseline_query_latency_ms = measure_query_latency(connection)
    control_reader_latency_ms = measure_parallel_query_latency(
        actor_count=readers
    )

    def reader_actor(reader_id: int) -> dict[str, Any]:
        queries = 0
        latencies_ms: list[float] = []
        try:
            with connect(socket_dir, port, autocommit=True) as connection:
                configure(connection, f'ii42-reader-{reader_id}')
                start_barrier.wait()
                while not stop_event.is_set():
                    started = time.perf_counter()
                    rows = model_search_ids(
                        connection,
                        'semantic retrieval database',
                    )
                    latencies_ms.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    if not rows:
                        raise AssertionError('model reader returned no rows')
                    queries += 1
            return {
                'actor': f'reader-{reader_id}',
                'ok': True,
                'queries': queries,
                'latency_ms': summarize_latency_ms(latencies_ms),
            }
        except Exception as error:
            stop_event.set()
            start_barrier.abort()
            return {
                'actor': f'reader-{reader_id}',
                'ok': False,
                'queries': queries,
                'latency_ms': summarize_latency_ms(latencies_ms),
                'error': str(error),
            }

    def writer_actor(writer_id: int) -> dict[str, Any]:
        operations = 0

        def require_visibility(
            connection: psycopg.Connection[Any],
            *,
            doc_id: str,
            query: str,
            expected: bool,
            phase: str,
        ) -> None:
            normal_ids = model_search_ids(connection, query)
            if (doc_id in normal_ids) is expected:
                return
            oracle_ids = model_search_ids(
                connection,
                query,
                oracle=True,
            )
            status = fetch_json(
                connection,
                "SELECT ii42_index_status("
                "'model_docs_body_idx'::regclass)",
            )
            cache = fetch_json(
                connection,
                "SELECT ii42_index_runtime_state_json("
                "'model_docs_body_idx'::regclass)",
            )
            raise AssertionError(json.dumps({
                'phase': phase,
                'doc_id': doc_id,
                'expected': expected,
                'normal_visible': doc_id in normal_ids,
                'oracle_visible': doc_id in oracle_ids,
                'normal_ids': normal_ids,
                'oracle_ids': oracle_ids,
                'status': status,
                'cache': cache,
            }, sort_keys=True))

        try:
            with connect(socket_dir, port, autocommit=True) as connection:
                configure(connection, f'ii42-writer-{writer_id}')
                start_barrier.wait()
                for cycle in range(cycles):
                    doc_id = f'model-writer-{writer_id}-{cycle}'
                    insert_query = 'inserted semantic retrieval'
                    update_query = 'updated database retrieval'
                    with connection.cursor() as cursor:
                        cursor.execute(
                            'INSERT INTO model_docs VALUES (%s, %s)',
                            (
                                doc_id,
                                f'{insert_query} writer {writer_id} '
                                f'cycle {cycle}',
                            ),
                        )
                    operations += 1
                    require_visibility(
                        connection,
                        doc_id=doc_id,
                        query=insert_query,
                        expected=True,
                        phase='insert',
                    )

                    with connection.cursor() as cursor:
                        cursor.execute(
                            'UPDATE model_docs SET body = %s WHERE id = %s',
                            (
                                f'{update_query} writer {writer_id} '
                                f'cycle {cycle}',
                                doc_id,
                            ),
                        )
                    operations += 1
                    require_visibility(
                        connection,
                        doc_id=doc_id,
                        query=update_query,
                        expected=True,
                        phase='update',
                    )

                    with connection.cursor() as cursor:
                        cursor.execute(
                            'DELETE FROM model_docs WHERE id = %s',
                            (doc_id,),
                        )
                    operations += 1
                    require_visibility(
                        connection,
                        doc_id=doc_id,
                        query=update_query,
                        expected=False,
                        phase='delete',
                    )
            return {
                'actor': f'writer-{writer_id}',
                'ok': True,
                'operations': operations,
            }
        except Exception as error:
            stop_event.set()
            start_barrier.abort()
            return {
                'actor': f'writer-{writer_id}',
                'ok': False,
                'operations': operations,
                'error': str(error),
            }

    def maintenance_actor() -> dict[str, Any]:
        calls = 0
        results: dict[str, int] = {}
        try:
            with connect(socket_dir, port, autocommit=True) as connection:
                configure(connection, 'ii42-maintenance')
                start_barrier.wait()
                while not stop_event.is_set():
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT ii42_index_try_maintain("
                            "'model_docs_body_idx'::regclass)"
                        )
                        row = cursor.fetchone()
                    result = str(row[0]) if row is not None else 'missing'
                    reason = result.split('reason=', 1)[-1].split(
                        ',',
                        1,
                    )[0].rstrip(')')
                    results[reason] = results.get(reason, 0) + 1
                    calls += 1
                    time.sleep(0.005)
            return {
                'actor': 'maintenance',
                'ok': True,
                'calls': calls,
                'reasons': results,
            }
        except Exception as error:
            stop_event.set()
            start_barrier.abort()
            return {
                'actor': 'maintenance',
                'ok': False,
                'calls': calls,
                'reasons': results,
                'error': str(error),
            }

    def vacuum_actor() -> dict[str, Any]:
        calls = 0
        try:
            with connect(socket_dir, port, autocommit=True) as connection:
                configure(connection, 'evoke-vacuum')
                start_barrier.wait()
                while not stop_event.is_set():
                    with connection.cursor() as cursor:
                        cursor.execute('VACUUM model_docs')
                    calls += 1
                    time.sleep(0.005)
            return {
                'actor': 'vacuum',
                'ok': True,
                'calls': calls,
            }
        except Exception as error:
            stop_event.set()
            start_barrier.abort()
            return {
                'actor': 'vacuum',
                'ok': False,
                'calls': calls,
                'error': str(error),
            }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=readers + writers + 2
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
        vacuum_future = executor.submit(vacuum_actor)
        writer_results = [future.result() for future in writer_futures]
        stop_event.set()
        reader_results = [future.result() for future in reader_futures]
        maintenance_result = maintenance_future.result()
        vacuum_result = vacuum_future.result()

    actors = reader_results + writer_results + [
        maintenance_result,
        vacuum_result,
    ]
    if not all(bool(actor.get('ok')) for actor in actors):
        raise AssertionError(
            'model-backed eventual concurrency actor failed: '
            f'{json.dumps(actors, sort_keys=True)}'
        )

    with connect(socket_dir, port, autocommit=True) as connection:
        configure(connection, 'ii42-finalizer')
        with connection.cursor() as cursor:
            cursor.execute('VACUUM model_docs')
            cursor.execute('SELECT count(*) FROM model_docs')
            final_rows = int(cursor.fetchone()[0])
        final_maintenance = maintain_model_until_clean(connection)
        staging_reuse = exercise_generation_staging_reuse(
            connection,
            cycles=staging_cycles,
        )
        final_status = fetch_json(
            connection,
            "SELECT ii42_index_status("
            "'model_docs_body_idx'::regclass)",
        )
        runtime_status = fetch_json(
            connection,
            'SELECT ii42_runtime_service_status()',
        )
        normal_rows = model_search_ids(
            connection,
            'semantic retrieval database',
        )
        oracle_rows = model_search_ids(
            connection,
            'semantic retrieval database',
            oracle=True,
        )
        final_query_latency_ms = measure_query_latency(connection)

    expected_operations = writers * cycles * 3
    observed_operations = sum(
        int(result.get('operations', 0))
        for result in writer_results
    )
    generation = final_status.get('generation', {})
    details = final_status.get('details', {})
    generation_docs = int(generation.get('docs', -1))
    generation_docs_scope = generation.get('docs_scope')
    document_slot_high_watermark = int(
        generation.get('document_slot_high_watermark', -1)
    )
    if generation_docs_scope == 'sealed_generation':
        generation_docs_passed = (
            generation_docs >= final_rows
            and generation_docs <= document_slot_high_watermark
        )
    else:
        generation_docs_passed = generation_docs == final_rows
    passed = (
        all(bool(actor.get('ok')) for actor in actors)
        and observed_operations == expected_operations
        and sum(
            int(actor.get('queries', 0))
            for actor in reader_results
        ) > 0
        and int(maintenance_result.get('calls', 0)) > 0
        and int(vacuum_result.get('calls', 0)) > 0
        and final_rows == 4
        and staging_reuse['passed']
        and generation_docs_passed
        and int(details.get('pending_writes', -1)) == 0
        and int(details.get('pending_deletes', -1)) == 0
        and int(details.get('delta_records', -1)) == 0
        and final_status.get('query_ready') is True
        and normal_rows == oracle_rows
        and runtime_status.get('worker_ready') is True
        and int(runtime_status.get('failures', -1)) == 0
        and int(runtime_status.get('busy_rejections', -1)) == 0
    )
    evidence = {
        'readers': readers,
        'writers': writers,
        'cycles_per_writer': cycles,
        'actors': actors,
        'baseline_query_latency_ms': baseline_query_latency_ms,
        'control_reader_latency_ms': control_reader_latency_ms,
        'active_reader_latency_ms': [
            result['latency_ms'] for result in reader_results
        ],
        'final_query_latency_ms': final_query_latency_ms,
        'expected_operations': expected_operations,
        'observed_operations': observed_operations,
        'final_rows': final_rows,
        'generation_docs': generation_docs,
        'generation_docs_scope': generation_docs_scope,
        'generation_docs_passed': generation_docs_passed,
        'document_slot_high_watermark': document_slot_high_watermark,
        'final_maintenance': final_maintenance['results'],
        'generation_staging_reuse': staging_reuse,
        'final_status': final_status,
        'normal_query_ids': normal_rows,
        'oracle_query_ids': oracle_rows,
        'runtime_status': runtime_status,
        'passed': passed,
    }
    if not passed:
        raise AssertionError(
            'model-backed eventual concurrency failed: '
            f'{json.dumps(evidence, sort_keys=True)}'
        )
    return evidence


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
        args.extension_libdir = args.extension_libdir.expanduser().resolve()
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir.expanduser().resolve()
        )
    pg_bin = args.pg_bin.expanduser().resolve()
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'
    for executable in (initdb, pg_ctl, pg_bin / 'postgres'):
        if not executable.is_file():
            raise FileNotFoundError(f'missing PostgreSQL executable: {executable}')

    socket_root = create_short_socket_root('ii42-writer-')
    try:
        with tempfile.TemporaryDirectory(prefix='ii42-writer-data-') as temp:
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
                configure_cluster(
                    data_dir,
                    socket_dir,
                    port,
                    args.extension_libdir,
                    args.extension_control_dir,
                )
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
                with connect(socket_dir, port, autocommit=True) as connection:
                    setup(connection)
                evidence = exercise_writer_protocol(
                    socket_dir,
                    port,
                    args.statement_timeout_ms,
                )
                evidence['shared_retirement_sequence_guard'] = (
                    exercise_shared_retirement_sequence_guard(
                        socket_dir,
                        port,
                    )
                )
                evidence['linked_l0_logical_prefix'] = (
                    exercise_linked_l0_logical_prefix(
                        socket_dir,
                        port,
                    )
                )
                evidence['retired_page_reader_fence'] = (
                    exercise_retired_page_reader_fence(
                        socket_dir,
                        port,
                        pg_ctl=pg_ctl,
                        data_dir=data_dir,
                        log_path=log_path,
                    )
                )
                if args.model_path is not None:
                    model_path = args.model_path.expanduser().resolve()
                    if not model_path.is_dir():
                        raise FileNotFoundError(
                            f'missing model checkout: {model_path}'
                        )
                    with connect(
                        socket_dir,
                        port,
                        autocommit=True,
                    ) as connection:
                        setup_model_eventual(connection, model_path)
                    tail_handoff = exercise_online_tail_handoff(
                        socket_dir,
                        port,
                    )
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
                    tail_handoff['restart_and_compaction'] = (
                        verify_online_tail_handoff_after_restart(
                            socket_dir,
                            port,
                        )
                        if tail_handoff.get('applicable', True)
                        else {
                            'applicable': False,
                            'reason': (
                                'linked_l0_replaces_v2_tail_handoff'
                            ),
                            'passed': True,
                        }
                    )
                    tail_handoff['passed'] = (
                        bool(tail_handoff['passed'])
                        and bool(
                            tail_handoff[
                                'restart_and_compaction'
                            ]['passed']
                        )
                    )
                    evidence['online_tail_handoff'] = tail_handoff
                    evidence['vacuum_maintenance_authority'] = (
                        exercise_vacuum_maintenance_authority(
                            socket_dir,
                            port,
                        )
                    )
                    evidence['model_eventual_concurrency'] = (
                        exercise_model_eventual_concurrency(
                            socket_dir,
                            port,
                            readers=args.model_readers,
                            writers=args.model_writers,
                            cycles=args.model_cycles,
                            staging_cycles=args.staging_cycles,
                        )
                    )
                    log_lines = log_path.read_text(
                        encoding='utf-8'
                    ).splitlines()
                    reader_lock_waits = [
                        line for line in log_lines
                        if (
                            'ii42-reader-' in line or
                            'ii42-control-reader-' in line
                        ) and (
                            'still waiting for' in line or
                            'acquired ' in line and ' after ' in line
                        )
                    ]
                    evidence['model_eventual_concurrency'][
                        'reader_lock_wait_log'
                    ] = reader_lock_waits
                    if reader_lock_waits:
                        raise AssertionError(
                            'public query waited on a PostgreSQL lock: '
                            f'{json.dumps(reader_lock_waits)}'
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

    output = json.dumps(evidence, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f'{output}\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
