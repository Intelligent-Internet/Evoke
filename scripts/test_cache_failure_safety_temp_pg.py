#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import psycopg

from evoke_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parents[1]
RSS_GROWTH_LIMIT_KIB = 32 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Exercise page-native query, L0, operator, scan, and failure '
            'cleanup in one surviving backend.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument('--iterations', type=int, default=32)
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share directory containing extension/evoke.control, '
            'or the extension directory itself.'
        ),
    )
    return parser.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


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
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        result.check_returncode()
    return result


def configure_cluster(
    data_dir: Path,
    port: int,
    extension_libdir: Path | None,
    extension_control_dir: Path | None,
) -> None:
    config = data_dir / 'postgresql.conf'
    with config.open('a', encoding='utf-8') as handle:
        handle.write("\nshared_preload_libraries = ''\n")
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
        handle.write("dynamic_shared_memory_type = 'mmap'\n")
        handle.write("listen_addresses = ''\n")
        handle.write(f"unix_socket_directories = '{data_dir}'\n")
        handle.write(f'port = {port}\n')


def connect(
    data_dir: Path,
    port: int,
) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname='postgres',
        user=os.environ.get('USER', 'postgres'),
        host=str(data_dir),
        port=port,
        autocommit=True,
    )


def backend_rss_kib(pid: int) -> int:
    result = run(['ps', '-o', 'rss=', '-p', str(pid)])
    return int(result.stdout.strip())


def query_count(connection: psycopg.Connection[Any]) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT count(*)
            FROM evoke_query_tokens(
                'failure_test.docs_idx'::regclass,
                ARRAY['failure', 'safety'],
                20,
                NULL
            )
            '''
        )
        return int(cursor.fetchone()[0])


def scan_count(connection: psycopg.Connection[Any]) -> int:
    with connection.cursor() as cursor:
        cursor.execute('SET enable_seqscan = off')
        cursor.execute(
            '''
            SELECT count(*)
            FROM failure_test.docs
            WHERE tokens @@ 'failure'
            '''
        )
        return int(cursor.fetchone()[0])


def set_fault(
    connection: psycopg.Connection[Any],
    setting: str,
    enabled: bool,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT set_config(%s, %s, false)',
            (setting, 'on' if enabled else 'off'),
        )


def clear_cache(connection: psycopg.Connection[Any]) -> None:
    with connection.cursor() as cursor:
        cursor.execute('SELECT evoke_runtime_cache_clear()')


def expect_fault(
    connection: psycopg.Connection[Any],
    setting: str,
    expected: str,
) -> None:
    set_fault(connection, setting, True)
    try:
        query_count(connection)
    except psycopg.Error as error:
        if expected not in str(error):
            raise AssertionError(
                f'unexpected {setting} error: {error}'
            ) from error
    else:
        raise AssertionError(f'{setting} did not fail')
    finally:
        set_fault(connection, setting, False)


def expect_scan_fault(connection: psycopg.Connection[Any]) -> None:
    setting = 'evoke.test_search_error_after_rank'

    set_fault(connection, setting, True)
    try:
        scan_count(connection)
    except psycopg.Error as error:
        expected = (
            'injected evoke page-native search result construction error'
        )
        if expected not in str(error):
            raise AssertionError(
                f'unexpected scan cleanup error: {error}'
            ) from error
    else:
        raise AssertionError('index scan cleanup fault did not fail')
    finally:
        set_fault(connection, setting, False)


def expect_query_operator_fault(
    connection: psycopg.Connection[Any],
    statement: str,
) -> None:
    setting = 'evoke.test_query_operator_error_after_parse'

    set_fault(connection, setting, True)
    try:
        with connection.cursor() as cursor:
            cursor.execute(statement)
    except psycopg.Error as error:
        expected = 'injected evoke query operator after parse error'
        if expected not in str(error):
            raise AssertionError(
                f'unexpected query operator cleanup error: {error}'
            ) from error
    else:
        raise AssertionError('query operator cleanup fault did not fail')
    finally:
        set_fault(connection, setting, False)


def query_operator_statements() -> tuple[str, ...]:
    return (
        '''
        SELECT evoke_op_match_query_tokens(
            ARRAY['failure', 'safety']::text[],
            'failure AND safety'
        )
        ''',
        '''
        SELECT evoke_match_prepared_query(
            ARRAY['failure', 'safety']::text[],
            evoke_prepared_query(
                'failure_test.docs_idx'::regclass,
                'failure AND safety'
            )
        )
        ''',
        '''
        SELECT evoke_highlight(
            ARRAY['failure', 'safety']::text[],
            'failure AND safety'
        )
        ''',
        '''
        SELECT evoke_snippet(
            ARRAY['failure', 'safety']::text[],
            'failure AND safety',
            8
        )
        ''',
    )


def assert_query_operators_retry(
    connection: psycopg.Connection[Any],
    statements: tuple[str, ...],
) -> None:
    with connection.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
            value = cursor.fetchone()[0]
            if value is None:
                raise AssertionError(
                    'query operator retry returned a null result'
                )


def setup(connection: psycopg.Connection[Any]) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            CREATE EXTENSION evoke;
            CREATE SCHEMA failure_test;
            CREATE TABLE failure_test.docs (
                id integer PRIMARY KEY,
                tokens text[] NOT NULL
            );
            INSERT INTO failure_test.docs
            SELECT
                value,
                ARRAY[
                    'failure',
                    'safety',
                    'document',
                    value::text
                ]
            FROM generate_series(1, 5000) AS value;
            CREATE INDEX docs_idx
            ON failure_test.docs
            USING evoke (tokens)
            WITH (
                consistency = 'eventual'
            );
            '''
        )


def run_failure_matrix(
    connection: psycopg.Connection[Any],
    iterations: int,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_backend_pid()')
        backend_pid = int(cursor.fetchone()[0])

    if query_count(connection) != 20:
        raise AssertionError('baseline query did not return 20 hits')
    clear_cache(connection)
    rss_before = backend_rss_kib(backend_pid)

    operator_statements = query_operator_statements()
    for _ in range(iterations):
        for statement in operator_statements:
            expect_query_operator_fault(connection, statement)
    assert_query_operators_retry(connection, operator_statements)

    for _ in range(iterations):
        expect_fault(
            connection,
            'evoke.test_search_error_after_rank',
            'injected evoke search result construction error',
        )
    if query_count(connection) != 20:
        raise AssertionError('search cleanup retry did not return 20 hits')

    for _ in range(iterations):
        expect_scan_fault(connection)
    if scan_count(connection) != 5000:
        raise AssertionError('index scan cleanup retry returned wrong count')

    with connection.cursor() as cursor:
        cursor.execute(
            '''
            INSERT INTO failure_test.docs
            SELECT
                value,
                ARRAY['failure', 'safety', 'delta', value::text]
            FROM generate_series(5001, 5020) AS value
            '''
        )

    clear_cache(connection)
    for _ in range(iterations):
        expect_fault(
            connection,
            'evoke.test_search_error_after_rank',
            'injected evoke search result construction error',
        )
        clear_cache(connection)

    if query_count(connection) != 20:
        raise AssertionError('L0 query retry did not return 20 hits')
    if scan_count(connection) != 5020:
        raise AssertionError('L0 scan retry returned wrong count')
    rss_after = backend_rss_kib(backend_pid)
    if rss_after - rss_before > RSS_GROWTH_LIMIT_KIB:
        raise AssertionError(
            'same-backend failure loop retained too much memory: '
            f'before={rss_before}KiB after={rss_after}KiB '
            f'growth={rss_after - rss_before}KiB'
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
    workdir = Path(tempfile.mkdtemp(prefix='evoke_cache_failure_', dir='/tmp'))
    data_dir = workdir / 'pgdata'
    port = free_port()
    pg_ctl = args.pg_bin / 'pg_ctl'

    try:
        run([
            str(args.pg_bin / 'initdb'),
            '-D',
            str(data_dir),
            '-A',
            'trust',
            '-U',
            os.environ.get('USER', 'postgres'),
        ])
        configure_cluster(
            data_dir,
            port,
            args.extension_libdir,
            args.extension_control_dir,
        )
        run([
            str(pg_ctl),
            '-D',
            str(data_dir),
            '-l',
            str(data_dir / 'postgres.log'),
            '-w',
            'start',
        ])
        with connect(data_dir, port) as connection:
            setup(connection)
            run_failure_matrix(
                connection,
                args.iterations,
            )
        print('cache failure-safety smoke passed')
    finally:
        run(
            [
                str(pg_ctl),
                '-D',
                str(data_dir),
                '-m',
                'fast',
                '-w',
                'stop',
            ],
            check=False,
        )
        if args.keep:
            print(f'kept temporary cluster at {workdir}')
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == '__main__':
    main()
