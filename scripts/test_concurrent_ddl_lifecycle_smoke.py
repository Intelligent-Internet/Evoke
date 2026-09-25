#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import psycopg

from evoke_test_support import extension_control_root


QUERY = 'rare zebra quantum flux capacitor semantic retrieval'
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Exercise concurrent and rewrite lifecycle operations for BM25 '
            'and semantic-enabled evoke indexes.'
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
    parser.add_argument('--output', type=Path)
    parser.add_argument('--keep', action='store_true')
    return parser.parse_args()


def run(
    command: list[str | Path],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def index_status(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_status(%s::regclass)',
            (index_name,),
        )
        value = cursor.fetchone()[0]
    if not isinstance(value, dict):
        raise AssertionError(f'invalid index status for {index_name}: {value}')
    return value


def assert_ready(
    connection: psycopg.Connection[Any],
    index_name: str,
    expected_type: str,
) -> dict[str, Any]:
    status = index_status(connection, index_name)
    if status.get('query_ready') is not True:
        raise AssertionError(f'{index_name} is not query-ready: {status}')
    if status.get('index_type') != expected_type:
        raise AssertionError(
            f'{index_name} type is {status.get("index_type")}, '
            f'expected {expected_type}'
        )
    generation = status.get('generation')
    if not isinstance(generation, dict) or generation.get('valid') is not True:
        raise AssertionError(f'{index_name} generation is invalid: {status}')
    return status


def search_ids(
    connection: psycopg.Connection[Any],
    index_name: str,
    query_text: str = QUERY,
) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT source.id
            FROM evoke_query(%s::regclass, %s, 20) AS hit
            JOIN ddl_lifecycle.docs AS source ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            ''',
            (index_name, query_text),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def assert_visible(
    connection: psycopg.Connection[Any],
    index_name: str,
    doc_id: str,
) -> None:
    hits = search_ids(connection, index_name)
    if doc_id not in hits:
        raise AssertionError(f'{doc_id} is not visible through {index_name}: {hits}')


def run_concurrent_reindex(conninfo: str, index_name: str) -> None:
    with psycopg.connect(conninfo, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET deadlock_timeout = '250ms'")
            cursor.execute(
                f'REINDEX INDEX CONCURRENTLY {index_name}'
            )


def main() -> int:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    if not (args.model_path / 'manifest.json').is_file():
        raise FileNotFoundError(
            f'model manifest is missing: {args.model_path / "manifest.json"}'
        )
    manifest = json.loads(
        (args.model_path / 'manifest.json').read_text(encoding='utf-8')
    )
    if (
        manifest.get('schema_version') != 1
        or manifest.get('api_version') != 'evoke_model_v1'
        or manifest.get('runtime_abi') != 'evoke_p2_unified_text_atoms_v2'
    ):
        raise ValueError(
            '--model-path must use the current Evoke model contract'
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

    root = Path(tempfile.mkdtemp(prefix='evoke_concurrent_ddl_'))
    data_dir = root / 'data'
    socket_dir = root / 'socket'
    log_path = root / 'postgres.log'
    socket_dir.mkdir()
    port = free_port()
    pg_ctl = args.pg_bin / 'pg_ctl'
    started = False
    report: dict[str, Any] = {
        'api_version': 'evoke_index_v1',
        'route': 'concurrent_ddl_and_relation_rewrite',
        'model_path': str(args.model_path),
        'gates': [],
    }

    try:
        run(
            [
                args.pg_bin / 'initdb',
                '-D',
                data_dir,
                '-A',
                'trust',
                '-U',
                'postgres',
            ]
        )
        with (data_dir / 'postgresql.conf').open('a', encoding='utf-8') as file:
            file.write("\nshared_preload_libraries = 'evoke'\n")
            if args.extension_libdir is not None:
                libdir = str(args.extension_libdir).replace("'", "''")
                file.write(
                    "dynamic_library_path = '"
                    f'{libdir}:$libdir'
                    "'\n"
                )
            if args.extension_control_dir is not None:
                control_dir = str(args.extension_control_dir).replace(
                    "'",
                    "''",
                )
                file.write(
                    "extension_control_path = '"
                    f'{control_dir}:$system'
                    "'\n"
                )
            file.write("listen_addresses = ''\n")
            file.write(f"unix_socket_directories = '{socket_dir}'\n")
            file.write(f'port = {port}\n')
            file.write('max_worker_processes = 16\n')
            file.write("evoke.shared_runtime_size = '256MB'\n")
        run([pg_ctl, '-D', data_dir, '-l', log_path, 'start', '-w'])
        started = True

        connection = psycopg.connect(
            host=str(socket_dir),
            port=port,
            dbname='postgres',
            user='postgres',
            autocommit=True,
        )
        conninfo = connection.info.dsn
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f'''
                    CREATE EXTENSION evoke;
                    CREATE SCHEMA ddl_lifecycle;
                    CREATE TABLE ddl_lifecycle.docs (
                        id text PRIMARY KEY,
                        body text NOT NULL
                    );
                    INSERT INTO ddl_lifecycle.docs VALUES
                        ('target', {sql_literal(QUERY)}),
                        ('medical', 'cardiovascular clinical evidence'),
                        ('database', 'postgresql index maintenance recovery'),
                        ('space', 'astronomy galaxy telescope observation');
                    ''',
                )
                cursor.execute(
                    '''
                    CREATE INDEX CONCURRENTLY docs_bm25_idx
                    ON ddl_lifecycle.docs USING evoke (body)
                    WITH (sae = false, consistency = realtime)
                    ''',
                )
                cursor.execute(
                    f'''
                    CREATE INDEX CONCURRENTLY docs_semantic_idx
                    ON ddl_lifecycle.docs USING evoke (body)
                    WITH (
                        sae = true,
                        model_path = {sql_literal(str(args.model_path))}
                    )
                    ''',
                )

            for name, expected_type in (
                ('ddl_lifecycle.docs_bm25_idx', 'bm25'),
                ('ddl_lifecycle.docs_semantic_idx', 'semantic'),
            ):
                assert_ready(connection, name, expected_type)
                assert_visible(connection, name, 'target')
            report['gates'].append('create_index_concurrently')

            with connection.cursor() as cursor:
                cursor.execute(
                    '''
                    CREATE TABLE ddl_lifecycle.concurrent_docs AS
                    SELECT
                        format('bulk-%s', item)::text AS id,
                        repeat(
                            format(
                                'concurrent lifecycle token %s ',
                                item
                            ),
                            4
                        )::text AS body
                    FROM generate_series(1, 40000) AS item;
                    ALTER TABLE ddl_lifecycle.concurrent_docs
                    ADD PRIMARY KEY (id);
                    CREATE INDEX concurrent_docs_idx
                    ON ddl_lifecycle.concurrent_docs USING evoke (body)
                    WITH (sae = false, consistency = realtime);
                    ''',
                )

            reindex_started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            ) as executor:
                reindex = executor.submit(
                    run_concurrent_reindex,
                    conninfo,
                    'ddl_lifecycle.concurrent_docs_idx',
                )
                deadline = time.monotonic() + 60.0
                while True:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            '''
                            SELECT phase
                            FROM pg_stat_progress_create_index
                            WHERE command = 'REINDEX CONCURRENTLY'
                              AND relid =
                                  'ddl_lifecycle.concurrent_docs'::regclass
                            ''',
                        )
                        progress = cursor.fetchone()
                    if progress is not None and progress[0] == 'building index':
                        break
                    if reindex.done():
                        reindex.result()
                        raise AssertionError(
                            'concurrent reindex completed before writer probe'
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            'concurrent reindex did not enter build phase'
                        )
                    time.sleep(0.01)

                with psycopg.connect(conninfo) as writer:
                    with writer.cursor() as cursor:
                        cursor.execute("SET deadlock_timeout = '250ms'")
                        cursor.execute(
                            '''
                            LOCK TABLE ddl_lifecycle.concurrent_docs
                            IN ROW EXCLUSIVE MODE
                            ''',
                        )

                        while True:
                            with connection.cursor() as observer:
                                observer.execute(
                                    '''
                                    SELECT new_index.indisready
                                    FROM pg_class AS relation
                                    JOIN pg_namespace AS namespace
                                      ON namespace.oid = relation.relnamespace
                                    JOIN pg_index AS new_index
                                      ON new_index.indexrelid = relation.oid
                                    WHERE namespace.nspname = 'ddl_lifecycle'
                                      AND relation.relname LIKE
                                          'concurrent_docs_idx_ccnew%'
                                    ''',
                                )
                                ready = observer.fetchone()
                            if ready is not None and ready[0] is True:
                                break
                            if reindex.done():
                                reindex.result()
                                raise AssertionError(
                                    'concurrent reindex did not expose a '
                                    'ready transient index'
                                )
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    'transient reindex did not become ready'
                                )
                            time.sleep(0.01)

                        cursor.execute(
                            '''
                            INSERT INTO ddl_lifecycle.concurrent_docs
                            VALUES (
                                'concurrent-writer',
                                'rare concurrent writer visibility token'
                            )
                            ''',
                        )
                    writer.commit()

                reindex.result(timeout=60.0)

            with connection.cursor() as cursor:
                cursor.execute(
                    '''
                    SELECT source.id
                    FROM evoke_query(
                        'ddl_lifecycle.concurrent_docs_idx'::regclass,
                        'rare concurrent writer visibility token',
                        20
                    ) AS hit
                    JOIN ddl_lifecycle.concurrent_docs AS source
                      ON source.ctid = hit.ctid
                    WHERE source.id = 'concurrent-writer'
                    ''',
                )
                if cursor.fetchone() is None:
                    raise AssertionError(
                        'writer committed during concurrent reindex is '
                        'not query-visible'
                    )
                cursor.execute(
                    'DROP TABLE ddl_lifecycle.concurrent_docs'
                )
            report['concurrent_reindex_writer_seconds'] = round(
                time.monotonic() - reindex_started,
                6,
            )
            report['gates'].append('reindex_concurrently_with_writer')

            with connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO ddl_lifecycle.docs VALUES (%s, %s)',
                    ('inserted', QUERY + ' inserted'),
                )
                cursor.execute(
                    'UPDATE ddl_lifecycle.docs SET body = %s WHERE id = %s',
                    (QUERY + ' updated', 'space'),
                )
                cursor.execute(
                    'DELETE FROM ddl_lifecycle.docs WHERE id = %s',
                    ('target',),
                )
            for name in (
                'ddl_lifecycle.docs_bm25_idx',
                'ddl_lifecycle.docs_semantic_idx',
            ):
                assert_visible(connection, name, 'inserted')
                assert_visible(connection, name, 'space')
                if 'target' in search_ids(connection, name):
                    raise AssertionError(f'deleted row remains visible: {name}')
            report['gates'].append('crud_before_reindex')

            with connection.cursor() as cursor:
                cursor.execute(
                    'REINDEX INDEX CONCURRENTLY ddl_lifecycle.docs_bm25_idx'
                )
                cursor.execute(
                    'REINDEX INDEX CONCURRENTLY ddl_lifecycle.docs_semantic_idx'
                )
            assert_ready(
                connection,
                'ddl_lifecycle.docs_bm25_idx',
                'bm25',
            )
            assert_ready(
                connection,
                'ddl_lifecycle.docs_semantic_idx',
                'semantic',
            )
            report['gates'].append('reindex_concurrently')

            with connection.cursor() as cursor:
                cursor.execute('VACUUM FULL ddl_lifecycle.docs')
            for name, expected_type in (
                ('ddl_lifecycle.docs_bm25_idx', 'bm25'),
                ('ddl_lifecycle.docs_semantic_idx', 'semantic'),
            ):
                assert_ready(connection, name, expected_type)
                assert_visible(connection, name, 'inserted')
            report['gates'].append('vacuum_full_relation_rewrite')

            with connection.cursor() as cursor:
                cursor.execute('TRUNCATE ddl_lifecycle.docs')
            for name, expected_type in (
                ('ddl_lifecycle.docs_bm25_idx', 'bm25'),
                ('ddl_lifecycle.docs_semantic_idx', 'semantic'),
            ):
                assert_ready(connection, name, expected_type)
                if search_ids(connection, name):
                    raise AssertionError(f'{name} is not empty after TRUNCATE')
            report['gates'].append('truncate_to_empty_generation')

            with connection.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO ddl_lifecycle.docs VALUES (%s, %s)',
                    ('post-truncate', QUERY),
                )
            for name in (
                'ddl_lifecycle.docs_bm25_idx',
                'ddl_lifecycle.docs_semantic_idx',
            ):
                assert_visible(connection, name, 'post-truncate')
            report['gates'].append('post_truncate_lexical_visibility')

            with connection.cursor() as cursor:
                cursor.execute(
                    'DROP INDEX CONCURRENTLY ddl_lifecycle.docs_bm25_idx'
                )
                cursor.execute(
                    'DROP INDEX CONCURRENTLY ddl_lifecycle.docs_semantic_idx'
                )
                cursor.execute(
                    '''
                    SELECT count(*)
                    FROM pg_catalog.pg_class AS relation
                    JOIN pg_catalog.pg_namespace AS namespace
                      ON namespace.oid = relation.relnamespace
                    JOIN pg_catalog.pg_am AS access
                      ON access.oid = relation.relam
                    WHERE namespace.nspname = 'ddl_lifecycle'
                      AND access.amname = 'evoke'
                    ''',
                )
                remaining = int(cursor.fetchone()[0])
            if remaining != 0:
                raise AssertionError(
                    f'{remaining} evoke indexes remain after concurrent drop'
                )
            report['gates'].append('drop_index_concurrently')

        report['passed'] = True
        report['gate_count'] = len(report['gates'])
    finally:
        if started:
            run(
                [pg_ctl, '-D', data_dir, 'stop', '-m', 'fast'],
                check=False,
            )
        if args.keep:
            report['kept_cluster'] = str(root)
        else:
            shutil.rmtree(root)

    output = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    print(output, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
