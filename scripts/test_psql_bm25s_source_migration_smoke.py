#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import psycopg

from evoke_test_support import (
    REPO_ROOT,
    create_short_socket_root,
    ensure_temp_root,
    extension_control_root,
)


Hit = tuple[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Verify side-by-side source-table migration from psql_bm25s '
            'to the current evoke index.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument(
        '--source-package-root',
        type=Path,
        required=True,
        help=(
            'DESTDIR-style, internally consistent psql_bm25s package used '
            'as the migration source.'
        ),
    )
    parser.add_argument(
        '--extension-libdir',
        type=Path,
        help=(
            'Directory containing the staged current evoke shared library. '
            'Must be supplied with --extension-control-dir.'
        ),
    )
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'Staged PostgreSQL share or extension directory containing '
            'evoke.control. Must be supplied with --extension-libdir.'
        ),
    )
    parser.add_argument(
        '--temp-root',
        type=Path,
        default=Path('/tmp'),
        help=(
            'Root for temporary cluster data and logs. PostgreSQL sockets '
            'use a separate short-lived path so long roots remain valid.'
        ),
    )
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def run(command: list[str], *, check: bool = True) -> str:
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
    return result.stdout


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def append_config(path: Path, lines: list[str]) -> None:
    content = path.read_text(encoding='utf-8')
    if not content.endswith('\n'):
        content += '\n'
    path.write_text(content + '\n'.join(lines) + '\n', encoding='utf-8')


def pg_config_value(pg_bin: Path, option: str) -> Path:
    value = run([str(pg_bin / 'pg_config'), option]).strip()
    if not value:
        raise RuntimeError(f'pg_config returned no value for {option}')
    return Path(value)


def staged_path(package_root: Path, install_path: Path) -> Path:
    if not install_path.is_absolute():
        raise ValueError(f'install path is not absolute: {install_path}')
    return package_root / install_path.relative_to(install_path.anchor)


def old_hits(conn: psycopg.Connection[Any], query: str) -> list[Hit]:
    rows = conn.execute(
        '''
        SELECT docs.id, hit.score
        FROM public.psql_bm25s_search(
            'docs_old_idx'::regclass,
            %s,
            100
        ) AS hit
        JOIN docs ON docs.ctid = hit.ctid
        ORDER BY hit.score DESC, docs.id
        ''',
        (query,),
    ).fetchall()
    return [(int(row[0]), float(row[1])) for row in rows]


def new_hits(conn: psycopg.Connection[Any], query: str) -> list[Hit]:
    rows = conn.execute(
        '''
        SELECT docs.id, hit.score
        FROM evoke_ext.evoke_query(
            'docs_new_idx'::regclass,
            %s,
            100
        ) AS hit
        JOIN docs ON docs.ctid = hit.ctid
        ORDER BY hit.score DESC, docs.id
        ''',
        (query,),
    ).fetchall()
    return [(int(row[0]), float(row[1])) for row in rows]


def assert_hits_match(old: list[Hit], new: list[Hit]) -> None:
    old_ids = [doc_id for doc_id, _score in old]
    new_ids = [doc_id for doc_id, _score in new]
    if old_ids != new_ids:
        raise AssertionError(
            f'old/new result identities differ: {old_ids} != {new_ids}'
        )
    for (old_id, old_score), (new_id, new_score) in zip(
        old,
        new,
        strict=True,
    ):
        if old_id != new_id or not math.isclose(
            old_score,
            new_score,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                'old/new BM25 scores differ: '
                f'{old_id}={old_score}, {new_id}={new_score}'
            )


def relation_exists(
    conn: psycopg.Connection[Any],
    relation_name: str,
) -> bool:
    row = conn.execute(
        'SELECT to_regclass(%s)',
        (relation_name,),
    ).fetchone()
    return row is not None and row[0] is not None


def require_current_index(
    conn: psycopg.Connection[Any],
) -> dict[str, Any]:
    row = conn.execute(
        "SELECT evoke_ext.evoke_index_status('docs_new_idx'::regclass)"
    ).fetchone()
    if row is None or not isinstance(row[0], dict):
        raise AssertionError('current evoke index status is unavailable')
    status = row[0]
    generation = status.get('generation')
    if (
        status.get('query_ready') is not True
        or status.get('index_valid') is not True
        or status.get('index_ready') is not True
        or status.get('index_type') != 'bm25'
        or not isinstance(generation, dict)
        or generation.get('atomic') is not True
        or generation.get('valid') is not True
    ):
        raise AssertionError(f'current evoke index is not ready: {status}')
    return status


def current_index_is_converged(
    status: dict[str, Any],
    expected_docs: int,
) -> bool:
    generation = status.get('generation')
    details = status.get('details')
    if not isinstance(generation, dict) or not isinstance(details, dict):
        return False
    delta = generation.get('delta')
    if not isinstance(delta, dict):
        return False
    return (
        generation.get('docs_scope') == 'sealed_generation'
        and int(generation.get('docs', -1)) == expected_docs
        and int(generation.get('sealed_docs', -1)) == expected_docs
        and int(delta.get('records', -1)) == 0
        and int(details.get('pending_writes', -1)) == 0
        and int(details.get('pending_deletes', -1)) == 0
        and int(details.get('delta_records', -1)) == 0
    )


def require_converged_current_index(
    conn: psycopg.Connection[Any],
    expected_docs: int,
) -> dict[str, Any]:
    status = require_current_index(conn)
    if not current_index_is_converged(status, expected_docs):
        raise AssertionError(
            f'current evoke index is not converged: {status}'
        )
    return status


def maintain_current_index_until_converged(
    conn: psycopg.Connection[Any],
    expected_docs: int,
    max_attempts: int = 16,
) -> tuple[dict[str, Any], list[str]]:
    phases: list[str] = []
    for _attempt in range(max_attempts):
        status = require_current_index(conn)
        if current_index_is_converged(status, expected_docs):
            return status, phases
        row = conn.execute(
            "SELECT evoke_ext.evoke_index_maintain("
            "'docs_new_idx'::regclass)"
        ).fetchone()
        phases.append(str(row[0]) if row is not None else '<no result>')

    status = require_current_index(conn)
    raise AssertionError(
        'current evoke index did not converge after '
        f'{max_attempts} maintenance phases: phases={phases}, '
        f'status={status}'
    )


def extension_version(
    conn: psycopg.Connection[Any],
    extension_name: str,
) -> str:
    row = conn.execute(
        'SELECT extversion FROM pg_extension WHERE extname = %s',
        (extension_name,),
    ).fetchone()
    if row is None:
        raise AssertionError(f'extension is not installed: {extension_name}')
    return str(row[0])


def assert_source_rows(
    conn: psycopg.Connection[Any],
    expected: list[tuple[int, str]],
) -> None:
    rows = [
        (int(row[0]), str(row[1]))
        for row in conn.execute(
            'SELECT id, body FROM docs ORDER BY id'
        ).fetchall()
    ]
    if rows != expected:
        raise AssertionError(f'source rows changed unexpectedly: {rows}')


def main() -> None:
    args = parse_args()
    if (args.extension_libdir is None) != (
        args.extension_control_dir is None
    ):
        raise ValueError(
            '--extension-libdir and --extension-control-dir must be '
            'provided together'
        )
    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    role = os.environ.get('USER') or 'postgres'
    source_package_root = args.source_package_root.expanduser().resolve()
    source_libdir = staged_path(
        source_package_root,
        pg_config_value(args.pg_bin, '--pkglibdir'),
    )
    source_sharedir = staged_path(
        source_package_root,
        pg_config_value(args.pg_bin, '--sharedir'),
    )
    source_control = (
        source_sharedir / 'extension' / 'psql_bm25s.control'
    )
    source_library = source_libdir / (
        'psql_bm25s.dylib'
        if os.uname().sysname == 'Darwin'
        else 'psql_bm25s.so'
    )
    if not source_control.is_file() or not source_library.is_file():
        raise FileNotFoundError(
            'source psql_bm25s package is incomplete: '
            f'{source_package_root}'
        )
    target_libdir: Path | None = None
    target_sharedir: Path | None = None
    if args.extension_libdir is not None:
        target_libdir = args.extension_libdir.expanduser().resolve()
        target_sharedir = extension_control_root(
            args.extension_control_dir
        )
        target_libraries = [
            target_libdir / name for name in ('evoke.so', 'evoke.dylib')
        ]
        if not any(path.is_file() for path in target_libraries):
            raise FileNotFoundError(
                f'current evoke library is missing from {target_libdir}'
            )
        if not (
            target_sharedir / 'extension' / 'evoke.control'
        ).is_file():
            raise FileNotFoundError(
                'current evoke control file is missing below '
                f'{target_sharedir}'
            )
    escaped_source_libdir = str(source_libdir).replace("'", "''")
    escaped_source_sharedir = str(source_sharedir).replace("'", "''")
    dynamic_library_path = f'{escaped_source_libdir}:$libdir'
    extension_control_path = f'{escaped_source_sharedir}:$system'
    target_binding = 'installed'
    if target_libdir is not None and target_sharedir is not None:
        escaped_target_libdir = str(target_libdir).replace("'", "''")
        escaped_target_sharedir = str(target_sharedir).replace("'", "''")
        dynamic_library_path = (
            f'{escaped_target_libdir}:{escaped_source_libdir}:$libdir'
        )
        extension_control_path = (
            f'{escaped_target_sharedir}:{escaped_source_sharedir}:$system'
        )
        target_binding = 'staged'
    temp_root = ensure_temp_root(args.temp_root)
    root = Path(
        tempfile.mkdtemp(
            prefix='evoke_source_migration_',
            dir=temp_root,
        )
    )
    socket_root = create_short_socket_root('evoke_source_migration_socket_')
    data_dir = root / 'data'
    socket_dir = socket_root / 's'
    log_path = root / 'postgres.log'
    socket_dir.mkdir()
    port = free_port()
    dsn = (
        f'dbname=postgres user={role} host={socket_dir} port={port}'
    )
    started = False
    summary: dict[str, Any] = {
        'api_version': 'evoke_index_v1',
        'suite': 'psql_bm25s_source_table_migration',
        'checks': {},
        'passed': False,
    }
    base_rows = [
        (1, 'alpha semantic migration target'),
        (2, 'beta lexical migration target'),
        (3, 'gamma source table remains online'),
        (4, 'unrelated document'),
    ]
    final_rows = [
        (1, 'alpha semantic migration target'),
        (2, 'beta lexical target updated'),
        (4, 'unrelated document'),
        (5, 'delta migration target'),
    ]
    query = 'semantic migration target'
    updated_query = 'lexical target updated'

    try:
        run(
            [
                str(initdb),
                '-D',
                str(data_dir),
                '-U',
                role,
                '-A',
                'trust',
                '--no-locale',
                '--encoding=UTF8',
            ]
        )
        append_config(
            data_dir / 'postgresql.conf',
            [
                "shared_preload_libraries = 'evoke'",
                (
                    "dynamic_library_path = '"
                    f"{dynamic_library_path}'"
                ),
                (
                    "extension_control_path = '"
                    f"{extension_control_path}'"
                ),
                "listen_addresses = ''",
                f"unix_socket_directories = '{socket_dir}'",
                f'port = {port}',
            ],
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

        with psycopg.connect(dsn, autocommit=True) as conn:
            available = conn.execute(
                '''
                SELECT name::text
                FROM pg_available_extensions
                WHERE name IN ('evoke', 'psql_bm25s')
                ORDER BY name
                '''
            ).fetchall()
            available_names = [str(row[0]) for row in available]
            if available_names != [
                'evoke',
                'psql_bm25s',
            ]:
                raise RuntimeError(
                    'both evoke and psql_bm25s extension packages are '
                    'required for this migration smoke; found '
                    f'{available_names}'
                )
            conn.execute(
                'CREATE EXTENSION psql_bm25s WITH SCHEMA public'
            )
            conn.execute('CREATE SCHEMA evoke_ext')
            conn.execute(
                'CREATE EXTENSION evoke WITH SCHEMA evoke_ext'
            )
            old_version = extension_version(conn, 'psql_bm25s')
            new_version = extension_version(conn, 'evoke')

            conn.execute(
                '''
                CREATE TABLE docs (
                    id int PRIMARY KEY,
                    body text NOT NULL
                )
                '''
            )
            with conn.cursor() as cursor:
                cursor.executemany(
                    'INSERT INTO docs (id, body) VALUES (%s, %s)',
                    base_rows,
                )
            conn.execute(
                '''
                CREATE INDEX docs_old_idx
                ON docs USING psql_bm25s (body)
                '''
            )
            old_before = old_hits(conn, query)

            conn.execute(
                '''
                CREATE INDEX CONCURRENTLY docs_new_idx
                ON docs USING evoke (body)
                WITH (sae = false)
                '''
            )
            require_converged_current_index(conn, expected_docs=4)
            new_before = new_hits(conn, query)
            assert_hits_match(old_before, new_before)
            assert_source_rows(conn, base_rows)

            conn.execute('DROP INDEX docs_new_idx')
            if (
                relation_exists(conn, 'docs_new_idx')
                or old_hits(conn, query) != old_before
            ):
                raise AssertionError(
                    'pre-cutover rollback changed the old serving index'
                )
            assert_source_rows(conn, base_rows)

            conn.execute(
                '''
                CREATE INDEX CONCURRENTLY docs_new_idx
                ON docs USING evoke (body)
                WITH (sae = false)
                '''
            )
            require_converged_current_index(conn, expected_docs=4)
            assert_hits_match(old_before, new_hits(conn, query))

            conn.execute(
                "INSERT INTO docs VALUES (5, 'delta migration target')"
            )
            conn.execute(
                '''
                UPDATE docs
                SET body = 'beta lexical target updated'
                WHERE id = 2
                '''
            )
            conn.execute('DELETE FROM docs WHERE id = 3')
            conn.execute('VACUUM docs')
            conn.execute(
                "SELECT public.psql_bm25s_maintain_index("
                "'docs_old_idx'::regclass)"
            )
            assert_hits_match(
                old_hits(conn, updated_query),
                new_hits(conn, updated_query),
            )
            _status, maintenance_phases = (
                maintain_current_index_until_converged(
                    conn,
                    expected_docs=4,
                )
            )
            assert_hits_match(
                old_hits(conn, updated_query),
                new_hits(conn, updated_query),
            )
            assert_source_rows(conn, final_rows)

            current_cutover_hits = new_hits(conn, updated_query)
            conn.execute('DROP INDEX docs_old_idx')
            conn.execute('DROP EXTENSION psql_bm25s')
            if relation_exists(conn, 'docs_old_idx'):
                raise AssertionError('old index remains after cutover')
            if extension_version(conn, 'evoke') != new_version:
                raise AssertionError(
                    'dropping psql_bm25s changed the evoke extension'
                )
            require_converged_current_index(conn, expected_docs=4)
            if new_hits(conn, updated_query) != current_cutover_hits:
                raise AssertionError(
                    'current query changed after removing psql_bm25s'
                )
            assert_source_rows(conn, final_rows)

            summary['checks'] = {
                'current_index_ready': True,
                'dual_index_crud_parity': True,
                'old_extension_version': old_version,
                'bounded_maintenance_phases': maintenance_phases,
                'pre_cutover_rollback': True,
                'side_by_side_score_parity': True,
                'source_rows_preserved': True,
                'target_extension_version': new_version,
                'target_package_binding': target_binding,
            }
            summary['passed'] = True

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + '\n',
                encoding='utf-8',
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception:
        if log_path.exists():
            print(log_path.read_text(encoding='utf-8'))
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
                ],
                check=False,
            )
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(socket_root, ignore_errors=True)


if __name__ == '__main__':
    main()
