#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
from pathlib import Path

from evoke_test_support import extension_control_root
from test_convergent_segment_read_smoke import (
    configure_cluster,
    pg_config_value,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')
BLCKSZ = 8192
PAGE_HEADER_SIZE = 24
META_STORAGE_VERSION_OFFSET = 72
STORAGE_VERSION_OFFSET = PAGE_HEADER_SIZE + META_STORAGE_VERSION_OFFSET
STORAGE_GENERATION_DELTA = 2
STORAGE_CONVERGENT_SEGMENTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Prove that retired generation-delta storage fails closed and '
            'explicit REINDEX restores page-native v3 storage.'
        ),
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--json-output', type=Path)
    return parser.parse_args()


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        return int(listener.getsockname()[1])


def run(
    command: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
    )


def psql_command(pg_bin: Path, socket_dir: Path, port: int) -> list[str]:
    return [
        str(pg_bin / 'psql'),
        '-X',
        '-Atq',
        '-h',
        str(socket_dir),
        '-p',
        str(port),
        '-d',
        'postgres',
        '-v',
        'ON_ERROR_STOP=1',
        '-v',
        'VERBOSITY=verbose',
    ]


def psql(
    pg_bin: Path,
    socket_dir: Path,
    port: int,
    sql: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run(
        psql_command(pg_bin, socket_dir, port),
        input_text=sql,
        check=check,
    )


def initialize_cluster(
    pg_bin: Path,
    pgdata: Path,
    socket_dir: Path,
    port: int,
    *,
    extension_libdir: Path,
    system_libdir: Path,
    extension_control_dir: Path,
    system_sharedir: Path,
) -> None:
    run([
        str(pg_bin / 'initdb'),
        '-D',
        str(pgdata),
        '-A',
        'trust',
        '--no-data-checksums',
        '-U',
        os.environ.get('USER', 'postgres'),
    ])
    configure_cluster(
        pgdata,
        socket_dir,
        port,
        extension_libdir=extension_libdir,
        system_libdir=system_libdir,
        extension_control_dir=extension_control_dir,
        system_sharedir=system_sharedir,
    )


def start_cluster(pg_bin: Path, pgdata: Path) -> None:
    run([
        str(pg_bin / 'pg_ctl'),
        '-D',
        str(pgdata),
        '-l',
        str(pgdata / 'postgres.log'),
        '-w',
        'start',
    ])


def stop_cluster(pg_bin: Path, pgdata: Path) -> None:
    subprocess.run(
        [
            str(pg_bin / 'pg_ctl'),
            '-D',
            str(pgdata),
            '-m',
            'fast',
            '-w',
            'stop',
        ],
        text=True,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )


def storage_version(index_path: Path) -> int:
    with index_path.open('rb') as index_file:
        page = index_file.read(BLCKSZ)
    if len(page) != BLCKSZ:
        raise AssertionError(
            f'evoke metapage is short: {len(page)} bytes at {index_path}'
        )
    return int(struct.unpack_from('<I', page, STORAGE_VERSION_OFFSET)[0])


def patch_storage_version(index_path: Path, version: int) -> None:
    with index_path.open('r+b') as index_file:
        page = bytearray(index_file.read(BLCKSZ))
        if len(page) != BLCKSZ:
            raise AssertionError(
                f'evoke metapage is short: {len(page)} bytes at {index_path}'
            )
        struct.pack_into('<I', page, STORAGE_VERSION_OFFSET, version)
        index_file.seek(0)
        index_file.write(page)
        index_file.flush()
        os.fsync(index_file.fileno())


def assert_storage_error(result: subprocess.CompletedProcess[str]) -> None:
    if result.returncode == 0:
        raise AssertionError(
            f'unsupported storage operation unexpectedly succeeded: '
            f'{result.stdout}'
        )
    expected = (
        '0A000',
        'unsupported evoke index storage layout',
        'REINDEX the evoke index to publish page-native v3 storage.',
    )
    missing = [part for part in expected if part not in result.stderr]
    if missing:
        raise AssertionError(
            f'unsupported storage error is missing {missing}: '
            f'{result.stderr}'
        )


def assert_rejected(
    pg_bin: Path,
    socket_dir: Path,
    port: int,
    name: str,
    sql: str,
    rejected: list[str],
) -> None:
    result = psql(pg_bin, socket_dir, port, sql, check=False)
    assert_storage_error(result)
    rejected.append(name)


def relation_path(
    pg_bin: Path,
    socket_dir: Path,
    port: int,
    pgdata: Path,
) -> Path:
    result = psql(
        pg_bin,
        socket_dir,
        port,
        "SELECT pg_relation_filepath('docs_bm25_idx'::regclass);",
    )
    relative = result.stdout.strip().splitlines()[-1]
    return pgdata / relative


def run_boundary(
    pg_bin: Path,
    pgdata: Path,
    socket_dir: Path,
    port: int,
) -> dict[str, object]:
    setup = psql(
        pg_bin,
        socket_dir,
        port,
        '''
        CREATE EXTENSION evoke;
        CREATE TABLE docs (
            id integer PRIMARY KEY,
            tokens text[] NOT NULL
        );
        INSERT INTO docs
        SELECT id, ARRAY['storage', 'boundary', id::text]
        FROM generate_series(1, 128) AS id;
        CREATE INDEX docs_bm25_idx
            ON docs USING evoke (tokens)
            WITH (consistency = 'eventual');
        CHECKPOINT;
        ''',
    )
    if setup.returncode != 0:
        raise AssertionError(setup.stderr)

    initial_path = relation_path(
        pg_bin,
        socket_dir,
        port,
        pgdata,
    )
    stop_cluster(pg_bin, pgdata)
    initial_version = storage_version(initial_path)
    if initial_version != STORAGE_CONVERGENT_SEGMENTS:
        raise AssertionError(
            f'fresh index storage version is {initial_version}, expected 3'
        )
    patch_storage_version(initial_path, STORAGE_GENERATION_DELTA)
    if storage_version(initial_path) != STORAGE_GENERATION_DELTA:
        raise AssertionError('failed to install retired storage fixture')
    start_cluster(pg_bin, pgdata)

    rejected: list[str] = []
    checks = {
        'query': '''
            SELECT count(*)
            FROM evoke_query_tokens(
                'docs_bm25_idx'::regclass,
                ARRAY['storage'],
                10,
                NULL
            );
        ''',
        'insert': '''
            INSERT INTO docs VALUES (129, ARRAY['storage', 'new']);
        ''',
        'vacuum': 'VACUUM docs;',
        'maintenance': '''
            SELECT evoke_index_try_maintain('docs_bm25_idx'::regclass);
        ''',
        'generation_status': '''
            SELECT evoke_index_generation_status_internal(
                'docs_bm25_idx'::regclass
            );
        ''',
        'cache_state': '''
            SELECT evoke_index_runtime_state(
                'docs_bm25_idx'::regclass
            );
        ''',
        'preload': '''
            SELECT evoke_index_preload(
                'docs_bm25_idx'::regclass
            );
        ''',
        'details': '''
            SELECT count(*)
            FROM evoke_index_details('docs_bm25_idx'::regclass);
        ''',
        'policy': '''
            SELECT count(*)
            FROM evoke_index_policy_recommend(
                'docs_bm25_idx'::regclass,
                'balanced'
            );
        ''',
        'generation_signature': '''
            SELECT evoke_index_generation_signature_internal(
                'docs_bm25_idx'::regclass
            );
        ''',
        'shared_resident': '''
            SELECT evoke_index_shared_preload_resident(
                'docs_bm25_idx'::regclass
            );
        ''',
    }
    for name, sql in checks.items():
        assert_rejected(
            pg_bin,
            socket_dir,
            port,
            name,
            sql,
            rejected,
        )

    runtime_signature = psql(
        pg_bin,
        socket_dir,
        port,
        '''
        SELECT evoke_index_runtime_signature_internal(
            'docs_bm25_idx'::regclass
        );
        ''',
    ).stdout.strip()
    if not runtime_signature:
        raise AssertionError(
            'runtime contract signature is unavailable for retired storage'
        )

    due_count = psql(
        pg_bin,
        socket_dir,
        port,
        'SELECT count(*) FROM evoke_index_maintain_due(16);',
    ).stdout.strip()
    if due_count != '0':
        raise AssertionError(
            f'background selector accepted retired storage: {due_count}'
        )

    psql(
        pg_bin,
        socket_dir,
        port,
        'REINDEX INDEX docs_bm25_idx; CHECKPOINT;',
    )
    recovered_path = relation_path(
        pg_bin,
        socket_dir,
        port,
        pgdata,
    )
    stop_cluster(pg_bin, pgdata)
    recovered_version = storage_version(recovered_path)
    if recovered_version != STORAGE_CONVERGENT_SEGMENTS:
        raise AssertionError(
            f'REINDEX published storage version {recovered_version}, '
            'expected 3'
        )
    start_cluster(pg_bin, pgdata)

    recovered_hits = psql(
        pg_bin,
        socket_dir,
        port,
        '''
        SELECT count(*)
        FROM evoke_query_tokens(
            'docs_bm25_idx'::regclass,
            ARRAY['storage'],
            10,
            NULL
        );
        ''',
    ).stdout.strip()
    if recovered_hits != '10':
        raise AssertionError(
            f'query did not recover after REINDEX: {recovered_hits}'
        )
    psql(
        pg_bin,
        socket_dir,
        port,
        "INSERT INTO docs VALUES (129, ARRAY['storage', 'recovered']);",
    )
    recovered_row = psql(
        pg_bin,
        socket_dir,
        port,
        'SELECT count(*) FROM docs WHERE id = 129;',
    ).stdout.strip()
    if recovered_row != '1':
        raise AssertionError('CRUD did not recover after REINDEX')

    return {
        'passed': True,
        'initial_storage_version': initial_version,
        'fixture_storage_version': STORAGE_GENERATION_DELTA,
        'recovered_storage_version': recovered_version,
        'rejected_boundaries': rejected,
        'runtime_signature': runtime_signature,
        'background_due_count': int(due_count),
        'recovered_hits': int(recovered_hits),
        'recovered_insert_count': int(recovered_row),
        'storage_version_offset': STORAGE_VERSION_OFFSET,
    }


def main() -> int:
    args = parse_args()
    pg_bin = args.pg_bin.expanduser().resolve()
    for executable in ('initdb', 'pg_ctl', 'psql'):
        path = pg_bin / executable
        if not path.is_file():
            raise FileNotFoundError(f'missing PostgreSQL executable: {path}')

    system_libdir = pg_config_value(pg_bin, '--pkglibdir')
    system_sharedir = pg_config_value(pg_bin, '--sharedir')
    extension_libdir = (
        args.extension_libdir.expanduser().resolve()
        if args.extension_libdir is not None
        else system_libdir
    )
    extension_control_dir = extension_control_root(
        args.extension_control_dir
        if args.extension_control_dir is not None
        else system_sharedir
    )

    work_root = Path(tempfile.mkdtemp(
        prefix='evoke-v3-storage-boundary-',
        dir='/tmp',
    ))
    pgdata = work_root / 'pgdata'
    socket_dir = work_root / 'socket'
    socket_dir.mkdir()
    port = reserve_port()
    result: dict[str, object] | None = None

    try:
        initialize_cluster(
            pg_bin,
            pgdata,
            socket_dir,
            port,
            extension_libdir=extension_libdir,
            system_libdir=system_libdir,
            extension_control_dir=extension_control_dir,
            system_sharedir=system_sharedir,
        )
        start_cluster(pg_bin, pgdata)
        result = run_boundary(pg_bin, pgdata, socket_dir, port)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + '\n',
                encoding='utf-8',
            )
        return 0
    finally:
        stop_cluster(pg_bin, pgdata)
        if args.keep:
            print(f'kept temporary cluster at {work_root}')
        else:
            shutil.rmtree(work_root, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
