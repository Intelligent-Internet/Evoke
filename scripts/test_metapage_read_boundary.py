#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

from evoke_test_support import extension_control_root
from test_convergent_segment_read_smoke import pg_config_value
from test_storage_layout_boundary import (
    BLCKSZ,
    DEFAULT_PG_BIN,
    initialize_cluster,
    psql,
    reserve_port,
    start_cluster,
    stop_cluster,
)


PAGE_HEADER_SIZE = 24
META_MAGIC_OFFSET = PAGE_HEADER_SIZE
META_TID_BYTES_OFFSET = PAGE_HEADER_SIZE + 24
META_ROOT_OFFSET = PAGE_HEADER_SIZE + 153


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate checked page-native v3 metapage reads.',
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--json-output', type=Path)
    return parser.parse_args()


def relation_path(
    pg_bin: Path,
    socket_dir: Path,
    port: int,
    pgdata: Path,
    index_name: str,
) -> Path:
    result = psql(
        pg_bin,
        socket_dir,
        port,
        f"SELECT pg_relation_filepath('{index_name}'::regclass);",
    )
    relative = result.stdout.strip().splitlines()[-1]
    return pgdata / relative


def write_bytes(path: Path, offset: int, value: bytes) -> None:
    with path.open('r+b') as index_file:
        index_file.seek(offset)
        index_file.write(value)
        index_file.flush()
        os.fsync(index_file.fileno())


def flip_byte(path: Path, offset: int) -> None:
    with path.open('r+b') as index_file:
        index_file.seek(offset)
        value = index_file.read(1)
        if len(value) != 1:
            raise AssertionError(f'missing metapage byte at {offset}: {path}')
        index_file.seek(offset)
        index_file.write(bytes([value[0] ^ 1]))
        index_file.flush()
        os.fsync(index_file.fileno())


def truncate_file(path: Path, size: int) -> None:
    with path.open('r+b') as index_file:
        index_file.truncate(size)
        index_file.flush()
        os.fsync(index_file.fileno())


def query_sql(index_name: str) -> str:
    return f'''
        SELECT count(*)
        FROM evoke_query_tokens(
            '{index_name}'::regclass,
            ARRAY['boundary'],
            10,
            NULL
        );
    '''


def assert_rejected(
    pg_bin: Path,
    socket_dir: Path,
    port: int,
    index_name: str,
    expected: tuple[str, ...],
) -> str:
    result = psql(
        pg_bin,
        socket_dir,
        port,
        query_sql(index_name),
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError(f'{index_name} unexpectedly remained queryable')
    missing = [part for part in expected if part not in result.stderr]
    if missing:
        raise AssertionError(
            f'{index_name} error is missing {missing}: {result.stderr}'
        )
    return result.stderr.strip().splitlines()[0]


def run_boundary(
    pg_bin: Path,
    pgdata: Path,
    socket_dir: Path,
    port: int,
) -> dict[str, object]:
    index_names = (
        'docs_control_idx',
        'docs_header_idx',
        'docs_retired_idx',
        'docs_root_idx',
        'docs_bounds_idx',
        'docs_empty_idx',
    )
    psql(
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
        SELECT id, ARRAY['metapage', 'boundary', id::text]
        FROM generate_series(1, 128) AS id;
        CREATE INDEX docs_control_idx ON docs USING evoke (tokens);
        CREATE INDEX docs_header_idx ON docs USING evoke (tokens);
        CREATE INDEX docs_retired_idx ON docs USING evoke (tokens);
        CREATE INDEX docs_root_idx ON docs USING evoke (tokens);
        CREATE INDEX docs_bounds_idx ON docs USING evoke (tokens);
        CREATE INDEX docs_empty_idx ON docs USING evoke (tokens);
        CHECKPOINT;
        ''',
    )
    paths = {
        name: relation_path(
            pg_bin,
            socket_dir,
            port,
            pgdata,
            name,
        )
        for name in index_names
    }

    stop_cluster(pg_bin, pgdata)
    write_bytes(
        paths['docs_header_idx'],
        META_MAGIC_OFFSET,
        struct.pack('<I', 0),
    )
    write_bytes(
        paths['docs_retired_idx'],
        META_TID_BYTES_OFFSET,
        struct.pack('<Q', 1),
    )
    flip_byte(paths['docs_root_idx'], META_ROOT_OFFSET)
    truncate_file(paths['docs_bounds_idx'], BLCKSZ)
    truncate_file(paths['docs_empty_idx'], 0)
    start_cluster(pg_bin, pgdata)

    rejected = {
        'malformed_header': assert_rejected(
            pg_bin,
            socket_dir,
            port,
            'docs_header_idx',
            ('invalid evoke index metapage',),
        ),
        'retired_field': assert_rejected(
            pg_bin,
            socket_dir,
            port,
            'docs_retired_idx',
            (
                'invalid evoke convergent segment payload',
                'Validation failed: marked_corrupt.',
            ),
        ),
        'invalid_root': assert_rejected(
            pg_bin,
            socket_dir,
            port,
            'docs_root_idx',
            (
                'invalid evoke convergent segment payload',
                'Validation failed: invalid_segment_read_root.',
            ),
        ),
        'out_of_bounds_root': assert_rejected(
            pg_bin,
            socket_dir,
            port,
            'docs_bounds_idx',
            (
                'invalid evoke convergent segment payload',
                'Validation failed: segment_root_out_of_bounds.',
            ),
        ),
        'empty_relation': assert_rejected(
            pg_bin,
            socket_dir,
            port,
            'docs_empty_idx',
            ('evoke index relation', 'is empty'),
        ),
    }

    before_reindex = psql(
        pg_bin,
        socket_dir,
        port,
        query_sql('docs_control_idx'),
    ).stdout.strip()
    psql(
        pg_bin,
        socket_dir,
        port,
        'REINDEX INDEX docs_control_idx;',
    )
    after_reindex = psql(
        pg_bin,
        socket_dir,
        port,
        query_sql('docs_control_idx'),
    ).stdout.strip()
    if before_reindex != '10' or after_reindex != '10':
        raise AssertionError(
            'control index did not preserve queryability across REINDEX: '
            f'{before_reindex!r} -> {after_reindex!r}'
        )

    return {
        'passed': True,
        'rejected': rejected,
        'control_hits_before_reindex': int(before_reindex),
        'control_hits_after_reindex': int(after_reindex),
        'offsets': {
            'magic': META_MAGIC_OFFSET,
            'retired_tid_bytes': META_TID_BYTES_OFFSET,
            'serialized_root': META_ROOT_OFFSET,
        },
    }


def main() -> int:
    args = parse_args()
    pg_bin = args.pg_bin.expanduser().resolve()
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
        prefix='evoke-v3-metapage-read-',
        dir='/tmp',
    ))
    pgdata = work_root / 'pgdata'
    socket_dir = work_root / 'socket'
    socket_dir.mkdir()
    port = reserve_port()

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
