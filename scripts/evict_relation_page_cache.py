#!/usr/bin/env python3
"""Evict one isolated Evoke relation from the Linux page cache."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import mmap
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg import sql


SEGMENT_SUFFIX = re.compile(r'^[0-9]+$')
MINCORE_CHUNK_BYTES = 1024 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Checkpoint one isolated PostgreSQL instance, evict only the '
            'target Evoke relation main fork, and record mincore evidence.'
        ),
    )
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--expected-port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-resident-ratio', type=float, default=0.01)
    parser.add_argument('--max-resident-mib', type=int, default=64)
    parser.add_argument(
        '--allow-page-cache-eviction',
        action='store_true',
        help='required acknowledgement for relation-specific cache eviction',
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.allow_page_cache_eviction:
        raise ValueError('--allow-page-cache-eviction is required')
    if not 1 <= args.expected_port <= 65535:
        raise ValueError('--expected-port is invalid')
    if not 0.0 <= args.max_resident_ratio <= 1.0:
        raise ValueError('--max-resident-ratio must be between zero and one')
    if args.max_resident_mib < 0:
        raise ValueError('--max-resident-mib must be non-negative')
    if not hasattr(os, 'posix_fadvise'):
        raise RuntimeError('POSIX_FADV_DONTNEED is unavailable on this host')


def extension_schema(connection: psycopg.Connection[Any]) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT namespace.nspname
            FROM pg_extension AS extension
            JOIN pg_namespace AS namespace
              ON namespace.oid = extension.extnamespace
            WHERE extension.extname = 'evoke'
            """
        )
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError('evoke is not installed')
    return catalog_text(row[0])


def catalog_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode('ascii')
    return str(value)


def call_json(
    connection: psycopg.Connection[Any],
    schema: str,
    function: str,
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL('SELECT {}(%s::regclass)::jsonb').format(
                sql.Identifier(schema, function)
            ),
            (index_name,),
        )
        value = cursor.fetchone()[0]
    if not isinstance(value, dict):
        raise RuntimeError(f'{function} returned invalid JSON')
    return value


def generation_identity(status: dict[str, Any]) -> dict[str, Any]:
    accelerator = status.get('semantic_accelerator') or {}
    primary = status.get('primary') or {}
    posting = status.get('posting') or {}
    return {
        'generation_id': status.get('generation_id'),
        'contract_signature': status.get('contract_signature'),
        'posting_signature': posting.get('signature'),
        'manifest_start_block': primary.get('manifest_start_block'),
        'accelerator_source_manifest_id': accelerator.get(
            'source_manifest_id'
        ),
        'accelerator_builder_policy_id': accelerator.get(
            'builder_policy_id'
        ),
    }


def publication_stable(
    generation_before: dict[str, Any],
    generation_after: dict[str, Any],
    relation_before: dict[str, Any],
    relation_after: dict[str, Any],
) -> bool:
    return bool(
        generation_identity(generation_before) ==
            generation_identity(generation_after)
        and relation_before == relation_after
    )


def runtime_ready(state: dict[str, Any]) -> bool:
    generation = state.get('generation') or {}
    preload = state.get('shared_preload') or {}
    resident_fold_ready = bool(
        preload.get('resident_fold_current') is True
        and preload.get('resident_fold_loading') is not True
    )
    page_metadata_ready = bool(
        preload.get('query_warm_marker_valid') is True
        and preload.get('query_metadata_warm') is True
    )
    return bool(
        int(generation.get('auto_preload_priority') or 0) > 0
        and preload.get('available') is True
        and preload.get('loading') is not True
        and (resident_fold_ready or page_metadata_ready)
    )


def qualification_guard(
    connection: psycopg.Connection[Any],
    expected_port: int,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                current_setting('port')::int4,
                current_user,
                current_setting('data_directory'),
                pg_postmaster_start_time(),
                (SELECT count(*) FROM pg_stat_progress_create_index),
                (SELECT count(*)
                 FROM pg_stat_activity
                 WHERE backend_type = 'client backend'
                   AND pid <> pg_backend_pid()
                   AND state <> 'idle')
            """
        )
        row = cursor.fetchone()
        cursor.execute(
            'SELECT rolsuper FROM pg_roles WHERE rolname = current_user'
        )
        superuser = bool(cursor.fetchone()[0])
    if int(row[0]) != expected_port:
        raise RuntimeError(
            f'connected port {row[0]} differs from {expected_port}'
        )
    if not superuser:
        raise RuntimeError('qualification requires a superuser connection')
    if int(row[4]) != 0:
        raise RuntimeError('an index build is active')
    if int(row[5]) != 0:
        raise RuntimeError('another active client backend is present')
    return {
        'port': int(row[0]),
        'user': catalog_text(row[1]),
        'data_directory': catalog_text(row[2]),
        'postmaster_started_at': str(row[3]),
        'active_index_builds': int(row[4]),
        'other_active_clients': int(row[5]),
    }


def relation_details(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                relation.oid,
                relation.relkind,
                access_method.amname,
                pg_relation_filepath(relation.oid),
                pg_relation_size(relation.oid, 'main')
            FROM pg_class AS relation
            JOIN pg_am AS access_method
              ON access_method.oid = relation.relam
            WHERE relation.oid = %s::regclass
            """,
            (index_name,),
        )
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError('target index does not exist')
    relkind = catalog_text(row[1])
    access_method = catalog_text(row[2])
    if relkind != 'i' or access_method != 'evoke':
        raise RuntimeError('target relation is not an Evoke index')
    return {
        'oid': int(row[0]),
        'relkind': relkind,
        'access_method': access_method,
        'relative_path': catalog_text(row[3]),
        'main_fork_bytes': int(row[4]),
    }


def segment_number(path: Path, base: Path) -> int:
    if path == base:
        return 0
    prefix = base.name + '.'
    if path.parent != base.parent or not path.name.startswith(prefix):
        raise RuntimeError(f'unexpected relation segment path: {path}')
    suffix = path.name[len(prefix):]
    if SEGMENT_SUFFIX.fullmatch(suffix) is None:
        raise RuntimeError(f'invalid relation segment suffix: {path}')
    return int(suffix)


def relation_segments(data_directory: Path, relative_path: str) -> list[Path]:
    base = data_directory / relative_path
    candidates = [base]
    candidates.extend(
        path
        for path in base.parent.glob(base.name + '.*')
        if SEGMENT_SUFFIX.fullmatch(path.name[len(base.name) + 1:])
    )
    paths = sorted(set(candidates), key=lambda path: segment_number(path, base))
    for expected, path in enumerate(paths):
        if segment_number(path, base) != expected:
            raise RuntimeError('relation segment sequence is incomplete')
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f'relation segment is not a regular file: {path}')
    return paths


def resident_pages(path: Path, chunk_bytes: int = MINCORE_CHUNK_BYTES) -> int:
    page_size = mmap.PAGESIZE
    file_size = path.stat().st_size
    if file_size == 0:
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    total = 0
    with path.open('rb') as handle:
        offset = 0
        while offset < file_size:
            length = min(chunk_bytes, file_size - offset)
            with mmap.mmap(
                handle.fileno(),
                length,
                flags=mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset,
            ) as mapping:
                pages = math.ceil(length / page_size)
                vector = (ctypes.c_ubyte * pages)()
                anchor = ctypes.c_char.from_buffer(mapping)
                result = libc.mincore(
                    ctypes.byref(anchor),
                    ctypes.c_size_t(length),
                    vector,
                )
                del anchor
                if result != 0:
                    error = ctypes.get_errno()
                    raise OSError(error, os.strerror(error), str(path))
                total += sum(1 for value in vector if value & 1)
            offset += length
    return total


def residency(paths: Iterable[Path]) -> dict[str, Any]:
    page_size = mmap.PAGESIZE
    files = []
    total_bytes = 0
    total_pages = 0
    total_resident_pages = 0
    for path in paths:
        size = path.stat().st_size
        pages = math.ceil(size / page_size)
        resident = resident_pages(path)
        files.append({
            'path': str(path),
            'bytes': size,
            'pages': pages,
            'resident_pages': resident,
        })
        total_bytes += size
        total_pages += pages
        total_resident_pages += resident
    return {
        'page_size': page_size,
        'bytes': total_bytes,
        'pages': total_pages,
        'resident_pages': total_resident_pages,
        'resident_bytes': total_resident_pages * page_size,
        'resident_ratio': (
            total_resident_pages / total_pages if total_pages else 0.0
        ),
        'files': files,
    }


def evict(paths: Iterable[Path]) -> None:
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.posix_fadvise(
                descriptor,
                0,
                0,
                os.POSIX_FADV_DONTNEED,
            )
        finally:
            os.close(descriptor)


def residency_qualified(
    state: dict[str, Any],
    max_ratio: float,
    max_bytes: int,
) -> bool:
    return bool(
        float(state.get('resident_ratio') or 0.0) <= max_ratio
        and int(state.get('resident_bytes') or 0) <= max_bytes
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    started = time.time()
    with psycopg.connect(args.dsn, autocommit=True) as connection:
        guard = qualification_guard(connection, args.expected_port)
        schema = extension_schema(connection)
        relation = relation_details(connection, args.index)
        generation_before = call_json(
            connection,
            schema,
            'evoke_index_generation_status_internal',
            args.index,
        )
        runtime_before = call_json(
            connection,
            schema,
            'evoke_index_runtime_state_json',
            args.index,
        )
        if generation_before.get('valid') is not True:
            raise RuntimeError('target generation is not valid')
        if not runtime_ready(runtime_before):
            raise RuntimeError('target shared query metadata is not ready')
        with connection.cursor() as cursor:
            cursor.execute('CHECKPOINT')

    data_directory = Path(guard['data_directory'])
    paths = relation_segments(data_directory, relation['relative_path'])
    physical_bytes = sum(path.stat().st_size for path in paths)
    if physical_bytes != relation['main_fork_bytes']:
        raise RuntimeError(
            'relation segment bytes differ from PostgreSQL main-fork size'
        )
    before = residency(paths)
    evict(paths)

    with psycopg.connect(args.dsn, autocommit=True) as connection:
        guard_after = qualification_guard(connection, args.expected_port)
        schema_after = extension_schema(connection)
        relation_after = relation_details(connection, args.index)
        generation_after = call_json(
            connection,
            schema_after,
            'evoke_index_generation_status_internal',
            args.index,
        )
        runtime_after = call_json(
            connection,
            schema_after,
            'evoke_index_runtime_state_json',
            args.index,
        )
    stable = publication_stable(
        generation_before,
        generation_after,
        relation,
        relation_after,
    )
    paths_after = relation_segments(
        Path(guard_after['data_directory']),
        relation_after['relative_path'],
    )
    after_physical_bytes = sum(path.stat().st_size for path in paths_after)
    relation_bytes_stable = bool(
        after_physical_bytes == relation_after['main_fork_bytes']
        and after_physical_bytes == physical_bytes
    )
    after = residency(paths_after)
    max_bytes = args.max_resident_mib * 1024 * 1024
    passed = bool(
        stable
        and relation_bytes_stable
        and generation_after.get('valid') is True
        and runtime_ready(runtime_after)
        and residency_qualified(
            after,
            args.max_resident_ratio,
            max_bytes,
        )
    )
    output = {
        'schema': 'evoke_relation_page_cache_eviction_v1',
        'started_at_epoch': started,
        'finished_at_epoch': time.time(),
        'guard': guard,
        'guard_after': guard_after,
        'index': args.index,
        'relation': relation,
        'relation_after': relation_after,
        'relation_bytes_stable': relation_bytes_stable,
        'generation_identity': generation_identity(generation_before),
        'generation_identity_after': generation_identity(generation_after),
        'publication_stable': stable,
        'runtime': runtime_before,
        'runtime_after': runtime_after,
        'checkpoint_completed': True,
        'before': before,
        'after': after,
        'limits': {
            'max_resident_ratio': args.max_resident_ratio,
            'max_resident_mib': args.max_resident_mib,
        },
        'passed': passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(args.output)
    print(json.dumps({
        'output': str(args.output),
        'before_resident_ratio': before['resident_ratio'],
        'after_resident_ratio': after['resident_ratio'],
        'after_resident_bytes': after['resident_bytes'],
        'passed': passed,
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
