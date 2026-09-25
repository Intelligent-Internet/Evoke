#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import psycopg

from evoke_test_support import (
    create_short_socket_root,
    extension_control_root,
)
from test_convergent_segment_read_smoke import (
    configure_cluster,
    connect,
    maintenance_result_fields,
    pg_config_value,
    reserve_port,
    run,
    start_cluster,
    stop_cluster,
)


DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')
DEFAULT_EXTENSION_LIBDIR = Path('/opt/homebrew/lib/postgresql@18')
DEFAULT_EXTENSION_CONTROL_DIR = Path(
    '/opt/homebrew/share/postgresql@18'
)


def parse_vocab_sizes(value: str) -> list[int]:
    sizes: list[int] = []

    for item in value.split(','):
        try:
            size = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f'invalid vocabulary size: {item}'
            ) from error
        if size <= 0:
            raise argparse.ArgumentTypeError(
                'vocabulary sizes must be positive'
            )
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError(
            'at least one vocabulary size is required'
        )
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Measure cold convergent text-seal cost as vocabulary grows.'
        )
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument(
        '--extension-libdir',
        type=Path,
        default=DEFAULT_EXTENSION_LIBDIR,
    )
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        default=DEFAULT_EXTENSION_CONTROL_DIR,
    )
    parser.add_argument(
        '--vocab-sizes',
        type=parse_vocab_sizes,
        default=parse_vocab_sizes('2000,20000,100000'),
    )
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error('--repeats must be positive')
    return args


def create_index(
    connection: psycopg.Connection[Any],
    vocabulary_size: int,
) -> tuple[str, float]:
    table_name = f'bench.docs_{vocabulary_size}'
    index_name = f'bench.docs_{vocabulary_size}_idx'

    with connection.cursor() as cursor:
        cursor.execute(
            f'CREATE TABLE {table_name} ('
            'id int PRIMARY KEY, body text NOT NULL)'
        )
        cursor.execute(
            f'INSERT INTO {table_name} '
            "SELECT 1, string_agg('v' || lpad(g::text, 9, '0'), ' ') "
            'FROM generate_series(1, %s) AS terms(g)',
            (vocabulary_size,),
        )
        started_at = time.perf_counter()
        cursor.execute(
            f'CREATE INDEX docs_{vocabulary_size}_idx '
            f'ON {table_name} USING evoke (body) '
            'WITH (sae=false, consistency=realtime)'
        )
        build_seconds = time.perf_counter() - started_at
    return index_name, build_seconds


def append_pending_row(
    connection: psycopg.Connection[Any],
    vocabulary_size: int,
    repeat: int,
) -> str:
    table_name = f'bench.docs_{vocabulary_size}'
    new_term = f'fresh_{vocabulary_size}_{repeat}'

    with connection.cursor() as cursor:
        cursor.execute(
            "SET evoke.test_convergent_l0_rotation_records = '1'"
        )
        cursor.execute(
            f'INSERT INTO {table_name} VALUES (%s, %s)',
            (
                repeat + 2,
                f'v{1:09d} {new_term}',
            ),
        )
        cursor.execute('RESET evoke.test_convergent_l0_rotation_records')
    return new_term


def maintain_until_sealed(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> tuple[float, list[dict[str, str]]]:
    seal_seconds = 0.0
    results: list[dict[str, str]] = []

    for _ in range(8):
        started_at = time.perf_counter()
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT evoke_index_try_maintain(%s::regclass)',
                (index_name,),
            )
            row = cursor.fetchone()
        elapsed = time.perf_counter() - started_at
        if row is None or not isinstance(row[0], str):
            raise AssertionError('invalid evoke maintenance result')
        fields = maintenance_result_fields(row[0])
        fields['elapsed_seconds'] = f'{elapsed:.9f}'
        results.append(fields)
        if fields.get('mode') != 'segment_seal':
            continue
        if fields.get('maintained') != 'true':
            raise AssertionError(
                f'cold pending seal did not publish: {results}'
            )
        seal_seconds = elapsed
        break
    if seal_seconds <= 0.0:
        raise AssertionError(
            f'pending frontier did not seal: {results}'
        )
    return seal_seconds, results


def assert_term_visible(
    connection: psycopg.Connection[Any],
    index_name: str,
    term: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT count(*) FROM evoke_query(%s::regclass, %s, 10)',
            (index_name, term),
        )
        row = cursor.fetchone()
    if row is None or int(row[0]) == 0:
        raise AssertionError(f'sealed term is not searchable: {term}')


def capture_storage(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> dict[str, int | str]:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT pg_relation_size(%s::regclass)::bigint, '
            'details.index_bytes::bigint, details.pages::bigint, '
            'pg_current_wal_insert_lsn()::text '
            'FROM evoke_index_details(%s::regclass) AS details',
            (index_name, index_name),
        )
        row = cursor.fetchone()
    if row is None:
        raise AssertionError('missing evoke storage measurement')
    return {
        'relation_bytes': int(row[0]),
        'logical_index_bytes': int(row[1]),
        'relation_pages': int(row[2]),
        'wal_lsn': str(row[3]),
    }


def wal_bytes_between(
    connection: psycopg.Connection[Any],
    before_lsn: str,
    after_lsn: str,
) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT pg_wal_lsn_diff(%s::pg_lsn, %s::pg_lsn)::bigint',
            (after_lsn, before_lsn),
        )
        row = cursor.fetchone()
    if row is None:
        raise AssertionError('missing evoke WAL measurement')
    return int(row[0])


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        'minimum': ordered[0],
        'median': statistics.median(ordered),
        'maximum': ordered[-1],
        'mean': statistics.fmean(ordered),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    extension_libdir = args.extension_libdir.expanduser().resolve()
    extension_control_dir = extension_control_root(
        args.extension_control_dir
    )
    if not any(
        (extension_libdir / name).is_file()
        for name in ('evoke.so', 'evoke.dylib')
    ):
        raise FileNotFoundError(
            f'evoke library is missing from {extension_libdir}'
        )

    root = create_short_socket_root('evoke-cold-text-seal-')
    data_dir = root / 'data'
    socket_dir = root / 's'
    log_path = root / 'postgres.log'
    socket_dir.mkdir()
    port = reserve_port()
    pg_ctl = args.pg_bin / 'pg_ctl'
    started = False
    connection: psycopg.Connection[Any] | None = None
    rows: list[dict[str, Any]] = []

    try:
        run(
            [
                str(args.pg_bin / 'initdb'),
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
            system_libdir=pg_config_value(args.pg_bin, '--pkglibdir'),
            extension_control_dir=extension_control_dir,
            system_sharedir=pg_config_value(args.pg_bin, '--sharedir'),
        )
        with (data_dir / 'postgresql.conf').open(
            'a',
            encoding='utf-8',
        ) as handle:
            handle.write("\nshared_preload_libraries = ''\n")
            handle.write('max_worker_processes = 0\n')
            handle.write(
                'evoke.maintenance_timer_interval_ms = 2147483647\n'
            )
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        with connection.cursor() as cursor:
            cursor.execute('CREATE EXTENSION evoke')
            cursor.execute('CREATE SCHEMA bench')

        for vocabulary_size in args.vocab_sizes:
            index_name, build_seconds = create_index(
                connection,
                vocabulary_size,
            )
            build_storage = capture_storage(connection, index_name)
            seal_samples: list[float] = []
            relation_byte_deltas: list[int] = []
            logical_byte_deltas: list[int] = []
            relation_page_deltas: list[int] = []
            wal_byte_deltas: list[int] = []
            maintenance: list[list[dict[str, str]]] = []
            for repeat in range(args.repeats):
                new_term = append_pending_row(
                    connection,
                    vocabulary_size,
                    repeat,
                )
                connection.close()
                connection = connect(socket_dir, port)
                storage_before = capture_storage(connection, index_name)
                seal_seconds, results = maintain_until_sealed(
                    connection,
                    index_name,
                )
                assert_term_visible(connection, index_name, new_term)
                storage_after = capture_storage(connection, index_name)
                seal_samples.append(seal_seconds)
                relation_byte_deltas.append(
                    int(storage_after['relation_bytes']) -
                    int(storage_before['relation_bytes'])
                )
                logical_byte_deltas.append(
                    int(storage_after['logical_index_bytes']) -
                    int(storage_before['logical_index_bytes'])
                )
                relation_page_deltas.append(
                    int(storage_after['relation_pages']) -
                    int(storage_before['relation_pages'])
                )
                wal_byte_deltas.append(wal_bytes_between(
                    connection,
                    str(storage_before['wal_lsn']),
                    str(storage_after['wal_lsn']),
                ))
                maintenance.append(results)
            rows.append(
                {
                    'vocabulary_size': vocabulary_size,
                    'changed_terms_per_seal': 2,
                    'build_seconds': build_seconds,
                    'build_storage': build_storage,
                    'seal_seconds': seal_samples,
                    'seal_summary': summarize(seal_samples),
                    'relation_byte_deltas': relation_byte_deltas,
                    'relation_byte_delta_summary': summarize(
                        relation_byte_deltas
                    ),
                    'logical_byte_deltas': logical_byte_deltas,
                    'logical_byte_delta_summary': summarize(
                        logical_byte_deltas
                    ),
                    'relation_page_deltas': relation_page_deltas,
                    'relation_page_delta_summary': summarize(
                        relation_page_deltas
                    ),
                    'wal_byte_deltas': wal_byte_deltas,
                    'wal_byte_delta_summary': summarize(wal_byte_deltas),
                    'maintenance': maintenance,
                }
            )
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
        'api_version': 'evoke_benchmark_v1',
        'benchmark': 'convergent cold text pending seal',
        'vocabulary_sizes': args.vocab_sizes,
        'repeats': args.repeats,
        'rows': rows,
    }


def main() -> None:
    args = parse_args()
    if args.output is not None:
        args.output.unlink(missing_ok=True)
    report = run_benchmark(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
