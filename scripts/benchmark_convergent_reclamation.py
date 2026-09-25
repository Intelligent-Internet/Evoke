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


def parse_positive_ints(value: str) -> list[int]:
    values: list[int] = []

    for item in value.split(','):
        try:
            parsed = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f'invalid positive integer: {item}'
            ) from error
        if parsed <= 0:
            raise argparse.ArgumentTypeError(
                'values must be positive'
            )
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError(
            'at least one value is required'
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Measure cold optional compaction/reclamation cost as the '
            'vocabulary grows.'
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
        type=parse_positive_ints,
        default=parse_positive_ints('2000,20000,100000'),
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
    repeat: int,
) -> tuple[str, str]:
    suffix = f'{vocabulary_size}_{repeat}'
    table_name = f'bench.docs_{suffix}'
    index_name = f'bench.docs_{suffix}_idx'

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
        cursor.execute(
            f'CREATE INDEX docs_{suffix}_idx '
            f'ON {table_name} USING evoke (body) '
            'WITH (sae=false, consistency=realtime)'
        )
    return table_name, index_name


def maintain_until_mode(
    connection: psycopg.Connection[Any],
    index_name: str,
    expected_mode: str,
    attempts: int = 16,
) -> tuple[float, dict[str, str], list[dict[str, str]]]:
    results: list[dict[str, str]] = []

    for _ in range(attempts):
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
        if fields.get('mode') != expected_mode:
            continue
        if fields.get('maintained') != 'true':
            raise AssertionError(
                f'{expected_mode} did not publish: {results}'
            )
        return elapsed, fields, results
    raise AssertionError(
        f'maintenance did not reach {expected_mode}: {results}'
    )


def append_segment(
    connection: psycopg.Connection[Any],
    table_name: str,
    index_name: str,
    vocabulary_size: int,
    ordinal: int,
) -> list[dict[str, str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SET evoke.test_convergent_l0_rotation_records = '1'"
        )
        cursor.execute(
            f'INSERT INTO {table_name} VALUES (%s, %s)',
            (
                ordinal + 1,
                f'v{1:09d} fresh_{vocabulary_size}_{ordinal}',
            ),
        )
        cursor.execute('RESET evoke.test_convergent_l0_rotation_records')
    _, _, maintenance = maintain_until_mode(
        connection,
        index_name,
        'segment_seal',
    )
    return maintenance


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

    root = create_short_socket_root('evoke-reclamation-bench-')
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
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        with connection.cursor() as cursor:
            cursor.execute('CREATE EXTENSION evoke')
            cursor.execute('CREATE SCHEMA bench')

        for vocabulary_size in args.vocab_sizes:
            samples: list[float] = []
            runs: list[dict[str, Any]] = []
            for repeat in range(args.repeats):
                table_name, index_name = create_index(
                    connection,
                    vocabulary_size,
                    repeat,
                )
                seal_maintenance = [
                    append_segment(
                        connection,
                        table_name,
                        index_name,
                        vocabulary_size,
                        ordinal,
                    )
                    for ordinal in range(1, 5)
                ]
                connection.close()
                connection = connect(socket_dir, port)
                elapsed, compaction, maintenance = maintain_until_mode(
                    connection,
                    index_name,
                    'segment_compaction',
                )
                with connection.cursor() as cursor:
                    cursor.execute(
                        'SELECT count(*) FROM evoke_query('
                        '%s::regclass, %s, 10)',
                        (index_name, 'v000000001'),
                    )
                    result_count = int(cursor.fetchone()[0])
                    cursor.execute(f'DROP TABLE {table_name}')
                if result_count != 5:
                    raise AssertionError(
                        'compaction changed exact result count: '
                        f'{result_count}'
                    )
                samples.append(elapsed)
                runs.append(
                    {
                        'seal_maintenance': seal_maintenance,
                        'compaction': compaction,
                        'maintenance': maintenance,
                        'result_count': result_count,
                    }
                )
            rows.append(
                {
                    'vocabulary_size': vocabulary_size,
                    'segments_compacted': 4,
                    'changed_terms_per_segment': 2,
                    'compaction_seconds': samples,
                    'compaction_summary': summarize(samples),
                    'runs': runs,
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
        'benchmark': 'convergent optional compaction reclamation',
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
