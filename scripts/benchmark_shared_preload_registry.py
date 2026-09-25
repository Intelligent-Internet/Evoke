#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psycopg

from test_unified_index_lifecycle_smoke import (
    configure_cluster,
    connect,
    run,
    start_cluster,
    stop_cluster,
)
SCHEMA = 'cap3_registry'
INDEX_NAME = f'{SCHEMA}.docs_idx'
DEFAULT_CAPACITIES = (1024, 10000, 65536)
DEFAULT_CLIENTS = (1, 16, 64)
QUERY = 'shared preload registry'


def raw_state_integer(state: dict[str, Any], name: str) -> int:
    raw_state = str(state.get('raw_state', ''))
    match = re.search(
        rf'(?:^|, ){re.escape(name)}=([0-9]+)',
        raw_state,
    )
    if match is None:
        raise AssertionError(
            f'shared registry state is missing {name}: {raw_state}'
        )
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Measure page-native query and exact-root marker behavior at '
            '1K/10K/65K shared-registry occupancy.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument('--extension-control-dir', type=Path)
    parser.add_argument('--port-base', type=int, default=55720)
    parser.add_argument(
        '--capacities',
        default=','.join(str(value) for value in DEFAULT_CAPACITIES),
    )
    parser.add_argument(
        '--clients',
        default=','.join(str(value) for value in DEFAULT_CLIENTS),
    )
    parser.add_argument('--queries-per-client', type=int, default=100)
    parser.add_argument('--warmup-queries', type=int, default=10)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument(
        '--max-65k-to-1k-p95-ratio',
        type=float,
        default=2.0,
    )
    parser.add_argument(
        '--min-64-client-throughput-scale',
        type=float,
        default=1.25,
    )
    parser.add_argument(
        '--min-65k-to-1k-64-client-qps-ratio',
        type=float,
        default=0.8,
    )
    parser.add_argument('--enforce-gates', action='store_true')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def parse_positive_csv(value: str, label: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(',')}))
    if not values or any(item <= 0 for item in values):
        raise ValueError(f'{label} must contain positive integers')
    return values


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(math.ceil(len(ordered) * fraction) - 1, 0)
    return ordered[index]


def query_rows(
    connection: psycopg.Connection[Any],
) -> tuple[tuple[int, float], ...]:
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT source.id, hit.score::float8
            FROM evoke_query(%s::regclass, %s, 10) AS hit
            JOIN {SCHEMA}.docs AS source
              ON source.ctid = hit.ctid
            ORDER BY hit.score DESC, source.id
            """,
            (INDEX_NAME, QUERY),
        )
        return tuple(
            (int(row[0]), round(float(row[1]), 12))
            for row in cursor.fetchall()
        )


def setup(connection: psycopg.Connection[Any]) -> tuple[tuple[int, float], ...]:
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE EXTENSION evoke;
            CREATE SCHEMA {SCHEMA};
            CREATE TABLE {SCHEMA}.docs (
                id integer PRIMARY KEY,
                body text NOT NULL
            );
            INSERT INTO {SCHEMA}.docs
            SELECT ordinal,
                   CASE
                       WHEN ordinal <= 20
                           THEN 'shared preload registry exact lookup '
                                || ordinal::text
                       ELSE 'unrelated document ' || ordinal::text
                   END
            FROM generate_series(1, 100) AS ordinal;
            CREATE INDEX docs_idx
            ON {SCHEMA}.docs
            USING evoke (body)
            WITH (auto_preload = 0);
            """
        )
        cursor.execute(
            'SELECT evoke_index_preload(%s::regclass)',
            (INDEX_NAME,),
        )
        preload = str(cursor.fetchone()[0])
    if 'tier=postgres_buffer_cache' not in preload:
        raise AssertionError(
            f'index did not use page-native buffer prewarm: {preload}'
        )
    state = registry_state(connection)
    warm_entries = raw_state_integer(
        {'raw_state': state},
        'shared_preload_unified_warm_entries',
    )
    synthetic_entries = raw_state_integer(
        {'raw_state': state},
        'shared_preload_test_registry_fill',
    )
    if warm_entries != synthetic_entries + 1:
        raise AssertionError(
            f'index did not publish one exact-root marker: {state}'
        )
    expected = query_rows(connection)
    if not expected:
        raise AssertionError('registry benchmark query returned no rows')
    return expected


def registry_state(
    connection: psycopg.Connection[Any],
) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_runtime_state(%s::regclass)',
            (INDEX_NAME,),
        )
        return str(cursor.fetchone()[0])


def registry_lifecycle_audit(
    connection: psycopg.Connection[Any],
    expected: tuple[tuple[int, float], ...],
    churn_indexes: int = 16,
) -> dict[str, Any]:
    before = registry_state(connection)
    before_evictions = raw_state_integer(
        {'raw_state': before},
        'shared_preload_relation_entry_evictions',
    )
    for ordinal in range(churn_indexes):
        index_name = f'docs_churn_{ordinal}'
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE INDEX {index_name}
                ON {SCHEMA}.docs
                USING evoke (body)
                WITH (auto_preload = 0)
                """
            )
            cursor.execute(
                'SELECT evoke_index_preload(%s::regclass)',
                (f'{SCHEMA}.{index_name}',),
            )
            preload = str(cursor.fetchone()[0])
        if 'tier=postgres_buffer_cache' not in preload:
            raise AssertionError(
                f'churn index {index_name} did not use page-native prewarm'
            )

    after_churn = registry_state(connection)
    after_evictions = raw_state_integer(
        {'raw_state': after_churn},
        'shared_preload_relation_entry_evictions',
    )
    churn_query_exact = query_rows(connection) == expected
    with connection.cursor() as cursor:
        cursor.execute('SELECT evoke_runtime_cache_clear()')
        clear_result = str(cursor.fetchone()[0])
    after_clear = registry_state(connection)
    clear_entries = raw_state_integer(
        {'raw_state': after_clear},
        'shared_preload_entries',
    )
    with connection.cursor() as cursor:
        cursor.execute(
            'SELECT evoke_index_preload(%s::regclass)',
            (INDEX_NAME,),
        )
        reload_result = str(cursor.fetchone()[0])
    recovered_query_exact = query_rows(connection) == expected
    after_reload = registry_state(connection)
    reload_entries = raw_state_integer(
        {'raw_state': after_reload},
        'shared_preload_entries',
    )
    passed = (
        after_evictions > before_evictions and
        churn_query_exact and
        clear_entries == 0 and
        'tier=postgres_buffer_cache' in reload_result and
        recovered_query_exact and
        reload_entries == 1
    )
    return {
        'churn_indexes': churn_indexes,
        'evictions_before': before_evictions,
        'evictions_after': after_evictions,
        'churn_query_exact': churn_query_exact,
        'clear_result': clear_result,
        'entries_after_clear': clear_entries,
        'reload_result': reload_result,
        'entries_after_reload': reload_entries,
        'recovered_query_exact': recovered_query_exact,
        'passed': passed,
    }


def invalid_startup_audit(
    args: argparse.Namespace,
    port: int,
) -> dict[str, Any]:
    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    with tempfile.TemporaryDirectory(
        prefix='evoke_cap3_invalid_',
    ) as tmp:
        root = Path(tmp)
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
            port,
            args.extension_libdir,
            args.extension_control_dir,
        )
        with (data_dir / 'postgresql.conf').open(
            'a',
            encoding='utf-8',
        ) as handle:
            handle.write(
                'evoke.test_shared_preload_registry_capacity = 1024\n'
            )
            handle.write(
                'evoke.test_shared_preload_registry_fill = 1024\n'
            )
        result = run(
            [
                str(pg_ctl),
                '-D',
                str(data_dir),
                '-l',
                str(log_path),
                'start',
                '-w',
            ],
            check=False,
        )
        started = result.returncode == 0
        if started:
            stop_cluster(pg_ctl, data_dir)
        log_text = (
            log_path.read_text(encoding='utf-8', errors='replace')
            if log_path.exists()
            else ''
        )
        expected_error = (
            'evoke test registry fill exceeds its capacity' in log_text
        )
        return {
            'capacity': 1024,
            'fill': 1024,
            'startup_returncode': result.returncode,
            'started': started,
            'expected_error': expected_error,
            'passed': not started and expected_error,
        }


def benchmark_client(
    socket_dir: Path,
    port: int,
    barrier: threading.Barrier,
    expected: tuple[tuple[int, float], ...],
    query_count: int,
    warmup_count: int,
) -> list[float]:
    connection = connect(socket_dir, port)
    try:
        for _ in range(warmup_count):
            if query_rows(connection) != expected:
                raise AssertionError('warmup result differs from baseline')
        barrier.wait(timeout=60.0)
        samples: list[float] = []
        for _ in range(query_count):
            started = time.perf_counter()
            rows = query_rows(connection)
            samples.append((time.perf_counter() - started) * 1000.0)
            if rows != expected:
                raise AssertionError('concurrent result differs from baseline')
        return samples
    finally:
        connection.close()


def benchmark_clients(
    socket_dir: Path,
    port: int,
    client_count: int,
    expected: tuple[tuple[int, float], ...],
    query_count: int,
    warmup_count: int,
) -> dict[str, float | int]:
    barrier = threading.Barrier(client_count + 1)
    started = 0.0
    finished = 0.0
    samples: list[float] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=client_count,
    ) as executor:
        futures = [
            executor.submit(
                benchmark_client,
                socket_dir,
                port,
                barrier,
                expected,
                query_count,
                warmup_count,
            )
            for _ in range(client_count)
        ]
        barrier.wait(timeout=60.0)
        started = time.perf_counter()
        for future in concurrent.futures.as_completed(futures):
            samples.extend(future.result())
        finished = time.perf_counter()
    elapsed_seconds = finished - started
    return {
        'clients': client_count,
        'queries': len(samples),
        'elapsed_seconds': round(elapsed_seconds, 6),
        'qps': round(len(samples) / elapsed_seconds, 3),
        'median_ms': round(statistics.median(samples), 6),
        'p95_ms': round(percentile(samples, 0.95), 6),
        'max_ms': round(max(samples), 6),
    }


def benchmark_client_trials(
    socket_dir: Path,
    port: int,
    client_count: int,
    expected: tuple[tuple[int, float], ...],
    query_count: int,
    warmup_count: int,
    trials: int,
) -> dict[str, Any]:
    samples = [
        benchmark_clients(
            socket_dir,
            port,
            client_count,
            expected,
            query_count,
            warmup_count,
        )
        for _ in range(trials)
    ]
    return {
        'clients': client_count,
        'trials': samples,
        'queries_per_trial': samples[0]['queries'],
        'qps': round(
            statistics.median(
                float(sample['qps'])
                for sample in samples
            ),
            3,
        ),
        'median_ms': round(
            statistics.median(
                float(sample['median_ms'])
                for sample in samples
            ),
            6,
        ),
        'p95_ms': round(
            statistics.median(
                float(sample['p95_ms'])
                for sample in samples
            ),
            6,
        ),
        'max_ms': round(
            max(float(sample['max_ms']) for sample in samples),
            6,
        ),
    }


def observe_capacity(
    args: argparse.Namespace,
    capacity: int,
    clients: tuple[int, ...],
    ordinal: int,
) -> dict[str, Any]:
    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    with tempfile.TemporaryDirectory(
        prefix=f'evoke_cap3_{capacity}_',
    ) as tmp:
        root = Path(tmp)
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
            args.port_base + ordinal,
            args.extension_libdir,
            args.extension_control_dir,
        )
        fill = capacity - 8
        with (data_dir / 'postgresql.conf').open(
            'a',
            encoding='utf-8',
        ) as handle:
            handle.write(
                'evoke.test_shared_preload_registry_capacity = '
                f'{capacity}\n'
            )
            handle.write(
                'evoke.test_shared_preload_registry_fill = '
                f'{fill}\n'
            )

        started = False
        connection: psycopg.Connection[Any] | None = None
        try:
            start_cluster(pg_ctl, data_dir, log_path)
            started = True
            connection = connect(socket_dir, args.port_base + ordinal)
            expected = setup(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT evoke_index_runtime_state(%s::regclass)',
                    (INDEX_NAME,),
                )
                raw_state = str(cursor.fetchone()[0])
            observed_capacity = raw_state_integer(
                {'raw_state': raw_state},
                'shared_preload_entry_capacity',
            )
            observed_fill = raw_state_integer(
                {'raw_state': raw_state},
                'shared_preload_test_registry_fill',
            )
            if observed_capacity != capacity or observed_fill != fill:
                raise AssertionError(
                    'registry startup occupancy differs from configuration: '
                    f'{raw_state}'
                )
            results = [
                benchmark_client_trials(
                    socket_dir,
                    args.port_base + ordinal,
                    client_count,
                    expected,
                    args.queries_per_client,
                    args.warmup_queries,
                    args.trials,
                )
                for client_count in clients
            ]
            lifecycle = (
                registry_lifecycle_audit(connection, expected)
                if ordinal == 0
                else None
            )
            return {
                'capacity': capacity,
                'synthetic_fill': fill,
                'real_entries': raw_state_integer(
                    {'raw_state': raw_state},
                    'shared_preload_entries',
                ) - fill,
                'result_fingerprint': expected,
                'clients': results,
                'lifecycle': lifecycle,
            }
        finally:
            if connection is not None:
                connection.close()
            if started:
                stop_cluster(pg_ctl, data_dir)


def client_result(
    observation: dict[str, Any],
    clients: int,
) -> dict[str, Any]:
    for result in observation['clients']:
        if result['clients'] == clients:
            return result
    raise KeyError(f'missing {clients}-client result')


def evaluate_gates(
    observations: list[dict[str, Any]],
    invalid_startup: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    failures: list[str] = []
    smallest = observations[0]
    largest = observations[-1]
    p95_ratio = (
        client_result(largest, 1)['p95_ms'] /
        client_result(smallest, 1)['p95_ms']
    )
    largest_single_qps = client_result(largest, 1)['qps']
    largest_concurrent_qps = client_result(largest, 64)['qps']
    throughput_scale = largest_concurrent_qps / largest_single_qps
    capacity_qps_ratio = (
        largest_concurrent_qps /
        client_result(smallest, 64)['qps']
    )
    for observation in observations:
        if observation['real_entries'] != 1:
            failures.append(
                f"{observation['capacity']}: expected one real entry"
            )
    lifecycle = smallest.get('lifecycle')
    if not isinstance(lifecycle, dict) or not lifecycle.get('passed'):
        failures.append('registry eviction/clear/reload lifecycle failed')
    if not invalid_startup['passed']:
        failures.append('invalid registry startup configuration was accepted')
    if p95_ratio > args.max_65k_to_1k_p95_ratio:
        failures.append(
            f'65K/1K single-client p95 ratio {p95_ratio:.3f} exceeds '
            f'{args.max_65k_to_1k_p95_ratio:.3f}'
        )
    if throughput_scale < args.min_64_client_throughput_scale:
        failures.append(
            f'65K 64-client throughput scale {throughput_scale:.3f} is '
            f'below {args.min_64_client_throughput_scale:.3f}'
        )
    if capacity_qps_ratio < args.min_65k_to_1k_64_client_qps_ratio:
        failures.append(
            f'65K/1K 64-client QPS ratio {capacity_qps_ratio:.3f} is '
            f'below {args.min_65k_to_1k_64_client_qps_ratio:.3f}'
        )
    return {
        'enforced': args.enforce_gates,
        '65k_to_1k_single_client_p95_ratio': round(p95_ratio, 4),
        '65k_64_client_throughput_scale': round(throughput_scale, 4),
        '65k_to_1k_64_client_qps_ratio': round(capacity_qps_ratio, 4),
        'max_p95_ratio': args.max_65k_to_1k_p95_ratio,
        'min_throughput_scale': args.min_64_client_throughput_scale,
        'min_65k_to_1k_64_client_qps_ratio': (
            args.min_65k_to_1k_64_client_qps_ratio
        ),
        'failures': failures,
        'passed': not failures or not args.enforce_gates,
        'meets_acceptance': not failures,
    }


def main() -> int:
    args = parse_args()
    capacities = parse_positive_csv(args.capacities, 'capacities')
    clients = parse_positive_csv(args.clients, 'clients')
    if capacities[0] < 1024 or capacities[-1] > 65536:
        raise ValueError('capacities must be between 1024 and 65536')
    if 1 not in clients or 64 not in clients:
        raise ValueError('clients must include 1 and 64')
    if (
        args.queries_per_client <= 0 or
        args.warmup_queries < 0 or
        args.trials <= 0
    ):
        raise ValueError('query counts are invalid')

    observations = [
        observe_capacity(args, capacity, clients, ordinal)
        for ordinal, capacity in enumerate(capacities)
    ]
    invalid_startup = invalid_startup_audit(
        args,
        args.port_base + len(capacities),
    )
    gates = evaluate_gates(observations, invalid_startup, args)
    report = {
        'route': 'shared preload registry capacity',
        'capacities': list(capacities),
        'client_counts': list(clients),
        'queries_per_client': args.queries_per_client,
        'trials': args.trials,
        'observations': observations,
        'invalid_startup': invalid_startup,
        'gates': gates,
        'passed': gates['passed'],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
