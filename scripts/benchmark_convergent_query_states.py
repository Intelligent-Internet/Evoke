#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import psycopg

from evoke_test_support import (
    create_short_socket_root,
    extension_control_root,
)
from test_convergent_segment_read_smoke import (
    acquire_maintenance_guard,
    assert_rows_close,
    configure_cluster,
    connect,
    fetch_generation_cache_state,
    fetch_ids,
    fetch_status,
    maintenance_guard_is_held,
    maintenance_result_fields,
    pg_config_value,
    release_maintenance_guard,
    reserve_port,
    run,
    start_cluster,
    stop_cluster,
    try_maintain,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')
QUERY_TERM = 1
ORDERED_QUERY_TERMS = tuple(range(1, 9))
QUERY_LIMIT = 100
LATENCY_EQUIVALENCE_LIMIT = 1.05
P99_EQUIVALENCE_LIMIT = 1.10
QPS_EQUIVALENCE_FLOOR = 0.95
POSTING_HEAT_BENEFIT_BYTES = 65_536
POSTGRES_BLOCK_BYTES = 8_192
QUERY_SQL = (
    'SELECT doc_id, score::float8 '
    'FROM evoke_query_ids(%s::regclass, %s::int4[], %s)'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Benchmark exact native Evoke queries across static, '
            'fragmented, workload-folded, impact-specialized, and '
            'page-native static-reference index states.'
        ),
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument('--extension-libdir', type=Path, required=True)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        required=True,
        help=(
            'PostgreSQL share root containing extension/evoke.control, '
            'or the extension directory itself.'
        ),
    )
    parser.add_argument('--base-docs', type=int, default=30_000)
    parser.add_argument('--delta-docs', type=int, default=5_000)
    parser.add_argument('--delta-batches', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        'base_docs',
        'delta_docs',
        'delta_batches',
        'samples',
        'trials',
    )
    for name in positive_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if args.warmup < 0:
        raise ValueError('--warmup must not be negative')
    if args.delta_batches != 2:
        raise ValueError(
            '--delta-batches must be 2 for the three-state matrix'
        )


def command_output(command: list[str]) -> str:
    return run(command).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_identity() -> dict[str, Any]:
    status = command_output([
        'git',
        'status',
        '--porcelain',
        '--untracked-files=normal',
    ])
    product_status = command_output([
        'git',
        'status',
        '--porcelain',
        '--',
        'Makefile',
        'evoke.control',
        'sql',
        'src',
    ])
    return {
        'commit': command_output(['git', 'rev-parse', 'HEAD']),
        'tree': 'clean' if not status else 'dirty',
        'product_tree': 'clean' if not product_status else 'dirty',
        'branch': command_output(['git', 'branch', '--show-current']),
    }


def host_snapshot() -> dict[str, Any]:
    load_1m, load_5m, load_15m = os.getloadavg()
    return {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'logical_cpus': os.cpu_count(),
        'load_average': {
            '1m': round(load_1m, 3),
            '5m': round(load_5m, 3),
            '15m': round(load_15m, 3),
        },
    }


def extension_library_exists(extension_libdir: Path) -> bool:
    return any(
        (extension_libdir / name).is_file()
        for name in ('evoke.so', 'evoke.dylib')
    )


def extension_library_identity(extension_libdir: Path) -> dict[str, Any]:
    libraries = [
        extension_libdir / name
        for name in ('evoke.so', 'evoke.dylib')
        if (extension_libdir / name).is_file()
    ]
    if len(libraries) != 1:
        raise FileNotFoundError(
            'expected exactly one staged evoke extension library in '
            f'{extension_libdir}'
        )
    library = libraries[0]
    return {
        'path': str(library),
        'size': library.stat().st_size,
        'sha256': sha256_file(library),
    }


def insert_documents(
    connection: psycopg.Connection[Any],
    start_id: int,
    document_count: int,
) -> None:
    end_id = start_id + document_count - 1
    statement = (
        'INSERT INTO bench.documents '
        'SELECT document_id, '
        'ARRAY['
        '1, '
        '2 + (document_id %% 2048), '
        '4096 + ((document_id * 17) %% 4096), '
        '8192 + ((document_id * 31) %% 8192)'
        ']::int4[] '
        'FROM generate_series(%s::int4, %s::int4) '
        'AS rows(document_id)'
    )
    with connection.cursor() as cursor:
        cursor.execute(statement, (start_id, end_id))


def create_convergent_index(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            f'CREATE INDEX {index_name} '
            'ON bench.documents USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )


def create_static_reference_index(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            f'CREATE INDEX {index_name} '
            'ON bench.documents USING evoke (tokens) '
            'WITH (sae=false, consistency=realtime)'
        )


def drain_l0(
    connection: psycopg.Connection[Any],
    index_name: str,
) -> list[str]:
    results: list[str] = []
    for _ in range(8):
        status = fetch_status(connection, index_name)
        delta = status['generation']['delta']
        if (
            delta['active']['records'] == 0
            and delta['pending']['records'] == 0
        ):
            return results
        result = try_maintain(connection, index_name)
        fields = maintenance_result_fields(result)
        results.append(result)
        if fields.get('mode') == 'segment_compaction':
            raise AssertionError(
                'benchmark setup unexpectedly compacted segments: '
                f'{result}'
            )
    raise AssertionError(
        f'benchmark L0 did not drain for {index_name}: {results}'
    )


def append_sealed_batch(
    connection: psycopg.Connection[Any],
    index_name: str,
    start_id: int,
    document_count: int,
) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_config("
            "'evoke.test_convergent_l0_rotation_records', %s, false)",
            (str(document_count),),
        )
    try:
        insert_documents(connection, start_id, document_count)
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                'RESET evoke.test_convergent_l0_rotation_records'
            )
    return drain_l0(connection, index_name)


def query_once(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    query_terms: tuple[int, ...] = (QUERY_TERM,),
) -> list[tuple[int, float]]:
    cursor.execute(
        QUERY_SQL,
        (index_name, list(query_terms), QUERY_LIMIT),
    )
    return [
        (int(row[0]), float(row[1]))
        for row in cursor.fetchall()
    ]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError('cannot compute a percentile of an empty sample')
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def result_id_sha256(rows: list[tuple[int, float]]) -> str:
    payload = ','.join(str(document_id) for document_id, _ in rows)
    return hashlib.sha256(payload.encode('ascii')).hexdigest()


def max_score_difference(
    left: list[tuple[int, float]],
    right: list[tuple[int, float]],
) -> float:
    if len(left) != len(right):
        raise AssertionError('cannot compare differently sized result sets')
    return max(
        (
            abs(left_row[1] - right_row[1])
            for left_row, right_row in zip(left, right)
        ),
        default=0.0,
    )


def measure_trial(
    connection: psycopg.Connection[Any],
    index_name: str,
    expected: list[tuple[int, float]],
    *,
    samples: int,
    query_terms: tuple[int, ...] = (QUERY_TERM,),
) -> dict[str, float]:
    latencies_ms: list[float] = []
    with connection.cursor() as cursor:
        for _ in range(samples):
            started = time.perf_counter_ns()
            rows = query_once(cursor, index_name, query_terms)
            elapsed_ms = (
                time.perf_counter_ns() - started
            ) / 1_000_000.0
            assert_rows_close(
                expected,
                rows,
                label=f'{index_name} timed query',
            )
            latencies_ms.append(elapsed_ms)
    elapsed_trial = sum(latencies_ms) / 1000.0
    return {
        'mean_ms': statistics.fmean(latencies_ms),
        'p50_ms': percentile(latencies_ms, 0.50),
        'p95_ms': percentile(latencies_ms, 0.95),
        'p99_ms': percentile(latencies_ms, 0.99),
        'qps': samples / elapsed_trial,
    }


def summarize_trials(
    index_name: str,
    expected: list[tuple[int, float]],
    *,
    warmup: int,
    samples: int,
    trial_rows: list[dict[str, float | int]],
) -> dict[str, Any]:
    metric_names = ('mean_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'qps')
    aggregate = {
        name: statistics.median(
            float(trial[name]) for trial in trial_rows
        )
        for name in metric_names
    }
    return {
        'index': index_name,
        'queries': warmup + samples * len(trial_rows),
        'warmup': warmup,
        'samples_per_trial': samples,
        'trials': trial_rows,
        'median_trial': aggregate,
        'result_count': len(expected),
        'result_id_sha256': result_id_sha256(expected),
    }


def benchmark_pair(
    connection: psycopg.Connection[Any],
    states: tuple[tuple[str, str], tuple[str, str]],
    expected: list[tuple[int, float]],
    *,
    warmup: int,
    samples: int,
    trials: int,
    query_terms: tuple[int, ...] = (QUERY_TERM,),
) -> dict[str, dict[str, Any]]:
    trial_rows: dict[str, list[dict[str, float | int]]] = {
        label: []
        for label, _ in states
    }
    with connection.cursor() as cursor:
        for label, index_name in states:
            for _ in range(warmup):
                rows = query_once(cursor, index_name, query_terms)
                assert_rows_close(
                    expected,
                    rows,
                    label=f'{label} warmup',
                )

    for trial in range(trials):
        ordered_states = states if trial % 2 == 0 else tuple(reversed(states))
        for label, index_name in ordered_states:
            row: dict[str, float | int] = {
                'trial': trial + 1,
            }
            row.update(
                measure_trial(
                    connection,
                    index_name,
                    expected,
                    samples=samples,
                    query_terms=query_terms,
                )
            )
            trial_rows[label].append(row)

    return {
        label: summarize_trials(
            index_name,
            expected,
            warmup=warmup,
            samples=samples,
            trial_rows=trial_rows[label],
        )
        for label, index_name in states
    }


def compact_status(status: dict[str, Any]) -> dict[str, Any]:
    generation = status['generation']
    primary = generation['primary']
    workload = generation['workload_fold']
    return {
        'query_ready': status['query_ready'],
        'index_bytes': status['details']['index_bytes'],
        'generation': generation['generation'],
        'docs': generation['docs'],
        'segment_count': primary['segment_count'],
        'reachable_blocks': primary['reachable_blocks'],
        'reachable_ranges': primary['reachable_ranges'],
        'physical_blocks': primary['physical_blocks'],
        'workload_fold': {
            'candidate': workload['candidate'],
            'term_id': workload['term_id'],
            'query_heat': workload['query_heat'],
            'root_heat': workload['root_heat'],
            'extents': workload['extents'],
            'blocks': workload['blocks'],
            'postings': workload['postings'],
        },
    }


def compact_static_reference_status(
    status: dict[str, Any],
) -> dict[str, Any]:
    generation = status['generation']
    layout = generation.get('layout') or {}
    return {
        'query_ready': status['query_ready'],
        'index_bytes': status['details']['index_bytes'],
        'generation': generation.get('generation'),
        'docs': generation.get('docs'),
        'storage': layout.get('storage'),
    }


def ratio(
    numerator: dict[str, Any],
    denominator: dict[str, Any],
    metric: str,
) -> float:
    top = float(numerator['median_trial'][metric])
    bottom = float(denominator['median_trial'][metric])
    if bottom == 0.0:
        raise ZeroDivisionError(f'zero benchmark denominator for {metric}')
    return top / bottom


def metric_ratios(
    numerator: dict[str, Any],
    denominator: dict[str, Any],
) -> dict[str, float]:
    return {
        metric: ratio(numerator, denominator, metric)
        for metric in ('mean_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'qps')
    }


def normalized_metric_ratios(
    numerator: dict[str, Any],
    numerator_control: dict[str, Any],
    denominator: dict[str, Any],
    denominator_control: dict[str, Any],
) -> dict[str, float]:
    metrics = ('mean_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'qps')
    return {
        metric: (
            ratio(numerator, numerator_control, metric)
            / ratio(denominator, denominator_control, metric)
        )
        for metric in metrics
    }


def ensure_workload_candidate(
    connection: psycopg.Connection[Any],
    index_name: str,
    expected: list[tuple[int, float]],
    *,
    minimum_root_heat: int = 32,
) -> tuple[dict[str, Any], int]:
    additional_queries = 0
    while additional_queries <= 2048:
        status = fetch_status(connection, index_name)
        candidate = status['generation']['workload_fold']
        if (
            candidate['candidate'] is True
            and int(candidate['root_heat']) >= minimum_root_heat
        ):
            return status, additional_queries
        with connection.cursor() as cursor:
            for _ in range(32):
                rows = query_once(cursor, index_name)
                assert_rows_close(
                    expected,
                    rows,
                    label='workload-fold admission',
                )
                additional_queries += 1
    raise AssertionError(
        f'workload fold did not become eligible for {index_name}'
    )


def maintain_until_workload_action(
    connection: psycopg.Connection[Any],
    index_name: str,
    expected: list[tuple[int, float]],
    *,
    target_mode: str,
    target_reason: str,
    minimum_root_heat: int = 32,
) -> tuple[
    str,
    dict[str, str],
    list[str],
    int,
    list[dict[str, Any]],
]:
    actions: list[str] = []
    admission_queries = 0
    candidates: list[dict[str, Any]] = []

    for _ in range(64):
        status, observed_queries = ensure_workload_candidate(
            connection,
            index_name,
            expected,
            minimum_root_heat=minimum_root_heat,
        )
        admission_queries += observed_queries
        candidate = status['generation']['workload_fold']
        candidates.append({
            'generation': status['generation']['generation'],
            'term_id': candidate['term_id'],
            'query_heat': candidate['query_heat'],
            'root_heat': candidate['root_heat'],
            'extents': candidate['extents'],
            'postings': candidate['postings'],
            'impact_specialization_ready': (
                candidate['impact_specialization_ready']
            ),
            'hot_cache_republish_ready': (
                candidate['hot_cache_republish_ready']
            ),
        })
        result = try_maintain(connection, index_name)
        fields = maintenance_result_fields(result)
        actions.append(result)
        if (
            fields.get('maintained') == 'true'
            and fields.get('mode') == target_mode
            and fields.get('reason') == target_reason
        ):
            return (
                result,
                fields,
                actions,
                admission_queries,
                candidates,
            )
        if fields.get('maintained') == 'true':
            continue
        if fields.get('reason') in {'lock_busy', 'xid_horizon'}:
            time.sleep(0.05)
            continue
        raise AssertionError(
            f'maintenance stopped before {target_mode}: {actions}; '
            f'candidates={candidates}'
        )
    raise AssertionError(
        f'maintenance did not reach {target_mode}: {actions}; '
        f'candidates={candidates}'
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    extension_libdir = args.extension_libdir.expanduser().resolve()
    extension_control_dir = extension_control_root(
        args.extension_control_dir
    )
    if not extension_library_exists(extension_libdir):
        raise FileNotFoundError(
            f'evoke library is missing from {extension_libdir}'
        )

    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    system_libdir = pg_config_value(args.pg_bin, '--pkglibdir')
    system_sharedir = pg_config_value(args.pg_bin, '--sharedir')
    root = create_short_socket_root('evoke-query-states-')
    data_dir = root / 'data'
    socket_dir = root / 's'
    log_path = root / 'postgres.log'
    port = reserve_port()
    socket_dir.mkdir()
    started = False
    guarded_indexes: list[str] = []
    connection: psycopg.Connection[Any] | None = None

    try:
        run([
            str(initdb),
            '-D',
            str(data_dir),
            '-A',
            'trust',
            '-U',
            'postgres',
        ])
        configure_cluster(
            data_dir,
            socket_dir,
            port,
            extension_libdir=extension_libdir,
            system_libdir=system_libdir,
            extension_control_dir=extension_control_dir,
            system_sharedir=system_sharedir,
        )
        start_cluster(pg_ctl, data_dir, log_path)
        started = True
        connection = connect(socket_dir, port)
        with connection.cursor() as cursor:
            cursor.execute('CREATE EXTENSION evoke')
            cursor.execute('CREATE SCHEMA bench')
            cursor.execute(
                'CREATE FUNCTION bench.test_query_page_native_topk('
                'regclass, int4[], int4) RETURNS jsonb '
                "AS '$libdir/evoke', 'evoke_test_query_page_native_topk' "
                'LANGUAGE C STRICT'
            )
            cursor.execute(
                'CREATE TABLE bench.documents ('
                'id int PRIMARY KEY, tokens int4[] NOT NULL)'
            )

        insert_documents(connection, 1, args.base_docs)
        create_convergent_index(connection, 'fragmented_idx')
        fragmented_index = 'bench.fragmented_idx'
        acquire_maintenance_guard(connection, fragmented_index)
        guarded_indexes.append(fragmented_index)
        if not maintenance_guard_is_held(
            connection,
            fragmented_index,
        ):
            raise AssertionError('benchmark maintenance guard was lost')

        maintenance: list[str] = []
        next_id = args.base_docs + 1
        with connection.cursor() as cursor:
            cursor.execute('SET evoke.test_force_structural_term_fold = true')
        try:
            for _ in range(args.delta_batches):
                maintenance.extend(
                    append_sealed_batch(
                        connection,
                        fragmented_index,
                        next_id,
                        args.delta_docs,
                    )
                )
                next_id += args.delta_docs
        finally:
            with connection.cursor() as cursor:
                cursor.execute(
                    'RESET evoke.test_force_structural_term_fold'
                )

        fragmented_status = fetch_status(
            connection,
            fragmented_index,
        )
        minimum_expected_segments = 2
        total_docs = (
            args.base_docs + args.delta_docs * args.delta_batches
        )
        fold_minimum_root_heat = 32
        observed_segments = fragmented_status[
            'generation'
        ]['primary']['segment_count']
        required_surface_reduction = observed_segments - 1
        if (
            observed_segments < minimum_expected_segments
        ):
            raise AssertionError(
                'fragmented benchmark index has too few segments: '
                f'{compact_status(fragmented_status)}'
            )

        create_convergent_index(connection, 'static_idx')
        static_index = 'bench.static_idx'
        acquire_maintenance_guard(connection, static_index)
        guarded_indexes.append(static_index)
        static_status = fetch_status(connection, static_index)
        static_primary = static_status['generation']['primary']
        static_is_single_surface = (
            static_primary['segment_count'] <= 1
            and static_primary['reachable_ranges'] == 1
        )
        if not static_is_single_surface:
            raise AssertionError(
                'static benchmark index is not a single-surface rebuild: '
                f'{compact_status(static_status)}'
            )

        create_static_reference_index(connection, 'static_reference_idx')
        static_reference_index = 'bench.static_reference_idx'
        acquire_maintenance_guard(connection, static_reference_index)
        guarded_indexes.append(static_reference_index)
        static_reference_status = fetch_status(
            connection,
            static_reference_index,
        )
        expected = fetch_ids(
            connection,
            static_reference_index,
            [QUERY_TERM],
        )
        static_rows = fetch_ids(connection, static_index, [QUERY_TERM])
        assert_rows_close(
            expected,
            static_rows,
            label='static-reference/static benchmark baseline',
        )
        fragmented_rows = fetch_ids(
            connection,
            fragmented_index,
            [QUERY_TERM],
        )
        assert_rows_close(
            expected,
            fragmented_rows,
            label='static-fragmented benchmark baseline',
        )
        pre_fold_benchmarks = benchmark_pair(
            connection,
            (
                ('static', static_index),
                ('fragmented', fragmented_index),
            ),
            expected,
            warmup=args.warmup,
            samples=args.samples,
            trials=args.trials,
        )
        static_benchmark = pre_fold_benchmarks['static']
        fragmented_benchmark = pre_fold_benchmarks['fragmented']

        ordered_expected = fetch_ids(
            connection,
            static_reference_index,
            list(ORDERED_QUERY_TERMS),
        )
        ordered_static_rows = fetch_ids(
            connection,
            static_index,
            list(ORDERED_QUERY_TERMS),
        )
        assert_rows_close(
            ordered_expected,
            ordered_static_rows,
            label='ordered static-reference/static benchmark baseline',
        )
        ordered_fragmented_rows = fetch_ids(
            connection,
            fragmented_index,
            list(ORDERED_QUERY_TERMS),
        )
        assert_rows_close(
            ordered_expected,
            ordered_fragmented_rows,
            label='ordered static-fragmented benchmark baseline',
        )
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT bench.test_query_page_native_topk('
                '%s::regclass, %s::int4[], %s)',
                (static_index, list(ORDERED_QUERY_TERMS), QUERY_LIMIT),
            )
            ordered_static_stats_row = cursor.fetchone()
            cursor.execute(
                'SELECT bench.test_query_page_native_topk('
                '%s::regclass, %s::int4[], %s)',
                (fragmented_index, list(ORDERED_QUERY_TERMS), QUERY_LIMIT),
            )
            ordered_fragmented_stats_row = cursor.fetchone()
        if (
            ordered_static_stats_row is None
            or not isinstance(ordered_static_stats_row[0], dict)
            or ordered_fragmented_stats_row is None
            or not isinstance(ordered_fragmented_stats_row[0], dict)
        ):
            raise AssertionError('invalid ordered query path statistics')
        ordered_query_stats = {
            'static': ordered_static_stats_row[0],
            'fragmented': ordered_fragmented_stats_row[0],
        }
        ordered_pre_fold_benchmarks = benchmark_pair(
            connection,
            (
                ('static', static_index),
                ('fragmented', fragmented_index),
            ),
            ordered_expected,
            warmup=args.warmup,
            samples=args.samples,
            trials=args.trials,
            query_terms=ORDERED_QUERY_TERMS,
        )
        ordered_static_benchmark = ordered_pre_fold_benchmarks['static']
        ordered_fragmented_benchmark = (
            ordered_pre_fold_benchmarks['fragmented']
        )

        candidate_status, admission_queries = ensure_workload_candidate(
            connection,
            fragmented_index,
            expected,
            minimum_root_heat=fold_minimum_root_heat,
        )
        candidate = candidate_status['generation']['workload_fold']
        if candidate['extents'] < observed_segments:
            raise AssertionError(
                'workload candidate did not observe every segment: '
                f'{candidate}'
            )
        (
            fold_result,
            fold_fields,
            fold_actions,
            fold_extra_admission_queries,
            fold_candidates,
        ) = maintain_until_workload_action(
            connection,
            fragmented_index,
            expected,
            target_mode='term_workload_fold',
            target_reason='workload_heat',
            minimum_root_heat=fold_minimum_root_heat,
        )
        admission_queries += fold_extra_admission_queries
        if (
            fold_fields.get('maintained') != 'true'
            or fold_fields.get('mode') != 'term_workload_fold'
            or fold_fields.get('reason') != 'workload_heat'
            or int(fold_fields.get('surface_reduction', '0'))
            < required_surface_reduction
        ):
            raise AssertionError(
                f'workload fold was not published: {fold_result}'
            )

        post_fold_status = fetch_status(
            connection,
            fragmented_index,
        )
        folded_rows = fetch_ids(
            connection,
            fragmented_index,
            [QUERY_TERM],
        )
        assert_rows_close(
            expected,
            folded_rows,
            label='static-workload-fold benchmark baseline',
        )
        post_fold_benchmarks = benchmark_pair(
            connection,
            (
                ('static', static_index),
                ('folded', fragmented_index),
            ),
            expected,
            warmup=args.warmup,
            samples=args.samples,
            trials=args.trials,
        )
        static_post_fold_benchmark = post_fold_benchmarks['static']
        folded_benchmark = post_fold_benchmarks['folded']
        folded_final_status = fetch_status(
            connection,
            fragmented_index,
        )
        impact_candidate_status, impact_admission_queries = (
            ensure_workload_candidate(
                connection,
                fragmented_index,
                expected,
            )
        )
        impact_candidate = impact_candidate_status[
            'generation'
        ]['workload_fold']
        if (
            impact_candidate['impact_specialization_ready'] is not True
        ):
            raise AssertionError(
                'folded term was not eligible for impact specialization: '
                f'{impact_candidate}'
            )
        (
            impact_result,
            impact_fields,
            impact_actions,
            impact_extra_admission_queries,
            impact_candidates,
        ) = maintain_until_workload_action(
            connection,
            fragmented_index,
            expected,
            target_mode='term_impact_specialization',
            target_reason='impact_specialization',
        )
        impact_admission_queries += impact_extra_admission_queries
        if (
            impact_fields.get('maintained') != 'true'
            or impact_fields.get('mode')
            != 'term_impact_specialization'
            or impact_fields.get('reason') != 'impact_specialization'
        ):
            raise AssertionError(
                'impact specialization was not published: '
                f'{impact_result}'
            )
        impact_cache_state = fetch_generation_cache_state(
            connection,
            fragmented_index,
        )
        if 'shared_hot_fold_current=true' not in impact_cache_state:
            raise AssertionError(
                'impact specialization did not publish its exact-root '
                f'shared hot fold: {impact_cache_state}'
            )
        impact_status = fetch_status(connection, fragmented_index)
        impact_rows = fetch_ids(
            connection,
            fragmented_index,
            [QUERY_TERM],
        )
        assert_rows_close(
            expected,
            impact_rows,
            label='static-reference/impact benchmark baseline',
        )
        impact_benchmarks = benchmark_pair(
            connection,
            (
                ('static_reference', static_reference_index),
                ('impact_specialized', fragmented_index),
            ),
            expected,
            warmup=args.warmup,
            samples=args.samples,
            trials=args.trials,
        )
        static_reference_benchmark = impact_benchmarks['static_reference']
        impact_benchmark = impact_benchmarks['impact_specialized']
        final_status = fetch_status(connection, fragmented_index)
        fragmented_over_static = metric_ratios(
            fragmented_benchmark,
            static_benchmark,
        )
        ordered_fragmented_over_static = metric_ratios(
            ordered_fragmented_benchmark,
            ordered_static_benchmark,
        )
        folded_over_static = metric_ratios(
            folded_benchmark,
            static_post_fold_benchmark,
        )
        folded_over_fragmented_raw = metric_ratios(
            folded_benchmark,
            fragmented_benchmark,
        )
        folded_over_fragmented_normalized = normalized_metric_ratios(
            folded_benchmark,
            static_post_fold_benchmark,
            fragmented_benchmark,
            static_benchmark,
        )
        impact_over_static_reference = metric_ratios(
            impact_benchmark,
            static_reference_benchmark,
        )
        fold_input_bytes = int(fold_fields['input_bytes'])
        fold_surface_reduction = int(fold_fields['surface_reduction'])
        byte_only_denominator = (
            fold_surface_reduction * POSTING_HEAT_BENEFIT_BYTES
        )
        byte_only_required_root_heat = (
            fold_input_bytes + byte_only_denominator - 1
        ) // byte_only_denominator
        fold_root_heat = int(fold_fields['root_heat'])
        unpruned_blocks = max(
            int(candidate['blocks_considered'])
            - int(candidate['blocks_skipped']),
            0,
        )
        observed_page_bytes = unpruned_blocks * POSTGRES_BLOCK_BYTES
        byte_only_cost_covered = (
            fold_root_heat >= byte_only_required_root_heat
        )
        observed_work_cost_covered = (
            candidate['impact_specialization_ready'] is True
            and observed_page_bytes >= fold_input_bytes
        )

        gates = {
            'static_is_single_surface': static_is_single_surface,
            'fragmented_has_expected_segments': (
                observed_segments >= minimum_expected_segments
            ),
            'static_fragmented_rows_and_scores_match': True,
            'ordered_static_fragmented_rows_and_scores_match': True,
            'static_reference_static_rows_and_scores_match': True,
            'workload_candidate_observed_all_extents': (
                candidate['extents'] >= observed_segments
            ),
            'workload_fold_reduced_surfaces': (
                int(fold_fields['surface_reduction'])
                >= required_surface_reduction
            ),
            'workload_fold_has_valid_cost_basis': (
                byte_only_cost_covered or observed_work_cost_covered
            ),
            'static_folded_rows_and_scores_match': True,
            'static_reference_impact_rows_and_scores_match': True,
            'impact_specialization_was_admitted': (
                impact_candidate['impact_specialization_ready'] is True
            ),
            'impact_hot_fold_is_exact_root_resident': (
                'shared_hot_fold_current=true' in impact_cache_state
            ),
            'maintenance_guard_remained_held': (
                all(
                    maintenance_guard_is_held(connection, guarded_index)
                    for guarded_index in guarded_indexes
                )
            ),
            'ordered_fragmented_mean_is_static_equivalent': (
                ordered_fragmented_over_static['mean_ms']
                <= LATENCY_EQUIVALENCE_LIMIT
            ),
            'ordered_fragmented_p50_is_static_equivalent': (
                ordered_fragmented_over_static['p50_ms']
                <= LATENCY_EQUIVALENCE_LIMIT
            ),
            'ordered_fragmented_p95_is_static_equivalent': (
                ordered_fragmented_over_static['p95_ms']
                <= LATENCY_EQUIVALENCE_LIMIT
            ),
            'ordered_fragmented_p99_is_static_equivalent': (
                ordered_fragmented_over_static['p99_ms']
                <= P99_EQUIVALENCE_LIMIT
            ),
            'ordered_fragmented_qps_is_static_equivalent': (
                ordered_fragmented_over_static['qps']
                >= QPS_EQUIVALENCE_FLOOR
            ),
            'folded_mean_is_static_equivalent': (
                folded_over_static['mean_ms']
                <= LATENCY_EQUIVALENCE_LIMIT
            ),
            'folded_p50_is_static_equivalent': (
                folded_over_static['p50_ms']
                <= LATENCY_EQUIVALENCE_LIMIT
            ),
            'folded_p95_is_static_equivalent': (
                folded_over_static['p95_ms']
                <= LATENCY_EQUIVALENCE_LIMIT
            ),
            'folded_p99_is_static_equivalent': (
                folded_over_static['p99_ms']
                <= P99_EQUIVALENCE_LIMIT
            ),
            'folded_qps_is_static_equivalent': (
                folded_over_static['qps']
                >= QPS_EQUIVALENCE_FLOOR
            ),
            'impact_mean_is_static_reference_equivalent': (
                impact_over_static_reference['mean_ms']
                <= LATENCY_EQUIVALENCE_LIMIT
            ),
            'impact_p50_is_static_reference_equivalent': (
                impact_over_static_reference['p50_ms']
                <= LATENCY_EQUIVALENCE_LIMIT
            ),
            'impact_p95_is_static_reference_equivalent': (
                impact_over_static_reference['p95_ms']
                <= LATENCY_EQUIVALENCE_LIMIT
            ),
            'impact_p99_is_static_reference_equivalent': (
                impact_over_static_reference['p99_ms']
                <= P99_EQUIVALENCE_LIMIT
            ),
            'impact_qps_is_static_reference_equivalent': (
                impact_over_static_reference['qps']
                >= QPS_EQUIVALENCE_FLOOR
            ),
        }
        return {
            'benchmark': 'evoke_convergent_query_states_v3',
            'api': 'evoke_query_ids',
            'source': source_identity(),
            'extension_library': extension_library_identity(
                extension_libdir
            ),
            'host': host_snapshot(),
            'postgresql': command_output([
                str(args.pg_bin / 'pg_config'),
                '--version',
            ]),
            'configuration': {
                'base_docs': args.base_docs,
                'delta_docs': args.delta_docs,
                'delta_batches': args.delta_batches,
                'minimum_expected_segments': minimum_expected_segments,
                'required_surface_reduction': required_surface_reduction,
                'observed_fragmented_segments': observed_segments,
                'total_docs': total_docs,
                'fold_minimum_root_heat': fold_minimum_root_heat,
                'byte_only_required_root_heat': (
                    byte_only_required_root_heat
                ),
                'byte_only_cost_covered': byte_only_cost_covered,
                'observed_page_bytes': observed_page_bytes,
                'observed_work_cost_covered': observed_work_cost_covered,
                'tokens_per_document': 4,
                'query': [QUERY_TERM],
                'ordered_query': list(ORDERED_QUERY_TERMS),
                'top_k': QUERY_LIMIT,
                'warmup': args.warmup,
                'samples': args.samples,
                'trials': args.trials,
                'performance_equivalence': {
                    'mean_p50_p95_latency_ratio_max': (
                        LATENCY_EQUIVALENCE_LIMIT
                    ),
                    'p99_latency_ratio_max': P99_EQUIVALENCE_LIMIT,
                    'qps_ratio_min': QPS_EQUIVALENCE_FLOOR,
                },
            },
            'gates': gates,
            'all_gates_passed': all(gates.values()),
            'states': {
                'static_reference_impact': {
                    'status': compact_static_reference_status(
                        static_reference_status
                    ),
                    'performance': static_reference_benchmark,
                },
                'static': {
                    'status': compact_status(static_status),
                    'performance_before_fold': static_benchmark,
                    'performance_after_fold': (
                        static_post_fold_benchmark
                    ),
                    'ordered_performance': ordered_static_benchmark,
                },
                'fragmented': {
                    'status': compact_status(fragmented_status),
                    'performance': fragmented_benchmark,
                    'ordered_performance': ordered_fragmented_benchmark,
                },
                'workload_folded': {
                    'status_after_publication': compact_status(
                        post_fold_status
                    ),
                    'status_after_benchmark': compact_status(
                        folded_final_status
                    ),
                    'performance': folded_benchmark,
                },
                'impact_specialized': {
                    'status_after_publication': compact_status(
                        impact_status
                    ),
                    'generation_cache_state': impact_cache_state,
                    'status_after_benchmark': compact_status(final_status),
                    'performance': impact_benchmark,
                },
            },
            'ratios': {
                'fragmented_over_static': fragmented_over_static,
                'ordered_fragmented_over_static': (
                    ordered_fragmented_over_static
                ),
                'folded_over_static': folded_over_static,
                'folded_over_fragmented_raw': (
                    folded_over_fragmented_raw
                ),
                'folded_over_fragmented_normalized_to_static': (
                    folded_over_fragmented_normalized
                ),
                'impact_over_static_reference': impact_over_static_reference,
            },
            'exactness': {
                'result_count': len(expected),
                'result_id_sha256': result_id_sha256(expected),
                'fragmented_max_score_abs_diff': max_score_difference(
                    expected,
                    fragmented_rows,
                ),
                'ordered_fragmented_max_score_abs_diff': (
                    max_score_difference(
                        ordered_expected,
                        ordered_fragmented_rows,
                    )
                ),
                'ordered_query_path_stats': ordered_query_stats,
                'folded_max_score_abs_diff': max_score_difference(
                    expected,
                    folded_rows,
                ),
                'impact_max_score_abs_diff': max_score_difference(
                    expected,
                    impact_rows,
                ),
            },
            'maintenance': {
                'setup': maintenance,
                'admission_queries': admission_queries,
                'candidate': candidate,
                'fold_actions': fold_actions,
                'fold_candidates': fold_candidates,
                'fold_result': fold_result,
                'fold_fields': fold_fields,
                'impact_admission_queries': impact_admission_queries,
                'impact_candidate': impact_candidate,
                'impact_actions': impact_actions,
                'impact_candidates': impact_candidates,
                'impact_result': impact_result,
                'impact_fields': impact_fields,
            },
        }
    except Exception:
        if log_path.is_file():
            print(log_path.read_text(encoding='utf-8'))
        raise
    finally:
        if connection is not None:
            for guarded_index in reversed(guarded_indexes):
                if maintenance_guard_is_held(connection, guarded_index):
                    release_maintenance_guard(connection, guarded_index)
            connection.close()
        if started:
            stop_cluster(pg_ctl, data_dir)
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    if not report['all_gates_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
