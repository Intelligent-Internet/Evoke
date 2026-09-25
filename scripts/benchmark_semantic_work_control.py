#!/usr/bin/env python3
"""Calibrate Evoke semantic work and concurrent latency on real queries."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import psycopg

from benchmark_page_native_block_cost import DEFAULT_QUERIES, expand_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--lexical-dims', type=int, required=True)
    parser.add_argument('--semantic-dims', type=int, required=True)
    parser.add_argument('--field-count', type=int, default=1)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--query', action='append', dest='queries')
    parser.add_argument('--queries-jsonl', type=Path)
    parser.add_argument('--query-limit', type=int, default=50)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--skip-serial', action='store_true')
    parser.add_argument(
        '--max-df-ratio',
        type=float,
        default=1.0,
        help='Hidden query-time DF control used only by this benchmark.',
    )
    parser.add_argument(
        '--semantic-work-target-postings',
        type=int,
        default=0,
        help='Activate semantic DF pruning only above this planned work.',
    )
    parser.add_argument(
        '--allow-approximate-results',
        action='store_true',
        help='Record top-k mismatch instead of failing the benchmark.',
    )
    parser.add_argument('--disable-semantic-bmp', action='store_true')
    parser.add_argument('--force-semantic-bmp', action='store_true')
    parser.add_argument('--bmp-super-batch', type=int, default=8)
    parser.add_argument('--semantic-error-budget', type=float, default=0.0)
    parser.add_argument('--semantic-min-support', type=float, default=0.0)
    parser.add_argument(
        '--concurrency',
        type=int,
        action='append',
        dest='concurrency_levels',
    )
    parser.add_argument(
        '--admission-limit',
        type=int,
        action='append',
        dest='admission_limits',
    )
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def bind_probes(cursor: psycopg.Cursor[Any]) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION pg_temp.evoke_query_block_cost(
            regclass,
            int4[],
            real[],
            int4
        ) RETURNS jsonb
        AS '$libdir/evoke', 'evoke_test_query_block_cost'
        LANGUAGE C STRICT
        """
    )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION pg_temp.evoke_query_topk(
            regclass,
            int4[],
            real[],
            int4
        ) RETURNS jsonb
        AS '$libdir/evoke', 'evoke_test_query_page_native_topk'
        LANGUAGE C STRICT
        """
    )


def read_query_texts(path: Path, limit: int) -> list[str]:
    queries: list[str] = []

    with path.open(encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get('text') or row.get('query')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('query JSONL row has no text/query string')
            queries.append(text)
            if len(queries) >= limit:
                break
    return queries


def percentile(values: Iterable[float], ratio: float) -> float:
    ordered = sorted(values)

    if not ordered:
        return 0.0
    position = math.ceil(ratio * len(ordered)) - 1
    return ordered[max(0, min(position, len(ordered) - 1))]


def encode_queries(
    connection: psycopg.Connection[Any],
    index_name: str,
    query_texts: list[str],
    lexical_dims: int,
    semantic_dims: int,
    field_count: int,
) -> list[dict[str, Any]]:
    encoded_queries: list[dict[str, Any]] = []

    with connection.cursor() as cursor:
        for query_text in query_texts:
            cursor.execute(
                'SELECT evoke_encode_text_internal(%s::regclass, %s)',
                (index_name, query_text),
            )
            encoded = cursor.fetchone()[0]
            query_ids, query_weights = expand_query(
                encoded,
                lexical_dims,
                semantic_dims,
                field_count,
            )
            encoded_queries.append(
                {
                    'text': query_text,
                    'ids': query_ids,
                    'weights': query_weights,
                    'atom_count': len(encoded['atoms']),
                }
            )
    return encoded_queries


def call_probe(
    cursor: psycopg.Cursor[Any],
    function_name: str,
    index_name: str,
    query: dict[str, Any],
    k: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    cursor.execute(
        f"""
        SELECT pg_temp.{function_name}(
            %s::regclass,
            %s::int4[],
            %s::real[],
            %s
        )
        """,
        (index_name, query['ids'], query['weights'], k),
    )
    row = cursor.fetchone()[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(row, dict):
        raise RuntimeError(f'{function_name} returned no JSON object')
    return row, elapsed_ms


def call_public_search(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    query_text: str,
    k: int,
) -> float:
    started = time.perf_counter()
    cursor.execute(
        """
        SELECT count(*), sum(score)
        FROM evoke_query(%s::regclass, %s, %s)
        """,
        (index_name, query_text, k),
    )
    cursor.fetchone()
    return (time.perf_counter() - started) * 1000.0


def collect_serial_baseline(
    connection: psycopg.Connection[Any],
    index_name: str,
    queries: list[dict[str, Any]],
    k: int,
    repetitions: int,
    require_exact: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    with connection.cursor() as cursor:
        bind_probes(cursor)
        for query in queries:
            audit, audit_ms = call_probe(
                cursor,
                'evoke_query_block_cost',
                index_name,
                query,
                k,
            )
            samples: list[float] = []
            stats: dict[str, Any] | None = None
            probe_samples: list[float] = []
            for _ in range(repetitions):
                stats, probe_ms = call_probe(
                    cursor,
                    'evoke_query_topk',
                    index_name,
                    query,
                    k,
                )
                probe_samples.append(probe_ms)
                samples.append(
                    call_public_search(
                        cursor,
                        index_name,
                        query['text'],
                        k,
                    )
                )
            assert stats is not None
            if require_exact and not bool(stats.get('matched', False)):
                raise RuntimeError(
                    f"page-native top-k mismatch for query: {query['text']}"
                )
            level16 = next(
                level
                for level in audit['levels']
                if int(level['block_size']) == 16
            )
            bmp_postings = int(
                stats.get('semantic_bmp_postings_examined', 0)
            )
            fallback_postings = int(stats.get('postings_examined', 0))
            actual_postings = bmp_postings + fallback_postings
            query_postings = int(audit['query_postings'])
            results.append(
                {
                    'query_text': query['text'],
                    'atom_count': query['atom_count'],
                    'query_postings': query_postings,
                    'b16_oracle_competitive_postings': int(
                        level16['competitive_postings']
                    ),
                    'b16_oracle_posting_fraction': float(
                        level16['posting_fraction']
                    ),
                    'actual_semantic_postings': actual_postings,
                    'bmp_attempt_postings': bmp_postings,
                    'bmp_super_ref_reads': int(
                        stats.get('semantic_bmp_super_ref_reads', 0)
                    ),
                    'bmp_ref_reads': int(
                        stats.get('semantic_bmp_ref_reads', 0)
                    ),
                    'bmp_record_reads': int(
                        stats.get('semantic_bmp_record_reads', 0)
                    ),
                    'blocks_considered': int(
                        stats.get('blocks_considered', 0)
                    ),
                    'blocks_scored': int(stats.get('blocks_scored', 0)),
                    'fallback_or_primary_postings': fallback_postings,
                    'documents_examined': int(
                        stats.get('documents_examined', 0)
                    ),
                    'positive_document_count': int(
                        stats.get('positive_document_count', 0)
                    ),
                    'query_term_count': int(
                        stats.get('query_term_count', 0)
                    ),
                    'query_run_count': int(
                        stats.get('query_run_count', 0)
                    ),
                    'term_at_a_time_score_bytes': int(
                        stats.get('term_at_a_time_score_bytes', 0)
                    ),
                    'term_at_a_time_document_length_bytes': int(
                        stats.get(
                            'term_at_a_time_document_length_bytes',
                            0,
                        )
                    ),
                    'actual_posting_fraction': (
                        actual_postings / query_postings
                        if query_postings
                        else 0.0
                    ),
                    'semantic_bmp_fallback': bool(
                        stats.get('semantic_bmp_fallback', False)
                    ),
                    'semantic_bmp_admission_skipped': bool(
                        stats.get(
                            'semantic_bmp_admission_skipped',
                            False,
                        )
                    ),
                    'semantic_bmp_query_path': bool(
                        stats.get('semantic_bmp_query_path', False)
                    ),
                    'matched_exact_topk': bool(stats.get('matched', False)),
                    'df_pruned_terms': int(
                        stats.get('query_df_pruned_term_count', 0)
                    ),
                    'df_pruned_postings': int(
                        stats.get('query_df_pruned_postings', 0)
                    ),
                    'audit_ms': audit_ms,
                    'latency_ms': {
                        'p50': statistics.median(samples),
                        'p95': percentile(samples, 0.95),
                        'samples': samples,
                    },
                    'probe_latency_ms': {
                        'p50': statistics.median(probe_samples),
                        'p95': percentile(probe_samples, 0.95),
                        'samples': probe_samples,
                    },
                }
            )
    return results


def concurrent_worker(
    dsn: str,
    index_name: str,
    queries: list[dict[str, Any]],
    k: int,
    barrier: threading.Barrier,
    admission: threading.Semaphore | None,
    bmp_super_batch: int,
    semantic_error_budget: float,
    semantic_min_support: float,
    max_df_ratio: float,
    semantic_work_target_postings: int,
    disable_semantic_bmp: bool,
    force_semantic_bmp: bool,
) -> dict[str, list[float]]:
    execution_samples: list[float] = []
    queue_samples: list[float] = []
    total_samples: list[float] = []

    with psycopg.connect(dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            bind_probes(cursor)
            cursor.execute(
                'SELECT set_config(%s, %s, false)',
                (
                    'evoke.test_query_max_df_ratio',
                    format(max_df_ratio, '.17g'),
                ),
            )
            cursor.execute(
                'SELECT set_config(%s, %s, false)',
                (
                    'evoke.test_query_semantic_work_target_postings',
                    str(semantic_work_target_postings),
                ),
            )
            cursor.execute(
                'SELECT set_config(%s, %s, false)',
                (
                    'evoke.test_disable_semantic_bmp',
                    'on' if disable_semantic_bmp else 'off',
                ),
            )
            cursor.execute(
                'SELECT set_config(%s, %s, false)',
                (
                    'evoke.test_force_semantic_bmp',
                    'on' if force_semantic_bmp else 'off',
                ),
            )
            cursor.execute(
                'SELECT set_config(%s, %s, false)',
                (
                    'evoke.test_query_semantic_bmp_super_batch',
                    str(bmp_super_batch),
                ),
            )
            cursor.execute(
                'SELECT set_config(%s, %s, false)',
                (
                    'evoke.test_query_semantic_error_budget_ratio',
                    format(semantic_error_budget, '.17g'),
                ),
            )
            cursor.execute(
                'SELECT set_config(%s, %s, false)',
                (
                    'evoke.test_query_semantic_min_support_ratio',
                    format(semantic_min_support, '.17g'),
                ),
            )
            barrier.wait()
            for query in queries:
                arrived = time.perf_counter()
                if admission is not None:
                    admission.acquire()
                admitted = time.perf_counter()
                try:
                    elapsed_ms = call_public_search(
                        cursor,
                        index_name,
                        query['text'],
                        k,
                    )
                finally:
                    if admission is not None:
                        admission.release()
                completed = time.perf_counter()
                execution_samples.append(elapsed_ms)
                queue_samples.append((admitted - arrived) * 1000.0)
                total_samples.append((completed - arrived) * 1000.0)
    return {
        'execution': execution_samples,
        'queue': queue_samples,
        'total': total_samples,
    }


def run_concurrent_level(
    dsn: str,
    index_name: str,
    queries: list[dict[str, Any]],
    k: int,
    repetitions: int,
    requested_workers: int,
    admission_limit: int | None = None,
    bmp_super_batch: int = 8,
    semantic_error_budget: float = 0.0,
    semantic_min_support: float = 0.0,
    max_df_ratio: float = 1.0,
    semantic_work_target_postings: int = 0,
    disable_semantic_bmp: bool = False,
    force_semantic_bmp: bool = False,
) -> dict[str, Any]:
    tasks = queries * repetitions
    worker_count = min(requested_workers, len(tasks))
    chunks = [tasks[index::worker_count] for index in range(worker_count)]
    barrier = threading.Barrier(worker_count)
    admission = (
        threading.Semaphore(admission_limit)
        if admission_limit is not None and admission_limit < worker_count
        else None
    )
    started = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:
        futures = [
            executor.submit(
                concurrent_worker,
                dsn,
                index_name,
                chunk,
                k,
                barrier,
                admission,
                bmp_super_batch,
                semantic_error_budget,
                semantic_min_support,
                max_df_ratio,
                semantic_work_target_postings,
                disable_semantic_bmp,
                force_semantic_bmp,
            )
            for chunk in chunks
        ]
        worker_results = [future.result() for future in futures]
        execution_samples = [
            sample
            for worker_result in worker_results
            for sample in worker_result['execution']
        ]
        queue_samples = [
            sample
            for worker_result in worker_results
            for sample in worker_result['queue']
        ]
        total_samples = [
            sample
            for worker_result in worker_results
            for sample in worker_result['total']
        ]
    elapsed = time.perf_counter() - started
    return {
        'requested_concurrency': requested_workers,
        'actual_workers': worker_count,
        'admission_limit': admission_limit or worker_count,
        'query_count': len(execution_samples),
        'elapsed_seconds': elapsed,
        'throughput_qps': (
            len(execution_samples) / elapsed if elapsed else 0.0
        ),
        'execution_latency_ms': {
            'p50': statistics.median(execution_samples),
            'p95': percentile(execution_samples, 0.95),
            'p99': percentile(execution_samples, 0.99),
            'max': max(execution_samples, default=0.0),
        },
        'queue_latency_ms': {
            'p50': statistics.median(queue_samples),
            'p95': percentile(queue_samples, 0.95),
            'p99': percentile(queue_samples, 0.99),
            'max': max(queue_samples, default=0.0),
        },
        'total_latency_ms': {
            'p50': statistics.median(total_samples),
            'p95': percentile(total_samples, 0.95),
            'p99': percentile(total_samples, 0.99),
            'max': max(total_samples, default=0.0),
        },
    }


def main() -> int:
    args = parse_args()
    if args.query_limit <= 0 or args.repetitions <= 0 or args.k <= 0:
        raise SystemExit('query limit, repetitions and k must be positive')
    if not 1 <= args.bmp_super_batch <= 8:
        raise SystemExit('BMP super batch must be between 1 and 8')
    if not 0.0 <= args.semantic_error_budget <= 1.0:
        raise SystemExit('semantic error budget must be between zero and one')
    if not 0.0 <= args.semantic_min_support <= 1.0:
        raise SystemExit('semantic minimum support must be between zero and one')
    if not 0.0 < args.max_df_ratio <= 1.0:
        raise SystemExit(
            'maximum DF ratio must be greater than zero and at most one'
        )
    if args.semantic_work_target_postings < 0:
        raise SystemExit('semantic work target postings must be non-negative')
    if args.queries_jsonl is not None:
        query_texts = read_query_texts(
            args.queries_jsonl,
            args.query_limit,
        )
    else:
        query_texts = list(args.queries or DEFAULT_QUERIES)
    if not query_texts:
        raise SystemExit('no queries selected')

    with psycopg.connect(args.dsn, autocommit=True) as connection:
        connection.execute(
            'SELECT set_config(%s, %s, false)',
            (
                'evoke.test_query_max_df_ratio',
                format(args.max_df_ratio, '.17g'),
            ),
        )
        connection.execute(
            'SELECT set_config(%s, %s, false)',
            (
                'evoke.test_query_semantic_work_target_postings',
                str(args.semantic_work_target_postings),
            ),
        )
        connection.execute(
            'SELECT set_config(%s, %s, false)',
            (
                'evoke.test_disable_semantic_bmp',
                'on' if args.disable_semantic_bmp else 'off',
            ),
        )
        connection.execute(
            'SELECT set_config(%s, %s, false)',
            (
                'evoke.test_force_semantic_bmp',
                'on' if args.force_semantic_bmp else 'off',
            ),
        )
        connection.execute(
            'SELECT set_config(%s, %s, false)',
            (
                'evoke.test_query_semantic_bmp_super_batch',
                str(args.bmp_super_batch),
            ),
        )
        connection.execute(
            'SELECT set_config(%s, %s, false)',
            (
                'evoke.test_query_semantic_error_budget_ratio',
                format(args.semantic_error_budget, '.17g'),
            ),
        )
        connection.execute(
            'SELECT set_config(%s, %s, false)',
            (
                'evoke.test_query_semantic_min_support_ratio',
                format(args.semantic_min_support, '.17g'),
            ),
        )
        queries = encode_queries(
            connection,
            args.index,
            query_texts,
            args.lexical_dims,
            args.semantic_dims,
            args.field_count,
        )
        serial = [] if args.skip_serial else collect_serial_baseline(
            connection,
            args.index,
            queries,
            args.k,
            args.repetitions,
            not args.allow_approximate_results and
            args.semantic_error_budget == 0.0 and
            args.max_df_ratio >= 1.0,
        )

    concurrency_levels = args.concurrency_levels or [1, 8, 32]
    admission_limits = args.admission_limits or [None]
    concurrent_results = [
        run_concurrent_level(
            args.dsn,
            args.index,
            queries,
            args.k,
            args.repetitions,
            level,
            admission_limit,
            args.bmp_super_batch,
            args.semantic_error_budget,
            args.semantic_min_support,
            args.max_df_ratio,
            args.semantic_work_target_postings,
            args.disable_semantic_bmp,
            args.force_semantic_bmp,
        )
        for level in concurrency_levels
        for admission_limit in admission_limits
        if admission_limit is None or admission_limit > 0
    ]
    result = {
        'index_name': args.index,
        'k': args.k,
        'bmp_super_batch': args.bmp_super_batch,
        'disable_semantic_bmp': args.disable_semantic_bmp,
        'force_semantic_bmp': args.force_semantic_bmp,
        'semantic_error_budget': args.semantic_error_budget,
        'semantic_min_support': args.semantic_min_support,
        'max_df_ratio': args.max_df_ratio,
        'semantic_work_target_postings': (
            args.semantic_work_target_postings
        ),
        'query_count': len(queries),
        'repetitions': args.repetitions,
        'serial': serial,
        'concurrency': concurrent_results,
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output + '\n', encoding='utf-8')
    else:
        print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
