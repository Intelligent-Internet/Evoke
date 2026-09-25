#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import conninfo, sql


DEFAULT_OUTPUT = Path(
    'docs/performance/data/diagnostics/sql-field-helpers-2026-03-26.json'
)
DEFAULT_QUERIES = ['bird', 'cat', 'bird OR cat', 'policy']
BENCHMARK_DB = 'evoke_sql_field_helpers'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark field-aware SQL helper surfaces.'
    )
    parser.add_argument(
        '--dsn',
        default=os.environ.get('EVOKE_BENCH_DSN', 'dbname=postgres'),
        help='PostgreSQL DSN used for the local benchmark session.',
    )
    parser.add_argument(
        '--docs',
        type=int,
        default=4000,
        help='Synthetic document count to generate.',
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Top-k returned by each benchmarked query.',
    )
    parser.add_argument(
        '--candidate-k',
        type=int,
        default=20,
        help='Candidate top-k used before result fusion.',
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=30,
        help='Repeats per query and per variant.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='JSON output path.',
    )
    return parser.parse_args()


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    ordered = sorted(latencies_ms)
    return {
        'count': len(ordered),
        'mean_ms': statistics.fmean(ordered),
        'p50_ms': ordered[len(ordered) // 2],
        'min_ms': ordered[0],
        'max_ms': ordered[-1],
    }


def drop_database_if_exists(cur: psycopg.Cursor[Any], db_name: str) -> None:
    cur.execute(
        """
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = %s
          AND pid <> pg_backend_pid()
        """,
        (db_name,),
    )
    cur.execute(
        sql.SQL('DROP DATABASE IF EXISTS {}').format(sql.Identifier(db_name))
    )


def setup_schema(cur: psycopg.Cursor[Any], docs: int) -> None:
    cur.execute('CREATE EXTENSION evoke')
    cur.execute('CREATE SCHEMA bench')
    cur.execute(
        """
        CREATE TABLE bench.docs (
            id integer PRIMARY KEY,
            title_tokens text[] NOT NULL,
            body_tokens text[] NOT NULL
        )
        """
    )
    cur.execute(
        """
        INSERT INTO bench.docs (id, title_tokens, body_tokens)
        SELECT
            i,
            ARRAY[
                CASE WHEN i %% 2 = 0 THEN 'bird' ELSE 'cat' END,
                CASE WHEN i %% 3 = 0 THEN 'policy' ELSE 'fruit' END,
                CASE WHEN i %% 5 = 0 THEN 'science' ELSE 'animal' END
            ],
            ARRAY[
                CASE WHEN i %% 2 = 0 THEN 'bird' ELSE 'dog' END,
                CASE WHEN i %% 3 = 0 THEN 'cat' ELSE 'fish' END,
                CASE WHEN i %% 7 = 0 THEN 'policy' ELSE 'forest' END,
                CASE WHEN i %% 11 = 0 THEN 'science' ELSE 'garden' END
            ]
        FROM generate_series(1, %s) AS g(i)
        """,
        (docs,),
    )
    cur.execute(
        """
        CREATE INDEX title_tokens_bm25_idx
        ON bench.docs USING evoke (title_tokens)
        WITH (
            method = 'lucene',
            idf_method = 'lucene',
            create_empty_token = true
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX body_tokens_bm25_idx
        ON bench.docs USING evoke (body_tokens)
        WITH (
            method = 'lucene',
            idf_method = 'lucene',
            create_empty_token = true
        )
        """
    )
    cur.execute('ANALYZE bench.docs')


def benchmark_variant(
    cur: psycopg.Cursor[Any],
    name: str,
    sql_text: str,
    queries: list[str],
    repeats: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    query_results: dict[str, dict[str, list[Any]]] = {}

    for query_text in queries:
        for attempt in range(repeats):
            started = time.perf_counter()
            cur.execute(sql_text, {'query': query_text})
            row = cur.fetchone()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            if attempt == 0:
                query_results[query_text] = {
                    'doc_ids': row[0] or [],
                    'scores': row[1] or [],
                }

    return {
        'name': name,
        'latency_ms': summarize_latencies(latencies_ms),
        'query_results': query_results,
    }


def compare_results(
    baseline: dict[str, dict[str, list[Any]]],
    candidate: dict[str, dict[str, list[Any]]],
) -> bool:
    for query_text, baseline_result in baseline.items():
        candidate_result = candidate.get(query_text)
        if candidate_result is None:
            return False
        if baseline_result['doc_ids'] != candidate_result['doc_ids']:
            return False
        baseline_scores = baseline_result['scores']
        candidate_scores = candidate_result['scores']
        if len(baseline_scores) != len(candidate_scores):
            return False
        for left, right in zip(baseline_scores, candidate_scores):
            if abs(float(left) - float(right)) > 1e-6:
                return False
    return True


def main() -> None:
    args = parse_args()

    queries = list(DEFAULT_QUERIES)
    sql_variants = {
        'baseline_fused': f"""
            WITH hits AS (
                SELECT *
                FROM public.evoke_fusion(
                    ARRAY(
                        SELECT h
                        FROM public.evoke_query(
                            'bench.title_tokens_bm25_idx'::regclass,
                            %(query)s::text,
                            {args.candidate_k},
                            NULL::real[]
                        ) AS h
                    ),
                    2.0,
                    ARRAY(
                        SELECT h
                        FROM public.evoke_query(
                            'bench.body_tokens_bm25_idx'::regclass,
                            %(query)s::text,
                            {args.candidate_k},
                            NULL::real[]
                        ) AS h
                    ),
                    1.0,
                    {args.k}
                )
            )
            SELECT
                COALESCE(
                    array_agg(doc_id ORDER BY score DESC, ctid::text),
                    ARRAY[]::int4[]
                ) AS doc_ids,
                COALESCE(
                    array_agg(score ORDER BY score DESC, ctid::text),
                    ARRAY[]::real[]
                ) AS scores
            FROM hits
        """,
        'weighted_queries': f"""
            WITH hits AS (
                SELECT *
                FROM public.evoke_fusion_query_weighted(
                    ARRAY[
                        public.evoke_fusion_weighted_query(
                            'bench.title_tokens_bm25_idx'::regclass,
                            %(query)s::text,
                            2.0
                        ),
                        public.evoke_fusion_weighted_query(
                            'bench.body_tokens_bm25_idx'::regclass,
                            %(query)s::text,
                            1.0
                        )
                    ]::public.evoke_result_fusion_weighted_query[],
                    {args.k},
                    {args.candidate_k},
                    NULL::real[]
                )
            )
            SELECT
                COALESCE(
                    array_agg(doc_id ORDER BY score DESC, ctid::text),
                    ARRAY[]::int4[]
                ) AS doc_ids,
                COALESCE(
                    array_agg(score ORDER BY score DESC, ctid::text),
                    ARRAY[]::real[]
                ) AS scores
            FROM hits
        """,
        'field_queries': f"""
            WITH hits AS (
                SELECT *
                FROM public.evoke_fusion_query_fields(
                    ARRAY[
                        public.evoke_fusion_field_query(
                            'title',
                            'bench.title_tokens_bm25_idx'::regclass,
                            %(query)s::text,
                            2.0
                        ),
                        public.evoke_fusion_field_query(
                            'body',
                            'bench.body_tokens_bm25_idx'::regclass,
                            %(query)s::text,
                            1.0
                        )
                    ]::public.evoke_result_fusion_field_query[],
                    {args.k},
                    {args.candidate_k},
                    NULL::real[]
                )
            )
            SELECT
                COALESCE(
                    array_agg(doc_id ORDER BY score DESC, ctid::text),
                    ARRAY[]::int4[]
                ) AS doc_ids,
                COALESCE(
                    array_agg(score ORDER BY score DESC, ctid::text),
                    ARRAY[]::real[]
                ) AS scores
            FROM hits
        """,
        'search_indexes_named': f"""
            WITH hits AS (
                SELECT *
                FROM public.evoke_fusion_query(
                    ARRAY['title', 'body']::text[],
                    ARRAY[
                        'bench.title_tokens_bm25_idx'::regclass,
                        'bench.body_tokens_bm25_idx'::regclass
                    ],
                    %(query)s::text,
                    ARRAY[2.0, 1.0]::real[],
                    {args.k},
                    {args.candidate_k},
                    NULL::real[]
                )
            )
            SELECT
                COALESCE(
                    array_agg(doc_id ORDER BY score DESC, ctid::text),
                    ARRAY[]::int4[]
                ) AS doc_ids,
                COALESCE(
                    array_agg(score ORDER BY score DESC, ctid::text),
                    ARRAY[]::real[]
                ) AS scores
            FROM hits
        """,
        'search_indexes_short': f"""
            WITH hits AS (
                SELECT *
                FROM public.evoke_fusion_query(
                    ARRAY[
                        'bench.title_tokens_bm25_idx'::regclass,
                        'bench.body_tokens_bm25_idx'::regclass
                    ],
                    %(query)s::text,
                    ARRAY[2.0, 1.0]::real[],
                    {args.k},
                    {args.candidate_k},
                    NULL::real[]
                )
            )
            SELECT
                COALESCE(
                    array_agg(doc_id ORDER BY score DESC, ctid::text),
                    ARRAY[]::int4[]
                ) AS doc_ids,
                COALESCE(
                    array_agg(score ORDER BY score DESC, ctid::text),
                    ARRAY[]::real[]
                ) AS scores
            FROM hits
        """,
    }

    results: dict[str, Any] = {
        'docs': args.docs,
        'queries': queries,
        'k': args.k,
        'candidate_k': args.candidate_k,
        'repeats': args.repeats,
        'variants': {},
        'equivalence': {},
    }

    admin_dsn = conninfo.make_conninfo(
        args.dsn,
        dbname='postgres',
        application_name='evoke_field_helper_admin',
    )
    benchmark_dsn = conninfo.make_conninfo(
        args.dsn,
        dbname=BENCHMARK_DB,
        application_name='evoke_field_helper_bench',
    )

    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            drop_database_if_exists(cur, BENCHMARK_DB)
            cur.execute(
                sql.SQL('CREATE DATABASE {}').format(
                    sql.Identifier(BENCHMARK_DB)
                )
            )

    try:
        with psycopg.connect(benchmark_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                setup_schema(cur, args.docs)
                for name, sql_text in sql_variants.items():
                    results['variants'][name] = benchmark_variant(
                        cur,
                        name,
                        sql_text,
                        queries,
                        args.repeats,
                    )
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                drop_database_if_exists(cur, BENCHMARK_DB)

    baseline = results['variants']['baseline_fused']['query_results']
    for name, variant in results['variants'].items():
        results['equivalence'][name] = compare_results(
            baseline,
            variant['query_results'],
        )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + '\n', 'utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
