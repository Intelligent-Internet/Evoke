#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psycopg
from psycopg import conninfo, sql

from benchmark_beir_official import (
    DEFAULT_DATASETS_DIR,
    PG_DSN,
    TOP_K,
    benchmark_upstream,
    dataset_stats,
    drop_database_if_exists,
    load_dataset,
    summarize_latencies,
    tokenize_dataset,
)


FOCUSED_DATASETS = ['arguana', 'scifact', 'webis-touche2020']


def benchmark_db_name(dataset: str) -> str:
    suffix = os.environ.get('EVOKE_ORDERBY_DB_SUFFIX', '')
    return f'evoke_orderby_{dataset.replace("-", "_")}{suffix}'


def decode_ids(vocab_by_id: list[str], token_ids: list[int]) -> list[str]:
    return [vocab_by_id[token_id] for token_id in token_ids]


def fetch_plan(
    cur: psycopg.Cursor[Any],
    sql_text: str,
    params: tuple[Any, ...],
) -> list[str]:
    cur.execute(f'EXPLAIN (COSTS OFF) {sql_text}', params)
    return [row[0] for row in cur.fetchall()]


def benchmark_ids_search(
    cur: psycopg.Cursor[Any],
    queries: list[list[int]],
    top_k: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    query_sql = """
        SELECT
            count(*),
            coalesce(min(doc_id), 0),
            coalesce(max(score), 0::real)
        FROM public.evoke_query_ids(
            'bench.docs_ids_bm25_idx'::regclass,
            %s::int4[],
            %s::int4,
            NULL
        )
    """

    for query_ids in queries:
        t0 = time.perf_counter()
        cur.execute(query_sql, (query_ids, top_k))
        cur.fetchone()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return asdict(summarize_latencies(latencies_ms))


def benchmark_text_search(
    cur: psycopg.Cursor[Any],
    queries: list[list[str]],
    top_k: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    query_sql = """
        SELECT
            count(*),
            coalesce(min(doc_id), 0),
            coalesce(max(score), 0::real)
        FROM public.evoke_query_tokens(
            'bench.docs_tokens_bm25_idx'::regclass,
            %s::text[],
            %s::int4,
            NULL
        )
    """

    for query_tokens in queries:
        t0 = time.perf_counter()
        cur.execute(query_sql, (query_tokens, top_k))
        cur.fetchone()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return asdict(summarize_latencies(latencies_ms))


def benchmark_ids_orderby(
    cur: psycopg.Cursor[Any],
    queries: list[list[int]],
    top_k: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    query_sql = """
        SELECT count(*), coalesce(min(id), 0)
        FROM (
            SELECT id
            FROM bench.docs_ids
            ORDER BY token_ids <=> %s::int4[] ASC
            LIMIT %s
        ) hits
    """

    cur.execute('SET enable_seqscan = off')
    cur.execute('SET enable_sort = off')
    for query_ids in queries:
        t0 = time.perf_counter()
        cur.execute(query_sql, (query_ids, top_k))
        cur.fetchone()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    plan = fetch_plan(cur, query_sql, (queries[0], top_k))
    cur.execute('RESET enable_sort')
    cur.execute('RESET enable_seqscan')

    return {
        'query': asdict(summarize_latencies(latencies_ms)),
        'plan_forced': plan,
    }


def benchmark_text_orderby(
    cur: psycopg.Cursor[Any],
    queries: list[list[str]],
    top_k: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    query_sql = """
        SELECT count(*), coalesce(min(id), 0)
        FROM (
            SELECT id
            FROM bench.docs_tokens
            ORDER BY tokens <=> %s::text[] ASC
            LIMIT %s
        ) hits
    """

    cur.execute('SET enable_seqscan = off')
    cur.execute('SET enable_sort = off')
    for query_tokens in queries:
        t0 = time.perf_counter()
        cur.execute(query_sql, (query_tokens, top_k))
        cur.fetchone()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    plan = fetch_plan(cur, query_sql, (queries[0], top_k))
    cur.execute('RESET enable_sort')
    cur.execute('RESET enable_seqscan')

    return {
        'query': asdict(summarize_latencies(latencies_ms)),
        'plan_forced': plan,
    }


def run_dataset(
    dataset: str,
    datasets_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    corpus_ids, corpus_texts, query_texts = load_dataset(dataset, datasets_dir)
    corpus_tokenized, query_ids, vocab_by_id = tokenize_dataset(
        corpus_texts,
        query_texts,
    )
    query_tokens = [decode_ids(vocab_by_id, ids) for ids in query_ids]
    db_name = benchmark_db_name(dataset)

    drop_database_if_exists(db_name)
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL('CREATE DATABASE {}').format(sql.Identifier(db_name))
            )

    db_dsn = conninfo.make_conninfo(
        PG_DSN,
        dbname=db_name,
        application_name='evoke_orderby_eval',
    )
    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION evoke')
            cur.execute('SELECT public.evoke_runtime_cache_clear()')
            cur.execute('CREATE SCHEMA bench')
            cur.execute(
                'CREATE TABLE bench.docs_ids ('
                'id integer PRIMARY KEY, '
                'token_ids int4[] NOT NULL'
                ')'
            )
            cur.execute(
                'CREATE TABLE bench.docs_tokens ('
                'id integer PRIMARY KEY, '
                'tokens text[] NOT NULL'
                ')'
            )

            with cur.copy('COPY bench.docs_ids (id, token_ids) FROM STDIN') as copy:
                for doc_id, token_ids in enumerate(
                    corpus_tokenized.ids,
                    start=1,
                ):
                    copy.write_row((doc_id, token_ids))

            with cur.copy('COPY bench.docs_tokens (id, tokens) FROM STDIN') as copy:
                for doc_id, token_ids in enumerate(
                    corpus_tokenized.ids,
                    start=1,
                ):
                    copy.write_row((doc_id, decode_ids(vocab_by_id, token_ids)))

            started = time.perf_counter()
            cur.execute(
                """
                CREATE INDEX docs_ids_bm25_idx
                ON bench.docs_ids USING evoke (token_ids)
                WITH (
                    method = 'lucene',
                    idf_method = 'lucene',
                    k1 = 1.5,
                    b = 0.75,
                    delta = 0.5,
                    create_empty_token = true
                )
                """
            )
            ids_build_ms = (time.perf_counter() - started) * 1000.0
            cur.execute(
                "SELECT pg_relation_size('bench.docs_ids_bm25_idx'::regclass)"
            )
            ids_build_bytes = cur.fetchone()[0]

            started = time.perf_counter()
            cur.execute(
                """
                CREATE INDEX docs_tokens_bm25_idx
                ON bench.docs_tokens USING evoke (tokens)
                WITH (
                    method = 'lucene',
                    idf_method = 'lucene',
                    k1 = 1.5,
                    b = 0.75,
                    delta = 0.5
                )
                """
            )
            text_build_ms = (time.perf_counter() - started) * 1000.0
            cur.execute(
                "SELECT pg_relation_size('bench.docs_tokens_bm25_idx'::regclass)"
            )
            text_build_bytes = cur.fetchone()[0]

            ids_default_plan = fetch_plan(
                cur,
                """
                SELECT id
                FROM bench.docs_ids
                ORDER BY token_ids <=> %s::int4[] ASC
                LIMIT %s
                """,
                (query_ids[0], top_k),
            )
            text_default_plan = fetch_plan(
                cur,
                """
                SELECT id
                FROM bench.docs_tokens
                ORDER BY tokens <=> %s::text[] ASC
                LIMIT %s
                """,
                (query_tokens[0], top_k),
            )

            ids_search = benchmark_ids_search(cur, query_ids, top_k)
            ids_orderby = benchmark_ids_orderby(cur, query_ids, top_k)
            text_search = benchmark_text_search(cur, query_tokens, top_k)
            text_orderby = benchmark_text_orderby(cur, query_tokens, top_k)

    upstream = benchmark_upstream(
        corpus_ids,
        corpus_tokenized.ids,
        query_ids,
        top_k,
    )
    result = {
        'dataset': dataset,
        'stats': dataset_stats(corpus_tokenized, query_ids),
        'upstream_bm25s_local': upstream,
        'evoke_ids': {
            'build_ms': ids_build_ms,
            'build_bytes': ids_build_bytes,
            'search_query': ids_search,
            'orderby_query': ids_orderby['query'],
            'plan_default': ids_default_plan,
            'plan_forced': ids_orderby['plan_forced'],
        },
        'evoke_text': {
            'build_ms': text_build_ms,
            'build_bytes': text_build_bytes,
            'search_query': text_search,
            'orderby_query': text_orderby['query'],
            'plan_default': text_default_plan,
            'plan_forced': text_orderby['plan_forced'],
        },
    }

    del corpus_ids
    del corpus_texts
    del query_texts
    del corpus_tokenized
    del query_ids
    del query_tokens
    del vocab_by_id
    gc.collect()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark native ORDER BY scans against result helpers.'
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=FOCUSED_DATASETS,
        help='Datasets to benchmark.',
    )
    parser.add_argument(
        '--datasets-dir',
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help='Directory used to cache BEIR datasets.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=TOP_K,
        help='Top-k used for retrieval benchmarking.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = {
        'top_k': args.top_k,
        'datasets': {},
    }

    for dataset in args.datasets:
        results['datasets'][dataset] = run_dataset(
            dataset,
            args.datasets_dir,
            args.top_k,
        )

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
