#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import getpass
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import psycopg
from psycopg import conninfo

from benchmark_beir_official import (
    TOP_K,
    combine_doc_text,
    dataset_stats,
    download_and_unpack_dataset,
    ensure_extension,
    load_existing_results,
    merge_cqadupstack,
    persist_results,
    prepare_benchmark_schema,
    read_jsonl_map,
    summarize_latencies,
    tokenize_dataset,
)


DEFAULT_STATS_SOURCE = Path(
    'docs/performance/data/canonical/official-beir-text-current-2026-04-02.json'
)
DEFAULT_DB_PREFIX = 'evoke_extcmp_'


def default_work_root() -> Path:
    return Path(tempfile.gettempdir()) / 'evoke_extcmp'


def benchmark_admin_dsn() -> str:
    base_dsn = os.environ.get('EVOKE_BENCH_DSN', 'dbname=postgres')
    params = conninfo.conninfo_to_dict(base_dsn)
    if not params.get('user'):
        params['user'] = getpass.getuser()
    if not params.get('dbname'):
        params['dbname'] = 'postgres'
    return conninfo.make_conninfo(**params)


ADMIN_DSN = benchmark_admin_dsn()


def load_stats_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {
        dataset: entry.get('stats', {})
        for dataset, entry in payload.get('results', {}).items()
    }


def ensure_local_database(db_name: str) -> None:
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT EXISTS ('
                'SELECT 1 FROM pg_database WHERE datname = %s'
                ')',
                (db_name,),
            )
            if cur.fetchone()[0]:
                return
            cur.execute(
                f'CREATE DATABASE "{db_name}"'
            )


def default_datasets(stats_source: Path) -> list[str]:
    stats_map = load_stats_map(stats_source)
    rows = sorted(
        (
            stats.get('documents', 0),
            dataset,
        )
        for dataset, stats in stats_map.items()
    )
    return [
        rows[0][1],
        rows[len(rows) // 2][1],
        rows[-1][1],
    ]


def decode_tokens(
    token_ids: list[int],
    vocab_by_id: list[str],
) -> list[str]:
    return [vocab_by_id[token_id] for token_id in token_ids]


def joined_query_text(
    token_ids: list[int],
    vocab_by_id: list[str],
) -> str:
    return ' '.join(decode_tokens(token_ids, vocab_by_id))


def docs_text_rows(
    corpus_id_tokens: list[list[int]],
    vocab_by_id: list[str],
) -> list[tuple[int, str]]:
    return [
        (doc_id, ' '.join(decode_tokens(token_ids, vocab_by_id)))
        for doc_id, token_ids in enumerate(corpus_id_tokens, start=1)
    ]


def ensure_pg_search_extension(
    cur: psycopg.Cursor[Any],
) -> None:
    cur.execute('DROP EXTENSION IF EXISTS pg_search CASCADE')
    cur.execute('CREATE EXTENSION pg_search')


def ensure_pg_bm25s_extension(
    cur: psycopg.Cursor[Any],
) -> None:
    cur.execute('DROP EXTENSION IF EXISTS pg_bm25s CASCADE')
    cur.execute('CREATE EXTENSION pg_bm25s')


def benchmark_pg_search(
    dataset: str,
    corpus_id_tokens: list[list[int]],
    query_id_tokens: list[list[int]],
    vocab_by_id: list[str],
    top_k: int,
    db_prefix: str,
) -> dict[str, Any]:
    db_name = f'{db_prefix}pg_search_{dataset.replace("-", "_")}'
    ensure_local_database(db_name)
    db_dsn = conninfo.make_conninfo(
        ADMIN_DSN,
        dbname=db_name,
        application_name='pg_search_extcmp',
    )
    rows = docs_text_rows(corpus_id_tokens, vocab_by_id)

    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            ensure_pg_search_extension(cur)
            cur.execute('DROP SCHEMA IF EXISTS bench CASCADE')
            cur.execute('CREATE SCHEMA bench')
            cur.execute(
                'CREATE TABLE bench.docs ('
                'id integer PRIMARY KEY, '
                'body text NOT NULL'
                ')'
            )
            with cur.copy(
                'COPY bench.docs (id, body) FROM STDIN'
            ) as copy:
                for row in rows:
                    copy.write_row(row)

            started = time.perf_counter()
            cur.execute(
                """
                CREATE INDEX docs_body_bm25_idx
                ON bench.docs USING bm25 (id, body)
                WITH (key_field = id)
                """
            )
            build_ms = (time.perf_counter() - started) * 1000.0

            cur.execute(
                "SELECT pg_relation_size('bench.docs_body_bm25_idx'::regclass)"
            )
            build_bytes = int(cur.fetchone()[0])

            latencies_ms: list[float] = []
            query_sql = """
                SELECT
                    count(*),
                    coalesce(max(score), 0::float8)
                FROM (
                    SELECT
                        id,
                        pdb.score(id) AS score
                    FROM bench.docs
                    WHERE body @@@ %s
                    ORDER BY score DESC, id
                    LIMIT %s
                ) AS hits
            """
            for query_ids in query_id_tokens:
                query_text = joined_query_text(query_ids, vocab_by_id)
                if not query_text:
                    started = time.perf_counter()
                    cur.execute('SELECT 0, 0::float8')
                    cur.fetchone()
                    latencies_ms.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    continue
                started = time.perf_counter()
                cur.execute(query_sql, (query_text, top_k))
                cur.fetchone()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        'build_ms': build_ms,
        'build_bytes': build_bytes,
        'query': summarize_latencies(latencies_ms).__dict__,
    }


def benchmark_evoke_ids(
    dataset: str,
    corpus_id_tokens: list[list[int]],
    query_id_tokens: list[list[int]],
    top_k: int,
    db_prefix: str,
) -> dict[str, Any]:
    db_name = f'{db_prefix}evoke_{dataset.replace("-", "_")}'
    ensure_local_database(db_name)
    db_dsn = conninfo.make_conninfo(
        ADMIN_DSN,
        dbname=db_name,
        application_name='evoke_extcmp',
    )

    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            ensure_extension(cur, reuse_db=False)
            prepare_benchmark_schema(cur, reuse_db=False)
            cur.execute(
                'CREATE TABLE IF NOT EXISTS bench.docs_ids ('
                'id integer PRIMARY KEY, '
                'token_ids int4[] NOT NULL'
                ')'
            )
            with cur.copy(
                'COPY bench.docs_ids (id, token_ids) FROM STDIN'
            ) as copy:
                for doc_id, token_ids in enumerate(
                    corpus_id_tokens,
                    start=1,
                ):
                    copy.write_row((doc_id, token_ids))

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
                    create_empty_token = true,
                    consistency = 'manual'
                )
                """
            )
            build_ms = (time.perf_counter() - started) * 1000.0
            cur.execute(
                "SELECT pg_relation_size('bench.docs_ids_bm25_idx'::regclass)"
            )
            build_bytes = int(cur.fetchone()[0])

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
            for query_ids in query_id_tokens:
                started = time.perf_counter()
                cur.execute(query_sql, (query_ids, top_k))
                cur.fetchone()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        'build_ms': build_ms,
        'build_bytes': build_bytes,
        'query': summarize_latencies(latencies_ms).__dict__,
    }


def benchmark_pg_bm25s(
    dataset: str,
    corpus_id_tokens: list[list[int]],
    query_id_tokens: list[list[int]],
    vocab_by_id: list[str],
    top_k: int,
    db_prefix: str,
) -> dict[str, Any]:
    db_name = f'{db_prefix}pg_bm25s_{dataset.replace("-", "_")}'
    ensure_local_database(db_name)
    db_dsn = conninfo.make_conninfo(
        ADMIN_DSN,
        dbname=db_name,
        application_name='pg_bm25s_extcmp',
    )
    rows = docs_text_rows(corpus_id_tokens, vocab_by_id)

    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            ensure_pg_bm25s_extension(cur)
            cur.execute('DROP SCHEMA IF EXISTS bench CASCADE')
            cur.execute('CREATE SCHEMA bench')
            cur.execute(
                'CREATE TABLE bench.docs ('
                'id integer PRIMARY KEY, '
                'body text NOT NULL'
                ')'
            )
            with cur.copy(
                'COPY bench.docs (id, body) FROM STDIN'
            ) as copy:
                for row in rows:
                    copy.write_row(row)

            started = time.perf_counter()
            cur.execute(
                """
                SELECT public.bm25s_build_index(
                    'docs_body_bm25_idx',
                    'bench.docs',
                    'body',
                    'lucene',
                    1.5,
                    0.75,
                    0.5,
                    'none',
                    false
                )
                """
            )
            build_ms = (time.perf_counter() - started) * 1000.0

            cur.execute(
                """
                SELECT coalesce(sum(pg_total_relation_size(oid)), 0)
                FROM pg_class
                WHERE oid IN (
                    'pg_bm25s_indexes'::regclass,
                    'pg_bm25s_terms'::regclass,
                    'pg_bm25s_postings'::regclass
                )
                """
            )
            build_bytes = int(cur.fetchone()[0])

            latencies_ms: list[float] = []
            query_sql = """
                SELECT
                    count(*),
                    coalesce(max(score), 0::float8)
                FROM public.bm25s_search(
                    'docs_body_bm25_idx',
                    %s::text,
                    %s::int4
                )
            """
            for query_ids in query_id_tokens:
                query_text = joined_query_text(query_ids, vocab_by_id)
                if not query_text:
                    started = time.perf_counter()
                    cur.execute('SELECT 0, 0::float8')
                    cur.fetchone()
                    latencies_ms.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    continue
                started = time.perf_counter()
                cur.execute(query_sql, (query_text, top_k))
                cur.fetchone()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        'build_ms': build_ms,
        'build_bytes': build_bytes,
        'query': summarize_latencies(latencies_ms).__dict__,
    }


def load_dataset(
    dataset: str,
    datasets_dir: Path,
) -> tuple[list[str], list[str], list[str]]:
    data_path = download_and_unpack_dataset(dataset, datasets_dir)
    if dataset == 'cqadupstack':
        merge_cqadupstack(data_path)

    corpus = read_jsonl_map(
        data_path / 'corpus.jsonl',
        lambda row: combine_doc_text(row),
    )
    queries = read_jsonl_map(
        data_path / 'queries.jsonl',
        lambda row: row['text'],
    )
    return list(corpus.keys()), list(corpus.values()), list(queries.values())


def run_dataset(
    dataset: str,
    datasets_dir: Path,
    top_k: int,
    db_prefix: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus_ids, corpus_texts, query_texts = load_dataset(dataset, datasets_dir)
    corpus_tokenized, query_ids, vocab_by_id = tokenize_dataset(
        corpus_texts,
        query_texts,
    )

    result = {
        'dataset': dataset,
        'stats': dataset_stats(corpus_tokenized, query_ids),
        'evoke_ids': benchmark_evoke_ids(
            dataset,
            corpus_tokenized.ids,
            query_ids,
            top_k,
            db_prefix,
        ),
        'pg_search': benchmark_pg_search(
            dataset,
            corpus_tokenized.ids,
            query_ids,
            vocab_by_id,
            top_k,
            db_prefix,
        ),
        'pg_bm25s': benchmark_pg_bm25s(
            dataset,
            corpus_tokenized.ids,
            query_ids,
            vocab_by_id,
            top_k,
            db_prefix,
        ),
        'wall_time_s': time.perf_counter() - started,
    }

    del corpus_ids
    del corpus_texts
    del query_texts
    del corpus_tokenized
    del query_ids
    del vocab_by_id
    gc.collect()

    return result


def parse_args() -> argparse.Namespace:
    root = default_work_root()
    parser = argparse.ArgumentParser(
        description='Compare evoke, pg_search, and pg_bm25s.'
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=None,
        help='Defaults to smallest, median, and largest official datasets.',
    )
    parser.add_argument(
        '--stats-source',
        type=Path,
        default=DEFAULT_STATS_SOURCE,
    )
    parser.add_argument(
        '--datasets-dir',
        type=Path,
        default=root / 'datasets',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=root / 'results' / 'three-scale-pg-search-compare.json',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=TOP_K,
    )
    parser.add_argument(
        '--db-prefix',
        default=DEFAULT_DB_PREFIX,
    )
    parser.add_argument(
        '--resume',
        action='store_true',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_existing_results(args.output)
    payload['paths'] = ['evoke_ids', 'pg_search', 'pg_bm25s']
    payload['datasets_dir'] = str(args.datasets_dir)
    payload['db_prefix'] = args.db_prefix

    datasets = args.datasets or default_datasets(args.stats_source)
    payload['selected_datasets'] = datasets

    for dataset in datasets:
        existing = payload['results'].get(dataset)
        if args.resume and existing and 'error' not in existing:
            continue
        try:
            payload['results'][dataset] = run_dataset(
                dataset,
                args.datasets_dir,
                args.top_k,
                args.db_prefix,
            )
        except Exception as exc:
            payload['results'][dataset] = {
                'dataset': dataset,
                'error': str(exc),
                'traceback': traceback.format_exc(),
            }
        persist_results(args.output, payload)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
