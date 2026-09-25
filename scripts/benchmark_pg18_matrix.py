#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
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
    benchmark_upstream,
    combine_doc_text,
    dataset_stats,
    ensure_extension,
    load_existing_results,
    merge_cqadupstack,
    persist_results,
    prepare_benchmark_schema,
    read_jsonl_map,
    summarize_latencies,
    tokenize_dataset,
)


DEFAULT_DB_PREFIX = 'evoke_pg18_matrix_'
DEFAULT_TEMP_ROOT = Path(tempfile.gettempdir())
DEFAULT_DATASETS_DIR = (
    DEFAULT_TEMP_ROOT / 'evoke_dataset_cache/beir_official'
)
DEFAULT_OUTPUT = DEFAULT_TEMP_ROOT / 'evoke_pg18_matrix/results/smoke.json'
OFFICIAL_ORDER = [
    'arguana',
    'climate-fever',
    'cqadupstack',
    'dbpedia-entity',
    'fever',
    'fiqa',
    'hotpotqa',
    'msmarco',
    'nfcorpus',
    'nq',
    'quora',
    'scidocs',
    'scifact',
    'trec-covid',
    'webis-touche2020',
]
ENGINE_ORDER = [
    'upstream_bm25s',
    'evoke_ids',
    'evoke_text',
    'pg_search',
    'vchord_bm25',
]


def benchmark_admin_dsn(
    env_name: str | None = None,
) -> str:
    env_dsn = os.environ.get(env_name) if env_name else None
    base_dsn = os.environ.get('EVOKE_BENCH_DSN', 'dbname=postgres')
    if env_dsn:
        base_dsn = env_dsn
    params = conninfo.conninfo_to_dict(base_dsn)
    if not params.get('user'):
        params['user'] = 'postgres'
    if not params.get('dbname'):
        params['dbname'] = 'postgres'
    return conninfo.make_conninfo(**params)


EVOKE_ADMIN_DSN = benchmark_admin_dsn(
    'EVOKE_MATRIX_EVOKE_DSN'
)
PG_SEARCH_ADMIN_DSN = benchmark_admin_dsn(
    'EVOKE_MATRIX_PG_SEARCH_DSN'
)
VCHORD_BM25_ADMIN_DSN = benchmark_admin_dsn(
    'EVOKE_MATRIX_VCHORD_BM25_DSN'
)


def make_database_dsn(
    admin_dsn: str,
    db_name: str,
    application_name: str,
) -> str:
    return conninfo.make_conninfo(
        admin_dsn,
        dbname=db_name,
        application_name=application_name,
    )


def ensure_local_database(
    admin_dsn: str,
    db_name: str,
) -> None:
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT EXISTS ('
                'SELECT 1 FROM pg_database WHERE datname = %s'
                ')',
                (db_name,),
            )
            if cur.fetchone()[0]:
                return
            cur.execute(f'CREATE DATABASE "{db_name}" TEMPLATE template0')


def load_local_dataset(
    dataset: str,
    datasets_dir: Path,
) -> tuple[list[str], list[str], list[str]]:
    data_path = datasets_dir / dataset
    if not data_path.exists():
        raise FileNotFoundError(
            f'Local dataset cache is missing {dataset}: {data_path}'
        )
    if dataset == 'cqadupstack':
        merge_cqadupstack(data_path)

    corpus_path = data_path / 'corpus.jsonl'
    queries_path = data_path / 'queries.jsonl'
    qrels_dir = data_path / 'qrels'
    if not corpus_path.exists():
        raise FileNotFoundError(f'Missing corpus.jsonl for {dataset}')
    if not queries_path.exists():
        raise FileNotFoundError(f'Missing queries.jsonl for {dataset}')
    if not qrels_dir.exists():
        raise FileNotFoundError(f'Missing qrels directory for {dataset}')

    corpus = read_jsonl_map(corpus_path, lambda row: combine_doc_text(row))
    queries = read_jsonl_map(queries_path, lambda row: row['text'])
    return list(corpus.keys()), list(corpus.values()), list(queries.values())


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


def ensure_vchord_bm25_extension(
    cur: psycopg.Cursor[Any],
) -> None:
    cur.execute('DROP EXTENSION IF EXISTS vchord_bm25 CASCADE')
    cur.execute('CREATE EXTENSION vchord_bm25')
    cur.execute(
        "SELECT set_config('search_path', 'public,bm25_catalog', false)"
    )


def benchmark_evoke_ids(
    dataset: str,
    corpus_id_tokens: list[list[int]],
    query_id_tokens: list[list[int]],
    top_k: int,
    db_prefix: str,
) -> dict[str, Any]:
    db_name = f'{db_prefix}evoke_{dataset.replace("-", "_")}'
    ensure_local_database(EVOKE_ADMIN_DSN, db_name)
    db_dsn = make_database_dsn(
        EVOKE_ADMIN_DSN,
        db_name,
        'evoke_matrix',
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


def benchmark_evoke_text(
    dataset: str,
    corpus_id_tokens: list[list[int]],
    query_id_tokens: list[list[int]],
    vocab_by_id: list[str],
    top_k: int,
    db_prefix: str,
) -> dict[str, Any]:
    db_name = f'{db_prefix}evoke_text_{dataset.replace("-", "_")}'
    ensure_local_database(EVOKE_ADMIN_DSN, db_name)
    db_dsn = make_database_dsn(
        EVOKE_ADMIN_DSN,
        db_name,
        'evoke_matrix',
    )

    def decode(token_ids: list[int]) -> list[str]:
        return decode_tokens(token_ids, vocab_by_id)

    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            ensure_extension(cur, reuse_db=False)
            prepare_benchmark_schema(cur, reuse_db=False)
            cur.execute(
                'CREATE TABLE IF NOT EXISTS bench.docs_tokens ('
                'id integer PRIMARY KEY, '
                'tokens text[] NOT NULL'
                ')'
            )
            with cur.copy(
                'COPY bench.docs_tokens (id, tokens) FROM STDIN'
            ) as copy:
                for doc_id, token_ids in enumerate(
                    corpus_id_tokens,
                    start=1,
                ):
                    copy.write_row((doc_id, decode(token_ids)))

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
                    delta = 0.5,
                    consistency = 'manual'
                )
                """
            )
            build_ms = (time.perf_counter() - started) * 1000.0
            cur.execute(
                "SELECT pg_relation_size('bench.docs_tokens_bm25_idx'::regclass)"
            )
            build_bytes = int(cur.fetchone()[0])

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
            for query_ids in query_id_tokens:
                started = time.perf_counter()
                cur.execute(query_sql, (decode(query_ids), top_k))
                cur.fetchone()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        'build_ms': build_ms,
        'build_bytes': build_bytes,
        'query': summarize_latencies(latencies_ms).__dict__,
    }


def benchmark_pg_search(
    dataset: str,
    corpus_id_tokens: list[list[int]],
    query_id_tokens: list[list[int]],
    vocab_by_id: list[str],
    top_k: int,
    db_prefix: str,
) -> dict[str, Any]:
    db_name = f'{db_prefix}pg_search_{dataset.replace("-", "_")}'
    ensure_local_database(PG_SEARCH_ADMIN_DSN, db_name)
    db_dsn = make_database_dsn(
        PG_SEARCH_ADMIN_DSN,
        db_name,
        'pg_search_matrix',
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
            with cur.copy('COPY bench.docs (id, body) FROM STDIN') as copy:
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


def benchmark_vchord_bm25(
    dataset: str,
    corpus_id_tokens: list[list[int]],
    query_id_tokens: list[list[int]],
    top_k: int,
    db_prefix: str,
) -> dict[str, Any]:
    db_name = f'{db_prefix}vchord_bm25_{dataset.replace("-", "_")}'
    ensure_local_database(VCHORD_BM25_ADMIN_DSN, db_name)
    db_dsn = make_database_dsn(
        VCHORD_BM25_ADMIN_DSN,
        db_name,
        'vchord_bm25_matrix',
    )

    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            ensure_vchord_bm25_extension(cur)
            cur.execute('DROP SCHEMA IF EXISTS bench CASCADE')
            cur.execute('CREATE SCHEMA bench')
            cur.execute(
                'CREATE TABLE bench.docs_raw ('
                'id integer PRIMARY KEY, '
                'token_ids int4[] NOT NULL'
                ')'
            )
            cur.execute(
                'CREATE TABLE bench.docs ('
                'id integer PRIMARY KEY, '
                'embedding bm25vector NOT NULL'
                ')'
            )
            with cur.copy(
                'COPY bench.docs_raw (id, token_ids) FROM STDIN'
            ) as copy:
                for doc_id, token_ids in enumerate(
                    corpus_id_tokens,
                    start=1,
                ):
                    copy.write_row((doc_id, token_ids))

            cur.execute(
                """
                INSERT INTO bench.docs (id, embedding)
                SELECT id, token_ids::bm25vector
                FROM bench.docs_raw
                """
            )

            started = time.perf_counter()
            cur.execute(
                """
                CREATE INDEX docs_embedding_bm25_idx
                ON bench.docs USING bm25 (embedding bm25_ops)
                """
            )
            build_ms = (time.perf_counter() - started) * 1000.0
            cur.execute(
                """
                SELECT
                    coalesce(
                        sum(pg_relation_size(indexrelid)),
                        0
                    )
                FROM pg_index
                WHERE indrelid = 'bench.docs'::regclass
                """
            )
            build_bytes = int(cur.fetchone()[0])

            latencies_ms: list[float] = []
            query_sql = """
                SELECT
                    count(*),
                    coalesce(min(rank), 0::real)
                FROM (
                    SELECT
                        id,
                        embedding <&> to_bm25query(
                            'bench.docs_embedding_bm25_idx'::regclass,
                            %s::int4[]::bm25vector
                        ) AS rank
                    FROM bench.docs
                    ORDER BY rank, id
                    LIMIT %s
                ) AS hits
            """
            for query_ids in query_id_tokens:
                if not query_ids:
                    started = time.perf_counter()
                    cur.execute('SELECT 0, 0::real')
                    cur.fetchone()
                    latencies_ms.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    continue
                started = time.perf_counter()
                cur.execute(query_sql, (query_ids, top_k))
                cur.fetchone()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        'build_ms': build_ms,
        'build_bytes': build_bytes,
        'query': summarize_latencies(latencies_ms).__dict__,
    }


def run_dataset(
    dataset: str,
    datasets_dir: Path,
    top_k: int,
    db_prefix: str,
    paths: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus_ids, corpus_texts, query_texts = load_local_dataset(
        dataset,
        datasets_dir,
    )
    corpus_tokenized, query_ids, vocab_by_id = tokenize_dataset(
        corpus_texts,
        query_texts,
    )

    result = {
        'dataset': dataset,
        'stats': dataset_stats(corpus_tokenized, query_ids),
        'wall_time_s': time.perf_counter() - started,
    }
    if 'upstream_bm25s' in paths:
        result['upstream_bm25s'] = benchmark_upstream(
            corpus_ids,
            corpus_tokenized,
            query_ids,
            top_k,
        )
    if 'evoke_ids' in paths:
        result['evoke_ids'] = benchmark_evoke_ids(
            dataset,
            corpus_tokenized.ids,
            query_ids,
            top_k,
            db_prefix,
        )
    if 'evoke_text' in paths:
        result['evoke_text'] = benchmark_evoke_text(
            dataset,
            corpus_tokenized.ids,
            query_ids,
            vocab_by_id,
            top_k,
            db_prefix,
        )
    if 'pg_search' in paths:
        result['pg_search'] = benchmark_pg_search(
            dataset,
            corpus_tokenized.ids,
            query_ids,
            vocab_by_id,
            top_k,
            db_prefix,
        )
    if 'vchord_bm25' in paths:
        result['vchord_bm25'] = benchmark_vchord_bm25(
            dataset,
            corpus_tokenized.ids,
            query_ids,
            top_k,
            db_prefix,
        )
    result['wall_time_s'] = time.perf_counter() - started

    del corpus_ids
    del corpus_texts
    del query_texts
    del corpus_tokenized
    del query_ids
    del vocab_by_id
    gc.collect()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare four BM25 engines on local BEIR caches.'
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=['arguana'],
        choices=OFFICIAL_ORDER,
        help='Datasets to benchmark. Defaults to a small smoke dataset.',
    )
    parser.add_argument(
        '--datasets-dir',
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help='Directory containing pre-populated BEIR datasets.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
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
    parser.add_argument(
        '--paths',
        nargs='+',
        default=ENGINE_ORDER,
        choices=ENGINE_ORDER,
        help='Subset of engines to run.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_existing_results(args.output)
    payload['paths'] = args.paths
    payload['datasets_dir'] = str(args.datasets_dir)
    payload['db_prefix'] = args.db_prefix

    for dataset in args.datasets:
        existing = payload['results'].get(dataset)
        if args.resume and existing and 'error' not in existing:
            continue
        try:
            payload['results'][dataset] = run_dataset(
                dataset,
                args.datasets_dir,
                args.top_k,
                args.db_prefix,
                args.paths,
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
