#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import bm25s
import psycopg
from psycopg import conninfo

try:
    import Stemmer
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        'This script requires benchmark dependencies. '
        'Install them in the benchmark environment with:\n'
        '  pip install beir PyStemmer tqdm\n'
        f'Import error: {exc}'
    ) from exc

from benchmark_beir_official import (
    DEFAULT_DATASETS_DIR,
    OFFICIAL_ORDER,
    OFFICIAL_QPS,
    PG_DSN,
    TOP_K,
    download_and_unpack_dataset,
    ensure_database,
    load_existing_results,
    merge_cqadupstack,
    persist_results,
    summarize_latencies,
)


DB_PREFIX = 'pg_bm25s_official_'
PSQL_DB_PREFIX = 'evoke_official_'
DEFAULT_OUTPUT = Path(
    'docs/performance/data/diagnostics/'
    'official-beir-pg-bm25s-current-2026-03-31.json'
)
DEFAULT_STATS_SOURCE = Path(
    'docs/performance/data/canonical/'
    'official-beir-text-current-2026-04-02.json'
)


def result_is_success(result: dict[str, Any]) -> bool:
    return 'error' not in result


def source_db_name(dataset: str) -> str:
    return f'{PSQL_DB_PREFIX}{dataset.replace("-", "_")}_text'


def target_db_name(dataset: str) -> str:
    return f'{DB_PREFIX}{dataset.replace("-", "_")}'


def ensure_extension(cur: psycopg.Cursor[Any]) -> None:
    cur.execute('DROP EXTENSION IF EXISTS pg_bm25s CASCADE')
    cur.execute('CREATE EXTENSION pg_bm25s')


def configure_session(cur: psycopg.Cursor[Any]) -> None:
    cur.execute('SET max_parallel_workers = 0')
    cur.execute('SET max_parallel_workers_per_gather = 0')


def load_stats_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {
        dataset: entry.get('stats', {})
        for dataset, entry in payload.get('results', {}).items()
    }


def load_query_texts(dataset: str, datasets_dir: Path) -> list[str]:
    data_path = download_and_unpack_dataset(dataset, datasets_dir)
    if dataset == 'cqadupstack':
        merge_cqadupstack(data_path)

    split = 'dev' if dataset == 'msmarco' else 'test'
    qrel_path = data_path / 'qrels' / f'{split}.tsv'
    query_ids: set[str] = set()
    with qrel_path.open('r', encoding='utf-8') as handle:
        next(handle)
        for line in handle:
            qid, _cid, _score = line.rstrip('\n').split('\t')
            query_ids.add(qid)

    queries_path = data_path / 'queries.jsonl'
    query_texts: list[str] = []
    with queries_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['_id'] in query_ids:
                query_texts.append(row.get('text', ''))

    return query_texts


def tokenize_queries(query_texts: list[str]) -> list[str]:
    tokenizer = bm25s.tokenization.Tokenizer(
        stopwords='en',
        stemmer=Stemmer.Stemmer('english'),
    )
    query_tokens = tokenizer.tokenize(
        query_texts,
        update_vocab=True,
        return_as='string',
    )
    return [' '.join(tokens) for tokens in query_tokens]


def prepare_documents(
    dataset: str,
    target_cur: psycopg.Cursor[Any],
    target_conn: psycopg.Connection[Any],
) -> None:
    src_db_dsn = conninfo.make_conninfo(
        PG_DSN,
        dbname=source_db_name(dataset),
        application_name='pg_bm25s_bench_source',
    )
    target_cur.execute('CREATE SCHEMA IF NOT EXISTS bench')
    target_cur.execute(
        'CREATE TABLE IF NOT EXISTS bench.docs ('
        'id integer NOT NULL, '
        'body text NOT NULL'
        ')'
    )
    target_cur.execute('SELECT count(*) FROM bench.docs')
    loaded_rows = target_cur.fetchone()[0]

    with psycopg.connect(src_db_dsn, autocommit=False) as src_conn:
        with src_conn.cursor() as src_cur:
            src_cur.execute('SELECT count(*) FROM bench.docs_tokens')
            source_rows = src_cur.fetchone()[0]

        if loaded_rows == source_rows:
            return

        target_cur.execute('TRUNCATE bench.docs')
        with src_conn.cursor(name='pg_bm25s_source_docs') as src_cur:
            src_cur.itersize = 10_000
            src_cur.execute(
                """
                SELECT
                    id,
                    array_to_string(tokens, ' ') AS body
                FROM bench.docs_tokens
                ORDER BY id
                """
            )
            with target_cur.copy(
                'COPY bench.docs (id, body) FROM STDIN'
            ) as copy:
                for row in src_cur:
                    copy.write_row(row)

    target_conn.commit()


def benchmark_pg_bm25s(
    dataset: str,
    query_texts: list[str],
    top_k: int,
) -> dict[str, Any]:
    db_name = target_db_name(dataset)
    ensure_database(db_name)
    db_dsn = conninfo.make_conninfo(
        PG_DSN,
        dbname=db_name,
        application_name='pg_bm25s_bench',
    )

    with psycopg.connect(db_dsn, autocommit=False) as conn:
        with conn.cursor() as cur:
            configure_session(cur)
            prepare_documents(dataset, cur, conn)
            ensure_extension(cur)
            cur.execute('DROP INDEX IF EXISTS bench.docs_body_bm25_idx')
            conn.commit()

            started = time.perf_counter()
            cur.execute(
                """
                CREATE INDEX docs_body_bm25_idx
                ON bench.docs USING bm25s (body)
                WITH (
                    method = 'lucene',
                    k1 = 1.5,
                    b = 0.75,
                    delta = 0.5,
                    stopwords = 'none',
                    stemming = false
                )
                """
            )
            conn.commit()
            build_ms = (time.perf_counter() - started) * 1000.0

            cur.execute(
                """
                SELECT coalesce(sum(pg_total_relation_size(oid)), 0)
                FROM pg_class
                WHERE oid IN (
                    'bench.docs_body_bm25_idx'::regclass,
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
            for query_text in query_texts:
                started = time.perf_counter()
                cur.execute(query_sql, (query_text, top_k))
                cur.fetchone()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        'build_ms': build_ms,
        'build_bytes': build_bytes,
        'query': asdict(summarize_latencies(latencies_ms)),
    }


def run_dataset(
    dataset: str,
    datasets_dir: Path,
    top_k: int,
    stats_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    query_texts = tokenize_queries(load_query_texts(dataset, datasets_dir))
    started = time.perf_counter()
    result = {
        'dataset': dataset,
        'official_qps': OFFICIAL_QPS[dataset],
        'stats': stats_map[dataset],
        'pg_bm25s': benchmark_pg_bm25s(dataset, query_texts, top_k),
    }
    result['wall_time_s'] = time.perf_counter() - started
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark pg_bm25s inside local PostgreSQL.'
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=OFFICIAL_ORDER,
        help='Datasets to benchmark. Defaults to the full official suite.',
    )
    parser.add_argument(
        '--datasets-dir',
        type=Path,
        default=DEFAULT_DATASETS_DIR,
    )
    parser.add_argument(
        '--stats-source',
        type=Path,
        default=DEFAULT_STATS_SOURCE,
        help='Canonical evoke raw data used for dataset stats.',
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
        '--resume',
        action='store_true',
        help='Skip datasets that already have successful results.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats_map = load_stats_map(args.stats_source)
    payload = load_existing_results(args.output)
    payload['paths'] = ['pg_bm25s']

    for dataset in args.datasets:
        existing = payload['results'].get(dataset)
        if (
            args.resume
            and existing is not None
            and result_is_success(existing)
        ):
            continue

        try:
            payload['results'][dataset] = run_dataset(
                dataset,
                args.datasets_dir,
                args.top_k,
                stats_map,
            )
        except KeyboardInterrupt:
            persist_results(args.output, payload)
            raise
        except Exception as exc:  # pragma: no cover - benchmark harness
            payload['results'][dataset] = {
                'dataset': dataset,
                'official_qps': OFFICIAL_QPS.get(dataset),
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            }
        persist_results(args.output, payload)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
