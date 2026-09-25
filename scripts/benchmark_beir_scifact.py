#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import statistics
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import bm25s
import psycopg


DATASET_URL = (
    'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
)
DATASET_ZIP = Path('/tmp/scifact.zip')
DATASET_DIR = Path('/tmp/scifact')
DB_NAME = 'evoke_beir_scifact'
TOP_K = 10
WARMUP = 50
PG_DSN = os.environ.get('EVOKE_BENCH_DSN', 'dbname=postgres')


@dataclass
class BenchmarkStats:
    avg_ms: float
    p50_ms: float
    p95_ms: float
    qps: float
    count: int


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_latencies(latencies_ms: list[float]) -> BenchmarkStats:
    ordered = sorted(latencies_ms)
    avg_ms = statistics.fmean(latencies_ms)
    return BenchmarkStats(
        avg_ms=avg_ms,
        p50_ms=percentile(ordered, 0.50),
        p95_ms=percentile(ordered, 0.95),
        qps=1000.0 / avg_ms,
        count=len(latencies_ms),
    )


def ensure_dataset() -> None:
    if DATASET_ZIP.exists() and DATASET_DIR.exists():
        return

    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_ZIP.exists():
        raise FileNotFoundError(
            f'Missing {DATASET_ZIP}. Download it from {DATASET_URL} first.'
        )

    with zipfile.ZipFile(DATASET_ZIP) as archive:
        archive.extractall(DATASET_DIR.parent)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def combine_doc_text(row: dict) -> str:
    title = (row.get('title') or '').strip()
    text = (row.get('text') or '').strip()
    if title and text:
        return f'{title}\n{text}'
    return title or text


def chunked(items: list[tuple], size: int) -> Iterable[list[tuple]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def benchmark_upstream(
    corpus_tokens: list[list[str]],
    query_tokens: list[list[str]],
) -> tuple[float, BenchmarkStats]:
    retriever = bm25s.BM25(method='lucene', idf_method='lucene')
    t0 = time.perf_counter()
    retriever.index(corpus_tokens, show_progress=False)
    build_ms = (time.perf_counter() - t0) * 1000.0

    latencies_ms: list[float] = []
    for i, query in enumerate(query_tokens):
        t0 = time.perf_counter()
        retriever.retrieve(
            [query],
            k=TOP_K,
            sorted=True,
            show_progress=False,
            leave_progress=False,
            return_as='tuple',
        )
        if i >= WARMUP:
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return build_ms, summarize_latencies(latencies_ms)


def recreate_database() -> None:
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS {DB_NAME}')
            cur.execute(f'CREATE DATABASE {DB_NAME}')


def benchmark_postgres(
    corpus_text_tokens: list[list[str]],
    corpus_id_tokens: list[list[int]],
    query_text_tokens: list[list[str]],
    query_id_tokens: list[list[int]],
) -> dict:
    recreate_database()
    db_dsn = PG_DSN.replace('dbname=postgres', f'dbname={DB_NAME}')

    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION evoke')
            cur.execute('CREATE SCHEMA bench')
            cur.execute('SET search_path = bench, public')
            cur.execute(
                'CREATE TABLE docs_tokens (id int primary key, tokens text[] not null)'
            )
            cur.execute(
                'CREATE TABLE docs_ids (id int primary key, token_ids int4[] not null)'
            )

            token_rows = [
                (doc_id, tokens)
                for doc_id, tokens in enumerate(corpus_text_tokens, start=1)
            ]
            id_rows = [
                (doc_id, token_ids)
                for doc_id, token_ids in enumerate(corpus_id_tokens, start=1)
            ]
            for batch in chunked(token_rows, 500):
                cur.executemany(
                    'INSERT INTO docs_tokens (id, tokens) VALUES (%s, %s)',
                    batch,
                )
            for batch in chunked(id_rows, 500):
                cur.executemany(
                    'INSERT INTO docs_ids (id, token_ids) VALUES (%s, %s)',
                    batch,
                )

            t0 = time.perf_counter()
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
            build_tokens_ms = (time.perf_counter() - t0) * 1000.0
            cur.execute(
                "SELECT pg_relation_size('bench.docs_tokens_bm25_idx'::regclass)"
            )
            build_tokens_bytes = cur.fetchone()[0]

            t0 = time.perf_counter()
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
            build_ids_ms = (time.perf_counter() - t0) * 1000.0
            cur.execute(
                "SELECT pg_relation_size('bench.docs_ids_bm25_idx'::regclass)"
            )
            build_ids_bytes = cur.fetchone()[0]

            text_latencies_ms: list[float] = []
            for i, query_tokens in enumerate(query_text_tokens):
                t0 = time.perf_counter()
                cur.execute(
                    """
                    SELECT doc_id, score
                    FROM public.evoke_query_tokens(
                        'bench.docs_tokens_bm25_idx'::regclass,
                        %s,
                        %s,
                        NULL
                    )
                    """,
                    (query_tokens, TOP_K),
                )
                cur.fetchall()
                if i >= WARMUP:
                    text_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            id_latencies_ms: list[float] = []
            for i, query_ids in enumerate(query_id_tokens):
                t0 = time.perf_counter()
                cur.execute(
                    """
                    SELECT doc_id, score
                    FROM public.evoke_query_ids(
                        'bench.docs_ids_bm25_idx'::regclass,
                        %s,
                        %s,
                        NULL
                    )
                    """,
                    (query_ids, TOP_K),
                )
                cur.fetchall()
                if i >= WARMUP:
                    id_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        'text': {
            'build_ms': build_tokens_ms,
            'build_bytes': build_tokens_bytes,
            'query': summarize_latencies(text_latencies_ms),
        },
        'ids': {
            'build_ms': build_ids_ms,
            'build_bytes': build_ids_bytes,
            'query': summarize_latencies(id_latencies_ms),
        },
    }


def main() -> None:
    ensure_dataset()
    corpus_rows = load_jsonl(DATASET_DIR / 'corpus.jsonl')
    query_rows = load_jsonl(DATASET_DIR / 'queries.jsonl')

    corpus_texts = [combine_doc_text(row) for row in corpus_rows]
    query_texts = [row['text'] for row in query_rows]

    corpus_tokens = bm25s.tokenize(
        corpus_texts,
        return_ids=False,
        show_progress=False,
    )
    query_tokens = bm25s.tokenize(
        query_texts,
        return_ids=False,
        show_progress=False,
    )

    vocab: dict[str, int] = {}
    corpus_ids: list[list[int]] = []
    for tokens in corpus_tokens:
        token_ids: list[int] = []
        for token in tokens:
            token_id = vocab.get(token)
            if token_id is None:
                token_id = len(vocab)
                vocab[token] = token_id
            token_ids.append(token_id)
        corpus_ids.append(token_ids)

    query_ids: list[list[int]] = []
    for tokens in query_tokens:
        query_ids.append([vocab[token] for token in tokens if token in vocab])

    upstream_build_ms, upstream_stats = benchmark_upstream(
        corpus_tokens=corpus_tokens,
        query_tokens=query_tokens,
    )
    pg_results = benchmark_postgres(
        corpus_text_tokens=corpus_tokens,
        corpus_id_tokens=corpus_ids,
        query_text_tokens=query_tokens,
        query_id_tokens=query_ids,
    )

    result = {
        'dataset': 'scifact',
        'documents': len(corpus_tokens),
        'queries_total': len(query_tokens),
        'queries_measured': max(0, len(query_tokens) - WARMUP),
        'vocab_size': len(vocab),
        'upstream_bm25s': {
            'build_ms': upstream_build_ms,
            'query': upstream_stats.__dict__,
        },
        'evoke_text': {
            'build_ms': pg_results['text']['build_ms'],
            'build_bytes': pg_results['text']['build_bytes'],
            'query': pg_results['text']['query'].__dict__,
        },
        'evoke_ids': {
            'build_ms': pg_results['ids']['build_ms'],
            'build_bytes': pg_results['ids']['build_bytes'],
            'query': pg_results['ids']['query'].__dict__,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
