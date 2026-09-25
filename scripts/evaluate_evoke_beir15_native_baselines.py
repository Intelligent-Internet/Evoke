#!/usr/bin/env python3
"""Evaluate native BEIR15 BM25 and dense baselines from PostgreSQL.

This script keeps the baseline side inside the existing product database:

- BM25 uses the Evoke index through evoke_query.
- Dense uses the stored halfvec embeddings and VectorChord/halfvec KNN SQL.

Its output uses the common native retrieval-matrix metric contract so current
Evoke, BM25, and dense rows can be compared without a research evaluator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

plain_query_module = importlib.import_module('evoke_plain_query')
plain_text_to_raw_terms = plain_query_module.plain_text_to_raw_terms


DEFAULT_DSN = 'dbname=postgres'
DEFAULT_SCHEMA = 'evoke_beir15'
PREPARE_THRESHOLD = 0
METRIC_KEYS = (
    'candidate_upper_bound',
    'map_at_100',
    'mrr_at_20',
    'ndcg_at_10',
    'recall_at_100',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate native BEIR15 BM25/dense baselines.',
    )
    parser.add_argument('--dsn', default=DEFAULT_DSN)
    parser.add_argument('--schema', default=DEFAULT_SCHEMA)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--sources', default='bm25,dense')
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--limit-queries', type=int, default=0)
    parser.add_argument('--query-ids-file', type=Path)
    parser.add_argument('--progress-every', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--dense-ranking-jsonl', type=Path)
    parser.add_argument('--dense-ranking-k', type=int, default=100)
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--output-md', type=Path)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def read_query_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    query_ids = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            query_ids.add(stripped)
    return query_ids


def filter_queries(
    queries: list[dict[str, Any]],
    *,
    query_ids: set[str],
) -> list[dict[str, Any]]:
    if not query_ids:
        return queries
    return [
        query
        for query in queries
        if str(query['query_id']) in query_ids
    ]


def qident(name: str) -> sql.Identifier:
    return sql.Identifier(name)


def fetch_dataset_info(
    conn: psycopg.Connection[Any],
    *,
    schema: str,
    dataset: str,
) -> dict[str, Any]:
    row = conn.execute(
        sql.SQL("""
            SELECT table_name, vector_probes
            FROM {}.dataset_manifest
            WHERE dataset = %s
        """).format(qident(schema)),
        (dataset,),
    ).fetchone()
    if row is None:
        raise ValueError(f'dataset missing from manifest: {dataset}')
    return {
        'table_name': str(row[0]),
        'vector_probes': None if row[1] is None else int(row[1]),
    }


def load_queries(
    conn: psycopg.Connection[Any],
    *,
    schema: str,
    dataset: str,
    limit: int,
) -> list[dict[str, Any]]:
    query = sql.SQL("""
        SELECT query_id, text_content, embedding::text
        FROM {}.queries
        WHERE dataset = %s
        ORDER BY query_id
    """).format(qident(schema))
    rows = conn.execute(query, (dataset,)).fetchall()
    if limit > 0:
        rows = rows[:limit]
    return [
        {
            'embedding': str(row[2]),
            'query_id': str(row[0]),
            'text': str(row[1]),
        }
        for row in rows
    ]


def load_qrels(
    conn: psycopg.Connection[Any],
    *,
    schema: str,
    dataset: str,
) -> dict[str, dict[str, float]]:
    rows = conn.execute(
        sql.SQL("""
            SELECT query_id, doc_id, score
            FROM {}.qrels
            WHERE dataset = %s
              AND score > 0
        """).format(qident(schema)),
        (dataset,),
    ).fetchall()
    qrels: dict[str, dict[str, float]] = {}
    for query_id, doc_id, score in rows:
        qrels.setdefault(str(query_id), {})[str(doc_id)] = float(score)
    return qrels


def query_bm25(
    conn: psycopg.Connection[Any],
    *,
    schema: str,
    doc_table: str,
    query_text: str,
    k: int,
) -> list[str]:
    rows = conn.execute(
        sql.SQL("""
            SELECT d.doc_id
            FROM evoke_query(%s::regclass, %s, %s, NULL) h
            JOIN {}.{} d ON d.ctid = h.ctid
            ORDER BY h.score DESC, d.doc_id
        """).format(qident(schema), qident(doc_table)),
        (f'{schema}.{doc_table}_bm25_idx', query_text, k),
    ).fetchall()
    return [str(row[0]) for row in rows]


def query_dense(
    conn: psycopg.Connection[Any],
    *,
    schema: str,
    doc_table: str,
    query_embedding: str,
    k: int,
) -> list[str]:
    rows = conn.execute(
        sql.SQL("""
            SELECT doc_id
            FROM (
                SELECT
                    doc_id,
                    embedding <=> %s::halfvec AS distance
                FROM {}.{}
                WHERE vector_norm(embedding::vector) > 0
                ORDER BY distance
                LIMIT %s
            ) AS nearest
            ORDER BY distance, doc_id
        """).format(qident(schema), qident(doc_table)),
        (query_embedding, k),
    ).fetchall()
    return [str(row[0]) for row in rows]


_WORKER_LOCAL = threading.local()


def worker_connection(
    dsn: str,
    *,
    vector_probes: int | None,
) -> psycopg.Connection[Any]:
    conn = getattr(_WORKER_LOCAL, 'connection', None)
    if conn is None or conn.closed:
        conn = psycopg.connect(dsn, prepare_threshold=PREPARE_THRESHOLD)
        conn.execute(
            'SELECT set_config(%s, %s, false)',
            ('statement_timeout', '0'),
        )
        if vector_probes is not None:
            conn.execute(
                'SELECT set_config(%s, %s, false)',
                ('vchordrq.probes', str(vector_probes)),
            )
        _WORKER_LOCAL.connection = conn
    return conn


def query_source_worker(
    args: argparse.Namespace,
    *,
    source: str,
    schema: str,
    doc_table: str,
    query: dict[str, Any],
    vector_probes: int | None,
) -> list[str]:
    conn = worker_connection(args.dsn, vector_probes=vector_probes)
    if source == 'bm25':
        return query_bm25(
            conn,
            schema=schema,
            doc_table=doc_table,
            query_text=plain_text_to_raw_terms(query['text']),
            k=args.k,
        )
    return query_dense(
        conn,
        schema=schema,
        doc_table=doc_table,
        query_embedding=query['embedding'],
        k=args.k,
    )


def dcg(relevance: list[float]) -> float:
    total = 0.0
    for offset, rel in enumerate(relevance, start=2):
        total += (math.pow(2.0, rel) - 1.0) / math.log2(offset)
    return total


def metrics_for_query(
    ranked_docs: list[str],
    qrels: dict[str, float],
    *,
    candidate_k: int,
) -> dict[str, float]:
    if not qrels:
        return {
            'candidate_upper_bound': 0.0,
            'map_at_100': 0.0,
            'mrr_at_20': 0.0,
            'ndcg_at_10': 0.0,
            'recall_at_100': 0.0,
        }
    rel_count = len(qrels)
    top10_rels = [qrels.get(doc_id, 0.0) for doc_id in ranked_docs[:10]]
    ideal_rels = sorted(qrels.values(), reverse=True)[:10]
    ideal = dcg(ideal_rels)
    ndcg = dcg(top10_rels) / ideal if ideal > 0.0 else 0.0

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_docs[:100], start=1):
        if doc_id in qrels:
            hits += 1
            precision_sum += hits / rank
    average_precision = precision_sum / rel_count if rel_count else 0.0

    reciprocal = 0.0
    for rank, doc_id in enumerate(ranked_docs[:20], start=1):
        if doc_id in qrels:
            reciprocal = 1.0 / rank
            break
    candidate_hits = sum(
        1
        for doc_id in ranked_docs[:candidate_k]
        if doc_id in qrels
    )
    return {
        'candidate_upper_bound': candidate_hits / rel_count,
        'map_at_100': average_precision,
        'mrr_at_20': reciprocal,
        'ndcg_at_10': ndcg,
        'recall_at_100': hits / rel_count,
    }


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.k <= 0:
        raise ValueError('--k must be positive')
    if args.workers <= 0:
        raise ValueError('--workers must be positive')
    started = time.monotonic()
    sources = parse_csv(args.sources)
    unsupported = sorted(set(sources) - {'bm25', 'dense'})
    if unsupported:
        raise ValueError(f'unsupported sources: {unsupported}')

    rows: list[dict[str, Any]] = []
    dense_rankings: list[dict[str, Any]] = []
    with psycopg.connect(
        args.dsn,
        prepare_threshold=PREPARE_THRESHOLD,
    ) as conn:
        conn.execute('SELECT set_config(%s, %s, false)', ('statement_timeout', '0'))
        dataset_info = fetch_dataset_info(
            conn,
            schema=args.schema,
            dataset=args.dataset,
        )
        doc_table = str(dataset_info['table_name'])
        queries = load_queries(
            conn,
            schema=args.schema,
            dataset=args.dataset,
            limit=0,
        )
        queries = filter_queries(
            queries,
            query_ids=read_query_ids(args.query_ids_file),
        )
        if args.limit_queries > 0:
            queries = queries[:args.limit_queries]
        qrels = load_qrels(conn, schema=args.schema, dataset=args.dataset)

        for source in sources:
            query_rows: list[dict[str, Any]] = []
            source_queries = [
                query for query in queries
                if query['query_id'] in qrels
            ]
            vector_probes = dataset_info['vector_probes']
            if source == 'dense' and args.workers == 1:
                if vector_probes is None:
                    conn.execute('RESET vchordrq.probes')
                else:
                    conn.execute(
                        'SELECT set_config(%s, %s, false)',
                        ('vchordrq.probes', str(vector_probes)),
                    )

            def run_query(query: dict[str, Any]) -> list[str]:
                if args.workers > 1:
                    return query_source_worker(
                        args,
                        source=source,
                        schema=args.schema,
                        doc_table=doc_table,
                        query=query,
                        vector_probes=vector_probes,
                    )
                if source == 'bm25':
                    return query_bm25(
                        conn,
                        schema=args.schema,
                        doc_table=doc_table,
                        query_text=plain_text_to_raw_terms(query['text']),
                        k=args.k,
                    )
                return query_dense(
                    conn,
                    schema=args.schema,
                    doc_table=doc_table,
                    query_embedding=query['embedding'],
                    k=args.k,
                )

            def consume_results(
                ranked_results: Any,
            ) -> None:
                for query, ranked_docs in zip(
                    source_queries,
                    ranked_results,
                ):
                    metrics = metrics_for_query(
                        ranked_docs,
                        qrels[query['query_id']],
                        candidate_k=args.k,
                    )
                    if (
                        source == 'dense'
                        and args.dense_ranking_jsonl is not None
                    ):
                        dense_rankings.append({
                            'doc_ids': ranked_docs[:args.dense_ranking_k],
                            'query_id': query['query_id'],
                        })
                    query_rows.append({
                        'candidate_count': len(ranked_docs),
                        'metrics': metrics,
                        'query_id': query['query_id'],
                    })
                    if (
                        args.progress_every > 0
                        and len(query_rows) % args.progress_every == 0
                    ):
                        elapsed = time.monotonic() - started
                        print(
                            '[native-baseline] '
                            f'{args.dataset} {source} '
                            f'{len(query_rows)}/{len(source_queries)} '
                            f'elapsed_s={elapsed:.1f}',
                            file=sys.stderr,
                            flush=True,
                        )

            if args.workers == 1:
                consume_results(map(run_query, source_queries))
            else:
                with ThreadPoolExecutor(
                    max_workers=args.workers,
                ) as executor:
                    consume_results(executor.map(run_query, source_queries))
            summary = {
                key: mean([
                    float(row['metrics'][key])
                    for row in query_rows
                ])
                for key in METRIC_KEYS
            }
            summary['query_count'] = len(query_rows)
            rows.append({
                'query_rows': query_rows,
                'source': source,
                'summary': summary,
            })

    return {
        'dataset': args.dataset,
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'k': args.k,
        'query_limit': args.limit_queries,
        'prepare_threshold': PREPARE_THRESHOLD,
        'query_ids_file': (
            str(args.query_ids_file) if args.query_ids_file else None
        ),
        'workers': args.workers,
        'dense_rankings': dense_rankings,
        'rows': rows,
        'schema': args.schema,
    }


def metric_cell(value: Any) -> str:
    return '-' if value is None else f'{float(value):.6f}'


def render_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        '# Native BEIR15 Baselines',
        '',
        f"- Dataset: `{payload['dataset']}`",
        '',
        '| Source | Queries | NDCG@10 | MAP@100 | R@100 | MRR@20 | CUB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in payload['rows']:
        summary = row['summary']
        lines.append(
            f'| `{row["source"]}` | {summary["query_count"]} | '
            f'{metric_cell(summary.get("ndcg_at_10"))} | '
            f'{metric_cell(summary.get("map_at_100"))} | '
            f'{metric_cell(summary.get("recall_at_100"))} | '
            f'{metric_cell(summary.get("mrr_at_20"))} | '
            f'{metric_cell(summary.get("candidate_upper_bound"))} |'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    args = parse_args()
    payload = evaluate(args)
    dense_rankings = payload.pop('dense_rankings', [])
    if args.dense_ranking_jsonl is not None:
        args.dense_ranking_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.dense_ranking_jsonl.write_text(
            ''.join(
                json.dumps(row, sort_keys=True) + '\n'
                for row in dense_rankings
            ),
            encoding='utf-8',
        )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    if args.output_md is not None:
        render_markdown(args.output_md, payload)
    print(json.dumps({
        row['source']: row['summary']
        for row in payload['rows']
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
