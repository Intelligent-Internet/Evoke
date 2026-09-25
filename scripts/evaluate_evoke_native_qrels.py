#!/usr/bin/env python3
"""Evaluate relation-owned evoke indexes through the product search API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from evoke_plain_query import plain_text_to_raw_terms


DEFAULT_DSN = 'dbname=postgres'
MAP_CUTOFF = 100
METRIC_KEYS = (
    'candidate_upper_bound',
    'map_at_100',
    'mrr_at_20',
    'ndcg_at_10',
    'recall_at_100',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate current relation-owned evoke indexes exclusively '
            'through evoke_query.'
        ),
    )
    parser.add_argument('--dsn', default=DEFAULT_DSN)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--table', required=True)
    parser.add_argument('--id-column', default='doc_id')
    parser.add_argument(
        '--index',
        action='append',
        required=True,
        metavar='LABEL=REGCLASS',
    )
    parser.add_argument('--queries-jsonl', type=Path, required=True)
    parser.add_argument('--qrels-json', type=Path, required=True)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--limit-queries', type=int, default=0)
    parser.add_argument('--progress-every', type=int, default=0)
    parser.add_argument('--expected-documents', type=int)
    parser.add_argument('--expected-queries', type=int)
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--output-md', type=Path)
    return parser.parse_args()


def parse_index_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    labels: set[str] = set()
    for value in values:
        label, separator, index_name = value.partition('=')
        label = label.strip()
        index_name = index_name.strip()
        if not separator or not label or not index_name:
            raise ValueError(
                f'invalid --index {value!r}; expected LABEL=REGCLASS'
            )
        if label in labels:
            raise ValueError(f'duplicate index label: {label}')
        labels.add(label)
        specs.append((label, index_name))
    return specs


def validate_expected_count(
    *,
    label: str,
    actual: int,
    expected: int | None,
) -> None:
    if expected is not None and expected <= 0:
        raise ValueError(f'--expected-{label} must be positive')
    if expected is not None and actual != expected:
        raise RuntimeError(
            f'expected {label} count mismatch: '
            f'expected={expected}, loaded={actual}'
        )


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('qrels JSON must be an object keyed by query id')
    qrels: dict[str, dict[str, float]] = {}
    for query_id, judgments in raw.items():
        if not isinstance(judgments, dict):
            raise ValueError(f'qrels for {query_id!r} must be an object')
        positive = {
            str(doc_id): float(score)
            for doc_id, score in judgments.items()
            if float(score) > 0.0
        }
        if positive:
            qrels[str(query_id)] = positive
    return qrels


def load_queries(
    path: Path,
    *,
    qrels: dict[str, dict[str, float]],
    limit: int,
) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get('id', row.get('query_id', '')))
            query_text = row.get('text', row.get('text_content'))
            if not query_id or not isinstance(query_text, str):
                raise ValueError(
                    f'invalid query row at {path}:{line_number}'
                )
            if query_id not in qrels:
                continue
            if query_id in seen:
                raise ValueError(f'duplicate query id: {query_id}')
            seen.add(query_id)
            queries.append({'query_id': query_id, 'text': query_text})
            if limit > 0 and len(queries) >= limit:
                break
    return queries


def dcg(relevance: list[float]) -> float:
    return sum(
        (math.pow(2.0, score) - 1.0) / math.log2(rank + 1)
        for rank, score in enumerate(relevance, start=1)
    )


def metrics_for_query(
    ranked_docs: list[str],
    qrels: dict[str, float],
    *,
    candidate_k: int,
) -> dict[str, float]:
    relevant_count = len(qrels)
    if relevant_count == 0:
        return {key: 0.0 for key in METRIC_KEYS}

    top10_relevance = [qrels.get(doc_id, 0.0) for doc_id in ranked_docs[:10]]
    ideal_relevance = sorted(qrels.values(), reverse=True)[:10]
    ideal_dcg = dcg(ideal_relevance)

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(
        ranked_docs[:MAP_CUTOFF],
        start=1,
    ):
        if doc_id in qrels:
            hits += 1
            precision_sum += hits / rank

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(ranked_docs[:20], start=1):
        if doc_id in qrels:
            reciprocal_rank = 1.0 / rank
            break

    candidate_hits = sum(
        doc_id in qrels for doc_id in ranked_docs[:candidate_k]
    )
    return {
        'candidate_upper_bound': candidate_hits / relevant_count,
        'map_at_100': precision_sum / min(relevant_count, MAP_CUTOFF),
        'mrr_at_20': reciprocal_rank,
        'ndcg_at_10': (
            dcg(top10_relevance) / ideal_dcg if ideal_dcg > 0.0 else 0.0
        ),
        'recall_at_100': hits / relevant_count,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def relevant_rank_diagnostics(
    ranked_docs: list[str],
    qrels: dict[str, float],
) -> dict[str, int | None]:
    ranks = {doc_id: rank for rank, doc_id in enumerate(ranked_docs, start=1)}
    return {
        doc_id: ranks.get(doc_id)
        for doc_id in sorted(qrels)
    }


def find_extension_schema(conn: psycopg.Connection[Any]) -> str:
    row = conn.execute("""
        SELECT namespace.nspname
        FROM pg_catalog.pg_extension AS extension
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = extension.extnamespace
        WHERE extension.extname = 'evoke'
    """).fetchone()
    if row is None:
        raise RuntimeError('evoke extension is not installed')
    return str(row[0])


def file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {
        'bytes': path.stat().st_size,
        'path': str(path),
        'sha256': digest.hexdigest(),
    }


def validate_index_metadata(metadata: dict[str, Any]) -> None:
    status = metadata['index_status']
    options = metadata['index_options']
    generation = status.get('generation') or {}
    failures: list[str] = []
    if status.get('query_ready') is not True:
        failures.append('query_ready is not true')
    if options.get('payload_owner') != 'index_relation':
        failures.append('payload_owner is not index_relation')
    if generation.get('atomic') is not True:
        failures.append('generation is not atomic')
    if generation.get('valid') is not True:
        failures.append('generation is not valid')
    if failures:
        raise RuntimeError(
            f'index {metadata["index_name"]} failed product gate: '
            + '; '.join(failures)
        )


def index_metadata(
    conn: psycopg.Connection[Any],
    *,
    extension_schema: str,
    index_name: str,
) -> dict[str, Any]:
    row = conn.execute(
        sql.SQL("""
        SELECT
            {}.evoke_index_status(%s::regclass),
            {}.evoke_index_options(%s::regclass),
            pg_relation_size(%s::regclass)
        """).format(
            sql.Identifier(extension_schema),
            sql.Identifier(extension_schema),
        ),
        (index_name, index_name, index_name),
    ).fetchone()
    if row is None:
        raise RuntimeError(f'failed to inspect index: {index_name}')
    metadata = {
        'index_name': index_name,
        'index_options': row[1],
        'index_size_bytes': int(row[2]),
        'index_status': row[0],
    }
    validate_index_metadata(metadata)
    return metadata


def query_index(
    conn: psycopg.Connection[Any],
    *,
    schema: str,
    table: str,
    id_column: str,
    extension_schema: str,
    index_name: str,
    query_text: str,
    k: int,
) -> tuple[list[str], list[float], float]:
    statement = sql.SQL("""
        SELECT source.{}, hit.score
        FROM {}.evoke_query(%s::regclass, %s, %s) AS hit
        JOIN {}.{} AS source ON source.ctid = hit.ctid
        ORDER BY hit.score DESC, source.{}
        LIMIT %s
    """).format(
        sql.Identifier(id_column),
        sql.Identifier(extension_schema),
        sql.Identifier(schema),
        sql.Identifier(table),
        sql.Identifier(id_column),
    )
    started = time.perf_counter()
    rows = conn.execute(
        statement,
        (index_name, query_text, k, k),
    ).fetchall()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return (
        [str(row[0]) for row in rows],
        [float(row[1]) for row in rows],
        elapsed_ms,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.k <= 0:
        raise ValueError('--k must be positive')
    if args.limit_queries < 0:
        raise ValueError('--limit-queries cannot be negative')
    validate_expected_count(
        label='documents',
        actual=args.expected_documents or 1,
        expected=args.expected_documents,
    )
    validate_expected_count(
        label='queries',
        actual=args.expected_queries or 1,
        expected=args.expected_queries,
    )

    specs = parse_index_specs(args.index)
    qrels = load_qrels(args.qrels_json)
    queries = load_queries(
        args.queries_jsonl,
        qrels=qrels,
        limit=args.limit_queries,
    )
    if not queries:
        raise ValueError('no qrels-bearing queries were loaded')
    validate_expected_count(
        label='queries',
        actual=len(queries),
        expected=args.expected_queries,
    )

    run_started = time.monotonic()
    methods: list[dict[str, Any]] = []
    with psycopg.connect(args.dsn, prepare_threshold=0) as conn:
        conn.execute(
            'SELECT set_config(%s, %s, false)',
            ('statement_timeout', '0'),
        )
        document_count = int(conn.execute(
            sql.SQL('SELECT count(*) FROM {}.{}').format(
                sql.Identifier(args.schema),
                sql.Identifier(args.table),
            )
        ).fetchone()[0])
        validate_expected_count(
            label='documents',
            actual=document_count,
            expected=args.expected_documents,
        )
        extension_schema = find_extension_schema(conn)
        server_row = conn.execute(
            sql.SQL("""
                SELECT
                    current_database(),
                    current_setting('server_version'),
                    extension.extversion
                FROM pg_catalog.pg_extension AS extension
                WHERE extension.extname = 'evoke'
            """)
        ).fetchone()
        if server_row is None:
            raise RuntimeError('failed to read evoke extension metadata')

        for label, index_name in specs:
            metadata = index_metadata(
                conn,
                extension_schema=extension_schema,
                index_name=index_name,
            )
            sae_enabled = bool(
                metadata['index_options'].get('sae_enabled')
            )
            query_input_mode = (
                'plain_text'
                if sae_enabled
                else 'plain_text_to_raw_terms'
            )

            def product_query_text(value: str) -> str:
                if sae_enabled:
                    return value
                return plain_text_to_raw_terms(value)

            # Warm the runtime and immutable generation outside measurements.
            query_index(
                conn,
                schema=args.schema,
                table=args.table,
                id_column=args.id_column,
                extension_schema=extension_schema,
                index_name=index_name,
                query_text=product_query_text(queries[0]['text']),
                k=min(args.k, 10),
            )

            query_rows: list[dict[str, Any]] = []
            for offset, query in enumerate(queries, start=1):
                try:
                    ranked_docs, scores, elapsed_ms = query_index(
                        conn,
                        schema=args.schema,
                        table=args.table,
                        id_column=args.id_column,
                        extension_schema=extension_schema,
                        index_name=index_name,
                        query_text=product_query_text(query['text']),
                        k=args.k,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f'{label} query {query["query_id"]} failed'
                    ) from exc
                query_qrels = qrels[query['query_id']]
                query_rows.append({
                    'candidate_count': len(ranked_docs),
                    'latency_ms': elapsed_ms,
                    'metrics': metrics_for_query(
                        ranked_docs,
                        query_qrels,
                        candidate_k=args.k,
                    ),
                    'query_id': query['query_id'],
                    'relevant_ranks': relevant_rank_diagnostics(
                        ranked_docs,
                        query_qrels,
                    ),
                    'score_max': scores[0] if scores else None,
                    'score_min': scores[-1] if scores else None,
                })
                if args.progress_every > 0 and offset % args.progress_every == 0:
                    print(
                        f'[native-qrels] {args.dataset} {label} '
                        f'{offset}/{len(queries)}',
                        flush=True,
                    )

            latencies = [float(row['latency_ms']) for row in query_rows]
            summary = {
                key: mean([
                    float(row['metrics'][key])
                    for row in query_rows
                ])
                for key in METRIC_KEYS
            }
            summary.update({
                'candidate_count_mean': mean([
                    float(row['candidate_count'])
                    for row in query_rows
                ]),
                'latency_ms_mean': mean(latencies),
                'latency_ms_p50': percentile(latencies, 0.50),
                'latency_ms_p95': percentile(latencies, 0.95),
                'query_count': len(query_rows),
            })
            methods.append({
                'label': label,
                'metadata': metadata,
                'query_input_mode': query_input_mode,
                'query_rows': query_rows,
                'summary': summary,
            })

    return {
        'candidate_k': args.k,
        'dataset': args.dataset,
        'document_count': document_count,
        'elapsed_seconds': round(time.monotonic() - run_started, 3),
        'expected_document_count': args.expected_documents,
        'expected_query_count': args.expected_queries,
        'extension_schema': extension_schema,
        'id_column': args.id_column,
        'inputs': {
            'qrels': file_metadata(args.qrels_json),
            'queries': file_metadata(args.queries_jsonl),
        },
        'methods': methods,
        'qrels_query_count': len(qrels),
        'route': 'evoke_query',
        'schema': args.schema,
        'server': {
            'database': str(server_row[0]),
            'evoke_version': str(server_row[2]),
            'postgresql_version': str(server_row[1]),
        },
        'table': args.table,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f'# Native Qrels: {payload["dataset"]}',
        '',
        f'- Route: `{payload["route"]}`',
        f'- Documents: `{payload["document_count"]}`',
        f'- Candidate k: `{payload["candidate_k"]}`',
        '',
        '| Method | Queries | NDCG@10 | MAP@100 | Recall@100 | '
        'MRR@20 | CUB | p50 ms | p95 ms | Index bytes |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | '
        '---: | ---: |',
    ]
    for method in payload['methods']:
        summary = method['summary']
        metadata = method['metadata']
        lines.append(
            f'| `{method["label"]}` | {summary["query_count"]} | '
            f'{summary["ndcg_at_10"]:.6f} | '
            f'{summary["map_at_100"]:.6f} | '
            f'{summary["recall_at_100"]:.6f} | '
            f'{summary["mrr_at_20"]:.6f} | '
            f'{summary["candidate_upper_bound"]:.6f} | '
            f'{summary["latency_ms_p50"]:.3f} | '
            f'{summary["latency_ms_p95"]:.3f} | '
            f'{metadata["index_size_bytes"]} |'
        )
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    payload = evaluate(args)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            render_markdown(payload),
            encoding='utf-8',
        )
    print(json.dumps({
        method['label']: method['summary']
        for method in payload['methods']
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
