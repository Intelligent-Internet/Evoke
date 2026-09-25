#!/usr/bin/env python3
"""Benchmark one exact evoke binary stage and compare ranked parity."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from evaluate_evoke_native_qrels import (
    file_metadata,
    find_extension_schema,
    index_metadata,
    query_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a deterministic exact evoke latency stage.',
    )
    parser.add_argument('--dsn', default='dbname=postgres')
    parser.add_argument('--index', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--table', required=True)
    parser.add_argument('--id-column', required=True)
    parser.add_argument('--queries-json', type=Path, required=True)
    parser.add_argument('--binary-path', type=Path, required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--expected-documents', type=int, required=True)
    parser.add_argument('--expected-queries', type=int, required=True)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--score-tolerance', type=float, default=0.0001)
    parser.add_argument('--required-improvement', type=float, default=20.0)
    parser.add_argument('--baseline-json', type=Path)
    parser.add_argument('--output-json', type=Path, required=True)
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    rows = raw.get('rows') if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError('query source must contain a rows array')
    queries: list[dict[str, str]] = []
    for offset, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f'query row {offset} is not an object')
        query_id = str(row.get('source_id', row.get('id', '')))
        text = row.get('title', row.get('text'))
        if not query_id or not isinstance(text, str) or not text.strip():
            raise ValueError(f'invalid query row {offset}')
        queries.append({'query_id': query_id, 'text': text})
    return queries


def latency_summary(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError('at least two latency samples are required')
    quantiles = statistics.quantiles(values, n=20, method='inclusive')
    return {
        'count': len(values),
        'max_ms': max(values),
        'mean_ms': statistics.fmean(values),
        'min_ms': min(values),
        'p50_ms': statistics.median(values),
        'p95_ms': quantiles[18],
    }


def compare_rankings(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    score_tolerance: float,
    required_improvement: float,
) -> dict[str, Any]:
    baseline_rows = baseline.get('ranked_queries', [])
    candidate_rows = candidate.get('ranked_queries', [])
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError('baseline and candidate query counts differ')

    identity_mismatches = 0
    ranked_rows = 0
    max_score_delta = 0.0
    for expected, actual in zip(baseline_rows, candidate_rows, strict=True):
        if expected.get('query_id') != actual.get('query_id'):
            raise ValueError('baseline and candidate query order differs')
        expected_docs = expected.get('doc_ids', [])
        actual_docs = actual.get('doc_ids', [])
        expected_scores = expected.get('scores', [])
        actual_scores = actual.get('scores', [])
        ranked_rows += max(len(expected_docs), len(actual_docs))
        identity_mismatches += sum(
            left != right
            for left, right in zip(
                expected_docs,
                actual_docs,
                strict=False,
            )
        )
        identity_mismatches += abs(len(expected_docs) - len(actual_docs))
        if len(expected_scores) != len(actual_scores):
            max_score_delta = float('inf')
        else:
            for left, right in zip(
                expected_scores,
                actual_scores,
                strict=True,
            ):
                max_score_delta = max(
                    max_score_delta,
                    abs(float(left) - float(right)),
                )

    baseline_latency = baseline['latency']
    candidate_latency = candidate['latency']
    improvement = {
        key: (
            (float(baseline_latency[key]) - float(candidate_latency[key]))
            / float(baseline_latency[key])
            * 100.0
        )
        for key in ('p50_ms', 'p95_ms')
    }
    return {
        'allowed_absolute_score_delta': score_tolerance,
        'identity_mismatches': identity_mismatches,
        'improvement_percent': improvement,
        'max_absolute_score_delta': max_score_delta,
        'passed': (
            identity_mismatches == 0
            and max_score_delta <= score_tolerance
            and improvement['p50_ms'] >= required_improvement
            and improvement['p95_ms'] >= required_improvement
        ),
        'ranked_rows': ranked_rows,
        'required_improvement_percent': required_improvement,
    }


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    if args.k <= 0 or args.repeats <= 0:
        raise ValueError('--k and --repeats must be positive')
    queries = load_queries(args.queries_json)
    if len(queries) != args.expected_queries:
        raise RuntimeError(
            'expected query count mismatch: '
            f'expected={args.expected_queries}, loaded={len(queries)}'
        )

    samples: list[dict[str, Any]] = []
    ranked_queries: list[dict[str, Any]] = []
    with psycopg.connect(args.dsn, prepare_threshold=0) as conn:
        conn.execute(
            'SELECT set_config(%s, %s, false)',
            ('statement_timeout', '0'),
        )
        count = int(conn.execute(
            sql.SQL('SELECT count(*) FROM {}.{}').format(
                sql.Identifier(args.schema),
                sql.Identifier(args.table),
            )
        ).fetchone()[0])
        if count != args.expected_documents:
            raise RuntimeError(
                'expected document count mismatch: '
                f'expected={args.expected_documents}, loaded={count}'
            )
        extension_schema = find_extension_schema(conn)
        metadata = index_metadata(
            conn,
            extension_schema=extension_schema,
            index_name=args.index,
        )
        if metadata['index_options'].get('sae_enabled') is not True:
            raise RuntimeError('exact stage requires a semantic evoke index')

        for query in queries:
            query_index(
                conn,
                schema=args.schema,
                table=args.table,
                id_column=args.id_column,
                extension_schema=extension_schema,
                index_name=args.index,
                query_text=query['text'],
                k=args.k,
            )
            canonical_docs: list[str] | None = None
            canonical_scores: list[float] | None = None
            for repeat in range(1, args.repeats + 1):
                docs, scores, elapsed_ms = query_index(
                    conn,
                    schema=args.schema,
                    table=args.table,
                    id_column=args.id_column,
                    extension_schema=extension_schema,
                    index_name=args.index,
                    query_text=query['text'],
                    k=args.k,
                )
                if canonical_docs is None:
                    canonical_docs = docs
                    canonical_scores = scores
                elif docs != canonical_docs or scores != canonical_scores:
                    raise RuntimeError(
                        f'non-deterministic ranking for {query["query_id"]}'
                    )
                samples.append({
                    'elapsed_ms': elapsed_ms,
                    'query_id': query['query_id'],
                    'repeat': repeat,
                })
            ranked_queries.append({
                'doc_ids': canonical_docs,
                'query_id': query['query_id'],
                'scores': canonical_scores,
            })

    payload = {
        'binary': file_metadata(args.binary_path),
        'documents': args.expected_documents,
        'index': args.index,
        'k': args.k,
        'latency': latency_summary([
            float(sample['elapsed_ms'])
            for sample in samples
        ]),
        'metadata': metadata,
        'queries': len(queries),
        'query_source': file_metadata(args.queries_json),
        'ranked_queries': ranked_queries,
        'repeats': args.repeats,
        'samples': samples,
        'stage': args.stage,
    }
    if args.baseline_json is not None:
        baseline = json.loads(
            args.baseline_json.read_text(encoding='utf-8')
        )
        payload['comparison'] = compare_rankings(
            baseline,
            payload,
            score_tolerance=args.score_tolerance,
            required_improvement=args.required_improvement,
        )
    return payload


def main() -> int:
    args = parse_args()
    payload = run_stage(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps({
        'comparison': payload.get('comparison'),
        'latency': payload['latency'],
        'stage': payload['stage'],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
