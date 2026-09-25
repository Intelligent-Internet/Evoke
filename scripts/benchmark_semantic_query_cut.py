#!/usr/bin/env python3
"""Measure query-component cuts as a sparse candidate generator."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import psycopg

from benchmark_semantic_error_budget import (
    encode_queries,
    load_queries,
    percentile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--queries-jsonl', type=Path, required=True)
    parser.add_argument('--lexical-dims', type=int, required=True)
    parser.add_argument('--semantic-dims', type=int, required=True)
    parser.add_argument('--field-count', type=int, default=1)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument(
        '--semantic-term-cap',
        type=int,
        action='append',
        dest='semantic_term_caps',
    )
    parser.add_argument(
        '--candidate-multiplier',
        type=int,
        action='append',
        dest='candidate_multipliers',
    )
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def cut_query(
    query: dict[str, Any],
    lexical_dimension_count: int,
    semantic_term_cap: int,
) -> dict[str, Any]:
    lexical: list[tuple[int, float]] = []
    semantic: list[tuple[int, float]] = []

    for query_id, weight in zip(
        query['ids'],
        query['weights'],
        strict=True,
    ):
        target = lexical if query_id < lexical_dimension_count else semantic
        target.append((int(query_id), float(weight)))
    if semantic_term_cap > 0 and len(semantic) > semantic_term_cap:
        semantic = sorted(
            semantic,
            key=lambda pair: (-abs(pair[1]), pair[0]),
        )[:semantic_term_cap]
    retained = sorted((*lexical, *semantic), key=lambda pair: pair[0])
    return {
        **query,
        'ids': [query_id for query_id, _weight in retained],
        'weights': [weight for _query_id, weight in retained],
        'lexical_terms': len(lexical),
        'semantic_terms': len(semantic),
    }


def bind_fast_probe(cursor: psycopg.Cursor[Any]) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION pg_temp.evoke_query_topk_fast(
            regclass,
            int4[],
            real[],
            int4,
            boolean
        ) RETURNS jsonb
        AS '$libdir/evoke', 'evoke_test_query_page_native_topk'
        LANGUAGE C STRICT
        """
    )


def call_fast_probe(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    query: dict[str, Any],
    k: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    cursor.execute(
        """
        SELECT pg_temp.evoke_query_topk_fast(
            %s::regclass,
            %s::int4[],
            %s::real[],
            %s,
            true
        )
        """,
        (index_name, query['ids'], query['weights'], k),
    )
    row = cursor.fetchone()[0]
    if not isinstance(row, dict):
        raise RuntimeError('fast page-native probe returned no JSON object')
    return row, (time.perf_counter() - started) * 1000.0


def summarize_cap(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    encoded_queries: list[dict[str, Any]],
    lexical_dimension_count: int,
    semantic_term_cap: int,
    k: int,
    candidate_multipliers: list[int],
    baseline_rankings: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    search_k = k * max(candidate_multipliers, default=1)
    latencies: list[float] = []
    overlaps: list[float] = []
    semantic_terms: list[int] = []
    postings_examined: list[int] = []
    documents_examined: list[int] = []
    candidate_coverages = {
        multiplier: [] for multiplier in candidate_multipliers
    }
    required_candidate_ks: list[int] = []
    rankings: dict[str, list[str]] = {}

    for encoded_query in encoded_queries:
        query = cut_query(
            encoded_query,
            lexical_dimension_count,
            semantic_term_cap,
        )
        stats, latency_ms = call_fast_probe(
            cursor,
            index_name,
            query,
            search_k,
        )
        ranked_ids = [
            str(document_id)
            for document_id in stats['page_native_doc_ids']
        ]
        rankings[query['id']] = ranked_ids[:k]
        latencies.append(latency_ms)
        semantic_terms.append(query['semantic_terms'])
        postings_examined.append(int(stats['postings_examined']))
        documents_examined.append(int(stats['documents_examined']))
        baseline = baseline_rankings.get(query['id'])
        if baseline is not None:
            baseline_set = set(baseline)
            overlaps.append(
                len(set(ranked_ids[:k]) & baseline_set) /
                max(1, len(baseline_set))
            )
            ranks = {
                document_id: rank
                for rank, document_id in enumerate(ranked_ids, start=1)
            }
            required_candidate_ks.append(
                max(
                    (
                        ranks.get(document_id, search_k + 1)
                        for document_id in baseline_set
                    ),
                    default=0,
                )
            )
            for multiplier in candidate_multipliers:
                candidate_set = set(ranked_ids[:k * multiplier])
                candidate_coverages[multiplier].append(
                    len(candidate_set & baseline_set) /
                    max(1, len(baseline_set))
                )
    summary: dict[str, Any] = {
        'semantic_term_cap': semantic_term_cap,
        'query_count': len(encoded_queries),
        'mean_semantic_terms': statistics.fmean(semantic_terms),
        'latency_ms': {
            'p50': statistics.median(latencies),
            'p95': percentile(latencies, 0.95),
        },
        'baseline_overlap_at_100': (
            statistics.fmean(overlaps) if overlaps else 1.0
        ),
        'mean_postings_examined': (
            statistics.fmean(postings_examined)
            if postings_examined else None
        ),
        'mean_documents_examined': (
            statistics.fmean(documents_examined)
            if documents_examined else None
        ),
    }
    if baseline_rankings and candidate_multipliers:
        summary['exact_topk_required_candidate_k'] = {
            'p50': statistics.median(required_candidate_ks),
            'p95': percentile(required_candidate_ks, 0.95),
            'max': max(required_candidate_ks, default=0),
        }
        summary['candidate_completion'] = {
            str(multiplier): {
                'candidate_k': k * multiplier,
                'mean_exact_topk_coverage': statistics.fmean(
                    candidate_coverages[multiplier]
                ),
                'full_coverage_queries': sum(
                    coverage == 1.0
                    for coverage in candidate_coverages[multiplier]
                ),
            }
            for multiplier in candidate_multipliers
        }
    return summary, rankings


def main() -> int:
    args = parse_args()
    caps = sorted(set(args.semantic_term_caps or [8, 16, 24]))
    if any(cap <= 0 for cap in caps):
        raise ValueError('semantic term caps must be positive')
    candidate_multipliers = sorted(
        set(args.candidate_multipliers or [2, 4])
    )
    if any(multiplier <= 0 for multiplier in candidate_multipliers):
        raise ValueError('candidate multipliers must be positive')
    queries = load_queries(args.queries_jsonl, args.query_limit)
    result: dict[str, Any] = {
        'index_name': args.index,
        'k': args.k,
        'candidate_multipliers': candidate_multipliers,
        'caps': [],
    }

    with psycopg.connect(args.dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            bind_fast_probe(cursor)
            encoded_queries = encode_queries(
                cursor,
                args.index,
                queries,
                args.lexical_dims,
                args.semantic_dims,
                args.field_count,
            )
            baseline, baseline_rankings = summarize_cap(
                cursor,
                args.index,
                encoded_queries,
                args.lexical_dims * args.field_count,
                0,
                args.k,
                [],
                {},
            )
            baseline['semantic_term_cap'] = 0
            result['baseline'] = baseline
            print(json.dumps(baseline, sort_keys=True), flush=True)
            for cap in caps:
                summary, _rankings = summarize_cap(
                    cursor,
                    args.index,
                    encoded_queries,
                    args.lexical_dims * args.field_count,
                    cap,
                    args.k,
                    candidate_multipliers,
                    baseline_rankings,
                )
                result['caps'].append(summary)
                print(json.dumps(summary, sort_keys=True), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
