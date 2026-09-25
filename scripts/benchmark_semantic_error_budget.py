#!/usr/bin/env python3
"""Measure semantic posting work savings against retrieval quality."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg import sql

from benchmark_page_native_block_cost import expand_query
from benchmark_semantic_work_control import bind_probes, call_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--queries-jsonl', type=Path, required=True)
    parser.add_argument('--qrels-json', type=Path, required=True)
    parser.add_argument('--lexical-dims', type=int, required=True)
    parser.add_argument('--semantic-dims', type=int, required=True)
    parser.add_argument('--field-count', type=int, default=1)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--support-ratio', type=float, default=0.0)
    parser.add_argument('--skip-work-probe', action='store_true')
    parser.add_argument(
        '--strategy',
        choices=('term-budget', 'impact-floor', 'df-limit'),
        default='term-budget',
    )
    parser.add_argument(
        '--semantic-work-target-postings',
        type=int,
        default=0,
        help='Keep DF-limited queries exact below this semantic work.',
    )
    parser.add_argument(
        '--budget-ratio',
        type=float,
        action='append',
        dest='budget_ratios',
    )
    parser.add_argument(
        '--candidate-multiplier',
        type=int,
        action='append',
        dest='candidate_multipliers',
    )
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def percentile(values: Iterable[float], ratio: float) -> float:
    ordered = sorted(values)

    if not ordered:
        return 0.0
    position = math.ceil(ratio * len(ordered)) - 1
    return ordered[max(0, min(position, len(ordered) - 1))]


def load_queries(path: Path, limit: int) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []

    with path.open(encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = row.get('id')
            text = row.get('text') or row.get('query')
            if not isinstance(query_id, str) or not isinstance(text, str):
                raise ValueError('query row must contain string id and text')
            queries.append({'id': query_id, 'text': text})
            if limit > 0 and len(queries) >= limit:
                break
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding='utf-8') as source:
        raw = json.load(source)
    return {
        str(query_id): {
            str(document_id): float(relevance)
            for document_id, relevance in documents.items()
        }
        for query_id, documents in raw.items()
    }


def resolve_table(
    cursor: psycopg.Cursor[Any],
    index_name: str,
) -> tuple[str, str]:
    cursor.execute(
        """
        SELECT namespace.nspname, relation.relname
        FROM pg_index index_catalog
        JOIN pg_class relation
          ON relation.oid = index_catalog.indrelid
        JOIN pg_namespace namespace
          ON namespace.oid = relation.relnamespace
        WHERE index_catalog.indexrelid = %s::regclass
        """,
        (index_name,),
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f'index does not exist: {index_name}')
    return str(row[0]), str(row[1])


def encode_queries(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    queries: list[dict[str, str]],
    lexical_dims: int,
    semantic_dims: int,
    field_count: int,
) -> list[dict[str, Any]]:
    encoded_queries: list[dict[str, Any]] = []

    for query in queries:
        cursor.execute(
            'SELECT evoke_encode_text_internal(%s::regclass, %s)',
            (index_name, query['text']),
        )
        encoded = cursor.fetchone()[0]
        query_ids, query_weights = expand_query(
            encoded,
            lexical_dims,
            semantic_dims,
            field_count,
        )
        encoded_queries.append(
            {
                **query,
                'ids': query_ids,
                'weights': query_weights,
            }
        )
    return encoded_queries


def search(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    table_name: tuple[str, str],
    query_text: str,
    k: int,
) -> tuple[list[tuple[str, float]], float]:
    statement = sql.SQL(
        """
        SELECT document.doc_id::text, hit.score::double precision
        FROM evoke_query(%s::regclass, %s, %s) AS hit
        JOIN {}.{} AS document ON document.ctid = hit.ctid
        ORDER BY hit.score DESC, hit.doc_id
        """
    ).format(sql.Identifier(table_name[0]), sql.Identifier(table_name[1]))
    started = time.perf_counter()
    cursor.execute(statement, (index_name, query_text, k))
    rows = [(str(row[0]), float(row[1])) for row in cursor.fetchall()]
    return rows, (time.perf_counter() - started) * 1000.0


def query_metrics(
    ranking: list[tuple[str, float]],
    qrels: dict[str, float],
) -> dict[str, float]:
    relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0.0}
    ranked_ids = [doc_id for doc_id, _score in ranking]
    recall_hits = sum(doc_id in relevant for doc_id in ranked_ids[:100])
    precision_sum = 0.0
    ap_hits = 0
    reciprocal_rank = 0.0
    dcg = 0.0

    for rank, doc_id in enumerate(ranked_ids[:100], start=1):
        grade = qrels.get(doc_id, 0.0)
        if grade > 0.0:
            ap_hits += 1
            precision_sum += ap_hits / rank
            if rank <= 20 and reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
        if rank <= 10 and grade > 0.0:
            dcg += (2.0**grade - 1.0) / math.log2(rank + 1.0)
    ideal_grades = sorted(qrels.values(), reverse=True)[:10]
    ideal_dcg = sum(
        (2.0**grade - 1.0) / math.log2(rank + 1.0)
        for rank, grade in enumerate(ideal_grades, start=1)
        if grade > 0.0
    )
    denominator = len(relevant)
    return {
        'recall_at_100': recall_hits / denominator if denominator else 0.0,
        'map_at_100': precision_sum / denominator if denominator else 0.0,
        'mrr_at_20': reciprocal_rank,
        'ndcg_at_10': dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def mean_metric(rows: list[dict[str, float]], name: str) -> float:
    return statistics.fmean(row[name] for row in rows) if rows else 0.0


def evaluate_ratio(
    cursor: psycopg.Cursor[Any],
    index_name: str,
    table_name: tuple[str, str],
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, float]],
    k: int,
    ratio: float,
    baseline_rankings: dict[str, list[str]],
    baseline_query_metrics: dict[str, dict[str, float]],
    strategy: str,
    support_ratio: float,
    candidate_multipliers: list[int],
    skip_work_probe: bool,
    semantic_work_target_postings: int,
) -> tuple[
    dict[str, Any],
    dict[str, list[str]],
    dict[str, dict[str, float]],
]:
    setting_name = {
        'term-budget': 'evoke.test_query_semantic_error_budget_ratio',
        'impact-floor': 'evoke.test_query_semantic_impact_floor_ratio',
        'df-limit': 'evoke.test_query_max_df_ratio',
    }[strategy]
    cursor.execute(
        'SELECT set_config(%s, %s, false)',
        (setting_name, format(ratio, '.17g')),
    )
    cursor.execute(
        'SELECT set_config(%s, %s, false)',
        (
            'evoke.test_query_semantic_work_target_postings',
            str(semantic_work_target_postings),
        ),
    )
    cursor.execute(
        'SELECT set_config(%s, %s, false)',
        (
            'evoke.test_query_semantic_min_support_ratio',
            format(support_ratio, '.17g'),
        ),
    )
    rows: list[dict[str, float]] = []
    latencies: list[float] = []
    overlaps: list[float] = []
    pruned_postings: list[int] = []
    pruned_terms: list[int] = []
    omitted_fractions: list[float] = []
    candidate_coverages: dict[int, list[float]] = {
        multiplier: [] for multiplier in candidate_multipliers
    }
    required_candidate_ks: list[int] = []
    rankings: dict[str, list[str]] = {}
    query_metric_rows: dict[str, dict[str, float]] = {}
    search_k = k

    if baseline_rankings and candidate_multipliers:
        search_k = k * max(candidate_multipliers)

    for query in queries:
        ranking, latency_ms = search(
            cursor,
            index_name,
            table_name,
            query['text'],
            search_k,
        )
        ranked_ids = [doc_id for doc_id, _score in ranking]
        rankings[query['id']] = ranked_ids[:k]
        latencies.append(latency_ms)
        metric_row = query_metrics(ranking, qrels.get(query['id'], {}))
        rows.append(metric_row)
        query_metric_rows[query['id']] = metric_row
        if query['id'] in baseline_rankings:
            baseline = baseline_rankings[query['id']]
            overlaps.append(
                len(set(ranked_ids[:k]) & set(baseline)) /
                max(1, len(baseline))
            )
            baseline_set = set(baseline)
            approximate_ranks = {
                document_id: rank
                for rank, document_id in enumerate(ranked_ids, start=1)
            }
            required_candidate_ks.append(
                max(
                    (
                        approximate_ranks.get(document_id, search_k + 1)
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
        if skip_work_probe:
            continue
        stats, _probe_ms = call_probe(
            cursor,
            'evoke_query_topk',
            index_name,
            query,
            k,
        )
        if strategy == 'term-budget':
            pruned_postings.append(
                int(stats.get('query_error_budget_pruned_postings', 0))
            )
            pruned_terms.append(
                int(stats.get('query_error_budget_pruned_term_count', 0))
            )
            total_bound = float(
                stats.get('query_semantic_total_absolute_bound', 0.0)
            )
            omitted_bound = float(
                stats.get('query_semantic_omitted_absolute_bound', 0.0)
            )
            omitted_fractions.append(
                omitted_bound / total_bound if total_bound > 0.0 else 0.0
            )
        elif strategy == 'impact-floor':
            pruned_postings.append(
                int(
                    stats.get(
                        'query_impact_floor_omitted_postings',
                        0,
                    )
                )
            )
            pruned_terms.append(0)
            omitted_fractions.append(ratio)
        else:
            pruned_postings.append(
                int(stats.get('query_df_pruned_postings', 0))
            )
            pruned_terms.append(
                int(stats.get('query_df_pruned_term_count', 0))
            )

    summary = {
        'budget_ratio': ratio,
        'strategy': strategy,
        'support_ratio': support_ratio,
        'semantic_work_target_postings': semantic_work_target_postings,
        'query_count': len(rows),
        'metrics': {
            name: mean_metric(rows, name)
            for name in (
                'ndcg_at_10',
                'map_at_100',
                'recall_at_100',
                'mrr_at_20',
            )
        },
        'baseline_overlap_at_100': (
            statistics.fmean(overlaps) if overlaps else 1.0
        ),
        'latency_ms': {
            'p50': statistics.median(latencies),
            'p95': percentile(latencies, 0.95),
        },
        'mean_pruned_postings': (
            statistics.fmean(pruned_postings)
            if pruned_postings else None
        ),
        'mean_pruned_terms': (
            statistics.fmean(pruned_terms) if pruned_terms else None
        ),
        'mean_omitted_bound_fraction': (
            statistics.fmean(omitted_fractions)
            if omitted_fractions else None
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
                'query_count': len(candidate_coverages[multiplier]),
            }
            for multiplier in candidate_multipliers
        }
    if baseline_query_metrics:
        summary['query_level_delta'] = {}
        for name in (
            'ndcg_at_10',
            'map_at_100',
            'recall_at_100',
            'mrr_at_20',
        ):
            deltas = [
                query_metric_rows[query_id][name] -
                baseline_query_metrics[query_id][name]
                for query_id in query_metric_rows
            ]
            summary['query_level_delta'][name] = {
                'gain_queries': sum(delta > 0.0 for delta in deltas),
                'harm_queries': sum(delta < 0.0 for delta in deltas),
                'worst': min(deltas, default=0.0),
                'best': max(deltas, default=0.0),
            }
    return summary, rankings, query_metric_rows


def main() -> int:
    args = parse_args()
    ratios = args.budget_ratios or [0.0, 0.002, 0.005, 0.01, 0.02]
    baseline_ratio = 1.0 if args.strategy == 'df-limit' else 0.0
    if args.strategy == 'df-limit' and args.budget_ratios is None:
        ratios = [0.35]
    if not ratios or ratios[0] != baseline_ratio:
        ratios = [baseline_ratio, *ratios]
    if args.strategy == 'df-limit':
        if any(
            not math.isfinite(ratio) or not 0.0 < ratio <= 1.0
            for ratio in ratios
        ):
            raise ValueError('DF ratios must be finite and in (0, 1]')
    elif any(
        not math.isfinite(ratio) or ratio < 0.0
        for ratio in ratios
    ):
        raise ValueError('budget ratios must be finite and non-negative')
    candidate_multipliers = sorted(set(args.candidate_multipliers or []))
    if any(multiplier <= 0 for multiplier in candidate_multipliers):
        raise ValueError('candidate multipliers must be positive')
    if (
        not math.isfinite(args.support_ratio) or
        not 0.0 <= args.support_ratio <= 1.0
    ):
        raise ValueError('support ratio must be between zero and one')
    if args.semantic_work_target_postings < 0:
        raise ValueError('semantic work target must be non-negative')
    queries = load_queries(args.queries_jsonl, args.query_limit)
    qrels = load_qrels(args.qrels_json)
    result: dict[str, Any] = {
        'index_name': args.index,
        'strategy': args.strategy,
        'support_ratio': args.support_ratio,
        'semantic_work_target_postings': (
            args.semantic_work_target_postings
        ),
        'k': args.k,
        'query_count': len(queries),
        'candidate_multipliers': candidate_multipliers,
        'ratios': [],
    }

    with psycopg.connect(args.dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            bind_probes(cursor)
            table_name = resolve_table(cursor, args.index)
            encoded = encode_queries(
                cursor,
                args.index,
                queries,
                args.lexical_dims,
                args.semantic_dims,
                args.field_count,
            )
            baseline_rankings: dict[str, list[str]] = {}
            baseline_query_metrics: dict[str, dict[str, float]] = {}
            baseline_metrics: dict[str, float] | None = None
            for ratio in ratios:
                summary, rankings, query_metrics_by_id = evaluate_ratio(
                    cursor,
                    args.index,
                    table_name,
                    encoded,
                    qrels,
                    args.k,
                    ratio,
                    baseline_rankings,
                    baseline_query_metrics,
                    args.strategy,
                    args.support_ratio,
                    candidate_multipliers,
                    args.skip_work_probe,
                    args.semantic_work_target_postings,
                )
                if not baseline_rankings:
                    baseline_rankings = rankings
                    baseline_query_metrics = query_metrics_by_id
                    baseline_metrics = summary['metrics']
                    summary['baseline_overlap_at_100'] = 1.0
                assert baseline_metrics is not None
                summary['metric_delta'] = {
                    name: summary['metrics'][name] - baseline_metrics[name]
                    for name in baseline_metrics
                }
                result['ratios'].append(summary)
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
