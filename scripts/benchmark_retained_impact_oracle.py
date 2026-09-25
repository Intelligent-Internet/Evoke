#!/usr/bin/env python3
"""Evaluate retained-impact postings as a bounded high-DF oracle."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from benchmark_semantic_error_budget import (
    load_qrels,
    mean_metric,
    query_metrics,
)
from benchmark_seismic_npz_oracle import load_csr, load_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--accelerated-posting-mass', type=float, default=0.30)
    parser.add_argument(
        '--pruning-strategy',
        choices=('fixed-size', 'global-threshold'),
        default='fixed-size',
    )
    parser.add_argument('--max-fraction', type=float, default=1.5)
    parser.add_argument(
        '--retained-count',
        action='append',
        type=int,
        dest='retained_counts',
    )
    parser.add_argument(
        '--candidate-multiplier',
        action='append',
        type=int,
        dest='candidate_multipliers',
    )
    return parser.parse_args()


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)

    if not ordered:
        return 0.0
    position = math.ceil(ratio * len(ordered)) - 1
    return ordered[max(0, min(position, len(ordered) - 1))]


def topk_offsets(scores: np.ndarray, k: int) -> np.ndarray:
    count = min(k, scores.size)

    if count == 0:
        return np.empty(0, dtype=np.int64)
    unordered = np.argpartition(-scores, count - 1)[:count]
    order = np.lexsort((unordered, -scores[unordered]))
    return unordered[order].astype(np.int64, copy=False)


def load_dataset(
    root: Path,
    query_limit: int,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    list[str],
    list[str],
    dict[str, dict[str, float]],
]:
    manifest = json.loads((root / 'manifest.json').read_text())
    document_matrices: list[sparse.csr_matrix] = []
    document_ids: list[str] = []

    for shard in manifest['documents']['shards']:
        document_matrices.append(load_csr(root / shard['matrix']))
        document_ids.extend(load_ids(root / shard['ids'], 0))
    documents = sparse.vstack(
        document_matrices,
        format='csr',
        dtype=np.float32,
    )
    queries = load_csr(root / manifest['queries']['matrix'])
    query_ids = load_ids(root / manifest['queries']['ids'], query_limit)
    if query_limit > 0:
        queries = queries[:query_limit].tocsr()
    if documents.shape[0] != len(document_ids):
        raise ValueError('document matrix and ID counts differ')
    if queries.shape[0] != len(query_ids):
        raise ValueError('query matrix and ID counts differ')
    return (
        documents,
        queries,
        document_ids,
        query_ids,
        load_qrels(root / manifest['queries']['qrels']),
    )


def select_high_df_terms(
    document_frequencies: np.ndarray,
    target_mass: float,
) -> tuple[np.ndarray, float]:
    total = int(document_frequencies.sum())
    selected = np.zeros(document_frequencies.size, dtype=np.bool_)

    if total == 0:
        return selected, 0.0
    term_ids = np.flatnonzero(document_frequencies)
    order = np.lexsort(
        (term_ids, -document_frequencies[term_ids].astype(np.int64)),
    )
    ordered = term_ids[order]
    cumulative = np.cumsum(document_frequencies[ordered], dtype=np.uint64)
    target = math.ceil(target_mass * total)
    count = int(np.searchsorted(cumulative, target, side='left')) + 1
    selected[ordered[:count]] = True
    return selected, float(cumulative[count - 1] / total)


def mask_query_terms(
    queries: sparse.csr_matrix,
    selected: np.ndarray,
) -> sparse.csr_matrix:
    result = queries.copy()
    keep = selected[result.indices]

    result.data = result.data * keep
    result.eliminate_zeros()
    return result


def build_retained_matrix(
    documents: sparse.csr_matrix,
    selected_terms: np.ndarray,
    retained_count: int,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, float | None]:
    columns = documents.tocsc()
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    tail_maxima = np.zeros(documents.shape[1], dtype=np.float32)
    retained_counts = np.zeros(documents.shape[1], dtype=np.int64)

    for term_id in np.flatnonzero(selected_terms):
        start = int(columns.indptr[term_id])
        end = int(columns.indptr[term_id + 1])
        values = columns.data[start:end]
        rows = columns.indices[start:end]
        if values.size <= retained_count:
            retained = np.arange(values.size)
        else:
            retained = np.argpartition(
                -values,
                retained_count - 1,
            )[:retained_count]
            omitted = np.ones(values.size, dtype=np.bool_)
            omitted[retained] = False
            tail_maxima[term_id] = float(values[omitted].max(initial=0.0))
        row_parts.append(rows[retained])
        column_parts.append(
            np.full(retained.size, term_id, dtype=np.int32),
        )
        value_parts.append(values[retained])
        retained_counts[term_id] = retained.size
    retained_matrix = sparse.csr_matrix(
        (
            np.concatenate(value_parts) if value_parts else np.empty(0),
            (
                np.concatenate(row_parts) if row_parts else np.empty(0),
                (
                    np.concatenate(column_parts)
                    if column_parts
                    else np.empty(0)
                ),
            ),
        ),
        shape=documents.shape,
        dtype=np.float32,
    )
    retained_matrix.sort_indices()
    return retained_matrix, tail_maxima, retained_counts, None


def build_global_threshold_matrix(
    documents: sparse.csr_matrix,
    selected_terms: np.ndarray,
    retained_count: int,
    max_fraction: float,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, float | None]:
    """Apply Seismic-style global threshold pruning to selected terms."""
    columns = documents.tocsc()
    selected_ids = np.flatnonzero(selected_terms)
    selected_postings = int(
        sum(
            int(columns.indptr[term_id + 1] - columns.indptr[term_id])
            for term_id in selected_ids
        )
    )
    target = min(selected_postings, retained_count * selected_ids.size)
    max_list_len = max(1, int(retained_count * max_fraction))
    tail_maxima = np.zeros(documents.shape[1], dtype=np.float32)
    retained_counts = np.zeros(documents.shape[1], dtype=np.int64)
    if target == 0:
        return (
            sparse.csr_matrix(documents.shape, dtype=np.float32),
            tail_maxima,
            retained_counts,
            None,
        )

    all_values = np.concatenate(
        [
            columns.data[
                int(columns.indptr[term_id]):
                int(columns.indptr[term_id + 1])
            ]
            for term_id in selected_ids
        ]
    )
    threshold_offset = all_values.size - target
    threshold = float(np.partition(all_values, threshold_offset)[
        threshold_offset
    ])
    del all_values

    retained_by_term: list[tuple[int, np.ndarray]] = []
    total_retained = 0
    for term_id in selected_ids:
        start = int(columns.indptr[term_id])
        end = int(columns.indptr[term_id + 1])
        values = columns.data[start:end]
        rows = columns.indices[start:end]
        eligible = np.flatnonzero(values >= threshold)
        order = np.lexsort((rows[eligible], -values[eligible]))
        retained = eligible[order[:max_list_len]]
        retained_by_term.append((int(term_id), retained))
        total_retained += retained.size

    excess = max(0, total_retained - target)
    if excess > 0:
        for list_offset, (term_id, retained) in enumerate(retained_by_term):
            if excess == 0 or retained.size == 0:
                continue
            start = int(columns.indptr[term_id])
            values = columns.data[start:int(columns.indptr[term_id + 1])]
            tied = int(np.count_nonzero(values[retained] == threshold))
            trim = min(excess, tied)
            if trim > 0:
                retained_by_term[list_offset] = (
                    term_id,
                    retained[:-trim],
                )
                excess -= trim

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for term_id, retained in retained_by_term:
        start = int(columns.indptr[term_id])
        end = int(columns.indptr[term_id + 1])
        values = columns.data[start:end]
        rows = columns.indices[start:end]
        omitted = np.ones(values.size, dtype=np.bool_)
        omitted[retained] = False
        tail_maxima[term_id] = float(values[omitted].max(initial=0.0))
        retained_counts[term_id] = retained.size
        row_parts.append(rows[retained])
        column_parts.append(
            np.full(retained.size, term_id, dtype=np.int32),
        )
        value_parts.append(values[retained])

    retained_matrix = sparse.csr_matrix(
        (
            np.concatenate(value_parts) if value_parts else np.empty(0),
            (
                np.concatenate(row_parts) if row_parts else np.empty(0),
                (
                    np.concatenate(column_parts)
                    if column_parts
                    else np.empty(0)
                ),
            ),
        ),
        shape=documents.shape,
        dtype=np.float32,
    )
    retained_matrix.sort_indices()
    return retained_matrix, tail_maxima, retained_counts, threshold


def metric_summary(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, float]],
) -> dict[str, float]:
    rows = [
        query_metrics(ranking, qrels.get(query_id, {}))
        for query_id, ranking in rankings.items()
    ]
    return {
        metric: mean_metric(rows, metric)
        for metric in (
            'ndcg_at_10',
            'map_at_100',
            'recall_at_100',
            'mrr_at_20',
        )
    }


def rank_to_external(
    offsets: np.ndarray,
    scores: np.ndarray,
    document_ids: list[str],
) -> list[tuple[str, float]]:
    return [
        (document_ids[int(offset)], float(scores[int(offset)]))
        for offset in offsets
    ]


def work_ratios(
    queries: sparse.csr_matrix,
    document_frequencies: np.ndarray,
    selected_terms: np.ndarray,
    retained_counts: np.ndarray,
) -> list[float]:
    ratios: list[float] = []

    for query in range(queries.shape[0]):
        start = int(queries.indptr[query])
        end = int(queries.indptr[query + 1])
        term_ids = queries.indices[start:end]
        exact = int(document_frequencies[term_ids].sum())
        bounded = int(
            np.where(
                selected_terms[term_ids],
                retained_counts[term_ids],
                document_frequencies[term_ids],
            ).sum()
        )
        ratios.append(1.0 if exact == 0 else bounded / exact)
    return ratios


def evaluate_variant(
    approximate: np.ndarray,
    exact: np.ndarray,
    exact_topk: list[np.ndarray],
    document_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, float]],
    k: int,
    candidate_multiplier: int,
) -> dict[str, Any]:
    approximate_rankings: dict[str, list[tuple[str, float]]] = {}
    completed_rankings: dict[str, list[tuple[str, float]]] = {}
    overlaps: list[float] = []
    candidate_coverages: list[float] = []
    candidate_count = min(k * candidate_multiplier, exact.shape[1])

    for query, query_id in enumerate(query_ids):
        approximate_top = topk_offsets(approximate[query], k)
        candidates = topk_offsets(approximate[query], candidate_count)
        candidate_exact = exact[query, candidates]
        local = topk_offsets(candidate_exact, k)
        completed_top = candidates[local]
        exact_set = set(int(value) for value in exact_topk[query])

        overlaps.append(len(set(approximate_top) & exact_set) / len(exact_set))
        candidate_coverages.append(
            len(set(candidates) & exact_set) / len(exact_set),
        )
        approximate_rankings[query_id] = rank_to_external(
            approximate_top,
            approximate[query],
            document_ids,
        )
        completed_rankings[query_id] = rank_to_external(
            completed_top,
            exact[query],
            document_ids,
        )
    return {
        'candidate_count': candidate_count,
        'mean_overlap_at_k': statistics.fmean(overlaps),
        'minimum_overlap_at_k': min(overlaps),
        'mean_exact_topk_candidate_coverage': statistics.fmean(
            candidate_coverages,
        ),
        'minimum_exact_topk_candidate_coverage': min(candidate_coverages),
        'approximate_metrics': metric_summary(approximate_rankings, qrels),
        'exact_completion_metrics': metric_summary(completed_rankings, qrels),
    }


def main() -> int:
    args = parse_args()
    retained_counts = tuple(args.retained_counts or (128, 256, 512, 1000))
    candidate_multipliers = tuple(args.candidate_multipliers or (8, 16, 32, 72))
    if args.k <= 0 or any(value <= 0 for value in retained_counts):
        raise SystemExit('k and retained counts must be positive')
    if any(value <= 0 for value in candidate_multipliers):
        raise SystemExit('candidate multipliers must be positive')
    if args.max_fraction <= 0.0:
        raise SystemExit('max fraction must be positive')
    if not 0.0 < args.accelerated_posting_mass <= 1.0:
        raise SystemExit('accelerated posting mass must be in (0, 1]')

    started = time.perf_counter()
    documents, queries, document_ids, query_ids, qrels = load_dataset(
        args.dataset_root,
        args.query_limit,
    )
    document_frequencies = np.asarray(
        documents.getnnz(axis=0),
        dtype=np.int64,
    ).reshape(-1)
    selected_terms, actual_mass = select_high_df_terms(
        document_frequencies,
        args.accelerated_posting_mass,
    )
    selected_queries = mask_query_terms(queries, selected_terms)
    exact = np.asarray((queries @ documents.transpose()).toarray())
    selected_exact = np.asarray(
        (selected_queries @ documents.transpose()).toarray(),
    )
    residual = exact - selected_exact
    exact_topk = [
        topk_offsets(exact[query], args.k)
        for query in range(queries.shape[0])
    ]
    exact_rankings = {
        query_id: rank_to_external(
            exact_topk[query],
            exact[query],
            document_ids,
        )
        for query, query_id in enumerate(query_ids)
    }
    variants: dict[str, Any] = {}

    for retained_count in retained_counts:
        variant_started = time.perf_counter()
        if args.pruning_strategy == 'global-threshold':
            (
                retained,
                tail_maxima,
                retained_per_term,
                threshold,
            ) = build_global_threshold_matrix(
                documents,
                selected_terms,
                retained_count,
                args.max_fraction,
            )
        else:
            (
                retained,
                tail_maxima,
                retained_per_term,
                threshold,
            ) = build_retained_matrix(
                documents,
                selected_terms,
                retained_count,
            )
        retained_scores = np.asarray(
            (selected_queries @ retained.transpose()).toarray(),
        )
        approximate = residual + retained_scores
        bounds = np.asarray(selected_queries @ tail_maxima).reshape(-1)
        work = work_ratios(
            queries,
            document_frequencies,
            selected_terms,
            retained_per_term,
        )
        variants[str(retained_count)] = {
            'pruning_strategy': args.pruning_strategy,
            'max_fraction': (
                args.max_fraction
                if args.pruning_strategy == 'global-threshold'
                else None
            ),
            'global_threshold': threshold,
            'retained_postings': int(retained.nnz),
            'maximum_retained_per_term': int(retained_per_term.max(initial=0)),
            'work_ratio': {
                'mean': statistics.fmean(work),
                'p50': statistics.median(work),
                'p95': percentile(work, 0.95),
                'maximum': max(work),
            },
            'omitted_score_bound': {
                'mean': float(np.mean(bounds)),
                'p95': float(np.percentile(bounds, 95)),
                'maximum': float(np.max(bounds)),
            },
            'candidate_multipliers': {
                str(multiplier): evaluate_variant(
                    approximate,
                    exact,
                    exact_topk,
                    document_ids,
                    query_ids,
                    qrels,
                    args.k,
                    multiplier,
                )
                for multiplier in candidate_multipliers
            },
            'elapsed_seconds': time.perf_counter() - variant_started,
        }
    output = {
        'schema': 'evoke_retained_impact_oracle_v2',
        'dataset': args.dataset_root.name,
        'documents': documents.shape[0],
        'queries': queries.shape[0],
        'vocabulary': documents.shape[1],
        'document_postings': int(documents.nnz),
        'accelerated_posting_mass_requested': args.accelerated_posting_mass,
        'accelerated_posting_mass_actual': actual_mass,
        'accelerated_term_count': int(selected_terms.sum()),
        'pruning_strategy': args.pruning_strategy,
        'max_fraction': (
            args.max_fraction
            if args.pruning_strategy == 'global-threshold'
            else None
        ),
        'exact_metrics': metric_summary(exact_rankings, qrels),
        'variants': variants,
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
