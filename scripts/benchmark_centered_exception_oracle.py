#!/usr/bin/env python3
"""Evaluate rank-invariant baselines for high-DF semantic postings."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from benchmark_retained_impact_oracle import (
    evaluate_variant,
    load_dataset,
    metric_summary,
    rank_to_external,
    select_high_df_terms,
    topk_offsets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--accelerated-posting-mass', type=float, default=0.3)
    parser.add_argument(
        '--tolerance-ratio',
        action='append',
        type=float,
        dest='tolerance_ratios',
    )
    parser.add_argument(
        '--candidate-multiplier',
        action='append',
        type=int,
        dest='candidate_multipliers',
    )
    return parser.parse_args()


def best_baseline(
    values: np.ndarray,
    document_count: int,
    tolerance: float,
) -> tuple[float, int]:
    """Return the fixed baseline covering the most values within tolerance."""
    if document_count <= 0 or tolerance < 0.0:
        raise ValueError('invalid document count or tolerance')
    ordered = np.sort(np.asarray(values, dtype=np.float32))
    missing = document_count - ordered.size
    zero_end = int(np.searchsorted(ordered, 2.0 * tolerance, side='right'))
    best_baseline_value = tolerance
    best_coverage = missing + zero_end
    left = 0

    for right in range(ordered.size):
        while ordered[right] - ordered[left] > 2.0 * tolerance:
            left += 1
        coverage = right - left + 1
        baseline = float((ordered[left] + ordered[right]) * 0.5)
        if coverage > best_coverage or (
            coverage == best_coverage and baseline < best_baseline_value
        ):
            best_coverage = coverage
            best_baseline_value = baseline
    return best_baseline_value, best_coverage


def build_centered_exceptions(
    documents: sparse.csr_matrix,
    selected_terms: np.ndarray,
    tolerance_ratio: float,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    columns = documents.tocsc()
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    baselines = np.zeros(documents.shape[1], dtype=np.float32)
    tolerances = np.zeros(documents.shape[1], dtype=np.float32)
    exception_counts = np.zeros(documents.shape[1], dtype=np.int64)
    all_rows = np.arange(documents.shape[0], dtype=np.int64)

    for term_id in np.flatnonzero(selected_terms):
        start = int(columns.indptr[term_id])
        end = int(columns.indptr[term_id + 1])
        values = columns.data[start:end]
        rows = columns.indices[start:end].astype(np.int64, copy=False)
        maximum = float(np.max(np.abs(values), initial=0.0))
        tolerance = tolerance_ratio * maximum
        baseline, _coverage = best_baseline(
            values,
            documents.shape[0],
            tolerance,
        )
        lower = baseline - tolerance
        upper = baseline + tolerance
        outside = (values < lower) | (values > upper)
        exception_rows = rows[outside]
        exception_values = values[outside] - baseline
        if lower > 0.0:
            present = np.zeros(documents.shape[0], dtype=np.bool_)
            present[rows] = True
            missing_rows = all_rows[~present]
            exception_rows = np.concatenate((exception_rows, missing_rows))
            exception_values = np.concatenate(
                (
                    exception_values,
                    np.full(missing_rows.size, -baseline, dtype=np.float32),
                )
            )
        baselines[term_id] = baseline
        tolerances[term_id] = tolerance
        exception_counts[term_id] = exception_rows.size
        row_parts.append(exception_rows)
        column_parts.append(
            np.full(exception_rows.size, term_id, dtype=np.int32)
        )
        value_parts.append(exception_values)
    matrix = sparse.csr_matrix(
        (
            np.concatenate(value_parts) if value_parts else np.empty(0),
            (
                np.concatenate(row_parts) if row_parts else np.empty(0),
                np.concatenate(column_parts) if column_parts else np.empty(0),
            ),
        ),
        shape=documents.shape,
        dtype=np.float32,
    )
    matrix.sort_indices()
    return matrix, baselines, tolerances, exception_counts


def query_work_ratios(
    queries: sparse.csr_matrix,
    document_frequencies: np.ndarray,
    selected_terms: np.ndarray,
    exception_counts: np.ndarray,
) -> list[float]:
    ratios: list[float] = []

    for query in range(queries.shape[0]):
        start = int(queries.indptr[query])
        end = int(queries.indptr[query + 1])
        terms = queries.indices[start:end]
        exact = int(document_frequencies[terms].sum())
        work = int(
            np.where(
                selected_terms[terms],
                exception_counts[terms],
                document_frequencies[terms],
            ).sum()
        )
        ratios.append(1.0 if exact == 0 else work / exact)
    return ratios


def main() -> int:
    args = parse_args()
    tolerance_ratios = tuple(args.tolerance_ratios or (0.02, 0.05))
    candidate_multipliers = tuple(args.candidate_multipliers or (8,))
    if args.k <= 0 or any(value <= 0 for value in candidate_multipliers):
        raise SystemExit('k and candidate multipliers must be positive')
    if any(value <= 0.0 for value in tolerance_ratios):
        raise SystemExit('tolerance ratios must be positive')

    started = time.perf_counter()
    documents, queries, document_ids, query_ids, qrels = load_dataset(
        args.dataset_root,
        args.query_limit,
    )
    frequencies = np.asarray(
        documents.getnnz(axis=0),
        dtype=np.int64,
    ).reshape(-1)
    selected_terms, actual_mass = select_high_df_terms(
        frequencies,
        args.accelerated_posting_mass,
    )
    selected_queries = queries.multiply(selected_terms).tocsr()
    exact = np.asarray((queries @ documents.transpose()).toarray())
    selected_exact = np.asarray(
        (selected_queries @ documents.transpose()).toarray()
    )
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
    selected_postings = int(frequencies[selected_terms].sum())
    variants: dict[str, Any] = {}

    for ratio in tolerance_ratios:
        variant_started = time.perf_counter()
        exceptions, baselines, tolerances, counts = (
            build_centered_exceptions(documents, selected_terms, ratio)
        )
        exception_scores = np.asarray(
            (selected_queries @ exceptions.transpose()).toarray()
        )
        approximate = exact - selected_exact + exception_scores
        work = query_work_ratios(
            queries,
            frequencies,
            selected_terms,
            counts,
        )
        query_bounds = np.asarray(
            np.abs(selected_queries) @ tolerances
        ).reshape(-1)
        variants[format(ratio, '.6g')] = {
            'exception_postings': int(exceptions.nnz),
            'selected_postings': selected_postings,
            'exception_to_selected_ratio': (
                exceptions.nnz / selected_postings
                if selected_postings
                else 0.0
            ),
            'nonzero_baseline_terms': int(np.count_nonzero(baselines)),
            'work_ratio': {
                'mean': statistics.fmean(work),
                'p50': statistics.median(work),
                'p95': float(np.percentile(work, 95)),
                'maximum': max(work),
            },
            'score_error_bound': {
                'mean': float(np.mean(query_bounds)),
                'p95': float(np.percentile(query_bounds, 95)),
                'maximum': float(np.max(query_bounds)),
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
        'schema': 'evoke_centered_exception_oracle_v1',
        'dataset': args.dataset_root.name,
        'documents': documents.shape[0],
        'queries': queries.shape[0],
        'document_postings': int(documents.nnz),
        'accelerated_posting_mass_requested': args.accelerated_posting_mass,
        'accelerated_posting_mass_actual': actual_mass,
        'accelerated_term_count': int(selected_terms.sum()),
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
