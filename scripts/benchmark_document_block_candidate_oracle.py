#!/usr/bin/env python3
"""Evaluate cross-term document-block candidates for sparse retrieval."""

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--block-shift', type=int, default=4)
    parser.add_argument(
        '--candidate-multiplier',
        action='append',
        type=int,
        dest='candidate_multipliers',
    )
    parser.add_argument('--include-query-rows', action='store_true')
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


def load_csr(path: Path) -> sparse.csr_matrix:
    with np.load(path, allow_pickle=False) as payload:
        matrix = sparse.csr_matrix(
            (
                payload['data'],
                payload['indices'],
                payload['indptr'],
            ),
            shape=tuple(int(value) for value in payload['shape']),
        )
    matrix.sort_indices()
    return matrix


def load_ids(path: Path, limit: int = 0) -> list[str]:
    with path.open(encoding='utf-8') as source:
        result = [line.rstrip('\n') for line in source if line.strip()]
    return result[:limit] if limit > 0 else result


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text())
    return {
        str(query_id): {
            str(document_id): float(relevance)
            for document_id, relevance in documents.items()
        }
        for query_id, documents in raw.items()
    }


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
        document_ids.extend(load_ids(root / shard['ids']))
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


def query_metrics(
    ranking: list[tuple[str, float]],
    qrels: dict[str, float],
) -> dict[str, float]:
    relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0.0}
    ranked_ids = [doc_id for doc_id, _score in ranking]
    recall_hits = sum(doc_id in relevant for doc_id in ranked_ids[:100])
    precision_sum = 0.0
    average_precision_hits = 0
    reciprocal_rank = 0.0
    dcg = 0.0

    for rank, doc_id in enumerate(ranked_ids[:100], start=1):
        grade = qrels.get(doc_id, 0.0)
        if grade > 0.0:
            average_precision_hits += 1
            precision_sum += average_precision_hits / rank
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


def metric_summary(
    rankings: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, float]],
) -> dict[str, float]:
    rows = [
        query_metrics(ranking, qrels.get(query_id, {}))
        for query_id, ranking in rankings.items()
    ]
    return {
        metric: statistics.fmean(row[metric] for row in rows)
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


def accumulate_block_maxima(
    columns: sparse.csc_matrix,
    query_ids: np.ndarray,
    query_weights: np.ndarray,
    block_count: int,
    block_shift: int,
) -> tuple[np.ndarray, int, int]:
    """Accumulate a safe nonnegative upper score for each document block."""
    scores = np.zeros(block_count, dtype=np.float32)
    posting_count = 0
    ref_count = 0

    for term_id, query_weight in zip(query_ids, query_weights, strict=True):
        start = int(columns.indptr[term_id])
        end = int(columns.indptr[term_id + 1])
        rows = columns.indices[start:end]
        values = columns.data[start:end]

        posting_count += rows.size
        if rows.size == 0 or query_weight <= 0.0:
            continue
        blocks = np.right_shift(rows, block_shift)
        starts = np.flatnonzero(
            np.concatenate(
                (
                    np.asarray([True]),
                    blocks[1:] != blocks[:-1],
                )
            )
        )
        maxima = np.maximum.reduceat(values, starts) * query_weight
        block_ids = blocks[starts]
        scores[block_ids] += maxima
        ref_count += block_ids.size
    return scores, posting_count, ref_count


def block_candidates(
    block_scores: np.ndarray,
    document_count: int,
    target_documents: int,
    block_shift: int,
) -> tuple[np.ndarray, int]:
    block_size = 1 << block_shift
    target_blocks = min(
        block_scores.size,
        math.ceil(target_documents / block_size),
    )
    positive = np.flatnonzero(block_scores > 0.0)

    if target_blocks == 0 or positive.size == 0:
        return np.empty(0, dtype=np.int64), 0
    if positive.size > target_blocks:
        selected_local = topk_offsets(
            block_scores[positive],
            target_blocks,
        )
        selected_blocks = positive[selected_local]
    else:
        selected_blocks = positive
    documents = np.concatenate(
        [
            np.arange(
                block_id << block_shift,
                min((block_id + 1) << block_shift, document_count),
                dtype=np.int64,
            )
            for block_id in selected_blocks
        ]
    )
    return documents, selected_blocks.size


def summarize(values: list[float]) -> dict[str, float]:
    return {
        'mean': statistics.fmean(values) if values else 0.0,
        'p50': statistics.median(values) if values else 0.0,
        'p95': percentile(values, 0.95),
        'maximum': max(values, default=0.0),
    }


def main() -> int:
    args = parse_args()
    multipliers = tuple(args.candidate_multipliers or (8, 16, 32, 64))
    if args.k <= 0 or args.block_shift <= 0:
        raise SystemExit('k and block shift must be positive')
    if any(multiplier <= 0 for multiplier in multipliers):
        raise SystemExit('candidate multipliers must be positive')

    started = time.perf_counter()
    (
        documents,
        queries,
        document_ids,
        query_ids,
        qrels,
    ) = load_dataset(args.dataset_root, args.query_limit)
    if documents.data.size and float(documents.data.min()) < 0.0:
        raise SystemExit('document-block oracle requires nonnegative impacts')
    if queries.data.size and float(queries.data.min()) < 0.0:
        raise SystemExit('document-block oracle requires nonnegative queries')

    columns = documents.tocsc()
    block_count = math.ceil(documents.shape[0] / (1 << args.block_shift))
    exact_rankings: dict[str, list[tuple[str, float]]] = {}
    completed_rankings = {multiplier: {} for multiplier in multipliers}
    overlaps = {multiplier: [] for multiplier in multipliers}
    coverages = {multiplier: [] for multiplier in multipliers}
    candidate_documents = {multiplier: [] for multiplier in multipliers}
    selected_blocks = {multiplier: [] for multiplier in multipliers}
    completion_latencies = {multiplier: [] for multiplier in multipliers}
    block_latencies: list[float] = []
    ref_ratios: list[float] = []
    work_ratios = {multiplier: [] for multiplier in multipliers}
    query_rows: list[dict[str, Any]] = []

    for query_offset, query_id in enumerate(query_ids):
        query = queries.getrow(query_offset)
        exact_started = time.perf_counter()
        exact_scores = np.asarray(
            (documents @ query.transpose()).toarray(),
        ).reshape(-1)
        exact_top = topk_offsets(exact_scores, args.k)
        exact_set = set(int(document) for document in exact_top)
        exact_rankings[query_id] = rank_to_external(
            exact_top,
            exact_scores,
            document_ids,
        )

        block_started = time.perf_counter()
        block_scores, postings, refs = accumulate_block_maxima(
            columns,
            query.indices,
            query.data,
            block_count,
            args.block_shift,
        )
        block_latencies.append((time.perf_counter() - block_started) * 1000.0)
        ref_ratios.append(refs / postings if postings else 0.0)
        row: dict[str, Any] = {
            'query_id': query_id,
            'query_terms': int(query.nnz),
            'query_postings': postings,
            'block_refs': refs,
            'exact_latency_ms': (
                time.perf_counter() - exact_started
            ) * 1000.0,
            'variants': {},
        }

        for multiplier in multipliers:
            completion_started = time.perf_counter()
            candidates, block_total = block_candidates(
                block_scores,
                documents.shape[0],
                args.k * multiplier,
                args.block_shift,
            )
            candidate_scores = exact_scores[candidates]
            local_top = topk_offsets(candidate_scores, args.k)
            completed_top = candidates[local_top]
            candidate_set = set(int(document) for document in candidates)
            overlap = len(set(int(value) for value in completed_top) & exact_set)
            coverage = len(candidate_set & exact_set)
            denominator = len(exact_set) or 1
            completion_ms = (
                time.perf_counter() - completion_started
            ) * 1000.0
            exact_score_work = candidates.size * query.nnz

            overlaps[multiplier].append(overlap / denominator)
            coverages[multiplier].append(coverage / denominator)
            candidate_documents[multiplier].append(int(candidates.size))
            selected_blocks[multiplier].append(block_total)
            completion_latencies[multiplier].append(completion_ms)
            work_ratios[multiplier].append(
                (refs + exact_score_work) / postings if postings else 0.0
            )
            completed_rankings[multiplier][query_id] = rank_to_external(
                completed_top,
                exact_scores,
                document_ids,
            )
            row['variants'][str(multiplier)] = {
                'candidate_documents': int(candidates.size),
                'selected_blocks': block_total,
                'overlap_at_k': overlap / denominator,
                'candidate_coverage_at_k': coverage / denominator,
                'completion_latency_ms': completion_ms,
            }
        if args.include_query_rows:
            query_rows.append(row)

    variants: dict[str, Any] = {}
    for multiplier in multipliers:
        variants[str(multiplier)] = {
            'candidate_multiplier': multiplier,
            'candidate_documents': summarize(candidate_documents[multiplier]),
            'selected_blocks': summarize(selected_blocks[multiplier]),
            'mean_overlap_at_k': statistics.fmean(overlaps[multiplier]),
            'minimum_overlap_at_k': min(overlaps[multiplier]),
            'mean_exact_topk_candidate_coverage': statistics.fmean(
                coverages[multiplier]
            ),
            'minimum_exact_topk_candidate_coverage': min(
                coverages[multiplier]
            ),
            'work_ratio': summarize(work_ratios[multiplier]),
            'completion_latency_ms': summarize(
                completion_latencies[multiplier]
            ),
            'metrics': metric_summary(
                completed_rankings[multiplier],
                qrels,
            ),
        }

    output = {
        'schema': 'evoke_document_block_candidate_oracle_v1',
        'dataset': args.dataset_root.name,
        'documents': documents.shape[0],
        'queries': queries.shape[0],
        'vocabulary': documents.shape[1],
        'document_postings': int(documents.nnz),
        'block_shift': args.block_shift,
        'block_size': 1 << args.block_shift,
        'block_count': block_count,
        'block_ref_ratio': summarize(ref_ratios),
        'block_accumulation_latency_ms': summarize(block_latencies),
        'exact_metrics': metric_summary(exact_rankings, qrels),
        'variants': variants,
        'query_rows': query_rows if args.include_query_rows else None,
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
