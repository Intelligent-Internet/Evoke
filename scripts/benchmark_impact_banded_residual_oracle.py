#!/usr/bin/env python3
"""Audit exact and bounded residual posting frontiers.

The oracle models the product accelerator in two stages. High-DF selected
terms seed an exact top-k threshold. Unselected terms then admit residual
documents. Every admitted document receives its exact sparse score, so the
experiment isolates posting decode and candidate-union work from scoring
fidelity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import numpy as np
from scipy import sparse

from benchmark_document_block_candidate_oracle import (
    load_dataset,
    query_metrics,
)
from benchmark_retained_impact_oracle import select_high_df_terms


DEFAULT_BLOCK_SHIFTS = (6,)
DEFAULT_ERROR_RATIOS = (0.0,)
DEFAULT_LAZY_BLOCK_BATCHES = (1024,)
DIRECTORY_WIDTHS = (8, 12, 16)
IMPACT_WIDTHS = (4, 8)
EVOKE_TERMS_PER_FRONTIER_SHARD = 64
EVOKE_FRONTIER_HEADER_SIZE = 64
EVOKE_PACKED_HEADER_SIZE = 136
EVOKE_PACKED_TERM_SIZE = 52
EVOKE_PACKED_REF_SIZE = 5
EVOKE_PACKED_SUPER_REF_SIZE = 16
EVOKE_PACKED_BLOCK_SHIFT = 4
EVOKE_PACKED_SUPERBLOCK_SHIFT = 8


@dataclass(frozen=True)
class PostingList:
    documents: np.ndarray
    impacts: np.ndarray


@dataclass(frozen=True)
class TermBlock:
    term_id: int
    query_weight: float
    documents: np.ndarray
    impacts: np.ndarray

    @property
    def upper_bound(self) -> float:
        if self.impacts.size == 0:
            return 0.0
        return self.query_weight * float(self.impacts[0])


@dataclass
class VariantRow:
    decoded_postings: int
    admitted_documents: int
    metadata_entries: int
    top_documents: np.ndarray


class ExactTopK:
    """Incremental deterministic top-k over already computed exact scores."""

    def __init__(self, scores: np.ndarray, k: int) -> None:
        self.scores = scores
        self.k = k
        self.seen = np.zeros(scores.size, dtype=np.bool_)
        self.heap: list[tuple[float, int]] = []

    def offer(self, document: int) -> bool:
        if self.seen[document]:
            return False
        self.seen[document] = True
        item = (float(self.scores[document]), -document)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)
        return True

    def offer_many(self, documents: Iterable[int]) -> int:
        admitted = 0
        for document in documents:
            admitted += self.offer(int(document))
        return admitted

    @property
    def threshold(self) -> float:
        if len(self.heap) < self.k:
            return -math.inf
        return self.heap[0][0]

    def can_stop(self, upper_bound: float, error_ratio: float) -> bool:
        threshold = self.threshold
        if not math.isfinite(threshold) or threshold <= 0.0:
            return False
        allowed = threshold * (1.0 + error_ratio)
        slack = 1e-6 * (1.0 + abs(upper_bound) + abs(allowed))
        return upper_bound + slack < allowed

    def finish(self) -> np.ndarray:
        if len(self.heap) < self.k:
            for document in range(self.scores.size):
                self.offer(document)
                if len(self.heap) == self.k:
                    break
        documents = np.asarray(
            [-negative_document for _score, negative_document in self.heap],
            dtype=np.int64,
        )
        order = np.lexsort((documents, -self.scores[documents]))
        return documents[order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-limit', type=int, default=32)
    parser.add_argument('--document-limit', type=int, default=0)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--accelerated-posting-mass', type=float, default=0.30)
    parser.add_argument('--seed-document-fraction', type=float, default=1.0)
    parser.add_argument(
        '--seed-per-selected-term',
        type=int,
        default=0,
        help=(
            'When positive, model the product retained surface by admitting '
            'at most this many highest-impact documents per selected term. '
            'This overrides --seed-document-fraction.'
        ),
    )
    parser.add_argument(
        '--frontier-scope',
        choices=('residual', 'all-unseeded'),
        default='residual',
        help=(
            'Residual models the current unselected-term tail. All-unseeded '
            'also traverses selected-term postings not admitted by a bounded '
            'first-stage seed and therefore supports exact bounded-seed '
            'experiments.'
        ),
    )
    parser.add_argument(
        '--block-shift',
        action='append',
        type=int,
        dest='block_shifts',
    )
    parser.add_argument(
        '--error-ratio',
        action='append',
        type=float,
        dest='error_ratios',
    )
    parser.add_argument(
        '--lazy-block-batch',
        action='append',
        type=int,
        dest='lazy_block_batches',
        help='Resolve partial block candidates every N blocks; zero means all.',
    )
    parser.add_argument(
        '--frontier-impact-bits',
        action='append',
        type=int,
        dest='frontier_impact_bits',
        help=(
            'Impact precision used only by block frontier bounds. Use 32 for '
            'f32 or 8 for conservative per-term upward bounds followed by '
            'exact score resolution.'
        ),
    )
    parser.add_argument(
        '--sole-authority-impact-bits',
        action='append',
        type=int,
        dest='sole_authority_impact_bits',
        help=(
            'Measure ranking fidelity when fp16 or per-term u8 impacts are '
            'the sole scoring authority rather than conservative guidance.'
        ),
    )
    parser.add_argument(
        '--include-fixed-u8-authority',
        action='store_true',
        help=(
            'Also measure the lifecycle-stable positive fixed-range u8 '
            'authority used by the product precision slice.'
        ),
    )
    parser.add_argument(
        '--include-document-order',
        action='store_true',
        help=(
            'Also traverse document blocks in natural order. This models a '
            'bounded streaming executor without a query-global block sort.'
        ),
    )
    parser.add_argument(
        '--hierarchical-super-shift',
        action='append',
        type=int,
        dest='hierarchical_super_shifts',
        help=(
            'Add a coarse document-range directory above each requested '
            'fine block size and descend only into competitive ranges.'
        ),
    )
    parser.add_argument('--include-query-rows', action='store_true')
    return parser.parse_args()


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = math.ceil(ratio * len(ordered)) - 1
    return ordered[max(0, min(position, len(ordered) - 1))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        'mean': statistics.fmean(values) if values else 0.0,
        'p50': statistics.median(values) if values else 0.0,
        'p95': percentile(values, 0.95),
        'minimum': min(values, default=0.0),
        'maximum': max(values, default=0.0),
    }


def varint_size(value: int) -> int:
    size = 1
    while value >= 0x80:
        value >>= 7
        size += 1
    return size


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    count = min(k, scores.size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    documents = np.arange(scores.size, dtype=np.int64)
    order = np.lexsort((documents, -scores))
    return order[:count].astype(np.int64, copy=False)


def overlap(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(1, min(left.size, right.size))
    return len(set(map(int, left)) & set(map(int, right))) / denominator


def metric_row(
    documents: np.ndarray,
    scores: np.ndarray,
    document_ids: list[str],
    qrels: dict[str, float],
) -> dict[str, float]:
    ranking = [
        (document_ids[int(document)], float(scores[int(document)]))
        for document in documents
    ]
    return query_metrics(ranking, qrels)


def impact_order(postings: PostingList) -> PostingList:
    if postings.documents.size <= 1:
        return postings
    order = np.lexsort((postings.documents, -postings.impacts))
    return PostingList(
        documents=postings.documents[order],
        impacts=postings.impacts[order],
    )


def quantize_impacts_up(postings: PostingList, bits: int) -> PostingList:
    """Return conservative per-term impact bounds at the requested width."""

    if bits == 32 or postings.impacts.size == 0:
        return postings
    if bits != 8:
        raise ValueError('frontier impact bits must be 8 or 32')
    levels = (1 << bits) - 1
    maximum = float(postings.impacts.max())
    scale = np.nextafter(maximum / levels, math.inf)
    quantized = np.ceil(postings.impacts / scale)
    quantized = np.clip(quantized, 1, levels)
    bounds = np.nextafter(quantized * scale, math.inf)
    bounds = np.maximum(bounds, postings.impacts)
    return PostingList(documents=postings.documents, impacts=bounds)


def quantize_impacts_nearest(
    impacts: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Quantize impacts for a sole scoring authority while retaining support."""

    if impacts.size == 0:
        return impacts.astype(np.float64, copy=False)
    if bits == 16:
        quantized = impacts.astype(np.float16).astype(np.float64)
        if not np.all(np.isfinite(quantized)):
            raise ValueError('impacts exceed the fp16 authority range')
        smallest = float(np.nextafter(np.float16(0), np.float16(1)))
        return np.maximum(quantized, smallest)
    if bits != 8:
        raise ValueError('sole authority impact bits must be 8 or 16')
    levels = (1 << bits) - 1
    maximum = float(impacts.max())
    scale = maximum / levels
    codes = np.rint(impacts / scale)
    codes = np.clip(codes, 1, levels)
    return codes.astype(np.float64, copy=False) * scale


def quantize_impacts_fixed_u8(impacts: np.ndarray) -> np.ndarray:
    """Quantize positive P2 impacts with a fixed 2.0 global range."""

    values = impacts.astype(np.float64, copy=False)
    if values.size == 0:
        return values
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError('fixed u8 requires finite positive impacts')
    codes = np.rint(values * 255.0 / 2.0)
    codes = np.clip(codes, 1, 255)
    return codes.astype(np.float64, copy=False) * (2.0 / 255.0)


def sole_authority_scores(
    columns: sparse.csc_matrix,
    query_ids: np.ndarray,
    query_weights: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Score one query using only its quantized document-impact authority."""

    scores = np.zeros(columns.shape[0], dtype=np.float64)
    for term_id, query_weight in zip(query_ids, query_weights, strict=True):
        postings = load_posting(columns, int(term_id))
        quantized = quantize_impacts_nearest(postings.impacts, bits)
        scores[postings.documents] += float(query_weight) * quantized
    return scores


def sole_fixed_u8_authority_scores(
    columns: sparse.csc_matrix,
    query_ids: np.ndarray,
    query_weights: np.ndarray,
) -> np.ndarray:
    """Score one query using only fixed-range u8 document impacts."""

    scores = np.zeros(columns.shape[0], dtype=np.float64)
    for term_id, query_weight in zip(query_ids, query_weights, strict=True):
        postings = load_posting(columns, int(term_id))
        quantized = quantize_impacts_fixed_u8(postings.impacts)
        scores[postings.documents] += float(query_weight) * quantized
    return scores


def load_posting(columns: sparse.csc_matrix, term_id: int) -> PostingList:
    start = int(columns.indptr[term_id])
    end = int(columns.indptr[term_id + 1])
    return PostingList(
        documents=columns.indices[start:end].astype(np.int64, copy=False),
        impacts=columns.data[start:end].astype(np.float64, copy=False),
    )


def selected_seed_documents(
    columns: sparse.csc_matrix,
    documents: sparse.csr_matrix,
    query_ids: np.ndarray,
    query_weights: np.ndarray,
    selected_terms: np.ndarray,
    fraction: float,
    per_term_cap: int,
    k: int,
) -> np.ndarray:
    parts = []
    for term_id in query_ids:
        if not selected_terms[int(term_id)]:
            continue
        postings = load_posting(columns, int(term_id))
        if per_term_cap > 0:
            postings = impact_order(postings)
            parts.append(postings.documents[:per_term_cap])
        else:
            parts.append(postings.documents)
    if not parts:
        return np.empty(0, dtype=np.int64)
    union = np.unique(np.concatenate(parts))
    if per_term_cap > 0:
        return union
    if fraction >= 1.0:
        return union
    selected_mask = selected_terms[query_ids]
    selected_ids = query_ids[selected_mask]
    selected_weights = query_weights[selected_mask]
    if selected_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    selected_scores = np.asarray(
        documents[:, selected_ids] @ selected_weights,
    ).reshape(-1)
    target = min(
        union.size,
        max(k, math.ceil(fraction * union.size)),
    )
    order = np.lexsort((union, -selected_scores[union]))
    return union[order[:target]]


def residual_query_terms(
    columns: sparse.csc_matrix,
    query_ids: np.ndarray,
    query_weights: np.ndarray,
    selected_terms: np.ndarray,
    include_selected_tail: bool,
) -> list[TermBlock]:
    result: list[TermBlock] = []
    for term_id, query_weight in zip(query_ids, query_weights, strict=True):
        if (
            (selected_terms[int(term_id)] and not include_selected_tail)
            or query_weight <= 0.0
        ):
            continue
        postings = impact_order(load_posting(columns, int(term_id)))
        if postings.documents.size == 0:
            continue
        result.append(
            TermBlock(
                term_id=int(term_id),
                query_weight=float(query_weight),
                documents=postings.documents,
                impacts=postings.impacts,
            )
        )
    return result


def full_residual_union(
    state: ExactTopK,
    terms: list[TermBlock],
) -> VariantRow:
    decoded = 0
    admitted = 0
    for term in terms:
        decoded += term.documents.size
        admitted += state.offer_many(term.documents)
    return VariantRow(decoded, admitted, 0, state.finish())


def global_impact_frontier(
    state: ExactTopK,
    terms: list[TermBlock],
    error_ratio: float,
) -> VariantRow:
    positions = np.zeros(len(terms), dtype=np.int64)
    heap: list[tuple[float, int]] = []
    upper_bound = 0.0
    for offset, term in enumerate(terms):
        contribution = term.upper_bound
        upper_bound += contribution
        heapq.heappush(heap, (-contribution, offset))

    decoded = 0
    admitted = 0
    while heap and not state.can_stop(upper_bound, error_ratio):
        negative_contribution, offset = heapq.heappop(heap)
        old_contribution = -negative_contribution
        position = int(positions[offset])
        term = terms[offset]
        admitted += state.offer(int(term.documents[position]))
        decoded += 1
        position += 1
        positions[offset] = position
        if position < term.documents.size:
            new_contribution = (
                term.query_weight * float(term.impacts[position])
            )
            heapq.heappush(heap, (-new_contribution, offset))
        else:
            new_contribution = 0.0
        upper_bound += new_contribution - old_contribution
        if upper_bound < 0.0 and upper_bound > -1e-9:
            upper_bound = 0.0
    return VariantRow(decoded, admitted, len(terms), state.finish())


def resolve_partial_candidates(
    state: ExactTopK,
    partial_scores: dict[int, float],
    remaining_upper_bound: float,
    error_ratio: float,
) -> int:
    candidates = [
        (-(partial_score + remaining_upper_bound), document)
        for document, partial_score in partial_scores.items()
        if not state.seen[document]
    ]
    heapq.heapify(candidates)
    resolved = 0
    while candidates:
        negative_bound, document = heapq.heappop(candidates)
        optimistic = -negative_bound
        if state.can_stop(optimistic, error_ratio):
            break
        resolved += state.offer(document)
    return resolved


def resolve_bounded_candidates(
    state: ExactTopK,
    optimistic_scores: dict[int, float],
    error_ratio: float,
) -> int:
    candidates = [
        (-optimistic, document)
        for document, optimistic in optimistic_scores.items()
        if not state.seen[document]
    ]
    heapq.heapify(candidates)
    resolved = 0
    while candidates:
        negative_bound, document = heapq.heappop(candidates)
        if state.can_stop(-negative_bound, error_ratio):
            break
        resolved += state.offer(document)
    return resolved


def global_impact_lazy_frontier(
    state: ExactTopK,
    terms: list[TermBlock],
    error_ratio: float,
) -> VariantRow:
    positions = np.zeros(len(terms), dtype=np.int64)
    heap: list[tuple[float, int]] = []
    partial_scores: dict[int, float] = {}
    upper_bound = 0.0
    for offset, term in enumerate(terms):
        contribution = term.upper_bound
        upper_bound += contribution
        heapq.heappush(heap, (-contribution, offset))

    decoded = 0
    while heap and not state.can_stop(upper_bound, error_ratio):
        negative_contribution, offset = heapq.heappop(heap)
        old_contribution = -negative_contribution
        position = int(positions[offset])
        term = terms[offset]
        document = int(term.documents[position])
        partial_scores[document] = partial_scores.get(document, 0.0) + (
            term.query_weight * float(term.impacts[position])
        )
        decoded += 1
        position += 1
        positions[offset] = position
        if position < term.documents.size:
            new_contribution = (
                term.query_weight * float(term.impacts[position])
            )
            heapq.heappush(heap, (-new_contribution, offset))
        else:
            new_contribution = 0.0
        upper_bound += new_contribution - old_contribution
        if upper_bound < 0.0 and upper_bound > -1e-9:
            upper_bound = 0.0
    resolved = resolve_partial_candidates(
        state,
        partial_scores,
        upper_bound,
        error_ratio,
    )
    return VariantRow(decoded, resolved, len(terms), state.finish())


def split_term_blocks(term: TermBlock, block_shift: int) -> list[TermBlock]:
    block_ids = np.right_shift(term.documents, block_shift)
    result: list[TermBlock] = []
    for block_id in np.unique(block_ids):
        selected = block_ids == block_id
        postings = impact_order(
            PostingList(
                documents=term.documents[selected],
                impacts=term.impacts[selected],
            )
        )
        result.append(
            TermBlock(
                term_id=term.term_id,
                query_weight=term.query_weight,
                documents=postings.documents,
                impacts=postings.impacts,
            )
        )
    return result


def query_blocks(
    terms: list[TermBlock],
    block_shift: int,
    impact_bits: int = 32,
) -> dict[int, list[TermBlock]]:
    result: dict[int, list[TermBlock]] = {}
    for term in terms:
        bounded = quantize_impacts_up(
            PostingList(term.documents, term.impacts),
            impact_bits,
        )
        bounded_term = TermBlock(
            term_id=term.term_id,
            query_weight=term.query_weight,
            documents=bounded.documents,
            impacts=bounded.impacts,
        )
        for block in split_term_blocks(bounded_term, block_shift):
            block_id = int(block.documents[0]) >> block_shift
            result.setdefault(block_id, []).append(block)
    return result


def block_only_frontier(
    state: ExactTopK,
    blocks: dict[int, list[TermBlock]],
    error_ratio: float,
) -> VariantRow:
    ordered = sorted(
        blocks.values(),
        key=lambda terms: -sum(term.upper_bound for term in terms),
    )
    decoded = 0
    admitted = 0
    for terms in ordered:
        upper_bound = sum(term.upper_bound for term in terms)
        if state.can_stop(upper_bound, error_ratio):
            break
        for term in terms:
            decoded += term.documents.size
            admitted += state.offer_many(term.documents)
    metadata = sum(len(terms) for terms in blocks.values())
    return VariantRow(decoded, admitted, metadata, state.finish())


def block_impact_frontier(
    state: ExactTopK,
    blocks: dict[int, list[TermBlock]],
    error_ratio: float,
) -> VariantRow:
    ordered = sorted(
        blocks.values(),
        key=lambda terms: -sum(term.upper_bound for term in terms),
    )
    decoded = 0
    admitted = 0
    for terms in ordered:
        initial_bound = sum(term.upper_bound for term in terms)
        if state.can_stop(initial_bound, error_ratio):
            break
        positions = np.zeros(len(terms), dtype=np.int64)
        heap: list[tuple[float, int]] = []
        upper_bound = initial_bound
        for offset, term in enumerate(terms):
            heapq.heappush(heap, (-term.upper_bound, offset))
        while heap and not state.can_stop(upper_bound, error_ratio):
            negative_contribution, offset = heapq.heappop(heap)
            old_contribution = -negative_contribution
            position = int(positions[offset])
            term = terms[offset]
            admitted += state.offer(int(term.documents[position]))
            decoded += 1
            position += 1
            positions[offset] = position
            if position < term.documents.size:
                new_contribution = (
                    term.query_weight * float(term.impacts[position])
                )
                heapq.heappush(heap, (-new_contribution, offset))
            else:
                new_contribution = 0.0
            upper_bound += new_contribution - old_contribution
            if upper_bound < 0.0 and upper_bound > -1e-9:
                upper_bound = 0.0
    metadata = sum(len(terms) for terms in blocks.values())
    return VariantRow(decoded, admitted, metadata, state.finish())


def block_impact_lazy_frontier(
    state: ExactTopK,
    blocks: dict[int, list[TermBlock]],
    error_ratio: float,
) -> VariantRow:
    return block_impact_batched_frontier(state, blocks, error_ratio, 1)


def block_impact_batched_frontier(
    state: ExactTopK,
    blocks: dict[int, list[TermBlock]],
    error_ratio: float,
    batch_size: int,
    order_by_upper_bound: bool = True,
) -> VariantRow:
    if order_by_upper_bound:
        ordered = sorted(
            blocks.values(),
            key=lambda terms: -sum(term.upper_bound for term in terms),
        )
    else:
        ordered = [terms for _block_id, terms in sorted(blocks.items())]
    decoded = 0
    resolved = 0
    pending: dict[int, float] = {}
    pending_blocks = 0
    for terms in ordered:
        initial_bound = sum(term.upper_bound for term in terms)
        if state.can_stop(initial_bound, error_ratio):
            if order_by_upper_bound:
                break
            continue
        positions = np.zeros(len(terms), dtype=np.int64)
        heap: list[tuple[float, int]] = []
        partial_scores: dict[int, float] = {}
        upper_bound = initial_bound
        for offset, term in enumerate(terms):
            heapq.heappush(heap, (-term.upper_bound, offset))
        while heap and not state.can_stop(upper_bound, error_ratio):
            negative_contribution, offset = heapq.heappop(heap)
            old_contribution = -negative_contribution
            position = int(positions[offset])
            term = terms[offset]
            document = int(term.documents[position])
            partial_scores[document] = partial_scores.get(document, 0.0) + (
                term.query_weight * float(term.impacts[position])
            )
            decoded += 1
            position += 1
            positions[offset] = position
            if position < term.documents.size:
                new_contribution = (
                    term.query_weight * float(term.impacts[position])
                )
                heapq.heappush(heap, (-new_contribution, offset))
            else:
                new_contribution = 0.0
            upper_bound += new_contribution - old_contribution
            if upper_bound < 0.0 and upper_bound > -1e-9:
                upper_bound = 0.0
        for document, partial_score in partial_scores.items():
            pending[document] = partial_score + upper_bound
        pending_blocks += 1
        if batch_size > 0 and pending_blocks >= batch_size:
            resolved += resolve_bounded_candidates(
                state,
                pending,
                error_ratio,
            )
            pending.clear()
            pending_blocks = 0
    resolved += resolve_bounded_candidates(
        state,
        pending,
        error_ratio,
    )
    metadata = sum(len(terms) for terms in blocks.values())
    return VariantRow(decoded, resolved, metadata, state.finish())


def hierarchical_block_impact_frontier(
    state: ExactTopK,
    fine_blocks: dict[int, list[TermBlock]],
    fine_shift: int,
    super_shift: int,
    error_ratio: float,
    batch_size: int,
    order_by_upper_bound: bool = False,
) -> VariantRow:
    """Traverse a coarse directory before reading fine impact frontiers."""

    if super_shift <= fine_shift:
        raise ValueError('super shift must exceed fine block shift')
    superblocks: dict[int, dict[int, list[TermBlock]]] = {}
    super_term_bounds: dict[int, dict[int, float]] = {}
    shift = super_shift - fine_shift
    for fine_block_id, terms in fine_blocks.items():
        superblock_id = fine_block_id >> shift
        superblocks.setdefault(superblock_id, {})[fine_block_id] = terms
        bounds = super_term_bounds.setdefault(superblock_id, {})
        for term in terms:
            bounds[term.term_id] = max(
                bounds.get(term.term_id, 0.0),
                term.upper_bound,
            )
    if order_by_upper_bound:
        ordered_superblocks = sorted(
            superblocks.items(),
            key=lambda item: -sum(super_term_bounds[item[0]].values()),
        )
    else:
        ordered_superblocks = sorted(superblocks.items())

    decoded = 0
    resolved = 0
    pending: dict[int, float] = {}
    pending_blocks = 0
    descended_metadata = 0
    for superblock_id, blocks in ordered_superblocks:
        super_bound = sum(super_term_bounds[superblock_id].values())
        if state.can_stop(super_bound, error_ratio):
            if order_by_upper_bound:
                break
            continue
        ordered_fine_blocks = sorted(
            blocks.values(),
            key=lambda terms: -sum(term.upper_bound for term in terms),
        )
        descended_metadata += sum(
            len(terms) for terms in ordered_fine_blocks
        )
        for terms in ordered_fine_blocks:
            initial_bound = sum(term.upper_bound for term in terms)
            if state.can_stop(initial_bound, error_ratio):
                break
            positions = np.zeros(len(terms), dtype=np.int64)
            heap: list[tuple[float, int]] = []
            partial_scores: dict[int, float] = {}
            upper_bound = initial_bound
            for offset, term in enumerate(terms):
                heapq.heappush(heap, (-term.upper_bound, offset))
            while heap and not state.can_stop(upper_bound, error_ratio):
                negative_contribution, offset = heapq.heappop(heap)
                old_contribution = -negative_contribution
                position = int(positions[offset])
                term = terms[offset]
                document = int(term.documents[position])
                partial_scores[document] = partial_scores.get(
                    document,
                    0.0,
                ) + term.query_weight * float(term.impacts[position])
                decoded += 1
                position += 1
                positions[offset] = position
                if position < term.documents.size:
                    new_contribution = (
                        term.query_weight * float(term.impacts[position])
                    )
                    heapq.heappush(
                        heap,
                        (-new_contribution, offset),
                    )
                else:
                    new_contribution = 0.0
                upper_bound += new_contribution - old_contribution
                if upper_bound < 0.0 and upper_bound > -1e-9:
                    upper_bound = 0.0
            for document, partial_score in partial_scores.items():
                pending[document] = partial_score + upper_bound
            pending_blocks += 1
            if batch_size > 0 and pending_blocks >= batch_size:
                resolved += resolve_bounded_candidates(
                    state,
                    pending,
                    error_ratio,
                )
                pending.clear()
                pending_blocks = 0
    resolved += resolve_bounded_candidates(state, pending, error_ratio)
    super_metadata = sum(len(bounds) for bounds in super_term_bounds.values())
    return VariantRow(
        decoded,
        resolved,
        super_metadata + descended_metadata,
        state.finish(),
    )


def publication_shape(
    columns: sparse.csc_matrix,
    selected_terms: np.ndarray,
    block_shift: int,
) -> dict[str, float | int]:
    posting_count = 0
    block_entries = 0
    compact_block_header_bytes = 0
    packed_term_count = 0
    packed_ref_count = 0
    packed_super_ref_count = 0
    packed_membership_bytes = 0
    packed_doc_delta_bytes = 0
    packed_block_count = math.ceil(
        columns.shape[0] / (1 << EVOKE_PACKED_BLOCK_SHIFT)
    )
    for term_id in np.flatnonzero(~selected_terms):
        postings = load_posting(columns, int(term_id))
        posting_count += postings.documents.size
        if postings.documents.size:
            packed_term_count += 1
            packed_blocks = np.right_shift(
                postings.documents,
                EVOKE_PACKED_BLOCK_SHIFT,
            )
            packed_superblocks = np.right_shift(
                postings.documents,
                EVOKE_PACKED_SUPERBLOCK_SHIFT,
            )
            packed_refs = np.unique(packed_blocks).size
            packed_ref_count += packed_refs
            packed_super_ref_count += np.unique(packed_superblocks).size
            packed_membership_bytes += min(
                math.ceil(packed_block_count / 8),
                packed_refs,
            )
            if postings.documents.size > 1:
                maximum_delta = int(np.diff(postings.documents).max())
                delta_width = (
                    1 if maximum_delta <= 0xff
                    else 2 if maximum_delta <= 0xffff
                    else 4
                )
                packed_doc_delta_bytes += (
                    postings.documents.size - 1
                ) * delta_width
            block_ids, counts = np.unique(
                np.right_shift(postings.documents, block_shift),
                return_counts=True,
            )
            block_entries += block_ids.size
            previous = 0
            for index, (block_id, count) in enumerate(
                zip(block_ids, counts, strict=True)
            ):
                delta = (
                    int(block_id)
                    if index == 0
                    else int(block_id - previous)
                )
                compact_block_header_bytes += varint_size(delta)
                compact_block_header_bytes += varint_size(int(count))
                previous = int(block_id)
    current_packed_bytes = (
        EVOKE_PACKED_HEADER_SIZE
        + packed_term_count * EVOKE_PACKED_TERM_SIZE
        + packed_super_ref_count * EVOKE_PACKED_SUPER_REF_SIZE
        + packed_ref_count * EVOKE_PACKED_REF_SIZE
        + packed_membership_bytes
        + packed_doc_delta_bytes
        + posting_count * 4
    )
    codecs: dict[str, dict[str, float | int]] = {}
    local_document_width = max(1, math.ceil(block_shift / 8))
    for impact_width in IMPACT_WIDTHS:
        posting_width = local_document_width + impact_width
        for directory_width in DIRECTORY_WIDTHS:
            posting_bytes = posting_count * posting_width
            directory_bytes = block_entries * directory_width
            total_bytes = posting_bytes + directory_bytes
            codecs[f'i{impact_width}_d{directory_width}'] = {
                'local_document_width': local_document_width,
                'impact_width': impact_width,
                'posting_width': posting_width,
                'posting_bytes': posting_bytes,
                'directory_bytes': directory_bytes,
                'total_bytes': total_bytes,
                'total_vs_current_packed': (
                    total_bytes / current_packed_bytes
                    if current_packed_bytes else 0.0
                ),
                'directory_bytes_per_residual_posting': (
                    directory_bytes / posting_count if posting_count else 0.0
                ),
            }
    shard_count = math.ceil(columns.shape[1] /
                            EVOKE_TERMS_PER_FRONTIER_SHARD)
    compact_directory_bytes = (
        compact_block_header_bytes
        + (columns.shape[1] + shard_count) * 8
        + shard_count * EVOKE_FRONTIER_HEADER_SIZE
    )
    compact_posting_width = max(1, math.ceil(block_shift / 8)) + 4
    compact_posting_bytes = posting_count * compact_posting_width
    compact_total_bytes = compact_posting_bytes + compact_directory_bytes
    compact_f16_posting_width = local_document_width + 2
    compact_f16_posting_bytes = posting_count * compact_f16_posting_width
    compact_f16_total_bytes = (
        compact_f16_posting_bytes + compact_directory_bytes
    )
    compact_u8_scale_bytes = packed_term_count * 4
    compact_u8_posting_width = local_document_width + 1
    compact_u8_posting_bytes = posting_count * compact_u8_posting_width
    compact_u8_total_bytes = (
        compact_u8_posting_bytes
        + compact_directory_bytes
        + compact_u8_scale_bytes
    )
    return {
        'residual_postings': posting_count,
        'term_block_entries': block_entries,
        'current_packed_estimate': {
            'bytes': current_packed_bytes,
            'bytes_per_posting': (
                current_packed_bytes / posting_count
                if posting_count else 0.0
            ),
            'term_count': packed_term_count,
            'ref_count': packed_ref_count,
            'super_ref_count': packed_super_ref_count,
            'membership_bytes': packed_membership_bytes,
            'doc_delta_bytes': packed_doc_delta_bytes,
        },
        'compact_f32': {
            'posting_width': compact_posting_width,
            'posting_bytes': compact_posting_bytes,
            'block_header_bytes': compact_block_header_bytes,
            'directory_bytes': compact_directory_bytes,
            'directory_bytes_per_residual_posting': (
                compact_directory_bytes / posting_count
                if posting_count else 0.0
            ),
            'total_bytes': compact_total_bytes,
            'total_vs_current_packed': (
                compact_total_bytes / current_packed_bytes
                if current_packed_bytes else 0.0
            ),
        },
        'compact_f16': {
            'posting_width': compact_f16_posting_width,
            'posting_bytes': compact_f16_posting_bytes,
            'directory_bytes': compact_directory_bytes,
            'total_bytes': compact_f16_total_bytes,
            'total_vs_current_packed': (
                compact_f16_total_bytes / current_packed_bytes
                if current_packed_bytes else 0.0
            ),
        },
        'compact_u8': {
            'posting_width': compact_u8_posting_width,
            'posting_bytes': compact_u8_posting_bytes,
            'scale_bytes': compact_u8_scale_bytes,
            'directory_bytes': compact_directory_bytes,
            'total_bytes': compact_u8_total_bytes,
            'total_vs_current_packed': (
                compact_u8_total_bytes / current_packed_bytes
                if current_packed_bytes else 0.0
            ),
        },
        'codecs': codecs,
    }


def variant_name(prefix: str, error_ratio: float) -> str:
    return f'{prefix}_e{error_ratio:g}'


def main() -> int:
    args = parse_args()
    block_shifts = tuple(args.block_shifts or DEFAULT_BLOCK_SHIFTS)
    error_ratios = tuple(args.error_ratios or DEFAULT_ERROR_RATIOS)
    lazy_block_batches = tuple(
        args.lazy_block_batches or DEFAULT_LAZY_BLOCK_BATCHES
    )
    hierarchical_super_shifts = tuple(
        args.hierarchical_super_shifts or ()
    )
    frontier_impact_bits = tuple(args.frontier_impact_bits or (32,))
    sole_authority_impact_bits = tuple(
        args.sole_authority_impact_bits or ()
    )
    if args.k <= 0 or args.query_limit < 0 or args.document_limit < 0:
        raise SystemExit('k must be positive and limits cannot be negative')
    if not 0.0 < args.accelerated_posting_mass < 1.0:
        raise SystemExit('accelerated posting mass must be between zero and one')
    if not 0.0 < args.seed_document_fraction <= 1.0:
        raise SystemExit('seed document fraction must be between zero and one')
    if args.seed_per_selected_term < 0:
        raise SystemExit('seed per selected term cannot be negative')
    if (
        (args.seed_document_fraction < 1.0 or
         args.seed_per_selected_term > 0)
        and args.frontier_scope != 'all-unseeded'
    ):
        raise SystemExit(
            'bounded seed experiments require --frontier-scope all-unseeded'
        )
    if any(shift < 1 or shift > 20 for shift in block_shifts):
        raise SystemExit('block shifts must be between 1 and 20')
    if any(shift < 2 or shift > 24 for shift in hierarchical_super_shifts):
        raise SystemExit('hierarchical super shifts must be between 2 and 24')
    if any(ratio < 0.0 or ratio > 1.0 for ratio in error_ratios):
        raise SystemExit('error ratios must be between zero and one')
    if any(batch < 0 for batch in lazy_block_batches):
        raise SystemExit('lazy block batches cannot be negative')
    if any(bits not in (8, 32) for bits in frontier_impact_bits):
        raise SystemExit('frontier impact bits must be 8 or 32')
    if any(bits not in (8, 16) for bits in sole_authority_impact_bits):
        raise SystemExit('sole authority impact bits must be 8 or 16')

    started = time.perf_counter()
    documents, queries, document_ids, query_ids, qrels = load_dataset(
        args.dataset_root,
        args.query_limit,
    )
    if args.document_limit > 0:
        documents = documents[:args.document_limit].tocsr()
        document_ids = document_ids[:args.document_limit]
    if documents.data.size and float(documents.data.min()) < 0.0:
        raise SystemExit('oracle requires nonnegative document impacts')
    if queries.data.size and float(queries.data.min()) < 0.0:
        raise SystemExit('oracle requires nonnegative query weights')

    columns = documents.tocsc()
    document_frequencies = np.diff(columns.indptr).astype(np.int64)
    selected_terms, actual_mass = select_high_df_terms(
        document_frequencies,
        args.accelerated_posting_mass,
    )
    publication_selected_terms = (
        selected_terms
        if args.frontier_scope == 'residual'
        else np.zeros_like(selected_terms)
    )
    publication = {
        str(1 << shift): publication_shape(
            columns,
            publication_selected_terms,
            shift,
        )
        for shift in block_shifts
    }
    variant_values: dict[str, dict[str, list[float]]] = {}
    metric_values: dict[str, list[dict[str, float]]] = {}
    authority_values: dict[str, dict[str, Any]] = {
        'f32': {'overlap_at_k': [], 'metrics': []},
        **{
            f'i{bits}': {'overlap_at_k': [], 'metrics': []}
            for bits in sole_authority_impact_bits
        },
    }
    if args.include_fixed_u8_authority:
        authority_values['u8_fixed'] = {
            'overlap_at_k': [],
            'metrics': [],
        }
    query_rows: list[dict[str, Any]] = []

    def record(
        name: str,
        row: VariantRow,
        baseline_postings: int,
        baseline_documents: int,
        exact_top: np.ndarray,
        scores: np.ndarray,
        query_qrels: dict[str, float],
    ) -> dict[str, float]:
        values = variant_values.setdefault(
            name,
            {
                'decoded_fraction': [],
                'candidate_fraction': [],
                'conservative_entry_fraction': [],
                'overlap_at_k': [],
                'metadata_entries': [],
            },
        )
        decoded_fraction = (
            row.decoded_postings / baseline_postings
            if baseline_postings else 0.0
        )
        candidate_fraction = (
            row.admitted_documents / baseline_documents
            if baseline_documents else 0.0
        )
        conservative_entry_fraction = (
            (row.decoded_postings + row.metadata_entries) /
            baseline_postings
            if baseline_postings else 0.0
        )
        row_overlap = overlap(row.top_documents, exact_top)
        values['decoded_fraction'].append(decoded_fraction)
        values['candidate_fraction'].append(candidate_fraction)
        values['conservative_entry_fraction'].append(
            conservative_entry_fraction
        )
        values['overlap_at_k'].append(row_overlap)
        values['metadata_entries'].append(float(row.metadata_entries))
        metric_values.setdefault(name, []).append(
            metric_row(
                row.top_documents,
                scores,
                document_ids,
                query_qrels,
            )
        )
        return {
            'decoded_postings': row.decoded_postings,
            'decoded_fraction': decoded_fraction,
            'admitted_documents': row.admitted_documents,
            'candidate_fraction': candidate_fraction,
            'conservative_entry_fraction': conservative_entry_fraction,
            'overlap_at_k': row_overlap,
            'metadata_entries': row.metadata_entries,
        }

    for query_offset, query_id in enumerate(query_ids):
        query = queries.getrow(query_offset)
        scores = np.asarray(
            (documents @ query.transpose()).toarray(),
        ).reshape(-1)
        exact_top = topk(scores, args.k)
        query_qrels = qrels.get(query_id, {})
        authority_values['f32']['overlap_at_k'].append(1.0)
        authority_values['f32']['metrics'].append(
            metric_row(exact_top, scores, document_ids, query_qrels)
        )
        for bits in sole_authority_impact_bits:
            authority_scores = sole_authority_scores(
                columns,
                query.indices,
                query.data,
                bits,
            )
            authority_top = topk(authority_scores, args.k)
            authority = authority_values[f'i{bits}']
            authority['overlap_at_k'].append(
                overlap(authority_top, exact_top)
            )
            authority['metrics'].append(
                metric_row(
                    authority_top,
                    authority_scores,
                    document_ids,
                    query_qrels,
                )
            )
        if args.include_fixed_u8_authority:
            authority_scores = sole_fixed_u8_authority_scores(
                columns,
                query.indices,
                query.data,
            )
            authority_top = topk(authority_scores, args.k)
            authority = authority_values['u8_fixed']
            authority['overlap_at_k'].append(
                overlap(authority_top, exact_top)
            )
            authority['metrics'].append(
                metric_row(
                    authority_top,
                    authority_scores,
                    document_ids,
                    query_qrels,
                )
            )
        seed_documents = selected_seed_documents(
            columns,
            documents,
            query.indices,
            query.data,
            selected_terms,
            args.seed_document_fraction,
            args.seed_per_selected_term,
            args.k,
        )
        residual_terms = residual_query_terms(
            columns,
            query.indices,
            query.data,
            selected_terms,
            args.frontier_scope == 'all-unseeded',
        )

        baseline_state = ExactTopK(scores, args.k)
        baseline_state.offer_many(seed_documents)
        baseline = full_residual_union(baseline_state, residual_terms)
        baseline_postings = baseline.decoded_postings
        baseline_documents = baseline.admitted_documents
        if overlap(baseline.top_documents, exact_top) != 1.0:
            raise RuntimeError(
                f'baseline candidate union is not exact for query {query_id}'
            )
        row: dict[str, Any] = {
            'query_id': query_id,
            'query_terms': int(query.nnz),
            'seed_documents': int(seed_documents.size),
            'residual_terms': len(residual_terms),
            'baseline_residual_postings': baseline_postings,
            'baseline_residual_documents': baseline_documents,
            'variants': {},
        }

        for error_ratio in error_ratios:
            state = ExactTopK(scores, args.k)
            state.offer_many(seed_documents)
            result = global_impact_frontier(
                state,
                residual_terms,
                error_ratio,
            )
            name = variant_name('global_impact', error_ratio)
            row['variants'][name] = record(
                name,
                result,
                baseline_postings,
                baseline_documents,
                exact_top,
                scores,
                query_qrels,
            )

            state = ExactTopK(scores, args.k)
            state.offer_many(seed_documents)
            result = global_impact_lazy_frontier(
                state,
                residual_terms,
                error_ratio,
            )
            name = variant_name('global_impact_lazy', error_ratio)
            row['variants'][name] = record(
                name,
                result,
                baseline_postings,
                baseline_documents,
                exact_top,
                scores,
                query_qrels,
            )

        for shift in block_shifts:
            for impact_bits in frontier_impact_bits:
                blocks = query_blocks(
                    residual_terms,
                    shift,
                    impact_bits,
                )
                precision = '' if impact_bits == 32 else f'_u{impact_bits}'
                for error_ratio in error_ratios:
                    state = ExactTopK(scores, args.k)
                    state.offer_many(seed_documents)
                    block_only = block_only_frontier(
                        state,
                        blocks,
                        error_ratio,
                    )
                    name = variant_name(
                        f'block{1 << shift}{precision}',
                        error_ratio,
                    )
                    row['variants'][name] = record(
                        name,
                        block_only,
                        baseline_postings,
                        baseline_documents,
                        exact_top,
                        scores,
                        query_qrels,
                    )

                    for batch in lazy_block_batches:
                        state = ExactTopK(scores, args.k)
                        state.offer_many(seed_documents)
                        block_impact_lazy = block_impact_batched_frontier(
                            state,
                            blocks,
                            error_ratio,
                            batch,
                        )
                        batch_name = 'all' if batch == 0 else str(batch)
                        name = variant_name(
                            f'block{1 << shift}_impact_lazy_'
                            f'b{batch_name}{precision}',
                            error_ratio,
                        )
                        row['variants'][name] = record(
                            name,
                            block_impact_lazy,
                            baseline_postings,
                            baseline_documents,
                            exact_top,
                            scores,
                            query_qrels,
                        )
                        if args.include_document_order:
                            state = ExactTopK(scores, args.k)
                            state.offer_many(seed_documents)
                            document_order = block_impact_batched_frontier(
                                state,
                                blocks,
                                error_ratio,
                                batch,
                                order_by_upper_bound=False,
                            )
                            name = variant_name(
                                f'block{1 << shift}_impact_document_order_'
                                f'b{batch_name}{precision}',
                                error_ratio,
                            )
                            row['variants'][name] = record(
                                name,
                                document_order,
                                baseline_postings,
                                baseline_documents,
                                exact_top,
                                scores,
                                query_qrels,
                            )
                        for super_shift in hierarchical_super_shifts:
                            if super_shift <= shift:
                                continue
                            state = ExactTopK(scores, args.k)
                            state.offer_many(seed_documents)
                            hierarchical = hierarchical_block_impact_frontier(
                                state,
                                blocks,
                                shift,
                                super_shift,
                                error_ratio,
                                batch,
                            )
                            name = variant_name(
                                f'block{1 << shift}_super'
                                f'{1 << super_shift}_impact_b{batch_name}'
                                f'{precision}',
                                error_ratio,
                            )
                            row['variants'][name] = record(
                                name,
                                hierarchical,
                                baseline_postings,
                                baseline_documents,
                                exact_top,
                                scores,
                                query_qrels,
                            )

                    state = ExactTopK(scores, args.k)
                    state.offer_many(seed_documents)
                    block_impact = block_impact_frontier(
                        state,
                        blocks,
                        error_ratio,
                    )
                    name = variant_name(
                        f'block{1 << shift}_impact{precision}',
                        error_ratio,
                    )
                    row['variants'][name] = record(
                        name,
                        block_impact,
                        baseline_postings,
                        baseline_documents,
                        exact_top,
                        scores,
                        query_qrels,
                    )
        if args.include_query_rows:
            query_rows.append(row)
        print(
            f'{query_offset + 1}/{len(query_ids)} {query_id}: '
            f'seed={seed_documents.size:,} '
            f'residual={baseline_postings:,}/{baseline_documents:,}',
            flush=True,
        )

    variants: dict[str, Any] = {}
    for name, values in variant_values.items():
        metrics = metric_values[name]
        variants[name] = {
            key: summarize(items)
            for key, items in values.items()
        }
        variants[name]['metrics'] = {
            metric: statistics.fmean(row[metric] for row in metrics)
            for metric in (
                'ndcg_at_10',
                'map_at_100',
                'recall_at_100',
                'mrr_at_20',
            )
        }

    authority_quality: dict[str, Any] = {}
    for name, values in authority_values.items():
        metrics = values['metrics']
        authority_quality[name] = {
            'overlap_at_k': summarize(values['overlap_at_k']),
            'metrics': {
                metric: statistics.fmean(row[metric] for row in metrics)
                for metric in (
                    'ndcg_at_10',
                    'map_at_100',
                    'recall_at_100',
                    'mrr_at_20',
                )
            },
        }

    result = {
        'schema': 'impact_banded_residual_oracle_v1',
        'dataset_root': str(args.dataset_root.resolve()),
        'document_count': documents.shape[0],
        'query_count': len(query_ids),
        'vocabulary_size': documents.shape[1],
        'k': args.k,
        'accelerated_posting_mass_target': args.accelerated_posting_mass,
        'accelerated_posting_mass_actual': actual_mass,
        'seed_document_fraction': args.seed_document_fraction,
        'seed_per_selected_term': args.seed_per_selected_term,
        'frontier_scope': args.frontier_scope,
        'selected_term_count': int(np.count_nonzero(selected_terms)),
        'block_shifts': list(block_shifts),
        'error_ratios': list(error_ratios),
        'lazy_block_batches': list(lazy_block_batches),
        'hierarchical_super_shifts': list(hierarchical_super_shifts),
        'frontier_impact_bits': list(frontier_impact_bits),
        'sole_authority_impact_bits': list(sole_authority_impact_bits),
        'include_fixed_u8_authority': args.include_fixed_u8_authority,
        'publication': publication,
        'sole_authority_quality': authority_quality,
        'variants': variants,
        'query_rows': query_rows,
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
