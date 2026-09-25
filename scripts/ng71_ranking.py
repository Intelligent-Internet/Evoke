"""NG71 binary-qrel objective reference, not a mining or training pipeline.

Ranks must come from a separately verified full-corpus snapshot. This module
cannot establish corpus provenance from an integer rank array. Scores are fresh
student scores; ranks, teacher targets and metric weights stay fixed during
differentiation. Unjudged documents are never implicit hard negatives.
"""

from __future__ import annotations

import math

import numpy as np


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def _ranks(values: np.ndarray, corpus_size: int) -> np.ndarray:
    _positive_int(corpus_size, 'corpus_size')
    ranks = np.asarray(values)
    if ranks.ndim != 1 or ranks.size == 0 or ranks.dtype.kind not in 'iu':
        raise ValueError('ranks must be a nonempty integer vector')
    if np.any(ranks < 1) or np.any(ranks > corpus_size):
        raise ValueError('ranks must be within the complete corpus')
    if np.unique(ranks).size != ranks.size:
        raise ValueError('ranks require a deterministic, unique tie-break')
    return ranks


def binary_metrics(
    positive_ranks: np.ndarray,
    corpus_size: int,
    *,
    ndcg_cutoff: int = 10,
    recall_cutoff: int = 100,
) -> tuple[float, float]:
    """Metrics on all known positives, including those outside the top-k."""
    ranks = _ranks(positive_ranks, corpus_size)
    _positive_int(ndcg_cutoff, 'ndcg_cutoff')
    _positive_int(recall_cutoff, 'recall_cutoff')
    ideal = sum(
        1 / math.log2(1 + rank)
        for rank in range(1, min(ranks.size, ndcg_cutoff) + 1)
    )
    dcg = sum(
        1 / math.log2(1 + int(rank))
        for rank in ranks if rank <= ndcg_cutoff
    )
    return dcg / ideal, float(np.mean(ranks <= recall_cutoff))


def pair_weight(
    positive_rank: int,
    competitor_rank: int,
    positive_count: int,
    *,
    ndcg_cutoff: int = 10,
    recall_cutoff: int = 100,
    recall_weight: float = 1.0,
    pair_floor: float = 0.05,
) -> float:
    """Binary positive/nonpositive swap priority; unjudged is only a proxy."""
    for name, value in (
        ('positive_rank', positive_rank),
        ('competitor_rank', competitor_rank),
        ('positive_count', positive_count),
        ('ndcg_cutoff', ndcg_cutoff),
        ('recall_cutoff', recall_cutoff),
    ):
        _positive_int(value, name)
    if positive_rank == competitor_rank:
        raise ValueError('a pair must contain distinct ranked documents')
    _nonnegative(recall_weight, 'recall_weight')
    _nonnegative(pair_floor, 'pair_floor')
    ideal = sum(
        1 / math.log2(1 + rank)
        for rank in range(1, min(positive_count, ndcg_cutoff) + 1)
    )

    def discount(rank: int) -> float:
        return 1 / math.log2(1 + rank) if rank <= ndcg_cutoff else 0.0

    ndcg = abs(discount(positive_rank) - discount(competitor_rank)) / ideal
    recall = abs(
        int(positive_rank <= recall_cutoff)
        - int(competitor_rank <= recall_cutoff)
    ) / positive_count
    return ndcg + recall_weight * recall + pair_floor / positive_count


def _sigmoid(value: float) -> float:
    return float(np.exp(-np.logaddexp(0.0, -value)))


def loss_and_gradient(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    global_ranks: np.ndarray,
    teacher_scores: np.ndarray,
    judged_negative_mask: np.ndarray,
    *,
    total_positives: int,
    corpus_size: int,
    rank_scope: str,
    student_temperature: float = 1.0,
    teacher_temperature: float = 0.04,
    ndcg_cutoff: int = 10,
    recall_cutoff: int = 100,
    recall_weight: float = 1.0,
    pair_floor: float = 0.05,
    metric_weights: bool = True,
) -> dict:
    """Return a single query's scalar loss and exact score-space gradient.

    The caller must bind total_positives to the complete qrel manifest. A
    missing eligible pair contributes zero without changing that denominator.
    Masks distinguish known positives, judged irrelevant and unjudged; label
    provenance must be checked before constructing the masks.
    """
    if rank_scope != 'full_corpus_reference_snapshot':
        raise ValueError('candidate-local ranks are not valid metric weights')
    ranks = _ranks(global_ranks, corpus_size)
    scores = np.asarray(scores, dtype=np.float64)
    teacher = np.asarray(teacher_scores, dtype=np.float64)
    positives = np.asarray(positive_mask)
    negatives = np.asarray(judged_negative_mask)
    for values in (scores, teacher, positives, negatives):
        if values.shape != ranks.shape:
            raise ValueError('all vectors must have identical one-dimensional shape')
    if positives.dtype.kind != 'b' or negatives.dtype.kind != 'b':
        raise ValueError('label masks must be boolean, not graded labels')
    if not np.isfinite(scores).all() or not np.isfinite(teacher).all():
        raise ValueError('scores must be finite')
    if np.any(positives & negatives):
        raise ValueError('positive and judged-negative labels conflict')
    _positive_int(total_positives, 'total_positives')
    if positives.sum() != total_positives:
        raise ValueError('the pool must retain all known positives')
    for name, value in (
        ('student_temperature', student_temperature),
        ('teacher_temperature', teacher_temperature),
    ):
        _nonnegative(value, name)
        if value == 0:
            raise ValueError(f'{name} must be positive')
    _positive_int(ndcg_cutoff, 'ndcg_cutoff')
    _positive_int(recall_cutoff, 'recall_cutoff')
    _nonnegative(recall_weight, 'recall_weight')
    _nonnegative(pair_floor, 'pair_floor')
    if not isinstance(metric_weights, bool):
        raise ValueError('metric_weights must be boolean')

    loss = 0.0
    gradient = np.zeros_like(scores)
    eligible_pairs = 0
    supervised_positives = 0
    unresolved_pairs = 0
    for p in np.flatnonzero(positives):
        pairs = []
        for n in np.flatnonzero(~positives):
            if negatives[n]:
                target, confidence = 1.0, 1.0
            else:
                margin = float((teacher[p] - teacher[n]) / teacher_temperature)
                if not math.isfinite(margin):
                    raise ValueError('teacher margin overflow')
                target = _sigmoid(margin)
                confidence = 2 * target - 1
                if confidence <= 0:
                    unresolved_pairs += 1
                    continue
            weight = pair_weight(
                int(ranks[p]), int(ranks[n]), total_positives,
                ndcg_cutoff=ndcg_cutoff, recall_cutoff=recall_cutoff,
                recall_weight=recall_weight, pair_floor=pair_floor,
            ) if metric_weights else 1.0
            if weight > 0:
                pairs.append((n, target, confidence, weight))
        # Confidence scales the update; normalizing it away amplifies weak
        # teacher evidence when a positive has only one eligible witness.
        denominator = sum(weight for _, _, _, weight in pairs)
        if not denominator:
            continue
        supervised_positives += 1
        eligible_pairs += len(pairs)
        for n, target, confidence, weight in pairs:
            z = float((scores[p] - scores[n]) / student_temperature)
            if not math.isfinite(z):
                raise ValueError('student margin overflow')
            scale = confidence * weight / denominator / total_positives
            loss += scale * float(np.logaddexp(0.0, z) - target * z)
            derivative = scale * (_sigmoid(z) - target) / student_temperature
            gradient[p] += derivative
            gradient[n] -= derivative
    return {
        'loss': loss,
        'gradient': gradient,
        'eligible_pairs': eligible_pairs,
        'supervised_positives': supervised_positives,
        'total_positives': total_positives,
        'unresolved_pairs': unresolved_pairs,
    }
