"""One-sided trusted-margin retention; no mining or training on import.

Pair identities, confidence and metric weights must be frozen before training.
The objective preserves selected score relations, not arbitrary shared-parameter
updates or full-corpus ranking quality. All known positives stay in the mean.
"""

import math

import numpy as np
import torch

from ng71_ranking import pair_weight


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preference(positive, rival, judged_negative, temperature):
    require(type(judged_negative) is bool, 'negative label must be boolean')
    require(math.isfinite(temperature) and temperature > 0,
            'invalid teacher temperature')
    for value in (positive, rival):
        require(value is None or math.isfinite(value), 'invalid teacher score')
    if judged_negative:
        return 'judged_negative', 1.
    if positive is None or rival is None:
        return 'unobserved', 0.
    margin = (positive - rival) / temperature
    require(math.isfinite(margin), 'teacher margin overflow')
    if margin <= 0:
        return ('tie' if margin == 0 else 'opposes'), 0.
    # Match original D confidence, including rounding of extremely small gaps.
    confidence = 2 * float(np.exp(-np.logaddexp(0., -margin))) - 1
    return 'agrees', confidence


def prepare_anchors(record, config):
    pool = record['pool']
    count = len(pool)
    require(count == len(set(pool)) > 0, 'duplicate or empty pool')
    require(record['rank_scope'] == 'full_corpus_reference_snapshot',
            'candidate-local ranks are not valid')
    for key in ('positive_mask', 'judged_negative_mask', 'teacher_scores',
                'global_ranks', 'hybrid_scores'):
        require(len(record[key]) == count, 'witness shape changed')
    pm, nm = record['positive_mask'], record['judged_negative_mask']
    require(all(type(v) is bool for v in pm + nm), 'labels must be boolean')
    require(not any(p and n for p, n in zip(pm, nm)), 'conflicting labels')
    total = record['total_positives']
    require(type(total) is int and sum(pm) == total > 0,
            'all positive denominator changed')
    require(set(record['positive_ids']) == {pool[i] for i, p in enumerate(pm) if p}
            and len(record['positive_ids']) == total, 'positive identity changed')
    ranks = record['global_ranks']
    require(len(set(ranks)) == count and all(type(r) is int
            and 1 <= r <= record['corpus_size'] for r in ranks), 'invalid global ranks')
    scores = record['hybrid_scores']
    require(all(math.isfinite(s) for s in scores), 'invalid baseline scores')
    temperature = config['student_temperature']
    require(math.isfinite(temperature) and temperature > 0, 'invalid student temperature')
    rows = []
    for p, positive in enumerate(pm):
        if not positive:
            continue
        pending = []
        for n, other_positive in enumerate(pm):
            if other_positive:
                continue
            status, confidence = preference(record['teacher_scores'][p],
                record['teacher_scores'][n], nm[n], config['teacher_temperature'])
            margin = scores[p] - scores[n]
            require(math.isfinite(margin / temperature), 'baseline margin overflow')
            if margin <= 0 or confidence <= 0:
                continue
            weight = pair_weight(ranks[p], ranks[n], total, **{
                k: config[k] for k in ('ndcg_cutoff', 'recall_cutoff',
                                      'recall_weight', 'pair_floor')})
            if weight > 0:
                pending.append(dict(indices=[p, n], positive_id=pool[p],
                    rival_id=pool[n], baseline_margin=margin, confidence=confidence,
                    metric_weight=weight, supervision=status))
        denominator = sum(r['metric_weight'] for r in pending)
        for row in pending:
            row['coefficient'] = (row['confidence'] * row['metric_weight']
                                  / denominator / total)
        rows.extend(pending)
    return dict(candidate_count=count, total_positives=total,
        covered_positives=len({r['positive_id'] for r in rows}),
        student_temperature=temperature, anchors=rows)


def bernoulli_gap(current, baseline):
    """FP64 convex gap and autograd, zero for margins at/above the baseline.

    Reversing the Bernoulli event avoids subtracting huge positive logits.
    For t <= 1e-3, the fifth-order cumulant expansion avoids cancellation of
    two O(t) terms to obtain an O(t**2) gap. Tested against Decimal references;
    this is a numerical approximation, not a different retention deadband.
    """
    require(current.shape == baseline.shape and current.dtype.is_floating_point
            and baseline.dtype.is_floating_point, 'invalid margin tensors')
    a, a0 = current.double(), baseline.detach().double()
    require(torch.isfinite(a).all() and torch.isfinite(a0).all()
            and (a0 > 0).all(), 'baseline must be finite and strictly positive')
    t = (a0 - a).clamp_min(0)
    require(torch.isfinite(t).all(), 'margin difference overflow')
    r = torch.sigmoid(-a0)
    u = t.clamp_max(1e-3)
    k2 = r * (1 - r)
    small = k2 * u.square() * (.5 + (1 - 2 * r) * u / 6
        + (1 - 6 * r + 6 * r.square()) * u.square() / 24
        + (1 - 2 * r) * (1 - 12 * r + 12 * r.square()) * u.pow(3) / 120)
    zero = torch.zeros_like(a)
    general = (torch.logaddexp(zero, -a) - torch.logaddexp(zero, -a0)
               - r * t)
    return torch.where(t <= 1e-3, small, general)


def retention_loss(scores, manifest):
    require(scores.ndim == 1 and len(scores) == manifest['candidate_count']
            and scores.dtype.is_floating_point and torch.isfinite(scores).all(),
            'invalid scores')
    total = manifest['total_positives']
    require(type(total) is int and total > 0, 'invalid positive denominator')
    temperature = manifest['student_temperature']
    require(math.isfinite(temperature) and temperature > 0, 'invalid temperature')
    rows = manifest['anchors']
    if not rows:
        return scores.double().sum() * 0
    raw = [r['indices'] for r in rows]
    require(all(len(pair) == 2 and pair[0] != pair[1]
                and all(type(i) is int and 0 <= i < len(scores) for i in pair)
                for pair in raw), 'invalid anchor indices')
    require(len({tuple(pair) for pair in raw}) == len(raw), 'duplicate anchor')
    indices = torch.tensor(raw, dtype=torch.long, device=scores.device)
    baseline = torch.tensor([r['baseline_margin'] for r in rows],
                            dtype=torch.float64, device=scores.device) / temperature
    weights = torch.tensor([r['coefficient'] for r in rows],
                           dtype=torch.float64, device=scores.device)
    require(torch.isfinite(weights).all() and (weights > 0).all()
            and weights.sum() <= 1 + 1e-12, 'invalid frozen coefficients')
    # Cast before subtraction; baseline was computed from CPU FP64 CSR scores.
    values = scores.double()
    current = (values[indices[:, 0]] - values[indices[:, 1]]) / temperature
    return (bernoulli_gap(current, baseline) * weights).sum()


def combined_loss(original_loss, scores, anchors, coefficient):
    require(type(coefficient) in (int, float) and math.isfinite(coefficient)
            and coefficient >= 0, 'invalid retention coefficient')
    keep = retention_loss(scores, anchors)
    # Identical bookkeeping, but retain original D's FP32 scalar/gradient at 0.
    total = original_loss if coefficient == 0 else original_loss.double() + coefficient * keep
    return total, keep
