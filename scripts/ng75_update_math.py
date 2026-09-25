"""Local margin diagnostics for actual shared-parameter displacements.

These helpers never step an optimizer or select probes from final outcomes.
The Taylor prediction is measured, not assumed to equal a finite change.
"""

import hashlib
import math

import numpy as np
import torch


DOMAINS = ('fever', 'hotpotqa', 'nq')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_sentinel(rows, per_domain=8):
    require(type(per_domain) is int and per_domain > 0, 'invalid probe count')
    require(len({r['query_id'] for r in rows}) == len(rows), 'duplicate query')
    require(all(r['split'] == 'TRAIN' and r['surface'] == 'TRAIN_SENTINEL'
                and r['domain'] in DOMAINS for r in rows), 'only TRAIN sentinel permitted')
    result = []
    for domain in DOMAINS:
        selected = sorted((r for r in rows if r['domain'] == domain), key=lambda r: (
            hashlib.sha256(('NG75-probes-v1\0' + r['query_id']).encode()).hexdigest(), r['query_id']))
        require(len(selected) >= per_domain, 'insufficient domain probes')
        result.extend(selected[:per_domain])
    return result


def initial_pairs(row):
    require(row['split'] == 'TRAIN', 'only TRAIN pair probes permitted')
    gold, head = row['gold_ids'], row['top100']
    require(gold and len(set(gold)) == len(gold), 'all unique positives required')
    require(head and len(set(head)) == len(head), 'unique initial head required')
    require(all(type(d) is int and d >= 0 for d in gold + head), 'invalid document ID')
    rivals = [d for d in head if d not in gold]
    require(rivals, 'no initial rival; do not silently skip query')
    selected = list(dict.fromkeys((rivals[0], rivals[-1])))
    return [(p, n) for p in gold for n in selected]


def parameter_layout(model, optimizer):
    named = list(model.named_parameters())
    ids = [id(p) for _, p in named]
    require(len(ids) == len(set(ids)), 'tied parameters counted more than once')
    require(len(optimizer.param_groups) == 2, 'expected trunk/head optimizer layout')
    groups = {id(p): label for group, label in zip(optimizer.param_groups, ('trunk', 'head'))
              for p in group['params']}
    flat = [id(p) for group in optimizer.param_groups for p in group['params']]
    require(len(flat) == len(set(flat)) and set(flat) == set(ids), 'optimizer parameter coverage changed')
    return named, [groups[id(p)] for _, p in named]


def copy_parameters(named):
    return [p.detach().clone() for _, p in named]


def restore_parameters(named, values):
    require(len(named) == len(values), 'parameter inventory changed')
    require(all(p.shape == v.shape and p.dtype == v.dtype
                for (_, p), v in zip(named, values, strict=True)),
            'parameter shape/precision changed')
    with torch.no_grad():
        for (_, parameter), value in zip(named, values, strict=True):
            parameter.copy_(value)


def displacement(before, after):
    require(before and len(before) == len(after), 'displacement inventory changed')
    result = []
    for a, b in zip(before, after, strict=True):
        require(a.shape == b.shape and a.dtype == b.dtype, 'displacement tensor identity changed')
        delta = b - a
        require(torch.isfinite(delta).all().item(), 'nonfinite actual displacement')
        result.append(delta)
    return result


def dot_groups(gradients, delta, groups):
    require(len(gradients) == len(delta) == len(groups), 'Jacobian layout mismatch')
    totals = dict(trunk=0., head=0.)
    for gradient, step, group in zip(gradients, delta, groups, strict=True):
        require(group in totals, 'unknown parameter group')
        if gradient is not None:
            require(gradient.shape == step.shape and torch.isfinite(gradient).all().item(),
                    'invalid margin Jacobian')
            totals[group] += float((gradient.double() * step.double()).sum())
    require(all(math.isfinite(v) for v in totals.values()), 'invalid projected margin')
    return totals


def margin_projection(query, positive, rival, named, delta, groups):
    """Differentiate both paths of one margin at the same parameter state."""
    require(query.ndim == positive.ndim == rival.ndim == 1, 'one vector per role required')
    require(query.shape == positive.shape == rival.shape, 'role shape mismatch')
    require(all(torch.isfinite(v).all().item() for v in (query, positive, rival)),
            'nonfinite role vector')
    q, difference = query.double(), positive.double() - rival.double()
    parameters = [p for _, p in named]
    full = torch.dot(q, difference)
    query_path = torch.dot(q, difference.detach())
    document_path = torch.dot(q.detach(), difference)
    gq = torch.autograd.grad(query_path, parameters, retain_graph=True, allow_unused=True)
    gd = torch.autograd.grad(document_path, parameters, retain_graph=True, allow_unused=True)
    gf = torch.autograd.grad(full, parameters, allow_unused=True)
    norm, error, max_error = 0., 0., 0.
    for parameter, a, b, c in zip(parameters, gq, gd, gf, strict=True):
        zero = torch.zeros_like(parameter)
        a, b, c = [zero if g is None else g for g in (a, b, c)]
        torch.testing.assert_close(a + b, c, rtol=2e-5, atol=2e-7)
        difference = (a.double() + b.double()) - c.double()
        norm += float(c.double().square().sum())
        error += float(difference.square().sum())
        max_error = max(max_error, float(difference.abs().max()))
    pieces = {role: dot_groups(g, delta, groups) for role, g in (('query', gq), ('document', gd))}
    direct = sum(dot_groups(gf, delta, groups).values())
    predicted = sum(sum(values.values()) for values in pieces.values())
    require(math.isclose(predicted, direct, rel_tol=2e-5, abs_tol=2e-7), 'projected chain rule mismatch')
    return dict(semantic_margin=float(full.detach()), predicted=direct, contributions=pieces,
                gradient_chain_max_abs=max_error, gradient_chain_relative_l2=math.sqrt(error / max(norm, 1e-30)))


def direction(value, floor):
    require(math.isfinite(value) and math.isfinite(floor) and floor >= 0, 'invalid direction boundary')
    return 'expand' if value > floor else 'shrink' if value < -floor else 'stationary'


def describe_change(before, after, prediction, no_update_error):
    require(all(math.isfinite(v) for v in (before, after, prediction, no_update_error))
            and no_update_error >= 0, 'invalid measured change')
    floor = max(1e-7, 10 * no_update_error)
    actual = after - before
    return dict(before_margin=before, after_margin=after, actual_change=actual,
                predicted_change=prediction, taylor_residual=actual - prediction,
                no_update_error=no_update_error, numerical_floor=floor,
                actual_direction=direction(actual, floor), predicted_direction=direction(prediction, floor))


def summarize(rows):
    result = {}
    for surface in sorted({r['surface'] for r in rows}):
        for domain in DOMAINS:
            selected = [r for r in rows if r['surface'] == surface and r['domain'] == domain]
            if not selected:
                continue
            origins = {}
            for origin in ('all', 'original_positive', 'added_vs_original'):
                group = [r for r in selected if origin == 'all' or r['origin'] == origin]
                origins[origin] = dict(
                    pairs=len(group), queries=len({r['query_id'] for r in group}),
                    positives=len({(r['query_id'], r['positive_id']) for r in group}),
                    directions={k: sum(r['actual_direction'] == k for r in group)
                                for k in ('expand', 'shrink', 'stationary')},
                    predicted_sign_matches=sum(r['actual_direction'] == r['predicted_direction'] for r in group),
                    mean_actual_change=float(np.mean([r['actual_change'] for r in group])) if group else None,
                    mean_abs_residual=float(np.mean([abs(r['taylor_residual']) for r in group])) if group else None)
                per_query = [np.mean([r['actual_change'] for r in group if r['query_id'] == q])
                             for q in sorted({r['query_id'] for r in group})]
                origins[origin]['query_balanced_mean_change'] = float(np.mean(per_query)) if per_query else None
            result[surface + '/' + domain] = origins
    return result
