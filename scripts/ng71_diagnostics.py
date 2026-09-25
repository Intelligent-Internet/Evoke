"""TRAIN snapshot score-space diagnostics, not parameter-gradient inference.

All four objectives see the same fixed initial encoder. NumPy derivatives are
checked against Torch autograd, without loading or updating any model weights.
"""

from collections import Counter

import numpy as np
import torch

import ng71_ranking as reference
import ng71_training as training


VECTORS = ('positive_mask', 'judged_negative_mask', 'teacher_scores',
           'global_ranks', 'hybrid_scores', 'lexical_scores')


def original_record(record):
    positions = {doc: i for i, doc in enumerate(record['pool'])}
    indices = [positions[doc] for doc in record['original_pool']]
    subset = {**record, 'pool': list(record['original_pool']), 'sources': {}}
    for name in VECTORS:
        subset[name] = [record[name][i] for i in indices]
    if sum(subset['positive_mask']) != record['total_positives']:
        raise ValueError('original arm would lose a known positive')
    return subset


def legacy_target(record, config):
    positives = np.asarray(record['positive_mask'], dtype=np.float64)
    teacher = np.asarray(record['teacher_scores'], dtype=np.float64)
    if (not positives.any() or positives.sum() != record['total_positives']
            or teacher.shape != positives.shape or not np.isfinite(teacher).all()):
        raise ValueError('invalid complete positive/teacher target')
    logits = teacher / config['teacher_temperature']
    probs = np.exp(logits - logits.max())
    return .5 * positives / positives.sum() + .5 * probs / probs.sum()


def assert_close(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    if (a.shape != b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all()):
        raise ValueError('invalid independent score derivative')
    error = float(np.max(np.abs(a - b)))
    if error > 1e-12:
        raise ValueError('independent score derivative exceeds 1e-12')
    return error


def source_gradients(record, pairs, scores):
    """Attribute pair derivatives before cancellation in candidate score space."""
    sources = {'original_pool': [doc for doc, positive in
                                zip(record['pool'], record['positive_mask'], strict=True)
                                if doc in record['original_pool'] and not positive],
               **record['sources']}
    owners = {}
    summaries = {}
    for name, docs in sources.items():
        for doc in docs:
            if doc in owners:
                raise ValueError('duplicate negative witness source')
            owners[doc] = name
        summaries[name] = dict(
            eligible_pairs=0, coefficient_mass=0., abs_pair_derivative_mass=0.,
            promote_positive_pairs=0, reduce_positive_margin_pairs=0,
            cross_top10_abs_derivative_mass=0.,
            cross_top100_abs_derivative_mass=0.)
    margins, derivatives = [], []
    temperature = pairs['student_temperature']
    reconstructed = np.zeros(len(scores), dtype=np.float64)
    for (p, n), target, coefficient in zip(
            pairs['indices'], pairs['targets'], pairs['coefficients'], strict=True):
        if not record['positive_mask'][p] or record['positive_mask'][n]:
            raise ValueError('invalid positive-negative pair attribution')
        summary = summaries[owners[record['pool'][n]]]
        margin = (scores[p] - scores[n]) / temperature
        probability = float(np.exp(-np.logaddexp(0., -margin)))
        derivative = coefficient * (probability - target) / temperature
        reconstructed[p] += derivative
        reconstructed[n] -= derivative
        margins.append(float(margin))
        derivatives.append(float(derivative))
        summary['eligible_pairs'] += 1
        summary['coefficient_mass'] += coefficient
        summary['abs_pair_derivative_mass'] += abs(derivative)
        summary['promote_positive_pairs'] += int(derivative < 0)
        summary['reduce_positive_margin_pairs'] += int(derivative > 0)
        for cutoff in (10, 100):
            if ((record['global_ranks'][p] <= cutoff)
                    != (record['global_ranks'][n] <= cutoff)):
                summary[f'cross_top{cutoff}_abs_derivative_mass'] += abs(derivative)
    return dict(sources=summaries, pair_student_margins=margins,
                pair_score_derivatives=derivatives), reconstructed


def objective(record, config, name):
    scores = np.asarray(record['hybrid_scores'], dtype=np.float64)
    if scores.shape != (len(record['pool']),) or not np.isfinite(scores).all():
        raise ValueError('complete finite snapshot scores required')
    variable = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    detail = {}
    if name == 'legacy_CE':
        target = legacy_target(record, config)
        loss = training.legacy_loss(variable, target)
        logsum = np.logaddexp.reduce(scores)
        expected_loss = -float(target @ (scores - logsum))
        expected_gradient = np.exp(scores - logsum) - target
        eligible_pairs = supervised = None
        detail['target'] = target.tolist()
    elif name in ('balanced_soft_pair', 'uniform_soft_pair'):
        uniform = name == 'uniform_soft_pair'
        params = {key: config[key] for key in (
            'ndcg_cutoff', 'recall_cutoff', 'recall_weight', 'pair_floor',
            'teacher_temperature', 'student_temperature')}
        audit = reference.loss_and_gradient(
            scores, np.asarray(record['positive_mask']),
            np.asarray(record['global_ranks']), np.asarray(record['teacher_scores']),
            np.asarray(record['judged_negative_mask']),
            total_positives=record['total_positives'],
            corpus_size=record['corpus_size'], rank_scope=record['rank_scope'],
            metric_weights=not uniform, **params)
        pairs = training.prepare_pairs(record, config, uniform=uniform)
        loss = training.pair_loss(variable, pairs)
        expected_loss, expected_gradient = audit['loss'], audit['gradient']
        eligible_pairs = audit['eligible_pairs']
        supervised = audit['supervised_positives']
        if eligible_pairs != len(pairs['indices']):
            raise ValueError('pair eligibility disagrees with independent reference')
        detail, attributed = source_gradients(record, pairs, scores)
        assert_close(attributed, expected_gradient)
        detail.update(pair_indices=pairs['indices'], pair_targets=pairs['targets'],
                      pair_coefficients=pairs['coefficients'])
    else:
        raise ValueError('unknown frozen NG71 objective')
    loss.backward()
    gradient = variable.grad.detach().numpy()
    error = max(assert_close(gradient, expected_gradient),
                assert_close(float(loss.detach()), expected_loss))
    positives = np.asarray(record['positive_mask'])
    return dict(
        objective=name, pool=record['pool'], loss=float(loss.detach()),
        score_gradient=gradient.tolist(), score_gradient_l1=float(np.abs(gradient).sum()),
        positive_score_gradient_sum=float(gradient[positives].sum()),
        positive_score_promote_count=int((gradient[positives] < 0).sum()),
        positive_score_reduce_count=int((gradient[positives] > 0).sum()),
        eligible_pairs=eligible_pairs, supervised_positives=supervised,
        max_independent_derivative_error=error, **detail)


def diagnose(record, config):
    original = original_record(record)
    result = {}
    for arm, spec in config['arms'].items():
        if spec['pool'] not in ('original', 'witness'):
            raise ValueError('unknown candidate pool definition')
        selected = original if spec['pool'] == 'original' else record
        result[arm] = objective(selected, config['ranking'], spec['objective'])
    if {'A', 'B', 'C', 'D'} <= set(result):
        if result['A']['pool'] != result['B']['pool'] or result['C']['pool'] != result['D']['pool']:
            raise ValueError('paired arm candidate identities changed')
    return result


def batch_audit(records, order, config):
    lookup = {r['query_id']: r for r in records}
    if (len(lookup) != len(records) or len(records) != config['pilot']['train_queries']
            or Counter(order) != Counter({q: config['pilot']['epochs'] for q in lookup})):
        raise ValueError('fixed TRAIN exposure manifest changed')
    batch_size = config['optimizer']['queries_per_update']
    if len(order) != config['pilot']['updates'] * batch_size:
        raise ValueError('fixed update count changed')
    empty = {arm: [] for arm, spec in config['arms'].items()
             if spec['objective'] in ('balanced_soft_pair', 'uniform_soft_pair')}
    for start in range(0, len(order), batch_size):
        for arm in empty:
            if not any(lookup[q]['arm_score_diagnostics'][arm]['eligible_pairs']
                       for q in order[start:start + batch_size]):
                empty[arm].append(start // batch_size)
    if any(empty.values()):
        raise ValueError('full manifest contains an unsupervised optimizer batch')
    return dict(passed=True, updates=config['pilot']['updates'],
                exposures=len(order), empty_batches_by_arm=empty)


def summarize(records, config):
    report = {}
    for domain in config['pilot']['domains']:
        rows = [r for r in records if r['domain'] == domain]
        report[domain] = {}
        for arm in config['arms']:
            values = [r['arm_score_diagnostics'][arm] for r in rows]
            if not values:
                raise ValueError('missing domain score diagnostics')
            totals = {key: sum(v[key] for v in values) for key in (
                'loss', 'score_gradient_l1', 'positive_score_gradient_sum',
                'positive_score_promote_count', 'positive_score_reduce_count')}
            sources = {}
            if arm in ('B', 'D'):
                for name in values[0]['sources']:
                    sources[name] = {
                        key: sum(v['sources'][name][key] for v in values)
                        for key in values[0]['sources'][name]}
            report[domain][arm] = dict(
                queries=len(rows), **totals, sources=sources,
                max_independent_derivative_error=max(
                    v['max_independent_derivative_error'] for v in values))
    return report
