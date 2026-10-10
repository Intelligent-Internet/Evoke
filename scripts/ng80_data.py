"""TRAIN-only identities and numerical controls for the NG80 campaign."""

import hashlib
import unicodedata

import numpy as np


DOMAINS = ('fever', 'hotpotqa', 'nq')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split()).casefold()


def selection(rows, per_domain=512):
    ids = [r['query_id'] for r in rows]
    require(len(ids) == len(set(ids)), 'duplicate query ID')
    require(set(r['subset'] for r in rows) == set(DOMAINS), 'domain set changed')
    hold = set()
    for domain in DOMAINS:
        candidates = [i for i, row in enumerate(rows) if row['subset'] == domain]
        require(len(candidates) > per_domain, 'insufficient domain rows')
        candidates.sort(key=lambda i: hashlib.sha256(
            ('NG80/split-v1/' + rows[i]['query_id']).encode()).hexdigest())
        hold.update(candidates[:per_domain])
    train = [i for i in range(len(rows)) if i not in hold]
    diagnostic = sorted(hold)
    a = {normalized(rows[i]['query']) for i in train}
    b = {normalized(rows[i]['query']) for i in diagnostic}
    require(not a & b, 'normalized query crosses new split')
    return {'train': train, 'diagnostic': diagnostic,
            'diagnostic_is_independent_holdout': False}


def panels(pools, labels, count, background_count=16):
    require(len(pools) == len(labels) > 0, 'pool/label alignment')
    background = sorted(range(count), key=lambda i: hashlib.sha256(
        ('NG80/background-v1/' + str(i)).encode()).hexdigest())[:background_count]
    output = []
    for pool, gold in zip(pools, labels, strict=True):
        require(pool and gold and len(pool) == len(set(pool)), 'invalid source pool')
        require(all(type(i) is int and 0 <= i < count for i in pool), 'invalid ID')
        require(len(gold) == len(set(gold)) and set(gold) <= set(pool), 'lost positive')
        output.append(list(dict.fromkeys([*pool, *background])))
    return output, background


def ridge(xq, yq, xd, yd, regularization=1e-3):
    """Equal role mass; unpenalized intercept, TRAIN-only fit."""
    require(regularization > 0, 'positive ridge required')
    p, out = xq.shape[1], yq.shape[1]
    gram, cross = np.zeros((p + 1, p + 1)), np.zeros((p + 1, out))
    for x, y in ((xq, yq), (xd, yd)):
        require(len(x) > 0 and x.shape[1] == p and y.shape == (len(x), out),
                'ridge dimensions')
        require(np.isfinite(x).all() and np.isfinite(y).all(), 'nonfinite ridge')
        design = np.column_stack((x.astype(np.float64), np.ones(len(x))))
        gram += .5 * (design.T @ design) / len(x)
        cross += .5 * (design.T @ y.astype(np.float64)) / len(x)
    gram[np.arange(p), np.arange(p)] += regularization
    weights = np.linalg.solve(gram, cross)
    require(np.isfinite(weights).all(), 'nonfinite projection')
    return weights


def project(x, weights):
    values = np.asarray(x, dtype=np.float64) @ weights[:-1] + weights[-1]
    norm = np.linalg.norm(values, axis=1)
    require(np.isfinite(values).all() and np.all(norm > 1e-12), 'invalid projection')
    return values / norm[:, None]


def centered_error(student, teacher):
    error = np.asarray(student, dtype=np.float64) - np.asarray(teacher, dtype=np.float64)
    require(error.ndim == 1 and len(error) > 1 and np.isfinite(error).all(),
            'invalid score field')
    error -= error.mean()
    return float(error @ error / len(error))


def metrics(scores, teacher, pool, gold):
    scores, teacher, pool = map(np.asarray, (scores, teacher, pool))
    require(scores.shape == teacher.shape == pool.shape and scores.ndim == 1,
            'metric alignment')
    require(np.isfinite(scores).all() and np.isfinite(teacher).all(), 'nonfinite scores')
    require(len(set(pool.tolist())) == len(pool) and set(gold) <= set(pool),
            'invalid metric IDs')
    order = pool[np.lexsort((pool, -scores))]
    other = pool[np.lexsort((pool, -teacher))]
    cutoff = min(10, len(pool))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(gold))))
    ndcg = sum(1 / np.log2(i + 2) for i, d in enumerate(order[:10]) if d in gold) / ideal
    return {'panel_ndcg10': float(ndcg),
            'teacher_top10_overlap': len(set(order[:cutoff]) & set(other[:cutoff])) / cutoff,
            'centered_score_mse': centered_error(scores, teacher)}
