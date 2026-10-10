"""Deterministic NG88 sparse budgets, full-background ranks and cost evidence."""

import numpy as np
from scipy import sparse

from ng85_data import DOMAINS, require, stable_top


PROFILES = {
    'full': (None, None),
    'wide': (96, 768),
    'compact': (64, 384),
    'document': (None, 384),
    'query': (64, None),
}


def prune(matrix, k):
    require(sparse.isspmatrix_csr(matrix) and matrix.has_canonical_format,
            'canonical CSR required')
    require(np.isfinite(matrix.data).all() and (matrix.data > 0).all(),
            'positive finite sparse values required')
    if k is None:
        return matrix
    require(type(k) is int and k > 0, 'positive integer support budget required')
    result = matrix.copy()
    for row in range(result.shape[0]):
        lo, hi = result.indptr[row:row + 2]
        if hi - lo <= k:
            continue
        values = result.data[lo:hi]
        cutoff = np.partition(values, len(values) - k)[len(values) - k]
        keep = values > cutoff
        # Canonical CSR orders coordinates by token ID, including tied values.
        tied = np.flatnonzero(values == cutoff)[:k - int(keep.sum())]
        keep[tied] = True
        values[~keep] = 0
    result.eliminate_zeros()
    return result


def normalized_query(raw, rms, k):
    require(rms.shape == (raw.shape[1],) and np.isfinite(rms).all()
            and (rms >= 0).all(), 'invalid fixed RMS vector')
    values = prune(raw, k)
    result = values.copy()
    for i in range(values.shape[0]):
        lo, hi = values.indptr[i:i + 2]
        weights = values.data[lo:hi]
        denominator = np.sum(weights * rms[values.indices[lo:hi]], dtype=np.float32)
        require(np.isfinite(denominator) and denominator > 0,
                'invalid masked fixed RMS normalization')
        result.data[lo:hi] = np.float32(.9) * weights / denominator
    require(np.isfinite(result.data).all(), 'nonfinite normalized query')
    return result


def score(matrix, query):
    require(sparse.isspmatrix_csc(matrix) and matrix.has_canonical_format
            and query.shape == (1, matrix.shape[1]), 'scoring shape/format')
    weights = query.data.astype(np.float64)
    values = np.asarray(matrix[:, query.indices] @ weights).ravel()
    work = int(np.diff(matrix.indptr)[query.indices].sum())
    require(np.isfinite(values).all(), 'nonfinite corpus scores')
    return values, work


def audit_scores(matrix, query, values, ids):
    expected = (matrix[ids].astype(np.float64)
                @ query.astype(np.float64).T).toarray().ravel()
    error = float(np.max(np.abs(expected - values[ids]), initial=0.))
    require(error <= 1e-9, 'independent selected-row score mismatch')
    return error


def ranked(values, positives):
    values = np.asarray(values)
    require(values.ndim == 1 and np.isfinite(values).all() and positives
            and len(positives) == len(set(positives))
            and all(type(p) is int and 0 <= p < len(values) for p in positives),
            'ranking universe or known-positive identity')
    head = stable_top(values, min(100, len(values)))
    ids = np.arange(len(values))
    ranks = [int(np.count_nonzero(values > values[p])
                 + np.count_nonzero((values == values[p]) & (ids < p)) + 1)
             for p in positives]
    ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(ranks))))
    return dict(gold_ids=positives, gold_ranks=ranks, top100=head,
                top100_scores=values[head].tolist(),
                ndcg10=float(sum(1 / np.log2(r + 1) for r in ranks if r <= 10) / ideal),
                recall10=float(np.mean(np.array(ranks) <= 10)),
                recall100=float(np.mean(np.array(ranks) <= 100)))


def check_rank(record, size):
    ids, ranks, head = (record[k] for k in ('gold_ids', 'gold_ranks', 'top100'))
    require(len(ids) == len(ranks) and ids and len(set(ids)) == len(ids)
            and all(type(i) is int and 0 <= i < size for i in ids)
            and len(set(ranks)) == len(ranks)
            and all(type(r) is int and 1 <= r <= size for r in ranks)
            and len(head) == min(size, 100) and len(head) == len(set(head))
            and all(0 <= i < size for i in head), 'invalid rank receipt')
    scores = record['top100_scores']
    require(len(scores) == len(head) and np.isfinite(scores).all()
            and head == sorted(head, key=lambda i: (-scores[head.index(i)], i)),
            'invalid ranked head order')
    for doc, rank in zip(ids, ranks, strict=True):
        require((rank <= len(head) and head[rank - 1] == doc)
                or (rank > len(head) and doc not in head), 'head/gold rank disagreement')
    ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(ranks))))
    expected = dict(ndcg10=sum(1 / np.log2(r + 1) for r in ranks if r <= 10) / ideal,
                    recall10=sum(r <= 10 for r in ranks) / len(ranks),
                    recall100=sum(r <= 100 for r in ranks) / len(ranks))
    require(all(abs(record[k] - v) <= 1e-12 for k, v in expected.items()),
            'independent metric recomputation failed')


def distribution(values):
    values = np.asarray(values)
    require(values.size > 0 and np.isfinite(values).all(), 'empty/nonfinite statistics')
    return dict(mean=float(values.mean()), p50=float(np.quantile(values, .5)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    deviation = values.std()
    return (values - values.mean()) / deviation if deviation > 0 else np.zeros_like(values)


def paired(left, right, domains, seed=88001):
    require(len(left) == len(right) == len(domains), 'paired rows mismatch')
    delta = np.asarray(right) - np.asarray(left)
    rng, samples, means = np.random.default_rng(seed), np.zeros(5000), []
    for domain in DOMAINS:
        part = delta[np.asarray(domains) == domain]
        require(part.size > 0 and np.isfinite(part).all(), 'paired domain missing')
        means.append(float(part.mean()))
        samples += rng.choice(part, (5000, len(part))).mean(1) / len(DOMAINS)
    return dict(delta=float(np.mean(means)),
                descriptive_interval_95=np.quantile(samples, [.025, .975]).tolist())
