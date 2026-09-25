"""Authentic three-checkpoint TRAIN diagnostics, not causal attribution."""

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank(scores, gold):
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.ndim == 1 and len(scores) >= 100 and np.isfinite(scores).all(), 'invalid scores')
    require(len(gold) == len(set(gold)) > 0 and all(type(i) is int and 0 <= i < len(scores) for i in gold),
            'invalid complete positive IDs')
    order = np.lexsort((np.arange(len(scores)), -scores))
    inverse = np.empty(len(scores), dtype=np.int64)
    inverse[order] = np.arange(1, len(scores) + 1)
    ranks = inverse[gold]
    ideal = sum(1 / np.log2(i + 1) for i in range(1, min(10, len(gold)) + 1))
    return dict(top100=order[:100].tolist(), gold_ids=gold, gold_ranks=ranks.tolist(),
                ndcg10=float(sum(1 / np.log2(i + 1) for i in ranks if i <= 10) / ideal),
                recall100=float(np.mean(ranks <= 100)))


def analyze(data, middle_scores, middle_rankings):
    require(set(middle_scores) == set(middle_rankings) == set(data['query_ids']), 'query coverage changed')
    pairs, per_query = [], []
    for q in data['query_ids']:
        pool, gold = data['pools'][q], data['gold'][q]
        values = [data['endpoint_scores']['initial'][q], middle_scores[q],
                  data['endpoint_scores']['D-192'][q]]
        require(all(len(v) == len(pool) and np.isfinite(v).all() for v in values), 'pool score mismatch')
        for positive in gold:
            pi = pool.index(positive)
            for ni, rival in enumerate(pool):
                if rival in gold:
                    continue
                margins = [v[pi] - v[ni] for v in values]
                increments = [margins[1] - margins[0], margins[2] - margins[1]]
                require(abs(sum(increments) - (margins[2] - margins[0])) < 1e-10, 'telescoping failed')
                ahead = [m > 0 or (m == 0 and positive < rival) for m in margins]
                pairs.append(dict(query_id=q, domain=data['queries'][q]['subset'], positive_id=positive,
                    rival_id=rival, margins=margins, increments=increments, ahead=ahead,
                    endpoint_lost=ahead[0] and not ahead[2], endpoint_gained=not ahead[0] and ahead[2],
                    lost_already_midpoint=ahead[0] and not ahead[1] and not ahead[2],
                    lost_after_midpoint=ahead[0] and ahead[1] and not ahead[2],
                    regained=ahead[0] and not ahead[1] and ahead[2],
                    transient_gain=not ahead[0] and ahead[1] and not ahead[2]))
        ranks = [data['endpoint_rankings']['initial'][q], middle_rankings[q],
                 data['endpoint_rankings']['D-192'][q]]
        require(all(r['gold_ids'] == gold for r in ranks), 'gold coverage changed')
        per_query.append(dict(query_id=q, domain=data['queries'][q]['subset'],
            ndcg10=[r['ndcg10'] for r in ranks], recall100=[r['recall100'] for r in ranks],
            middle_only_top100=sorted(set(ranks[1]['top100']) - set(pool))))
    domains = {}
    for domain in ('fever', 'hotpotqa', 'nq'):
        ps, qs = [r for r in pairs if r['domain'] == domain], [r for r in per_query if r['domain'] == domain]
        require(qs and ps, 'missing diagnostic domain')
        totals = {k: sum(r[k] for r in ps) for k in (
            'endpoint_lost', 'endpoint_gained', 'lost_already_midpoint', 'lost_after_midpoint',
            'regained', 'transient_gain')}
        domains[domain] = dict(queries=len(qs), pairs=len(ps), **totals,
            ndcg10=np.mean([r['ndcg10'] for r in qs], axis=0).tolist(),
            recall100=np.mean([r['recall100'] for r in qs], axis=0).tolist(),
            middle_only_top100=sum(len(r['middle_only_top100']) for r in qs))
    return dict(checkpoints=[0, 96, 192], pairs=pairs, per_query=per_query, domains=domains,
                diagnostic_TRAIN_only=True, causal_domain_attribution=False,
                heldout_quality_evaluation=False, overall_goal_qualified=False)
