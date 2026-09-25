"""Deterministic bounded witness selection from complete corpus score vectors.

The execution layer must independently verify score accumulation and bind model,
corpus, teacher, and qrel hashes. This module does not certify those inputs and
never computes candidate-local ranks or truncates the original positive set.
"""

import hashlib

import numpy as np


def full_order(scores, stable_ids, corpus_size):
    values = np.asarray(scores)
    if (values.shape != (corpus_size,) or values.dtype.kind != 'f'
            or not np.isfinite(values).all()):
        raise ValueError('expected complete finite floating corpus score vector')
    if len(stable_ids) != corpus_size or len(set(stable_ids)) != corpus_size:
        raise ValueError('corpus identities missing or duplicated')
    if not all(isinstance(value, str) and value for value in stable_ids):
        raise ValueError('stable document IDs must be nonempty strings')
    order = np.lexsort((np.asarray(stable_ids), -values))
    ranks = np.empty(corpus_size, dtype=np.int64)
    ranks[order] = np.arange(1, corpus_size + 1)
    return order, ranks


def mine(query_id, original_pool, positive_ids, stable_ids, scores, config,
         *, corpus_size):
    if not isinstance(query_id, str) or not query_id:
        raise ValueError('query identity required')
    for values in (original_pool, positive_ids):
        if not values or len(set(values)) != len(values):
            raise ValueError('empty or duplicate original pool/positive IDs')
        if any(type(i) is not int or i < 0 or i >= corpus_size for i in values):
            raise ValueError('document ID outside complete corpus')
    if not config['retain_original_pool'] or not config['retain_all_positives']:
        raise ValueError('cannot drop original candidates or positives')
    if config['cross_source_backfill']:
        raise ValueError('cross-source backfill is not this protocol')
    if sum(s['quota'] for s in config['sources']) > config['max_new_documents']:
        raise ValueError('witness quotas exceed frozen budget')
    orders, ranks = {}, {}
    for name in ('hybrid', 'pplx', 'bm25'):
        orders[name], ranks[name] = full_order(scores[name], stable_ids, corpus_size)
    pool = list(original_pool)
    pool.extend(sorted(set(positive_ids) - set(pool), key=lambda i: stable_ids[i]))
    retained = len(pool)
    used = set(pool)
    chosen = {}
    for source in config['sources']:
        name, quota = source['name'], source['quota']
        if type(quota) is not int or quota < 0:
            raise ValueError('invalid source quota')
        if name == 'remaining_hash_sample':
            candidates = sorted(
                (i for i in range(corpus_size) if i not in used),
                key=lambda i: (hashlib.sha256(
                    ('NG-0071/witness-v1/' + query_id + '/' + stable_ids[i]).encode()
                ).hexdigest(), stable_ids[i]))
        else:
            key = name.split('_')[0]
            if key not in orders or name not in (
                    'hybrid_head', 'hybrid_boundary', 'pplx_head', 'bm25_head'):
                raise ValueError('unknown witness source')
            low, high = source['rank_min'], source['rank_max']
            if not 1 <= low <= high:
                raise ValueError('invalid corpus rank interval')
            candidates = [int(i) for i in orders[key][low - 1:high]]
            if name == 'hybrid_boundary':
                candidates.sort(key=lambda i: (abs(int(ranks[key][i]) - 100), stable_ids[i]))
        selected = [i for i in candidates if i not in used][:quota]
        if name in chosen:
            raise ValueError('duplicate witness source')
        chosen[name] = selected
        used.update(selected)
        pool.extend(selected)
    if len(pool) > config['max_total_documents']:
        raise ValueError('pool overflow; never drop positives or original candidates')
    return {
        'query_id': query_id, 'pool': pool, 'original_pool': list(original_pool),
        'positive_ids': list(positive_ids),
        'positive_mask': [i in set(positive_ids) for i in pool],
        'judged_negative_mask': [False] * len(pool),
        'teacher_scores': [float(scores['pplx'][i]) for i in pool],
        'global_ranks': [int(ranks['hybrid'][i]) for i in pool],
        'rank_scope': 'full_corpus_reference_snapshot',
        'total_positives': len(positive_ids), 'corpus_size': corpus_size,
        'new_witness_count': len(pool) - retained, 'sources': chosen,
        'independent_score_verification_required': True,
    }
