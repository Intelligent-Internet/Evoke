"""NG71 TRAIN-only selection and teacher-supervision feasibility audit.

This does not mine witnesses, infer on DEV/LOCKED, or certify global ranks.
Source-mined negatives stay unjudged; NG70 provenance is not a human qrel.
"""

from collections import Counter
import hashlib
import json
import unicodedata

import numpy as np

import audit_ng70_provenance as provenance


ANCHORS = {
    'NG-0069/teacher-v1/targets.json':
        'dda350fe906ba9bf84d606ab36ea4038c419126e6ed0ed3b3bdb8d0d6ca478a8',
    'NG-0070/provenance-v1/pair-provenance.json':
        'd6a7dcac3916e8b121e277754c28f3109e6ae3b415a4e87b114e4867c4acfcb7',
    'NG-0067/data/documents.jsonl':
        '16a008e102028b2cd04872dc24bfc60e0f2f9e508a563748020df81499a5b961',
    'NG-0059/data/train.jsonl':
        '146fa7d3d27202debc317faf0338de22395428947af5ce388636a0f2b5fab491',
    'NG-0067/data/train.jsonl':
        '01fbe89cbb8c9ba30330033fb4711dcaa780547c0fa394baba045d74d06ea775',
}


def same_source_text(left, right):
    # The frozen corpus key deduplicates Unicode-equivalent source spellings.
    # Do not normalize the actual model input or permit fuzzy/whitespace drift.
    return unicodedata.normalize('NFC', left) == unicodedata.normalize('NFC', right)


def select(rows, config):
    """One frozen hash order per domain; sentinel is also text-disjoint."""
    if len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate TRAIN identity')
    result = {'pilot': [], 'sentinel': [], 'canary': []}
    count = config['pilot']['queries_per_domain']
    canary = config['pilot']['preflight_train_queries'] // 3
    seen_text = set()
    for domain in config['pilot']['domains']:
        ordered = sorted(
            (r for r in rows if r['subset'] == domain),
            key=lambda r: (hashlib.sha256(
                ('NG-0071/selection-v1/' + r['query_id']).encode()).hexdigest(),
                r['query_id']))
        chosen = []
        for row in ordered:
            identity = provenance.normalized(row['query'])
            if identity not in seen_text:
                chosen.append(row['query_id'])
                seen_text.add(identity)
            if len(chosen) == count * 2:
                break
        if len(chosen) != count * 2:
            raise ValueError('insufficient disjoint TRAIN queries')
        result['pilot'].extend(chosen[:count])
        result['sentinel'].extend(chosen[count:])
        result['canary'].extend(chosen[:canary])
    return result


def audit(base, config):
    for name, expected in ANCHORS.items():
        if provenance.sha(base / name) != expected:
            raise ValueError('input changed: ' + name)
    raw, checked = provenance.selected_train(base)
    targets = json.loads((base / 'NG-0069/teacher-v1/targets.json').read_text())
    history = json.loads((base / 'NG-0070/provenance-v1/pair-provenance.json').read_text())
    source = {r['query_id']: r for r in history}
    if len(source) != len(history) or set(source) != {r['query_id'] for r in raw}:
        raise ValueError('provenance identities changed')
    wanted = {doc for row in targets for doc in row['pool']}
    documents = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                documents[index] = json.loads(line)
    if index + 1 != config['pilot']['corpus_documents'] or set(documents) != wanted:
        raise ValueError('corpus inventory changed')
    resolved_only = config['pilot']['selection'] == 'resolved_provenance_then_TRAIN_stable_hash_v2'
    if not resolved_only:
        raise ValueError('scientific selection requires the v2 provenance contract')
    excluded = [r['query_id'] for r in raw if source[r['query_id']]['status'] != 'resolved']
    excluded_set = set(excluded)
    selection = select([r for r in raw if r['query_id'] not in excluded_set], config)
    records, unicode_aliases = [], []
    for row, target in zip(raw, targets):
        if row['query_id'] != target['query_id']:
            raise ValueError('query order changed')
        candidates = {r['key']: r for r in row['candidates']}
        if len(candidates) != len(row['candidates']):
            raise ValueError('duplicate original candidate')
        pool = target['pool']
        if len(set(pool)) != len(pool) or len(pool) != len(candidates):
            raise ValueError('original pool changed')
        if len(pool) + config['witness']['max_new_documents'] > config['witness']['max_total_documents']:
            raise ValueError('potential witness overflow; never truncate positives')
        actual_positive_keys = set()
        for doc, positive in zip(pool, target['positive_mask'], strict=True):
            item = documents[doc]
            original = candidates[item['key']]
            if item['text'] != original['text']:
                if not same_source_text(item['text'], original['text']):
                    raise ValueError('document text changed beyond Unicode equivalence')
                unicode_aliases.append({
                    'query_id': row['query_id'], 'document_id': doc,
                    'label': original['label'],
                    'corpus_sha256': hashlib.sha256(item['text'].encode()).hexdigest(),
                    'source_sha256': hashlib.sha256(original['text'].encode()).hexdigest(),
                    'actual_model_input': 'unchanged_frozen_corpus_text',
                })
            if type(positive) is not bool or positive != (original['label'] == 'positive'):
                raise ValueError('positive mask changed')
            if positive:
                actual_positive_keys.add(item['key'])
        lineage = source[row['query_id']]
        if {p['document_key'] for p in lineage['pairs']} != actual_positive_keys:
            raise ValueError('positive provenance coverage changed')
        for positive in lineage['pairs']:
            text = candidates[positive['document_key']]['text']
            if hashlib.sha256(text.encode()).hexdigest() != positive['text_sha256']:
                raise ValueError('positive provenance text changed')
        positives = np.asarray(target['positive_mask'], dtype=bool)
        teacher = np.asarray(target['scores'], dtype=np.float64)
        if not positives.any() or teacher.shape != positives.shape or not np.isfinite(teacher).all():
            raise ValueError('invalid teacher or labels')
        margins = (teacher[positives, None] - teacher[None, ~positives])
        confidence = np.tanh(margins / (2 * config['ranking']['teacher_temperature']))
        eligible = confidence > 0
        records.append({
            'query_id': row['query_id'], 'domain': row['subset'],
            'pool': len(pool), 'positive_count': int(positives.sum()),
            'supervised_positives': int(eligible.any(axis=1).sum()),
            'eligible_pairs': int(eligible.sum()),
            'unresolved_pairs': int((~eligible).sum()),
            'confidence_sum': float(confidence[eligible].sum()),
            'near_tie_eligible_pairs': int((eligible & (confidence < .1)).sum()),
            'target_above_099_pairs': int((confidence > .98).sum()),
            'provenance_status': lineage['status'],
            'positive_origins': dict(Counter(p['provenance'] for p in lineage['pairs'])),
        })
    summaries = {}
    for surface, identities in {'all_train': [r['query_id'] for r in raw], **selection}.items():
        selected = set(identities)
        summaries[surface] = {}
        for domain in config['pilot']['domains']:
            subset = [r for r in records if r['query_id'] in selected and r['domain'] == domain]
            totals = {k: sum(r[k] for r in subset) for k in (
                'positive_count', 'supervised_positives', 'eligible_pairs',
                'unresolved_pairs', 'confidence_sum', 'near_tie_eligible_pairs',
                'target_above_099_pairs')}
            totals.update(
                queries=len(subset), max_pool=max(r['pool'] for r in subset),
                no_supervision_queries=sum(r['eligible_pairs'] == 0 for r in subset),
                unresolved_provenance_queries=sum(r['provenance_status'] != 'resolved' for r in subset),
                positive_origins=dict(sum((Counter(r['positive_origins']) for r in subset), Counter())),
                eligible_positive_fraction=totals['supervised_positives'] / totals['positive_count'],
                mean_eligible_pair_confidence=totals['confidence_sum'] / max(totals['eligible_pairs'], 1),
                near_tie_eligible_pair_fraction=totals['near_tie_eligible_pairs'] / max(totals['eligible_pairs'], 1),
                target_above_099_fraction=totals['target_above_099_pairs'] / max(totals['eligible_pairs'], 1))
            summaries[surface][domain] = totals
    safe = all(
        s['eligible_positive_fraction'] >= config['gates']['eligible_positive_fraction_min_per_domain']
        and s['unresolved_provenance_queries'] == 0
        for surface in ('pilot', 'sentinel', 'canary') for s in summaries[surface].values())
    lookup = {r['query_id']: r for r in records}
    rng = np.random.default_rng(config['pilot']['query_order_seed'])
    order = [str(q) for _ in range(config['pilot']['epochs'])
             for q in rng.permutation(selection['pilot'])]
    batch_size = config['optimizer']['queries_per_update']
    empty_batches = [start // batch_size for start in range(0, len(order), batch_size)
                     if not any(lookup[q]['eligible_pairs'] for q in order[start:start + batch_size])]
    safe = safe and not empty_batches
    return {'selection': selection, 'query_order': order,
            'excluded_unresolved_provenance_queries': excluded,
            'all_unsupervised_update_batches': empty_batches,
            'records': records, 'inputs': {**checked, **ANCHORS},
            'summaries': summaries, 'eligibility_gate_passed': safe,
            'unicode_equivalent_source_aliases': unicode_aliases,
            'model_inference': False, 'locked_test_access': False,
            'global_witness_audit_passed': False, 'training_authorized_by_audit': False}
