"""Independent TRAIN selection, lineage and scalar-loss review of NG79.

No generator selection/loss helper, Torch, encoder, or corpus inference is used.
Full-corpus rank certification remains the sealed dual-score preparation proof.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import unicodedata

import numpy as np

from review_ng75_displacement import compare, read, require, sha, verify as verify_files
from review_ng77_preparation import probability, weight


INPUT = 'f6e0430ea6de939f48c76c153f2f45362d4b7a07761a33a1550e2a647d2fe87c'
INVENTORY = '8cc056c9c5d8f70dd045b1e191292d53abb05e0c31db2c3dbecaf9dd083c49e8'
DOMAINS = ('fever', 'hotpotqa', 'nq')


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def audit_selection(selected, queries, lineage, old, prefix, *, per_domain=128):
    lookup = {q['query_id']: q for q in queries}
    require(len(lookup) == len(queries), 'duplicate query identity')
    require(selected['repeated'] == old['pilot'] and selected['sentinel'] == old['sentinel'],
            'original cohort changed')
    protected = old['pilot'] + old['sentinel']
    require(len(protected) == len(set(protected)) == 6 * per_domain, 'historical identities overlap')
    used = {normalized(lookup[q]['query']) for q in protected}
    require(len(used) == len(protected), 'historical texts overlap')
    require(all(lookup[q]['split'] == 'TRAIN' and lineage[q]['status'] == 'resolved'
                for q in protected), 'invalid historical TRAIN source')
    require(len(prefix) == len(set(prefix)) == 3 * per_domain and set(prefix) == set(old['pilot']),
            'prefix identity changed')
    added = {}
    for d in DOMAINS:
        ranks = {}
        for row in queries:
            q = row['query_id']
            if row['subset'] == d and row['split'] == 'TRAIN' and lineage[q]['status'] == 'resolved':
                ranks[q] = hashlib.sha256(('NG-0079/breadth-v1/' + q).encode()).hexdigest()
        chosen = []
        for q in sorted(ranks, key=lambda q: (ranks[q], q)):
            text = normalized(lookup[q]['query'])
            if text not in used:
                chosen.append(q)
                used.add(text)
            if len(chosen) == 3 * per_domain:
                break
        require(len(chosen) == 3 * per_domain, 'insufficient eligible TRAIN queries')
        added[d] = chosen
    require(selected['added'] == added
            and selected['breadth'] == old['pilot'] + sum(added.values(), []),
            'stable-hash breadth selection differs')
    rng = np.random.default_rng(79001)
    r_order = list(prefix)
    b_order = list(prefix)
    for quarter in range(3):
        exposure = list(map(str, rng.permutation(old['pilot'])))
        r_order.extend(exposure)
        domain_offset = Counter()
        for q in exposure:
            d = lookup[q]['subset']
            b_order.append(added[d][quarter * per_domain + domain_offset[d]])
            domain_offset[d] += 1
    require(selected['orders'] == {'R': r_order, 'B': b_order}, 'seed or domain-matched order differs')
    require(Counter(r_order) == Counter({q: 4 for q in old['pilot']})
            and len(b_order) == len(set(b_order)) == 12 * per_domain, 'exposure count differs')
    require(not set(b_order).intersection(old['sentinel']), 'sentinel entered gradients')
    for surface, count in (('repeated', per_domain), ('breadth', per_domain * 4), ('sentinel', per_domain)):
        require(Counter(lookup[q]['subset'] for q in selected[surface])
                == Counter({d: count for d in DOMAINS}), 'domain balance differs')


def scalar_objective(record, config):
    positive = [i for i, yes in enumerate(record['positive_mask']) if yes]
    require(len(positive) == record['total_positives'] > 0, 'all positives required')
    require(not any(record['judged_negative_mask']), 'source negatives are not judged')
    ranks, scores, teacher = (record[k] for k in ('global_ranks', 'hybrid_scores', 'teacher_scores'))
    require(len(set(ranks)) == len(ranks) and all(type(r) is int and 1 <= r <= 233009 for r in ranks),
            'invalid full-corpus ranks')
    gradient, terms, pairs, targets, coefficients = [0.] * len(scores), [], [], [], []
    supervised = 0
    for p in positive:
        pending = []
        for n in range(len(scores)):
            if n in positive:
                continue
            target = probability((teacher[p] - teacher[n]) / config['teacher_temperature'])
            if target <= .5:
                continue
            w = weight(ranks[p], ranks[n], len(positive), config)
            pending.append((n, target, w))
        supervised += bool(pending)
        denominator = math.fsum(w for _, _, w in pending)
        for n, target, w in pending:
            coefficient = (2 * target - 1) * w / denominator / len(positive)
            z = (scores[p] - scores[n]) / config['student_temperature']
            terms.append(coefficient * (max(z, 0.) + math.log1p(math.exp(-abs(z))) - target * z))
            derivative = coefficient * (probability(z) - target) / config['student_temperature']
            gradient[p] += derivative
            gradient[n] -= derivative
            pairs.append([p, n])
            targets.append(target)
            coefficients.append(coefficient)
    return dict(loss=math.fsum(terms), score_gradient=gradient, eligible_pairs=len(pairs),
                supervised_positives=supervised, pair_indices=pairs,
                pair_targets=targets, pair_coefficients=coefficients)


def review(base, run):
    started = time.monotonic()
    inventory_path = run.with_name(run.name + '-remote-inventory.json')
    require(sha(run / 'inputs.json') == INPUT and sha(inventory_path) == INVENTORY, 'wrong frozen unit')
    inventory, manifest = read(inventory_path), read(run / 'inputs.json')
    verify_files(run, inventory['files'], exact=True)
    verify_files(base, manifest['dependencies'])
    verify_files(run, manifest['source'])
    phase = run / 'witnesses'
    complete, exited, tracking = (read(phase / n) for n in ('complete.json', 'exit.json', 'clearml.json'))
    require(complete['passed'] and exited['exit_code'] == 0 and exited['error'] is None
            and exited['owned_group_closed'] and exited['elapsed_seconds'] <= 5400
            and exited['peak_tree_rss_bytes'] <= 16 * 2 ** 30, 'phase is not safely closed')
    verify_files(phase, complete['files'], exact=True, exclude=('complete.json',))
    require(tracking['actual_start'] and tracking['closed'] and tracking['outcome'] == 'passed'
            and read(run / 'controller-exit.json')['status'] == 'all_phases_complete', 'closure missing')
    selected = read(run / 'selection-audit.json')['selection']
    queries = read(base / 'NG-0069/lexical-v1/queries.json')[:6144]
    qmap = {q['query_id']: q for q in queries}
    old = read(base / 'NG-0071/pilot-v1/selection.json')
    prefix = read(base / 'NG-0071/pilot-step0-v1/training-order.json')[:384]
    lineage = {r['query_id']: r for r in read(base / 'NG-0070/provenance-v1/pair-provenance.json')}
    audit_selection(selected, queries, lineage, old, prefix)
    labels = read(base / 'NG-0069/lexical-v1/labels.json')
    targets = {r['query_id']: r for r in read(base / 'NG-0069/teacher-v1/targets.json')}
    position = {q['query_id']: i for i, q in enumerate(queries)}
    with (phase / 'witnesses.jsonl').open() as stream:
        identities, wanted = [], set()
        for line in stream:
            r = json.loads(line)
            identities.append(r['query_id'])
            wanted.update(r['pool'])
    require(identities == selected['breadth'], 'witness order or coverage changed')
    docs = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                d = json.loads(line)
                docs[i] = (d['key'], hashlib.sha256(d['text'].encode()).hexdigest())
    require(i + 1 == 233009 and set(docs) == wanted, 'corpus inventory differs')
    canonical_sha = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    historical = {}
    with (base / 'NG-0071/pilot-step0-v1/witnesses.jsonl').open() as stream:
        for r in map(json.loads, stream):
            r['arm_score_diagnostics'] = {'D': r['arm_score_diagnostics']['D']}
            historical[r['query_id']] = canonical_sha(r)
    config = read(run / 'config.json')
    summaries = {d: Counter() for d in DOMAINS}
    errors = []
    eligible = {}
    def records():
        with (phase / 'witnesses.jsonl').open() as stream:
            yield from map(json.loads, stream)

    for r in records():
        q, pool = r['query_id'], r['pool']
        require(r['split'] == 'TRAIN' and qmap[q]['split'] == 'TRAIN' and r['reference_update'] == 0
                and r['checkpoint_state_sha256'] == config['base']['state_sha256'], 'TRAIN/reference changed')
        require(r['query_sha256'] == hashlib.sha256(qmap[q]['query'].encode()).hexdigest(), 'query text changed')
        require(r['positive_ids'] == labels[position[q]] and r['original_pool'] == targets[q]['pool']
                and r['total_positives'] == len(r['positive_ids']), 'all-positive/original pool mismatch')
        require(len(pool) == len(set(pool)) <= 256 and pool[:len(r['original_pool'])] == r['original_pool'],
                'original candidates missing or duplicate')
        require(r['positive_mask'] == [d in r['positive_ids'] for d in pool], 'positive mask changed')
        require(r['new_witness_count'] <= 64 and r['new_witness_count'] == len(pool) - len(r['original_pool']),
                'witness budget differs')
        attributed = sum(r['sources'].values(), [])
        require(attributed == pool[len(r['original_pool']):], 'witness source attribution differs')
        require(r['document_text_sha256'] == [docs[d][1] for d in pool], 'document text changed')
        origins = {p['document_key']: p['provenance'] for p in lineage[q]['pairs']}
        require(set(origins) == {docs[d][0] for d in r['positive_ids']}, 'positive lineage differs')
        compare(r['teacher_scores'][:len(r['original_pool'])], targets[q]['scores'])
        audit = scalar_objective(r, config['ranking'])
        errors.append(compare({k: r['arm_score_diagnostics']['D'][k] for k in audit}, audit))
        eligible[q] = audit['eligible_pairs']
        totals = summaries[r['domain']]
        totals.update(queries=1, positives=r['total_positives'],
                      supervised_positives=audit['supervised_positives'], eligible_pairs=audit['eligible_pairs'],
                      candidate_documents=len(pool), no_supervision_queries=int(not audit['eligible_pairs']))
        totals.update(Counter(origins.values()))
        if q in historical:
            require(canonical_sha(r) == historical[q], 'historical witness differs')
    for d, totals in summaries.items():
        require(totals['queries'] == 512 and totals['supervised_positives'] / totals['positives'] >= .8,
                'domain eligibility failed')
    for arm, order in selected['orders'].items():
        require(not any(not any(eligible[q] for q in order[i:i + 4]) for i in range(0,1536,4)),
                'unsupervised update in ' + arm)
    result = read(phase / 'results.json')
    require(result['passed'] and result['eligibility_gate_passed']
            and result['historical_384_witness_max_error'] == 0.
            and result['max_independent_score_error'] <= 1e-12
            and result['actual_optimizer_updates'] == 0, 'invalid preparation result')
    return dict(passed=True, input_sha256=INPUT, inventory_sha256=INVENTORY,
                reviewer_sha256=sha(Path(__file__)), summaries={d: dict(v) for d, v in summaries.items()},
                queries=1536, positives=sum(v['positives'] for v in summaries.values()),
                max_independent_scalar_error=max(errors), historical_384_exact=True,
                independent_full_corpus_rerank=False, new_inference=False, dev_access=False,
                locked_test_access=False, native_cost_evaluated=False, seconds=time.monotonic() - started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.research_root, args.run)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))
