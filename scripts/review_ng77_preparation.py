"""Independent scalar audit of frozen NG77 anchors and head coverage.

NumPy is used only to memory-map existing teacher vectors. No generator math,
Torch, model loading, new model inference or optimizer execution is imported.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root, inventory):
    for name, digest in inventory.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and sha(path) == digest,
                'frozen content differs: ' + name)


def close(actual, expected):
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= 1e-12, 'scalar arithmetic differs')


def probability(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def weight(p, n, count, config):
    k, cutoff = config['ndcg_cutoff'], config['recall_cutoff']
    ideal = math.fsum(1 / math.log2(i + 1) for i in range(1, min(k, count) + 1))
    dp, dn = (1 / math.log2(p + 1) if p <= k else 0), (1 / math.log2(n + 1) if n <= k else 0)
    return (abs(dp - dn) / ideal + config['recall_weight'] * abs((p <= cutoff) - (n <= cutoff)) / count
            + config['pair_floor'] / count)


def review(base, run, procedure, expected_manifest, expected_procedure):
    require(sha(run / 'inputs.json') == expected_manifest
            and sha(procedure / 'complete.json') == expected_procedure, 'wrong input identity')
    receipt, frozen = read(procedure / 'complete.json'), read(run / 'inputs.json')
    require(receipt['passed'] and frozen['training_enabled'] is False, 'preparation gate failed')
    verify(procedure, receipt['files'])
    actual = {str(p.relative_to(procedure)) for p in procedure.rglob('*')
              if p.is_file() and p.name != 'complete.json'}
    require(actual == set(receipt['files']), 'procedure inventory changed')
    inventory = read(procedure / 'run-inventory.json')['files']
    verify(run, inventory)
    require({str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()} == set(inventory),
            'run inventory changed')
    verify(base, frozen['dependencies'])
    verify(run, frozen['source'])
    exit_record, tracking = read(procedure / 'exit.json'), read(procedure / 'clearml.json')
    require(exit_record['exit_code'] == 0 and exit_record['error'] is None
            and exit_record['owned_group_closed'] and exit_record['elapsed_seconds'] < 900
            and exit_record['peak_tree_rss_bytes'] <= 8 * 2 ** 30, 'unclosed or out-of-bounds CPU job')
    require(tracking['actual_start'] and tracking['closed'] and tracking['outcome'] == 'passed',
            'tracking not closed')
    config = read(base / 'NG-0071/pilot-step0-v1/config.json')['ranking']
    require(config['student_temperature'] == 1., 'synthetic scale audit assumes frozen temperature 1')
    records = {r['query_id']: r for line in
               (base / 'NG-0071/pilot-step0-v1/witnesses.jsonl').read_text().splitlines()
               if (r := json.loads(line))}
    selection = read(base / 'NG-0071/pilot-v1/selection.json')
    require(set(records) == set(selection['pilot']) and not set(records) & set(selection['sentinel']),
            'TRAIN pilot boundary differs')
    heads = {}
    with (base / 'NG-0069/evaluation-v2/rank-initial/rankings.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            if r['query_id'] in records:
                require(r['split'] == 'TRAIN' and r['query_id'] not in heads, 'invalid saved TRAIN head')
                heads[r['query_id']] = r
    qs = read(base / 'NG-0069/lexical-v1/queries.json')
    positions = {q['query_id']: i for i, q in enumerate(qs)}
    dense = base / 'NG-0069/evaluation-v2/encode-dense'
    dq = np.load(dense / 'query.npy', mmap_mode='r')
    dd = np.load(dense / 'document.npy', mmap_mode='r')
    anchors = read(run / 'audit/anchors.json')
    per_query = {r['query_id']: r for r in read(run / 'audit/per-query.json')}
    require(set(anchors) == set(per_query) == set(records) == set(heads) and len(records) == 384,
            'missing or additional query')
    recorded_pairs = defaultdict(dict)
    with (run / 'audit/pair-audit.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            key = (r['positive_id'], r['rival_id'])
            require(key not in recorded_pairs[r['query_id']], 'duplicate recorded pair')
            recorded_pairs[r['query_id']][key] = r
    require(set(recorded_pairs) == set(records), 'unknown query in pair audit')
    counts, domain_positives, domain_anchors, shrinking = Counter(), Counter(), Counter(), Counter()
    coverage = defaultdict(lambda: defaultdict(lambda: Counter()))
    max_teacher_error = 0.
    for q, r in records.items():
        require(r['split'] == 'TRAIN' and positions[q] < 6144 and qs[positions[q]]['split'] == 'TRAIN',
                'non-TRAIN source')
        head, pool, gold = heads[q], r['pool'], r['positive_ids']
        rank = dict(zip(pool, r['global_ranks'], strict=True))
        score = dict(zip(pool, r['hybrid_scores'], strict=True))
        teacher = dict(zip(pool, r['teacher_scores'], strict=True))
        for index, (d, s) in enumerate(zip(head['top100'], head['top100_scores'], strict=True), 1):
            if d in pool:
                close(score[d], s)
                require(rank[d] == index, 'saved head rank differs')
            rank[d], score[d] = index, s
        require(dict(zip(head['gold_ids'], head['gold_ranks'], strict=True)) == {p: rank[p] for p in gold},
                'full positive ranks differ')
        ids = sorted(score)
        vector = dq[positions[q]].tolist()
        for d in ids:
            value = math.fsum(a * b for a, b in zip(vector, dd[d].tolist(), strict=True))
            if d in teacher:
                max_teacher_error = max(max_teacher_error, abs(value - teacher[d]))
                close(value, teacher[d])
            else:
                teacher[d] = value
        expected_keys = {(p, n) for p in gold for n in ids if n not in gold}
        require(set(recorded_pairs[q]) == expected_keys, 'pair universe changed')
        expected, per_positive = {}, defaultdict(list)
        for p, n in sorted(expected_keys):
            row = recorded_pairs[q][p, n]
            in_pool = n in pool
            judged = r['judged_negative_mask'][pool.index(n)] if in_pool else False
            margin = score[p] - score[n]
            gap = (teacher[p] - teacher[n]) / config['teacher_temperature']
            status = 'judged_negative' if judged else 'agrees' if gap > 0 else 'opposes' if gap < 0 else 'tie'
            confidence = 1. if judged else max(0., 2 * probability(gap) - 1)
            w = weight(rank[p], rank[n], len(gold), config)
            trusted = margin > 0 and confidence > 0 and w > 0
            target = 1. if judged else (confidence + 1) / 2
            shrinks = trusted and in_pool and probability(margin) > target
            require(row['teacher_status'] == status and row['trusted'] == trusted
                    and row['in_D_pool'] == in_pool and row['used_as_anchor'] == (trusted and in_pool)
                    and row['D_direct_pressure_shrinks_anchor'] == shrinks,
                    'trust, pool or anchor label differs')
            shrinking[r['domain']] += int(shrinks)
            for actual, value in ((row['confidence'], confidence), (row['metric_weight'], w),
                                  (row['baseline_margin'], margin)):
                close(actual, value)
            b = '1-10' if rank[n] <= 10 else '11-100' if rank[n] <= 100 else 'outside100'
            require(row['rival_band'] == b, 'rank band differs')
            coverage[r['domain']][b]['trusted'] += int(trusted)
            coverage[r['domain']][b]['trusted_in_pool'] += int(trusted and in_pool)
            if trusted and in_pool:
                item = dict(indices=[pool.index(p), pool.index(n)], margin=margin,
                            confidence=confidence, weight=w)
                expected[p, n] = item
                per_positive[p].append(item)
        observed = {(a['positive_id'], a['rival_id']): a for a in anchors[q]['anchors']}
        require(len(observed) == len(anchors[q]['anchors']) and set(observed) == set(expected),
                'exact anchor identity differs')
        for key, item in expected.items():
            denominator = math.fsum(v['weight'] for v in per_positive[key[0]])
            coefficient = item['confidence'] * item['weight'] / denominator / len(gold)
            item['coefficient'] = coefficient
            a = observed[key]
            require(a['indices'] == item['indices'], 'forward indices differ')
            for actual, value in ((a['baseline_margin'], item['margin']), (a['coefficient'], coefficient),
                                  (a['confidence'], item['confidence']), (a['metric_weight'], item['weight'])):
                close(actual, value)
        require(anchors[q]['total_positives'] == per_query[q]['total_positives'] == len(gold)
                and anchors[q]['covered_positives'] == per_query[q]['covered_positives'] == len(per_positive)
                and anchors[q]['candidate_count'] == per_query[q]['forward_documents'] == len(pool),
                'all-positive or forward denominator changed')
        for drop in (.1, .5, 1.):
            losses, gradients = [], []
            for item in expected.values():
                a0, a = item['margin'], item['margin'] - drop
                def softplus(x):
                    return max(x, 0) + math.log1p(math.exp(-abs(x)))
                losses.append(item['coefficient'] * (softplus(-a) - softplus(-a0) - probability(-a0) * drop))
                gradients.append(2 * item['coefficient'] * (probability(-a) - probability(-a0)))
            recorded = per_query[q]['synthetic_positive_margin_drop'][str(drop)]
            close(recorded['loss'], math.fsum(losses))
            close(recorded['score_gradient_l1'], math.fsum(gradients))
        counts['queries'] += 1
        counts['anchors'] += len(expected)
        counts['audited_pairs'] += len(expected_keys)
        domain_positives[r['domain']] += len(gold)
        domain_anchors[r['domain']] += len(expected)
    result = read(run / 'audit/results.json')
    require(all(result[k] == v for k, v in counts.items()), 'summary count differs')
    for domain, bands in coverage.items():
        for b, counters in bands.items():
            for k, count in counters.items():
                require(result['domains'][domain]['rivals'][b][k] == count, 'coverage reduction differs')
        require(result['domains'][domain]['positives'] == domain_positives[domain]
                and result['domains'][domain]['anchors'] == domain_anchors[domain]
                and result['domains'][domain]['D_direct_pressure_shrinks_anchor'] == shrinking[domain],
                'domain totals differ')
    return dict(passed=True, **counts, all_positive_relations=sum(domain_positives.values()),
        all_anchor_identities_margins_confidence_weights_coefficients_verified=True,
        cached_teacher_fsum_checked=True, max_in_pool_teacher_error=max_teacher_error,
        synthetic_score_space_loss_and_gradient_checked=True,
        full_head_coverage_verified=True, no_new_model_inference=True,
        generator_math_imported=False, training_enabled=False,
        independent_heldout_quality=False, overall_goal_qualified=False,
        manifest_sha256=expected_manifest, procedure_sha256=expected_procedure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('research-root', 'run', 'procedure', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--procedure-sha', required=True)
    args = parser.parse_args()
    result = review(args.research_root, args.run, args.procedure, args.manifest_sha, args.procedure_sha)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
