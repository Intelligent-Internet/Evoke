"""Audit original TRAIN pools and trusted initial margins without new inference.

Outside-pool top100 competitors are audited with cached PPLX vectors only;
they do not silently become training examples. No sentinel/DEV outcomes select
anchors. A successful preparation receipt does not authorize model training.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

import ng71_ranking as ranking
import ng77_retention as retention
import review_ng75_displacement as io


PARENT = 'NG-0071/pilot-step0-v1'
PILOT = 'NG-0071/pilot-v1'
PARENT_SHA = '5a736aedeff4270bec4e070640725d67236db52cdd7f82fc7684dcdc7ae5fc96'
PILOT_SHA = '5c44c86e13d688972100f1eb1a85b562ba8a631bdf31a2b2e4073ef9b304fb2a'
REVIEW_SHA = 'c3b847295d6f32596c5863e122c6ee2691bd82d6c0f40f6eaf631d8cee445629'
SOURCES = ('prepare_ng77_retention.py', 'ng77_retention.py',
           'ng71_ranking.py', 'review_ng75_displacement.py')
PLAN = 'ng0077-trusted-margin-retention-plan.zh.md'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def freeze(base, run):
    parent = base / PARENT
    io.require(io.sha(parent / 'complete.json') == PARENT_SHA
               and io.sha(base / PILOT / 'inputs.json') == PILOT_SHA
               and io.sha(base / PILOT / 'step0-review.json') == REVIEW_SHA,
               'original initial witness or selection proof changed')
    receipt = io.read(parent / 'complete.json')
    io.require(receipt['passed'], 'original snapshot failed')
    io.verify(parent, receipt['files'], exact=True, exclude=('complete.json',))
    dependencies = dict(io.read(parent / 'dependencies.json'))
    dependencies.update({f'{PARENT}/{name}': digest for name, digest in receipt['files'].items()})
    dependencies[f'{PARENT}/complete.json'] = PARENT_SHA
    pilot = io.read(base / PILOT / 'inputs.json')
    for name in ('selection.json', 'training-order.json', 'step0-review.json'):
        dependencies[f'{PILOT}/{name}'] = pilot['source'][name]
    dependencies[f'{PILOT}/inputs.json'] = PILOT_SHA
    io.verify(base, dependencies)
    run.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    for name in SOURCES:
        shutil.copy2(source / name, run / name)
    shutil.copy2(source.parent / 'tests/test_ng77_retention.py', run / 'test_ng77_retention.py')
    shutil.copy2(source.parent / 'tests/test_ng77_preparation.py', run / 'test_ng77_preparation.py')
    shutil.copy2(source.parent / 'docs/research-sae/reports/ng0001-ng0099' / PLAN, run / PLAN)
    # The loss fixture checks parity with the unchanged original D objective.
    shutil.copy2(source / 'ng71_training.py', run / 'ng71_training.py')
    write(run / 'inputs.json', dict(protocol='NG77_trusted_margin_CPU_preparation_v1',
        dependencies=dependencies, source={p.name: io.sha(p) for p in run.iterdir() if p.is_file()},
        training_enabled=False, new_model_inference=False, gpu_used=False,
        locked_test_access=False, sentinel_or_dev_selection=False,
        forward_pool='unchanged_original_D_step0', outside_pool='audit_only_not_anchors',
        student_temperature=1., retention_coefficient=None,
        sensitivity_margin_drops=[.1, .5, 1.], sensitivity_not_real_training=True,
        wall_seconds=900, tree_rss_gib=8, host_available_gib=12, disk_available_gib=40))
    return dict(passed=True, manifest_sha256=io.sha(run / 'inputs.json'),
                dependency_files=len(dependencies), training_enabled=False)


def selected_rankings(path, identities):
    selected = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['query_id'] in identities:
                io.require(row['split'] == 'TRAIN' and row['query_id'] not in selected,
                           'non-TRAIN or duplicate initial ranking')
                selected[row['query_id']] = row
    io.require(set(selected) == set(identities), 'missing original TRAIN ranking')
    return selected


def validate_cohort(records, selection, order, queries):
    ids = [r['query_id'] for r in records]
    io.require(len(ids) == len(set(ids)) == 384 and set(ids) == set(selection['pilot']),
               'original 384-query pilot changed')
    io.require(not set(ids) & set(selection['sentinel']), 'sentinel anchor contamination')
    io.require(len(order) == 768 and len(set(order[:384])) == 384
               and set(order[:384]) == set(ids), 'original first-epoch order changed')
    io.require(Counter(r['domain'] for r in records) == dict.fromkeys(('fever', 'hotpotqa', 'nq'), 128),
               'domain cohort changed')
    for r in records:
        q = queries[r['query_id']]
        io.require(r['split'] == q['split'] == 'TRAIN' and r['domain'] == q['subset']
                   and r['reference_update'] == 0
                   and r['independent_score_and_rank_verification_passed'] is True,
                   'unverified or wrong initial TRAIN reference')


def band(rank):
    return '1-10' if rank <= 10 else '11-100' if rank <= 100 else 'outside100'


def query_audit(record, head, outside_teacher, config):
    io.require(head['query_id'] == record['query_id'] and head['split'] == 'TRAIN'
               and head['domain'] == record['domain'], 'ranking identity mismatch')
    pool, positives = record['pool'], record['positive_ids']
    io.require(set(head['gold_ids']) == set(positives)
               and len(head['gold_ids']) == len(positives), 'full positive universe changed')
    io.require(len(head['top100']) == len(set(head['top100'])) == 100
               and len(head['top100_scores']) == 100, 'incomplete full-corpus head')
    locations = {d: i for i, d in enumerate(pool)}
    scores = dict(zip(pool, record['hybrid_scores'], strict=True))
    ranks = dict(zip(pool, record['global_ranks'], strict=True))
    teacher = dict(zip(pool, record['teacher_scores'], strict=True))
    teacher.update(outside_teacher)
    io.require(not set(pool) & set(outside_teacher), 'outside teacher overwrote old scores')
    for d, rank in zip(head['gold_ids'], head['gold_ranks'], strict=True):
        io.require(ranks[d] == rank, 'initial full positive rank mismatch')
    for rank, (d, score) in enumerate(zip(head['top100'], head['top100_scores'], strict=True), 1):
        if d in locations:
            io.require(ranks[d] == rank and abs(scores[d] - score) <= 1e-12,
                       'saved initial head differs from witness')
        scores[d], ranks[d] = score, rank
    anchors = retention.prepare_anchors(record, config)
    identities = {(r['positive_id'], r['rival_id']) for r in anchors['anchors']}
    origins = {r['document_id']: r for r in record['positive_visibility']}
    io.require(set(origins) == set(positives), 'positive provenance/visibility missing')
    rows = []
    for p in positives:
        for n in sorted(set(pool) | set(head['top100'])):
            if n in positives:
                continue
            in_pool = n in locations
            judged = record['judged_negative_mask'][locations[n]] if in_pool else False
            status, confidence = retention.preference(teacher[p], teacher.get(n), judged,
                                                      config['teacher_temperature'])
            margin = scores[p] - scores[n]
            weight = ranking.pair_weight(ranks[p], ranks[n], len(positives), **{
                k: config[k] for k in ('ndcg_cutoff', 'recall_cutoff', 'recall_weight', 'pair_floor')})
            trusted = margin > 0 and confidence > 0 and weight > 0
            io.require(((p, n) in identities) == (trusted and in_pool), 'anchor audit disagrees')
            target = 1. if judged else (confidence + 1) / 2
            derivative = float(np.exp(-np.logaddexp(0., -margin / config['student_temperature']))) - target
            rows.append(dict(query_id=record['query_id'], domain=record['domain'],
                positive_id=p, rival_id=n, positive_rank=ranks[p], rival_rank=ranks[n],
                positive_band=band(ranks[p]), rival_band=band(ranks[n]),
                positive_origin=origins[p]['origin'], positive_truncated=origins[p]['truncated'],
                evidence_visibility=origins[p]['supporting_evidence_visible'],
                baseline_margin=margin, baseline_positive=margin > 0,
                teacher_status=status, confidence=confidence, metric_weight=weight,
                trusted=trusted, in_D_pool=in_pool,
                in_original_pool=p in record['original_pool'] and n in record['original_pool'],
                original_D_eligible=in_pool and confidence > 0 and weight > 0,
                D_direct_pressure_shrinks_anchor=trusted and in_pool and derivative > 0,
                used_as_anchor=(p, n) in identities,
                crosses_top100=(ranks[p] <= 100) != (ranks[n] <= 100)))
    base = ranking.loss_and_gradient(np.array(record['hybrid_scores']),
        np.array(record['positive_mask']), np.array(record['global_ranks']),
        np.array(record['teacher_scores']), np.array(record['judged_negative_mask']),
        total_positives=len(positives), corpus_size=record['corpus_size'],
        rank_scope=record['rank_scope'], **{k: config[k] for k in (
            'student_temperature', 'teacher_temperature', 'ndcg_cutoff',
            'recall_cutoff', 'recall_weight', 'pair_floor')})
    sensitivity = {}
    for drop in (.1, .5, 1.):
        values = np.array(record['hybrid_scores']) - drop * np.array(record['positive_mask'])
        scores_tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        loss = retention.retention_loss(scores_tensor, anchors)
        loss.backward()
        sensitivity[str(drop)] = dict(loss=float(loss.detach()),
            score_gradient_l1=float(scores_tensor.grad.abs().sum()))
    return anchors, rows, dict(query_id=record['query_id'], domain=record['domain'],
        total_positives=len(positives), covered_positives=anchors['covered_positives'],
        forward_documents=len(pool), anchors=len(identities),
        baseline_D_loss=base['loss'], baseline_D_score_gradient_l1=float(abs(base['gradient']).sum()),
        synthetic_positive_margin_drop=sensitivity, positive_visibility=record['positive_visibility'])


def summarize(pairs, queries):
    result = {}
    for domain in ('fever', 'hotpotqa', 'nq'):
        ps = [r for r in pairs if r['domain'] == domain]
        qs = [r for r in queries if r['domain'] == domain]
        covered = [r for r in ps if r['used_as_anchor']]
        metrics = {}
        for b in ('1-10', '11-100', 'outside100'):
            group = [r for r in ps if r['rival_band'] == b]
            trusted = [r for r in group if r['trusted']]
            inside = [r for r in trusted if r['in_D_pool']]
            metrics[b] = dict(audited_pairs=len(group), trusted=len(trusted),
                trusted_in_pool=len(inside), trusted_missing_pool=len(trusted) - len(inside),
                trusted_pair_coverage=len(inside) / len(trusted) if trusted else None,
                trusted_metric_weight_coverage=(sum(r['metric_weight'] for r in inside)
                    / sum(r['metric_weight'] for r in trusted)) if trusted else None,
                teacher_states=dict(Counter(r['teacher_status'] for r in group)))
        result[domain] = dict(queries=len(qs), positives=sum(r['total_positives'] for r in qs),
            covered_positives=sum(r['covered_positives'] for r in qs),
            queries_without_anchors=sum(r['anchors'] == 0 for r in qs),
            anchors=len(covered), original_pool_anchors=sum(r['in_original_pool'] for r in covered),
            old_D_seen_eligible_anchors=sum(r['original_D_eligible'] for r in covered),
            D_direct_pressure_shrinks_anchor=sum(r['D_direct_pressure_shrinks_anchor'] for r in covered),
            rivals=metrics, baseline_margin_quantiles=np.quantile(
                [r['baseline_margin'] for r in covered], [0, .1, .5, .9, 1]).tolist() if covered else [],
            positive_origins=dict(Counter(v['origin'] for r in qs for v in r['positive_visibility'])),
            truncated_positives=sum(v['truncated'] for r in qs for v in r['positive_visibility']),
            baseline_D_query_mean_loss=float(np.mean([r['baseline_D_loss'] for r in qs])),
            baseline_D_query_mean_score_gradient_l1=float(np.mean([
                r['baseline_D_score_gradient_l1'] for r in qs])),
            synthetic_positive_margin_drop={str(drop): {key: float(np.mean([
                r['synthetic_positive_margin_drop'][str(drop)][key] for r in qs]))
                for key in ('loss', 'score_gradient_l1')} for drop in (.1, .5, 1.)})
    return result


def prepare(base, run, output):
    frozen = io.read(run / 'inputs.json')
    io.require(frozen['training_enabled'] is False and frozen['new_model_inference'] is False,
               'preparation cannot enable training')
    io.verify(run, frozen['source'])
    io.verify(base, frozen['dependencies'])
    parent = base / PARENT
    config = io.read(parent / 'config.json')
    records = [json.loads(line) for line in (parent / 'witnesses.jsonl').read_text().splitlines()]
    selection = io.read(base / PILOT / 'selection.json')
    order = io.read(base / PILOT / 'training-order.json')
    io.require(order == io.read(parent / 'training-order.json'), 'original D order differs')
    all_queries = io.read(base / 'NG-0069/lexical-v1/queries.json')
    positions = {q['query_id']: i for i, q in enumerate(all_queries)}
    queries = {q['query_id']: q for q in all_queries}
    validate_cohort(records, selection, order, queries)
    for record in records:
        io.require(record['checkpoint_state_sha256'] == config['base']['state_sha256']
                   and record['query_sha256'] == hashlib.sha256(
                       queries[record['query_id']]['query'].encode()).hexdigest(),
                   'initial model or query text identity changed')
    heads = selected_rankings(base / 'NG-0069/evaluation-v2/rank-initial/rankings.jsonl', selection['pilot'])
    dense = base / 'NG-0069/evaluation-v2/encode-dense'
    dq = np.load(dense / 'query.npy', mmap_mode='r')
    dd = np.load(dense / 'document.npy', mmap_mode='r')
    io.require(dd.shape == (233009, 1024) and dq.shape == (len(all_queries), 1024), 'teacher cache shape changed')
    anchors, pairs, per_query, errors = {}, [], [], []
    for record in records:
        q = record['query_id']
        io.require(positions[q] < 6144, 'non-TRAIN teacher query access')
        head = heads[q]
        ids = sorted(set(record['pool']) | set(head['top100']))
        vector = dq[positions[q]].astype(np.float64)
        matrix = dd[ids].astype(np.float64)
        values = matrix @ vector
        independent = np.sum(matrix * vector, axis=1)
        error = float(abs(values - independent).max())
        io.require(error <= 1e-12 and np.isfinite(values).all(), 'cached teacher dot check failed')
        teacher = dict(zip(ids, values.tolist(), strict=True))
        error = max(error, max(abs(teacher[d] - s) for d, s in
                              zip(record['pool'], record['teacher_scores'], strict=True)))
        io.require(error <= 1e-12, 'cached teacher differs from original witness')
        errors.append(error)
        anchor, rows, summary = query_audit(record, head,
            {d: teacher[d] for d in ids if d not in record['pool']}, config['ranking'])
        anchors[q] = anchor
        pairs.extend(rows)
        per_query.append(summary)
    output.mkdir(exist_ok=False)
    write(output / 'anchors.json', anchors)
    write(output / 'per-query.json', per_query)
    with (output / 'pair-audit.jsonl').open('x') as stream:
        for row in pairs:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    results = dict(passed=True, domains=summarize(pairs, per_query),
        max_cached_teacher_dot_error=max(errors), queries=len(records),
        audited_pairs=len(pairs), anchors=sum(len(a['anchors']) for a in anchors.values()),
        original_D_order_sha256=io.sha(parent / 'training-order.json'),
        initial_snapshot_sha256=PARENT_SHA, new_model_inference=False,
        teacher_cache_access='384 TRAIN_PILOT rows and their original-pool/top100 document union only',
        outside100_coverage='original D witness subset only; not complete outside100 universe',
        global_positive_denominator=sum(r['total_positives'] for r in records),
        unanchored_positives_retained_in_denominator=True,
        forward_pool_changed=False, outside_pool_added_as_anchor=False,
        training_enabled=False, cuda_no_update_canary_passed=False,
        full_training_manifest_frozen=False, coefficient_selected=False,
        sentinel_or_dev_outcomes_used=False, locked_test_scored=False,
        heldout_quality_evaluation=False, overall_goal_qualified=False)
    write(output / 'results.json', results)
    io.verify(run, frozen['source'])
    io.verify(base, frozen['dependencies'])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'prepare'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    base, run = args.research_root.resolve(strict=True), args.run.resolve()
    io.require(run.parent == base / 'NG-0077', 'fresh NG77 unit required')
    result = freeze(base, run) if args.mode == 'freeze' else prepare(base, run, run / 'audit')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
