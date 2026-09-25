"""CPU-only independent NG77 actual-loss, exposure and closure audit.

No Torch, checkpoint inference, optimizer update or quality evaluation. The
original D reference is NumPy; the keep gap is independently evaluated with
50-digit Decimal arithmetic, not the training Taylor approximation.
"""

import argparse
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path

import numpy as np

import ng71_ranking as reference
from review_ng75_displacement import require, read, sha, verify
from review_ng77_preparation import probability


INPUT = '16f9bf709176a88d46627ea56e24466820430d352a1008d080b488ab65691e97'
INVENTORY = '53649fef49af83ee1a0def472ec719f8940a0ad0c49022d5b001c5e1108c8c8b'
PHASES = ('train-Z-96', 'train-K-96')
OLD_COMPLETE = 'cbb8e9a0c77985359ce30f36914b8aeaf3fc9fe8608c4471c78ddcd6779419c0'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sealed(root):
    complete = read(root / 'complete.json')
    require(complete['passed'] is True, 'failed phase')
    require({'results.json', 'exit.json', 'clearml.json', 'started.json'}
            <= set(complete['files']), 'unbound closure')
    verify(root, complete['files'], exact=True, exclude=('complete.json',))
    ended, started, tracking = [read(root / n) for n in
                               ('exit.json', 'started.json', 'clearml.json')]
    require(ended['exit_code'] == 0 and ended['error'] is None
            and ended['owned_group_closed'] is True
            and ended['elapsed_seconds'] < 1800
            and ended['peak_tree_rss_bytes'] <= 16 * 2 ** 30
            and started['limit_seconds'] == 1800
            and ended['finished_unix'] > started['started_unix'], 'unsafe phase closure')
    require(tracking['actual_start'] and tracking['closed']
            and tracking['outcome'] == 'passed', 'tracking not closed')
    return read(root / 'results.json')


def keep_reference(scores, anchors):
    require(len(scores) == anchors['candidate_count'] and np.isfinite(scores).all(),
            'invalid candidate score surface')
    losses, gradient = [], np.zeros(len(scores), dtype=np.float64)
    temperature = anchors['student_temperature']
    require(temperature > 0 and math.isfinite(temperature), 'invalid temperature')
    with localcontext() as context:
        context.prec = 50
        one, t = Decimal(1), Decimal.from_float(float(temperature))
        for row in anchors['anchors']:
            p, n = row['indices']
            a0 = Decimal.from_float(row['baseline_margin']) / t
            a = (Decimal.from_float(float(scores[p])) - Decimal.from_float(float(scores[n]))) / t
            require(a0 > 0 and row['coefficient'] > 0, 'invalid frozen anchor')
            if a >= a0:
                continue
            # Preserve log(1 + exp(-large_margin)) rather than rounding it to 0.
            context.prec = max(50, int(max(abs(float(a)), float(a0)) / math.log(10)) + 50)
            r = one / (one + a0.exp())
            gap = (one + (-a).exp()).ln() - (one + (-a0).exp()).ln() - r * (a0 - a)
            require(gap >= 0, 'negative convex gap')
            losses.append(row['coefficient'] * float(gap))
            derivative = row['coefficient'] * (probability(float(a)) - probability(float(a0))) / temperature
            gradient[p] += derivative
            gradient[n] -= derivative
    return math.fsum(losses), gradient


def audit_example(example, record, anchors, config, coefficient):
    require(example['query_id'] == record['query_id'] and record['split'] == 'TRAIN'
            and example['vjp_replay_exact'] is True
            and example['documents'] == len(record['pool'])
            and example['objective'] == 'balanced_soft_pair', 'example identity or VJP changed')
    expected = reference.loss_and_gradient(
        np.asarray(example['scores']), np.asarray(record['positive_mask']),
        np.asarray(record['global_ranks']), np.asarray(record['teacher_scores']),
        np.asarray(record['judged_negative_mask']), total_positives=record['total_positives'],
        corpus_size=record['corpus_size'], rank_scope=record['rank_scope'], **{
            k: config[k] for k in ('student_temperature', 'teacher_temperature',
                'ndcg_cutoff', 'recall_cutoff', 'recall_weight', 'pair_floor')})
    keep, derivative = keep_reference(example['scores'], anchors)
    component = example['objective_components']
    require(component['retention_coefficient'] == coefficient
            and example['eligible_pairs'] == expected['eligible_pairs']
            and example['supervised_positives'] == expected['supervised_positives'],
            'loss coefficient or all-positive supervision changed')
    np.testing.assert_allclose(component['original_loss'], expected['loss'], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(component['keep_loss'], keep, rtol=2e-10, atol=1e-13)
    np.testing.assert_allclose(example['loss'], expected['loss'] + coefficient * keep,
                               rtol=2e-6, atol=2e-7)
    target = (expected['gradient'] + coefficient * derivative) / 4
    observed = np.asarray(example['score_gradient'])
    np.testing.assert_allclose(observed, target, rtol=2e-5, atol=2e-7)
    return dict(score_gradient_max_error=float(np.max(np.abs(observed - target))),
                keep_loss_error=abs(component['keep_loss'] - keep),
                keep_gradient_l1=float(np.abs(derivative).sum()),
                original_gradient_l1=float(np.abs(expected['gradient']).sum()),
                keep_loss=keep)


def review(base, run):
    require(sha(run / 'inputs.json') == INPUT, 'training identity changed')
    inventory = run.with_name(run.name + '-remote-inventory.json')
    require(sha(inventory) == INVENTORY, 'remote inventory changed')
    verify(run, read(inventory)['files'], exact=True)
    manifest = read(run / 'inputs.json')
    verify(base, manifest['dependencies'])
    verify(run, manifest['source'])
    require(read(run / 'controller-exit.json')['completed'] == list(PHASES)
            and read(run / 'controller-exit.json')['status'] == 'all_phases_complete', 'controller incomplete')
    config, order = read(run / 'config.json'), read(run / 'training-order.json')
    records = {r['query_id']: r for r in rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')}
    anchors = read(base / 'NG-0077/preparation-v1/audit/anchors.json')
    require(set(order[:384]) == set(records) == set(anchors) and len(set(order[:384])) == 384
            and sum(r['total_positives'] for r in records.values()) == 660
            and sum(len(r['anchors']) for r in anchors.values()) == 45175, 'cohort changed')
    totals, previous_end = {}, None
    for arm, coefficient in (('Z', 0.), ('K', 1.)):
        root = run / f'train-{arm}-96'
        result = sealed(root)
        require(result['arm'] == arm and result['steps'] == result['cumulative_steps'] == 96
                and result['query_exposures'] == 384 and result['retention_coefficient'] == coefficient
                and result['initial_model_sha256'] == config['base']['state_sha256']
                and result['initial_optimizer_fingerprint'] is None
                and result['previous_complete_sha256'] is None
                and result['checkpoint_optimizer_replay_exact']
                and sha(root / 'optimizer.pt') == result['optimizer_sha256']
                and not result['gpu_observer'] and not result['quality_evaluation']
                and not result['locked_test_access'], 'training chain or contract changed')
        start = read(root / 'started.json')['started_unix']
        if previous_end is not None:
            require(start >= previous_end, 'training processes overlapped')
        previous_end = read(root / 'exit.json')['finished_unix']
        progress, details = rows(root / 'progress.jsonl'), []
        require(len(progress) == 96, 'incomplete optimizer log')
        tokens, documents, seen = dict(query=0, document=0), 0, []
        for i, step in enumerate(progress):
            identities = [r['query_id'] for r in step['examples']]
            require(step['step'] == i + 1 and step['reference_update'] == 0
                    and identities == order[i * 4:(i + 1) * 4], 'exposure/order changed')
            require(all(math.isfinite(step[k]) and step[k] > 0
                        for k in ('update_l2', 'gradient_before_clip', 'seconds')), 'invalid update')
            for example in step['examples']:
                q = example['query_id']
                details.append(audit_example(example, records[q], anchors[q], config['ranking'], coefficient))
                documents += example['documents']
                for role in tokens:
                    tokens[role] += example[role + '_tokens']
            seen.extend(identities)
        require(seen == order[:384] and tokens == result['tokens']
                and documents == result['candidate_pairs'], 'actual counters differ')
        totals[arm] = dict(updates=96, unique_train_queries=384, tokens=tokens,
            document_exposures=documents, model_sha256=result['model_sha256'],
            optimizer_fingerprint=result['optimizer_fingerprint'],
            max_score_gradient_error=max(d['score_gradient_max_error'] for d in details),
            max_keep_loss_error=max(d['keep_loss_error'] for d in details),
            mean_keep_loss=float(np.mean([d['keep_loss'] for d in details])),
            mean_keep_gradient_l1=float(np.mean([d['keep_gradient_l1'] for d in details])),
            mean_D_gradient_l1=float(np.mean([d['original_gradient_l1'] for d in details])),
            seconds=read(root / 'exit.json')['elapsed_seconds'])
    historical = base / 'NG-0071/pilot-v1/train-D-96'
    require(sha(historical / 'complete.json') == OLD_COMPLETE, 'historical control changed')
    verify(historical, read(historical / 'complete.json')['files'], exact=True, exclude=('complete.json',))
    old = read(historical / 'results.json')
    totals['zero_matches_historical_D96'] = all(totals['Z'][k] == old[k]
                                               for k in ('model_sha256', 'optimizer_fingerprint'))
    require(totals['Z']['model_sha256'] != totals['K']['model_sha256'], 'retention did not change model')
    return dict(passed=True, input_sha256=INPUT, remote_inventory_sha256=INVENTORY,
                files_verified=len(read(inventory)['files']), arms=totals,
                parameter_vjp_independently_recomputed=False, inference_performed=False,
                quality_evaluation=False, locked_test_access=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(review(args.research_root, args.run), stream, indent=2, allow_nan=False)
        stream.write('\n')
