"""Independent scalar-loss, exposure and sealed optimizer-chain audit of NG79.

No Torch, encoder inference, parameter VJP or optimizer-tensor loading occurs.
Raw parameter/readout replay remains the separately sealed worker evidence.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

import ng79_pilot as protocol
from review_ng75_displacement import compare, read, require, sha, verify
from review_ng79_preparation import scalar_objective


INPUT = 'fab3122d21c270dda659bef84e9e35bcaebe5d5a2ada1b8bc448437f97ef7068'
TRAIN = 'NG-0079/matched-breadth-384-v1'


def rows(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def sealed(root):
    complete = read(root / 'complete.json')
    require(complete['passed'] is True and {'results.json', 'exit.json', 'clearml.json', 'started.json'}
            <= set(complete['files']), 'missing successful closure')
    verify(root, complete['files'], exact=True, exclude=('complete.json',))
    end, start, tracking = [read(root / n) for n in ('exit.json', 'started.json', 'clearml.json')]
    require(end['exit_code'] == 0 and end['error'] is None and end['owned_group_closed']
            and end['elapsed_seconds'] < 1800 and end['peak_tree_rss_bytes'] <= 16 * 2 ** 30
            and start['limit_seconds'] == 1800 and end['finished_unix'] > start['started_unix'],
            'unsafe or unbounded phase closure')
    require(tracking['actual_start'] and tracking['closed'] and tracking['outcome'] == 'passed',
            'tracking did not close')
    return read(root / 'results.json')


def audit_example(example, record, config):
    require(example['query_id'] == record['query_id'] and record['split'] == 'TRAIN'
            and record['reference_update'] == 0 and example['vjp_replay_exact'] is True
            and example['documents'] == len(record['pool'])
            and example['objective'] == 'balanced_soft_pair'
            and 'objective_components' not in example, 'example or objective changed')
    expected = scalar_objective(dict(record, hybrid_scores=example['scores']), config)
    require(example['eligible_pairs'] == expected['eligible_pairs']
            and example['supervised_positives'] == expected['supervised_positives'],
            'all-positive supervision changed')
    np.testing.assert_allclose(example['loss'], expected['loss'], rtol=2e-6, atol=2e-7)
    target = np.asarray(expected['score_gradient']) / 4
    observed = np.asarray(example['score_gradient'])
    np.testing.assert_allclose(observed, target, rtol=2e-5, atol=2e-7)
    return float(np.max(np.abs(observed - target)))


def audit_phase(root, result, config, order, records, previous):
    _, arm, end = root.name.split('-')
    end = int(end)
    require(result['arm'] == arm and result['cumulative_steps'] == end and result['steps'] == 96
            and result['query_exposures'] == 384 and result['checkpoint_optimizer_replay_exact']
            and sha(root / 'optimizer.pt') == result['optimizer_sha256']
            and result['static_reference_update'] == 0
            and not any(result[k] for k in ('gpu_observer', 'quality_evaluation', 'dev_access',
                                           'locked_test_access')), 'training phase contract differs')
    if previous is None:
        require(end == 96 and result['initial_model_sha256'] == config['base']['state_sha256']
                and result['initial_optimizer_fingerprint'] is None
                and result['previous_complete_sha256'] is None, 'fresh initialization required')
        compare({k: result[k] for k in protocol.PREFIX}, protocol.PREFIX, atol=0.)
        require(result['common_prefix_reproduced'], 'prefix check not recorded')
    else:
        prior_path, prior = previous
        require(prior['arm'] == arm and prior['cumulative_steps'] == end - 96
                and result['initial_model_sha256'] == prior['model_sha256']
                and result['initial_optimizer_fingerprint'] == prior['optimizer_fingerprint']
                and result['previous_complete_sha256'] == sha(prior_path / 'complete.json'),
                'same-arm model/optimizer continuation broken')
    seen, tokens, candidates, max_error, elapsed = [], dict(query=0, document=0), 0, 0., 0.
    for offset, step in enumerate(rows(root / 'progress.jsonl')):
        update = end - 96 + offset
        identities = [e['query_id'] for e in step['examples']]
        require(offset < 96 and step['step'] == update + 1 and step['reference_update'] == 0
                and identities == order[update * 4:(update + 1) * 4], 'update identity/order differs')
        require(all(math.isfinite(step[k]) and step[k] > 0
                    for k in ('update_l2', 'gradient_before_clip', 'seconds')), 'invalid update')
        elapsed += step['seconds']
        for example in step['examples']:
            max_error = max(max_error, audit_example(example, records[example['query_id']], config['ranking']))
            candidates += example['documents']
            for role in tokens:
                value = example[role + '_tokens']
                require(type(value) is int and value > 0, 'invalid measured token count')
                tokens[role] += value
        seen.extend(identities)
    require(len(seen) == 384 and seen == order[(end - 96) * 4:end * 4]
            and tokens == result['tokens'] and candidates == result['candidate_pairs'],
            'actual exposure counts differ')
    return dict(queries=seen, tokens=tokens, document_exposures=candidates,
                compute_seconds=elapsed, max_score_gradient_error=max_error)


def review(base, run, inventory_sha):
    require(run == base / TRAIN and sha(run / 'inputs.json') == INPUT, 'wrong training unit')
    inventory = run.with_name(run.name + '-remote-inventory.json')
    require(len(inventory_sha) == 64 and sha(inventory) == inventory_sha, 'training inventory changed')
    verify(run, read(inventory)['files'], exact=True)
    protocol.verify(base, run)
    control = read(run / 'controller-exit.json')
    require(control['status'] == 'all_phases_complete' and control['completed'] == list(protocol.PHASES),
            'training controller is not fully closed')
    configs, selection = read(run / 'configs.json'), read(run / 'selection.json')
    records = {r['query_id']: r for r in rows(base / protocol.PREPARATION / 'witnesses/witnesses.jsonl')}
    require(len(records) == 1536 and set(records) == set(selection['breadth'])
            and sum(r['total_positives'] for r in records.values()) == 2680, 'all-positive cohort changed')
    last, previous_end, stages = {}, None, {}
    for phase in protocol.PHASES:
        root = run / phase
        result = sealed(root)
        arm = result['arm']
        start, ended = read(root / 'started.json'), read(root / 'exit.json')
        require(previous_end is None or start['started_unix'] >= previous_end, 'workers overlapped')
        previous_end = ended['finished_unix']
        detail = audit_phase(root, result, configs[arm], selection['orders'][arm], records, last.get(arm))
        stages[phase] = dict(detail, model_sha256=result['model_sha256'],
                            optimizer_fingerprint=result['optimizer_fingerprint'],
                            worker_seconds=ended['elapsed_seconds'])
        last[arm] = (root, result)
    totals = {}
    for arm, cohort, repeats in (('R', 'repeated', 4), ('B', 'breadth', 1)):
        parts = [stages[f'train-{arm}-{n}'] for n in (96, 192, 288, 384)]
        seen = sum((p['queries'] for p in parts), [])
        require(seen == selection['orders'][arm]
                and Counter(seen) == Counter({q: repeats for q in selection[cohort]}), 'arm exposure differs')
        totals[arm] = dict(updates=384, query_exposures=len(seen), unique_train_queries=len(set(seen)),
            tokens={r: sum(p['tokens'][r] for p in parts) for r in ('query', 'document')},
            document_exposures=sum(p['document_exposures'] for p in parts),
            worker_seconds=sum(p['worker_seconds'] for p in parts),
            compute_seconds=sum(p['compute_seconds'] for p in parts),
            max_score_gradient_error=max(p['max_score_gradient_error'] for p in parts),
            terminal_model_sha256=last[arm][1]['model_sha256'],
            terminal_optimizer_fingerprint=last[arm][1]['optimizer_fingerprint'])
    return dict(passed=True, input_sha256=INPUT, remote_inventory_sha256=inventory_sha,
                arms=totals, scalar_reviewer_sha256=sha(Path(__file__)),
                actual_parameter_vjp_recomputed=False, optimizer_tensors_independently_loaded=False,
                inference_performed=False, quality_evaluation=False, dev_access=False,
                locked_test_access=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inventory-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(review(args.research_root, args.run, args.inventory_sha), stream, indent=2, allow_nan=False)
        stream.write('\n')
