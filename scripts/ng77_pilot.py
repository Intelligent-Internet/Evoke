"""Frozen NG77 matched 96-update arms; endpoint evaluation is a later stage.

No observer, pool refresh, DEV, LOCKED_TEST, automatic extension or promotion.
The wider TRAIN endpoint identities and decision rules are frozen here, before
either training outcome exists. Both training processes must close first.
"""

import argparse
import copy
from pathlib import Path
import shutil

import numpy as np

import ng71_execution as execution
import ng71_observation as observation
import ng71_pilot as pipeline
import ng71_preflight as io
import ng77_cuda_canary as canary
from ng77_training import RetainedObjective
from review_ng75_displacement import require, verify as verify_files


CANARY = 'NG-0077/context-qualified-canary-v2'
CANARY_INPUT = '362a94b730e511bdb652e31c0bcae27feaaadc6b42c11ef0bed3f08d6cfe31ca'
CANARY_COMPLETE = '7fd7c380d1e9a75c4e8275ed5f2c219b76833ec7fce0337e947e496f81abbb9c'
REVIEW = '89449f53e778adb3c13925612709c815e2ed6e03f7af9ffe42412c71372146d5'
PROCEDURE = '94ffa81a07228821e654b412ba878e1d22fb852484cd1c5eaa4cb186d3fcb54a'
PARENT = 'NG-0071/pilot-v1'
PHASES = ('train-Z-96', 'train-K-96')
COEFFICIENTS = {'Z': 0., 'K': 1.}
GATES = dict(primary_surface='TRAIN_SENTINEL', bootstrap_seed=71071, bootstrap_replicates=10000,
    keep_minus_zero_ndcg_point_min=.005, keep_minus_zero_ndcg_95_lower_strictly_above=0.,
    keep_minus_zero_recall_95_lower_min=-.002, per_domain_ndcg_floor_vs_zero_and_initial=-.005,
    per_domain_recall_floor_vs_zero_and_initial=-.005, document_nnz_ratio_vs_initial_max=1.25,
    sentinel_query_df_ratio_vs_initial_max=1.25, automatic_192_updates=False,
    automatic_dev_access=False, independent_holdout=False, native_cost_evaluated=False)


def configuration(parent):
    result = copy.deepcopy(parent)
    result['arms'] = {a: copy.deepcopy(parent['arms']['D']) for a in COEFFICIENTS}
    result['status'] = 'frozen_ng77_matched_retention_96'
    require(result['training_enabled'] and result['arms']['Z']
            == dict(pool='witness', objective='balanced_soft_pair'), 'original D contract differs')
    return result


def qualified_canary(base):
    root = base / CANARY
    require(io.sha(root / 'inputs.json') == CANARY_INPUT
            and io.sha(root / 'canary/complete.json') == CANARY_COMPLETE, 'wrong accepted CUDA gate')
    m = canary.verify(base, root)
    result = execution.sealed(root / 'canary')
    require(result['passed'] and result['optimizer_updates'] == 0
            and result['model_and_rng_unchanged'] and result['context_qualified_comparison'],
            'canary is not qualified')
    reviewed = root.with_name(root.name + '-mirror-review.json')
    procedure = root.with_name(root.name + '-mirror-review-procedure.json')
    require(io.sha(reviewed) == REVIEW and io.sha(procedure) == PROCEDURE, 'CPU mirror review changed')
    r, p = execution.read(reviewed), execution.read(procedure)
    require(r['passed'] and r['input_sha256'] == CANARY_INPUT
            and r['sealed_cuda_vjp_receipts_verified'] and p['exit_code'] == 0
            and p['error'] is None and p['owned_group_closed'], 'CPU review not safely completed')
    inventory = root.with_name(root.name + '-remote-inventory.json')
    verify_files(root, execution.read(inventory)['files'], exact=True)
    deps = dict(m['dependencies'])
    deps.update({str(p.relative_to(base)): io.sha(p) for p in root.rglob('*') if p.is_file()})
    deps.update({str(p.relative_to(base)): io.sha(p) for p in (reviewed, procedure, inventory)})
    return deps


def validate_cohort(selection, order, records, endpoints):
    require(len(records) == 384 and len(order) == 768 and len(set(order[:384])) == 384
            and set(order[:384]) == set(records) == set(selection['pilot'])
            and set(order[384:]) == set(records), 'original training order/coverage changed')
    require(len(endpoints) == 768 and [q['query_id'] for q in endpoints]
            == selection['pilot'] + selection['sentinel']
            and len({q['query_id'] for q in endpoints}) == 768
            and all(q['split'] == 'TRAIN' for q in endpoints), 'endpoint identity or TRAIN split changed')
    require(sum(r['total_positives'] for r in records.values()) == 660
            and all(r['split'] == 'TRAIN' and r['reference_update'] == 0
                    and len(r['positive_ids']) == r['total_positives'] for r in records.values()),
            'complete original positive/reference contract changed')


def verify(base, run):
    m = execution.read(run / 'inputs.json')
    require(m['protocol'] == 'NG77_matched_retention_training_v1' and m['phases'] == list(PHASES)
            and m['retention_coefficients'] == COEFFICIENTS and m['endpoint_gates'] == GATES
            and m['updates_per_arm'] == 96 and not m['quality_evaluation_enabled']
            and not m['locked_test_access'] and not m['gpu_observer']
            and m['stage_limit_seconds'] == 1800,
            'frozen study or staging contract changed')
    verify_files(base, m['dependencies'])
    verify_files(run, m['source'])
    require(execution.read(run / 'config.json') == configuration(execution.read(base / PARENT / 'config.json')),
            'single-factor scientific configuration changed')
    for name in ('selection.json', 'training-order.json'):
        require(io.sha(run / name) == io.sha(base / PARENT / name), 'training cohort/order changed')
    selection = execution.read(run / 'selection.json')
    endpoints = observation.surface_queries(base, selection, True, include_dev=False)
    require(execution.read(run / 'endpoint-queries.json') == endpoints, 'wide TRAIN endpoint changed')
    records = {r['query_id']: r for r in pipeline.rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')}
    validate_cohort(selection, execution.read(run / 'training-order.json'), records, endpoints)
    return m


def freeze(base, run):
    dependencies = qualified_canary(base)
    config = configuration(execution.read(base / PARENT / 'config.json'))
    selection = execution.read(base / PARENT / 'selection.json')
    endpoints = observation.surface_queries(base, selection, True, include_dev=False)
    run.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    names = [p.name for p in source.glob('ng71_*.py')]
    names += ['ng77_pilot.py', 'ng77_retention.py', 'ng77_training.py', 'ng77_cuda_canary.py',
              'review_ng75_displacement.py', 'audit_ng70_provenance.py', 'analyze_ng66_learning_curves.py']
    for name in names:
        shutil.copy2(source / name, run / name)
    for name in ('test_ng77_pilot.py', 'test_ng77_execution.py', 'test_ng77_cuda.py'):
        shutil.copy2(source.parent / 'tests' / name, run / name)
    for name in ('selection.json', 'training-order.json'):
        shutil.copy2(base / PARENT / name, run / name)
    plan = source.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0077-matched-retention-pilot.zh.md'
    shutil.copy2(plan, run / plan.name)
    io.write(run / 'config.json', config)
    io.write(run / 'endpoint-queries.json', endpoints)
    io.write(run / 'inputs.json', dict(protocol='NG77_matched_retention_training_v1',
        dependencies=dependencies, source={p.name: io.sha(p) for p in run.iterdir() if p.is_file()},
        phases=list(PHASES), retention_coefficients=COEFFICIENTS, endpoint_gates=GATES,
        updates_per_arm=96, quality_evaluation_enabled=False, locked_test_access=False,
        stage_limit_seconds=1800, gpu_observer=False, actual_training_started=False))
    verify(base, run)
    print('NG77_MATCHED_TRAINING_FROZEN', io.sha(run / 'inputs.json'), flush=True)


def worker(args):
    import torch
    from clearml import Task

    verify(args.research_root, args.run)
    require(args.phase in PHASES, 'only the two frozen training phases are enabled')
    if args.phase == PHASES[1]:
        observation.training_closed(args.run, phases=[PHASES[0]])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(71001)
    Task.set_offline(True)
    task = Task.init(project_name='Evoke-NG', task_name='NG-0077/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False,
        auto_connect_arg_parser=False, auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    io.write(output / 'clearml-start.json', receipt)
    try:
        base, run = args.research_root, args.run
        config = execution.read(run / 'config.json')
        arm = args.phase.split('-')[1]
        task.connect(dict(config=config, input_sha256=io.sha(run / 'inputs.json'), coefficient=COEFFICIENTS[arm]))
        records = {r['query_id']: r for r in pipeline.rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')}
        anchors = execution.read(base / 'NG-0077/preparation-v1/audit/anchors.json')
        require(set(anchors) == set(records) and sum(len(v['anchors']) for v in anchors.values()) == 45175,
                'frozen retention anchors changed')
        transforms = {q: RetainedObjective(anchors[q], COEFFICIENTS[arm]) for q in records}
        selection = execution.read(run / 'selection.json')
        queries = observation.surface_queries(base, selection, False)
        documents = pipeline.rows(base / 'NG-0067/data/documents.jsonl')
        result = execution.train_chunk(base, config, output, arm, 0, 96, records,
            execution.read(run / 'training-order.json'), {q['query_id']: q for q in queries},
            documents, task, loss_transforms=transforms, stage_limit_seconds=1800)
        result.update(retention_coefficient=COEFFICIENTS[arm], anchors=45175,
                      gpu_observer=False, quality_evaluation=False, locked_test_access=False)
        verify(base, run)
        io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        io.write(output / 'clearml.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3))
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    require(args.run.parent == args.research_root / 'NG-0077', 'new NG77 unit required')
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    else:
        manifest = verify(args.research_root, args.run)
        require(io.sha(Path(__file__)) == manifest['source']['ng77_pilot.py'], 'executing source not frozen')
        if args.mode == 'worker':
            worker(args)
        elif args.mode == 'supervise':
            require(args.gpu is not None, 'explicit physical GPU required')
            pipeline.supervise_graph(args, PHASES, 'ng77_pilot.py', limit_seconds=1800)
