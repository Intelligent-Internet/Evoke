"""Bounded single-factor uniform-pair control against frozen NG71 D."""

import argparse
import copy
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np

import ng71_diagnostics as diagnostics
import ng71_execution as execution
import ng71_observation as observation
import ng71_pilot as pipeline
import ng71_preflight as io
import ng71_training as training


PARENT = 'NG-0071/pilot-v1'
INPUT_SHA = '5c44c86e13d688972100f1eb1a85b562ba8a631bdf31a2b2e4073ef9b304fb2a'
REVIEW_SHA = 'f99d5a33670d3a9cb018eba86ec5780256f4c51d14fcfeb14345ff51c7e6fda3'
TRAIN_PHASES = ('train-U-96', 'train-U-192')
PHASES = ('train-preflight', *TRAIN_PHASES, 'encode-U-192', 'rank-U-192', 'review')


def require(test, message):
    if not test:
        raise ValueError(message)


def configuration(parent):
    config = copy.deepcopy(parent)
    config['arms'] = {'U': dict(pool='witness', objective='uniform_soft_pair')}
    config['status'] = 'frozen_uniform_pair_control'
    return config


def verify(base, run):
    frozen = execution.read(run / 'inputs.json')
    require(frozen['phases'] == list(PHASES) and frozen['locked_test_access'] is False,
            'execution graph or test boundary changed')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in frozen[key].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve()) and io.sha(path) == digest,
                    'frozen input changed: ' + name)
    require(execution.read(run / 'config.json') == configuration(execution.read(base / PARENT / 'config.json')),
            'scientific configuration changed beyond uniform rank weight')
    require(execution.read(run / 'math-preflight.json')['passed'] is True, 'math gate not passed')
    return frozen


def math_check(base, config, order):
    references = {0: base / 'NG-0071/pilot-step0-v1', 96: base / PARENT / 'snapshot-96'}
    counts, max_error = {}, 0.
    for start, folder in references.items():
        records = {r['query_id']: r for r in pipeline.rows(folder / 'witnesses.jsonl')}
        require(len(records) == 384 and set(records) == set(order), 'pilot surface changed')
        indices, targets, changed_coefficients = 0, 0, 0
        eligible = {}
        for identity, record in records.items():
            require(record['split'] == 'TRAIN' and record['reference_update'] == start, 'wrong reference/split')
            m = training.prepare_pairs(record, config['ranking'])
            u = training.prepare_pairs(record, config['ranking'], uniform=True)
            require(m['indices'] == u['indices'] and m['targets'] == u['targets']
                    and m['supervised_positives'] == u['supervised_positives']
                    and m['unresolved_pairs'] == u['unresolved_pairs'], 'uniform changed supervision')
            checked = diagnostics.objective(record, config['ranking'], 'uniform_soft_pair')
            max_error = max(max_error, checked['max_independent_derivative_error'])
            indices += len(u['indices'])
            targets += len(u['targets'])
            changed_coefficients += m['coefficients'] != u['coefficients']
            eligible[identity] = len(u['indices'])
        for step in range(start, start + 96):
            require(any(eligible[q] for q in order[step * 4:(step + 1) * 4]), 'empty optimizer batch')
        counts[str(start)] = dict(queries=len(records), eligible_pairs=indices,
                                  targets=targets, queries_with_changed_coefficients=changed_coefficients)
    return dict(passed=True, references=counts, max_independent_derivative_error=max_error,
                training_updates=0, new_model_inference=False, locked_test_scored=False)


def freeze(args):
    base, run = args.research_root, args.run
    require(io.sha(base / PARENT / 'inputs.json') == INPUT_SHA
            and io.sha(base / PARENT / 'review/complete.json') == REVIEW_SHA, 'parent anchors changed')
    parent = pipeline.verify(base, base / PARENT)
    config = configuration(execution.read(base / PARENT / 'config.json'))
    require(config['training_enabled'] is True, 'parent training not enabled')
    dependencies = dict(parent['dependencies'])
    dependencies.update({PARENT + '/' + n: h for n, h in parent['source'].items()})
    dependencies[PARENT + '/inputs.json'] = INPUT_SHA
    for phase in ('review', 'snapshot-96', 'train-D-96', 'train-D-192',
                  'rank-initial', 'rank-dense', 'rank-A-192', 'rank-D-192'):
        folder = base / PARENT / phase
        execution.sealed(folder)
        dependencies[PARENT + '/' + phase + '/complete.json'] = io.sha(folder / 'complete.json')
        dependencies.update({PARENT + '/' + phase + '/' + n: h
                             for n, h in execution.read(folder / 'complete.json')['files'].items()})
    require(args.run.parent == base / 'NG-0074', 'fresh NG74 unit required')
    run.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    names = [p.name for p in source.glob('ng71_*.py')]
    names += ['ng74_uniform_control.py', 'review_ng74_uniform_control.py',
              'review_ng71_pilot.py', 'audit_ng70_provenance.py', 'analyze_ng66_learning_curves.py']
    for name in names:
        shutil.copy2(source / name, run / name)
    for name in ('selection.json', 'training-order.json'):
        shutil.copy2(base / PARENT / name, run / name)
    for path in (source.parent / 'tests').glob('*ng74*.py'):
        shutil.copy2(path, run / path.name)
    plan = source.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0074-uniform-control-plan.zh.md'
    shutil.copy2(plan, run / plan.name)
    io.write(run / 'config.json', config)
    io.write(run / 'math-preflight.json', math_check(base, config, execution.read(run / 'training-order.json')))
    io.write(run / 'inputs.json', dict(
        phases=list(PHASES), dependencies=dependencies,
        source={str(p.relative_to(run)): io.sha(p) for p in run.rglob('*') if p.is_file()},
        locked_test_access=False, actual_training_started=False,
        scientific_factor='rank_weights_only_uniform_1', parent_review_sha256=REVIEW_SHA))
    verify(base, run)


def dispatch(args, config, task):
    base, run, phase = args.research_root, args.run, args.phase
    output = run / phase
    if phase == 'train-preflight':
        return io.model_check(base, config, output, device='cuda')
    preflight = execution.sealed(run / 'train-preflight')
    require(preflight['passed'] is True and preflight['scientific_training_enabled'] is False
            and preflight['initial_state_sha256'] == config['base']['state_sha256']
            and preflight['checkpoint_reload_bit_exact'] is True
            and preflight['optimizer_reload_bit_exact'] is True, 'CUDA fixture not passed')
    if phase == 'review':
        from review_ng74_uniform_control import review
        return review(base, run)
    witness = base / PARENT / 'snapshot-96'
    if phase == 'rank-U-192':
        return observation.rank(base, config, run, output, 'U-192',
                                training_phases=TRAIN_PHASES, witness_source=witness)
    selection = execution.read(run / 'selection.json')
    documents = pipeline.rows(base / 'NG-0067/data/documents.jsonl')
    if phase == 'encode-U-192':
        observation.training_closed(run, TRAIN_PHASES)
        queries = observation.surface_queries(base, selection, True)
        return execution.encode_checkpoint(base, config, output, run / 'train-U-192', queries, documents)
    require(phase in TRAIN_PHASES, 'unknown phase')
    end = int(phase.split('-')[-1])
    reference = base / 'NG-0071/pilot-step0-v1' if end == 96 else witness
    require(execution.sealed(reference)['eligibility_gate_passed'], 'witness eligibility changed')
    records = {r['query_id']: r for r in pipeline.rows(reference / 'witnesses.jsonl')}
    queries = observation.surface_queries(base, selection, False)
    require(not set(records).intersection(selection['sentinel']), 'sentinel gradient leakage')
    return execution.train_chunk(base, config, output, 'U', end - 96, end, records,
                                  execution.read(run / 'training-order.json'),
                                  {q['query_id']: q for q in queries}, documents, task,
                                  previous=run / 'train-U-96' if end == 192 else None)


def worker(args):
    import torch
    from clearml import Task
    verify(args.research_root, args.run)
    if args.phase != 'train-preflight':
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(71001)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    np.random.seed(71001)
    Task.set_offline(True)
    task = Task.init(project_name='Evoke-NG', task_name='NG-0074/' + args.phase,
                     reuse_last_task_id=False, auto_connect_frameworks=False,
                     auto_connect_arg_parser=False, auto_connect_streams=False,
                     auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    io.write(output / 'clearml-start.json', receipt)
    try:
        config = execution.read(args.run / 'config.json')
        task.connect(dict(config=config, input_sha256=io.sha(args.run / 'inputs.json')))
        result = dispatch(args, config, task)
        verify(args.research_root, args.run)
        io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException:
        receipt['outcome'] = 'failed'
        raise
    finally:
        task.close()
        receipt['closed'] = True
        io.write(output / 'clearml.json', receipt)


def supervise(args):
    import fcntl
    verify(args.research_root, args.run)
    stopped = []
    handlers = {s: signal.signal(s, lambda number, frame: stopped.append(number))
                for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (args.research_root / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            memory, utilization = pipeline.gpu_state(args.gpu)
            require(memory < 100 and utilization == 0, 'GPU occupied; do not evict')
            io.write(args.run / 'controller-start.json', dict(
                pid=os.getpid(), gpu=args.gpu, started_unix=time.time()))
            completed = []
            args.worker_script = Path(__file__).name
            try:
                for phase in PHASES:
                    print('NG74_PHASE_START', phase, flush=True)
                    pipeline.phase(args, phase, stopped)
                    completed.append(phase)
                    print('NG74_PHASE_COMPLETE', phase, flush=True)
                io.write(args.run / 'controller-exit.json', dict(
                    status='all_phases_complete', completed=completed, finished_unix=time.time()))
            except BaseException as exc:
                io.write(args.run / 'controller-exit.json', dict(
                    status='failed_preserve_attempt', completed=completed,
                    error=f'{type(exc).__name__}: {exc}', finished_unix=time.time()))
                raise
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'supervise', 'worker', 'verify'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3))
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    require(args.run.parent == args.research_root / 'NG-0074', 'new NG74 path required')
    if args.mode == 'freeze':
        freeze(args)
    else:
        verify(args.research_root, args.run)
        require(io.sha(Path(__file__)) == execution.read(args.run / 'inputs.json')['source'][Path(__file__).name],
                'executing controller is not frozen source')
        if args.mode == 'supervise':
            require(args.gpu is not None, 'explicit physical GPU required')
            supervise(args)
        elif args.mode == 'worker':
            require(args.phase is not None, 'explicit phase required')
            worker(args)


if __name__ == '__main__':
    main()
