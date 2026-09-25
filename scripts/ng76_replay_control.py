"""One disposable D96 control without observations, never a model promotion."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np

import audit_ng76_replay as audit
import ng71_execution as execution
import ng71_observation as observation
import ng71_pilot as pipeline
import ng71_preflight as io
import ng76_trajectory as trajectory


PARENT = 'NG-0076/historical-trajectory-v1'
PARENT_SHA = 'b53a8a6fea1710b260855626e9baced73ed7ddef81dfb15802a33c47b3135822'
INVENTORY = 'NG-0076/historical-trajectory-v1-remote-failed-final.json'
INVENTORY_SHA = 'b3f3a3cf288833b624a1e387912ec2b819636c9249fbaa72217b807515ef08cb'
PHASE = 'control-D-96'
require = audit.require


def verify(base, run):
    manifest = execution.read(run / 'inputs.json')
    require(manifest['protocol'] == 'NG76_no_observer_control_v1'
            and manifest['disposable_optimizer_updates'] == 96
            and manifest['observer_enabled'] is False
            and manifest['locked_test_access'] is False, 'control contract changed')
    trajectory.verify_files(base, manifest['dependencies'])
    trajectory.verify_files(run, manifest['source'])
    require(io.sha(base / PARENT / 'inputs.json') == PARENT_SHA
            and io.sha(base / INVENTORY) == INVENTORY_SHA, 'failed attempt identity changed')
    require(execution.read(run / 'config.json') == execution.read(base / PARENT / 'config.json'),
            'historical configuration changed')
    return manifest


def freeze(base, run):
    parent = base / PARENT
    require(io.sha(parent / 'inputs.json') == PARENT_SHA
            and io.sha(base / INVENTORY) == INVENTORY_SHA, 'wrong frozen failure')
    original = trajectory.verify(base, parent)
    inventory = execution.read(base / INVENTORY)
    trajectory.verify_files(parent, inventory['files'])
    require({str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file()}
            == set(inventory['files']), 'failed attempt inventory changed')
    require(execution.read(parent / 'controller-exit.json')['status'] == 'failed_preserve_attempt',
            'do not duplicate a running observer')
    run.mkdir(parents=True, exist_ok=False)
    for name in original['source']:
        shutil.copy2(parent / name, run / name)
    source = Path(__file__).resolve().parent
    for name in ('audit_ng76_replay.py', 'ng76_replay_control.py'):
        shutil.copy2(source / name, run / name)
    shutil.copy2(source.parent / 'tests/test_ng76_replay_control.py', run)
    shutil.copy2(source.parent / ('docs/research-sae/reports/ng0001-ng0099/'
                                 'ng0076-replay-control-plan.zh.md'), run)
    dependencies = dict(original['dependencies'])
    dependencies.update({f'{PARENT}/{name}': digest for name, digest in inventory['files'].items()})
    dependencies[INVENTORY] = INVENTORY_SHA
    io.write(run / 'inputs.json', dict(protocol='NG76_no_observer_control_v1',
        disposable_optimizer_updates=96, observer_enabled=False, locked_test_access=False,
        dependencies=dependencies, source={p.name: io.sha(p) for p in run.iterdir() if p.is_file()}))
    verify(base, run)


def summarize(result, historical, actual_trace, historical_trace):
    compared = audit.audit(actual_trace, historical_trace)
    require(compared['steps'] == 96 and compared['complete_trace'], 'control incomplete')
    keys = ('arm', 'cumulative_steps', 'initial_model_sha256', 'initial_optimizer_fingerprint',
            'tokens', 'candidate_pairs', 'query_exposures', 'model_sha256', 'optimizer_fingerprint')
    endpoint = {k: result[k] == historical[k] for k in keys}
    qualified = (all(endpoint.values()) and compared['numeric_tolerance_passed']
                 and not compared['identity_differences'] and not compared['readout_differences'])
    return dict(procedure_completed=True, historical_replay_qualified=qualified,
                endpoint_matches=endpoint, trace=compared, disposable_optimizer_updates=96,
                new_scientific_training_updates=0, overall_goal_qualified=False,
                quality_evaluation=False, locked_test_scored=False)


def run_control(base, run, task):
    parent = base / trajectory.PARENT
    config = execution.read(run / 'config.json')
    records = {r['query_id']: r for r in pipeline.rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')}
    wanted = {d for row in records.values() for d in row['pool']}
    documents = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                documents[i] = json.loads(line)
    require(i + 1 == 233009 and set(documents) == wanted, 'canonical TRAIN documents incomplete')
    queries = observation.surface_queries(base, execution.read(parent / 'selection.json'), False)
    output = run / PHASE
    # Explicit None is the treatment: no state hashes, CSR or sentinel inference in the loop.
    result = execution.train_chunk(base, config, output, 'D', 0, 96, records,
        execution.read(parent / 'training-order.json'), {q['query_id']: q for q in queries},
        documents, task, observer=None)
    comparison = summarize(result, execution.sealed(parent / 'train-D-96'),
        pipeline.rows(output / 'progress.jsonl'), pipeline.rows(parent / 'train-D-96/progress.jsonl'))
    io.write(output / 'comparison.json', comparison)
    return dict(result, **{k: v for k, v in comparison.items() if k != 'trace'})


def worker(args):
    import torch
    import transformers
    from clearml import Task
    verify(args.research_root, args.run)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    np.random.seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.run / PHASE
    io.write(output / 'runtime.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        transformers=transformers.__version__, device=torch.cuda.get_device_name(0),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        tf32=False, observer_enabled=False, computational_contract_changed=False))
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0076/no-observer-D96-control',
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    try:
        io.write(output / 'clearml-start.json', receipt)
        task.connect(dict(input_sha256=io.sha(args.run / 'inputs.json'), diagnostic_only=True))
        result = run_control(args.research_root, args.run, task)
        verify(args.research_root, args.run)
        io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'  # Procedure success, not historical or scientific qualification.
    except BaseException:
        receipt['outcome'] = 'failed'
        raise
    finally:
        task.close()
        receipt['closed'] = True
        io.write(output / 'clearml.json', receipt)


def supervise(args):
    stopped = []
    handlers = {s: signal.signal(s, lambda number, frame: stopped.append(number))
                for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (args.research_root / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            memory, utilization = pipeline.gpu_state(args.gpu)
            require(memory < 100 and utilization == 0, 'GPU occupied; do not evict')
            io.write(args.run / 'controller-start.json', dict(pid=os.getpid(), gpu=args.gpu,
                                                            started_unix=time.time()))
            args.worker_script = Path(__file__).name
            try:
                pipeline.phase(args, PHASE, stopped, cuda=True, limit_seconds=1800)
                io.write(args.run / 'controller-exit.json', dict(status='diagnostic_procedure_complete',
                    finished_unix=time.time(), no_automatic_followup=True))
            except BaseException as exc:
                io.write(args.run / 'controller-exit.json', dict(status='failed_preserve_attempt',
                    error=f'{type(exc).__name__}: {exc}', finished_unix=time.time()))
                raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=(PHASE,))
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3))
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    require(args.run.parent == args.research_root / 'NG-0076'
            and args.run.name == 'no-observer-control-v1', 'fresh dedicated control path required')
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
        return
    manifest = verify(args.research_root, args.run)
    require(io.sha(Path(__file__)) == manifest['source'][Path(__file__).name], 'source not frozen')
    if args.mode == 'worker':
        require(args.phase == PHASE, 'explicit control phase required')
        worker(args)
    elif args.mode == 'supervise':
        require(args.gpu is not None, 'explicit physical GPU required')
        supervise(args)


if __name__ == '__main__':
    main()
