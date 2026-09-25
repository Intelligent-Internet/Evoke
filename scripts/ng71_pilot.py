#!/usr/bin/env python3
"""Frozen, fail-closed NG71 pilot; no retries, test access or model selection."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

sys.dont_write_bytecode = True

import ng71_preflight as io


PROOFS = {
    'NG-0071/pilot-step0-v1/complete.json':
        '5a736aedeff4270bec4e070640725d67236db52cdd7f82fc7684dcdc7ae5fc96',
    'NG-0071/cuda-model-preflight-v1/complete.json':
        '8688c029c056cba220d40f8d034674c601076fd9b5d3ee260d7d17ac7dbf9b50',
}
STEP0_REVIEW = 'c3b847295d6f32596c5863e122c6ee2691bd82d6c0f40f6eaf631d8cee445629'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def schedule():
    phases = ['train-A-96', 'encode-A-96', 'snapshot-96', 'train-A-192']
    phases += [f'train-{arm}-{step}' for arm in 'BCD' for step in (96, 192)]
    phases += ['rank-initial', 'rank-dense', 'rank-bm25', 'rank-A-96']
    phases += [f'{action}-{arm}-96' for arm in 'BCD' for action in ('encode', 'rank')]
    phases += [f'{action}-{arm}-192' for arm in 'ABCD' for action in ('encode', 'rank')]
    return phases + ['review']


def verify(base, run):
    frozen = read(run / 'inputs.json')
    if frozen['phases'] != schedule() or frozen['locked_test_access'] is not False:
        raise ValueError('frozen execution graph changed')
    for root, field in ((run, 'source'), (base, 'dependencies')):
        for name, digest in frozen[field].items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()) or io.sha(path) != digest:
                raise ValueError('frozen input changed: ' + name)
    core, config = read(run / 'preparation-config.json'), read(run / 'config.json')
    expected = dict(core, training_enabled=True, status='frozen_four_arm_pilot')
    if core['training_enabled'] is not False or config != expected:
        raise ValueError('scientific configuration changed beyond execution enablement')
    return frozen


def freeze(args):
    import ng71_data as data
    import ng71_execution as execution
    import ng71_snapshot as snapshot

    core = read(args.config)
    if (core['training_enabled'] is not False or core['schema_version'] != 2
            or io.sha(args.step0_review) != STEP0_REVIEW):
        raise ValueError('preparation configuration or independent review changed')
    for name, digest in PROOFS.items():
        if io.sha(args.research_root / name) != digest:
            raise ValueError('accepted preparation proof changed')
        proof = execution.sealed((args.research_root / name).parent)
        if proof['passed'] is not True:
            raise ValueError('accepted preparation proof is not successful')
    step0 = execution.sealed(args.research_root / 'NG-0071/pilot-step0-v1')
    if not step0['eligibility_gate_passed'] or not read(args.step0_review)['passed']:
        raise ValueError('step-zero scientific eligibility was not established')
    prepared = data.audit(args.research_root, core)
    if (not prepared['eligibility_gate_passed'] or prepared['query_order'] != read(
            args.research_root / 'NG-0071/pilot-step0-v1/training-order.json')):
        raise ValueError('TRAIN selection/order no longer matches accepted step zero')
    dependencies = snapshot.dependencies(args.research_root, core)
    dependencies.update(prepared['inputs'])
    for name in PROOFS:
        parent = (args.research_root / name).parent
        dependencies[name] = PROOFS[name]
        for relative, digest in read(parent / 'complete.json')['files'].items():
            dependencies[str((parent / relative).relative_to(args.research_root))] = digest
    if args.run.parent != args.research_root / 'NG-0071':
        raise ValueError('a fresh NG71 research unit is required')
    args.run.mkdir(exist_ok=False)
    source = Path(__file__).resolve().parent
    names = sorted(p.name for p in source.glob('ng71_*.py'))
    names += ['audit_ng70_provenance.py', 'analyze_ng66_learning_curves.py', 'review_ng71_pilot.py']
    for name in names:
        shutil.copy2(source / name, args.run / name)
    shutil.copy2(args.config, args.run / 'preparation-config.json')
    shutil.copy2(args.step0_review, args.run / 'step0-review.json')
    for path in sorted(args.config.parent.glob('ng0071-*.md')):
        shutil.copy2(path, args.run / path.name)
    io.write(args.run / 'config.json', dict(core, training_enabled=True, status='frozen_four_arm_pilot'))
    io.write(args.run / 'selection.json', prepared['selection'])
    io.write(args.run / 'training-order.json', prepared['query_order'])
    tests = source.parent / 'tests'
    (args.run / 'test-source').mkdir()
    for path in sorted(tests.glob('*ng71*.py')):
        shutil.copy2(path, args.run / 'test-source' / path.name)
    io.write(args.run / 'inputs.json', dict(
        phases=schedule(), dependencies=dependencies,
        source={str(p.relative_to(args.run)): io.sha(p) for p in args.run.rglob('*') if p.is_file()},
        actual_training_started=False, locked_test_access=False,
        bootstrap_replicates=10000, bootstrap_seed=71071,
        terminal_surfaces=['TRAIN_PILOT', 'TRAIN_SENTINEL', 'DEV_NEW']))
    verify(args.research_root, args.run)
    print('NG71_FROZEN', io.sha(args.run / 'inputs.json'), flush=True)


def dispatch(args, config, task):
    import ng71_execution as execution
    import ng71_observation as observation
    import ng71_snapshot as snapshot

    run, base, phase = args.run, args.research_root, args.phase
    output = run / phase
    if phase == 'review':
        from review_ng71_pilot import review
        return review(base, run)
    if phase.startswith('rank-'):
        return observation.rank(base, config, run, output, phase.removeprefix('rank-'))
    if phase == 'snapshot-96':
        result = snapshot.build(base, config, output, surface='pilot', reference=run / 'encode-A-96')
        if not result['eligibility_gate_passed']:
            raise ValueError('A96 full TRAIN witness eligibility failed')
        return result
    documents = rows(base / 'NG-0067/data/documents.jsonl')
    selection = read(run / 'selection.json')
    _, arm, end = phase.split('-')
    end = int(end)
    if phase.startswith('encode-'):
        if phase != 'encode-A-96':
            observation.training_closed(run)
        queries = observation.surface_queries(base, selection, end == 192)
        encoder = execution.encode_reference if phase == 'encode-A-96' else execution.encode_checkpoint
        return encoder(base, config, output, run / f'train-{arm}-{end}', queries, documents)
    if not phase.startswith('train-'):
        raise ValueError('unrecognized phase')
    start = end - 96
    reference = base / 'NG-0071/pilot-step0-v1' if start == 0 else run / 'snapshot-96'
    proof = execution.sealed(reference)
    if not proof['eligibility_gate_passed']:
        raise ValueError('training reference gate failed')
    records = {r['query_id']: r for r in rows(reference / 'witnesses.jsonl')}
    queries = observation.surface_queries(base, selection, False)
    return execution.train_chunk(
        base, config, output, arm, start, end, records, read(run / 'training-order.json'),
        {q['query_id']: q for q in queries}, documents, task,
        previous=run / f'train-{arm}-96' if start else None)


def worker(args):
    import numpy as np
    import torch
    from clearml import Task

    verify(args.research_root, args.run)
    if io.sha(Path(__file__)) != read(args.run / 'inputs.json')['source']['ng71_pilot.py']:
        raise ValueError('executing controller differs from frozen source')
    config = read(args.run / 'config.json')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    np.random.seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='Evoke-NG', task_name='NG-0071/' + args.phase,
                     reuse_last_task_id=False, auto_connect_frameworks=False,
                     auto_connect_arg_parser=False, auto_connect_streams=False,
                     auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, actual_start=True, post_hoc=False, offline=True,
                   remote_synced=False, closed=False,
                   reason='online authentication not yet qualified; explicit offline tracking')
    output = args.run / args.phase
    io.write(output / 'clearml-start.json', receipt)
    task.connect(dict(config=config, frozen_input_sha256=io.sha(args.run / 'inputs.json')))
    try:
        result = dispatch(args, config, task)
        verify(args.research_root, args.run)
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


def gpu_state(gpu):
    value = subprocess.check_output([
        'nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    memory, utilization = (int(v.strip()) for v in value.strip().split(','))
    return memory, utilization


def terminate_group(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
    process.wait()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(.1)
    return False


def phase(args, name, stopped, *, cuda=None, limit_seconds=5400, rss_limit_gib=16):
    if type(limit_seconds) is not int or not 0 < limit_seconds <= 5400:
        raise ValueError('phase timeout must not relax the existing maximum')
    if cuda is not None and type(cuda) is not bool:
        raise ValueError('explicit CUDA phase flag must be boolean')
    if type(rss_limit_gib) is not int or not 0 < rss_limit_gib <= 16:
        raise ValueError('RSS limit must not relax the existing maximum')
    if stopped:
        raise ValueError('controller stopped before phase launch')
    if (psutil.virtual_memory().available <= 24 * 1024 ** 3
            or shutil.disk_usage(args.run).free <= 40 * 1024 ** 3):
        raise ValueError('insufficient host memory or artifact space')
    output = args.run / name
    output.mkdir(exist_ok=False)
    cuda = name.startswith(('train-', 'encode-')) if cuda is None else cuda
    if cuda and gpu_state(args.gpu)[0] >= 100:
        raise ValueError('reserved GPU was occupied between phases; do not evict')
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('CLEARML_TASK_', 'TRAINS_TASK_', 'CLEARML_PROC_', 'TRAINS_PROC_'))}
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4',
               MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CLEARML_OFFLINE_MODE='1',
               CLEARML_CACHE_DIR=str(output / 'tracking-cache'),
               CUDA_VISIBLE_DEVICES=str(args.gpu) if cuda else '')
    peak, error, closed = 0, None, False
    started = time.monotonic()
    with (output / 'stdout.log').open('x') as log:
        process = subprocess.Popen([
            sys.executable, '-B', str(args.run / getattr(args, 'worker_script', 'ng71_pilot.py')), 'worker',
            '--research-root', str(args.research_root), '--run', str(args.run), '--phase', name],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            io.write(output / 'started.json', dict(
                worker_pid=process.pid, controller_pid=os.getpid(), started_unix=time.time(),
                physical_gpu=args.gpu if cuda else None, limit_seconds=limit_seconds,
                rss_limit_bytes=rss_limit_gib * 1024 ** 3))
            while process.poll() is None:
                if stopped:
                    raise ValueError('controller stop requested')
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)])
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    rss = 0
                if rss > rss_limit_gib * 1024 ** 3 or time.monotonic() - started > limit_seconds:
                    raise ValueError('worker exceeded RSS or phase time bound')
                if (psutil.virtual_memory().available <= 24 * 1024 ** 3
                        or shutil.disk_usage(args.run).free <= 40 * 1024 ** 3):
                    raise ValueError('host memory or artifact space crossed safety floor')
                if cuda and gpu_state(args.gpu)[0] > 20 * 1024:
                    raise ValueError('selected GPU total use exceeded 20 GiB')
                time.sleep(.5)
            if stopped:
                raise ValueError('controller stop requested at phase completion')
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            closed = terminate_group(process)
    io.write(output / 'exit.json', dict(
        exit_code=process.returncode, error=error, owned_group_closed=closed,
        finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - started, peak_tree_rss_bytes=peak,
        scientific_training_bounds_relaxed=False))
    tracking = read(output / 'clearml.json') if (output / 'clearml.json').exists() else {}
    passed = (process.returncode == 0 and error is None and closed
              and tracking.get('closed') is True and tracking.get('outcome') == 'passed')
    io.write(output / 'complete.json', dict(
        passed=passed, files={str(p.relative_to(output)): io.sha(p)
                              for p in sorted(output.rglob('*')) if p.is_file()}))
    if not passed:
        raise ValueError('phase failed; preserve attempt without retry: ' + name)


def supervise(args):
    verify(args.research_root, args.run)
    if io.sha(Path(__file__)) != read(args.run / 'inputs.json')['source']['ng71_pilot.py']:
        raise ValueError('executing controller differs from frozen source')
    supervise_graph(args, schedule(), 'ng71_pilot.py')


def supervise_graph(args, phases, worker_script, *, limit_seconds=5400):
    """Execute a caller-verified frozen graph using the existing resource guard."""
    import fcntl

    if (not phases or len(set(phases)) != len(phases)
            or any(not isinstance(n, str) or not n.replace('-', '').isalnum() for n in phases)):
        raise ValueError('phase graph must contain safe, unique names')
    if (Path(worker_script).name != worker_script
            or worker_script not in read(args.run / 'inputs.json')['source']):
        raise ValueError('worker must be a frozen source file')
    args.worker_script = worker_script
    stopped = []
    previous = {sig: signal.signal(sig, lambda value, frame: stopped.append(value))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    with (args.research_root / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory, utilization = gpu_state(args.gpu)
        if memory >= 100 or utilization != 0:
            raise ValueError('requested GPU is occupied; do not evict other jobs')
        io.write(args.run / 'controller-start.json', dict(
            pid=os.getpid(), gpu=args.gpu, started_unix=time.time()))
        completed = []
        try:
            for name in phases:
                print('NG71_PHASE_START', name, flush=True)
                phase(args, name, stopped, limit_seconds=limit_seconds)
                completed.append(name)
                print('NG71_PHASE_COMPLETE', name, flush=True)
            io.write(args.run / 'controller-exit.json', dict(
                status='all_phases_complete', completed=completed, finished_unix=time.time()))
        except BaseException as exc:
            io.write(args.run / 'controller-exit.json', dict(
                status='failed_preserve_attempt', completed=completed,
                error=f'{type(exc).__name__}: {exc}', finished_unix=time.time()))
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'supervise', 'worker'))
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--step0-review', type=Path)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3))
    parser.add_argument('--phase', choices=schedule())
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    if args.mode == 'freeze':
        if args.config is None or args.step0_review is None:
            parser.error('freeze requires --config and --step0-review')
        freeze(args)
    elif args.mode == 'supervise':
        if args.gpu is None:
            parser.error('supervise requires an explicit physical --gpu')
        supervise(args)
    else:
        if args.phase is None:
            parser.error('worker requires --phase')
        worker(args)


if __name__ == '__main__':
    main()
