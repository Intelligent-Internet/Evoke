"""Immutable NG87 first-seed campaign; bounded, sequential and non-preemptive."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from ng85_data import read, require, rows, sha, write
from ng85_diagnose import PARENT_SHA
from ng85_run import closed, verify as verify_parent
from ng86_intervention import select_panel


PHASES = ('preflight-A', 'preflight-B', 'profile-base', 'train-A',
          'profile-A', 'train-B', 'profile-B')
CAPS = {'preflight': 1800, 'train': 12 * 3600, 'profile': 3600}
EVIDENCE_SHA = '10ae379dc050a3783861d0ae9fc4d80c5e09d5174af2f7655c0210dd4e2eb33b'


def panels(queries):
    monitor = select_panel(queries)
    validation = [i for i, q in enumerate(queries) if q['split'] == 'validation']
    require(len(validation) == 2048 and all(
        sum(queries[i]['subset'] == domain for i in validation) == 512
        for domain in ('fever', 'hotpotqa', 'nq', 'fiqa')), 'validation inventory')
    endpoint = [i for i in monitor if queries[i]['split'] == 'fit'] + validation
    require(len(endpoint) == len(set(endpoint)) == 2176
            and set(monitor) <= set(endpoint), 'endpoint panel identity')
    require(len({q['query_id'] for q in queries}) == len(queries), 'duplicate query ID')
    return monitor, endpoint


def check_inputs(args):
    require(args.parent == args.base / 'NG-0085/student-ab-v4'
            and sha(args.parent / 'inputs.json') == PARENT_SHA, 'canonical sealed parent required')
    verify_parent(args.base, args.parent)
    phases = ('prepare', 'bank', *(f'rankt5-{i}' for i in range(4)))
    for phase in phases:
        closed(args.parent, phase)
    evidence = args.base / 'NG-0086/evaluation-v2'
    require(sha(evidence / 'inputs.json') == EVIDENCE_SHA, 'NG86 evidence identity')
    for phase in ('profile-control', 'review'):
        closed(evidence, phase)
    return {str(path): sha(path) for path in (
        args.parent / 'inputs.json', args.parent / 'config.json',
        *(args.parent / p / 'complete.json' for p in phases),
        evidence / 'inputs.json', evidence / 'profile-control/complete.json',
        evidence / 'review/complete.json')}


def freeze(args):
    require(args.out.parent == args.base / 'NG-0087' and not args.out.exists(),
            'new immutable NG87 unit required')
    dependencies = check_inputs(args)
    monitor, endpoint = panels(rows(args.parent / 'prepare/queries.jsonl'))
    old = read(args.parent / 'config.json')
    args.out.mkdir(parents=True)
    for name in read(args.parent / 'inputs.json')['source']:
        if name.endswith('.py'):
            shutil.copy2(args.parent / name, args.out / name)
    source = Path(__file__).parent
    for name in ('ng85_diagnose.py', 'ng86_intervention.py', 'ng86_evaluation.py',
                 'review_ng86_intervention.py', 'ng87_run.py', 'ng87_train.py',
                 'review_ng87.py', 'test_ng87.py', 'ng0087-protocol.md'):
        shutil.copy2(source / name, args.out / name)
    # Only these two verified NG71 sections are consumed; do not carry stale
    # NG71 disabled-training, arm, gate or resource declarations into NG87.
    write(args.out / 'config.json', dict(experiment='NG-0087', training_enabled=True,
        base=old['base'], optimizer=old['optimizer']))
    write(args.out / 'inputs.json', dict(protocol='NG87_clean_base_fullstd_AB_v1',
        parent=str(args.parent), phases=list(PHASES), caps=CAPS,
        dependencies=dependencies, monitor_panel=monitor, endpoint_panel=endpoint,
        source={p.name: sha(p) for p in args.out.iterdir() if p.is_file()},
        seed=85001, epochs=2, updates=8192, fit_queries=16384,
        primary_step=8192, saved_steps=[2048, 4096, 8192],
        arms={'A': 'PPLX', 'B': 'RankT5'}, loss='full_std_gradient',
        fresh_optimizer=True, cost_regularizer=0., rms_unchanged=True,
        independent_validation=False, locked_test_access=False,
        automatic_research_expansion=False, production_deployment=False))
    verify(args)


def verify(args):
    m = read(args.out / 'inputs.json')
    require(m['protocol'] == 'NG87_clean_base_fullstd_AB_v1'
            and m['parent'] == str(args.parent) and m['phases'] == list(PHASES)
            and m['caps'] == CAPS and m['seed'] == 85001 and m['updates'] == 8192
            and m['epochs'] == 2 and m['fit_queries'] == 16384
            and not m['independent_validation']
            and m['primary_step'] == 8192 and m['saved_steps'] == [2048, 4096, 8192]
            and m['arms'] == {'A': 'PPLX', 'B': 'RankT5'}
            and m['loss'] == 'full_std_gradient' and m['fresh_optimizer']
            and m['cost_regularizer'] == 0. and m['rms_unchanged']
            and not m['locked_test_access'] and not m['automatic_research_expansion']
            and not m['production_deployment'], 'NG87 contract changed')
    for name, digest in m['source'].items():
        require(Path(name).name == name and sha(args.out / name) == digest,
                'changed frozen source: ' + name)
    require(check_inputs(args) == m['dependencies'], 'changed parent receipts')
    monitor, endpoint = panels(rows(args.parent / 'prepare/queries.jsonl'))
    require(monitor == m['monitor_panel'] and endpoint == m['endpoint_panel'], 'changed panels')
    return m


def prerequisites(args):
    require(args.phase in PHASES, 'registered phase required')
    for phase in PHASES[:PHASES.index(args.phase)]:
        closed(args.out, phase)


def worker(args):
    import torch
    from clearml import Task
    from ng85_train import load_targets
    from ng87_train import execute, profile

    m = verify(args)
    prerequisites(args)
    require(torch.cuda.device_count() == 1, 'one explicit GPU required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(m['seed'])
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0087/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, offline=True, remote_synced=False, closed=False)
    folder = args.out / args.phase
    write(folder / 'clearml-start.json', receipt)
    try:
        task.connect(dict(inputs_sha256=sha(args.out / 'inputs.json'), phase=args.phase))
        queries = rows(args.parent / 'prepare/queries.jsonl')
        documents = rows(args.parent / 'prepare/documents.jsonl')
        with np.load(args.parent / 'bank/bank.npz') as data:
            bank = {key: data[key] for key in data.files}
        config = read(args.out / 'config.json')
        if args.phase.startswith('profile-'):
            profile(args, m, config, queries, documents, bank, task)
        else:
            execute(args, config, queries, documents, bank,
                    load_targets(args.parent, queries, bank), task)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        write(folder / 'clearml.json', receipt)


def supervise(args):
    import fcntl
    import psutil
    from ng85_queue import check_resources, gpu_snapshot

    verify(args)
    prerequisites(args)
    require(args.gpu in range(4), 'explicit single GPU required')
    with (args.base / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_resources(args.out, args.phase, args.gpu)
        folder = args.out / args.phase
        folder.mkdir()
        cap = CAPS[args.phase.split('-')[0]]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
            PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', CLEARML_OFFLINE_MODE='1',
            CLEARML_CACHE_DIR=str(folder / 'tracking-cache'))
        process, error, peak, group_closed = None, None, 0, False
        stopped = []
        handlers = {s: signal.signal(s, lambda v, frame: stopped.append(v))
                    for s in (signal.SIGTERM, signal.SIGINT)}
        begin = time.monotonic()
        try:
            with (folder / 'stdout.log').open('x') as log:
                process = subprocess.Popen(command(args, 'worker'), env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                write(folder / 'started.json', dict(pid=process.pid, gpu=args.gpu,
                    started_unix=time.time(), cap=cap, inputs_sha256=sha(args.out / 'inputs.json')))
                while process.poll() is None:
                    require(not stopped, 'stop requested')
                    try:
                        tree = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in [tree, *tree.children(recursive=True)])
                    except psutil.NoSuchProcess:
                        rss = 0
                    peak = max(peak, rss)
                    require(rss < 24 * 1024 ** 3, 'owned RSS limit')
                    require(time.monotonic() - begin < cap, 'phase wall limit')
                    require(psutil.virtual_memory().available > 20 * 1024 ** 3, 'host memory floor')
                    require(shutil.disk_usage(args.out).free > 40 * 1024 ** 3, 'disk floor')
                    require(gpu_snapshot(args.gpu)['memory_mib'] < 22 * 1024, 'GPU memory limit')
                    time.sleep(5)
                require(process.returncode == 0, 'worker failed')
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            if process is not None:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    group_closed = True
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            receipt = read(folder / 'clearml.json') if (folder / 'clearml.json').exists() else {}
            passed = bool(error is None and group_closed and receipt.get('closed')
                          and receipt.get('outcome') == 'passed')
            write(folder / 'exit.json', dict(passed=passed, error=error,
                exit_code=process.returncode if process else None, owned_group_closed=group_closed,
                peak_tree_rss_bytes=peak, seconds=time.monotonic() - begin, finished_unix=time.time()))
            write(folder / 'complete.json', dict(passed=passed,
                files={str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()}))
        require(passed, 'failed immutable attempt; no automatic retry')


def command(args, action):
    result = [sys.executable, '-B', str(args.out / 'ng87_run.py'), action,
              '--base', str(args.base), '--parent', str(args.parent), '--out', str(args.out)]
    if args.phase is not None:
        result += ['--phase', args.phase]
    if args.gpu is not None:
        result += ['--gpu', str(args.gpu)]
    return result


def campaign(args):
    verify(args)
    require(not (args.out / 'campaign-start.json').exists(), 'campaign already started')
    write(args.out / 'campaign-start.json', dict(unix=time.time(), gpu=args.gpu,
        phases=PHASES, retries=False, inputs_sha256=sha(args.out / 'inputs.json')))
    completed, error, current = [], None, None
    stopped = []

    def stop(signum, frame):
        stopped.append(signum)
        if current is not None and current.poll() is None:
            current.terminate()

    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for phase in PHASES:
            require(not stopped, 'campaign stop requested')
            args.phase = phase
            current = subprocess.Popen(command(args, 'supervise'))
            require(current.wait() == 0 and not stopped, 'phase stopped or failed: ' + phase)
            current = None
            closed(args.out, phase)
            completed.append(phase)
        from review_ng87 import review
        review(args.out)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if current is not None and current.poll() is None:
            current.terminate()
            current.wait(timeout=60)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        write(args.out / 'campaign-exit.json', dict(passed=error is None,
            completed=completed, error=error, finished_unix=time.time(),
            automatic_training_continuation=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'verify', 'worker', 'supervise', 'campaign'))
    for name in ('base', 'parent', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--gpu', type=int, choices=range(4))
    args = parser.parse_args()
    if args.action in ('worker', 'supervise') and args.phase is None:
        parser.error('--phase required')
    if args.action in ('supervise', 'campaign') and args.gpu is None:
        parser.error('--gpu required')
    globals()[args.action](args)


if __name__ == '__main__':
    main()
