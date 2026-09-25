"""Bounded replay of the original D trajectory, with no new objective."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from scipy import sparse

import ng71_execution as execution
import ng71_observation as observation
import ng71_pilot as pipeline
import ng71_preflight as io
import ng76_trajectory_observer as trace


PREPARATION = 'NG-0076/preparation-v1'
PREP_SHA = '28f676cb3f858094a097f74c772b7ee509c10fcdb1aa3e79a113f907f197d80d'
PROCEDURE_SHA = '78e8b143024dc2aab54d69c305374bff6e65537154992581e18097491a51e94e'
PARENT = 'NG-0071/pilot-v1'
PHASES = ('replay-D-96', 'replay-D-192', 'review')
require = trace.require


def verify_files(root, files):
    for name, expected in files.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and io.sha(path) == expected,
                'frozen input changed: ' + name)


def verify(base, run):
    manifest = execution.read(run / 'inputs.json')
    require(manifest['protocol'] == 'NG76_historical_trajectory_v1'
            and manifest['phases'] == list(PHASES)
            and manifest['historical_optimizer_updates'] == 192
            and manifest['scientific_training_updates'] == 0
            and manifest['locked_test_access'] is False, 'execution boundary changed')
    verify_files(base, manifest['dependencies'])
    verify_files(run, manifest['source'])
    require(io.sha(base / PREPARATION / 'inputs.json') == PREP_SHA, 'preparation changed')
    require(execution.read(run / 'config.json') == execution.read(base / PARENT / 'config.json'),
            'original training config changed')
    trace.validate_payload(execution.read(run / 'prepared.json'))
    return manifest


def freeze(base, run):
    parent = base / PREPARATION
    require(io.sha(parent / 'inputs.json') == PREP_SHA, 'wrong preparation')
    manifest = execution.read(parent / 'inputs.json')
    verify_files(base, manifest['dependencies'])
    verify_files(parent, manifest['source'])
    procedure = base / 'NG-0076/preparation-v1-procedure'
    require(io.sha(procedure / 'complete.json') == PROCEDURE_SHA, 'wrong preparation receipt')
    complete = execution.read(procedure / 'complete.json')
    require(complete['passed'] is True, 'preparation failed')
    verify_files(procedure, complete['files'])
    dependencies = dict(manifest['dependencies'])
    for folder in (parent, procedure, *(base / PARENT / f'train-D-{end}' for end in (96, 192))):
        if folder.name.startswith('train-'):
            execution.sealed(folder)
        dependencies.update({str(p.relative_to(base)): io.sha(p)
                             for p in folder.rglob('*') if p.is_file()})
    run.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    names = [p.name for p in source.glob('ng71_*.py')]
    names += ['audit_ng70_provenance.py', 'analyze_ng66_learning_curves.py', 'review_ng71_pilot.py',
              'ng74_uniform_control.py', 'review_ng74_uniform_control.py', 'ng75_update_diagnosis.py',
              'ng75_update_math.py', 'review_ng75_displacement.py', 'prepare_ng76_trajectory.py',
              'ng76_trajectory_observer.py', 'ng76_trajectory.py']
    for name in names:
        shutil.copy2(source / name, run / name)
    for p in (source.parent / 'tests').glob('*ng76*.py'):
        shutil.copy2(p, run / p.name)
    shutil.copy2(parent / 'prepared.json', run / 'prepared.json')
    shutil.copy2(base / PARENT / 'config.json', run / 'config.json')
    plan = source.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0076-endpoint-union-trajectory-plan.zh.md'
    shutil.copy2(plan, run / plan.name)
    io.write(run / 'inputs.json', dict(protocol='NG76_historical_trajectory_v1', phases=list(PHASES),
        dependencies=dependencies, source={p.name: io.sha(p) for p in run.iterdir() if p.is_file()},
        historical_optimizer_updates=192, scientific_training_updates=0, locked_test_access=False))
    verify(base, run)
    print('NG76_FROZEN', io.sha(run / 'inputs.json'), flush=True)


def replay(base, run, end, task):
    parent = base / PARENT
    config = execution.read(run / 'config.json')
    data = execution.read(run / 'prepared.json')
    start = end - 96
    if start:
        prior = execution.sealed(run / 'replay-D-96')
        require(prior['historical_endpoint_exact'] is True, 'first replay not qualified')
    original = execution.sealed(parent / f'train-D-{end}')
    reference = parent / 'snapshot-96' if start else base / 'NG-0071/pilot-step0-v1'
    require(execution.sealed(reference)['eligibility_gate_passed'], 'original witnesses not qualified')
    records = {r['query_id']: r for r in pipeline.rows(reference / 'witnesses.jsonl')}
    wanted = {d for row in records.values() for d in row['pool']}
    documents = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                documents[i] = json.loads(line)
    require(i + 1 == 233009 and set(documents) == wanted, 'canonical training documents incomplete')
    selection = execution.read(parent / 'selection.json')
    queries = observation.surface_queries(base, selection, False)
    require(not set(records).intersection(data['query_ids']), 'sentinel must never supply loss')
    output = run / f'replay-D-{end}'
    observer = trace.Observer(data, output, pipeline.rows(parent / f'train-D-{end}/progress.jsonl'),
                              start, end, execution.read(output / 'started.json')['started_unix'])
    result = execution.train_chunk(base, config, output, 'D', start, end, records,
        execution.read(parent / 'training-order.json'), {q['query_id']: q for q in queries}, documents,
        task, previous=parent / 'train-D-96' if start else None, observer=observer)
    return observer.finish(result, original)


def reduce_path(data, scores):
    require(set(scores) == set(range(0, 193, 16)), 'missing trajectory checkpoint')
    steps = sorted(scores)
    values = {s: trace.margins(data, scores[s]) for s in steps}
    pairs, domains = [], {}
    for identity, initial in values[0].items():
        q, p, n = identity
        sequence = [values[s][identity] for s in steps]
        increments = np.diff(sequence)
        residual = abs(float(increments.sum()) - (sequence[-1] - initial))
        require(residual <= 1e-10 * max(1., abs(initial), abs(sequence[-1])), 'telescoping mismatch')
        ahead = [value > 0 or (value == 0 and p < n) for value in sequence]
        pairs.append(dict(query_id=q, domain=data['queries'][q]['subset'], positive_id=p, rival_id=n,
            margins=sequence, increments=increments.tolist(), ahead=ahead,
            endpoint_lost=ahead[0] and not ahead[-1], endpoint_gained=not ahead[0] and ahead[-1],
            lost_intervals=[steps[i + 1] for i in range(12) if ahead[i] and not ahead[i + 1]],
            gained_intervals=[steps[i + 1] for i in range(12) if not ahead[i] and ahead[i + 1]],
            telescoping_residual=residual))
    for domain in ('fever', 'hotpotqa', 'nq'):
        rows = [p for p in pairs if p['domain'] == domain]
        domains[domain] = dict(pairs=len(rows), lost=sum(p['endpoint_lost'] for p in rows),
            gained=sum(p['endpoint_gained'] for p in rows),
            lost_by_interval={str(s): sum(s in p['lost_intervals'] for p in rows) for s in steps[1:]},
            final_lost_by_last_loss_interval={str(s): sum(p['endpoint_lost'] and p['lost_intervals'][-1] == s
                for p in rows) for s in steps[1:]})
    return pairs, domains


def review(base, run):
    data = execution.read(run / 'prepared.json')
    scores, states = {}, {}
    for end in (96, 192):
        phase = run / f'replay-D-{end}'
        result = execution.sealed(phase)
        require(result['passed'] and result['historical_endpoint_exact']
                and result['historical_trace_matches'], 'historical replay not qualified')
        old = execution.sealed(base / PARENT / f'train-D-{end}')
        for key in ('model_sha256', 'optimizer_fingerprint'):
            require(result[key] == old[key], 'sealed endpoint differs')
        for step in range(end - 96, end + 1, 16):
            folder = phase / f'observe-{step:03d}'
            record = execution.read(folder / 'observation.json')
            verify_files(folder, record['files'])
            require(record['state_unchanged'] is True and record['step'] == step,
                    'observer invariance not established')
            actual = trace.pool_scores(data, sparse.load_npz(folder / 'query.npz'),
                                       sparse.load_npz(folder / 'document.npz'))
            require(actual == execution.read(folder / 'scores.json'), 'saved CSR score reduction differs')
            if step in scores:
                require(actual == scores[step], '96-boundary scores differ')
                for key in ('model_sha256', 'optimizer_sha256'):
                    require(states[step][key] == record['state'][key], '96-boundary state differs')
                for role in ('query', 'document'):
                    before = sparse.load_npz(run / 'replay-D-96/observe-096' / f'{role}.npz')
                    after = sparse.load_npz(folder / f'{role}.npz')
                    require(before.shape == after.shape and (before != after).nnz == 0,
                            '96-boundary codes differ')
            scores[step], states[step] = actual, record['state']
    for step, endpoint in ((0, 'initial'), (192, 'D-192')):
        for q in data['query_ids']:
            trace.endpoint_rank(data['pools'][q], scores[step][q], data['endpoint_rankings'][endpoint][q])
    pairs, domains = reduce_path(data, scores)
    output = run / 'review'
    with (output / 'pairs.jsonl').open('x') as stream:
        for row in pairs:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    io.write(output / 'scores.json', scores)
    return dict(passed=True, observations=13, pairs=len(pairs), domains=domains,
                historical_optimizer_updates=192, scientific_training_updates=0,
                endpoint_full_corpus_top100_parity=True, intermediate_full_corpus_quality=False,
                locked_test_scored=False, causal_domain_attribution=False, overall_goal_qualified=False)


def worker(args):
    import torch
    from clearml import Task
    verify(args.research_root, args.run)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    np.random.seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0076/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    try:
        io.write(output / 'clearml-start.json', receipt)
        task.connect(dict(input_sha256=io.sha(args.run / 'inputs.json'), phase=args.phase))
        result = review(args.research_root, args.run) if args.phase == 'review' else replay(
            args.research_root, args.run, int(args.phase.split('-')[-1]), task)
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
    stopped, completed = [], []
    handlers = {s: signal.signal(s, lambda number, frame: stopped.append(number))
                for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (args.research_root / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            memory, utilization = pipeline.gpu_state(args.gpu)
            require(memory < 100 and utilization == 0, 'GPU occupied; do not evict')
            io.write(args.run / 'controller-start.json', dict(pid=os.getpid(), gpu=args.gpu, started_unix=time.time()))
            args.worker_script = Path(__file__).name
            try:
                for phase in PHASES:
                    print('NG76_PHASE_START', phase, flush=True)
                    pipeline.phase(args, phase, stopped, cuda=phase != 'review', limit_seconds=1800)
                    completed.append(phase)
                    print('NG76_PHASE_COMPLETE', phase, flush=True)
                io.write(args.run / 'controller-exit.json', dict(status='all_phases_complete',
                    completed=completed, finished_unix=time.time()))
            except BaseException as exc:
                io.write(args.run / 'controller-exit.json', dict(status='failed_preserve_attempt',
                    completed=completed, error=f'{type(exc).__name__}: {exc}', finished_unix=time.time()))
                raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3))
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    require(args.run.parent == args.research_root / 'NG-0076' and args.run.name != 'preparation-v1',
            'fresh NG76 execution path required')
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    else:
        verify(args.research_root, args.run)
        require(io.sha(Path(__file__)) == execution.read(args.run / 'inputs.json')['source'][Path(__file__).name],
                'controller is not frozen source')
        if args.mode == 'supervise':
            require(args.gpu is not None, 'explicit physical GPU required')
            supervise(args)
        elif args.mode == 'worker':
            require(args.phase is not None, 'explicit worker phase required')
            worker(args)


if __name__ == '__main__':
    main()
