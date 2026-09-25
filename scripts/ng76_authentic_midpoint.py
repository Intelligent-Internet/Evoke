"""Encode only 26 queries at original D96; reuse frozen full-corpus documents."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from scipy import sparse

import ng71_execution as execution
import ng71_pilot as pipeline
import ng71_preflight as io
import ng71_training as training
import ng76_replay_control as parent_control
import ng76_trajectory as trajectory
import ng76_trajectory_observer as observer
import ng76_midpoint_math as math


PARENT = 'NG-0076/no-observer-control-v1'
PARENT_SHA = '5831092424a3f68c8bb309322ef0bb94d04729209aaf9d8d3fd9d9bb4f79671d'
MODEL_SHA = '92e574071092f08f6288a980ac27a1f0e17b82b0274acac40d3cf8cc228abd49'
ENCODE = 'NG-0071/pilot-v1/encode-D-96'
ENCODE_SHA = 'cd9ecb1f0349e9d0f3b3744dcda82c56da50048b5403e3c9d8003ba52b4df4ea'
CHECKPOINT = 'NG-0071/pilot-v1/train-D-96'
LEXICAL = 'NG-0069/lexical-v1'
PHASES = ('encode-query', 'rank')
PARITY = dict(rtol=2e-5, atol=2e-7, exact_support=True)
require = math.require


def query_contract(data, selected, controls):
    observer.validate_payload(data)
    expected = controls + data['query_ids']
    require(len(controls) == 2 and len(set(expected)) == 26
            and [r['query_id'] for r in selected] == expected, 'query identity/order changed')
    for row in selected:
        require(row['split'] == 'TRAIN' and hashlib.sha256(row['query'].encode()).hexdigest()
                == row['text_sha256'], 'forbidden split or changed text')
    for row in selected[2:]:
        require(row['query'] == data['queries'][row['query_id']]['query'], 'sentinel text mismatch')


def verify(base, run):
    m = execution.read(run / 'inputs.json')
    require(m['protocol'] == 'NG76_authentic_midpoint_v1' and m['phases'] == list(PHASES)
            and m['optimizer_updates'] == 0 and m['document_forwards'] == 0
            and m['query_forwards'] == 26 and m['locked_test_access'] is False
            and m['parity'] == PARITY, 'midpoint contract changed')
    trajectory.verify_files(base, m['dependencies'])
    trajectory.verify_files(run, m['source'])
    require(io.sha(base / ENCODE / 'complete.json') == ENCODE_SHA
            and io.sha(base / PARENT / 'inputs.json') == PARENT_SHA, 'parent identity changed')
    require(execution.read(run / 'config.json')
            == execution.read(base / 'NG-0071/pilot-v1/config.json'), 'original config changed')
    query_contract(execution.read(run / 'prepared.json'), execution.read(run / 'queries.json'), m['controls'])
    return m


def freeze(base, run):
    parent = base / PARENT
    require(io.sha(parent / 'inputs.json') == PARENT_SHA, 'wrong completed control')
    old = parent_control.verify(base, parent)
    require(execution.sealed(parent / 'control-D-96')['historical_replay_qualified'], 'control not qualified')
    require(io.sha(base / ENCODE / 'complete.json') == ENCODE_SHA, 'original encoding anchor changed')
    encoded, checkpoint = execution.sealed(base / ENCODE), execution.sealed(base / CHECKPOINT)
    require(encoded['model_sha256'] == checkpoint['model_sha256'] == MODEL_SHA
            and encoded['checkpoint_complete_sha256'] == io.sha(base / CHECKPOINT / 'complete.json')
            and encoded['counts']['document']['rows'] == 233009, 'document/model lineage mismatch')
    require(execution.read(base / ENCODE / 'clearml.json')['outcome'] == 'passed', 'encoding tracking failed')
    controls = execution.read(base / ENCODE / 'query-ids.json')[:2]
    data = execution.read(parent / 'prepared.json')
    raw = execution.read(base / LEXICAL / 'queries.json')
    by_id = {r['query_id']: r for r in raw}
    require(len(by_id) == len(raw) == 7872, 'canonical query mapping changed')
    selected = [dict(by_id[q], text_sha256=hashlib.sha256(by_id[q]['query'].encode()).hexdigest())
                for q in controls + data['query_ids']]
    query_contract(data, selected, controls)
    require(not set(data['query_ids']) & set(execution.read(base / ENCODE / 'query-ids.json')),
            'sentinel cache availability changed; reconsider unnecessary inference')
    run.mkdir(parents=True, exist_ok=False)
    for name in old['source']:
        shutil.copy2(parent / name, run / name)
    source = Path(__file__).resolve().parent
    for name in ('ng76_authentic_midpoint.py', 'ng76_midpoint_math.py',
                 'review_ng76_midpoint.py', 'ng71_pilot.py'):
        shutil.copy2(source / name, run / name)
    shutil.copy2(source.parent / 'tests/test_ng76_midpoint.py', run)
    shutil.copy2(source.parent / ('docs/research-sae/reports/ng0001-ng0099/'
                                 'ng0076-authentic-midpoint-plan.zh.md'), run)
    io.write(run / 'queries.json', selected)
    dependencies = dict(old['dependencies'])
    for folder in (base / ENCODE, parent):
        dependencies.update({str(p.relative_to(base)): io.sha(p) for p in folder.rglob('*') if p.is_file()})
    for name in ('queries.json', 'lexical-query.npz', 'lexical-document.npz'):
        dependencies[f'{LEXICAL}/{name}'] = io.sha(base / LEXICAL / name)
    io.write(run / 'inputs.json', dict(protocol='NG76_authentic_midpoint_v1', phases=list(PHASES),
        dependencies=dependencies, source={p.name: io.sha(p) for p in run.iterdir() if p.is_file()},
        controls=controls, query_forwards=26, document_forwards=0, optimizer_updates=0,
        locked_test_access=False, parity=PARITY))
    verify(base, run)


def compare_codes(actual, expected):
    require(actual.shape == expected.shape and actual.has_canonical_format and expected.has_canonical_format,
            'query code shape/format changed')
    require(all(np.isfinite(m.data).all() and (m.data > 0).all() for m in (actual, expected)),
            'invalid query control values')
    require(np.array_equal(actual.indptr, expected.indptr)
            and np.array_equal(actual.indices, expected.indices), 'query control support changed')
    np.testing.assert_allclose(actual.data, expected.data, rtol=PARITY['rtol'], atol=PARITY['atol'])
    return float(np.max(np.abs(actual.data - expected.data), initial=0))


def encode(base, run):
    import torch
    config, queries = execution.read(run / 'config.json'), execution.read(run / 'queries.json')
    encoder = training.Encoder(base, config, device='cuda', checkpoint=base / CHECKPOINT / 'checkpoint')
    require(training.parameter_hash(encoder.model) == MODEL_SHA, 'not original D96 model')
    require(not any(m.training for m in encoder.model.modules()), 'dropout/eval mode changed')
    before = observer.tree_hash((torch.get_rng_state(), torch.cuda.get_rng_state_all()))
    output, pieces = run / 'encode-query', []
    tick = time.monotonic()
    with torch.no_grad():
        for rows in (queries[:2], queries[2:]):
            values = encoder.encode([r['query'] for r in rows], 'query')
            require(values.dtype == torch.float32 and not values.requires_grad
                    and torch.isfinite(values).all() and (values >= 0).all(), 'invalid query readout')
            pieces.append(sparse.csr_matrix(values.cpu().numpy()))
            if len(pieces) == 1:
                error = compare_codes(pieces[0], sparse.load_npz(base / ENCODE / 'query.npz')[:2])
                forecast = (time.time() - execution.read(output / 'started.json')['started_unix']
                            + 13 * (time.monotonic() - tick) * 1.5 + 360)
                io.write(output / 'canary.json', dict(passed=forecast < 1740,
                    predicted_remaining_seconds=forecast, controls=2, max_abs_error=error))
                require(forecast < 1740, 'query canary exceeds bound')
    require(training.parameter_hash(encoder.model) == MODEL_SHA
            and all(p.grad is None for p in encoder.model.parameters()), 'forward changed model or gradients')
    require(before == observer.tree_hash((torch.get_rng_state(), torch.cuda.get_rng_state_all())),
            'query forward changed RNG')
    require(torch.cuda.max_memory_allocated() < 20 * 1024 ** 3, 'GPU allocation exceeded bound')
    result = sparse.vstack(pieces, format='csr')
    require(result.shape[0] == 26 and np.all(result.getnnz(axis=1) > 0), 'query coverage/readout incomplete')
    sparse.save_npz(output / 'query.npz', result)
    io.write(output / 'query-ids.json', [q['query_id'] for q in queries])
    return dict(passed=True, model_sha256=MODEL_SHA, query_forwards=26, document_forwards=0,
        optimizer_updates=0, parity_error=error, forward_seconds=time.monotonic() - tick,
        locked_test_scored=False, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())


def rank(base, run):
    require(execution.sealed(run / 'encode-query')['model_sha256'] == MODEL_SHA, 'query phase not qualified')
    data = execution.read(run / 'prepared.json')
    query = sparse.load_npz(run / 'encode-query/query.npz')[2:].astype(np.float64)
    document = sparse.load_npz(base / ENCODE / 'document.npz').astype(np.float64)
    require(document.shape[0] == 233009 and query.shape[0] == 24, 'corpus/query coverage changed')
    raw = execution.read(base / LEXICAL / 'queries.json')
    pos = {q['query_id']: i for i, q in enumerate(raw)}
    lexical_q = sparse.load_npz(base / LEXICAL / 'lexical-query.npz')[[pos[q] for q in data['query_ids']]]
    lexical_d = sparse.load_npz(base / LEXICAL / 'lexical-document.npz').astype(np.float64)
    require(lexical_d.shape[0] == document.shape[0], 'lexical corpus changed')
    output = run / 'rank'
    scores = np.lib.format.open_memmap(output / 'scores.npy', mode='w+', dtype=np.float64, shape=(24, 233009))
    fixed, rankings = {}, {}
    tick = time.monotonic()
    for i, q in enumerate(data['query_ids']):
        values = (query.getrow(i) @ document.T).toarray().ravel()
        lexical = (lexical_q.getrow(i).astype(np.float64) @ lexical_d.T).toarray().ravel()
        np.testing.assert_allclose(lexical[data['pools'][q]], data['lexical_scores'][q], rtol=0, atol=1e-12)
        scores[i] = values + lexical
        rankings[q] = math.rank(scores[i], data['gold'][q])
        fixed[q] = scores[i, data['pools'][q]].tolist()
        if i == 0:
            forecast = (time.time() - execution.read(output / 'started.json')['started_unix']
                        + (time.monotonic() - tick) * 24 * 1.5 + 180)
            io.write(output / 'canary.json', dict(passed=forecast < 840, predicted_remaining_seconds=forecast))
            require(forecast < 840, 'full rank canary exceeds bound')
        print('NG76_MIDPOINT_RANK', i + 1, time.monotonic() - tick, flush=True)
    scores.flush()
    io.write(output / 'rankings.json', rankings)
    io.write(output / 'fixed-scores.json', fixed)
    analysis = math.analyze(data, fixed, rankings)
    require(len(analysis['pairs']) == 5226, 'pair coverage changed')
    io.write(output / 'analysis.json', analysis)
    return dict(passed=True, queries=24, documents=233009, pairs=5226, domains=analysis['domains'],
        original_model_sha256=MODEL_SHA, optimizer_updates=0, document_forwards=0,
        diagnostic_TRAIN_only=True, heldout_quality_evaluation=False, overall_goal_qualified=False)


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
    task = Task.init(project_name='II42-NG', task_name='NG-0076/authentic-midpoint-' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    try:
        io.write(output / 'clearml-start.json', receipt)
        task.connect(dict(input_sha256=io.sha(args.run / 'inputs.json'), inference_only=True))
        result = (encode if args.phase == 'encode-query' else rank)(args.research_root, args.run)
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
                for phase in PHASES:
                    pipeline.phase(args, phase, stopped, cuda=phase == 'encode-query',
                        limit_seconds=1800 if phase == 'encode-query' else 900,
                        rss_limit_gib=16 if phase == 'encode-query' else 8)
                io.write(args.run / 'controller-exit.json', dict(status='all_phases_complete',
                    finished_unix=time.time(), independent_review_pending=True))
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
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3))
    args = parser.parse_args()
    args.research_root, args.run = args.research_root.resolve(strict=True), args.run.resolve()
    require(args.run.parent == args.research_root / 'NG-0076'
            and args.run.name == 'authentic-midpoint-v1', 'fresh dedicated midpoint path required')
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
        return
    m = verify(args.research_root, args.run)
    require(io.sha(Path(__file__)) == m['source'][Path(__file__).name], 'source not frozen')
    if args.mode == 'worker':
        require(args.phase is not None, 'explicit phase required')
        worker(args)
    elif args.mode == 'supervise':
        require(args.gpu is not None, 'explicit physical GPU required')
        supervise(args)


if __name__ == '__main__':
    main()
