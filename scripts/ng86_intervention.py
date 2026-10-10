"""Frozen, paired NG86 std-gradient intervention; no automatic expansion."""

import argparse
import gc
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from ng85_data import read, require, rows, sha, target_distribution, write
from ng85_diagnose import (PARENT_SHA, QueryObserver, append, candidate_metrics,
                           checkpoint_files, full_std_loss, geometry)
from ng85_run import closed, verify as verify_parent


ARMS = ('control', 'fullstd')
STEPS = (2560, 2816, 3072)
CAPS = {'train': 5400, 'profile': 1800}
DIAG_SHA = 'bcf88bf364b6275a4903239a81fb77661538871220528cb34d69408edfbeec49'
READOUT_ERRORS = ('invalid fixed RMS normalization', 'invalid sparse readout')


def select_panel(queries):
    selected = []
    for split, count in (('fit', 32), ('validation', 128)):
        for domain in ('fever', 'hotpotqa', 'nq', 'fiqa'):
            candidates = [i for i, q in enumerate(queries)
                          if q['split'] == split and q['subset'] == domain]
            require(len(candidates) >= count, 'incomplete fixed panel')
            candidates.sort(key=lambda i: hashlib.sha256(
                ('NG85-diagnosis-v1/' + queries[i]['query_id']).encode()).hexdigest())
            selected.extend(candidates[:count])
    require(len(selected) == len(set(selected)) == 640, 'duplicate panel identity')
    return selected


def support_failure(event, parameters_finite):
    return bool(parameters_finite and event and event['raw_finite']
                and event['denominator_finite'] and event['denominator'] == 0.
                and event['raw_nnz'] > 0 and event['supported_nnz'] == 0
                and event['unsupported_nnz'] == event['raw_nnz'])


def freeze(args):
    require(args.parent == args.base / 'NG-0085/student-ab-v4'
            and args.out.parent == args.base / 'NG-0086' and not args.out.exists(),
            'new NG86 unit and canonical parent required')
    require(sha(args.parent / 'inputs.json') == PARENT_SHA, 'unexpected NG85 parent')
    verify_parent(args.base, args.parent)
    diagnosis = args.base / 'NG-0085/loss-diagnosis-v2'
    require(sha(diagnosis / 'inputs.json') == DIAG_SHA, 'unexpected diagnosis')
    evidence = closed(diagnosis, 'replay')
    failure = evidence['failure']
    require(evidence['observed_failure'] and not evidence['scientific_changes']
            and failure['step'] == 2840
            and support_failure(failure['query_raw'][-1], failure['parameters_finite']),
            'sealed finite zero-support failure required')
    dependencies = {str(p): sha(p) for p in (
        args.parent / 'inputs.json', diagnosis / 'inputs.json',
        diagnosis / 'replay/complete.json',
        args.parent / 'train-B-85001/order.json')}
    for name, digest in checkpoint_files(args.parent, 'B', 2048).items():
        dependencies[str(args.parent / name)] = digest
    for phase in ('prepare', 'bank', *(f'rankt5-{i}' for i in range(4))):
        closed(args.parent, phase)
        dependencies[str(args.parent / phase / 'complete.json')] = sha(
            args.parent / phase / 'complete.json')
    panel = select_panel(rows(args.parent / 'prepare/queries.jsonl'))
    args.out.mkdir(parents=True)
    for name in read(diagnosis / 'inputs.json')['source']:
        require(sha(diagnosis / name) == read(diagnosis / 'inputs.json')['source'][name],
                'changed diagnostic helper')
        shutil.copy2(diagnosis / name, args.out / name)
    shutil.copy2(__file__, args.out / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('review_ng86_intervention.py'),
                 args.out / 'review_ng86_intervention.py')
    shutil.copy2(Path(__file__).with_name('ng0086-protocol.md'), args.out / 'protocol.md')
    write(args.out / 'inputs.json', dict(
        protocol='NG86_prospective_std_gradient_v1', parent=str(args.parent),
        source={p.name: sha(p) for p in args.out.iterdir() if p.is_file()},
        dependencies=dependencies, panel=panel, arms=list(ARMS),
        start_step=2048, end_step=3072, saved_steps=list(STEPS),
        primary_matched_step=2816, seed=85001, caps=CAPS,
        sole_treatment='differentiate_student_std', rms_unchanged=True,
        candidate_only=True, validation_independent=False, locked_test_access=False,
        automatic_research_expansion=False, production_deployment=False))


def verify(args):
    manifest = read(args.out / 'inputs.json')
    require(manifest['protocol'] == 'NG86_prospective_std_gradient_v1'
            and manifest['parent'] == str(args.parent)
            and manifest['arms'] == list(ARMS) and manifest['saved_steps'] == list(STEPS)
            and manifest['caps'] == CAPS and manifest['rms_unchanged']
            and not manifest['locked_test_access'], 'changed NG86 contract')
    for name, digest in manifest['source'].items():
        path = args.out / name
        require(path.resolve().is_relative_to(args.out.resolve()) and sha(path) == digest,
                'changed NG86 source: ' + name)
    for name, digest in manifest['dependencies'].items():
        path = Path(name)
        require(path.resolve().is_relative_to(args.base.resolve()) and sha(path) == digest,
                'changed NG86 dependency: ' + name)
    verify_parent(args.base, args.parent)
    require(select_panel(rows(args.parent / 'prepare/queries.jsonl')) == manifest['panel'],
            'changed fixed panel')
    for phase in ('prepare', 'bank'):
        closed(args.parent, phase)
    return manifest


def load_encoder(args, config, label, arm):
    from ng71_training import Encoder, parameter_hash

    folder = None
    if label == 'start':
        folder = args.parent / 'train-B-85001/checkpoint-02048'
    elif label != 'base':
        folder = args.out / f'train-{arm}/checkpoint-{int(label):05d}'
    encoder = Encoder(args.base, config, device='cuda',
                      checkpoint=None if folder is None else folder / 'model')
    if folder is not None:
        require(parameter_hash(encoder.model) == read(folder / 'receipt.json')['state_sha256'],
                'loaded checkpoint state mismatch')
    return encoder


def train(args, manifest, config, queries, documents, bank, targets, task):
    import torch
    import ng85_train
    from ng71_execution import restore_optimizer, save_optimizer
    from ng71_training import parameter_hash
    from ng80_training import Readout

    arm = args.phase.split('-')[1]
    original_loss = ng85_train.calibrated_loss
    if arm == 'fullstd':
        ng85_train.calibrated_loss = full_std_loss
    folder = args.out / args.phase
    order = read(args.parent / 'train-B-85001/order.json')
    encoder = load_encoder(args, config, 'start', arm)
    optimizer = Readout(encoder, 'S').optimizer()
    checkpoint = args.parent / 'train-B-85001/checkpoint-02048'
    receipt = read(checkpoint / 'receipt.json')
    require(restore_optimizer(checkpoint / 'optimizer.pt', encoder, optimizer, 2048)
            == receipt['optimizer_sha256'], 'optimizer identity mismatch')
    initial_hash = parameter_hash(encoder.model)
    initial_items = [ng85_train.example(i, 'B', queries, documents, bank, targets)
                     for i in order[8192:8196]]
    with torch.no_grad():
        for item in initial_items:
            q = encoder.encode([item['query']], 'query')
            d = encoder.encode(item['documents'][:4], 'document')
            s = (d @ q.T).ravel() + torch.as_tensor(
                item['lexical'][:4], dtype=torch.float32, device='cuda')
            target = target_distribution([0., 1., 2., 3.])
            require(torch.equal(original_loss(s, target), full_std_loss(s, target)),
                    'treatment changed loss forward')
    parity = [ng85_train.gradient_parity(encoder, item) for item in initial_items]
    require(parameter_hash(encoder.model) == initial_hash, 'preflight changed model')
    write(folder / 'preflight.json', dict(passed=True, parity=parity,
        model_sha256=initial_hash, optimizer_sha256=receipt['optimizer_sha256'],
        forward_equal=True, arm=arm))
    observer = QueryObserver(encoder)
    start, saved, failure, completed = time.monotonic(), [], None, 2048
    with (folder / 'steps.jsonl').open('x') as stream:
        for step in range(2049, 3073):
            require(time.monotonic() - start < CAPS['train'], 'train budget exceeded')
            indices = order[(step - 1) * 4:step * 4]
            observer.events = []
            stage = 'update'
            write(folder / 'intent.json', dict(step=step, query_indices=indices))
            items = [ng85_train.example(i, 'B', queries, documents, bank, targets)
                     for i in indices]
            try:
                records, norm = ng85_train.update(encoder, optimizer, items)
                completed = step
                row = dict(step=step, gradient_norm=norm, query_indices=indices,
                    losses=[r['loss'] for r in records], query_raw=observer.events,
                    query_nnz=[r['query_nnz'] for r in records],
                    document_nnz=[r['document_nnz'] / r['documents'] for r in records],
                    geometry=[geometry(r['scores'], item['target'])
                              for r, item in zip(records, items, strict=True)],
                    seconds=time.monotonic() - start)
                append(stream, row)
                if step % 16 == 0:
                    write(folder / 'progress.json', row)
                    task.get_logger().report_scalar('train', 'step', step, step)
                    print('NG86_TRAIN', arm, step, flush=True)
                if step == 2064:
                    projected = 1.5 * (time.monotonic() - start) * 1024 / 16 + 600
                    write(folder / 'eta.json', dict(projected_seconds=projected, cap=CAPS['train']))
                    require(projected < CAPS['train'], 'train ETA exceeds fixed budget')
                if step in STEPS:
                    stage = 'checkpoint_reload'
                    encoder, optimizer = ng85_train.save_reload(
                        args.base, config, folder, encoder, optimizer, step, items[0])
                    saved.append(step)
                    observer = QueryObserver(encoder)
            except ValueError as exc:
                if str(exc) not in READOUT_ERRORS:
                    raise
                finite = all(bool(torch.isfinite(p).all()) for p in encoder.model.parameters())
                event = observer.events[-1] if observer.events else None
                failure = dict(step=step, stage=stage, completed_update=completed,
                    error=str(exc), parameters_finite=finite, query_raw=observer.events,
                    expected_support_failure=support_failure(event, finite),
                    query_indices=indices)
                write(folder / 'failure.json', failure)
                if observer.raw is not None:
                    np.save(folder / 'failure-raw-query.npy', observer.raw)
                encoder.model.save_pretrained(folder / 'failure-model')
                save_optimizer(folder / 'failure-optimizer.pt', encoder, optimizer, completed)
                require(failure['expected_support_failure'], 'unexpected numerical failure')
                break
    write(folder / 'results.json', dict(passed=True, arm=arm, failure=failure,
        training_complete=completed == 3072 and failure is None, saved_steps=saved,
        completed_update=completed, candidate_only=True, production_qualified=False,
        conclusion='bounded_scientific_observation_not_training_success'))


def profile(args, manifest, config, queries, documents, bank, targets, task):
    import torch
    from ng71_training import parameter_hash

    arm = args.phase.split('-')[1]
    result = closed(args.out, 'train-' + arm)
    labels = (['base', 'start'] if arm == 'control' else []) + result['saved_steps']
    require(all(step in STEPS for step in result['saved_steps']), 'unplanned checkpoint')
    folder = args.out / args.phase
    begin, count = time.monotonic(), 0
    with (folder / 'queries.jsonl').open('x') as stream:
        for label in labels:
            encoder = load_encoder(args, config, label, arm)
            initial = parameter_hash(encoder.model)
            observer = QueryObserver(encoder)
            for i in manifest['panel']:
                require(time.monotonic() - begin < CAPS['profile'], 'profile budget exceeded')
                query = queries[i]
                left, right = bank['offsets'][i:i + 2]
                ids = bank['document_ids'][left:right]
                observer.events = []
                row = dict(arm=arm, checkpoint=str(label), query_index=i,
                    subset=query['subset'], split=query['split'], document_ids=ids.tolist())
                row['bm25'] = candidate_metrics(ids, bank['lexical'][left:right], query['positives'])
                row['pplx'] = candidate_metrics(ids, bank['pplx'][left:right], query['positives'])
                try:
                    with torch.no_grad():
                        q = encoder.encode([query['query']], 'query')
                        d = encoder.encode([documents[j]['text'] for j in ids], 'document')
                        semantic = (d @ q.T).ravel().cpu().numpy()
                        scores = semantic + bank['lexical'][left:right].astype(np.float32)
                        row.update(candidate_metrics(ids, scores, query['positives']))
                        row.update(valid=True, semantic=candidate_metrics(ids, semantic, query['positives']),
                            document_nnz_mean=float((d > 0).sum(1).float().mean()),
                            query_nnz=int((q > 0).sum()), score_std=float(scores.std()),
                            semantic_std=float(semantic.std()),
                            lexical_std=float(bank['lexical'][left:right].std()),
                            scores=scores.astype(float).tolist())
                except ValueError as exc:
                    if str(exc) not in READOUT_ERRORS:
                        raise
                    row.update(valid=False, error=str(exc), candidate_ndcg10=0., candidate_recall10=0.)
                row['query_raw'] = observer.events
                append(stream, row)
                count += 1
                if count == 16:
                    projected = 1.5 * (time.monotonic() - begin) * len(labels) * 640 / 16 + 120
                    write(folder / 'eta.json', dict(projected_seconds=projected, cap=CAPS['profile']))
                    require(projected < CAPS['profile'], 'profile ETA exceeds fixed budget')
                if count % 32 == 0:
                    write(folder / 'progress.json', dict(checkpoint=label, rows=count,
                                                        planned_rows=len(labels) * 640))
                    task.get_logger().report_scalar('profile', 'rows', count, count)
                    print('NG86_PROFILE', arm, label, count, flush=True)
            require(parameter_hash(encoder.model) == initial, 'profile mutated model')
            encoder.encode = observer.original
            del observer, encoder
            gc.collect()
            torch.cuda.empty_cache()
    write(folder / 'results.json', dict(passed=True, labels=labels, rows=count,
        candidate_only=True, full_corpus_evaluation=False, validation_independent=False))


def worker(args):
    import torch
    from clearml import Task
    from ng85_train import load_targets

    manifest = verify(args)
    require(torch.cuda.device_count() == 1, 'one explicit GPU required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(85001)
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0086/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, offline=True, remote_synced=False, closed=False)
    write(args.out / args.phase / 'clearml-start.json', receipt)
    try:
        task.connect(dict(inputs_sha256=sha(args.out / 'inputs.json'), phase=args.phase))
        queries = rows(args.parent / 'prepare/queries.jsonl')
        documents = rows(args.parent / 'prepare/documents.jsonl')
        with np.load(args.parent / 'bank/bank.npz') as archive:
            bank = {key: archive[key] for key in archive.files}
        targets = load_targets(args.parent, queries, bank)
        operation = args.phase.split('-')[0]
        {'train': train, 'profile': profile}[operation](
            args, manifest, read(args.parent / 'config.json'), queries, documents, bank, targets, task)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        write(args.out / args.phase / 'clearml.json', receipt)


def supervise(args):
    import fcntl
    import psutil
    from ng85_queue import check_resources, gpu_snapshot

    verify(args)
    require(args.gpu in (0, 3), 'explicit available 0/3 GPU required')
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
                command = [sys.executable, '-B', str(args.out / Path(__file__).name), 'worker',
                    '--base', str(args.base), '--parent', str(args.parent), '--out', str(args.out),
                    '--phase', args.phase]
                process = subprocess.Popen(command, env=env, stdout=log,
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
                require(process.returncode == 0, 'worker execution failed')
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
            passed = error is None and group_closed and receipt.get('outcome') == 'passed'
            write(folder / 'exit.json', dict(passed=passed, error=error,
                exit_code=process.returncode if process else None, owned_group_closed=group_closed,
                peak_tree_rss_bytes=peak, seconds=time.monotonic() - begin, finished_unix=time.time()))
            write(folder / 'complete.json', dict(passed=passed,
                files={str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()}))
        require(passed, 'execution failed; preserve immutable attempt')


def lane(args):
    verify(args)
    require(args.arm in ARMS, 'registered arm required')
    for operation in ('train', 'profile'):
        phase = operation + '-' + args.arm
        command = [sys.executable, '-B', str(args.out / Path(__file__).name), 'supervise',
            '--base', str(args.base), '--parent', str(args.parent), '--out', str(args.out),
            '--phase', phase, '--gpu', str(args.gpu)]
        subprocess.run(command, check=True)
        closed(args.out, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'worker', 'supervise', 'lane'))
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=[p + '-' + a for p in CAPS for a in ARMS])
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.action in ('worker', 'supervise') and args.phase is None:
        parser.error('--phase required')
    {'freeze': freeze, 'worker': worker, 'supervise': supervise, 'lane': lane}[args.action](args)


if __name__ == '__main__':
    main()
