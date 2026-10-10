"""Read-only continuation of NG86's fixed checkpoint evaluation."""

import argparse
import gc
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from ng85_data import read, require, rows, sha, write
from ng85_run import closed
from ng86_intervention import (QueryObserver, READOUT_ERRORS, append,
                               candidate_metrics, load_encoder)
from review_ng86_intervention import summarize


SOURCE_SHA = '859dbcdc6aed659f74936551bdd6436e2018723f91b61ea71dec583972135783'
LABELS = ('base', 'start', '2560', '2816')
CAP = 1800
PHASE = 'profile-control'


def checked_phase(source, phase, passed=True):
    folder = source / phase
    receipt = read(folder / 'complete.json')
    exit_record = read(folder / 'exit.json')
    require(receipt['passed'] is passed and exit_record['passed'] is passed
            and exit_record['exit_code'] == (0 if passed else 1)
            and exit_record['owned_group_closed'], 'unexpected phase outcome')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p != folder / 'complete.json'}
    require(actual == set(receipt['files']), 'phase inventory changed')
    for name, digest in receipt['files'].items():
        path = folder / name
        require(path.resolve().is_relative_to(folder.resolve()) and sha(path) == digest,
                'changed phase payload: ' + phase + '/' + name)
    return receipt


def original_args(args):
    return SimpleNamespace(base=args.base, parent=args.base / 'NG-0085/student-ab-v4',
                           out=args.source)


def check_original(args):
    from ng86_intervention import verify

    require(args.source == args.base / 'NG-0086/paired-std-v1'
            and sha(args.source / 'inputs.json') == SOURCE_SHA, 'unexpected source')
    manifest = verify(original_args(args))
    for phase in ('train-control', 'train-fullstd', 'profile-fullstd'):
        checked_phase(args.source, phase)
    checked_phase(args.source, PHASE, passed=False)
    require(read(args.source / 'train-control/results.json')['saved_steps'] == [2560, 2816]
            and read(args.source / 'profile-fullstd/results.json')['labels'] == [2560, 2816, 3072],
            'fixed endpoints changed')
    return manifest


def freeze(args):
    require(args.out.parent == args.base / 'NG-0086' and not args.out.exists(),
            'new immutable NG86 evaluation unit required')
    old = check_original(args)
    args.out.mkdir()
    for name in old['source']:
        shutil.copy2(args.source / name, args.out / name)
    shutil.copy2(__file__, args.out / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('ng0086-evaluation-protocol.md'),
                 args.out / 'evaluation-protocol.md')
    write(args.out / 'inputs.json', dict(
        protocol='NG86_readonly_evaluation_continuation_v2', source_run=str(args.source),
        source_inputs_sha256=SOURCE_SHA, panel=old['panel'], labels=list(LABELS),
        primary_step=2816, wall_cap_seconds=CAP, scientific_changes=False,
        locked_test_access=False, training=False, automatic_expansion=False,
        source={p.name: sha(p) for p in args.out.iterdir() if p.is_file()},
        receipts={phase: sha(args.source / phase / 'complete.json') for phase in (
            'train-control', 'train-fullstd', 'profile-control', 'profile-fullstd')}))


def verify(args):
    manifest = read(args.out / 'inputs.json')
    require(manifest['protocol'] == 'NG86_readonly_evaluation_continuation_v2'
            and manifest['source_run'] == str(args.source)
            and manifest['source_inputs_sha256'] == SOURCE_SHA
            and manifest['labels'] == list(LABELS) and manifest['wall_cap_seconds'] == CAP
            and manifest['primary_step'] == 2816 and not manifest['scientific_changes']
            and not manifest['training'] and not manifest['locked_test_access']
            and not manifest['automatic_expansion'], 'changed continuation contract')
    for name, digest in manifest['source'].items():
        path = args.out / name
        require(path.resolve().is_relative_to(args.out.resolve()) and sha(path) == digest,
                'changed continuation source: ' + name)
    for phase, digest in manifest['receipts'].items():
        require(sha(args.source / phase / 'complete.json') == digest, 'changed source receipt')
    require(check_original(args)['panel'] == manifest['panel'], 'changed fixed panel')
    return manifest


def projected_seconds(elapsed, row_seconds, completed, total, load_seconds, loads_left):
    require(elapsed >= 0 and row_seconds > 0 and 0 < completed <= total
            and load_seconds >= 0 and loads_left >= 0, 'invalid timing sample')
    # Charge initialization once per remaining model, not once per query.
    return elapsed + 1.5 * (row_seconds * (total - completed)
                           + load_seconds * loads_left) + 120


def compare_rows(actual, expected):
    for key in ('query_index', 'subset', 'split', 'document_ids', 'valid'):
        require(actual[key] == expected[key], 'parity identity mismatch: ' + key)
    require(actual['valid'], 'parity anchor unexpectedly invalid')
    a, b = np.asarray(actual['scores']), np.asarray(expected['scores'])
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=1e-5, atol=1e-5), 'raw score parity failed')
    for key in ('candidate_ndcg10', 'candidate_recall10'):
        require(abs(actual[key] - expected[key]) < 1e-12, 'metric parity failed')
    return float(np.max(np.abs(a - b)))


def evaluate_row(encoder, observer, index, label, queries, documents, bank):
    import torch

    query = queries[index]
    left, right = bank['offsets'][index:index + 2]
    ids = bank['document_ids'][left:right]
    observer.events = []
    row = dict(arm='control', checkpoint=label, query_index=index,
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
                semantic_std=float(semantic.std()), lexical_std=float(bank['lexical'][left:right].std()),
                scores=scores.astype(float).tolist())
    except ValueError as exc:
        if str(exc) not in READOUT_ERRORS:
            raise
        row.update(valid=False, error=str(exc), candidate_ndcg10=0., candidate_recall10=0.)
    row['query_raw'] = observer.events
    return row


def release_encoder(encoder, observer):
    encoder.encode = observer.original


def evaluate(args, manifest, task):
    import torch
    from ng71_training import parameter_hash

    old_args = original_args(args)
    config = read(old_args.parent / 'config.json')
    queries = rows(old_args.parent / 'prepare/queries.jsonl')
    documents = rows(old_args.parent / 'prepare/documents.jsonl')
    with np.load(old_args.parent / 'bank/bank.npz') as archive:
        bank = {key: archive[key] for key in archive.files}
    folder = args.out / PHASE
    reference = rows(args.source / 'profile-fullstd/queries.jsonl')
    anchors = [next(r for r in reference if r['checkpoint'] == '2816'
                    and r['split'] == split and r['subset'] == domain)
               for split in ('fit', 'validation')
               for domain in ('fever', 'hotpotqa', 'nq', 'fiqa')]
    encoder = load_encoder(old_args, config, '2816', 'fullstd')
    initial = parameter_hash(encoder.model)
    observer = QueryObserver(encoder)
    errors = [compare_rows(evaluate_row(encoder, observer, r['query_index'], '2816',
               queries, documents, bank), r) for r in anchors]
    require(parameter_hash(encoder.model) == initial, 'parity mutated model')
    write(folder / 'parity.json', dict(passed=True, queries=[r['query_index'] for r in anchors],
        max_abs_score_errors=errors, rtol=1e-5, atol=1e-5, exact_metrics=True,
        model_sha256=initial))
    release_encoder(encoder, observer)
    del observer, encoder
    gc.collect()
    torch.cuda.empty_cache()
    prior_base = {r['query_index']: r for r in rows(args.source / PHASE / 'queries.jsonl')}
    require(len(prior_base) == 16 and all(r['checkpoint'] == 'base' for r in prior_base.values()),
            'unexpected partial control reference')
    count, parity_base, load_times = 0, [], []
    start = read(folder / 'started.json')['started_unix']
    with (folder / 'queries.jsonl').open('x') as stream:
        for label_index, label in enumerate(LABELS):
            load_start = time.monotonic()
            encoder = load_encoder(old_args, config, label, 'control')
            initial = parameter_hash(encoder.model)
            observer = QueryObserver(encoder)
            load_times.append(time.monotonic() - load_start)
            query_start = time.monotonic()
            for label_count, i in enumerate(manifest['panel'], 1):
                require(time.time() - start < CAP, 'evaluation wall limit')
                row = evaluate_row(encoder, observer, i, label, queries, documents, bank)
                if label == 'base' and i in prior_base:
                    parity_base.append(compare_rows(row, prior_base[i]))
                append(stream, row)
                count += 1
                if label_count == 16:
                    elapsed = time.time() - start
                    seconds_per_row = (time.monotonic() - query_start) / label_count
                    estimate = projected_seconds(elapsed, seconds_per_row, count, 2560,
                                                 max(load_times), len(LABELS) - label_index - 1)
                    write(folder / ('eta-' + label + '.json'), dict(
                        projected_seconds=estimate, cap=CAP, elapsed_seconds=elapsed,
                        seconds_per_row=seconds_per_row, load_seconds=load_times[-1]))
                    require(estimate < CAP, 'evaluation ETA exceeds fixed budget')
                if count % 64 == 0:
                    write(folder / 'progress.json', dict(checkpoint=label, rows=count,
                        planned_rows=2560, seconds=time.time() - start))
                    task.get_logger().report_scalar('profile', 'rows', count, count)
                    print('NG86_EVALUATION', label, count, flush=True)
            require(parameter_hash(encoder.model) == initial, 'evaluation mutated model')
            release_encoder(encoder, observer)
            del observer, encoder
            gc.collect()
            torch.cuda.empty_cache()
    require(len(parity_base) == 16, 'missing original partial parity')
    write(folder / 'base-parity.json', dict(passed=True, max_abs_score_errors=parity_base))
    write(folder / 'results.json', dict(passed=True, labels=list(LABELS), rows=count,
        candidate_only=True, validation_independent=False, training=False))


def worker(args):
    import torch
    from clearml import Task

    manifest = verify(args)
    require(torch.cuda.device_count() == 1, 'one explicit GPU required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(85001)
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0086/evaluation-v2',
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, offline=True, remote_synced=False, closed=False)
    write(args.out / PHASE / 'clearml-start.json', receipt)
    try:
        task.connect(dict(inputs_sha256=sha(args.out / 'inputs.json'), training=False))
        evaluate(args, manifest, task)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        write(args.out / PHASE / 'clearml.json', receipt)


def supervise(args):
    import fcntl
    import psutil
    from ng85_queue import check_resources, gpu_snapshot

    verify(args)
    require(args.gpu in (0, 1, 2, 3), 'explicit GPU required')
    with (args.base / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_resources(args.out, PHASE, args.gpu)
        folder = args.out / PHASE
        folder.mkdir()
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
                    '--base', str(args.base), '--source', str(args.source), '--out', str(args.out)]
                process = subprocess.Popen(command, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                write(folder / 'started.json', dict(pid=process.pid, gpu=args.gpu,
                    started_unix=time.time(), cap=CAP, inputs_sha256=sha(args.out / 'inputs.json')))
                while process.poll() is None:
                    require(not stopped and time.monotonic() - begin < CAP, 'stop or wall limit')
                    try:
                        tree = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in [tree, *tree.children(recursive=True)])
                    except psutil.NoSuchProcess:
                        rss = 0
                    peak = max(peak, rss)
                    require(rss < 24 * 1024 ** 3, 'owned RSS limit')
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
        require(passed, 'evaluation failed; preserve immutable attempt')


def validate_records(control, fullstd, panel):
    for records, arm, labels in ((control, 'control', LABELS),
                                 (fullstd, 'fullstd', ('2560', '2816', '3072'))):
        require(len(records) == len(labels) * len(panel)
                and {r['checkpoint'] for r in records} == set(labels)
                and all(r['arm'] == arm for r in records), 'wrong profile labels or rows')
        for label in labels:
            selected = [r for r in records if r['checkpoint'] == label]
            require(len(selected) == len(panel)
                    and {r['query_index'] for r in selected} == set(panel), 'wrong profile panel')
    identity = {r['query_index']: (r['subset'], r['split'], r['document_ids'])
                for r in control if r['checkpoint'] == 'base'}
    require(all((r['subset'], r['split'], r['document_ids']) == identity[r['query_index']]
                for r in control + fullstd), 'candidate universe changed')


def review(args):
    manifest = verify(args)
    closed(args.out, PHASE)
    control = rows(args.out / PHASE / 'queries.jsonl')
    fullstd = rows(args.source / 'profile-fullstd/queries.jsonl')
    validate_records(control, fullstd, manifest['panel'])
    result = summarize(control + fullstd)
    result.update(passed=True, passed_means='paired observation, not model qualification',
                  training_restarted=False, control_evaluation_continuation=True)
    require(result['paired_validation']['2816']['available'], 'missing primary comparison')
    folder = args.out / 'review'
    folder.mkdir()
    write(folder / 'inputs.json', dict(inputs_sha256=sha(args.out / 'inputs.json'),
        control_receipt_sha256=sha(args.out / PHASE / 'complete.json'),
        fullstd_receipt_sha256=sha(args.source / 'profile-fullstd/complete.json')))
    write(folder / 'results.json', result)
    write(folder / 'exit.json', dict(passed=True, exit_code=0, finished_unix=time.time()))
    write(folder / 'complete.json', dict(passed=True,
        files={p.name: sha(p) for p in folder.iterdir() if p.is_file()}))
    print('NG86_PAIRED_REVIEW_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'worker', 'run', 'review'))
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.action == 'run':
        supervise(args)
        review(args)
    else:
        {'freeze': freeze, 'worker': worker, 'review': review}[args.action](args)


if __name__ == '__main__':
    main()
