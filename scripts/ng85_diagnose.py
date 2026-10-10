"""Immutable NG85 checkpoint diagnostics; no automatic training continuation."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np


PARENT_SHA = '5b5f7fa82c7e449e6e01f5309cf833919341fa20a69e3d9a74bd9c0c5ce50387'
CAPS = {'profile': 1800, 'replay': 5400}


def full_std_loss(scores, target):
    """Single-factor intervention: identical forward, differentiate the std."""
    import torch
    from ng85_data import require

    target = torch.as_tensor(target, dtype=scores.dtype, device=scores.device)
    require(scores.ndim == 1 and scores.shape == target.shape
            and bool(torch.isfinite(scores).all())
            and bool(torch.isfinite(target).all())
            and bool((target >= 0).all())
            and bool(torch.isclose(target.sum(), target.new_tensor(1.))),
            'invalid calibrated loss')
    scale = scores.std(correction=0).clamp_min(1e-6)
    return -(target * torch.log_softmax(scores / scale, dim=0)).sum()


def geometry(scores, target):
    """Score-space derivatives, not a proposal to change the frozen objective."""
    s, t = np.asarray(scores, dtype=float), np.asarray(target, dtype=float)
    if (s.ndim != 1 or len(s) < 2 or s.shape != t.shape
            or not np.isfinite(s).all() or not np.isfinite(t).all()
            or (t < 0).any() or not np.isclose(t.sum(), 1.)):
        raise ValueError('invalid geometry inputs')
    centered = s - s.mean()
    std = float(s.std())
    scale = max(std, 1e-6)
    logits = centered / scale
    logp = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
    detached = (np.exp(logp) - t) / scale
    full = detached.copy()
    if std > 1e-6:
        full -= centered * np.dot(detached, centered) / (len(s) * std ** 2)
    return dict(std=std, ce=float(-np.dot(t, logp)),
                detached_scale_derivative=float(np.dot(detached, centered)),
                full_scale_derivative=float(np.dot(full, centered)),
                detached_gradient_norm=float(np.linalg.norm(detached)),
                full_gradient_norm=float(np.linalg.norm(full)))


def panel_indices(queries):
    """Choose before inference, balanced and independent of scores or labels."""
    selected = []
    for split in ('fit', 'validation'):
        for domain in ('fever', 'hotpotqa', 'nq', 'fiqa'):
            candidates = [i for i, q in enumerate(queries)
                          if q['split'] == split and q['subset'] == domain]
            if len(candidates) < 16:
                raise ValueError('incomplete diagnostic panel')
            selected.extend(sorted(candidates, key=lambda i: hashlib.sha256(
                ('NG85-diagnosis-v1/' + queries[i]['query_id']).encode()).hexdigest())[:16])
    return selected


def candidate_metrics(ids, scores, positives):
    ids, scores, positives = np.asarray(ids), np.asarray(scores), set(positives)
    if not positives or not positives <= set(ids.tolist()):
        raise ValueError('candidate bank lost known positives')
    order = np.lexsort((ids, -scores))[:10]
    hits = np.array([int(ids[i]) in positives for i in order], dtype=float)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(positives))))
    return dict(candidate_ndcg10=float(np.sum(hits / np.log2(
        np.arange(len(hits)) + 2)) / ideal),
        candidate_recall10=float(hits.sum() / len(positives)))


def checkpoint_files(parent, arm, step):
    from ng85_data import read, require, sha

    folder = parent / f'train-{arm}-85001/checkpoint-{step:05d}'
    receipt = read(folder / 'receipt.json')
    require(receipt['step'] == step and receipt['exact_reload'], 'checkpoint receipt')
    files = {str((folder / name).relative_to(parent)): digest
             for name, digest in receipt['files'].items()}
    files[str((folder / 'receipt.json').relative_to(parent))] = sha(folder / 'receipt.json')
    for name, digest in files.items():
        require((parent / name).resolve().is_relative_to(parent.resolve())
                and sha(parent / name) == digest, 'checkpoint changed: ' + name)
    return files


def drift_reference(folder):
    """Verify a failed strict replay without relabeling it as successful."""
    from ng85_data import read, require, sha

    receipt = read(folder / 'replay/complete.json')
    exit_record = read(folder / 'replay/exit.json')
    manifest = read(folder / 'inputs.json')
    require(not receipt['passed'] and exit_record['owned_group_closed']
            and exit_record['exit_code'] == 1
            and not manifest['scientific_changes']
            and manifest['parent_artifacts']['inputs.json'] == PARENT_SHA,
            'closed unchanged failed replay required')
    for root, entries in ((folder, manifest['source']),
                          (folder / 'replay', receipt['files'])):
        for name, digest in entries.items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve()) and sha(path) == digest,
                    'changed drift reference: ' + name)
    history = [json.loads(line) for line in
               (folder / 'replay/steps.jsonl').read_text().splitlines()]
    require(history and not history[-1]['trace_matches']
            and all(row['trace_matches'] for row in history[:-1]),
            'reference must stop at first trace mismatch')
    return dict(path=str(folder), first_mismatch_step=history[-1]['step'],
        files={name: sha(folder / name) for name in
               ('inputs.json', 'replay/complete.json')})


def freeze(args):
    from ng85_data import read, require, rows, sha, write
    from ng85_run import closed, verify

    require(args.parent.parent == args.base / 'NG-0085'
            and args.out.parent == args.parent.parent and not args.out.exists(),
            'new NG85 diagnostic unit required')
    require(sha(args.parent / 'inputs.json') == PARENT_SHA, 'unexpected parent')
    verify(args.base, args.parent)
    artifacts = {'inputs.json': PARENT_SHA}
    for phase in ('prepare', 'bank', *(f'rankt5-{i}' for i in range(4))):
        closed(args.parent, phase)
        artifacts[f'{phase}/complete.json'] = sha(args.parent / phase / 'complete.json')
    for arm in 'AB':
        for step in (1024, 2048):
            artifacts.update(checkpoint_files(args.parent, arm, step))
        for name in ('order.json', 'steps.jsonl', 'exit.json', 'complete.json'):
            path = args.parent / f'train-{arm}-85001' / name
            artifacts[str(path.relative_to(args.parent))] = sha(path)
    queries = rows(args.parent / 'prepare/queries.jsonl')
    control = None
    drift = None
    if args.drift_reference is not None:
        require(args.control is None and args.drift_reference.parent == args.parent.parent,
                'observational replay cannot change the objective')
        drift = drift_reference(args.drift_reference)
    if args.control is not None:
        require(args.control.parent == args.parent.parent, 'same NG85 control required')
        evidence = closed(args.control, 'replay')
        closed(args.control, 'profile')
        control_manifest = read(args.control / 'inputs.json')
        require(evidence['observed_failure'] and evidence['reproduced_original_step']
                and evidence['failure']['parameters_finite']
                and not control_manifest['scientific_changes']
                and control_manifest['parent_artifacts']['inputs.json'] == PARENT_SHA,
                'exact finite-parameter failure reproduction required')
        for name, digest in control_manifest['source'].items():
            require(sha(args.control / name) == digest, 'changed control source')
        history = rows(args.control / 'replay/steps.jsonl')
        require(history and all(all(error == 0. for error in row['trace_max_abs_errors'].values())
                                for row in history), 'bit-exact control trace required')
        require(control_manifest['panel'] == panel_indices(queries), 'same diagnostic panel required')
        control = dict(path=str(args.control), files={name: sha(args.control / name)
            for name in ('inputs.json', 'replay/complete.json', 'profile/complete.json')})
    args.out.mkdir()
    shutil.copy2(__file__, args.out / Path(__file__).name)
    for name in read(args.parent / 'inputs.json')['source']:
        if name.endswith('.py'):
            shutil.copy2(args.parent / name, args.out / name)
    write(args.out / 'inputs.json', dict(
        parent=str(args.parent), parent_artifacts=artifacts,
        source={p.name: sha(p) for p in args.out.glob('*.py')},
        panel=panel_indices(queries), panel_fit=64, panel_validation=64,
        validation_independent=False, candidate_only=True,
        checkpoints=['corrected-3072'] if control else ['base', 'A-1024', 'A-2048', 'B-1024', 'B-2048'],
        replay_arm='B', replay_seed=85001, replay_start=2048, replay_end=3072 if control else 2864,
        objective='full_std_gradient' if control else 'original_detached_std',
        trace_policy='observe' if drift else 'strict', drift_reference=drift,
        control=control, scientific_changes=bool(control), locked_test_access=False,
        maximum_seconds=CAPS, automatic_research_expansion=False))


def validate(args):
    from ng85_data import read, require, sha
    from ng85_run import closed, verify

    manifest = read(args.out / 'inputs.json')
    require(manifest['parent'] == str(args.parent)
            and not manifest['locked_test_access'], 'diagnostic contract changed')
    for root, files in ((args.out, manifest['source']),
                        (args.parent, manifest['parent_artifacts'])):
        for name, digest in files.items():
            require((root / name).resolve().is_relative_to(root.resolve())
                    and sha(root / name) == digest, 'changed diagnostic input: ' + name)
    verify(args.base, args.parent)
    if manifest.get('drift_reference'):
        reference = manifest['drift_reference']
        require(drift_reference(Path(reference['path'])) == reference,
                'changed drift reference')
    if manifest.get('control'):
        control = manifest['control']
        for name, digest in control['files'].items():
            require(sha(Path(control['path']) / name) == digest, 'changed control receipt')
        for phase in ('replay', 'profile'):
            closed(Path(control['path']), phase)
    for phase in ('prepare', 'bank'):
        closed(args.parent, phase)
    return manifest


class QueryObserver:
    """Read-only hooks around the original encoder, preserving its graph."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.original = encoder.encode
        self.events = []
        self.raw = None
        encoder.encode = self.encode

    def encode(self, texts, role):
        if role != 'query':
            return self.original(texts, role)
        import torch

        mask = None

        def capture_mask(module, inputs, kwargs):
            nonlocal mask
            mask = kwargs['attention_mask'].detach().bool()

        def capture_values(module, inputs, output):
            from ng85_data import require

            require(mask is not None, 'query hook missing attention mask')
            with torch.no_grad():
                raw = output.detach().relu().log1p().masked_fill(
                    ~mask[..., None], 0).amax(1)
                rms = self.encoder.rms
                denominator = raw @ rms
                self.raw = raw.cpu().numpy()
                for vector, den in zip(raw, denominator, strict=True):
                    finite = bool(torch.isfinite(vector).all())
                    valid_den = bool(torch.isfinite(den))
                    self.events.append(dict(
                        raw_finite=finite, denominator_finite=valid_den,
                        denominator=float(den) if valid_den else None,
                        raw_nnz=int((vector > 0).sum()),
                        supported_nnz=int(((vector > 0) & (rms > 0)).sum()),
                        unsupported_nnz=int(((vector > 0) & (rms == 0)).sum()),
                        raw_sum=float(vector.sum()) if finite else None,
                        unsupported_mass=float(vector[rms == 0].sum()) if finite else None))

        a = self.encoder.model.base_model.register_forward_pre_hook(capture_mask, with_kwargs=True)
        b = self.encoder.model.lm_head.register_forward_hook(capture_values)
        try:
            return self.original(texts, role)
        finally:
            a.remove()
            b.remove()


def load_encoder(args, config, label):
    import torch
    from ng71_training import Encoder, parameter_hash
    from ng85_data import read, require

    torch.manual_seed(85001)
    folder = None
    if label == 'corrected-3072':
        from ng85_run import closed

        closed(args.out, 'replay')
        folder = args.out / 'replay/checkpoint-03072'
    elif label != 'base':
        arm, step = label.split('-')
        folder = args.parent / f'train-{arm}-85001/checkpoint-{int(step):05d}'
    encoder = Encoder(args.base, config, device='cuda',
                      checkpoint=None if folder is None else folder / 'model')
    if folder is not None:
        require(parameter_hash(encoder.model) == read(folder / 'receipt.json')['state_sha256'],
                'loaded checkpoint state mismatch')
    return encoder


def append(stream, row):
    stream.write(json.dumps(row, allow_nan=False) + '\n')
    stream.flush()


def profile(args, manifest, config, queries, documents, bank, targets, task):
    import torch
    from ng71_training import parameter_hash
    from ng85_data import require, target_distribution, write

    folder = args.out / args.job
    with (folder / 'queries.jsonl').open('x') as stream:
        for label in manifest['checkpoints']:
            encoder = load_encoder(args, config, label)
            initial = parameter_hash(encoder.model)
            observer = QueryObserver(encoder)
            sample = queries[manifest['panel'][0]]['query']
            with torch.no_grad():
                require(torch.equal(observer.original([sample], 'query'),
                                    encoder.encode([sample], 'query')), 'observer changed forward')
            for number, i in enumerate(manifest['panel']):
                query = queries[i]
                left, right = bank['offsets'][i:i + 2]
                ids = bank['document_ids'][left:right]
                observer.events = []
                row = dict(checkpoint=label, query_index=i, subset=query['subset'],
                           split=query['split'], candidates=len(ids))
                try:
                    with torch.no_grad():
                        q = encoder.encode([query['query']], 'query')
                        d = encoder.encode([documents[j]['text'] for j in ids], 'document')
                        semantic = (d @ q.T).ravel().cpu().numpy()
                        scores = semantic + bank['lexical'][left:right].astype(np.float32)
                        row.update(candidate_metrics(ids, scores, query['positives']))
                        row.update(document_nnz_mean=float((d > 0).sum(1).float().mean()),
                                   query_nnz=int((q > 0).sum()), score_std=float(scores.std()),
                                   semantic_std=float(semantic.std()),
                                   lexical_std=float(bank['lexical'][left:right].std()),
                                   scores=scores.astype(float).tolist(), document_ids=ids.tolist())
                    row['geometry_A'] = geometry(scores, target_distribution(bank['pplx'][left:right]))
                    if i in targets:
                        row['geometry_B'] = geometry(scores, target_distribution(targets[i]))
                    row['valid'] = True
                except ValueError as exc:
                    if str(exc) not in ('invalid fixed RMS normalization', 'invalid sparse readout'):
                        raise
                    row.update(valid=False, error=str(exc))
                row['query_raw'] = observer.events
                append(stream, row)
                write(folder / 'progress.json', dict(checkpoint=label, queries=number + 1))
                if (number + 1) % 16 == 0:
                    task.get_logger().report_scalar('profile', label, number + 1, number + 1)
                    print('PROFILE', label, number + 1, flush=True)
            require(parameter_hash(encoder.model) == initial, 'read-only profile mutated model')
            encoder.encode = observer.original
            del encoder, observer
            gc.collect()
            torch.cuda.empty_cache()
    write(folder / 'results.json', dict(passed=True, candidate_only=True,
        rows=len(manifest['panel']) * len(manifest['checkpoints']),
        full_corpus_evaluation=False, locked_test_access=False))


def replay(args, manifest, config, queries, documents, bank, targets, task):
    import torch
    from ng71_execution import restore_optimizer, save_optimizer
    from ng80_training import Readout
    from ng85_data import read, require, rows, write
    from ng85_train import example, update, gradient_parity

    changed = manifest.get('objective') == 'full_std_gradient'
    if changed:
        import ng85_train

        # Replace only this worker's callable; parent sources stay immutable.
        ng85_train.calibrated_loss = full_std_loss

    folder = args.out / args.job
    original = args.parent / 'train-B-85001'
    order = read(original / 'order.json')
    trace = {r['step']: r for r in rows(original / 'steps.jsonl')}
    encoder = load_encoder(args, config, 'B-2048')
    optimizer = Readout(encoder, 'S').optimizer()
    receipt = read(original / 'checkpoint-02048/receipt.json')
    restored = restore_optimizer(original / 'checkpoint-02048/optimizer.pt', encoder, optimizer, 2048)
    require(restored == receipt['optimizer_sha256'], 'optimizer mismatch')
    parity = [gradient_parity(encoder, example(i, 'B', queries, documents, bank, targets))
              for i in order[8192:8196]]
    write(folder / 'parity.json', parity)
    observer = QueryObserver(encoder)
    first_drift = None
    exact_trace = True
    with (folder / 'steps.jsonl').open('x') as stream:
        for step in range(manifest['replay_start'] + 1, manifest['replay_end'] + 1):
            indices = order[4 * (step - 1):4 * step]
            observer.events = []
            write(folder / 'intent.json', dict(step=step, query_indices=indices))
            items = [example(i, 'B', queries, documents, bank, targets) for i in indices]
            try:
                records, norm = update(encoder, optimizer, items)
            except ValueError as exc:
                if str(exc) not in ('invalid fixed RMS normalization', 'invalid sparse readout'):
                    raise
                fail = dict(step=step, error=str(exc), query_indices=indices,
                            query_raw=observer.events, last_successful_step=step - 1)
                fail['parameters_finite'] = all(bool(torch.isfinite(p).all())
                                                for p in encoder.model.parameters())
                write(folder / 'failure.json', fail)
                if observer.raw is not None:
                    np.save(folder / 'failure-raw-query.npy', observer.raw)
                encoder.model.save_pretrained(folder / 'failure-model')
                save_optimizer(folder / 'failure-optimizer.pt', encoder, optimizer, step - 1)
                write(folder / 'results.json', dict(passed=True,
                    observed_failure=True, reproduced_original_step=(step == max(trace) + 1),
                    failure=fail, first_trace_drift=first_drift,
                    exact_original_reproduction=exact_trace and step == max(trace) + 1,
                    locked_test_access=False, scientific_changes=changed))
                return
            row = dict(step=step, query_indices=indices, gradient_norm=norm,
                losses=[r['loss'] for r in records],
                student_score_std=[float(np.std(r['scores'])) for r in records],
                query_nnz=[r['query_nnz'] for r in records],
                document_nnz_per_query=[r['document_nnz'] / r['documents'] for r in records],
                query_raw=observer.events,
                geometry=[geometry(r['scores'], item['target'])
                          for r, item in zip(records, items, strict=True)])
            if step in trace:
                errors = {key: float(np.max(np.abs(np.asarray(row[key]) - trace[step][key])))
                          for key in ('gradient_norm', 'losses', 'student_score_std',
                                      'query_nnz', 'document_nnz_per_query')}
                row['trace_max_abs_errors'] = errors
                exact_trace = exact_trace and all(value == 0. for value in errors.values())
                row['trace_matches'] = all(np.allclose(row[key], trace[step][key], rtol=1e-6, atol=1e-6)
                                           for key in errors)
                append(stream, row)
                if not row['trace_matches'] and first_drift is None:
                    first_drift = step
                    if manifest.get('trace_policy') == 'observe':
                        encoder.model.save_pretrained(folder / 'first-drift-model')
                        save_optimizer(folder / 'first-drift-optimizer.pt', encoder, optimizer, step)
                        write(folder / 'first-drift.json', dict(
                            step=step, state_after_update=True,
                            trace_max_abs_errors=errors, exact_original_reproduction=False))
                if not changed:
                    require(row['trace_matches'] or manifest.get('trace_policy') == 'observe',
                            'replay diverged from original trace')
            else:
                append(stream, row)
            if step % 16 == 0:
                write(folder / 'progress.json', row)
                task.get_logger().report_scalar('replay', 'step', step, step)
                print('REPLAY', step, flush=True)
    if changed:
        from ng85_train import save_reload

        encoder, optimizer = save_reload(args.base, config, folder, encoder, optimizer,
                                         manifest['replay_end'], items[0])
    write(folder / 'results.json', dict(passed=True, observed_failure=False,
        reached_step=manifest['replay_end'], locked_test_access=False,
        first_trace_drift=first_drift, exact_original_reproduction=False,
        scientific_changes=changed,
        conclusion='bounded_intervention_complete_not_full_training'
            if changed else 'not_reproduced_within_fixed_budget'))


def worker(args):
    import torch
    from clearml import Task
    from ng85_data import read, require, rows, sha, write
    from ng85_train import load_targets

    manifest = validate(args)
    require(torch.cuda.device_count() == 1, 'one visible GPU required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0085/' + args.out.name + '/' + args.job,
        reuse_last_task_id=False, auto_connect_frameworks=False,
        auto_connect_arg_parser=False, auto_connect_streams=False, auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, offline=True, remote_synced=False, closed=False)
    write(args.out / args.job / 'clearml-start.json', receipt)
    try:
        task.connect(dict(inputs_sha256=sha(args.out / 'inputs.json'), job=args.job))
        config = read(args.parent / 'config.json')
        queries = rows(args.parent / 'prepare/queries.jsonl')
        documents = rows(args.parent / 'prepare/documents.jsonl')
        with np.load(args.parent / 'bank/bank.npz') as archive:
            bank = {key: archive[key] for key in archive.files}
        targets = load_targets(args.parent, queries, bank)
        {'profile': profile, 'replay': replay}[args.job](
            args, manifest, config, queries, documents, bank, targets, task)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        write(args.out / args.job / 'clearml.json', receipt)


def supervise(args):
    import fcntl
    import psutil
    from ng85_data import read, require, sha, write
    from ng85_queue import check_resources, gpu_snapshot

    validate(args)
    require(args.gpu in range(4), 'explicit GPU required')
    with (args.base / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_resources(args.out, args.job, args.gpu)
        folder = args.out / args.job
        folder.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', CLEARML_OFFLINE_MODE='1',
            CLEARML_CACHE_DIR=str(folder / 'tracking-cache'))
        process, error, peak, closed_group = None, None, 0, False
        stopped = []
        handlers = {s: signal.signal(s, lambda v, frame: stopped.append(v))
                    for s in (signal.SIGTERM, signal.SIGINT)}
        start = time.monotonic()
        try:
            with (folder / 'stdout.log').open('x') as log:
                command = [sys.executable, '-B', str(args.out / Path(__file__).name),
                    'worker', '--base', str(args.base), '--parent', str(args.parent),
                    '--out', str(args.out), '--job', args.job]
                process = subprocess.Popen(command, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                write(folder / 'started.json', dict(pid=process.pid, gpu=args.gpu,
                    started_unix=time.time(), wall_cap_seconds=CAPS[args.job],
                    inputs_sha256=sha(args.out / 'inputs.json')))
                while process.poll() is None:
                    require(not stopped, 'stop requested')
                    try:
                        tree = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in [tree, *tree.children(recursive=True)])
                    except psutil.NoSuchProcess:
                        rss = 0
                    peak = max(peak, rss)
                    require(rss < 24 * 1024 ** 3, 'owned RSS limit')
                    require(time.monotonic() - start < CAPS[args.job], 'wall limit')
                    require(psutil.virtual_memory().available > 20 * 1024 ** 3, 'host memory floor')
                    require(shutil.disk_usage(args.out).free > 40 * 1024 ** 3, 'disk floor')
                    require(gpu_snapshot(args.gpu)['memory_mib'] < 22 * 1024, 'GPU memory limit')
                    time.sleep(5)
                require(process.returncode == 0, 'diagnostic worker failed')
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
                    closed_group = True
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            receipt = read(folder / 'clearml.json') if (folder / 'clearml.json').exists() else {}
            passed = error is None and closed_group and receipt.get('outcome') == 'passed'
            write(folder / 'exit.json', dict(error=error, passed=passed,
                exit_code=process.returncode if process else None,
                owned_group_closed=closed_group, peak_tree_rss_bytes=peak,
                seconds=time.monotonic() - start, finished_unix=time.time()))
            write(folder / 'complete.json', dict(passed=passed,
                files={str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()}))
        require(passed, 'diagnostic stopped; preserve attempt')


def campaign(args):
    from ng85_data import read, require
    from ng85_run import closed

    manifest = validate(args)
    require(manifest.get('objective') == 'full_std_gradient', 'bounded intervention only')
    for job in ('replay', 'profile'):
        command = [sys.executable, '-B', str(args.out / Path(__file__).name), 'supervise',
            '--base', str(args.base), '--parent', str(args.parent), '--out', str(args.out),
            '--job', job, '--gpu', str(args.gpu)]
        subprocess.run(command, check=True)
        closed(args.out, job)
        if job == 'replay':
            require(not read(args.out / job / 'results.json')['observed_failure'],
                    'intervention failed; no endpoint profile')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'supervise', 'worker', 'campaign'])
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--job', choices=list(CAPS))
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--drift-reference', type=Path)
    args = parser.parse_args()
    sys.path.insert(1, str(args.parent))
    if args.action not in ('freeze', 'campaign') and args.job is None:
        parser.error('--job required')
    {'freeze': freeze, 'supervise': supervise, 'worker': worker,
     'campaign': campaign}[args.action](args)


if __name__ == '__main__':
    main()
