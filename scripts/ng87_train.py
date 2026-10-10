"""NG87 clean-base, full-std teacher comparison on sealed NG85 inputs."""

from contextlib import contextmanager
import gc
import time

import numpy as np

from ng85_data import DOMAINS, read, require, rows, sha, training_order, write
from ng85_diagnose import QueryObserver, append, full_std_loss, geometry


SEED = 85001
STEPS = (2048, 4096, 8192)


@contextmanager
def corrected_objective():
    """Select the common treatment without editing the frozen NG85 helper."""
    import ng85_train

    original = ng85_train.calibrated_loss
    ng85_train.calibrated_loss = full_std_loss
    try:
        yield
    finally:
        ng85_train.calibrated_loss = original


def reset(encoder, observer=None):
    import torch

    if observer is not None:
        encoder.encode = observer.original
        observer.encoder = None
        observer.original = None
    encoder.model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()


def capture_failure(folder, encoder, optimizer, observer, completed, step, indices, exc):
    import torch
    from ng71_execution import save_optimizer

    write(folder / 'failure.json', dict(step=step, completed_update=completed,
        query_indices=indices, error=f'{type(exc).__name__}: {exc}',
        parameters_finite=all(bool(torch.isfinite(p).all())
                              for p in encoder.model.parameters()),
        query_raw=observer.events, automatic_retry=False))
    if observer.raw is not None:
        np.save(folder / 'failure-raw-query.npy', observer.raw)
    encoder.model.save_pretrained(folder / 'failure-model')
    save_optimizer(folder / 'failure-optimizer.pt', encoder, optimizer, completed)


def execute(args, config, queries, documents, bank, targets, task):
    import ng85_train as training
    from ng71_training import Encoder, parameter_hash
    from ng80_training import Readout

    operation, arm = args.phase.split('-')
    require(operation in ('preflight', 'train') and arm in ('A', 'B'),
            'registered clean-base arm required')
    folder = args.out / args.phase
    order = training_order(queries, SEED)
    require(len(order) == 32768, 'two full epochs required')
    write(folder / 'order.json', order)
    encoder = Encoder(args.base, config, device='cuda')
    optimizer = Readout(encoder, 'S').optimizer()
    initial = parameter_hash(encoder.model)
    require(initial == config['base']['state_sha256'] and len(optimizer.state) == 0,
            'clean base and fresh optimizer required')
    observer = QueryObserver(encoder)
    write(folder / 'initial.json', dict(model_sha256=initial,
        optimizer_state_entries=len(optimizer.state), seed=SEED,
        loss='full_std_gradient', arm=arm))
    count = 16 if operation == 'preflight' else STEPS[-1]
    samples = [training.example(i, arm, queries, documents, bank, targets)
               for i in order[:4]]
    with corrected_objective():
        if operation == 'preflight':
            parity = {domain: training.gradient_parity(encoder, item)
                      for domain, item in zip(DOMAINS, samples, strict=True)}
            require(parameter_hash(encoder.model) == initial,
                    'gradient preflight mutated base')
            write(folder / 'parity.json', dict(passed=True, domains=parity))
        begin, completed, saved = time.monotonic(), 0, []
        with (folder / 'steps.jsonl').open('x') as stream:
            for step in range(1, count + 1):
                indices = order[(step - 1) * 4:step * 4]
                observer.events = []
                observer.raw = None
                write(folder / 'intent.json', dict(step=step, query_indices=indices))
                items = [training.example(i, arm, queries, documents, bank, targets)
                         for i in indices]
                try:
                    records, norm = training.update(encoder, optimizer, items)
                    completed = step
                except Exception as exc:
                    capture_failure(folder, encoder, optimizer, observer,
                                    completed, step, indices, exc)
                    raise
                record = dict(step=step, query_indices=indices, gradient_norm=norm,
                    query_raw=observer.events,
                    query_nnz=[r['query_nnz'] for r in records],
                    document_nnz=[r['document_nnz'] / r['documents'] for r in records],
                    geometry=[geometry(r['scores'], item['target'])
                              for r, item in zip(records, items, strict=True)],
                    seconds=time.monotonic() - begin)
                append(stream, record)
                if step % 16 == 0:
                    write(folder / 'progress.json', record)
                    logger = task.get_logger()
                    logger.report_scalar('train', 'step', step, step)
                    logger.report_scalar('train', 'loss', np.mean([
                        r['ce'] for r in record['geometry']]), step)
                    logger.report_scalar('train', 'document_nnz',
                                         np.mean(record['document_nnz']), step)
                    print('NG87', operation, arm, step, flush=True)
                if step == 16:
                    projected = 1.5 * (time.monotonic() - begin) * 8192 / 16 + 1800
                    write(folder / 'eta.json', dict(projected_seconds=projected,
                        cap=12 * 3600, disposable=operation == 'preflight'))
                    require(projected < 12 * 3600, 'training ETA exceeds frozen cap')
                if step in STEPS or operation == 'preflight' and step == 16:
                    # Break the observer's bound-method cycle before releasing
                    # the old model during exact checkpoint/optimizer reload.
                    encoder.encode = observer.original
                    observer.encoder = None
                    observer.original = None
                    encoder, optimizer = training.save_reload(
                        args.base, config, folder, encoder, optimizer, step, items[0])
                    observer = QueryObserver(encoder)
                    saved.append(step)
    final = parameter_hash(encoder.model)
    require(final != initial, 'training did not update model')
    write(folder / 'results.json', dict(passed=True, arm=arm, updates=completed,
        exposures=completed * 4, initial_sha256=initial, final_sha256=final,
        saved_steps=saved, discard_updates=operation == 'preflight',
        teacher='PPLX' if arm == 'A' else 'RankT5',
        seed=SEED, loss='full_std_gradient', cost_regularizer=0.,
        endpoint_complete=False, production_qualified=False))


def profile(args, manifest, config, queries, documents, bank, task):
    import torch
    from ng71_training import Encoder, parameter_hash
    from ng86_evaluation import compare_rows, evaluate_row, projected_seconds

    arm = args.phase.split('-')[1]
    labels = ['base'] if arm == 'base' else [str(step) for step in STEPS]
    reference = {}
    if arm == 'base':
        old = args.base / 'NG-0086/evaluation-v2/profile-control/queries.jsonl'
        reference = {r['query_index']: r for r in rows(old)
                     if r['checkpoint'] == 'base'}
    folder = args.out / args.phase
    total = sum(len(manifest['endpoint_panel'] if label in ('base', '8192')
                    else manifest['monitor_panel']) for label in labels)
    started, count, load_times, anchors = time.monotonic(), 0, [], []
    with (folder / 'queries.jsonl').open('x') as stream:
        for position, label in enumerate(labels):
            checkpoint = None
            if label != 'base':
                checkpoint = args.out / f'train-{arm}/checkpoint-{int(label):05d}'
                receipt = read(checkpoint / 'receipt.json')
                for name, digest in receipt['files'].items():
                    require(sha(checkpoint / name) == digest, 'checkpoint changed')
            tick = time.monotonic()
            encoder = Encoder(args.base, config, device='cuda',
                              checkpoint=None if checkpoint is None else checkpoint / 'model')
            initial = parameter_hash(encoder.model)
            require(initial == (config['base']['state_sha256'] if checkpoint is None
                                 else receipt['state_sha256']), 'profile model identity')
            observer = QueryObserver(encoder)
            load_times.append(time.monotonic() - tick)
            panel = manifest['endpoint_panel'] if label in ('base', '8192') else manifest['monitor_panel']
            tick = time.monotonic()
            for row_count, index in enumerate(panel, 1):
                row = evaluate_row(encoder, observer, index, label, queries, documents, bank)
                row['arm'] = arm
                if label == 'base' and index in reference:
                    anchors.append(compare_rows(row, reference[index]))
                append(stream, row)
                count += 1
                if row_count == 64:
                    estimate = projected_seconds(time.monotonic() - started,
                        (time.monotonic() - tick) / row_count, count, total,
                        max(load_times), len(labels) - position - 1)
                    write(folder / f'eta-{label}.json', dict(projected_seconds=estimate,
                        cap=3600, rows_timed=row_count))
                    require(estimate < 3600, 'profile ETA exceeds frozen cap')
                if count % 64 == 0:
                    write(folder / 'progress.json', dict(checkpoint=label, rows=count, total=total))
                    task.get_logger().report_scalar('profile', 'rows', count, count)
                    print('NG87_PROFILE', arm, label, count, flush=True)
            require(parameter_hash(encoder.model) == initial, 'profile changed model')
            reset(encoder, observer)
            del encoder, observer
            gc.collect()
            torch.cuda.empty_cache()
    if arm == 'base':
        require(len(anchors) == 640, 'missing prior-base parity anchors')
    write(folder / 'results.json', dict(passed=True, labels=labels, rows=count,
        anchors=len(anchors), max_anchor_error=max(anchors, default=0.),
        full_corpus_evaluation=False, validation_independent=False))
