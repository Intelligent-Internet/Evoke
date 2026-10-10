"""NG85 mature sparse student: fixed two epochs, teacher-only A/B contrast."""

import gc
import json
import time

import numpy as np

from ng85_data import (DOMAINS, read, require, rows, sha, target_distribution,
                       training_order, write)


def calibrated_loss(scores, target):
    import torch

    target = torch.as_tensor(target, dtype=scores.dtype, device=scores.device)
    require(scores.ndim == 1 and scores.shape == target.shape
            and bool(torch.isfinite(scores).all())
            and bool(torch.isfinite(target).all())
            and bool((target >= 0).all())
            and bool(torch.isclose(target.sum(), target.new_tensor(1.))),
            'invalid calibrated loss')
    scale = scores.std(correction=0).detach().clamp_min(1e-6)
    return -(target * torch.log_softmax(scores / scale, dim=0)).sum()


def load_targets(run, queries, bank):
    from ng85_run import closed

    output = {}
    for shard in range(4):
        result = closed(run, f'rankt5-{shard}')
        require(result['bank_sha256'] == sha(run / 'bank/bank.npz'), 'target bank mismatch')
        values = np.load(run / f'rankt5-{shard}/scores.npz')
        for j, raw in enumerate(values['query_indices']):
            i = int(raw)
            left, right = values['offsets'][j:j + 2]
            target = values['scores'][left:right]
            require(i not in output and queries[i]['split'] == 'fit'
                    and len(target) == bank['offsets'][i + 1] - bank['offsets'][i]
                    and np.isfinite(target).all(), 'target identity or shape mismatch')
            output[i] = target.copy()
    require(set(output) == {i for i, q in enumerate(queries) if q['split'] == 'fit'},
            'incomplete RankT5 targets')
    return output


def save_reload(base, config, out, encoder, optimizer, step, sample):
    import torch
    from ng71_training import Encoder, parameter_hash
    from ng71_execution import save_optimizer, restore_optimizer
    from ng80_training import Readout

    folder = out / f'checkpoint-{step:05d}'
    folder.mkdir()
    state = parameter_hash(encoder.model)
    with torch.no_grad():
        q = encoder.encode([sample['query']], 'query').cpu()
        d = encoder.encode(sample['documents'][:2], 'document').cpu()
    encoder.model.save_pretrained(folder / 'model')
    moment = save_optimizer(folder / 'optimizer.pt', encoder, optimizer, step)
    encoder.model.to('cpu')
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    resumed = Encoder(base, config, device='cuda', checkpoint=folder / 'model')
    new_optimizer = Readout(resumed, 'S').optimizer()
    require(parameter_hash(resumed.model) == state, 'checkpoint parameter reload mismatch')
    require(restore_optimizer(folder / 'optimizer.pt', resumed, new_optimizer, step) == moment,
            'optimizer reload mismatch')
    with torch.no_grad():
        require(torch.equal(q, resumed.encode([sample['query']], 'query').cpu())
                and torch.equal(d, resumed.encode(sample['documents'][:2], 'document').cpu()),
                'query/document checkpoint reload mismatch')
    write(folder / 'receipt.json', dict(step=step, state_sha256=state,
        optimizer_sha256=moment, exact_reload=True,
        files={str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
    return resumed, new_optimizer


def example(i, arm, queries, documents, bank, task_targets):
    left, right = bank['offsets'][i:i + 2]
    ids = bank['document_ids'][left:right]
    target = bank['pplx'][left:right] if arm == 'A' else task_targets[i]
    require(set(queries[i]['positives']) <= set(ids), 'lost known positive')
    return dict(query=queries[i]['query'], documents=[documents[j]['text'] for j in ids],
        lexical=bank['lexical'][left:right], target=target_distribution(target))


def update(encoder, optimizer, examples):
    import torch
    from ng71_training import backward_query

    require(len(examples) == 4, 'one query per domain per update')
    optimizer.zero_grad(set_to_none=True)
    records = []
    for item in examples:
        records.append(backward_query(encoder, **item, accumulation=4,
            loss_transform=lambda scores, unused, target=item['target']:
                calibrated_loss(scores, target)))
    norm = torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.)
    require(bool(torch.isfinite(norm)) and float(norm) > 0, 'invalid gradient norm')
    optimizer.step()
    return records, float(norm)


def gradient_parity(encoder, sample):
    import torch
    from ng71_training import backward_query

    item = dict(sample, documents=sample['documents'][:4],
                lexical=sample['lexical'][:4], target=target_distribution([0., 1., 2., 3.]))
    require(len(item['documents']) == 4, 'parity fixture requires four real candidates')
    reference = None
    losses = []
    for replay in (False, True):
        encoder.model.zero_grad(set_to_none=True)
        result = backward_query(encoder, **item, accumulation=1, replay=replay,
            loss_transform=lambda scores, unused: calibrated_loss(scores, item['target']))
        gradient = torch.cat([p.grad.detach().cpu().ravel() for p in encoder.model.parameters()])
        losses.append(result['loss'])
        if reference is None:
            reference = gradient
        else:
            relative = float((gradient - reference).norm() / reference.norm().clamp_min(1e-12))
    encoder.model.zero_grad(set_to_none=True)
    require(np.isfinite(relative) and relative < 1e-5
            and abs(losses[0] - losses[1]) < 1e-5, 'real VJP gradient parity failed')
    return dict(relative_gradient_l2=relative, losses=losses, passed=True)


def execute(base, run, phase, task):
    import torch
    from ng71_training import Encoder, parameter_hash
    from ng80_training import Readout
    from ng85_run import closed

    closed(run, 'prepare')
    closed(run, 'bank')
    _, arm, raw_seed = phase.split('-')
    seed = int(raw_seed)
    require(arm in ('A', 'B') and seed in (85001, 85002), 'unregistered arm')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.device_count() == 1, 'one visible GPU required')
    torch.manual_seed(seed)
    config, out = read(run / 'config.json'), run / phase
    queries = rows(run / 'prepare/queries.jsonl')
    documents = rows(run / 'prepare/documents.jsonl')
    bank = np.load(run / 'bank/bank.npz')
    targets = load_targets(run, queries, bank)
    order = training_order(queries, seed)
    require(len(order) == 32768, 'two complete epochs required')
    write(out / 'order.json', order)
    encoder = Encoder(base, config, device='cuda')
    optimizer = Readout(encoder, 'S').optimizer()
    first = [example(i, arm, queries, documents, bank, targets) for i in order[:4]]
    parity = {domain: gradient_parity(encoder, item)
              for domain, item in zip(DOMAINS, first, strict=True)}
    require(parameter_hash(encoder.model) == config['base']['state_sha256'],
            'gradient-only preflight changed parent')
    tick = time.monotonic()
    for step in range(16):
        items = [example(i, arm, queries, documents, bank, targets)
                 for i in order[4 * step:4 * step + 4]]
        update(encoder, optimizer, items)
    elapsed = time.monotonic() - tick
    projected = 1.5 * elapsed * 8192 / 16 + 1800
    write(out / 'canary.json', dict(discard_updates=True, updates=16, parity=parity,
        elapsed_seconds=elapsed, projected_training_seconds=projected,
        budget_seconds=12 * 3600, passed=projected < 12 * 3600))
    require(projected < 12 * 3600, 'full training cannot fit frozen budget')
    encoder, optimizer = save_reload(base, config, out, encoder, optimizer, 16, first[0])
    del encoder, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    encoder = Encoder(base, config, device='cuda')
    optimizer = Readout(encoder, 'S').optimizer()
    initial = parameter_hash(encoder.model)
    start = time.monotonic()
    with (out / 'steps.jsonl').open('x') as stream:
        for step in range(1, 8193):
            indices = order[4 * (step - 1):4 * step]
            items = [example(i, arm, queries, documents, bank, targets) for i in indices]
            records, norm = update(encoder, optimizer, items)
            record = dict(step=step, epoch=1 if step <= 4096 else 2,
                query_indices=indices, gradient_norm=norm,
                losses=[r['loss'] for r in records],
                student_score_std=[float(np.std(r['scores'])) for r in records],
                teacher_entropy=[float(-np.sum(item['target'] * np.log(
                    np.maximum(item['target'], 1e-300)))) for item in items],
                query_nnz=[r['query_nnz'] for r in records],
                document_nnz_per_query=[r['document_nnz'] / r['documents'] for r in records],
                seconds=time.monotonic() - start)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            if step % 16 == 0:
                stream.flush()
                write(out / 'progress.json', record)
                task.get_logger().report_scalar('fit', 'mean_loss', float(np.mean(record['losses'])), step)
                print('NG85_FIT', arm, seed, step, round(record['seconds']), flush=True)
            if step % 1024 == 0:
                encoder, optimizer = save_reload(base, config, out, encoder, optimizer, step, first[0])
    state = parameter_hash(encoder.model)
    require(state != initial, 'no actual parameter update')
    write(out / 'results.json', dict(arm=arm, seed=seed, updates=8192,
        query_exposures=32768, independent_fit_queries=16384,
        initial_model_sha256=initial, final_model_sha256=state,
        checkpoint='checkpoint-08192/model', order_sha256=sha(out / 'order.json'),
        objective='teacher KL on complete hybrid scores; no fabricated human negatives',
        cost_regularizer_weight=0., full_corpus_evaluation_complete=False,
        production_qualified=False, locked_test_access=False,
        seconds=time.monotonic() - start, max_vram_bytes=torch.cuda.max_memory_allocated()))
