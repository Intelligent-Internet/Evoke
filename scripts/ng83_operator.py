"""Matched rank-256 score operators: only relation weighting differs."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse

import ng82_diagnostics as prior


campaign, old, require = prior.campaign, prior.old, prior.require
PARENT = 'NG-0082/diagnostics-v1'
PARENT_SHA = '5f591075a122242f85b126abe3d4d12a62da974424978dcaf70dc360e2605804'
PROTOCOL = 'ng0083-operator-execution.zh.md'
PHASES = ('prepare', 'fit-field', 'fit-boundary', 'rank-field', 'rank-boundary')
POSITIONS = tuple(sorted(set(range(16)) | {19, 29, 39, 49, 59, 69, 79, 89, 99, 100, 127, 199}))
WEIGHTS = {'field': 0., 'boundary': .5}


def config():
    return dict(protocol='NG83_operator_v1', phases=list(PHASES), rank=256,
        train_queries=12288, diagnostic_queries=1536, batch_queries=4, updates=3072,
        seed=83083, learning_rate=1e-4, weight_decay=0., clip_norm=1.,
        boundary_weights=WEIGHTS, positions=list(POSITIONS), pairs=list(map(list, prior.PAIRS)),
        source_encoder_updates=0, locked_test_access=False, independent_holdout=False,
        automatic_promotion=False, native_cost_evaluated=False)


def paths(base):
    files = prior.paths(base)
    files['pca'] = base / PARENT / 'geometry/pca-256.npz'
    files['panels'] = base / old.dense_fit.PARENT / 'prepare/panels.json'
    return files


def verify(base, run):
    manifest = campaign.read(run / 'inputs.json')
    require(manifest['config'] == config(), 'operator config changed')
    require(manifest['files'] == {k: str(v.relative_to(base)) for k, v in paths(base).items()},
            'operator input paths changed')
    required = set(manifest['files'].values()) | {PARENT + '/inputs.json'}
    require(required <= set(manifest['dependencies']), 'unbound operator input')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in manifest[key].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve())
                    and campaign.io.sha(path) == digest, 'changed input: ' + name)
    require(manifest['dependencies'][PARENT + '/inputs.json'] == PARENT_SHA, 'parent changed')
    return {k: base / v for k, v in manifest['files'].items()}


def freeze(base, run):
    parent = base / PARENT
    require(run.parent == base / 'NG-0083' and not run.exists(), 'fresh operator unit')
    require(campaign.io.sha(parent / 'inputs.json') == PARENT_SHA, 'NG82 identity')
    prior.verify(base, parent)
    prior.audit.sealed(parent / 'geometry')
    prior.audit.sealed(base / old.dense_fit.PARENT / 'prepare')
    run.mkdir(parents=True)
    for name in campaign.read(parent / 'inputs.json')['source']:
        shutil.copy2(parent / name, run / name)
    source = Path(__file__).resolve().parent
    for p in (Path(__file__), source.parent / 'tests/test_ng83_operator.py',
              source.parent / 'docs/research-sae/reports/ng0001-ng0099' / PROTOCOL):
        shutil.copy2(p, run / p.name)
    files = paths(base)
    anchors = [parent / 'inputs.json', parent / 'geometry/complete.json',
               base / old.dense_fit.PARENT / 'prepare/complete.json']
    campaign.io.write(run / 'inputs.json', dict(config=config(),
        files={k: str(p.relative_to(base)) for k, p in files.items()},
        dependencies={str(p.relative_to(base)): campaign.io.sha(p)
                      for p in [*files.values(), *anchors]},
        source={p.name: campaign.io.sha(p) for p in run.iterdir() if p.is_file()}))
    verify(base, run)
    print('NG83_OPERATOR_FROZEN', campaign.io.sha(run / 'inputs.json'), flush=True)


def candidate_bank(order, panel, gold):
    require(len(order) >= 200 and len(set(order.tolist())) == len(order), 'teacher order')
    bank = list(dict.fromkeys([*(int(order[p]) for p in POSITIONS), *panel, *gold]))
    mapping = {doc: i for i, doc in enumerate(bank)}
    pairs = [[mapping[int(order[p])], mapping[int(order[n])]] for p, n in prior.PAIRS]
    require(set(gold) <= set(bank) and len(bank) <= 256, 'candidate bank bound')
    return bank, pairs


def relation_loss(student, target, pairs, weight, scale):
    require(weight in WEIGHTS.values() and scale > 0, 'relation weighting contract')
    error = student - target
    field = ((error - error.mean()) ** 2).mean()
    boundary = .5 * ((error[pairs[:, 0]] - error[pairs[:, 1]]) ** 2).mean()
    return ((1 - weight) * field + weight * boundary) / scale, field, boundary


def prepare(files, output):
    records, labels, _, split, dd, dq = prior.load_context(files)
    panels = campaign.read(files['panels'])
    documents = dd.astype(np.float64)
    train = split['train']
    ids, offsets, targets, relations, variances = [], [0], [], [], []
    started = time.monotonic()
    for lo in range(0, len(train), 32):
        block = dq[train[lo:lo + 32]].astype(np.float64) @ documents.T
        for offset, index in enumerate(train[lo:lo + 32]):
            score = block[offset]
            order = np.lexsort((np.arange(len(score)), -score))[:200]
            bank, pairs = candidate_bank(order, panels[index], labels[index])
            target = score[bank]
            independent = np.einsum('ij,j->i', documents[bank], dq[index].astype(np.float64))
            np.testing.assert_allclose(target, independent, rtol=0, atol=1e-12)
            ids.extend(bank)
            offsets.append(len(ids))
            targets.extend(target.tolist())
            relations.append(pairs)
            variances.append(float(np.var(target)))
        if lo == 32:
            estimate = 1.5 * (time.monotonic() - started) * len(train) / 64 + 300
            campaign.io.write(output / 'canary.json', dict(projected_seconds=estimate))
            require(estimate < 5100, 'candidate preparation exceeds bound')
        if lo % 1024 == 0:
            print('NG83_BANK', lo + 32, flush=True)
    rng = np.random.default_rng(config()['seed'])
    np.savez(output / 'bank.npz', query_indices=np.array(train, dtype=np.int64),
        document_ids=np.array(ids, dtype=np.int64), offsets=np.array(offsets, dtype=np.int64),
        targets=np.array(targets, dtype=np.float64), pairs=np.array(relations, dtype=np.int64),
        training_order=rng.permutation(len(train)))
    scale = float(np.mean(variances))
    require(np.isfinite(scale) and scale > 1e-10, 'TRAIN scale degenerate')
    return dict(queries=len(train), documents=len(dd), candidate_exposures=len(ids),
        unique_documents=len(set(ids)), candidate_min=int(np.diff(offsets).min()),
        candidate_max=int(np.diff(offsets).max()), field_variance_scale=scale,
        human_positives_preserved=True, teacher_preferences_are_human_negatives=False,
        independent_holdout=False)


def fit(files, run, output, arm):
    import os

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch

    prepared = prior.audit.sealed(run / 'prepare')
    bank = np.load(run / 'prepare/bank.npz')
    source = np.load(files['pca'])
    dd, dq = np.load(files['documents'], mmap_mode='r'), np.load(files['queries'], mmap_mode='r')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(config()['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available(), 'explicit GPU required')
    mean = torch.as_tensor(source['mean'], device='cuda', dtype=torch.float32)
    a = torch.nn.Parameter(torch.tensor(source['query_map'], device='cuda', dtype=torch.float32))
    b = torch.nn.Parameter(torch.tensor(source['document_map'], device='cuda', dtype=torch.float32))
    optimizer = torch.optim.AdamW([a, b], lr=config()['learning_rate'], weight_decay=0.)
    scale, weight = prepared['field_variance_scale'], WEIGHTS[arm]
    order = bank['training_order']
    require(len(order) == config()['train_queries'] and sorted(order.tolist()) == list(range(len(order))),
            'training exposure/order')
    rows, norms, started = [], [], time.monotonic()
    seen = []
    with (output / 'updates.jsonl').open('x') as stream:
        for step, lo in enumerate(range(0, len(order), 4), start=1):
            optimizer.zero_grad(set_to_none=True)
            field_value, boundary_value, loss_value = 0., 0., 0.
            for position in order[lo:lo + 4]:
                index = int(bank['query_indices'][position])
                start, end = bank['offsets'][position:position + 2]
                docids = bank['document_ids'][start:end]
                q = torch.tensor(np.array(dq[index]), device='cuda', dtype=torch.float32)
                d = torch.tensor(np.array(dd[docids]), device='cuda', dtype=torch.float32)
                target = torch.tensor(bank['targets'][start:end], device='cuda', dtype=torch.float32)
                pairs = torch.tensor(bank['pairs'][position], device='cuda', dtype=torch.long)
                score = ((d - mean) @ b) @ (q @ a)
                loss, field, boundary = relation_loss(score, target, pairs, weight, scale)
                require(torch.isfinite(loss).item(), 'nonfinite loss')
                (loss / 4).backward()
                field_value += field.item() / 4
                boundary_value += boundary.item() / 4
                loss_value += loss.item() / 4
                seen.append(index)
            norm = torch.nn.utils.clip_grad_norm_([a, b], 1., error_if_nonfinite=True).item()
            norms.append(norm)
            optimizer.step()
            record = dict(step=step, loss=loss_value, field=field_value,
                          boundary=boundary_value, unclipped_gradient_norm=norm)
            rows.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            if step == 16:
                torch.cuda.synchronize()
                estimate = 1.5 * (time.monotonic() - started) * config()['updates'] / 16 + 300
                campaign.io.write(output / 'canary.json', dict(projected_seconds=estimate,
                    gradients_finite=True, first_updates_are_part_of_fixed_training=True))
                require(estimate < 5100, 'fit cannot fit bound')
            if step % 256 == 0:
                print('NG83_FIT', arm, step, loss_value, flush=True)
    require(len(rows) == 3072 and len(seen) == len(set(seen)) == 12288
            and sorted(seen) == sorted(bank['query_indices'].tolist()), 'incomplete training')
    maps = dict(mean=source['mean'], query_map=a.detach().cpu().numpy().astype(np.float64),
                document_map=b.detach().cpu().numpy().astype(np.float64))
    np.savez(output / 'maps.npz', **maps)
    torch.save(dict(query_map=a.detach().cpu(), document_map=b.detach().cpu(),
                    optimizer=optimizer.state_dict(), updates=3072), output / 'checkpoint.pt')
    saved = np.load(output / 'maps.npz')
    exported = ((d.detach().cpu().numpy().astype(np.float64) - saved['mean'])
                @ saved['document_map']) @ (q.detach().cpu().numpy().astype(np.float64)
                                            @ saved['query_map'])
    actual = (((d - mean) @ b) @ (q @ a)).detach().cpu().numpy()
    np.testing.assert_allclose(exported, actual, rtol=0, atol=1e-5)
    checkpoint = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
    np.testing.assert_array_equal(checkpoint['query_map'].numpy(), saved['query_map'])
    np.testing.assert_array_equal(checkpoint['document_map'].numpy(), saved['document_map'])
    require(all(int(v['step']) == 3072 for v in checkpoint['optimizer']['state'].values()),
            'optimizer steps not serialized')
    campaign.io.write(output / 'seen-query-indices.json', seen)
    return dict(arm=arm, boundary_weight=weight, updates=3072, distinct_train_queries=len(set(seen)),
        candidate_exposures=prepared['candidate_exposures'], trainable_parameters=a.numel() + b.numel(),
        exported_score_max_error=float(np.max(np.abs(exported - actual))),
        gradient_norm_max=max(norms), gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        bank_sha256=campaign.io.sha(run / 'prepare/bank.npz'),
        initial_map_sha256=campaign.io.sha(files['pca']),
        final_online_loss=rows[-1]['loss'], source_encoder_updates=0,
        independent_holdout=False, native_cost_evaluated=False)


def rank(files, run, output, arm):
    prior.audit.sealed(run / ('fit-' + arm))
    records, labels, selected, _, dd, dq = prior.load_context(files)
    models = {'initial': files['pca'], arm: run / ('fit-' + arm) / 'maps.npz'}
    ld = sparse.load_npz(files['ld']).astype(np.float64).tocsc()
    results = []
    started = time.monotonic()
    for role in ('calibration', 'diagnostic'):
        ids = selected[role]
        lq = sparse.load_npz(files['lcq' if role == 'calibration' else 'lq'])
        queries = dq[ids].astype(np.float64)
        documents = dd.astype(np.float64)
        output_rows = [dict(query_id=records[i]['query_id'], domain=records[i]['subset'],
                            role=role, rankings={}, geometry={}) for i in ids]
        for name, path in models.items():
            maps = np.load(path)
            d = (documents - maps['mean']) @ maps['document_map']
            q = queries @ maps['query_map']
            for lo in range(0, len(ids), 16):
                block, target = q[lo:lo + 16] @ d.T, queries[lo:lo + 16] @ documents.T
                for offset, index in enumerate(ids[lo:lo + 16]):
                    row = lo + offset
                    score, teacher = block[offset], target[offset]
                    lex, _, _ = old.sparse_scores(ld, lq[row])
                    entry = output_rows[row]
                    if name == 'initial':
                        entry['rankings']['dense'] = prior.compact_rank(teacher, labels[index])
                        entry['rankings']['dense_hybrid'] = prior.compact_rank(
                            prior.prev.mix(teacher, lex, .1, 'zscore'), labels[index])
                    entry['rankings'][name] = prior.compact_rank(score, labels[index])
                    entry['rankings'][name + '_hybrid'] = prior.compact_rank(
                        prior.prev.mix(score, lex, .1, 'zscore'), labels[index])
                    order = np.lexsort((np.arange(len(teacher)), -teacher))[:200]
                    p, n = np.array([(order[p], order[n]) for p, n in prior.PAIRS]).T
                    tm, sm = teacher[p] - teacher[n], score[p] - score[n]
                    entry['geometry'][name] = dict(
                        centered_score_mse=float(np.var(score - teacher)),
                        boundary_margin_mse=float(np.mean((sm - tm) ** 2)),
                        positive_teacher_pairs=int((tm > 1e-12).sum()),
                        preserved_positive_pairs=int(((tm > 1e-12) & (sm > 0)).sum()))
                if role == 'calibration' and name == 'initial' and lo == 48:
                    estimate = 1.5 * (time.monotonic() - started) * (384 + 1536) * 2 / 64 + 300
                    campaign.io.write(output / 'canary.json', dict(projected_seconds=estimate))
                    require(estimate < 5100, 'full ranking cannot fit bound')
                if lo % 256 == 0:
                    print('NG83_RANK', arm, role, name, lo + 16, flush=True)
            del d, q
            gc.collect()
        results.extend(output_rows)
    with (output / 'rankings.jsonl').open('x') as stream:
        for record in results:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    groups = {role: [r for r in results if r['role'] == role]
              for role in ('calibration', 'diagnostic')}
    return dict(arm=arm, quality={role: prior.summarize(rows, 'rankings') for role, rows in groups.items()},
        versus_initial={role: {suffix or 'pure': prior.paired(rows, arm + suffix, 'initial' + suffix)
                               for suffix in ('', '_hybrid')} for role, rows in groups.items()},
        queries=len(results), independent_holdout=False, native_cost_evaluated=False)


def worker(args):
    from clearml import Task

    files = verify(args.research_root, args.run)
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0083/operator-' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=old.TRACKING_FRAMEWORKS.copy(),
        auto_connect_arg_parser=False, auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    campaign.io.write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(config=config(), input_sha256=campaign.io.sha(args.run / 'inputs.json')))
        if args.phase == 'prepare':
            result = prepare(files, output)
        else:
            phase, arm = args.phase.split('-')
            result = (fit if phase == 'fit' else rank)(files, args.run, output, arm)
        verify(args.research_root, args.run)
        campaign.io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        campaign.io.write(output / 'clearml.json', receipt)


def supervise(args):
    import fcntl
    import os
    import signal

    verify(args.research_root, args.run)
    require(campaign.io.sha(Path(__file__)) == campaign.read(args.run / 'inputs.json')[
            'source']['ng83_operator.py'], 'frozen operator controller')
    name, stopped = args.phase, []
    require(name in PHASES and not (args.run / (name + '-controller-start.json')).exists(),
            'fresh operator lane')
    cuda = name.startswith('fit-')
    require(not cuda or args.gpu in (0, 3), 'GPU must be explicitly reserved')
    lock_name = '.ng-gpu-' + str(args.gpu) if cuda else '.ng83-operator-' + name
    handlers = {s: signal.signal(s, lambda *_: stopped.append(True))
                for s in (signal.SIGINT, signal.SIGTERM)}
    args.worker_script = 'ng83_operator.py'
    result = dict(status='failed_preserve_attempt', completed=[])
    try:
        with (args.research_root / '.ng80-cpu.lock').open('a') as shared:
            fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with (args.research_root / (lock_name + '.lock')).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if cuda:
                    memory, utilization = campaign.guard.gpu_state(args.gpu)
                    require(memory < 100 and utilization == 0, 'GPU occupied; no eviction')
                campaign.io.write(args.run / (name + '-controller-start.json'),
                    dict(pid=os.getpid(), started_unix=time.time(), physical_gpu=args.gpu if cuda else None))
                campaign.guard.phase(args, name, stopped, cuda=cuda, limit_seconds=5400)
                result = dict(status='all_phases_complete', completed=[name])
    except BaseException as exc:
        result['error'] = str(exc)
        raise
    finally:
        result['finished_unix'] = time.time()
        campaign.io.write(args.run / (name + '-controller-exit.json'), result)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'worker', 'supervise'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--gpu', type=int, choices=(0, 3))
    args = parser.parse_args()
    args.research_root, args.run = args.research_root.resolve(strict=True), args.run.resolve()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    elif args.mode == 'verify':
        verify(args.research_root, args.run)
    elif args.mode == 'worker':
        require(args.phase in PHASES, 'phase required')
        worker(args)
    else:
        supervise(args)
