"""Frozen TRAIN_DIAGNOSTIC endpoints for the completed NG80 F/D controls."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np

import ng80_campaign as campaign
import ng80_fit as fit
from ng80_data import DOMAINS, metrics as panel_metrics, require
import review_ng80_fit as audit


MODELS = ('teacher', 'initial', 'F', 'D')
LANES = {'audit': ('audit-training',),
         'frozen': ('encode-initial', 'encode-F'), 'adapted': ('encode-D',),
         'ranking': (*(f'rank-{m}' for m in MODELS), 'review')}
PHASES = tuple(p for values in LANES.values() for p in values)


def verify(base, run):
    manifest = campaign.read(run / 'inputs.json')
    require(manifest['protocol'] == 'NG80_FD_terminal3072_diagnostic_v1'
            and manifest['phases'] == list(PHASES)
            and manifest['models'] == list(MODELS)
            and manifest['training_inputs'] == audit.INPUTS
            and manifest['optimizer_updates'] == 0
            and manifest['bootstrap_seed'] == 80080
            and manifest['bootstrap_replicates'] == 10000
            and not manifest['locked_test_access']
            and not manifest['independent_holdout'], 'endpoint contract changed')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in manifest[key].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve())
                    and campaign.io.sha(path) == digest, 'changed input: ' + name)
    return manifest


def freeze(base, run):
    require(run.parent == base / 'NG-0080', 'fresh NG80 endpoint required')
    deps, sources = {}, {}
    for arm, digest in audit.INPUTS.items():
        root = base / f'NG-0080/dense-{arm}-v1'
        require(campaign.io.sha(root / 'inputs.json') == digest, 'training input')
        manifest = fit.verify(base, root)
        controller = campaign.read(root / 'controller-exit.json')
        require(controller['status'] == 'all_phases_complete'
                and controller['completed'] == list(fit.PHASES), 'training unfinished')
        for phase in fit.PHASES:
            audit.sealed(root / phase)
        deps.update(manifest['dependencies'])
        for path in root.rglob('*'):
            if path.is_file():
                deps[str(path.relative_to(base))] = campaign.io.sha(path)
        for name in manifest['source']:
            if name.endswith('.py') or name == 'ng0071-ranking-config.json':
                sources.setdefault(name, root / name)
    source = Path(__file__).resolve().parent
    for name in ('ng80_endpoints.py', 'review_ng80_fit.py'):
        sources[name] = source / name
    sources['test_ng80_endpoints.py'] = source.parent / 'tests/test_ng80_endpoints.py'
    sources['ng0080-dense-endpoints.zh.md'] = (source.parent /
        'docs/research-sae/reports/ng0001-ng0099/ng0080-dense-endpoints.zh.md')
    parent = base / fit.PARENT
    queries = campaign.read(parent / 'prepare/queries.json')
    selected = campaign.read(parent / 'prepare/selection.json')
    indices = selected['diagnostic']
    require(len(indices) == len(set(indices)) == 1536
            and not set(indices) & set(selected['train'])
            and Counter(queries[i]['subset'] for i in indices)
            == dict.fromkeys(DOMAINS, 512), 'diagnostic split')
    run.mkdir(exist_ok=False)
    for name, path in sources.items():
        shutil.copy2(path, run / name)
    campaign.io.write(run / 'query-indices.json', indices)
    campaign.io.write(run / 'inputs.json', dict(
        protocol='NG80_FD_terminal3072_diagnostic_v1', phases=list(PHASES),
        models=list(MODELS), training_inputs=audit.INPUTS, dependencies=deps,
        source={p.name: campaign.io.sha(p) for p in run.iterdir() if p.is_file()},
        optimizer_updates=0, bootstrap_seed=80080, bootstrap_replicates=10000,
        locked_test_access=False, independent_holdout=False,
        automatic_promotion=False, full_hybrid_evaluated=False,
        created_unix=time.time()))
    verify(base, run)
    print('NG80_ENDPOINTS_FROZEN', campaign.io.sha(run / 'inputs.json'), flush=True)


def encode(base, run, output, arm, task):
    import torch
    from ng71_training import Encoder, parameter_hash
    from ng80_training import Readout

    root = base / f'NG-0080/dense-{"F" if arm == "initial" else arm}-v1'
    terminal = root / 'train-3072'
    initial = arm == 'initial'
    config = campaign.read(root / 'config.json')
    weights = np.load(base / fit.PARENT / 'diagnose/mean-projection.npy')
    model = Readout(Encoder(base, config, 'cuda',
                    checkpoint=None if initial else terminal / 'checkpoint'),
                    'F' if initial else arm, weights)
    expected = campaign.read(root / ('train-canary' if initial else 'train-3072') / 'results.json')
    if not initial:
        model.projection.load_state_dict(torch.load(terminal / 'projection.pt',
            weights_only=True, map_location='cuda'), strict=True)
    before = (parameter_hash(model.encoder.model), parameter_hash(model.projection))
    prefix = 'initial_' if initial else ''
    require(before == (expected[prefix + 'model_sha256'],
                       expected[prefix + 'projection_sha256']), 'encoded parameter identity')
    queries = campaign.read(base / fit.PARENT / 'prepare/queries.json')
    indices = campaign.read(run / 'query-indices.json')
    documents = campaign.rows(base / campaign.DATA / 'documents.jsonl')
    require(len(documents) == 233009, 'full corpus changed')
    counts = {}
    with torch.no_grad():
        for role, texts in (('query', [queries[i]['query'] for i in indices]),
                            ('document', [r['text'] for r in documents])):
            destination = output / f'{role}.npy'
            require(not destination.exists(), 'never overwrite encoded output')
            values = np.lib.format.open_memmap(destination, mode='w+',
                dtype=np.float32, shape=(len(texts), 1024))
            tick = time.monotonic()
            for start in range(0, len(texts), 4):
                chunk = model.encode(texts[start:start + 4], role).cpu().numpy()
                require(np.isfinite(chunk).all(), 'nonfinite fresh codes')
                np.testing.assert_allclose(np.linalg.norm(chunk, axis=1), 1.,
                                            rtol=0, atol=2e-6)
                values[start:start + len(chunk)] = chunk
                if start == 60:
                    campaign.canary(output, role, time.monotonic() - tick, 64, len(texts))
                if start % 4096 == 0:
                    print('NG80_ENDPOINT_ENCODE', arm, role, start + len(chunk), len(texts), flush=True)
                    task.get_logger().report_scalar('encode', role, start + len(chunk), start)
            values.flush()
            counts[role] = dict(rows=len(texts), seconds=time.monotonic() - tick)
    require(before == (parameter_hash(model.encoder.model), parameter_hash(model.projection)),
            'inference changed model')
    campaign.io.write(output / 'query-ids.json', [queries[i]['query_id'] for i in indices])
    return dict(model=arm, model_sha256=before[0], projection_sha256=before[1],
        checkpoint_complete_sha256=None if initial else campaign.io.sha(terminal / 'complete.json'),
        roles=counts, optimizer_updates=0, fresh=True, locked_test_access=False)


def rank_row(scores, alternate, gold):
    scores, alternate = np.asarray(scores), np.asarray(alternate)
    require(scores.ndim == 1 and scores.shape == alternate.shape
            and np.isfinite(scores).all() and np.isfinite(alternate).all()
            and gold and len(gold) == len(set(gold))
            and all(type(g) is int and 0 <= g < len(scores) for g in gold), 'ranking universe')
    error = float(np.abs(scores - alternate).max())
    require(error <= 1e-12, 'full-background reductions differ')
    order = np.lexsort((np.arange(len(scores)), -scores))
    inverse = np.empty(len(scores), dtype=np.int64)
    inverse[order] = np.arange(1, len(scores) + 1)
    ranks = inverse[gold]
    ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(gold))))
    return dict(gold_ids=gold, gold_ranks=ranks.tolist(),
        ndcg10=float(np.sum(1 / np.log2(ranks[ranks <= 10] + 1)) / ideal),
        recall100=float(np.mean(ranks <= 100)), top100=order[:100].tolist(),
        top100_scores=scores[order[:100]].tolist(), reduction_error=error)


def codes(base, run, model):
    if model == 'teacher':
        q = np.load(base / fit.PARENT / 'encode-teacher/query.npy', mmap_mode='r')
        q = q[campaign.read(run / 'query-indices.json')]
        d = np.load(base / campaign.DENSE, mmap_mode='r')
    else:
        folder = run / ('encode-' + model)
        q, d = [np.load(folder / f'{r}.npy', mmap_mode='r') for r in ('query', 'document')]
    require(q.shape == (1536, 1024) and d.shape == (233009, 1024), 'code dimensions')
    return q, d


def rank(base, run, output, model):
    q, d = [a.astype(np.float64) for a in codes(base, run, model)]
    teacher_q, teacher_d = codes(base, run, 'teacher')
    parent = base / fit.PARENT
    queries = campaign.read(parent / 'prepare/queries.json')
    panels, labels = [campaign.read(parent / f'prepare/{n}.json') for n in ('panels', 'labels')]
    indices = campaign.read(run / 'query-indices.json')
    teacher_rows = (None if model == 'teacher' else
                    campaign.rows(run / 'rank-teacher/rankings.jsonl'))
    tick, error = time.monotonic(), 0.
    with (output / 'rankings.jsonl').open('x') as stream:
        for i, index in enumerate(indices):
            scores = d @ q[i]
            other = np.einsum('ij,j->i', d, q[i], optimize=False)
            row = rank_row(scores, other, labels[index])
            pool = panels[index]
            target = teacher_d[pool].astype(np.float64) @ teacher_q[i].astype(np.float64)
            row.update(panel_metrics(scores[pool], target, pool, labels[index]))
            teacher_head = row['top100'] if teacher_rows is None else teacher_rows[i]['top100']
            row.update(query_id=queries[index]['query_id'], domain=queries[index]['subset'],
                surface='TRAIN_DIAGNOSTIC', teacher_full_top10_overlap=
                len(set(row['top100'][:10]) & set(teacher_head[:10])) / 10)
            error = max(error, row['reduction_error'])
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            if i == 15:
                campaign.canary(output, 'ranking', time.monotonic() - tick, 16, len(indices))
            if i % 64 == 0:
                stream.flush()
                print('NG80_ENDPOINT_RANK', model, i + 1, len(indices), flush=True)
    return dict(model=model, queries=len(indices), full_background_documents=len(d),
        max_reduction_error=error, full_hybrid_evaluated=False,
        native_cost_evaluated=False, independent_holdout=False, locked_test_access=False)


def paired(delta, domains):
    delta, domains = np.asarray(delta), np.asarray(domains)
    require(delta.shape == (len(domains), 2) and np.isfinite(delta).all(), 'paired shape')
    rng, means, by_domain = np.random.default_rng(80080), [], {}
    for domain in DOMAINS:
        values = delta[domains == domain]
        require(len(values) > 0, 'paired domain missing')
        by_domain[domain] = values.mean(0).tolist()
        draws = rng.integers(len(values), size=(10000, len(values)))
        means.append(values[draws].mean(1))
    return dict(metrics=['ndcg10', 'recall100'], domains=by_domain,
        macro=np.mean(list(by_domain.values()), axis=0).tolist(),
        ci95=np.quantile(np.mean(means, axis=0), [.025, .975], axis=0).T.tolist(),
        bootstrap_seed=80080, replicates=10000, independent_holdout=False,
        conditional_on_single_initialization_and_order=True)


def review(base, run):
    parent = base / fit.PARENT
    labels = campaign.read(parent / 'prepare/labels.json')
    queries = campaign.read(parent / 'prepare/queries.json')
    indices = campaign.read(run / 'query-indices.json')
    arrays, quality, max_error = {}, {}, 0.
    last_train = max(campaign.read(base / f'NG-0080/dense-{a}-v1/train-3072/exit.json')
                     ['finished_unix'] for a in ('F', 'D'))
    for phase in PHASES[:-1]:
        audit.sealed(run / phase)
        require(campaign.read(run / phase / 'started.json')['started_unix'] >= last_train,
                'endpoint overlaps unfinished training')
    fields = ('ndcg10', 'recall100', 'centered_score_mse',
              'panel_ndcg10', 'teacher_full_top10_overlap')
    for model in MODELS:
        records = campaign.rows(run / f'rank-{model}/rankings.jsonl')
        require(len(records) == len(indices), 'incomplete rankings')
        q, d = codes(base, run, model)
        for i, (record, index) in enumerate(zip(records, indices, strict=True)):
            gold = labels[index]
            require(record['query_id'] == queries[index]['query_id']
                    and record['domain'] == queries[index]['subset']
                    and record['gold_ids'] == gold, 'rank identity/labels')
            ranks = record['gold_ranks']
            require(len(ranks) == len(gold) and len(set(ranks)) == len(ranks)
                    and all(type(r) is int and 1 <= r <= len(d) for r in ranks), 'gold ranks')
            ndcg = sum(1 / np.log2(r + 1) for r in ranks if r <= 10)
            ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(gold))))
            np.testing.assert_allclose([record['ndcg10'], record['recall100']],
                [ndcg / ideal, sum(r <= 100 for r in ranks) / len(gold)], rtol=0, atol=1e-12)
            head = record['top100']
            require(len(head) == len(set(head)) == 100
                    and all(type(h) is int and 0 <= h < len(d) for h in head), 'head IDs')
            for doc, position in zip(gold, ranks, strict=True):
                require((position <= 100 and head[position - 1] == doc)
                        or (position > 100 and doc not in head), 'gold/head disagreement')
            coordinates = np.sum(d[head].astype(np.float64) * q[i].astype(np.float64), axis=1)
            difference = float(np.abs(coordinates - record['top100_scores']).max())
            require(difference <= 1e-12, 'independent head score mismatch')
            max_error = max(max_error, difference)
        arrays[model] = np.asarray([[r[k] for k in fields[:2]] for r in records])
        quality[model] = {domain: {k: float(np.mean([r[k] for r in records
            if domain == 'macro' or r['domain'] == domain])) for k in fields}
            for domain in ('macro', *DOMAINS)}
    domains = [queries[i]['subset'] for i in indices]
    differences = {a + '-minus-' + b: paired(arrays[a] - arrays[b], domains)
                   for a, b in (('F', 'initial'), ('D', 'initial'), ('D', 'F'),
                                 ('F', 'teacher'), ('D', 'teacher'))}
    return dict(passed=True, quality=quality, comparisons=differences,
        independent_coordinate_error=max_error, full_hybrid_evaluated=False,
        native_cost_evaluated=False, independent_holdout=False, automatic_promotion=False,
        locked_test_access=False, remaining=['S/P actual sparse training',
            'multi-source supervision', 'context control', 'DF joint mask', 'native hybrid cost'])


def worker(args):
    import torch
    from clearml import Task

    verify(args.research_root, args.run)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(80080)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0080/FD-endpoint/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output, base, run = args.run / args.phase, args.research_root, args.run
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    campaign.io.write(output / 'clearml-start.json', receipt)
    task.connect(dict(input_sha256=campaign.io.sha(run / 'inputs.json'), phase=args.phase))
    try:
        if args.phase == 'audit-training':
            result = audit.review(base)
        else:
            require(audit.sealed(run / 'audit-training')['passed'], 'audit barrier')
            if args.phase.startswith('encode-'):
                result = encode(base, run, output, args.phase.removeprefix('encode-'), task)
            else:
                for model in ('initial', 'F', 'D'):
                    audit.sealed(run / ('encode-' + model))
                if args.phase == 'review':
                    result = review(base, run)
                else:
                    if args.phase != 'rank-teacher':
                        audit.sealed(run / 'rank-teacher')
                    result = rank(base, run, output, args.phase.removeprefix('rank-'))
        verify(base, run)
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
    require(args.lane in LANES, 'lane required')
    cuda = args.lane in ('frozen', 'adapted')
    require(not cuda or args.gpu in (0, 3), 'permitted explicit GPU required')
    require(not (args.run / f'{args.lane}-controller-start.json').exists(), 'never restart lane')
    stopped = []
    handlers = {sig: signal.signal(sig, lambda *_: stopped.append(True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    path = args.research_root / (f'.ng-gpu-{args.gpu}.lock' if cuda else '.ng80-cpu.lock')
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cuda:
            memory, use = campaign.guard.gpu_state(args.gpu)
            require(memory < 100 and use == 0, 'GPU occupied; never evict')
        campaign.io.write(args.run / f'{args.lane}-controller-start.json',
            dict(pid=os.getpid(), physical_gpu=args.gpu if cuda else None, started_unix=time.time()))
        args.worker_script = 'ng80_endpoints.py'
        done = []
        try:
            for phase in LANES[args.lane]:
                campaign.guard.phase(args, phase, stopped, cuda=cuda, limit_seconds=5400)
                done.append(phase)
            campaign.io.write(args.run / f'{args.lane}-controller-exit.json',
                dict(status='all_phases_complete', completed=done, finished_unix=time.time()))
        except BaseException as exc:
            campaign.io.write(args.run / f'{args.lane}-controller-exit.json',
                dict(status='failed_preserve_attempt', completed=done, error=str(exc), finished_unix=time.time()))
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--lane', choices=LANES)
    parser.add_argument('--gpu', type=int, choices=(0, 3))
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    elif args.mode == 'worker':
        require(args.phase in PHASES, 'phase required')
        worker(args)
    elif args.mode == 'supervise':
        supervise(args)
    else:
        verify(args.research_root, args.run)
