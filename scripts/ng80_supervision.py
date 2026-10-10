"""Closed-TRAIN-only geometry, supervision connectivity and visibility audit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np

import ng80_campaign as campaign
import ng80_data as data


PARENT = 'NG-0080/readout-v1'
PARENT_SHA = 'd8dca9c36ff49047c0fc0acfe0ee9cbcd607537a65596d9013300537f2e8dff1'


def spectrum(values):
    data.require(values.ndim == 2 and len(values) > 0
                 and np.isfinite(values).all(), 'invalid teacher vectors')
    second = values.astype(np.float64).T @ values.astype(np.float64) / len(values)
    eigen = np.linalg.eigvalsh(second).clip(0)[::-1]
    data.require(eigen.sum() > 0, 'empty geometry')
    probability = eigen / eigen.sum()
    positive = probability[probability > 0]
    return {'rows': len(values), 'dimensions': values.shape[1],
            'participation_rank': float(1 / (probability @ probability)),
            'entropy_rank': float(np.exp(-(positive * np.log(positive)).sum())),
            'mass_at': {str(k): float(probability[:k].sum())
                        for k in (32, 64, 128, 256, 384, 512)},
            'eigenvalues': eigen.tolist(), 'query_centered': False}


def connectivity(scores, positive):
    scores, positive = np.asarray(scores), np.asarray(positive, dtype=bool)
    data.require(scores.ndim == 1 and scores.shape == positive.shape
                 and len(scores) > 1 and positive.any() and np.isfinite(scores).all(),
                 'invalid supervision graph')
    parent = list(range(len(scores)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges, saturated = 0, 0
    for p in np.flatnonzero(positive):
        for n in np.flatnonzero(~positive):
            margin = float(scores[p] - scores[n])
            if margin > 0:
                parent[find(p)] = find(n)
                edges += 1
                saturated += int(margin / .04 >= np.log(99))
    components = len({find(i) for i in range(len(scores))})
    possible = int(positive.sum() * (~positive).sum())
    return {'documents': len(scores), 'positive_nonpositive_pairs': possible,
            'eligible_pairs': edges, 'teacher_opposed_or_tied_pairs': possible - edges,
            'target_at_least_099_pairs': saturated, 'components': components,
            'independent_exact_margin_constraints': len(scores) - components,
            'full_field_constraints': len(scores) - 1,
            'unconstrained_relative_degrees': components - 1}


def audit(base, output):
    parent = base / PARENT
    read = campaign.read
    queries = read(parent / 'prepare/queries.json')
    selection = read(parent / 'prepare/selection.json')
    panels = read(parent / 'prepare/panels.json')
    labels = read(parent / 'prepare/labels.json')
    q = np.load(parent / 'encode-teacher/query.npy', mmap_mode='r')
    d = np.load(base / campaign.DENSE, mmap_mode='r')
    data.require(q.shape == (13824, 1024) and d.shape == (233009, 1024), 'corpus alignment')
    train = selection['train']
    populations = {'old_prefix': list(range(1536)), 'all_fit': train}
    fresh = []
    for domain in data.DOMAINS:
        eligible = [i for i in train if i >= 1536 and queries[i]['subset'] == domain]
        eligible.sort(key=lambda i: hashlib.sha256(
            ('NG80/geometry-fresh-v1/' + queries[i]['query_id']).encode()).hexdigest())
        data.require(len(eligible) >= 512, 'fresh domain shortfall')
        fresh += eligible[:512]
    populations['fresh_1536'] = fresh
    geometry = {name: spectrum(q[ids]) for name, ids in populations.items()}
    # Raw query second moment is the score-observability object, not centered PCA.
    old = q[:1536].astype(np.float64)
    _, directions = np.linalg.eigh(old.T @ old / len(old))
    top = directions[:, -128:]
    projection_residual = {}
    for name, ids in populations.items():
        values = q[ids].astype(np.float64)
        residual = (values * values).sum(1) - ((values @ top) ** 2).sum(1)
        projection_residual[name] = float(residual.clip(0).mean())
    records = []
    with (output / 'connectivity.jsonl').open('x') as stream:
        for i in train:
            pool = panels[i]
            scores = d[pool].astype(np.float64) @ q[i].astype(np.float64)
            row = connectivity(scores, [doc in labels[i] for doc in pool])
            row.update(query_id=queries[i]['query_id'], domain=queries[i]['subset'])
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            records.append(row)
    graph = {}
    for domain in ('macro', *data.DOMAINS):
        rows = [r for r in records if domain == 'macro' or r['domain'] == domain]
        graph[domain] = {'queries': len(rows),
            'fully_connected_queries': sum(r['components'] == 1 for r in rows),
            'mean': {key: float(np.mean([r[key] for r in rows])) for key in (
                'documents', 'positive_nonpositive_pairs', 'eligible_pairs',
                'teacher_opposed_or_tied_pairs', 'target_at_least_099_pairs',
                'independent_exact_margin_constraints', 'full_field_constraints',
                'unconstrained_relative_degrees')}}
    hidden = parent / 'encode-hidden'
    dq = np.load(hidden / 'query-token-lengths.npy')
    dd = np.load(hidden / 'document-token-lengths.npy')
    tq = np.load(parent / 'encode-teacher/query-token-lengths.npy')
    visibility = {}
    for domain in data.DOMAINS:
        ids = [i for i in train if queries[i]['subset'] == domain]
        positive_lengths = [int(dd[doc]) for i in ids for doc in labels[i]]
        visibility[domain] = {'queries': len(ids),
            'student_query_over_64': int((dq[ids] > 64).sum()),
            'teacher_query_over_64': int((tq[ids] > 64).sum()),
            'positive_document_exposures': len(positive_lengths),
            'positive_document_exposures_over_256': sum(n > 256 for n in positive_lengths)}
    return {'teacher_geometry': geometry, 'old_top128_residual_energy': projection_residual,
            'supervision_graph': graph, 'input_visibility': visibility,
            'normalized_unique_fit_queries': len({data.normalized(queries[i]['query']) for i in train}),
            'independent_holdout': False, 'task_diversity_measured': False,
            'label_error_rate_measured': False, 'true_evidence_visibility_measured': False,
            'original_D_metric_weights_replayed': False, 'training_updates': 0,
            'locked_test_access': False, 'native_cost_measured': False}


def verify(base, run):
    manifest = campaign.read(run / 'inputs.json')
    data.require(manifest['protocol'] == 'NG80_cached_supervision_v1'
                 and manifest['parent_input_sha256'] == PARENT_SHA, 'protocol changed')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in manifest[key].items():
            path = root / name
            data.require(path.resolve().is_relative_to(root.resolve()), 'path escape')
            data.require(campaign.io.sha(path) == digest, 'changed input: ' + name)


def freeze(base, run):
    parent = base / PARENT
    data.require(campaign.io.sha(parent / 'inputs.json') == PARENT_SHA, 'parent changed')
    original = campaign.verify(base, parent)
    dependencies = dict(original['dependencies'])
    for phase in ('prepare', 'encode-teacher', 'encode-hidden'):
        campaign.closed(parent, phase)
        proof = parent / phase / 'complete.json'
        for name, digest in campaign.read(proof)['files'].items():
            dependencies[str((parent / phase / name).relative_to(base))] = digest
        dependencies[str(proof.relative_to(base))] = campaign.io.sha(proof)
    dependencies[PARENT + '/inputs.json'] = PARENT_SHA
    data.require(run.parent == base / 'NG-0080', 'fresh NG80 unit required')
    run.mkdir(exist_ok=False)
    for name in original['source']:
        shutil.copy2(parent / name, run / name)
    source = Path(__file__).resolve().parent
    for path in (Path(__file__), source.parent / 'tests/test_ng80_supervision.py',
                 source.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0080-supervision-audit.zh.md'):
        shutil.copy2(path, run / path.name)
    campaign.io.write(run / 'inputs.json', {
        'protocol': 'NG80_cached_supervision_v1', 'parent_input_sha256': PARENT_SHA,
        'dependencies': dependencies,
        'source': {p.name: campaign.io.sha(p) for p in run.iterdir() if p.is_file()},
        'locked_test_access': False, 'training_enabled': False})
    verify(base, run)
    print('NG80_AUDIT_FROZEN', campaign.io.sha(run / 'inputs.json'), flush=True)


def worker(args):
    from clearml import Task

    verify(args.research_root, args.run)
    data.require(args.phase == 'audit', 'unknown phase')
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0080/supervision-audit',
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    campaign.io.write(output / 'clearml-start.json', receipt)
    task.connect({'input_sha256': campaign.io.sha(args.run / 'inputs.json')})
    try:
        result = audit(args.research_root, output)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'worker', 'supervise', 'verify'))
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--phase')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    elif args.mode == 'worker':
        worker(args)
    else:
        verify(args.research_root, args.run)
        if args.mode == 'supervise':
            args.worker_script, args.gpu = 'ng80_supervision.py', None
            stopped = []
            handlers = {sig: signal.signal(sig, lambda *_: stopped.append(True))
                        for sig in (signal.SIGINT, signal.SIGTERM)}
            campaign.io.write(args.run / 'controller-start.json',
                              {'pid': os.getpid(), 'started_unix': time.time()})
            try:
                campaign.guard.phase(args, 'audit', stopped, cuda=False)
                campaign.io.write(args.run / 'controller-exit.json',
                                  {'status': 'all_phases_complete', 'finished_unix': time.time()})
            except BaseException as exc:
                campaign.io.write(args.run / 'controller-exit.json',
                    {'status': 'failed_preserve_attempt', 'error': str(exc),
                     'finished_unix': time.time()})
                raise
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


if __name__ == '__main__':
    main()
