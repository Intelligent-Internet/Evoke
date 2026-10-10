"""NG82 teacher geometry and joint-mask diagnostics; no model updates."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse

import ng81_fusion as prev


old, campaign, audit, require = prev.old, prev.campaign, prev.audit, prev.require
PARENT = 'NG-0081/fusion-v1'
INPUT_SHA = 'ba73d53e688c49ec17c40a41cafabdd17eab2a900b9ea54e9f9f38e1aa831bca'
REVIEW_SHA = '4bb1e6783776fc79e7a2a6ab41d6fce0ea5eb8922c3b20f0ffb56d0cfeb47a84'
PHASES = ('mask', 'geometry')
PAIRS = ((0, 9), (9, 10), (9, 49), (49, 99), (99, 100), (99, 199))
DIMS = (128, 256)
PROTOCOL = 'ng0082-potion-review-and-execution.zh.md'


def projection(documents, queries, dimension, weighted):
    documents = np.asarray(documents, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    width = documents.shape[1]
    require(documents.ndim == queries.ndim == 2 and queries.shape[1] == width
            and 0 < dimension <= width and np.isfinite(documents).all()
            and np.isfinite(queries).all(), 'projection inputs')
    mean = documents.mean(0)
    centered = documents - mean
    covariance = centered.T @ centered / len(centered)
    if weighted:
        moment = queries.T @ queries / len(queries)
        ridge = max(float(np.trace(moment) / width * 1e-4), 1e-12)
        values, vectors = np.linalg.eigh(moment + ridge * np.eye(width))
        root = (vectors * np.sqrt(values)) @ vectors.T
        inverse = (vectors / np.sqrt(values)) @ vectors.T
        metric = root @ covariance @ root
    else:
        ridge = 0.
        root = inverse = np.eye(width)
        metric = covariance
    values, vectors = np.linalg.eigh((metric + metric.T) / 2)
    basis = vectors[:, -dimension:][:, ::-1]
    return mean, root @ basis, inverse @ basis, dict(
        ridge=ridge, retained_objective_fraction=float(
            values[-dimension:].sum() / max(values.sum(), 1e-30)),
        fractions={str(k): float(values[-k:].sum() / max(values.sum(), 1e-30))
                   for k in DIMS if k <= dimension})


def utility(contributions, margins, tau):
    require(sparse.isspmatrix_csr(contributions)
            and contributions.shape[0] == len(margins)
            and np.isfinite(contributions.data).all()
            and np.isfinite(margins).all() and np.isfinite(tau) and tau > 0,
            'utility inputs')
    row = np.repeat(np.arange(len(margins)), np.diff(contributions.indptr))
    before = np.logaddexp(0, -margins[row] / tau)
    after = np.logaddexp(0, -(margins[row] - contributions.data) / tau)
    return np.bincount(contributions.indices, weights=after - before,
                       minlength=contributions.shape[1]) / len(margins)


def mask_to_budget(cost, order, target):
    cost = np.asarray(cost, dtype=np.float64)
    require(cost.ndim == 1 and (cost >= 0).all() and np.isfinite(cost).all()
            and np.isfinite(target) and 0 <= target <= cost.sum()
            and sorted(order) == list(range(len(cost))), 'mask budget inputs')
    keep = np.ones(len(cost), dtype=bool)
    remaining = float(cost.sum())
    for term in order:
        if remaining <= target:
            break
        if cost[term] > 0:
            keep[term] = False
            remaining -= cost[term]
    return keep


def masked_query(query, keep):
    result = query.copy()
    result.data *= keep[result.indices]
    result.eliminate_zeros()
    return result


def compact_rank(scores, labels):
    result = prev.ranked(scores, labels)
    return {k: result[k] for k in ('gold_ids', 'gold_ranks', 'ndcg10',
                                   'recall100', 'top100')}


def summarize(records, key):
    domains = [r['domain'] for r in records]
    arrays = {p: np.array([prev.metrics(r[key][p]['gold_ranks']) for r in records])
              for p in records[0][key]}
    result = {}
    for p, values in arrays.items():
        result[p] = dict(macro=prev.macro(values, domains).tolist(), domains={
            d: values[np.array(domains) == d].mean(0).tolist() for d in old.DOMAINS})
    return result


def paired(records, left, right, key='rankings'):
    differences = np.array([np.array(prev.metrics(r[key][left]['gold_ranks']))
                            - prev.metrics(r[key][right]['gold_ranks']) for r in records])
    rng = np.random.default_rng(82082)
    groups = [np.array([i for i, r in enumerate(records) if r['domain'] == d])
              for d in old.DOMAINS]
    require(all(len(g) for g in groups), 'bootstrap domains')
    samples = []
    for _ in range(2000):
        samples.append(np.mean([differences[rng.choice(g, len(g))].mean(0)
                                for g in groups], axis=0))
    return dict(mean=differences.mean(0).tolist(),
                ci95=np.quantile(samples, [.025, .975], axis=0).T.tolist())


def paths(base):
    endpoint = base / prev.PARENT
    encoded = old.phase_root(base, endpoint, 'encode-S')
    initial = old.phase_root(base, endpoint, 'encode-initial')
    root = base / old.dense_fit.PARENT
    return dict(
        documents=base / campaign.DENSE,
        queries=root / 'encode-teacher/query.npy',
        records=root / 'prepare/queries.json', labels=root / 'prepare/labels.json',
        split=root / 'prepare/selection.json',
        selection=base / PARENT / 'selection.json',
        sd=encoded / 'document.npz', sq=encoded / 'query.npz',
        sq_ids=encoded / 'query-ids.json',
        scq=base / PARENT / 'encode-calibration/S.npz',
        scq_ids=base / PARENT / 'encode-calibration/query-ids.json',
        initial_d=initial / 'document.npz',
        initial_q=base / PARENT / 'encode-calibration/initial.npz',
        ld=base / old.fit.preflight.LD,
        lq=endpoint / 'prepare-lexical/diagnostic.npz',
        lcq=endpoint / 'prepare-lexical/calibration.npz',
        baselines=endpoint / 'rank-baselines/rankings.jsonl',
        s_baseline=endpoint / 'rank-S/rankings.jsonl')


def freeze(base, run):
    parent = base / PARENT
    require(run.parent == base / 'NG-0082' and not run.exists(), 'fresh NG82 run')
    require(campaign.io.sha(parent / 'inputs.json') == INPUT_SHA
            and campaign.io.sha(parent / 'review/results.json') == REVIEW_SHA,
            'NG81 parent identity')
    previous = prev.verify(base, parent)
    for phase in prev.PHASES:
        audit.sealed(parent / phase)
    require(campaign.read(parent / 'controller-exit.json')['status']
            == 'all_phases_complete', 'NG81 unfinished')
    run.mkdir(parents=True, exist_ok=False)
    for name in previous['source']:
        shutil.copy2(parent / name, run / name)
    source = Path(__file__).resolve().parent
    for name, location in (('ng82_diagnostics.py', source),
            ('test_ng82_diagnostics.py', source.parent / 'tests'),
            (PROTOCOL, source.parent / 'docs/research-sae/reports/ng0001-ng0099')):
        shutil.copy2(location / name, run / name)
    files = paths(base)
    require(all(str(p.relative_to(base)) in previous['dependencies']
                or p.is_relative_to(parent) for p in files.values()),
            'new input not audited by NG81')
    dependencies = {str(p.relative_to(base)): campaign.io.sha(p) for p in files.values()}
    dependencies.update({PARENT + '/' + name: campaign.io.sha(parent / name)
                         for name in ('inputs.json', 'review/results.json')})
    campaign.io.write(run / 'inputs.json', dict(protocol='NG82_bounded_diagnostics_v1',
        phases=list(PHASES), dimensions=list(DIMS), teacher_pairs=list(map(list, PAIRS)),
        files={k: str(p.relative_to(base)) for k, p in files.items()},
        dependencies=dependencies, source={p.name: campaign.io.sha(p)
                                          for p in run.iterdir() if p.is_file()},
        optimizer_updates=0, locked_test_access=False, independent_holdout=False,
        native_cost_evaluated=False, automatic_promotion=False))
    verify(base, run)
    print('NG82_FROZEN', campaign.io.sha(run / 'inputs.json'), flush=True)


def verify(base, run):
    data = campaign.read(run / 'inputs.json')
    require(data['protocol'] == 'NG82_bounded_diagnostics_v1'
            and data['phases'] == list(PHASES) and data['dimensions'] == list(DIMS)
            and data['teacher_pairs'] == list(map(list, PAIRS))
            and data['optimizer_updates'] == 0 and not data['locked_test_access']
            and not data['independent_holdout'] and not data['automatic_promotion'],
            'NG82 protocol identity')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in data[key].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve())
                    and campaign.io.sha(path) == digest, 'changed input: ' + name)
    require(data['dependencies'][PARENT + '/inputs.json'] == INPUT_SHA
            and data['dependencies'][PARENT + '/review/results.json'] == REVIEW_SHA,
            'NG81 anchor changed')
    require(data['files'] == {k: str(p.relative_to(base)) for k, p in paths(base).items()},
            'input paths changed')
    return {k: base / v for k, v in data['files'].items()}


def load_context(files):
    records = campaign.read(files['records'])
    labels = campaign.read(files['labels'])
    selected = campaign.read(files['selection'])
    split = campaign.read(files['split'])
    require(len(selected['calibration']) == 384 and len(selected['diagnostic']) == 1536
            and set(selected['calibration']) <= set(split['train'])
            and selected['diagnostic'] == split['diagnostic']
            and not set(split['train']) & set(split['diagnostic']), 'split identity')
    for role, name in (('calibration', 'scq_ids'), ('diagnostic', 'sq_ids')):
        require(campaign.read(files[name]) == [records[i]['query_id']
                for i in selected[role]], 'sparse/query identity')
    dense_d = np.load(files['documents'], mmap_mode='r')
    dense_q = np.load(files['queries'], mmap_mode='r')
    require(dense_d.shape == (233009, 1024) and dense_q.shape == (13824, 1024),
            'teacher shape')
    return records, labels, selected, split, dense_d, dense_q


def geometry(files, output):
    records, labels, selected, split, source_d, source_q = load_context(files)
    started = time.monotonic()
    sample = np.random.default_rng(82082).choice(len(source_d), 65536, replace=False)
    campaign.io.write(output / 'fit-document-ids.json', sample.tolist())
    fit_q = source_q[split['train']].astype(np.float64)
    fit_d = source_d[sample].astype(np.float64)
    ids = selected['diagnostic']
    queries = source_q[ids].astype(np.float64)
    documents = source_d.astype(np.float64)
    ld = sparse.load_npz(files['ld']).astype(np.float64).tocsc()
    lq = sparse.load_npz(files['lq'])
    previous = campaign.rows(files['baselines'])
    results = [dict(query_id=records[i]['query_id'], domain=records[i]['subset'],
                    rankings={}, geometry={}) for i in ids]
    fits, checks = {}, []
    for weighted in (False, True):
        mean, dm, qm, stats = projection(fit_d, fit_q, max(DIMS), weighted)
        for k in DIMS:
            name = ('query_weighted' if weighted else 'pca') + '-' + str(k)
            doccode = (documents - mean) @ dm[:, :k]
            querycode = queries @ qm[:, :k]
            np.savez(output / (name + '.npz'), mean=mean, document_map=dm[:, :k],
                     query_map=qm[:, :k])
            fits[name] = dict(stats, retained_objective_fraction=stats['fractions'][str(k)])
            rank_started = time.monotonic()
            for lo in range(0, len(ids), 16):
                block = querycode[lo:lo + 16] @ doccode.T
                target = queries[lo:lo + 16] @ documents.T
                for offset, index in enumerate(ids[lo:lo + 16]):
                    row = lo + offset
                    score, exact = block[offset], target[offset]
                    lex, _, _ = old.sparse_scores(ld, lq[row])
                    if not weighted and k == DIMS[0]:
                        require(previous[row]['query_id'] == records[index]['query_id'],
                                'teacher ranking identity')
                        reference = compact_rank(exact, labels[index])
                        require(reference['top100'] == previous[row]['rankings']['dense']['top100'],
                                'cached teacher replay')
                        results[row]['rankings']['dense'] = reference
                        results[row]['rankings']['dense_hybrid'] = compact_rank(
                            prev.mix(exact, lex, .1, 'zscore'), labels[index])
                    results[row]['rankings'][name] = compact_rank(score, labels[index])
                    results[row]['rankings'][name + '_hybrid'] = compact_rank(
                        prev.mix(score, lex, .1, 'zscore'), labels[index])
                    top = results[row]['rankings'][name]['top100']
                    goldtop = results[row]['rankings']['dense']['top100']
                    centered = score - score.mean() - exact + exact.mean()
                    results[row]['geometry'][name] = dict(
                        centered_score_mse=float(np.mean(centered ** 2)),
                        top10_overlap=len(set(top[:10]) & set(goldtop[:10])) / 10,
                        top100_overlap=len(set(top) & set(goldtop)) / 100)
                    if row < 4:
                        independent = ((documents[:256] - mean)
                            @ (dm[:, :k] @ qm[:, :k].T)) @ queries[row]
                        np.testing.assert_allclose(score[:256], independent, rtol=0, atol=1e-12)
                        checks.append(float(np.max(np.abs(score[:256] - independent))))
                if lo == 48:
                    rank_seconds = time.monotonic() - rank_started
                    remaining = (5 - len(fits)) * len(ids) - 64
                    estimate = 1.5 * (time.monotonic() - started
                                      + rank_seconds * remaining / 64) + 300
                    campaign.io.write(output / (name + '-canary.json'),
                        dict(projected_phase_seconds=estimate, limit_seconds=5100))
                    require(estimate < 5100, 'geometry phase cannot fit original budget')
                if lo % 256 == 0:
                    print('NG82_GEOMETRY', name, lo + 16, flush=True)
            del doccode, querycode, block, target
            gc.collect()
    # Full-rank weighted maps must recover the original centered score exactly.
    mean, dm, qm, _ = projection(fit_d[:4096], fit_q, 1024, True)
    expected = queries[:8] @ (documents[:128] - mean).T
    actual = (queries[:8] @ qm) @ ((documents[:128] - mean) @ dm).T
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    with (output / 'rankings.jsonl').open('x') as stream:
        for row in results:
            stream.write(json.dumps(row) + '\n')
    quality = summarize(results, 'rankings')
    comparisons = {p: paired(results, p, 'dense') for p in quality if p != 'dense'}
    return dict(quality=quality, versus_dense=comparisons, fitted= fits,
                full_rank_error=float(np.max(np.abs(actual - expected))),
                independent_score_max_error=max(checks), queries=len(ids),
                train_queries=len(split['train']), fitted_documents=len(sample),
                independent_holdout=False, native_cost_evaluated=False)


def mask(files, output):
    records, labels, selected, _, source_d, source_q = load_context(files)
    sd_csr = sparse.load_npz(files['sd']).astype(np.float64)
    sd = sd_csr.tocsc()
    ld = sparse.load_npz(files['ld']).astype(np.float64).tocsc()
    cq = sparse.load_npz(files['scq'])
    lcq = sparse.load_npz(files['lcq'])
    df = np.diff(sd.indptr)
    rate = np.bincount(cq.indices, minlength=sd.shape[1]) / cq.shape[0]
    cost = df * rate
    contributions, margins, teacher_pairs = [], [], []
    started = time.monotonic()
    documents = source_d.astype(np.float64)
    for row, index in enumerate(selected['calibration']):
        teacher = documents @ source_q[index].astype(np.float64)
        order = np.lexsort((np.arange(len(teacher)), -teacher))[:200]
        positive, negative = np.array([(order[p], order[n]) for p, n in PAIRS]).T
        sem = sd_csr[positive] - sd_csr[negative]
        contrib = sem.multiply(cq[row]).tocsr()
        lex, _, _ = old.sparse_scores(ld, lcq[row])
        margin = np.asarray(contrib.sum(1)).ravel() + lex[positive] - lex[negative]
        contributions.append(contrib)
        margins.extend(margin.tolist())
        teacher_pairs.append(dict(query_id=records[index]['query_id'],
                                  positive=positive.tolist(), negative=negative.tolist()))
        if row % 64 == 0:
            print('NG82_MASK_TRAIN', row + 1, flush=True)
    contributions = sparse.vstack(contributions, format='csr')
    contributions.eliminate_zeros()
    margins = np.array(margins)
    tau = max(float(np.median(np.abs(margins))), 1e-8)
    benefits = utility(contributions, margins, tau)
    initial_d = sparse.load_npz(files['initial_d']).tocsc()
    initial_q = sparse.load_npz(files['initial_q'])
    initial_work = float(np.diff(initial_d.indptr)[initial_q.indices].sum() / initial_q.shape[0])
    del initial_d, initial_q, documents, sd_csr
    gc.collect()
    masks = {'S': np.ones(sd.shape[1], dtype=bool),
             'drop-df90': df < .9 * sd.shape[0],
             'drop-df50': df < .5 * sd.shape[0]}
    for policy in ('utility', 'df'):
        priority = benefits / np.maximum(cost, 1e-30) if policy == 'utility' else -df
        order = np.lexsort((np.arange(len(df)), priority))
        for label, budget in (('half', cost.sum() * .5), ('initial', initial_work)):
            masks[policy + '-' + label] = mask_to_budget(cost, order, budget)
    baseline_loss = float(np.logaddexp(0, -margins / tau).mean())
    design = {}
    for name, keep in masks.items():
        deleted = np.asarray(contributions[:, ~keep].sum(1)).ravel()
        exact_delta = float(np.logaddexp(0, -(margins - deleted) / tau).mean() - baseline_loss)
        additive = float(benefits[~keep].sum())
        design[name] = dict(features_removed=int((~keep).sum()),
            document_nnz_retained=int(df[keep].sum()), train_semantic_work=float(cost[keep].sum()),
            joint_train_loss_delta=exact_delta, additive_prediction=additive,
            nonadditive_residual=exact_delta - additive)
    campaign.io.write(output / 'mask-design.json', dict(tau=tau, pair_count=len(margins),
        initial_train_semantic_work=initial_work, parent_train_loss=baseline_loss, profiles=design))
    campaign.io.write(output / 'teacher-pairs.json', teacher_pairs)
    np.savez(output / 'masks.npz', **masks)
    np.savez(output / 'utility.npz', utility=benefits, df=df, cost=cost)
    del contributions
    q = sparse.load_npz(files['sq'])
    lq = sparse.load_npz(files['lq'])
    previous = campaign.rows(files['s_baseline'])
    results, rank_started = [], time.monotonic()
    with (output / 'rankings.jsonl').open('x') as stream:
        for row, index in enumerate(selected['diagnostic']):
            lex, _, lw = old.sparse_scores(ld, lq[row])
            lex_touch = np.asarray(ld[:, lq[row].indices].getnnz(axis=1)).ravel() > 0
            result = dict(query_id=records[index]['query_id'], domain=records[index]['subset'],
                          rankings={}, costs={})
            for name, keep in masks.items():
                query = masked_query(q[row], keep)
                sem, _, work = old.sparse_scores(sd, query)
                ranked = compact_rank(sem + lex, labels[index])
                result['rankings'][name] = ranked
                touched = np.asarray(sd[:, query.indices].getnnz(axis=1)).ravel() > 0
                result['costs'][name] = dict(semantic_df_work=work, lexical_df_work=lw,
                    semantic_touched=int(touched.sum()), union_touched=int((touched | lex_touch).sum()),
                    query_nnz=query.nnz, unseen_train_work=int(df[
                        query.indices[rate[query.indices] == 0]].sum()))
                if name == 'S':
                    require(result['query_id'] == previous[row]['query_id']
                            and ranked['top100'] == previous[row]['rankings']['hybrid']['top100'],
                            'S historical replay')
            results.append(result)
            stream.write(json.dumps(result) + '\n')
            if row == 63:
                campaign.canary(output, 'mask-ranking', time.monotonic() - rank_started,
                                64, len(selected['diagnostic']))
            if row % 128 == 0:
                print('NG82_MASK_RANK', row + 1, flush=True)
    costs = {p: {k: float(np.mean([r['costs'][p][k] for r in results]))
                 for k in results[0]['costs'][p]} for p in masks}
    return dict(quality=summarize(results, 'rankings'), design=design, costs=costs,
                versus_S={p: paired(results, p, 'S') for p in masks if p != 'S'},
                elapsed_seconds=time.monotonic() - started,
                queries=len(results), independent_holdout=False, native_cost_evaluated=False)


def worker(args):
    from clearml import Task
    files = verify(args.research_root, args.run)
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0082/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=old.TRACKING_FRAMEWORKS.copy(),
        auto_connect_arg_parser=False, auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    campaign.io.write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(input_sha256=campaign.io.sha(args.run / 'inputs.json'), phase=args.phase))
        result = (mask if args.phase == 'mask' else geometry)(files, output)
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
    name = args.phase
    require(name in PHASES and not (args.run / (name + '-controller-start.json')).exists(),
            'fresh lane required')
    require(campaign.io.sha(Path(__file__))
            == campaign.read(args.run / 'inputs.json')['source']['ng82_diagnostics.py'],
            'controller must use frozen source')
    args.worker_script = 'ng82_diagnostics.py'
    stopped = []
    handlers = {sig: signal.signal(sig, lambda *_: stopped.append(True))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    result = dict(status='failed_preserve_attempt', completed=[])
    try:
        with (args.research_root / '.ng80-cpu.lock').open('a') as shared:
            fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with (args.research_root / ('.ng82-' + name + '.lock')).open('a') as exclusive:
                fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
                campaign.io.write(args.run / (name + '-controller-start.json'),
                    dict(pid=os.getpid(), started_unix=time.time(), physical_gpu=None))
                campaign.guard.phase(args, name, stopped, cuda=False, limit_seconds=5400)
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
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    elif args.mode == 'verify':
        verify(args.research_root, args.run)
    elif args.mode == 'worker':
        require(args.phase in PHASES, 'phase required')
        worker(args)
    else:
        supervise(args)
