"""NG81 fixed-endpoint fusion control; no optimizer or locked-test access."""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse

import ng80_sparse_endpoints as old


campaign, audit, require = old.campaign, old.audit, old.require
PARENT = 'NG-0080/sparse-endpoints-v3'
PARENT_SHA = 'aa98a6bf917c2e5e0ea901d0fd20223f2a33ecc974a3120aba6f0a6b6f123642'
REVIEW_SHA = '365aad14fe4738786d7d38f699333a026e7932a51e51db8910ed50f0133814f7'
MODELS = old.MODELS
NORMS = ('serving', 'zscore')
PROFILES = ('historical', 'serving_calibrated', 'zscore_calibrated')
PHASES = ('encode-calibration', *('calibrate-' + m for m in MODELS),
          *('rank-' + m for m in MODELS), 'review')


def mix(semantic, lexical, alpha, normalization):
    semantic = np.asarray(semantic, dtype=np.float64)
    lexical = np.asarray(lexical, dtype=np.float64)
    require(semantic.ndim == 1 and semantic.shape == lexical.shape
            and len(semantic) > 1 and np.isfinite(semantic).all()
            and np.isfinite(lexical).all() and np.isfinite(alpha)
            and 0 <= alpha <= 1 and normalization in NORMS, 'fusion inputs')
    if normalization == 'serving':
        # Cached query codes already include NG80's 0.9/0.1 weights.
        return semantic * ((1 - alpha) / .9) + lexical * (alpha / .1)
    return (1 - alpha) * old.normalize(semantic) + alpha * old.normalize(lexical)


def metrics(ranks):
    ranks = np.asarray(ranks)
    require(ranks.ndim == 1 and len(ranks) > 0
            and np.issubdtype(ranks.dtype, np.integer)
            and (ranks > 0).all() and len(set(ranks.tolist())) == len(ranks),
            'positive rank list')
    ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(ranks))))
    return [float(sum(1 / np.log2(r + 1) for r in ranks if r <= 10) / ideal),
            float(np.mean(ranks <= 100))]


def ranked(score, labels):
    result = old.dense.rank_row(score, score, labels)
    positions = np.arange(len(score))
    direct = [1 + int(np.count_nonzero((score > score[g])
              | ((score == score[g]) & (positions < g)))) for g in labels]
    require(direct == result['gold_ranks'], 'independent all-positive ranks')
    np.testing.assert_allclose(metrics(direct),
        [result['ndcg10'], result['recall100']], rtol=0, atol=1e-12)
    return result


def macro(values, domains):
    values, domains = np.asarray(values), np.asarray(domains)
    require(values.shape[0] == len(domains) and set(domains) == set(old.DOMAINS)
            and np.isfinite(values).all(), 'macro population')
    return np.mean([values[domains == d].mean(0) for d in old.DOMAINS], axis=0)


def select(records, domains):
    require(len(records) == len(domains), 'calibration count')
    selected = {}
    for normalization in NORMS:
        arrays = []
        for record in records:
            require(set(record['candidates']) == set(NORMS), 'calibration modes')
            rows = record['candidates'][normalization]
            require([r['alpha'] for r in rows] == list(old.ALPHAS), 'alpha grid')
            values = []
            for row in rows:
                actual = metrics(row['gold_ranks'])
                np.testing.assert_allclose(actual, row['metrics'], rtol=0, atol=1e-12)
                values.append(actual)
            arrays.append(values)
        means = macro(arrays, domains)
        rows = [dict(alpha=a, ndcg10=float(v[0]), recall100=float(v[1]))
                for a, v in zip(old.ALPHAS, means, strict=True)]
        selected[normalization] = dict(alpha=old.choose_alpha(rows), candidates=rows)
    return selected


def verify(base, run):
    manifest = campaign.read(run / 'inputs.json')
    require(manifest['protocol'] == 'NG81_frozen_endpoint_fusion_v1'
            and manifest['phases'] == list(PHASES)
            and manifest['alphas'] == list(old.ALPHAS)
            and manifest['normalizations'] == list(NORMS)
            and manifest['profiles'] == list(PROFILES)
            and manifest['optimizer_updates'] == 0
            and manifest['locked_test_access'] is False
            and manifest['independent_holdout'] is False, 'NG81 protocol')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in manifest[key].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve())
                    and campaign.io.sha(path) == digest, 'changed input: ' + name)
    require(campaign.io.sha(base / PARENT / 'inputs.json') == PARENT_SHA
            and campaign.io.sha(base / PARENT / 'review/results.json') == REVIEW_SHA,
            'closed NG80 identities')
    require(campaign.read(run / 'selection.json')
            == campaign.read(base / PARENT / 'selection.json'), 'selection changed')
    require(campaign.read(run / 'config.json')['base']['semantic_weight'] == .9,
            'cached semantic weight changed')
    return manifest


def freeze(base, run):
    parent = base / PARENT
    require(run.parent == base / 'NG-0081' and not run.exists(), 'fresh NG81 run')
    require(campaign.io.sha(parent / 'inputs.json') == PARENT_SHA
            and campaign.io.sha(parent / 'review/results.json') == REVIEW_SHA,
            'parent identity')
    previous = old.verify(base, parent)
    for phase in old.PHASES:
        audit.sealed(old.phase_root(base, parent, phase))
    deps = previous['dependencies'].copy()
    deps.update({str(p.relative_to(base)): campaign.io.sha(p)
                 for p in parent.rglob('*') if p.is_file()})
    run.mkdir(parents=True, exist_ok=False)
    for name in previous['source']:
        if name.endswith('.py') or name == 'ng0071-ranking-config.json':
            shutil.copy2(parent / name, run / name)
    for name in ('config.json', 'selection.json'):
        shutil.copy2(parent / name, run / name)
    source = Path(__file__).resolve().parent
    for name in ('ng81_fusion.py', 'test_ng81_fusion.py', 'ng0081-fusion-execution.zh.md'):
        candidates = (source / name, source.parent / 'tests' / name,
                      source.parent / 'docs/research-sae/reports/ng0001-ng0099' / name)
        origin = next((p for p in candidates if p.is_file()), None)
        require(origin is not None, 'missing freeze source: ' + name)
        shutil.copy2(origin, run / name)
    campaign.io.write(run / 'inputs.json', dict(
        protocol='NG81_frozen_endpoint_fusion_v1', phases=list(PHASES),
        alphas=list(old.ALPHAS), normalizations=list(NORMS), profiles=list(PROFILES),
        dependencies=deps, source={p.name: campaign.io.sha(p)
                                  for p in run.iterdir() if p.is_file()},
        optimizer_updates=0, locked_test_access=False, independent_holdout=False,
        native_cost_evaluated=False, automatic_promotion=False))
    verify(base, run)
    print('NG81_FROZEN', campaign.io.sha(run / 'inputs.json'), flush=True)


def inputs(base, run, role):
    require(role in ('calibration', 'diagnostic'), 'query role')
    root = base / old.dense_fit.PARENT
    queries = campaign.read(root / 'prepare/queries.json')
    labels = campaign.read(root / 'prepare/labels.json')
    ids = campaign.read(run / 'selection.json')[role]
    count = 384 if role == 'calibration' else 1536
    require(len(ids) == len(set(ids)) == count
            and Counter(queries[i]['subset'] for i in ids)
            == dict.fromkeys(old.DOMAINS, count // 3), 'query population')
    return queries, labels, ids


def encode_calibration(base, run, output):
    import torch
    from ng71_training import Encoder, parameter_hash

    queries, _, ids = inputs(base, run, 'calibration')
    result = {}
    for arm in MODELS:
        root = base / f'NG-0080/sparse-{arm}-v1/train-3072' if arm != 'initial' else None
        expected = (campaign.read(root / 'results.json')['model_sha256']
                    if root else old.fit.preflight.INITIAL)
        encoder = Encoder(base, campaign.read(run / 'config.json'), 'cuda',
                          checkpoint=root / 'checkpoint' if root else None)
        require(parameter_hash(encoder.model) == expected, 'model identity')
        parts, started = [], time.monotonic()
        for offset in range(0, len(ids), 8):
            texts = [queries[i]['query'] for i in ids[offset:offset + 8]]
            with torch.no_grad():
                values = encoder.encode(texts, 'query').cpu().numpy()
            parts.append(sparse.csr_matrix(values))
            if offset == 56:
                campaign.canary(output, arm, time.monotonic() - started, 64, len(ids))
        codes = sparse.vstack(parts, format='csr')
        codes.eliminate_zeros()
        codes.sort_indices()
        require(codes.shape == (384, 50265) and codes.dtype == np.float32
                and codes.has_canonical_format and np.isfinite(codes.data).all()
                and (codes.data > 0).all() and (np.diff(codes.indptr) > 0).all(), 'query CSR')
        require(parameter_hash(encoder.model) == expected, 'encoding mutated model')
        sparse.save_npz(output / f'{arm}.npz', codes)
        result[arm] = dict(model_sha256=expected, nnz=int(codes.nnz),
                           elapsed_seconds=time.monotonic() - started)
        print('NG81_ENCODE_CALIBRATION', arm, result[arm], flush=True)
        del encoder, codes, parts
        gc.collect()
        torch.cuda.empty_cache()
    campaign.io.write(output / 'query-ids.json', [queries[i]['query_id'] for i in ids])
    return dict(models=result, queries=384, optimizer_updates=0, documents_reencoded=0)


def score_phase(base, run, output, arm, role):
    queries, labels, ids = inputs(base, run, role)
    parent = base / PARENT
    encoded = old.phase_root(base, parent, 'encode-' + arm)
    sd = sparse.load_npz(encoded / 'document.npz').astype(np.float64).tocsc()
    ld = sparse.load_npz(base / old.fit.preflight.LD).astype(np.float64).tocsc()
    lq = sparse.load_npz(parent / f'prepare-lexical/{role}.npz')
    if role == 'calibration':
        audit.sealed(run / 'encode-calibration')
        sq = sparse.load_npz(run / f'encode-calibration/{arm}.npz')
        identity = run / 'encode-calibration/query-ids.json'
        alpha = None
    else:
        alpha = audit.sealed(run / ('calibrate-' + arm))['selected']
        sq = sparse.load_npz(encoded / 'query.npz')
        identity = encoded / 'query-ids.json'
    require(campaign.read(identity) == [queries[i]['query_id'] for i in ids]
            and sq.shape == (len(ids), 50265) and sd.shape == (233009, 50265)
            and lq.shape == (len(ids), ld.shape[1]) and ld.shape[0] == 233009,
            'code/query/corpus alignment')
    records, started = [], time.monotonic()
    historical = (campaign.rows(parent / f'rank-{arm}/rankings.jsonl')
                  if role == 'diagnostic' else None)
    with (output / 'rankings.jsonl').open('x') as stream:
        for row, index in enumerate(ids):
            sem, _, sw = old.sparse_scores(sd, sq[row])
            lex, _, lw = old.sparse_scores(ld, lq[row])
            record = dict(query_id=queries[index]['query_id'], query_index=index,
                          domain=queries[index]['subset'])
            if role == 'calibration':
                candidates = {}
                for norm in NORMS:
                    candidates[norm] = []
                    for a in old.ALPHAS:
                        entry = ranked(mix(sem, lex, a, norm), labels[index])
                        candidates[norm].append(dict(alpha=a, gold_ranks=entry['gold_ranks'],
                            metrics=[entry['ndcg10'], entry['recall100']]))
                record['candidates'] = candidates
            else:
                profiles = {'historical': (.1, 'serving')}
                profiles.update({n + '_calibrated': (alpha[n]['alpha'], n) for n in NORMS})
                record['rankings'] = {name: ranked(mix(sem, lex, a, n), labels[index])
                                      for name, (a, n) in profiles.items()}
                prior = historical[row]
                require(prior['query_index'] == index
                        and prior['query_id'] == record['query_id'], 'prior query identity')
                for key in ('gold_ranks', 'gold_ids', 'top100', 'top100_scores'):
                    require(record['rankings']['historical'][key]
                            == prior['rankings']['hybrid'][key], 'historical ranking replay')
                record['normalization'] = dict(semantic=old.normalization_stats(sem),
                                                lexical=old.normalization_stats(lex))
                record['work'] = dict(semantic_df_sum=sw, lexical_df_sum=lw,
                    semantic_touch=int(np.count_nonzero(sem)),
                    lexical_touch=int(np.count_nonzero(lex)),
                    union_touch=int(np.count_nonzero((sem != 0) | (lex != 0))))
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            if role == 'calibration':
                records.append(record)
            if row == 15:
                campaign.canary(output, role, time.monotonic() - started, 16, len(ids))
            if (row + 1) % 128 == 0:
                stream.flush()
                print('NG81_SCORE', arm, role, row + 1, len(ids), flush=True)
    df = np.diff(sd.indptr)
    result = dict(model=arm, queries=len(ids), documents=233009, optimizer_updates=0,
                  semantic_document_nnz=int(sd.nnz), query_nnz=int(sq.nnz),
                  max_df=int(df.max()), universal_features=int(np.sum(df == len(sem))),
                  df_ge_90pct=int(np.sum(df >= .9 * len(sem))), native_cost_evaluated=False)
    if role == 'calibration':
        result['selected'] = select(records, [queries[i]['subset'] for i in ids])
    return result


def review(base, run):
    queries, labels, ids = inputs(base, run, 'diagnostic')
    _, _, calids = inputs(base, run, 'calibration')
    domains = [queries[i]['subset'] for i in ids]
    parent = base / PARENT
    baseline = campaign.rows(parent / 'rank-baselines/rankings.jsonl')
    arrays, quality, proxies = {}, {}, {}
    for key in ('bm25', 'dense', 'dense_hybrid'):
        arrays[key] = np.array([metrics(r['rankings'][key]['gold_ranks']) for r in baseline])
    ld = sparse.load_npz(base / old.fit.preflight.LD).astype(np.float64)
    lq = sparse.load_npz(parent / 'prepare-lexical/diagnostic.npz')
    peak = 0.
    for arm in MODELS:
        calibrated = audit.sealed(run / ('calibrate-' + arm))
        calrows = campaign.rows(run / f'calibrate-{arm}/rankings.jsonl')
        require([r['query_index'] for r in calrows] == calids
                and [r['query_id'] for r in calrows] == [queries[i]['query_id'] for i in calids],
                'calibration identities')
        chosen = select(calrows, [queries[i]['subset'] for i in calids])
        require(chosen == calibrated['selected'], 'TRAIN selection replay')
        proxies[arm] = audit.sealed(run / ('rank-' + arm))
        rows = campaign.rows(run / f'rank-{arm}/rankings.jsonl')
        require(len(rows) == len(ids), 'diagnostic cardinality')
        encoded = old.phase_root(base, parent, 'encode-' + arm)
        sd = sparse.load_npz(encoded / 'document.npz').astype(np.float64)
        sq = sparse.load_npz(encoded / 'query.npz').astype(np.float64)
        values = {p: [] for p in PROFILES}
        for row, (record, index) in enumerate(zip(rows, ids, strict=True)):
            require(record['query_index'] == index and record['query_id'] == queries[index]['query_id']
                    and record['domain'] == domains[row]
                    and set(record['rankings']) == set(PROFILES), 'diagnostic identity')
            for profile in PROFILES:
                entry = record['rankings'][profile]
                head, ranks = entry['top100'], entry['gold_ranks']
                require(entry['gold_ids'] == labels[index]
                        and len(head) == len(set(head)) == 100
                        and all(type(h) is int and 0 <= h < 233009 for h in head)
                        and all(1 <= r <= 233009 for r in ranks), 'head/gold inventory')
                for doc, rank in zip(labels[index], ranks, strict=True):
                    require((rank <= 100 and head[rank - 1] == doc)
                            or (rank > 100 and doc not in head), 'head/gold agreement')
                value = metrics(ranks)
                np.testing.assert_allclose(value, [entry['ndcg10'], entry['recall100']],
                                           rtol=0, atol=1e-12)
                sem = np.asarray(sd[head].multiply(sq[row]).sum(1)).ravel()
                lex = np.asarray(ld[head].multiply(lq[row]).sum(1)).ravel()
                if profile == 'zscore_calibrated':
                    a = chosen['zscore']['alpha']
                    stats = record['normalization']
                    score = ((1 - a) * old.normalize_head(sem, stats['semantic'])
                             + a * old.normalize_head(lex, stats['lexical']))
                else:
                    a = .1 if profile == 'historical' else chosen['serving']['alpha']
                    score = (1 - a) / .9 * sem + a / .1 * lex
                error = float(np.max(np.abs(score - entry['top100_scores'])))
                require(error <= 1e-10, 'independent head coordinate score')
                peak = max(peak, error)
                values[profile].append(value)
        proxies[arm]['observed_work'] = {
            k: float(np.mean([r['work'][k] for r in rows])) for k in rows[0]['work']}
        for profile in PROFILES:
            arrays[arm + '_' + profile] = np.array(values[profile])
        del sd, sq
        gc.collect()
    for key, values in arrays.items():
        quality[key] = {d: values[np.array(domains) == d].mean(0).tolist() for d in old.DOMAINS}
        quality[key]['macro'] = macro(values, domains).tolist()
    comparisons = {}
    for arm in MODELS:
        for norm in NORMS:
            key = arm + '_' + norm + '_calibrated'
            for reference in (arm + '_historical', 'dense', 'dense_hybrid'):
                comparisons[key + '-minus-' + reference] = old.dense.paired(
                    arrays[key] - arrays[reference], domains)
    return dict(passed=True, quality=quality, comparisons=comparisons, proxies=proxies,
        max_independent_head_error=peak, metrics=['ndcg10', 'recall100'],
        alpha_selected_on_train_only=True, independent_holdout=False,
        native_cost_evaluated=False, locked_test_access=False, automatic_promotion=False,
        stopping_rule='no_finer_alpha_or_rrf_sweep; review_before_training')


def worker(args):
    import torch
    from clearml import Task

    base, run, phase = args.research_root, args.run, args.phase
    verify(base, run)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(81081)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0081/fusion/' + phase,
        reuse_last_task_id=False, auto_connect_frameworks=old.TRACKING_FRAMEWORKS.copy(),
        auto_connect_arg_parser=False, auto_connect_streams=False, auto_resource_monitoring=False)
    output = run / phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    campaign.io.write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(input_sha256=campaign.io.sha(run / 'inputs.json'), phase=phase))
        if phase == 'encode-calibration':
            result = encode_calibration(base, run, output)
        elif phase == 'review':
            for previous in PHASES[:-1]:
                audit.sealed(run / previous)
            result = review(base, run)
        else:
            kind, arm = phase.split('-', 1)
            result = score_phase(base, run, output, arm,
                                 'calibration' if kind == 'calibrate' else 'diagnostic')
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
    require(args.gpu in (0, 3) and not (args.run / 'controller-start.json').exists(),
            'fresh controller and permitted physical GPU required')
    require(campaign.io.sha(Path(__file__))
            == campaign.read(args.run / 'inputs.json')['source']['ng81_fusion.py'], 'frozen controller')
    stopped, done = [], []
    handlers = {sig: signal.signal(sig, lambda *_: stopped.append(True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    args.worker_script = 'ng81_fusion.py'
    campaign.io.write(args.run / 'controller-start.json',
        dict(pid=os.getpid(), started_unix=time.time(), physical_gpu=args.gpu))
    try:
        for phase in PHASES:
            cuda = phase == 'encode-calibration'
            name = f'.ng-gpu-{args.gpu}.lock' if cuda else '.ng80-cpu.lock'
            with (args.research_root / name).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if cuda:
                    memory, use = campaign.guard.gpu_state(args.gpu)
                    require(memory < 100 and use == 0, 'GPU occupied; do not evict')
                print('NG81_PHASE_START', phase, flush=True)
                campaign.guard.phase(args, phase, stopped, cuda=cuda, limit_seconds=5400)
                done.append(phase)
                print('NG81_PHASE_COMPLETE', phase, flush=True)
        result = dict(status='all_phases_complete', completed=done)
    except BaseException as exc:
        result = dict(status='failed_preserve_attempt', completed=done, error=str(exc))
        raise
    finally:
        result['finished_unix'] = time.time()
        campaign.io.write(args.run / 'controller-exit.json', result)
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
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    elif args.mode == 'supervise':
        supervise(args)
    elif args.mode == 'worker':
        require(args.phase in PHASES, 'phase required')
        worker(args)
    else:
        verify(args.research_root, args.run)
