"""Frozen full-corpus S/P endpoints; no training or diagnostic selection."""

import argparse
from collections import Counter
import gc
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse

import ng80_campaign as campaign
import ng80_endpoints as dense
import ng80_fit as dense_fit
import ng80_sparse_fit as fit
from ng80_data import DOMAINS, require
import review_ng80_sparse_fit as audit


MODELS = ('initial', 'S', 'P')
ALPHAS = (0., .1, .25, .5, .75, .9, 1.)
LANES = {
    'audit': ('audit-training',),
    'prepare': ('prepare-lexical', 'calibrate-dense'),
    'first': ('encode-initial', 'encode-S'),
    'second': ('encode-P',),
    'ranking': ('rank-baselines', 'rank-initial', 'rank-S', 'rank-P', 'review'),
}
PHASES = tuple(p for values in LANES.values() for p in values)
REUSE_PARENT = 'NG-0080/sparse-endpoints-v2'
REUSE_INPUT = '44c8feeae1c437652cb4baff685bcf635f32244834bd8cf8c078ad02238b52f2'
REUSE_PHASES = ('audit-training', 'encode-initial', 'encode-S', 'encode-P')
TRACKING_FRAMEWORKS = dict.fromkeys((
    'detect_repository', 'hydra', 'scikit', 'joblib', 'matplotlib',
    'tensorflow', 'tensorboard', 'tfdefines', 'pytorch', 'megengine',
    'xgboost', 'catboost', 'fastai', 'lightgbm', 'gradio'), False)


def phase_root(base, run, phase):
    require(phase in PHASES, 'unknown endpoint phase')
    reused = campaign.read(run / 'inputs.json').get('reused_phases', {})
    if phase not in reused:
        return run / phase
    require(phase in REUSE_PHASES
            and reused[phase] == REUSE_PARENT + '/' + phase,
            'invalid reused phase path')
    return base / reused[phase]


def reuse_dependencies(base):
    parent = base / REUSE_PARENT
    require(campaign.io.sha(parent / 'inputs.json') == REUSE_INPUT,
            'wrong reuse parent')
    manifest = verify(base, parent)
    for lane in ('audit', 'first', 'second'):
        state = campaign.read(parent / f'{lane}-controller-exit.json')
        require(state['status'] == 'all_phases_complete'
                and state['completed'] == list(LANES[lane]),
                'reused controller unfinished')
    for phase in REUSE_PHASES:
        audit.sealed(parent / phase)
    failed = campaign.read(parent / 'prepare-lexical/complete.json')
    exit_record = campaign.read(parent / 'prepare-lexical/exit.json')
    controller = campaign.read(parent / 'prepare-controller-exit.json')
    require(failed['passed'] is False and exit_record['exit_code'] == -6
            and exit_record['error'] is None
            and exit_record['owned_group_closed'] is True
            and controller['status'] == 'failed_preserve_attempt'
            and controller['completed'] == []
            and not (parent / 'calibrate-dense').exists()
            and not (parent / 'ranking-controller-start.json').exists(),
            'reuse requires preserved preparation abort, no ranking')
    folder = parent / 'prepare-lexical'
    require({str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
            == set(failed['files']) | {'complete.json'}, 'failed inventory changed')
    for name, digest in failed['files'].items():
        path = folder / name
        require(path.resolve().is_relative_to(folder.resolve())
                and campaign.io.sha(path) == digest, 'failed evidence changed')
    files = {str(p.relative_to(base)): campaign.io.sha(p)
             for p in parent.rglob('*') if p.is_file()}
    files.update(manifest['dependencies'])
    return files


def selection(queries, selected):
    diagnostic = selected['diagnostic']
    calibration = dense_fit.order(queries, selected)[::32]
    require(len(diagnostic) == 1536 and len(calibration) == 384
            and len(set(diagnostic + calibration)) == 1920
            and not set(diagnostic) & set(selected['train'])
            and set(calibration) <= set(selected['train']), 'endpoint split')
    for ids, count in ((diagnostic, 512), (calibration, 128)):
        require(Counter(queries[i]['subset'] for i in ids)
                == dict.fromkeys(DOMAINS, count), 'endpoint domain balance')
    return dict(diagnostic=diagnostic, calibration=calibration)


def verify(base, run):
    m = campaign.read(run / 'inputs.json')
    require(m['protocol'] in ('NG80_SP_full_corpus_terminal3072_v2',
                             'NG80_SP_full_corpus_terminal3072_v3')
            and m['audit_target_arithmetic'] == 'fp32_cuda'
            and m['phases'] == list(PHASES) and m['training_inputs'] == audit.INPUTS
            and m['models'] == list(MODELS) and m['alphas'] == list(ALPHAS)
            and m['optimizer_updates'] == 0 and not m['locked_test_access']
            and not m['independent_holdout'], 'endpoint contract')
    if m['protocol'].endswith('_v3'):
        require(m['tracking_frameworks'] == TRACKING_FRAMEWORKS,
                'tracking policy changed')
        reused = m['reused_phases']
        require(reused == {} or reused == {
            phase: REUSE_PARENT + '/' + phase for phase in REUSE_PHASES},
            'reused phase inventory')
        if reused:
            require(m['dependencies'][REUSE_PARENT + '/inputs.json']
                    == REUSE_INPUT, 'reuse identity')
            for phase in REUSE_PHASES:
                require(not (run / phase).exists(), 'reused output shadowed')
                audit.sealed(phase_root(base, run, phase))
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in m[key].items():
            p = root / name
            require(p.resolve().is_relative_to(root.resolve())
                    and campaign.io.sha(p) == digest, 'changed input: ' + name)
    parent = base / dense_fit.PARENT
    require(campaign.read(run / 'selection.json') == selection(
        campaign.read(parent / 'prepare/queries.json'),
        campaign.read(parent / 'prepare/selection.json')), 'selection changed')
    return m


def freeze(base, run, reuse=False):
    require(run.parent == base / 'NG-0080' and not run.exists(), 'fresh endpoint required')
    deps, sources = reuse_dependencies(base) if reuse else {}, {}
    for arm, digest in audit.INPUTS.items():
        root = base / f'NG-0080/sparse-{arm}-v1'
        require(campaign.io.sha(root / 'inputs.json') == digest, 'training input')
        m = fit.verify(base, root)
        control = campaign.read(root / 'controller-exit.json')
        require(control['status'] == 'all_phases_complete'
                and control['completed'] == list(fit.PHASES), 'training unfinished')
        for phase in fit.PHASES:
            audit.sealed(root / phase)
        deps.update(m['dependencies'])
        deps.update({str(p.relative_to(base)): campaign.io.sha(p)
                     for p in root.rglob('*') if p.is_file()})
        for name in m['source']:
            if name.endswith('.py') or name == 'ng0071-ranking-config.json':
                sources.setdefault(name, root / name)
    source = Path(__file__).resolve().parent
    for name in ('ng80_sparse_endpoints.py', 'review_ng80_sparse_fit.py',
                 'review_ng80_fit.py', 'ng80_endpoints.py'):
        sources[name] = source / name
    for name in ('test_ng80_sparse_endpoints.py', 'test_ng80_sparse_review.py'):
        sources[name] = source.parent / 'tests' / name
    for name in ('ng0080-sparse-endpoints.zh.md', 'ng0080-sparse-terminal-review.zh.md'):
        sources[name] = source.parent / 'docs/research-sae/reports/ng0001-ng0099' / name
    parent = base / dense_fit.PARENT
    chosen = selection(campaign.read(parent / 'prepare/queries.json'),
                       campaign.read(parent / 'prepare/selection.json'))
    run.mkdir()
    for name, path in sources.items():
        shutil.copy2(path, run / name)
    shutil.copy2(base / 'NG-0080/sparse-S-v1/config.json', run / 'config.json')
    campaign.io.write(run / 'selection.json', chosen)
    campaign.io.write(run / 'inputs.json', dict(
        protocol='NG80_SP_full_corpus_terminal3072_v3', phases=list(PHASES),
        tracking_frameworks=TRACKING_FRAMEWORKS,
        reused_phases={phase: REUSE_PARENT + '/' + phase
                       for phase in REUSE_PHASES} if reuse else {},
        audit_target_arithmetic='fp32_cuda',
        training_inputs=audit.INPUTS, models=list(MODELS), alphas=list(ALPHAS),
        dependencies=deps, source={p.name: campaign.io.sha(p)
                                  for p in run.iterdir() if p.is_file()},
        optimizer_updates=0, locked_test_access=False, independent_holdout=False,
        native_cost_evaluated=False, automatic_promotion=False))
    verify(base, run)
    print('NG80_SP_ENDPOINT_FROZEN', campaign.io.sha(run / 'inputs.json'), flush=True)


def prepare_lexical(base, run, output):
    spec = importlib.util.spec_from_file_location('ng8_lexical', base / 'NG-0008/ng8_lexical.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vocab = campaign.read(base / 'NG-0009/surfaces/vocabulary.json')
    rms = np.load(base / 'NG-0009/surfaces/train-rms.npy')[:len(vocab)]
    queries = campaign.read(base / dense_fit.PARENT / 'prepare/queries.json')
    chosen = campaign.read(run / 'selection.json')
    tokenizer = fit.preflight.native.Tokenizer(base / fit.preflight.LEXICAL_RUNTIME / 'libtext.so')
    counts = {}
    reference = sparse.load_npz(base / fit.PREFLIGHT / 'calibrate/lexical-query.npz')
    positions = {i: n for n, i in enumerate(campaign.read(base / 'NG-0080/sparse-S-v1/order.json'))}
    for role, ids in chosen.items():
        raw, _, oov = module.count_matrix([queries[i]['query'] for i in ids], tokenizer, vocab)
        matrix = raw.multiply((.1 / module.scale(raw, rms))[:, None]).tocsr()
        if role == 'calibration':
            prior = reference[[positions[i] for i in ids]]
            require((matrix != prior).nnz == 0, 'same-platform lexical replay')
        sparse.save_npz(output / f'{role}.npz', matrix)
        campaign.io.write(output / f'{role}-ids.json', [queries[i]['query_id'] for i in ids])
        counts[role] = dict(rows=matrix.shape[0], nnz=int(matrix.nnz), oov=oov)
    return dict(counts=counts, locale=fit.preflight.native.ICU_LOCALE,
                calibration_replay_exact=True, serving_rms_changed=False)


def encode(base, run, output, arm):
    import torch
    from ng71_training import Encoder, parameter_hash

    phase = base / f'NG-0080/sparse-{arm}-v1/train-3072' if arm != 'initial' else None
    expected = campaign.read(phase / 'results.json')['model_sha256'] if phase else fit.preflight.INITIAL
    encoder = Encoder(base, campaign.read(run / 'config.json'), 'cuda',
                      checkpoint=phase / 'checkpoint' if phase else None)
    require(parameter_hash(encoder.model) == expected, 'model identity')
    queries = campaign.read(base / dense_fit.PARENT / 'prepare/queries.json')
    ids = campaign.read(run / 'selection.json')['diagnostic']
    documents = campaign.rows(base / campaign.DATA / 'documents.jsonl')
    require(len(documents) == 233009, 'corpus cardinality')
    counts = {}
    for role, texts, batch in (('document', [d['text'] for d in documents], 96),
                              ('query', [queries[i]['query'] for i in ids], 8)):
        parts, tick = [], time.monotonic()
        for start in range(0, len(texts), batch):
            with torch.no_grad():
                values = encoder.encode(texts[start:start + batch], role).cpu().numpy()
            parts.append(sparse.csr_matrix(values))
            if start == 7 * batch:
                campaign.canary(output, role, time.monotonic() - tick, 8 * batch, len(texts))
            if start % (32 * batch) == 0:
                print('NG80_SP_ENCODE', arm, role, start + len(values), len(texts), flush=True)
        matrix = sparse.vstack(parts, format='csr')
        matrix.eliminate_zeros()
        matrix.sort_indices()
        require(matrix.shape == (len(texts), 50265) and matrix.dtype == np.float32
                and matrix.has_canonical_format and np.isfinite(matrix.data).all()
                and (matrix.data > 0).all() and (np.diff(matrix.indptr) > 0).all(), 'CSR contract')
        sparse.save_npz(output / f'{role}.npz', matrix)
        counts[role] = dict(rows=len(texts), nnz=int(matrix.nnz), seconds=time.monotonic() - tick,
            csr_bytes=matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
            compressed_bytes=(output / f'{role}.npz').stat().st_size)
        del matrix, parts
        gc.collect()
    require(parameter_hash(encoder.model) == expected, 'encoding changed parameters')
    campaign.io.write(output / 'query-ids.json', [queries[i]['query_id'] for i in ids])
    return dict(model=arm, model_sha256=expected, counts=counts, fresh=True,
                optimizer_updates=0, locked_test_access=False, native_cost_evaluated=False)


def sparse_scores(documents, query):
    require(sparse.isspmatrix_csc(documents) and documents.has_canonical_format
            and query.shape == (1, documents.shape[1]) and query.has_canonical_format,
            'scoring sparse contract')
    query = query.astype(np.float64)
    score = np.asarray(documents[:, query.indices] @ query.data).ravel()
    alternate = np.zeros(documents.shape[0], dtype=np.float64)
    work = 0
    for term, weight in zip(query.indices, query.data, strict=True):
        lo, hi = documents.indptr[term:term + 2]
        alternate[documents.indices[lo:hi]] += documents.data[lo:hi] * weight
        work += int(hi - lo)
    require(np.isfinite(score).all() and np.isfinite(alternate).all()
            and np.max(np.abs(score - alternate), initial=0.) <= 1e-12,
            'independent posting accumulation')
    return score, alternate, work


def normalize(score):
    score = np.asarray(score, dtype=np.float64)
    require(score.ndim == 1 and len(score) > 1 and np.isfinite(score).all(), 'normalization input')
    centered = score - score.mean()
    deviation = float(np.sqrt(np.mean(centered ** 2)))
    return centered / deviation if deviation > 0 else np.zeros_like(centered)


def normalization_stats(score):
    return dict(mean=float(np.mean(score)), std=float(np.std(score)))


def normalize_head(score, stats):
    require(np.isfinite([stats['mean'], stats['std']]).all()
            and stats['std'] >= 0, 'stored normalization')
    return ((score - stats['mean']) / stats['std']
            if stats['std'] > 0 else np.zeros_like(score))


def choose_alpha(rows):
    require(len(rows) == len(ALPHAS), 'calibration grid')
    for row, alpha in zip(rows, ALPHAS, strict=True):
        require(row['alpha'] == alpha and np.isfinite([row['ndcg10'], row['recall100']]).all(),
                'calibration result')
    # Prefer less lexical weight if both predeclared TRAIN metrics tie.
    return max(rows, key=lambda r: (r['ndcg10'], r['recall100'], -r['alpha']))['alpha']


def teacher_scores(documents, query):
    score = documents @ query.astype(np.float64)
    alternate = np.einsum('ij,j->i', documents, query.astype(np.float64), optimize=False)
    require(np.max(np.abs(score - alternate)) <= 1e-12, 'teacher reduction parity')
    return score, alternate


def calibrate_dense(base, run, output):
    audit.sealed(run / 'prepare-lexical')
    ids = campaign.read(run / 'selection.json')['calibration']
    parent = base / dense_fit.PARENT
    labels = campaign.read(parent / 'prepare/labels.json')
    tq = np.load(parent / 'encode-teacher/query.npy', mmap_mode='r')
    td = np.load(base / campaign.DENSE).astype(np.float64)
    ld = sparse.load_npz(base / fit.preflight.LD).astype(np.float64).tocsc()
    lq = sparse.load_npz(run / 'prepare-lexical/calibration.npz')
    metrics, tick = [], time.monotonic()
    with (output / 'calibration.jsonl').open('x') as stream:
        for row, index in enumerate(ids):
            t, _ = teacher_scores(td, tq[index])
            lex, _, _ = sparse_scores(ld, lq[row])
            t, lex = normalize(t), normalize(lex)
            results = []
            for alpha in ALPHAS:
                mixed = (1 - alpha) * t + alpha * lex
                ranked = dense.rank_row(mixed, mixed, labels[index])
                results.append([ranked['ndcg10'], ranked['recall100']])
            stream.write(json.dumps(dict(query_index=index, metrics=results)) + '\n')
            metrics.append(results)
            if row == 15:
                campaign.canary(output, 'calibration', time.monotonic() - tick, 16, len(ids))
    averages = np.mean(metrics, axis=0)
    rows = [dict(alpha=a, ndcg10=float(v[0]), recall100=float(v[1]))
            for a, v in zip(ALPHAS, averages, strict=True)]
    return dict(alpha=choose_alpha(rows), candidates=rows, calibration_queries=len(ids),
                normalization='per_query_full_corpus_zscore', diagnostic_used=False,
                independent_holdout=False, exhaustive_fusion_optimum_claimed=False)


def rank(base, run, output, model):
    parent = base / dense_fit.PARENT
    queries = campaign.read(parent / 'prepare/queries.json')
    labels = campaign.read(parent / 'prepare/labels.json')
    panels = campaign.read(parent / 'prepare/panels.json')
    ids = campaign.read(run / 'selection.json')['diagnostic']
    ld = sparse.load_npz(base / fit.preflight.LD).astype(np.float64).tocsc()
    lq = sparse.load_npz(run / 'prepare-lexical/diagnostic.npz')
    tq = np.load(parent / 'encode-teacher/query.npy', mmap_mode='r')
    td = np.load(base / campaign.DENSE, mmap_mode='r')
    if model == 'baselines':
        td = td.astype(np.float64)
        alpha = audit.sealed(run / 'calibrate-dense')['alpha']
    else:
        encoded = phase_root(base, run, 'encode-' + model)
        audit.sealed(encoded)
        sd = sparse.load_npz(encoded / 'document.npz').astype(np.float64).tocsc()
        sq = sparse.load_npz(encoded / 'query.npz')
        require(sq.shape == (1536, 50265) and sd.shape == (233009, 50265), 'semantic dimensions')
        require(campaign.read(encoded / 'query-ids.json')
                == [queries[i]['query_id'] for i in ids], 'query code identity')
    tick = time.monotonic()
    scale = campaign.read(base / fit.PREFLIGHT / 'calibrate/scale.json')['scale']
    with (output / 'rankings.jsonl').open('x') as stream:
        for row, index in enumerate(ids):
            lex, lex_alt, lex_work = sparse_scores(ld, lq[row])
            normalization = None
            if model == 'baselines':
                score, alternate = teacher_scores(td, tq[index])
                mixed = (1 - alpha) * normalize(score) + alpha * normalize(lex)
                pairs = dict(bm25=(lex, lex_alt), dense=(score, alternate),
                             dense_hybrid=(mixed, mixed))
                normalization = dict(lexical=normalization_stats(lex),
                                     dense=normalization_stats(score))
                geometry, semantic_work = None, None
            else:
                score, alternate, semantic_work = sparse_scores(sd, sq[row])
                pairs = dict(semantic=(score, alternate), hybrid=(score + lex, alternate + lex_alt))
                target = td[panels[index]].astype(np.float64) @ tq[index].astype(np.float64)
                error = score[panels[index]] / scale - target
                geometry = float(np.mean((error - error.mean()) ** 2))
            ranked = {key: dense.rank_row(*values, labels[index]) for key, values in pairs.items()}
            stream.write(json.dumps(dict(query_id=queries[index]['query_id'], query_index=index,
                domain=queries[index]['subset'], rankings=ranked, centered_semantic_mse=geometry,
                lexical_df_work=lex_work, semantic_df_work=semantic_work,
                normalization=normalization), allow_nan=False) + '\n')
            if row == 15:
                campaign.canary(output, 'ranking', time.monotonic() - tick, 16, len(ids))
            if (row + 1) % 128 == 0:
                stream.flush()
                print('NG80_SP_RANK', model, row + 1, len(ids), flush=True)
    return dict(model=model, queries=len(ids), full_corpus_documents=233009,
                all_positives=True, native_cost_evaluated=False, independent_holdout=False)


def review(base, run):
    ids = campaign.read(run / 'selection.json')['diagnostic']
    parent = base / dense_fit.PARENT
    queries = campaign.read(parent / 'prepare/queries.json')
    labels = campaign.read(parent / 'prepare/labels.json')
    arrays, quality, work, teacher_heads = {}, {}, {}, []
    td = np.load(base / campaign.DENSE, mmap_mode='r')
    tq = np.load(parent / 'encode-teacher/query.npy', mmap_mode='r')
    ld = sparse.load_npz(base / fit.preflight.LD).astype(np.float64).tocsc()
    lq = sparse.load_npz(run / 'prepare-lexical/diagnostic.npz')
    alpha = audit.sealed(run / 'calibrate-dense')['alpha']
    peak_error = 0.
    for phase in PHASES[:-1]:
        audit.sealed(phase_root(base, run, phase))
    for model in ('baselines', *MODELS):
        records = campaign.rows(run / f'rank-{model}/rankings.jsonl')
        require(len(records) == len(ids), 'ranking count')
        names = ('bm25', 'dense', 'dense_hybrid') if model == 'baselines' else ('semantic', 'hybrid')
        values = {n: [] for n in names}
        overlaps = {n: [] for n in names}
        if model != 'baselines':
            encoded = phase_root(base, run, 'encode-' + model)
            sd = sparse.load_npz(encoded / 'document.npz').astype(np.float64).tocsc()
            sq = sparse.load_npz(encoded / 'query.npz')
        for row, (record, index) in enumerate(zip(records, ids, strict=True)):
            require(record['query_id'] == queries[index]['query_id']
                    and record['query_index'] == index and record['domain'] == queries[index]['subset']
                    and set(record['rankings']) == set(names), 'ranking identity')
            for name in names:
                entry = record['rankings'][name]
                ranks, head = entry['gold_ranks'], entry['top100']
                require(entry['gold_ids'] == labels[index] and len(ranks) == len(labels[index])
                        and len(set(ranks)) == len(ranks) and all(1 <= r <= 233009 for r in ranks)
                        and len(head) == len(set(head)) == 100
                        and all(type(h) is int and 0 <= h < 233009 for h in head), 'gold ranks/head')
                for doc, position in zip(labels[index], ranks, strict=True):
                    require((position <= 100 and head[position - 1] == doc)
                            or (position > 100 and doc not in head), 'rank/head mismatch')
                ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(ranks))))
                actual = [sum(1 / np.log2(r + 1) for r in ranks if r <= 10) / ideal,
                          sum(r <= 100 for r in ranks) / len(ranks)]
                np.testing.assert_allclose(actual, [entry['ndcg10'], entry['recall100']],
                                            rtol=0, atol=1e-12)
                lexical = np.asarray(ld[head].multiply(lq[row]).sum(1)).ravel()
                if model == 'baselines':
                    teacher = np.sum(td[head].astype(np.float64) * tq[index].astype(np.float64), axis=1)
                    if name == 'dense_hybrid':
                        score = ((1 - alpha) * normalize_head(teacher, record['normalization']['dense'])
                                 + alpha * normalize_head(lexical, record['normalization']['lexical']))
                    else:
                        score = teacher if name == 'dense' else lexical
                else:
                    score = np.asarray(sd[head].multiply(sq[row].astype(np.float64)).sum(1)).ravel()
                    if name == 'hybrid':
                        score += lexical
                error = float(np.max(np.abs(score - entry['top100_scores'])))
                require(np.isfinite(score).all() and error <= 1e-10, 'head coordinate mismatch')
                peak_error = max(peak_error, error)
                values[name].append(actual)
                if model != 'baselines':
                    overlaps[name].append(len(set(head[:10]) & set(teacher_heads[row])) / 10)
            if model == 'baselines':
                teacher_heads.append(record['rankings']['dense']['top100'][:10])
        domains = [queries[i]['subset'] for i in ids]
        for name, value in values.items():
            key = name if model == 'baselines' else model + '_' + name
            arrays[key] = np.asarray(value)
            quality[key] = {d: arrays[key][np.asarray(domains) == d].mean(0).tolist() for d in DOMAINS}
            quality[key]['macro'] = arrays[key].mean(0).tolist()
            if model != 'baselines':
                quality[key]['teacher_top10_overlap'] = float(np.mean(overlaps[name]))
        work[model] = dict(lexical_df_work_mean=float(np.mean([r['lexical_df_work'] for r in records])))
        if model != 'baselines':
            work[model].update(semantic_df_work_mean=float(np.mean([r['semantic_df_work'] for r in records])),
                centered_semantic_mse=float(np.mean([r['centered_semantic_mse'] for r in records])))
            del sd, sq
            gc.collect()
    comparisons = {a + '-minus-' + b: dense.paired(arrays[a] - arrays[b], domains)
        for a, b in ((a + '_hybrid', b) for a in ('S', 'P')
                     for b in ('initial_hybrid', 'dense', 'dense_hybrid'))}
    comparisons['S_hybrid-minus-P_hybrid'] = dense.paired(arrays['S_hybrid'] - arrays['P_hybrid'], domains)
    return dict(passed=True, quality=quality, metrics=['ndcg10', 'recall100'], comparisons=comparisons,
        proxies=work, max_independent_head_error=peak_error,
        full_hybrid_evaluated=True, native_cost_evaluated=False,
        independent_holdout=False, automatic_promotion=False, locked_test_access=False)


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
    task = Task.init(project_name='II42-NG', task_name='NG-0080/SP-endpoint/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=TRACKING_FRAMEWORKS.copy(),
        auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output, base, run = args.run / args.phase, args.research_root, args.run
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    campaign.io.write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(input_sha256=campaign.io.sha(run / 'inputs.json'), phase=args.phase))
        if args.phase == 'audit-training':
            result = audit.review(base, probability_device='cuda')
        else:
            require(audit.sealed(phase_root(base, run, 'audit-training'))['passed'], 'audit barrier')
            if args.phase == 'prepare-lexical':
                result = prepare_lexical(base, run, output)
            elif args.phase == 'calibrate-dense':
                result = calibrate_dense(base, run, output)
            elif args.phase.startswith('encode-'):
                result = encode(base, run, output, args.phase.removeprefix('encode-'))
            else:
                audit.sealed(run / 'calibrate-dense')
                for model in MODELS:
                    audit.sealed(phase_root(base, run, 'encode-' + model))
                result = (review(base, run) if args.phase == 'review'
                          else rank(base, run, output, args.phase.removeprefix('rank-')))
        verify(base, run)
        campaign.io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        tick = time.monotonic()
        task.close()
        receipt.update(closed=True, close_seconds=time.monotonic() - tick)
        campaign.io.write(output / 'clearml.json', receipt)


def supervise(args):
    import fcntl
    import os
    import signal

    verify(args.research_root, args.run)
    reused = campaign.read(args.run / 'inputs.json').get('reused_phases', {})
    require(not any(phase in reused for phase in LANES[args.lane]),
            'reused phases must not execute again')
    cuda = args.lane in ('audit', 'first', 'second')
    require(not cuda or args.gpu in (0, 3), 'explicit permitted GPU')
    require(not (args.run / f'{args.lane}-controller-start.json').exists(), 'never restart lane')
    stopped = []
    handlers = {sig: signal.signal(sig, lambda *_: stopped.append(True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    path = args.research_root / (f'.ng-gpu-{args.gpu}.lock' if cuda else '.ng80-cpu.lock')
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cuda:
            memory, use = campaign.guard.gpu_state(args.gpu)
            require(memory < 100 and use == 0, 'GPU occupied')
        campaign.io.write(args.run / f'{args.lane}-controller-start.json',
            dict(pid=os.getpid(), physical_gpu=args.gpu if cuda else None, started_unix=time.time()))
        args.worker_script = 'ng80_sparse_endpoints.py'
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
    parser.add_argument('mode', choices=('freeze', 'verify', 'worker', 'supervise'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--lane', choices=LANES)
    parser.add_argument('--gpu', type=int, choices=(0, 3))
    parser.add_argument('--reuse-v2', action='store_true')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run, reuse=args.reuse_v2)
    elif args.mode == 'worker':
        require(args.phase in PHASES, 'phase required')
        worker(args)
    elif args.mode == 'supervise':
        require(args.lane in LANES, 'lane required')
        supervise(args)
    else:
        verify(args.research_root, args.run)
