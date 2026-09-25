"""Recompute NG77 TRAIN-only endpoint metrics against pre-training gates.

No optimizer or model inference. Confidence intervals are conditional on one
initialization/order and exposed TRAIN queries, not independent generalization.
"""

from collections import defaultdict
import argparse
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import psutil
from scipy import sparse

import analyze_ng66_learning_curves as metrics
import ng71_execution as execution
import ng71_observation as observation
import ng77_pilot as training
from review_ng71_pilot import paired
from review_ng75_displacement import require, read, sha
from review_ng77_training import rows
from review_ng75_displacement import compare, verify


MODELS = ('initial', 'dense', 'bm25', 'Z-96', 'K-96')
SURFACES = ('TRAIN_PILOT', 'TRAIN_SENTINEL')
COMPARISONS = (('K-96', 'Z-96'), ('K-96', 'initial'), ('K-96', 'dense'),
               ('Z-96', 'initial'), ('Z-96', 'dense'))


def advancement(comparisons, doc_ratio, df_ratio, gates):
    kz, ki = comparisons['K-96-minus-Z-96'], comparisons['K-96-minus-initial']
    require(all(np.isfinite(v) and v > 0 for v in (doc_ratio, df_ratio)), 'invalid cost ratios')
    checks = dict(
        ndcg_point=kz['macro']['ndcg10'] >= gates['keep_minus_zero_ndcg_point_min'],
        ndcg_interval=kz['ci95']['ndcg10'][0] > gates['keep_minus_zero_ndcg_95_lower_strictly_above'],
        recall_interval=kz['ci95']['recall100'][0] >= gates['keep_minus_zero_recall_95_lower_min'],
        per_domain_ndcg=all(c['domains'][d]['ndcg10'] >= gates['per_domain_ndcg_floor_vs_zero_and_initial']
                            for c in (kz, ki) for d in metrics.DOMAINS),
        per_domain_recall=all(c['domains'][d]['recall100'] >= gates['per_domain_recall_floor_vs_zero_and_initial']
                              for c in (kz, ki) for d in metrics.DOMAINS),
        document_nnz=doc_ratio <= gates['document_nnz_ratio_vs_initial_max'],
        query_df=df_ratio <= gates['sentinel_query_df_ratio_vs_initial_max'])
    return dict(checks=checks, exploratory_gate_passed=all(checks.values()),
                automatic_scale_authorized=False, independent_holdout=False,
                overall_goal_qualified=False, native_cost_evaluated=False)


def audit_counts(base, run, model, queries, counts):
    root = base / 'NG-0069/evaluation-v2/encode-initial' if model == 'initial' else run / ('encode-' + model)
    query, document = [sparse.load_npz(root / f'{r}.npz') for r in ('query', 'document')]
    if model == 'initial':
        query = query[[q['lexical_index'] for q in queries]]
    require(query.shape[0] == 768 and document.shape[0] == 233009
            and query.shape[1] == document.shape[1], 'cost matrix universe changed')
    for matrix in (query, document):
        require(matrix.has_canonical_format and np.isfinite(matrix.data).all()
                and (matrix.data > 0).all(), 'noncanonical cost payload')
    require(counts['document_nnz'] == len(document.indices)
            and counts['query_nnz'] == len(query.indices)
            and counts['document_csr_bytes'] == sum(x.nbytes for x in
                (document.indices, document.indptr, document.data)), 'NNZ/bytes counts changed')
    df = np.bincount(document.indices, minlength=document.shape[1])
    work = [sum(int(df[j]) for j in query.indices[query.indptr[i]:query.indptr[i + 1]])
            for i in range(len(queries))]
    for surface in SURFACES:
        values = [w for w, q in zip(work, queries, strict=True) if q['surface'] == surface]
        expected = dict(mean=float(np.mean(values)), p95=float(np.quantile(values, .95)))
        require(expected == counts['query_df_by_surface'][surface], 'query-DF work changed')


def trusted_transfer(base, run, rankings, queries):
    """Actual terminal anchor margins, not claims about optimizer causality."""
    witnesses = {r['query_id']: r for r in rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')}
    anchors = read(base / 'NG-0077/preparation-v1/audit/anchors.json')
    lexical = base / 'NG-0069/lexical-v1'
    lq, ld = [sparse.load_npz(lexical / f'lexical-{r}.npz').astype(np.float64)
              for r in ('query', 'document')]
    summaries = {}
    for model in ('Z-96', 'K-96'):
        sq, sd = [sparse.load_npz(run / ('encode-' + model) / f'{r}.npz').astype(np.float64)
                  for r in ('query', 'document')]
        grouped = defaultdict(list)
        for i, q in enumerate(queries):
            if q['surface'] != 'TRAIN_PILOT':
                continue
            record, manifest = witnesses[q['query_id']], anchors[q['query_id']]
            pool, idx = record['pool'], q['lexical_index']
            score = (sq.getrow(i) @ sd[pool].T + lq.getrow(idx) @ ld[pool].T).toarray().ravel()
            other = sd[pool] @ sq.getrow(i).toarray().ravel() + ld[pool] @ lq.getrow(idx).toarray().ravel()
            np.testing.assert_allclose(score, other, rtol=0, atol=1e-12)
            row = rankings[model][i]
            current_ranks = dict(zip(pool, row['reference_staleness']['actual_ranks'], strict=True))
            for a in manifest['anchors']:
                p, n = a['indices']
                margin = float(score[p] - score[n])
                grouped[q['subset']].append(dict(
                    change=margin - a['baseline_margin'], ordered=margin > 0,
                    p_in10=current_ranks[a['positive_id']] <= 10,
                    n_in10=current_ranks[a['rival_id']] <= 10))
        summaries[model] = {d: dict(pairs=len(v), strict_margin_reductions=sum(r['change'] < 0 for r in v),
            mean_margin_change=float(np.mean([r['change'] for r in v])),
            previously_correct_order_lost=sum(not r['ordered'] for r in v),
            rival_in10_positive_out10=sum(r['n_in10'] and not r['p_in10'] for r in v))
            for d, v in grouped.items()}
    return dict(domains=summaries, pool_only=True, shared_parameter_causality_proven=False,
                strict_changes_include_numerical_noise=True)


def review(base, run):
    manifest = read(run / 'inputs.json')
    require(manifest['gates'] == training.GATES, 'pre-training gates changed')
    queries = read(run / 'endpoint-queries.json')
    expected = observation.surface_queries(base, read(run / 'selection.json'), True, include_dev=False)
    require(queries == expected and len(queries) == 768, 'TRAIN-only endpoint selection changed')
    parent = base / 'NG-0077/matched-retention-96-v1'
    training_results = observation.training_closed(parent, training.PHASES)
    last_train = max(read(parent / p / 'exit.json')['finished_unix'] for p in training.PHASES)
    for p in ('audit-training', 'encode-Z-96', 'encode-K-96', *(f'rank-{m}' for m in MODELS)):
        execution.sealed(run / p)
        require(read(run / p / 'started.json')['started_unix'] >= last_train,
                'evaluation overlapped training')
    for m in ('Z-96', 'K-96'):
        result = execution.sealed(run / ('encode-' + m))
        require(result['model_sha256'] == training_results['train-' + m]['model_sha256']
                and result['checkpoint_complete_sha256'] == sha(parent / ('train-' + m) / 'complete.json'),
                'encoding checkpoint identity differs')
    labels = read(base / 'NG-0069/lexical-v1/labels.json')
    rankings, quality, counts = {}, {}, {}
    for model in MODELS:
        root = run / ('rank-' + model)
        result = execution.sealed(root)
        records = rows(root / 'rankings.jsonl')
        require(len(records) == len(queries) and result['queries'] == len(queries)
                and result['all_query_score_and_rank_checks_passed']
                and result['max_independent_score_error'] <= 1e-12
                and not result['locked_test_scored'], 'incomplete independent ranking evidence')
        for row, q in zip(records, queries, strict=True):
            require(row['query_id'] == q['query_id'] and row['domain'] == q['subset']
                    and row['surface'] == q['surface'] and row['split'] == 'TRAIN', 'ranking identity differs')
            metrics.audit_row(row, labels[q['lexical_index']], 233009)
        rankings[model] = records
        quality[model] = {s: metrics.summary([r for r in records if r['surface'] == s]) for s in SURFACES}
        counts[model] = result['counts']
        if model not in ('dense', 'bm25'):
            audit_counts(base, run, model, queries, counts[model])
    comparisons, harms = {}, {}
    for surface in SURFACES:
        selected = {m: [r for r in rankings[m] if r['surface'] == surface] for m in MODELS}
        domains = [r['domain'] for r in selected['initial']]
        comparisons[surface], harms[surface] = {}, {}
        for a, b in COMPARISONS:
            name = a + '-minus-' + b
            differences = [[x[k] - y[k] for k in metrics.METRICS]
                           for x, y in zip(selected[a], selected[b], strict=True)]
            comparisons[surface][name] = paired(differences, domains)
            harms[surface][name] = metrics.harm(selected[b], selected[a])
    ratios = {m: dict(document_nnz=counts[m]['document_nnz'] / counts['initial']['document_nnz'],
        sentinel_query_df=counts[m]['query_df_by_surface']['TRAIN_SENTINEL']['mean']
            / counts['initial']['query_df_by_surface']['TRAIN_SENTINEL']['mean']) for m in ('Z-96', 'K-96')}
    gate = advancement(comparisons['TRAIN_SENTINEL'], ratios['K-96']['document_nnz'],
                       ratios['K-96']['sentinel_query_df'], manifest['gates'])
    outside = {m: {d: dict(queries=sum(r['domain'] == d for r in rankings[m][:384]),
        mean_outside_pool_top10=float(np.mean([len(set(r['top100'][:10]) &
            set(r['reference_staleness']['outside_reference_pool_top100']))
            for r in rankings[m][:384] if r['domain'] == d]))) for d in metrics.DOMAINS}
        for m in ('Z-96', 'K-96')}
    return dict(passed=True, input_sha256=sha(run / 'inputs.json'), quality=quality,
        comparisons=comparisons, harm=harms, counts=counts, cost_ratios=ratios, decision=gate,
        trusted_transfer=trusted_transfer(base, run, rankings, queries), outside_pool=outside,
        quality_gate_is_not_execution_pass=True, independent_holdout=False,
        native_cost_evaluated=False, locked_test_access=False, training_updates=0)


def coordinate_scores(query, documents, identities):
    """Independent coordinate-product reduction, not the producer's sparse dot."""
    selected = documents[identities].astype(np.float64)
    return np.asarray(selected.multiply(query.astype(np.float64)).sum(axis=1)).ravel()


def check_head_scores(base, run, guard):
    queries = read(run / 'endpoint-queries.json')
    indices = [q['lexical_index'] for q in queries]
    require(len(queries) == 768 and all(q['split'] == 'TRAIN' for q in queries),
            'head review must use the frozen TRAIN surface')
    lexical = base / 'NG-0069/lexical-v1'
    lq, ld = [sparse.load_npz(lexical / f'lexical-{r}.npz') for r in ('query', 'document')]
    checked, errors = {}, {}
    for model in MODELS:
        guard()
        sq, sd, dq, dd = None, None, None, None
        if model == 'dense':
            root = base / 'NG-0069/evaluation-v2/encode-dense'
            dq = np.load(root / 'query.npy', mmap_mode='r')
            dd = np.load(root / 'document.npy', mmap_mode='r')
        elif model != 'bm25':
            root = (base / 'NG-0069/evaluation-v2/encode-initial' if model == 'initial'
                    else run / ('encode-' + model))
            sq, sd = [sparse.load_npz(root / f'{r}.npz') for r in ('query', 'document')]
        records = rows(run / ('rank-' + model) / 'rankings.jsonl')
        require(len(records) == len(queries), 'head rows incomplete')
        largest = 0.
        for i, (q, row) in enumerate(zip(queries, records, strict=True)):
            guard()
            require(row['query_id'] == q['query_id'] and row['split'] == 'TRAIN',
                    'head row identity differs')
            ids, index = row['top100'], indices[i]
            require(len(ids) == len(set(ids)) == 100, 'head IDs incomplete')
            if model == 'dense':
                score = np.sum(dd[ids].astype(np.float64) * dq[index].astype(np.float64), axis=1)
            else:
                score = coordinate_scores(lq.getrow(index), ld, ids)
                if sq is not None:
                    score += coordinate_scores(sq.getrow(index if model == 'initial' else i), sd, ids)
            np.testing.assert_allclose(score, row['top100_scores'], rtol=0, atol=1e-12)
            largest = max(largest, float(np.max(abs(score - row['top100_scores']))))
        checked[model], errors[model] = len(records) * 100, largest
    return dict(checked_head_scores=checked, max_head_score_errors=errors,
                independent_full_corpus_rerank=False, new_model_inference=False)


def review_mirror(base, run):
    import ng77_endpoints as endpoint
    import review_ng77_training as actual_training

    started, peak = time.monotonic(), 0

    def guard():
        nonlocal peak
        peak = max(peak, psutil.Process().memory_info().rss)
        require(peak < 16 * 1024 ** 3 and psutil.virtual_memory().available >= 24 * 1024 ** 3
                and time.monotonic() - started < 900, 'bounded local review stopped')

    guard()
    inventory_path = run.with_name(run.name + '-remote-inventory.json')
    require(sha(inventory_path) == 'dd8e23dc862f57ef0e54779c4e89df170973a928525ec7c258469b08c2f19454',
            'endpoint mirror inventory changed')
    inventory = read(inventory_path)
    require(sha(run / 'inputs.json') == '46fed531851ee1b9d7a49f7ef9ed32436aec1e55fe52d293e4d7759cdc1400d7'
            and sha(run / 'review/complete.json') == 'ab4ead031c36d890379d01a8f46bf3e4f8debde8db012675fff86cbd8aa4ae3a',
            'endpoint identity changed')
    verify(run, inventory['files'], exact=True)
    manifest = endpoint.verify(base, run)
    require(read(run / 'controller-exit.json')['status'] == 'all_phases_complete',
            'endpoint controller not closed')
    observed = execution.sealed(run / 'review')
    training_result = actual_training.review(base, base / endpoint.TRAIN)
    training_error = compare(training_result, execution.sealed(run / 'audit-training'))
    guard()
    reduced = review(base, run)
    reduction_error = compare(reduced, observed)
    heads = check_head_scores(base, run, guard)
    verify(run, inventory['files'], exact=True)
    endpoint.verify(base, run)
    guard()
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    high_water = int(usage if sys.platform == 'darwin' else usage * 1024)
    require(high_water < 16 * 1024 ** 3, 'local review RSS high-water exceeded')
    sources = {Path(m.__file__).name: sha(Path(m.__file__)) for m in list(sys.modules.values())
               if getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py'
               and Path(m.__file__).resolve().parent == Path(__file__).resolve().parent}
    return dict(passed=True, exact_mirror_files=len(inventory['files']),
        exact_mirror_bytes=sum((run / p).stat().st_size for p in inventory['files']),
        dependencies=len(manifest['dependencies']), frozen_sources=len(manifest['source']),
        imported_review_sources=sources, training_reduction_error=training_error,
        endpoint_reduction_error=reduction_error, **heads, decision=reduced['decision'],
        peak_rss_bytes=max(peak, high_water), elapsed_seconds=time.monotonic() - started,
        locked_test_access=False, dev_access=False, training_updates=0,
        actual_parameter_vjp_recomputed=False, native_cost_evaluated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and args.output.parent.resolve() == args.run.parent.resolve(),
            'new review output must remain outside frozen run')

    def timeout(signum, frame):
        raise TimeoutError('local review exceeded 900 seconds')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    result = review_mirror(args.research_root, args.run)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    signal.alarm(0)
    print(json.dumps(result, allow_nan=False))
