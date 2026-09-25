"""Independent metric, coordinate-head and semantic-cost review of NG79.

The score producer checks two full-corpus reductions. This additional reviewer
reconstructs saved all-gold metrics and head scores, not a second corpus rerank.
"""

import time

import numpy as np
from scipy import sparse

import analyze_ng66_learning_curves as metrics
import ng79_endpoints as protocol
from review_ng75_displacement import compare, read, require, sha
from review_ng79_training import rows, sealed


COMPARISONS = (('B-384', 'R-384'), ('B-384', 'initial'), ('B-384', 'dense'),
               ('R-384', 'initial'), ('R-384', 'dense'))


def paired(delta, domains):
    delta, domains = np.asarray(delta), np.asarray(domains)
    require(delta.shape == (len(domains), 2) and np.isfinite(delta).all(), 'invalid matched deltas')
    rng = np.random.default_rng(79079)
    means, by_domain = [], {}
    for domain in metrics.DOMAINS:
        values = delta[domains == domain]
        require(len(values) > 0, 'missing paired domain')
        by_domain[domain] = dict(zip(metrics.METRICS, values.mean(0).tolist(), strict=True))
        draws = rng.integers(len(values), size=(10000, len(values)))
        means.append(values[draws].mean(1))
    intervals = np.quantile(np.mean(means, axis=0), [.025, .975], axis=0)
    return dict(macro={m: float(np.mean([by_domain[d][m] for d in metrics.DOMAINS])) for m in metrics.METRICS},
        domains=by_domain, ci95={m: intervals[:, i].tolist() for i, m in enumerate(metrics.METRICS)},
        queries=len(domains), bootstrap_replicates=10000, bootstrap_seed=79079,
        conditional_on_single_initialization_and_order=True, independent_holdout=False)


def advancement(comparisons, doc_ratio, df_ratio, gates):
    require(gates == protocol.GATES, 'pre-training gates changed')
    br, bi = comparisons['B-384-minus-R-384'], comparisons['B-384-minus-initial']
    require(all(np.isfinite(v) and v > 0 for v in (doc_ratio, df_ratio)), 'invalid cost ratios')
    floor = gates['per_domain_ndcg_and_recall_floor_vs_R_and_initial']
    checks = dict(
        ndcg_point=br['macro']['ndcg10'] >= gates['ndcg_point_min'],
        ndcg_interval=br['ci95']['ndcg10'][0] > gates['ndcg_95_lower_strictly_above'],
        recall_interval=br['ci95']['recall100'][0] >= gates['recall_95_lower_min'],
        per_domain_ndcg=all(c['domains'][d]['ndcg10'] >= floor for c in (br, bi) for d in metrics.DOMAINS),
        per_domain_recall=all(c['domains'][d]['recall100'] >= floor for c in (br, bi) for d in metrics.DOMAINS),
        document_nnz=doc_ratio <= gates['document_nnz_ratio_vs_initial_max'],
        query_df=df_ratio <= gates['sentinel_query_df_ratio_vs_initial_max'])
    return dict(checks=checks, exploratory_gate_passed=all(checks.values()),
                automatic_scale_authorized=False, independent_holdout=False,
                overall_goal_qualified=False, native_cost_evaluated=False)


def audit_counts(query, document, queries, counts):
    require(query.shape[0] == len(queries) and document.shape[0] == 233009
            and query.shape[1] == document.shape[1], 'cost universe changed')
    for matrix in (query, document):
        require(matrix.has_canonical_format and np.isfinite(matrix.data).all()
                and (matrix.data > 0).all(), 'invalid sparse cost payload')
    require(counts['document_nnz'] == len(document.indices) and counts['query_nnz'] == len(query.indices)
            and counts['document_csr_bytes'] == sum(a.nbytes for a in
                (document.data, document.indices, document.indptr)), 'physical CSR counters differ')
    df = np.bincount(document.indices, minlength=document.shape[1])
    work = [sum(int(df[j]) for j in query.indices[query.indptr[i]:query.indptr[i + 1]])
            for i in range(len(queries))]
    for surface in protocol.SURFACES:
        values = [w for w, q in zip(work, queries, strict=True) if q['surface'] == surface]
        expected = dict(mean=float(np.mean(values)), p95=float(np.quantile(values, .95)))
        require(expected == counts['query_df_by_surface'][surface], 'per-surface query-DF differs')


def coordinates(query, document, ids):
    selected = document[ids].astype(np.float64)
    return np.asarray(selected.multiply(query.astype(np.float64)).sum(axis=1)).ravel()


def review(base, run):
    manifest = read(run / 'inputs.json')
    require(manifest['gates'] == protocol.GATES, 'gates changed')
    queries = read(run / 'endpoint-queries.json')
    selection = read(run / 'selection.json')
    ids = selection['repeated'] + selection['sentinel'] + [
        q for q in selection['breadth'] if q not in set(selection['repeated'])]
    require(len(ids) == len(set(ids)) == 1920 and [q['query_id'] for q in queries] == ids,
            'endpoint identities changed')
    lexical = base / 'NG-0069/lexical-v1'
    original = read(lexical / 'queries.json')
    labels = read(lexical / 'labels.json')
    for i, q in enumerate(queries):
        surface = protocol.SURFACES[0 if i < 384 else 1 if i < 768 else 2]
        require(q == dict(original[q['lexical_index']], surface=surface, lexical_index=q['lexical_index'])
                and q['split'] == 'TRAIN', 'query text/role/split changed')
    parent = base / protocol.TRAIN
    last_train = max(read(parent / p / 'exit.json')['finished_unix'] for p in protocol.training.PHASES)
    for phase in protocol.PHASES[:-1]:
        sealed(run / phase)
        require(read(run / phase / 'started.json')['started_unix'] >= last_train, 'evaluation overlapped training')
    for model in ('R-384', 'B-384'):
        encoded, trained = sealed(run / ('encode-' + model)), sealed(parent / ('train-' + model))
        require(encoded['model_sha256'] == trained['model_sha256']
                and encoded['checkpoint_complete_sha256'] == sha(parent / ('train-' + model) / 'complete.json')
                and read(run / ('encode-' + model) / 'query-ids.json') == ids, 'encoded model/query mismatch')
    started = time.monotonic()
    lq, ld = [sparse.load_npz(lexical / f'lexical-{role}.npz') for role in ('query', 'document')]
    rankings, quality, counts, head_errors = {}, {}, {}, {}
    for model in protocol.MODELS:
        result = sealed(run / ('rank-' + model))
        records = list(rows(run / ('rank-' + model) / 'rankings.jsonl'))
        require(len(records) == result['queries'] == 1920 and result['all_query_score_and_rank_checks_passed']
                and result['max_independent_score_error'] <= 1e-12 and not result['locked_test_scored'],
                'incomplete dual full-corpus score/rank evidence')
        sq = sd = dq = dd = None
        if model == 'dense':
            folder = base / 'NG-0069/evaluation-v2/encode-dense'
            dq, dd = [np.load(folder / f'{role}.npy', mmap_mode='r') for role in ('query', 'document')]
        elif model != 'bm25':
            folder = base / 'NG-0069/evaluation-v2/encode-initial' if model == 'initial' else run / ('encode-' + model)
            sq, sd = [sparse.load_npz(folder / f'{role}.npz') for role in ('query', 'document')]
            if model == 'initial':
                sq = sq[[q['lexical_index'] for q in queries]]
            audit_counts(sq, sd, queries, result['counts'])
        largest = 0.
        for i, (row, q) in enumerate(zip(records, queries, strict=True)):
            require(time.monotonic() - started < 1500, 'bounded endpoint review exceeded budget')
            require(row['query_id'] == q['query_id'] and row['domain'] == q['subset']
                    and row['surface'] == q['surface'] and row['split'] == 'TRAIN', 'rank row identity changed')
            metrics.audit_row(row, labels[q['lexical_index']], 233009)
            head, index = row['top100'], q['lexical_index']
            require(len(head) == len(set(head)) == 100, 'incomplete head')
            if model == 'dense':
                scores = np.sum(dd[head].astype(np.float64) * dq[index].astype(np.float64), axis=1)
            else:
                scores = coordinates(lq.getrow(index), ld, head)
                if sq is not None:
                    scores += coordinates(sq.getrow(i), sd, head)
            np.testing.assert_allclose(scores, row['top100_scores'], rtol=0, atol=1e-12)
            largest = max(largest, float(np.max(abs(scores - row['top100_scores']))))
        if model in ('initial', 'dense', 'bm25'):
            previous = list(rows(base / 'NG-0077/train-endpoints-v1' / ('rank-' + model) / 'rankings.jsonl'))
            compare(records[:768], previous, atol=1e-12)
        rankings[model], counts[model], head_errors[model] = records, result['counts'], largest
        quality[model] = {s: metrics.summary([r for r in records if r['surface'] == s]) for s in protocol.SURFACES}
    comparisons, harms = {}, {}
    for surface in protocol.SURFACES:
        selected = {m: [r for r in rankings[m] if r['surface'] == surface] for m in protocol.MODELS}
        domains = [r['domain'] for r in selected['initial']]
        comparisons[surface], harms[surface] = {}, {}
        for a, b in COMPARISONS:
            differences = [[x[k] - y[k] for k in metrics.METRICS]
                           for x, y in zip(selected[a], selected[b], strict=True)]
            comparisons[surface][a + '-minus-' + b] = paired(differences, domains)
            harms[surface][a + '-minus-' + b] = metrics.harm(selected[b], selected[a])
    ratios = {m: dict(document_nnz=counts[m]['document_nnz'] / counts['initial']['document_nnz'],
        sentinel_query_df=counts[m]['query_df_by_surface']['TRAIN_SENTINEL']['mean'] /
            counts['initial']['query_df_by_surface']['TRAIN_SENTINEL']['mean']) for m in ('R-384', 'B-384')}
    decision = advancement(comparisons['TRAIN_SENTINEL'], ratios['B-384']['document_nnz'],
                           ratios['B-384']['sentinel_query_df'], manifest['gates'])
    return dict(passed=True, input_sha256=sha(run / 'inputs.json'), quality=quality,
        comparisons=comparisons, harm=harms, counts=counts, cost_ratios=ratios, decision=decision,
        max_independent_head_score_errors=head_errors, checked_head_scores_per_model=192000,
        baseline_old768_reproduced=True, independent_full_corpus_rerank=False,
        quality_gate_is_not_execution_pass=True, independent_holdout=False,
        native_cost_evaluated=False, locked_test_access=False, dev_access=False, training_updates=0)
