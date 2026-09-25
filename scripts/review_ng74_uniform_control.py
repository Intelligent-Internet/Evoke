"""Terminal comparison of uniform U to frozen matched NG71 baselines."""

import numpy as np
from scipy import sparse

import analyze_ng66_learning_curves as audit
import ng71_execution as execution
import ng71_observation as observation
import ng71_pilot as pipeline
import ng71_preflight as io
import ng74_uniform_control as control
import review_ng71_pilot as previous


def review(base, run):
    control.verify(base, run)
    config = execution.read(run / 'config.json')
    phases = {p: execution.sealed(run / p) for p in control.PHASES if p != 'review'}
    last_exit = max(execution.read(run / p / 'exit.json')['finished_unix'] for p in control.TRAIN_PHASES)
    for name in ('encode-U-192', 'rank-U-192'):
        audit.require(execution.read(run / name / 'started.json')['started_unix'] >= last_exit,
                      'observation preceded terminal training')
    trained = previous.audit_training(base, run, config, phases, arms='U',
                                      witness_source=base / control.PARENT / 'snapshot-96')
    selection = execution.read(run / 'selection.json')
    queries = observation.surface_queries(base, selection, True)
    labels = execution.read(base / 'NG-0069/lexical-v1/labels.json')
    records = {'U': pipeline.rows(run / 'rank-U-192/rankings.jsonl')}
    for name, phase in (('D', 'D-192'), ('A', 'A-192'), ('initial', 'initial'), ('dense', 'dense')):
        records[name] = pipeline.rows(base / control.PARENT / ('rank-' + phase) / 'rankings.jsonl')
    for name, rows in records.items():
        audit.require(len(rows) == len(queries), 'incomplete ranking surface')
        for row, q in zip(rows, queries, strict=True):
            audit.require(row['query_id'] == q['query_id'] and row['domain'] == q['subset']
                          and row['split'] == q['split'] and row['surface'] == q['surface'],
                          'query identity or split differs')
            audit.audit_row(row, labels[q['lexical_index']], config['pilot']['corpus_documents'])
    encoded = phases['encode-U-192']
    audit.require(encoded['model_sha256'] == phases['train-U-192']['model_sha256'], 'wrong model encoded')
    audit.require(execution.read(run / 'encode-U-192/query-ids.json') == [q['query_id'] for q in queries],
                  'encoded query identity differs')
    ranked = phases['rank-U-192']
    audit.require(ranked['all_query_score_and_rank_checks_passed']
                  and ranked['max_independent_score_error'] <= 1e-12, 'full-corpus scores not verified')
    q, d = [sparse.load_npz(run / 'encode-U-192' / (role + '.npz')) for role in ('query', 'document')]
    counts = observation.sparse_counts(q, d, queries)
    audit.require(counts == ranked['counts'], 'sparse counts disagree')
    df = np.diff(d.tocsc().indptr).astype(np.int64)
    work = np.array([sum(df[q.indices[q.indptr[i]:q.indptr[i + 1]]]) for i in range(len(queries))])
    for surface in counts['query_df_by_surface']:
        values = work[[r['surface'] == surface for r in queries]]
        audit.require(float(values.mean()) == counts['query_df_by_surface'][surface]['mean'], 'CSC DF differs')
    parent_review = execution.read(base / control.PARENT / 'review/results.json')
    initial_counts = parent_review['sparse_cost_proxy']['initial']
    ratios = dict(document_nnz_vs_init=counts['document_nnz'] / initial_counts['document_nnz'],
                  dev_query_df_vs_init=counts['query_df_by_surface']['DEV_NEW']['mean']
                  / initial_counts['query_df_by_surface']['DEV_NEW']['mean'])
    quality, comparisons, harm = {}, {}, {}
    for surface in ('TRAIN_PILOT', 'TRAIN_SENTINEL', 'DEV_NEW'):
        chosen = {name: [r for r in rows if r['surface'] == surface] for name, rows in records.items()}
        quality[surface] = {name: audit.summary(rows) for name, rows in chosen.items()}
        values = {name: np.array([[r[m] for m in audit.METRICS] for r in rows]) for name, rows in chosen.items()}
        domains = [r['domain'] for r in chosen['U']]
        comparisons[surface] = {name: previous.paired(values['U'] - values[name], domains)
                                for name in ('D', 'A', 'initial', 'dense')}
        harm[surface] = {name: audit.harm(chosen[name], chosen['U']) for name in ('D', 'A', 'initial', 'dense')}
    gate = previous.advancement(comparisons['DEV_NEW']['A'], ratios['document_nnz_vs_init'],
                                ratios['dev_query_df_vs_init'], config['gates'])
    gate['uniform_pair_ablation_required'] = False
    # A completed control is not automatic confirmation of the underlying model.
    return dict(passed=True, stage='NG74_uniform_weight_terminal', quality=quality,
                comparisons_U_minus=comparisons, harm_U_vs=harm, training_cost=trained,
                sparse_cost_proxy=counts, cost_ratios_vs_initial=ratios,
                inherited_quality_gate_against_A=gate, native_cost_evaluated=False,
                rows_audited=sum(map(len, records.values())), locked_test_scored=False,
                overall_goal_qualified=False, automatic_scale_authorized=False)
