"""Independent terminal NG71 metric, exposure, objective and cost review.

Intervals are conditional on one initialization/order, not multi-seed evidence.
No model inference, witness mining, retuning or promotion occurs here.
"""

import json
import math
import sys

import numpy as np
from scipy import sparse

sys.dont_write_bytecode = True

import analyze_ng66_learning_curves as audit
import ng71_diagnostics as diagnostics
import ng71_execution as execution
import ng71_observation as observation
import ng71_pilot as pipeline
import ng71_preflight as io


def paired(delta, domains):
    delta, domains = np.asarray(delta), np.asarray(domains)
    if delta.shape != (len(domains), 2) or not np.isfinite(delta).all():
        raise ValueError('finite matched query metric pairs required')
    rng = np.random.default_rng(71071)
    means, by_domain = [], {}
    for domain in audit.DOMAINS:
        values = delta[domains == domain]
        if not len(values):
            raise ValueError('missing paired domain')
        by_domain[domain] = dict(zip(audit.METRICS, values.mean(0).tolist()))
        draws = rng.integers(len(values), size=(10000, len(values)))
        means.append(values[draws].mean(1))
    intervals = np.quantile(np.mean(means, axis=0), [.025, .975], axis=0)
    return dict(
        macro={m: float(np.mean([by_domain[d][m] for d in audit.DOMAINS])) for m in audit.METRICS},
        domains=by_domain, ci95={m: intervals[:, i].tolist() for i, m in enumerate(audit.METRICS)},
        queries=len(domains), bootstrap_replicates=10000, bootstrap_seed=71071,
        conditional_on_single_initialization_and_order=True)


def advancement(comparison, doc_ratio, query_df_ratio, gates):
    checks = dict(
        ndcg_point=comparison['macro']['ndcg10'] >= gates['D_minus_A_ndcg_point_min'],
        ndcg_interval=comparison['ci95']['ndcg10'][0] > gates['D_minus_A_ndcg_95_lower_strictly_above'],
        recall_interval=comparison['ci95']['recall100'][0] >= gates['D_minus_A_recall_95_lower_min'],
        domain_ndcg=all(r['ndcg10'] >= gates['D_minus_A_domain_ndcg_point_min']
                        for r in comparison['domains'].values()),
        domain_recall=all(r['recall100'] >= gates['D_minus_A_domain_recall_point_min']
                          for r in comparison['domains'].values()))
    if not all(math.isfinite(v) and v > 0 for v in (doc_ratio, query_df_ratio)):
        raise ValueError('invalid cost ratio against common initialization')
    cost = (doc_ratio <= gates['scale_block_doc_nnz_ratio_vs_init_above']
            and query_df_ratio <= gates['scale_block_query_df_mean_ratio_vs_init_above'])
    return dict(checks=checks, quality_passed=all(checks.values()),
                relative_cost_proxy_passed=cost, uniform_pair_ablation_required=all(checks.values()),
                automatic_scale_authorized=False, overall_goal_qualified=False,
                native_cost_evaluated=False)


def audit_training(base, run, config, phases, *, arms='ABCD', witness_source=None):
    order = pipeline.read(run / 'training-order.json')
    snapshots = {}
    source = run / 'snapshot-96' if witness_source is None else witness_source
    for start, folder in ((0, base / 'NG-0071/pilot-step0-v1'), (96, source)):
        snapshots[start] = {r['query_id']: r for r in pipeline.rows(folder / 'witnesses.jsonl')}
    totals = {}
    for arm in arms:
        seen, previous = [], None
        totals[arm] = dict(updates=0, query_tokens=0, document_tokens=0,
                           candidate_pairs=0, worker_seconds=0., clipped_updates=0,
                           min_update_l2=None, max_update_l2=0., max_score_gradient_error=0.)
        for end in (96, 192):
            name = f'train-{arm}-{end}'
            result = phases[name]
            initial = previous['model_sha256'] if previous else config['base']['state_sha256']
            audit.require(result['initial_model_sha256'] == initial
                          and result['checkpoint_optimizer_replay_exact'], 'model chain changed')
            if previous:
                audit.require(result['initial_optimizer_fingerprint'] == previous['optimizer_fingerprint']
                              and result['previous_complete_sha256'] == io.sha(
                                  run / f'train-{arm}-96/complete.json'), 'optimizer chain changed')
            audit.require(io.sha(run / name / 'optimizer.pt') == result['optimizer_sha256'],
                          'optimizer artifact fingerprint changed')
            progress = pipeline.rows(run / name / 'progress.jsonl')
            audit.require(len(progress) == 96, 'optimizer progress incomplete')
            start = end - 96
            phase_tokens, candidates = dict(query=0, document=0), 0
            for index, step in enumerate(progress, start):
                expected_ids = order[index * 4:(index + 1) * 4]
                audit.require(step['step'] == index + 1 and step['reference_update'] == start
                              and [e['query_id'] for e in step['examples']] == expected_ids,
                              'actual query exposure or reference changed')
                audit.require(math.isfinite(step['update_l2']) and step['update_l2'] > 0
                              and math.isfinite(step['gradient_before_clip'])
                              and step['gradient_before_clip'] > 0, 'invalid actual parameter update')
                totals[arm]['clipped_updates'] += step['gradient_before_clip'] > 1
                values = [step['update_l2'], totals[arm]['min_update_l2']]
                totals[arm]['min_update_l2'] = min(v for v in values if v is not None)
                totals[arm]['max_update_l2'] = max(totals[arm]['max_update_l2'], step['update_l2'])
                for example in step['examples']:
                    record = snapshots[start][example['query_id']]
                    if config['arms'][arm]['pool'] == 'original':
                        record = diagnostics.original_record(record)
                    expected = diagnostics.objective(
                        dict(record, hybrid_scores=example['scores']), config['ranking'],
                        config['arms'][arm]['objective'])
                    # Logged gradients include four-query accumulation. Reference is float64.
                    target = np.asarray(expected['score_gradient']) / 4
                    observed = np.asarray(example['score_gradient'])
                    np.testing.assert_allclose(observed, target, rtol=2e-5, atol=2e-7)
                    np.testing.assert_allclose(example['loss'], expected['loss'], rtol=2e-6, atol=2e-7)
                    error = float(np.max(np.abs(observed - target)))
                    totals[arm]['max_score_gradient_error'] = max(
                        totals[arm]['max_score_gradient_error'], error)
                    audit.require(example['vjp_replay_exact'] and example['documents'] == len(record['pool'])
                                  and example['objective'] == config['arms'][arm]['objective'],
                                  'fresh-forward objective or candidate coverage changed')
                    for role in phase_tokens:
                        phase_tokens[role] += example[role + '_tokens']
                    candidates += example['documents']
                seen.extend(expected_ids)
            audit.require(phase_tokens == result['tokens'] and candidates == result['candidate_pairs'],
                          'actual exposure cost counters changed')
            totals[arm]['updates'] += len(progress)
            totals[arm]['query_tokens'] += phase_tokens['query']
            totals[arm]['document_tokens'] += phase_tokens['document']
            totals[arm]['candidate_pairs'] += candidates
            totals[arm]['worker_seconds'] += pipeline.read(run / name / 'exit.json')['elapsed_seconds']
            previous = result
        audit.require(seen == order and totals[arm]['updates'] == 192, 'training coverage changed')
    return totals


def review(base, run):
    pipeline.verify(base, run)
    config = pipeline.read(run / 'config.json')
    phases = {name: execution.sealed(run / name) for name in pipeline.schedule() if name != 'review'}
    last_training_exit = max(pipeline.read(run / name / 'exit.json')['finished_unix']
                             for name in observation.TRAIN_PHASES)
    for name in phases:
        if name.startswith('rank-') or name.startswith('encode-') and name != 'encode-A-96':
            audit.require(pipeline.read(run / name / 'started.json')['started_unix'] >= last_training_exit,
                          'terminal observation started before all training was sealed')
    train = audit_training(base, run, config, phases)
    labels = pipeline.read(base / 'NG-0069/lexical-v1/labels.json')
    selection = pipeline.read(run / 'selection.json')
    rankings, counts, quality = {}, {}, {}
    for name in pipeline.schedule():
        if not name.startswith('rank-'):
            continue
        model = name.removeprefix('rank-')
        queries = observation.surface_queries(base, selection, not model.endswith('-96'))
        records = pipeline.rows(run / name / 'rankings.jsonl')
        audit.require(len(records) == len(queries), 'incomplete ranking surface')
        for record, query in zip(records, queries, strict=True):
            audit.require(record['query_id'] == query['query_id'] and record['domain'] == query['subset']
                          and record['split'] == query['split'] and record['surface'] == query['surface'],
                          'evaluation query identity changed')
            audit.audit_row(record, labels[query['lexical_index']], config['pilot']['corpus_documents'])
        audit.require(phases[name]['all_query_score_and_rank_checks_passed']
                      and phases[name]['max_independent_score_error'] <= 1e-12,
                      'full-background independent accumulation failed')
        rankings[model] = records
        quality[model] = {surface: audit.summary([r for r in records if r['surface'] == surface])
                          for surface in sorted({q['surface'] for q in queries})}
        if model not in ('dense', 'bm25'):
            folder = base / 'NG-0069/evaluation-v2/encode-initial' if model == 'initial' else run / ('encode-' + model)
            q, d = [sparse.load_npz(folder / (role + '.npz')) for role in ('query', 'document')]
            if model == 'initial':
                q = q[[row['lexical_index'] for row in queries]]
            else:
                encoded = phases['encode-' + model]
                arm, update = model.split('-')
                audit.require(encoded['model_sha256'] == phases[f'train-{arm}-{update}']['model_sha256'],
                              'observation used wrong model')
                audit.require(pipeline.read(folder / 'query-ids.json') == [r['query_id'] for r in queries],
                              'observation used wrong query surface')
            counts[model] = observation.sparse_counts(q, d, queries)
            audit.require(counts[model] == phases[name]['counts'], 'sparse payload cost mismatch')
            # A CSC column count independently checks the posting support proxy.
            independent_df = np.diff(d.tocsc().indptr).astype(np.int64)
            independent_work = np.array([sum(independent_df[q.indices[q.indptr[i]:q.indptr[i + 1]]])
                                         for i in range(len(queries))], dtype=np.float64)
            for surface in counts[model]['query_df_by_surface']:
                values = independent_work[[q['surface'] == surface for q in queries]]
                audit.require(float(values.mean()) == counts[model]['query_df_by_surface'][surface]['mean'],
                              'independent CSC posting support differs')
            del q, d
    comparisons, harm, trajectories = {}, {}, {}
    for surface in ('TRAIN_PILOT', 'TRAIN_SENTINEL', 'DEV_NEW'):
        selected = {name: [r for r in rows if r['surface'] == surface]
                    for name, rows in rankings.items() if not name.endswith('-96')}
        values = {name: np.asarray([[r[m] for m in audit.METRICS] for r in rows])
                  for name, rows in selected.items()}
        domains = [r['domain'] for r in selected['initial']]
        pairs = [('B-192', 'A-192'), ('C-192', 'A-192'), ('D-192', 'C-192'),
                 ('D-192', 'A-192'), ('D-192', 'dense'), ('D-192', 'initial')]
        comparisons[surface] = {a + '_minus_' + b: paired(values[a] - values[b], domains) for a, b in pairs}
        comparisons[surface]['interaction'] = paired(
            (values['D-192'] - values['C-192']) - (values['B-192'] - values['A-192']), domains)
        harm[surface] = {a + '_vs_' + b: audit.harm(selected[b], selected[a]) for a, b in pairs}
    initial = [r for r in rankings['initial'] if r['surface'] == 'TRAIN_PILOT']
    for arm in 'ABCD':
        mid = rankings[f'{arm}-96']
        final = [r for r in rankings[f'{arm}-192'] if r['surface'] == 'TRAIN_PILOT']
        trajectories[arm] = {'0_to_96': audit.harm(initial, mid), '96_to_192': audit.harm(mid, final)}
        for key, before, after in (('0_to_96', initial, mid), ('96_to_192', mid, final)):
            for cutoff in (10, 100):
                trajectories[arm][key][f'all_document_entrants_top{cutoff}'] = sum(
                    len(set(b['top100'][:cutoff]) - set(a['top100'][:cutoff]))
                    for a, b in zip(before, after, strict=True))
    cost_ratios = {}
    for arm in 'ABCD':
        value = counts[f'{arm}-192']
        cost_ratios[arm] = dict(
            document_nnz_vs_init=value['document_nnz'] / counts['initial']['document_nnz'],
            dev_query_df_vs_init=value['query_df_by_surface']['DEV_NEW']['mean']
            / counts['initial']['query_df_by_surface']['DEV_NEW']['mean'])
    gate = advancement(comparisons['DEV_NEW']['D-192_minus_A-192'],
                       cost_ratios['D']['document_nnz_vs_init'], cost_ratios['D']['dev_query_df_vs_init'],
                       config['gates'])
    return dict(passed=True, stage='NG71_terminal_four_arm_review', quality=quality,
                comparisons=comparisons, harm=harm, train_trajectories=trajectories,
                training_cost=train, sparse_cost_proxy=counts, cost_ratios_vs_common_init=cost_ratios,
                advancement=gate, locked_test_scored=False, native_cost_evaluated=False,
                overall_goal_qualified=False, input_sha256=io.sha(run / 'inputs.json'),
                rows_audited=sum(map(len, rankings.values())))
