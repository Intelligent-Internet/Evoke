#!/usr/bin/env python3
"""Audit completed NG69 artifacts and apply the predeclared breadth gate."""

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_ng66_learning_curves as audit
import ng69_pipeline as pipeline


SEEDS = pipeline.SEEDS
DOMAINS = audit.DOMAINS
SPLITS = ('TRAIN', 'DEV_NEW', 'DEV_EXPOSED')
METRICS = audit.METRICS


def paired(candidate, baseline, domains):
    """Three order seeds share queries; resample queries, never seed/query rows."""
    audit.require(candidate.shape == baseline.shape == (3, len(domains), 2),
                  'Expected three matched order seeds')
    audit.require(np.isfinite(candidate).all() and np.isfinite(baseline).all(),
                  'Non-finite paired metrics')
    delta = (candidate - baseline).mean(axis=0)
    result = {'domains': {}}
    for domain in DOMAINS:
        part = delta[np.asarray(domains) == domain]
        audit.require(len(part) > 0, 'Missing domain')
        result['domains'][domain] = dict(zip(METRICS, part.mean(axis=0).tolist()))
    result['macro'] = {key: float(np.mean([result['domains'][d][key]
                                          for d in DOMAINS])) for key in METRICS}
    result['ndcg10_ci95'] = audit.paired_interval(delta[:, 0], domains)
    result['queries'] = len(domains)
    result['order_seeds'] = list(SEEDS)
    result['bootstrap_seed'] = 66066
    result['bootstrap_replicates'] = 10000
    result['interval_conditional_on_order_seeds'] = True
    return result


def breadth_gate(comparison):
    checks = {
        'positive_macro_ndcg': comparison['macro']['ndcg10'] > 0,
        'positive_paired_interval': comparison['ndcg10_ci95'][0] > 0,
        'macro_recall_floor': comparison['macro']['recall100'] >= -.005,
        'domain_ndcg_floors': all(comparison['domains'][d]['ndcg10'] >= -.01
                                  for d in DOMAINS),
    }
    return {'checks': checks, 'passed': all(checks.values()),
            'overall_goal_qualified': False, 'native_cost_evaluated': False}


def array(records, split):
    return np.array([[r[key] for key in METRICS] for r in records
                     if r['split'] == split])


def sparse_cost(folder, encoded, ranked):
    from scipy import sparse

    matrices, counts = {}, {}
    for role, count in (('query', 7872), ('document', 233009)):
        matrix = sparse.load_npz(folder / (role + '.npz'))
        audit.require(matrix.shape[0] == count and matrix.has_canonical_format,
                      'Sparse shape or canonical format changed')
        audit.require(matrix.dtype == np.float32 and np.isfinite(matrix.data).all()
                      and (matrix.data > 0).all(), 'Invalid sparse readout')
        values = {'nnz': matrix.nnz, 'csr_bytes': sum(a.nbytes for a in
                  (matrix.data, matrix.indices, matrix.indptr))}
        audit.require(values == encoded['counts'][role], 'Stored sparse cost changed')
        matrices[role], counts[role] = matrix, values
    q, d = matrices['query'], matrices['document']
    audit.require(q.shape[1] == d.shape[1], 'Readout dimension changed')
    df = np.bincount(d.indices, minlength=d.shape[1])
    work = q.sign().astype(np.float64) @ df
    values = {'mean': float(work.mean()), 'p95': float(np.quantile(work, .95)),
              'dev_new_mean': float(work[6144:7680].mean())}
    for key, value in values.items():
        audit.require(abs(value - ranked['counts']['semantic_df_proxy'][key]) < 1e-9,
                      'DF work proxy changed')
    counts['semantic_df_proxy'] = values
    return counts


def analyze(base, run):
    frozen = pipeline.verify(base, run)
    state = pipeline.read(run / 'continuation-state.json')
    audit.require(state['status'] == 'all_phases_complete', 'Pipeline incomplete')
    results = {phase: pipeline.completed(run, phase) for phase in pipeline.schedule()}
    lexical = base / 'NG-0069/lexical-v1'
    queries = pipeline.read(lexical / 'queries.json')
    labels = pipeline.read(lexical / 'labels.json')
    audit.require(len(queries) == len(labels) == 7872, 'Population changed')
    costs, initial = {}, set()
    for seed in SEEDS:
        previous, seen = None, []
        totals = {k: 0 for k in ('query_exposures', 'query_tokens', 'document_tokens',
                                 'candidate_pairs', 'seconds')}
        for quarter in (1, 2, 3, 4):
            phase = f'train-{seed}-{quarter}'
            result = results[phase]
            audit.require(result['steps'] == 384
                          and result['cumulative_steps'] == quarter * 384
                          and result['query_exposures'] == 1536
                          and result['checkpoint_optimizer_replay_exact'],
                          'Training count or restoration changed')
            if previous:
                audit.require(result['initial_model_sha256'] == previous['model_sha256']
                              and result['initial_optimizer_sha256']
                              == previous['optimizer_sha256'], 'Optimizer chain changed')
            else:
                initial.add(result['initial_model_sha256'])
            order = pipeline.read(run / phase / 'query-order.json')
            audit.require(order == pipeline.order(seed, quarter), 'Order changed')
            seen.extend(order)
            progress = pipeline.rows(run / phase / 'progress.jsonl')
            audit.require(len(progress) == 384, 'Missing optimizer progress')
            examples = []
            for step, record in enumerate(progress, (quarter - 1) * 384 + 1):
                audit.require(record['step'] == step and record['update_l2'] > 0,
                              'Missing actual parameter update')
                audit.require(len(record['examples']) == 4, 'Accumulation changed')
                examples.extend(record['examples'])
            audit.require([e['train_index'] for e in examples] == order,
                          'Actual query exposure changed')
            for example in examples:
                audit.require(example['query_id'] == queries[example['train_index']]['query_id']
                              and example['vjp_replay_exact'], 'Example or replay changed')
            for key, field in (('query_tokens', 'query_tokens'),
                               ('document_tokens', 'document_tokens'),
                               ('candidate_pairs', 'documents')):
                audit.require(result[key] == sum(e[field] for e in examples),
                              'Cost counter changed')
            for key in totals:
                totals[key] += result[key]
            previous = result
        audit.require(sorted(seen) == list(range(6144)), 'Missing unique TRAIN query')
        audit.require(previous['model_sha256']
                      == results[f'encode-B-{seed}']['model_sha256'], 'Wrong B checkpoint')
        costs[seed] = totals
    audit.require(len(initial) == 1 and initial.pop()
                  == results['encode-initial']['model_sha256'], 'Initialization changed')
    rankings, quality, counts = {}, {}, {}
    for phase in pipeline.schedule():
        if not phase.startswith('rank-'):
            continue
        model = phase.removeprefix('rank-')
        records = pipeline.rows(run / phase / 'rankings.jsonl')
        audit.require(len(records) == 7872, 'Missing ranked query')
        for record, query, gold in zip(records, queries, labels):
            audit.require(record['query_id'] == query['query_id']
                          and record['domain'] == query['subset']
                          and record['split'] == query['split'], 'Rank identity changed')
            audit.audit_row(record, gold, 233009)
        rankings[model] = records
        quality[model] = {}
        for split in SPLITS:
            summary = audit.summary([r for r in records if r['split'] == split])
            reported = results[phase]['quality'][split]
            for key in METRICS:
                audit.require(abs(summary['macro'][key] - reported['macro'][key]) < 1e-12,
                              'Reported macro changed')
                for domain in DOMAINS:
                    audit.require(abs(summary['domains'][domain][key]
                                      - reported['domains'][domain][key]) < 1e-12,
                                  'Reported domain changed')
            quality[model][split] = summary
        if model not in ('bm25', 'dense'):
            counts[model] = sparse_cost(run / ('encode-' + model),
                                       results['encode-' + model], results[phase])
    comparisons, harms = {}, {}
    for split in SPLITS:
        domains = [q['subset'] for q in queries if q['split'] == split]
        a = np.stack([array(rankings[f'A-{s}'], split) for s in SEEDS])
        b = np.stack([array(rankings[f'B-{s}'], split) for s in SEEDS])
        dense = np.repeat(array(rankings['dense'], split)[None, :, :], 3, axis=0)
        comparisons[split] = {'B_minus_A': paired(b, a, domains),
                              'B_minus_dense': paired(b, dense, domains),
                              'A_minus_dense': paired(a, dense, domains)}
        harms[split] = {}
        for seed in SEEDS:
            def select(model):
                return [r for r in rankings[model] if r['split'] == split]

            harms[split][seed] = {
                'B_vs_A': audit.harm(select(f'A-{seed}'), select(f'B-{seed}')),
                'B_vs_dense': audit.harm(select('dense'), select(f'B-{seed}'))}
    gate = breadth_gate(comparisons['DEV_NEW']['B_minus_A'])
    return {'stage': 'NG-0069', 'input_sha256': pipeline.sha(run / 'inputs.json'),
            'pipeline_source_sha256': frozen['source_sha256'],
            'inherited_successes': frozen.get('inherited'),
            'audit_source_sha256': pipeline.sha(Path(__file__)),
            'audit_helper_sha256': pipeline.sha(Path(audit.__file__)),
            'quality': quality, 'comparisons': comparisons, 'rank_harm': harms,
            'breadth_gate': gate, 'B_training_cost': costs, 'cost_proxy': counts,
            'A_token_cost_measured_here': False,
            'all_query_records_audited': 70848,
            'locked_test_scored': False, 'native_cost_evaluated': False,
            'overall_goal_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit.require(not args.output.exists(), 'Never overwrite a review')
    result = analyze(args.research_root.resolve(), args.run.resolve())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('NG69_PAIRED_REVIEW_PASSED', result['breadth_gate'])


if __name__ == '__main__':
    main()
