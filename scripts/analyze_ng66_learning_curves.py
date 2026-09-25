#!/usr/bin/env python3
"""Independently review frozen NG66 rankings; never train or score a model."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


SEEDS = (59059, 66061, 66067)
EPOCHS = (1, 2, 4)
DOMAINS = ('fever', 'hotpotqa', 'nq')
SPLITS = ('TRAIN', 'DEV_EXPOSED')
METRICS = ('ndcg10', 'recall100')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_row(row, gold, count):
    """Recompute metrics and check the stored top-100/gold-rank contract."""
    require(row['gold_ids'] == gold and len(set(gold)) == len(gold),
            'Gold identities changed')
    ranks = row['gold_ranks']
    require(len(ranks) == len(gold) > 0, 'Missing positive ranks')
    require(all(type(r) is int and 1 <= r <= count for r in ranks),
            'Invalid positive rank')
    require(len(set(ranks)) == len(ranks), 'Duplicate positive rank')
    top, scores = row['top100'], row['top100_scores']
    require(len(top) == len(scores) == 100 and len(set(top)) == 100,
            'Invalid top-100 size or duplicate IDs')
    require(all(type(i) is int and 0 <= i < count for i in top),
            'Invalid document ID')
    require(all(math.isfinite(s) for s in scores), 'Non-finite score')
    require(list(zip(top, scores)) == sorted(
        zip(top, scores), key=lambda x: (-x[1], x[0])),
        'Score or tie order changed')
    for doc, rank in zip(gold, ranks):
        require((rank <= 100 and top[rank - 1] == doc)
                or (rank > 100 and doc not in top), 'Gold/top-100 mismatch')
    ideal = math.fsum(1 / math.log2(r + 1)
                      for r in range(1, min(10, len(gold)) + 1))
    metrics = {
        'ndcg10': math.fsum(1 / math.log2(r + 1)
                            for r in ranks if r <= 10) / ideal,
        'recall100': sum(r <= 100 for r in ranks) / len(gold),
    }
    for key, value in metrics.items():
        require(math.isclose(row[key], value, abs_tol=1e-13, rel_tol=0),
                'Stored metric disagrees with positive ranks')


def summary(data):
    domains = {}
    for domain in DOMAINS:
        part = [r for r in data if r['domain'] == domain]
        require(bool(part), 'Empty domain')
        domains[domain] = {key: statistics.mean(r[key] for r in part)
                           for key in METRICS}
    return {'domains': domains, 'macro': {
        key: statistics.mean(domains[d][key] for d in DOMAINS)
        for key in METRICS}}


def harm(before, after):
    require(len(before) == len(after) > 0, 'Population length changed')
    ranks, dropped, entered = [], [], []
    for a, b in zip(before, after):
        require(a['query_id'] == b['query_id']
                and a['gold_ids'] == b['gold_ids'], 'Pair identity changed')
        for doc, old, new in zip(a['gold_ids'], a['gold_ranks'], b['gold_ranks']):
            ranks.append((old, new))
            event = {'query_id': a['query_id'], 'domain': a['domain'],
                     'gold_id': doc, 'before_rank': old, 'after_rank': new}
            if old <= 100 < new:
                dropped.append(event)
            if new <= 100 < old:
                entered.append(event)
    return {
        'queries': len(before), 'positive_pairs': len(ranks),
        'ndcg_improved': sum(b['ndcg10'] > a['ndcg10'] + 1e-13
                             for a, b in zip(before, after)),
        'ndcg_worsened': sum(b['ndcg10'] < a['ndcg10'] - 1e-13
                             for a, b in zip(before, after)),
        'positive_rank_improved': sum(b < a for a, b in ranks),
        'positive_rank_worsened': sum(b > a for a, b in ranks),
        'entered_top100': entered, 'dropped_top100': dropped,
    }


def paired_interval(differences, domains):
    # Average the three order seeds per query before resampling queries.
    # The interval is conditional on these seeds, not initialization variance.
    rng = np.random.default_rng(66066)
    means = []
    for domain in DOMAINS:
        values = np.asarray([v for v, d in zip(differences, domains)
                             if d == domain])
        require(len(values) > 0, 'Empty bootstrap domain')
        draws = rng.integers(len(values), size=(10000, len(values)))
        means.append(values[draws].mean(axis=1))
    return np.quantile(np.mean(means, axis=0), [.025, .975]).tolist()


def verify_artifacts(root, manifest):
    remote = read(manifest)
    expected = remote['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual == set(expected), 'Local/remote file inventory differs')
    for name, record in expected.items():
        path = root / name
        require(not path.is_symlink(), 'Unexpected symlink')
        require(path.stat().st_size == record['size']
                and sha(path) == record['sha256'], 'Copy mismatch: ' + name)
    frozen = [root / 'source-frozen.json',
              *sorted(root.glob('seed-*/epoch-*/*/complete.json'))]
    checked = set()
    for path in frozen:
        record = read(path)
        require(record.get('passed', True), 'Failed phase')
        for name, digest in record['files'].items():
            target = root / name
            require(sha(target) == digest, 'Frozen source/output mismatch: ' + name)
            checked.add(target.resolve())
    from scipy import sparse

    for seed in SEEDS:
        for epoch in EPOCHS:
            path = root / f'seed-{seed}/epoch-{epoch}/eval'
            result = read(path / 'results.json')
            matrices = {}
            for role, count in (('query', 1728), ('document', 59111)):
                matrix = sparse.load_npz(path / (role + '.npz'))
                require(matrix.shape[0] == count and matrix.has_canonical_format,
                        'Sparse matrix shape or canonical form changed')
                require(np.isfinite(matrix.data).all() and (matrix.data > 0).all(),
                        'Invalid nonnegative sparse readout')
                saved = result['counts'][role]
                size = sum(a.nbytes for a in (matrix.data, matrix.indices,
                                              matrix.indptr))
                require(matrix.nnz == saved['nnz'] and size == saved['csr_bytes'],
                        'Stored posting count or CSR bytes changed')
                matrices[role] = matrix
            query, document = matrices['query'], matrices['document']
            df = np.bincount(document.indices, minlength=document.shape[1])
            work = query.sign().astype(np.float64) @ df
            expected_work = result['counts']['semantic_posting_work_proxy']
            for key, actual_work in (('mean', work.mean()),
                                     ('p95', np.quantile(work, .95)),
                                     ('DEV_mean', work[1536:].mean())):
                require(abs(actual_work - expected_work[key]) < 1e-9,
                        'DF work proxy disagrees with saved posting matrices')
    return {'file_count': len(expected),
            'bytes': sum(x['size'] for x in expected.values()),
            'remote_manifest_sha256': sha(manifest),
            'frozen_source_and_phase_files': len(checked),
            'source_deleted': False, 'restore_tested': False}


def analyze(root, reference):
    queries = rows(reference / 'data/train.jsonl')
    queries += rows(reference / 'data/sealed.jsonl')
    labels = read(reference / 'data/positive-labels.json')
    require(len(queries) == len(labels) == 1728, 'Population size changed')
    curves = read(root / 'learning-curves.json')
    require(curves['all_seeds_finished']
            and curves['completed_evaluations'] == 9, 'Incomplete learning curve')
    data, results = {}, {}
    phases = sorted(root.glob('seed-*/epoch-*/*/exit.json'))
    require(len(phases) == 21, 'Expected 12 training and nine evaluation phases')
    for path in phases:
        exit_record = read(path)
        tracking = read(path.parent / 'clearml.json')
        require(exit_record['exit_code'] == 0
                and exit_record['owned_group_closed']
                and not exit_record['resource_limit']
                and exit_record['error'] is None, 'Unclean exit')
        require(tracking['actual_start'] and tracking['closed']
                and not tracking['post_hoc'], 'Invalid tracking lifecycle')
    initial_models = set()
    for seed in SEEDS:
        previous = None
        for epoch in range(1, 5):
            path = root / f'seed-{seed}/epoch-{epoch}/train'
            result = read(path / 'results.json')
            require(result['steps'] == 384
                    and result['cumulative_steps'] == epoch * 384
                    and result['unique_queries_this_epoch'] == 1536,
                    'Training exposure or update count changed')
            if previous is None:
                initial_models.add(result['initial_model_sha256'])
            else:
                require(result['moments_and_checkpoint_replay_exact']
                        and result['initial_model_sha256'] == previous['model_sha256']
                        and result['initial_optimizer_sha256']
                        == previous['optimizer_sha256'], 'Optimizer chain changed')
            previous = result
    require(len(initial_models) == 1, 'Training initialization changed')
    for seed in SEEDS:
        for epoch in EPOCHS:
            path = root / f'seed-{seed}/epoch-{epoch}/eval'
            value = rows(path / 'rankings.jsonl')
            require(len(value) == 1728, 'Missing query')
            result = read(path / 'results.json')
            for i, (row, query, gold) in enumerate(zip(value, queries, labels)):
                require(row['query_id'] == query['query_id']
                        and row['domain'] == query['subset']
                        and row['split'] == (SPLITS[0] if i < 1536 else SPLITS[1]),
                        'Query identity or split changed')
                audit_row(row, gold, 59111)
            for split in SPLITS:
                current = summary([r for r in value if r['split'] == split])
                old = result['populations'][split]['quality']
                for domain in DOMAINS:
                    for key in METRICS:
                        require(abs(current['domains'][domain][key]
                                    - old['domains'][domain][key]) < 1e-12,
                                'Domain metric mismatch')
                for key in METRICS:
                    require(abs(current['macro'][key] - old['macro'][key]) < 1e-12,
                            'Macro metric mismatch')
            data[seed, epoch], results[seed, epoch] = value, result
    aggregate = {}
    for epoch in EPOCHS:
        aggregate[epoch] = {}
        for split in SPLITS:
            summaries = [summary([r for r in data[seed, epoch]
                                  if r['split'] == split]) for seed in SEEDS]
            aggregate[epoch][split] = {
                'macro': {key: {'mean': statistics.mean(s['macro'][key]
                                                       for s in summaries),
                                'order_seed_std': statistics.stdev(
                                    s['macro'][key] for s in summaries)}
                          for key in METRICS},
                'domains': {domain: {key: statistics.mean(
                    s['domains'][domain][key] for s in summaries)
                    for key in METRICS} for domain in DOMAINS}}
    changes = []
    for first, last in ((1, 2), (1, 4), (2, 4)):
        for split in SPLITS:
            selected = [i for i, r in enumerate(data[SEEDS[0], first])
                        if r['split'] == split]
            diffs = [statistics.mean(data[s, last][i]['ndcg10']
                                     - data[s, first][i]['ndcg10'] for s in SEEDS)
                     for i in selected]
            domains = [data[SEEDS[0], first][i]['domain'] for i in selected]
            changes.append({'epochs': [first, last], 'split': split,
                            'ndcg10_mean_delta': statistics.mean(diffs),
                            'query_bootstrap_ci95_conditional_on_order_seeds':
                                paired_interval(diffs, domains)})
    harms = {}
    for seed in SEEDS:
        harms[seed] = {}
        for split in SPLITS:
            harms[seed][split] = {}
            for domain in ('ALL', *DOMAINS):
                a, b = [[r for r in data[seed, epoch] if r['split'] == split
                         and (domain == 'ALL' or r['domain'] == domain)]
                        for epoch in (1, 4)]
                harms[seed][split][domain] = harm(a, b)
    return {'stage': 'NG-0066/attempt-v2', 'all_15552_query_records_audited': True,
            'audit_scope': 'Recomputed metrics and top100/gold consistency; '
                           'full-score independent accumulation was run remotely.',
            'aggregate': aggregate, 'paired_changes': changes,
            'epoch1_to4_harm': harms, 'bootstrap_replicates': 10000,
            'bootstrap_seed': 66066, 'fresh_holdout_evaluated': False,
            'native_cost_evaluated': False, 'overall_goal_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.reference)
    if args.manifest:
        result['copy_verification'] = verify_artifacts(args.run, args.manifest)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('NG66_REVIEW_PASSED', args.output)


if __name__ == '__main__':
    main()
