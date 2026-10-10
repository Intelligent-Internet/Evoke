"""Fixed NG87 endpoint review; candidate evidence never qualifies deployment."""

from collections import defaultdict
import time

import numpy as np

from ng85_data import read, require, rows, sha, write
from ng85_run import closed


DOMAINS = ('fever', 'hotpotqa', 'nq', 'fiqa')


def paired(left, right, metric, seed):
    require(left.keys() == right.keys(), 'paired query identity mismatch')
    rng = np.random.default_rng(seed)
    samples, mean = np.zeros(5000), 0.
    for domain in DOMAINS:
        ids = sorted(i for i in left if left[i]['subset'] == domain)
        require(len(ids) == 512 and all(right[i]['subset'] == domain for i in ids),
                'paired domain mismatch')
        require(all(left[i]['document_ids'] == right[i]['document_ids'] for i in ids),
                'paired candidate mismatch')
        delta = np.array([right[i][metric] - left[i][metric] for i in ids])
        require(np.isfinite(delta).all(), 'nonfinite paired metric')
        mean += float(delta.mean()) / 4
        samples += rng.choice(delta, size=(5000, len(delta)), replace=True).mean(1) / 4
    return dict(delta=mean, descriptive_interval_95=np.quantile(samples, [.025, .975]).tolist())


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        label = 'base' if row['checkpoint'] == 'base' else row['arm'] + '-' + row['checkpoint']
        grouped[(row['split'], label)].append(row)
    labels = {'base', *(arm + '-' + str(step) for arm in 'AB' for step in (2048, 4096, 8192))}
    require(set(grouped) == {(split, label) for split in ('fit', 'validation') for label in labels},
            'missing fixed profile label')
    metrics = {}
    for (split, label), values in grouped.items():
        expected = 128 if split == 'fit' else 2048 if label in ('base', 'A-8192', 'B-8192') else 512
        ids = [v['query_index'] for v in values]
        require(len(ids) == len(set(ids)) == expected and all(
            sum(v['subset'] == d for v in values) == expected // 4 for d in DOMAINS),
            'missing, duplicate or unbalanced profile row')
        for v in values:
            require(all(np.isfinite(v[k]) and 0 <= v[k] <= 1 for k in (
                'candidate_ndcg10', 'candidate_recall10')), 'invalid quality metric')
            require(v['valid'] or v['candidate_ndcg10'] == v['candidate_recall10'] == 0.,
                    'invalid queries must remain in denominator')
        valid = [v for v in values if v['valid']]
        item = dict(queries=expected, invalid=expected - len(valid),
            ndcg10=float(np.mean([v['candidate_ndcg10'] for v in values])),
            recall10=float(np.mean([v['candidate_recall10'] for v in values])),
            document_nnz=float(np.mean([v['document_nnz_mean'] for v in valid])) if valid else None,
            query_nnz=float(np.mean([v['query_nnz'] for v in valid])) if valid else None,
            zero_rms_mass=float(np.mean([v['query_raw'][0]['unsupported_mass'] /
                v['query_raw'][0]['raw_sum'] for v in valid])) if valid else None,
            by_domain={d: dict(ndcg10=float(np.mean([
                v['candidate_ndcg10'] for v in values if v['subset'] == d])),
                recall10=float(np.mean([v['candidate_recall10'] for v in values
                                       if v['subset'] == d]))) for d in DOMAINS})
        metrics.setdefault(split, {})[label] = item
    endpoint = {label: {r['query_index']: r for r in grouped['validation', label]}
                for label in ('base', 'A-8192', 'B-8192')}
    for name in ('pplx', 'bm25'):
        endpoint[name] = {i: dict(row, **row[name]) for i, row in endpoint['base'].items()}
    comparisons = {}
    for left, right in (('base', 'A-8192'), ('base', 'B-8192'), ('A-8192', 'B-8192'),
                        ('pplx', 'A-8192'), ('pplx', 'B-8192')):
        comparisons[right + '_minus_' + left] = {key: paired(
            endpoint[left], endpoint[right], key, 87001 + n)
            for n, key in enumerate(('candidate_ndcg10', 'candidate_recall10'))}
    decisions = {}
    for arm in 'AB':
        key = arm + '-8192'
        comp = comparisons[key + '_minus_base']
        quality = (metrics['validation'][key]['invalid'] == 0
            and comp['candidate_ndcg10']['delta'] >= .005
            and comp['candidate_ndcg10']['descriptive_interval_95'][0] > 0
            and comp['candidate_recall10']['descriptive_interval_95'][0] >= -.002)
        nnz = metrics['validation'][key]['document_nnz']
        ratio = None if nnz is None else nnz / metrics['validation']['base']['document_nnz']
        decisions[arm] = dict(candidate_quality_signal=quality,
            candidate_nnz_ratio=ratio, candidate_cost_screen_pass=ratio is not None and ratio <= 1.5,
            full_corpus_and_second_seed_required=True, automatic_promotion=False)
    return dict(passed=True, passed_means='complete fixed candidate review, not model qualification',
        primary_step=8192, metrics=metrics, comparisons=comparisons, decisions=decisions,
        references={name: dict(ndcg10=float(np.mean([v['candidate_ndcg10'] for v in group.values()])),
            recall10=float(np.mean([v['candidate_recall10'] for v in group.values()])))
            for name, group in endpoint.items() if name in ('pplx', 'bm25')},
        independent_generalization=False, full_corpus_evaluation=False,
        native_cost_evaluated=False, production_qualified=False)


def review(run):
    manifest = read(run / 'inputs.json')
    require(manifest['protocol'] == 'NG87_clean_base_fullstd_AB_v1', 'NG87 protocol required')
    for name, digest in manifest['source'].items():
        require(sha(run / name) == digest, 'changed review source')
    for phase in manifest['phases']:
        result = closed(run, phase)
        if phase.startswith('train-'):
            require(result['updates'] == 8192 and not result['discard_updates'], 'incomplete training')
    records = []
    for arm in ('base', 'A', 'B'):
        data = rows(run / f'profile-{arm}/queries.jsonl')
        labels = ['base'] if arm == 'base' else ['2048', '4096', '8192']
        require(set(r['checkpoint'] for r in data) == set(labels)
                and all(r['arm'] == arm for r in data), 'profile label mismatch')
        for label in labels:
            expected = manifest['endpoint_panel'] if label in ('base', '8192') else manifest['monitor_panel']
            require({r['query_index'] for r in data if r['checkpoint'] == label} == set(expected),
                    'profile panel mismatch')
        records.extend(data)
    result = summarize(records)
    folder = run / 'review'
    folder.mkdir()
    write(folder / 'inputs.json', dict(inputs_sha256=sha(run / 'inputs.json'),
        receipts={p: sha(run / p / 'complete.json') for p in manifest['phases']}))
    write(folder / 'results.json', result)
    write(folder / 'exit.json', dict(passed=True, exit_code=0, finished_unix=time.time()))
    write(folder / 'complete.json', dict(passed=True,
        files={p.name: sha(p) for p in folder.iterdir() if p.is_file()}))
