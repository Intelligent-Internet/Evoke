"""Read sealed NG86 phases and report fixed comparisons, without new inference."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from ng85_data import read, require, rows, sha, write
from ng85_run import closed


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        label = row['checkpoint']
        if label not in ('base', 'start'):
            label = row['arm'] + '-' + label
        grouped[(row['split'], label)].append(row)
    output = {'fit': {}, 'validation': {}}
    for (split, label), values in grouped.items():
        require(split in output, 'unexpected evaluation split')
        expected = 128 if split == 'fit' else 512
        ids = [v['query_index'] for v in values]
        require(len(ids) == len(set(ids)) == expected, 'missing or duplicate profile row')
        domains = sorted(set(v['subset'] for v in values))
        require(domains == ['fever', 'fiqa', 'hotpotqa', 'nq']
                and all(sum(v['subset'] == d for v in values) == expected // 4 for d in domains),
                'unbalanced panel')
        for v in values:
            if not v['valid']:
                require(v['candidate_ndcg10'] == 0. and v['candidate_recall10'] == 0.,
                        'invalid queries must remain in the quality denominator')
        valid = [v for v in values if v['valid']]
        mass = [v['query_raw'][0]['unsupported_mass'] / v['query_raw'][0]['raw_sum']
                for v in valid if v['query_raw'][0]['raw_sum'] > 0]
        output[split][label] = dict(queries=expected, invalid=expected - len(valid),
            candidate_ndcg10=float(np.mean([v['candidate_ndcg10'] for v in values])),
            candidate_recall10=float(np.mean([v['candidate_recall10'] for v in values])),
            by_domain={d: dict(
                ndcg10=float(np.mean([v['candidate_ndcg10'] for v in values if v['subset'] == d])),
                recall10=float(np.mean([v['candidate_recall10'] for v in values if v['subset'] == d])))
                for d in domains},
            valid_query_document_nnz=float(np.mean([v['document_nnz_mean'] for v in valid]))
                if valid else None,
            valid_query_zero_rms_mass=float(np.mean(mass)) if mass else None,
            cost_is_candidate_proxy_only=True)
    pairs = {}
    for step in (2560, 2816, 3072):
        a = grouped.get(('validation', f'control-{step}'))
        b = grouped.get(('validation', f'fullstd-{step}'))
        if a is None or b is None:
            pairs[str(step)] = dict(available=False, reason='fixed checkpoint not reached by both arms')
            continue
        left, right = ({v['query_index']: v for v in group} for group in (a, b))
        require(left.keys() == right.keys(), 'paired query identity mismatch')
        rng = np.random.default_rng(86001 + step)
        samples, mean = np.zeros(5000), 0.
        for domain in ('fever', 'hotpotqa', 'nq', 'fiqa'):
            ids = sorted(i for i, v in left.items() if v['subset'] == domain)
            require(all(right[i]['subset'] == domain for i in ids), 'paired domain mismatch')
            delta = np.array([right[i]['candidate_ndcg10'] - left[i]['candidate_ndcg10'] for i in ids])
            mean += float(delta.mean()) / 4
            samples += rng.choice(delta, size=(5000, len(delta)), replace=True).mean(1) / 4
        pairs[str(step)] = dict(available=True, ndcg_delta=mean,
            descriptive_paired_interval_95=np.quantile(samples, [.025, .975]).tolist(),
            independent_generalization=False)
    return dict(metrics=output, paired_validation=pairs, primary_step=2816,
                no_posthoc_checkpoint_selection=True, production_qualified=False)


def run(args):
    manifest = read(args.out / 'inputs.json')
    require(manifest['protocol'] == 'NG86_prospective_std_gradient_v1'
            and not manifest['locked_test_access'], 'unexpected protocol')
    expected_phases = [p + '-' + a for a in ('control', 'fullstd') for p in ('train', 'profile')]
    deadline = time.monotonic() + args.wait_seconds
    while True:
        present = [p for p in expected_phases if (args.out / p / 'complete.json').exists()]
        for phase in present:
            require(read(args.out / phase / 'complete.json')['passed'], 'execution failed: ' + phase)
        if len(present) == len(expected_phases):
            break
        require(time.monotonic() < deadline, 'bounded review wait expired; no result fabricated')
        time.sleep(min(120, max(0., deadline - time.monotonic())))
    receipts = {p: closed(args.out, p) for p in expected_phases}
    for name, digest in manifest['source'].items():
        require(sha(args.out / name) == digest, 'source changed before review')
    records = []
    for arm in ('control', 'fullstd'):
        source = rows(args.out / f'profile-{arm}/queries.jsonl')
        labels = set(str(x) for x in receipts['profile-' + arm]['labels'])
        require(set(r['checkpoint'] for r in source) == labels
                and all(r['arm'] == arm for r in source), 'profile labels changed')
        for label in labels:
            require({r['query_index'] for r in source if r['checkpoint'] == label}
                    == set(manifest['panel']), 'profile panel changed')
        records.extend(source)
    result = summarize(records)
    result['training'] = {a: receipts['train-' + a] for a in ('control', 'fullstd')}
    result['passed'] = True
    result['passed_means'] = 'sealed observation reviewed; not model qualification'
    folder = args.out / 'review'
    folder.mkdir()
    write(folder / 'inputs.json', dict(inputs_sha256=sha(args.out / 'inputs.json'),
        reviewer_sha256=sha(Path(__file__)),
        receipts={p: sha(args.out / p / 'complete.json') for p in expected_phases}))
    write(folder / 'results.json', result)
    write(folder / 'exit.json', dict(exit_code=0, passed=True, finished_unix=time.time()))
    write(folder / 'complete.json', dict(passed=True,
        files={p.name: sha(p) for p in folder.iterdir() if p.is_file()}))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args()
    require(0 <= args.wait_seconds <= 9000, 'review wait cap exceeded')
    run(args)


if __name__ == '__main__':
    main()
