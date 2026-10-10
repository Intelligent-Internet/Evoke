"""Independently replay the closed NG84 admission without importing its scorer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DOMAINS = ('fever', 'hotpotqa', 'nq')
INPUT_SHA = '8b4d9d147cfdf5ca20636249c837f65dbb2de4bbad2e0efaba4723c4b905328e'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_files(root, entries):
    for relative, expected in entries.items():
        path = root / relative
        check(path.resolve().is_relative_to(root.resolve()), 'unsafe artifact path')
        check(digest(path) == expected, 'artifact hash mismatch: ' + relative)


def replay(ids, scores, positives):
    check(len(ids) == len(scores) and len(ids) == len(set(ids)), 'score alignment')
    check(all(type(i) is int and i >= 0 for i in ids), 'invalid document ID')
    check(positives and set(positives) <= set(ids), 'missing positive')
    check(np.isfinite(scores).all(), 'nonfinite score')
    order = np.lexsort((np.asarray(ids), -np.asarray(scores, dtype=np.float64)))
    relevant = np.isin(np.asarray(ids)[order], positives)
    ranks = np.flatnonzero(relevant) + 1
    discounts = 1 / np.log2(np.arange(2, min(10, len(ids)) + 2))
    ideal = (1 / np.log2(np.arange(2, min(10, len(set(positives))) + 2))).sum()
    return dict(ndcg10=float(relevant[:10] @ discounts / ideal),
                recall10=float(relevant[:10].sum() / len(set(positives))),
                mrr=float(1 / ranks[0]), positive_ranks=ranks.tolist())


def decide(delta, interval):
    if delta >= .005 and interval[0] > 0:
        return 'admit_conditional_training'
    if interval[1] <= 0:
        return 'no_useful_signal_on_this_surface'
    return 'inconclusive_do_not_auto_launch'


def review(run):
    check(digest(run / 'inputs.json') == INPUT_SHA, 'wrong frozen admission')
    verify_files(run, load(run / 'inputs.json')['source'])
    out = run / 'teacher'
    complete = load(out / 'complete.json')
    check(complete['passed'] is True, 'execution incomplete')
    verify_files(out, complete['files'])
    terminal, tracking = load(out / 'exit.json'), load(out / 'clearml.json')
    check(terminal['exit_code'] == 0 and terminal['error'] is None
          and terminal['owned_group_closed'] is True, 'unclean terminal state')
    check(tracking['actual_start'] and tracking['closed']
          and tracking['outcome'] == 'passed', 'tracking not closed')
    expected = load(run / 'queries.json')
    rows = [json.loads(line) for line in (out / 'scores.jsonl').read_text().splitlines()]
    check(len(rows) == len(expected) == 384, 'incomplete query cohort')
    check(len({r['query_id'] for r in rows}) == 384, 'duplicate query')
    for row, original in zip(rows, expected, strict=True):
        check(row['query_id'] == original['query_id']
              and row['domain'] == original['subset'], 'changed query identity')
        for key in ('document_ids', 'positives', 'pplx_scores'):
            check(row[key] == original[key], 'changed input: ' + key)
        for label, key in (('pplx', 'pplx_scores'), ('rankt5', 'scores')):
            measured = replay(row['document_ids'], row[key], row['positives'])
            check(measured['positive_ranks'] == row[label]['positive_ranks'],
                  'positive rank mismatch')
            for metric in ('ndcg10', 'recall10', 'mrr'):
                check(abs(measured[metric] - row[label][metric]) < 1e-12,
                      'metric mismatch: ' + metric)
            row[label] = measured
    rng = np.random.default_rng(84001)
    domains, bootstrap = {}, []
    for domain in DOMAINS:
        cohort = [row for row in rows if row['domain'] == domain]
        check(len(cohort) == 128, 'unbalanced domain')
        values = np.asarray([[r['pplx']['ndcg10'], r['rankt5']['ndcg10']] for r in cohort])
        delta = values[:, 1] - values[:, 0]
        draws = rng.integers(0, 128, size=(10000, 128))
        bootstrap.append(delta[draws].mean(axis=1))
        domains[domain] = dict(
            pplx_ndcg10=float(values[:, 0].mean()),
            rankt5_ndcg10=float(values[:, 1].mean()), delta=float(delta.mean()),
            wins=int((delta > 0).sum()), ties=int((delta == 0).sum()),
            losses=int((delta < 0).sum()),
            all_positive_recall10={key: float(np.mean([r[key]['recall10'] for r in cohort]))
                                   for key in ('pplx', 'rankt5')},
            mrr={key: float(np.mean([r[key]['mrr'] for r in cohort]))
                 for key in ('pplx', 'rankt5')})
    delta = float(np.mean([d['delta'] for d in domains.values()]))
    interval = np.quantile(np.asarray(bootstrap).mean(axis=0), [.025, .975]).tolist()
    decision = decide(delta, interval)
    original_result = load(out / 'results.json')
    check(abs(delta - original_result['macro_delta']) < 1e-12, 'macro mismatch')
    check(np.allclose(interval, original_result['paired_interval'], atol=1e-12, rtol=0),
          'bootstrap mismatch')
    check(decision == original_result['decision'], 'gate mismatch')
    check(digest(out / 'scores.jsonl') == original_result['scores_sha256'], 'score hash')
    return dict(passed=True, input_sha256=INPUT_SHA,
                complete_sha256=digest(out / 'complete.json'),
                scores_sha256=digest(out / 'scores.jsonl'),
                reviewer_sha256=digest(Path(__file__)), queries=len(rows),
                pairs=sum(len(r['document_ids']) for r in rows),
                known_positive_memberships=sum(len(set(r['positives'])) for r in rows),
                domains=domains, macro_delta=delta, paired_interval=interval,
                decision=decision, independent_metric_implementation=True,
                independent_holdout=False, optimizer_updates=0,
                native_cost_evaluated=False, tracking=tracking, terminal=terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
