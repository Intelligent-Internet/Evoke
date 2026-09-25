"""Independent frozen-row/gradient alignment and denominator audit for NG73."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

import ng73_gradient_replay as io


def review(base, run):
    io.verify(base, run)
    output = run / 'audit'
    complete = io.read(output / 'complete.json')
    io.require(complete['passed'] is True, 'producer did not pass')
    inventory = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    io.require(inventory == set(complete['files']) | {'complete.json'}, 'output inventory changed')
    for name, digest in complete['files'].items():
        path = output / name
        io.require(path.resolve().is_relative_to(output.resolve()) and io.sha(path) == digest,
                   'output hash changed')
    exit_row, tracking = [io.read(output / n) for n in ('exit.json', 'clearml.json')]
    io.require(exit_row['exit_code'] == 0 and exit_row['error'] is None
               and exit_row['owned_group_closed'] is True
               and tracking['actual_start'] is True and tracking['closed'] is True,
               'incomplete execution lifecycle')
    parent = base / io.PARENT
    progress = io.rows(parent / 'train-D-96/progress.jsonl') + io.rows(
        parent / 'train-D-192/progress.jsonl')
    actual = {(u['step'], e['query_id']): e for u in progress for e in u['examples']}
    source = {ref: io.unique(io.rows(path)) for ref, path in (
        (0, base / 'NG-0071/pilot-step0-v1/witnesses.jsonl'),
        (96, parent / 'snapshot-96/witnesses.jsonl'))}
    margins = {(r['query_id'], r['positive_id']): r
               for r in io.rows(base / io.DIAG / 'diagnosis/margins.jsonl')
               if r['surface'] == 'TRAIN_PILOT'}
    records = io.rows(output / 'pairs.jsonl')
    expected = {(step, q, positive) for step, q in actual for identity, positive in margins if q == identity}
    observed = {(r['step'], r['query_id'], r['positive_id']) for r in records}
    io.require(expected == observed and len(records) == len(observed), 'positive/exposure coverage changed')
    config = io.read(parent / 'config.json')['ranking']
    max_error = 0.
    counters, totals, samples = defaultdict(Counter), defaultdict(Counter), defaultdict(dict)
    for row in records:
        q, p = row['query_id'], row['positive_id']
        example = actual[row['step'], q]
        reference = 0 if row['step'] <= 96 else 96
        witness = source[reference][q]
        old = margins[q, p]
        io.require(row['reference_update'] == reference and row['domain'] == witness['domain']
                   and row['origin'] == old['origin'], 'lineage changed')
        for field in ('competitor_ids', 'before_margin', 'after_margin'):
            io.require(row[field] == old[field], 'frozen diagnostic row changed')
        positions = {doc: i for i, doc in enumerate(witness['pool'])}
        # Targets/coefficients are independently rebuilt by the existing NumPy
        # reference through prepare_pairs, rather than trusted from NG73 rows.
        pairs = io.diagnostics.training.prepare_pairs(witness, config)
        lookup = {tuple(pair): (target, coefficient) for pair, target, coefficient in zip(
            pairs['indices'], pairs['targets'], pairs['coefficients'], strict=True)}
        own = np.zeros(len(witness['pool']))
        for (i, j), (target, coefficient) in lookup.items():
            z = (example['scores'][i] - example['scores'][j]) / config['student_temperature']
            own[i] += coefficient * (np.exp(-np.logaddexp(0., -z)) - target) / config['student_temperature']
            own[j] -= coefficient * (np.exp(-np.logaddexp(0., -z)) - target) / config['student_temperature']
        np.testing.assert_allclose(own / 4, example['score_gradient'], rtol=2e-5, atol=2e-7)
        for j, rival in enumerate(row['competitor_ids']):
            i, n = positions[p], positions.get(rival)
            pair = lookup.get((i, n))
            if pair:
                target, coefficient = pair
                z = (example['scores'][i] - example['scores'][n]) / config['student_temperature']
                derivative = coefficient * (np.exp(-np.logaddexp(0., -z)) - target) / config['student_temperature']
                np.testing.assert_allclose([row['target'][j], row['coefficient'][j], row['own_derivative'][j]],
                                           [target, coefficient, derivative], rtol=0, atol=1e-14)
                max_error = max(max_error, abs(row['own_derivative'][j] - derivative))
                state = 'shrink' if derivative > 1e-12 else 'expand' if derivative < -1e-12 else 'stationary'
            else:
                state = 'absent' if n is None else 'ineligible'
                io.require(all(row[f][j] is None for f in ('target', 'coefficient', 'own_derivative')),
                           'invented direct gradient')
                derivative = 0.
            group = {(True, True): 'retained', (True, False): 'lost',
                     (False, True): 'gained', (False, False): 'stayed_behind'}[
                         old['baseline_ahead'][j], old['final_ahead'][j]]
            io.require(row['outcome'][j] == group and row['state'][j] == state, 'pair state changed')
            pressure = None if n is None else own[i] - own[n]
            if pressure is None:
                io.require(row['aggregate_margin_pressure'][j] is None, 'invented unseen rival pressure')
            else:
                np.testing.assert_allclose(row['aggregate_margin_pressure'][j], pressure, rtol=0, atol=1e-13)
            for origin in ('all', row['origin']):
                for g in ('all', group):
                    key = row['domain'], origin, g
                    counters[key][state] += 1
                    if state in ('shrink', 'expand'):
                        totals[key][state] += abs(derivative)
        for origin in ('all', row['origin']):
            for group in ('all', *io.GROUPS):
                key = row['domain'], origin, group
                counts = Counter(s for s, g in zip(row['state'], row['outcome'], strict=True)
                                 if group == 'all' or g == group)
                samples[key].setdefault(q, []).append({s: counts[s] / len(row['competitor_ids'])
                                                     for s in io.STATES})
    result = io.read(output / 'results.json')
    for key, query_rows in samples.items():
        domain, origin, group = key
        reported = result['summary'][domain][origin]['groups'][group]
        io.require(reported['pair_exposures'] == sum(counters[key].values()), 'pair exposure count differs')
        for state in io.STATES:
            io.require(reported['state_counts'][state] == counters[key][state], 'state count differs')
            average = sum(sum(r[state] for r in parts) / len(parts)
                          for parts in query_rows.values()) / len(query_rows)
            np.testing.assert_allclose(reported['full_rival_query_balanced'][state], average, rtol=0, atol=1e-14)
        for state in ('expand', 'shrink'):
            np.testing.assert_allclose(reported['absolute_pair_derivative_mass'][state],
                                       totals[key][state], rtol=1e-12, atol=1e-14)
    io.require(all(result[k] is False for k in ('new_model_inference', 'dev_scored', 'locked_test_scored',
                                               'parameter_update_causality_established')),
               'scope claim changed')
    io.verify(base, run)
    for name, digest in complete['files'].items():
        io.require(io.sha(output / name) == digest, 'output changed during review')
    return dict(passed=True, positive_exposures=len(records), query_exposures=len(actual),
                max_pair_derivative_error=max_error, input_sha256=io.sha(run / 'inputs.json'),
                producer_complete_sha256=io.sha(output / 'complete.json'),
                independent_row_and_count_review=True, full_corpus_reranked=False,
                parameter_update_causality_established=False, locked_test_scored=False)


if __name__ == '__main__':
    print(json.dumps(review(Path(sys.argv[1]), Path(sys.argv[2])), indent=2, allow_nan=False))
