"""Independent, standard-library-only reduction of sealed NG75 evidence.

This reviewer does not import the generator's math helpers or encode text.
It checks recorded finite differences and projections, not the raw Jacobians.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


CASES = ('probe-D-1', 'probe-U-1', 'probe-D-97', 'probe-U-97')
MANIFEST = '93087ed7480f320b762d296f9f1e36cf7fa891546cab8b36a6181c14f87aa822'
INVENTORY = '488c70dbe85742768f63dd223f6e0cf72189d35c6badf2f956fc2e096f9ada6d'
ORIGINS = ('all', 'original_positive', 'added_vs_original')
DIRECTIONS = ('expand', 'shrink', 'stationary')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root, files, *, exact=False, exclude=()):
    if exact:
        actual = {str(p.relative_to(root)) for p in root.rglob('*')
                  if p.is_file() and str(p.relative_to(root)) not in exclude}
        require(actual == set(files), 'file inventory changed')
    for name, digest in files.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and sha(path) == digest,
                'file content changed: ' + name)


def compare(actual, expected, *, rtol=0., atol=1e-12):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'mapping changed')
        return max((compare(actual[k], v, rtol=rtol, atol=atol) for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'list shape changed')
        return max((compare(a, b, rtol=rtol, atol=atol)
                    for a, b in zip(actual, expected, strict=True)), default=0.)
    if type(expected) in (float, int):
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected),
                'invalid numeric evidence')
        error = abs(actual - expected)
        require(error <= atol + rtol * abs(expected), 'numeric reduction differs')
        return error
    require(type(actual) is type(expected) and actual == expected, 'categorical evidence changed')
    return 0.


def sign(value, floor):
    require(math.isfinite(value) and math.isfinite(floor) and floor >= 0, 'invalid direction')
    if abs(value) <= floor:
        return 'stationary'
    return 'expand' if value > 0 else 'shrink'


def reduce_group(rows):
    per_query = defaultdict(list)
    for r in rows:
        per_query[r['query_id']].append(r['after_margin'] - r['before_margin'])
    return dict(
        pairs=len(rows), queries=len(per_query),
        positives=len({(r['query_id'], r['positive_id']) for r in rows}),
        directions={k: sum(r['actual_direction'] == k for r in rows) for k in DIRECTIONS},
        predicted_sign_matches=sum(r['actual_direction'] == r['predicted_direction'] for r in rows),
        mean_actual_change=fmean(r['after_margin'] - r['before_margin'] for r in rows) if rows else None,
        mean_abs_residual=fmean(abs(r['taylor_residual']) for r in rows) if rows else None,
        query_balanced_mean_change=fmean(fmean(per_query[q]) for q in sorted(per_query)) if rows else None)


def reduce_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['surface'] + '/' + row['domain']].append(row)
    return {key: {origin: reduce_group([r for r in values if origin == 'all' or r['origin'] == origin])
                  for origin in ORIGINS} for key, values in sorted(groups.items())}


def audit_row(row, spec, case, trace):
    compare({k: row[k] for k in spec}, spec, atol=0.)
    require(row['split'] == 'TRAIN' and row['origin'] in ORIGINS[1:], 'non-TRAIN or unresolved pair')
    require(row['no_update_error'] == 0. and row['rivals_are_judged_negative'] is False,
            'no-update or rival label boundary changed')
    delta = row['after_margin'] - row['before_margin']
    floor = max(1e-7, 10 * row['no_update_error'])
    expected = dict(actual_change=delta, taylor_residual=delta - row['predicted'],
                    predicted_change=row['predicted'], numerical_floor=floor,
                    actual_direction=sign(delta, floor), predicted_direction=sign(row['predicted'], floor))
    error = compare({k: row[k] for k in expected}, expected)
    compare(row['before_margin'], row['semantic_margin'] + row['lexical_margin'])
    require(set(row['contributions']) == {'query', 'document'}, 'missing gradient role')
    for pieces in row['contributions'].values():
        require(set(pieces) == {'trunk', 'head'}, 'missing parameter group')
    compare(sum(v for pieces in row['contributions'].values() for v in pieces.values()),
            row['predicted'], rtol=2e-5, atol=2e-7)
    for key in ('gradient_chain_max_abs', 'gradient_chain_relative_l2'):
        require(math.isfinite(row[key]) and row[key] >= 0, 'invalid recorded chain check')
    pressure = None
    if row['query_id'] in case['batch_ids']:
        j = case['batch_ids'].index(row['query_id'])
        pool = case['witnesses'][j]['pool']
        if row['positive_id'] in pool and row['rival_id'] in pool:
            gradient = trace['examples'][j]['score_gradient']
            require(len(gradient) == len(pool), 'score gradient coverage differs')
            pressure = gradient[pool.index(row['rival_id'])] - gradient[pool.index(row['positive_id'])]
    compare(row['same_batch_score_pressure'], pressure)
    for key in ('teacher_original', 'teacher_reference'):
        item = row[key]
        if item['status'] == 'unobserved':
            require(item['margin'] is None, 'unobserved teacher has a margin')
        else:
            require(math.isfinite(item['margin']), 'invalid teacher margin')
            label = 'agrees' if item['margin'] > 0 else 'opposes' if item['margin'] < 0 else 'tie'
            require(item['status'] == label, 'teacher status changed')
    return error


def characterize(rows):
    """Descriptive strata, not new selection gates or a quality evaluation."""
    negative = [r for r in rows if r['actual_direction'] == 'shrink']
    pressures = {}
    for label in (*DIRECTIONS, 'unobserved'):
        group = [r for r in rows if (sign(r['same_batch_score_pressure'], 1e-12)
                 if r['same_batch_score_pressure'] is not None else 'unobserved') == label]
        pressures[label] = reduce_group(group)
    teacher = {s: reduce_group([r for r in rows if r['teacher_original']['status'] == s])
               for s in ('agrees', 'opposes', 'tie', 'unobserved')}
    absolute = sum(abs(r['actual_change']) for r in rows)
    return dict(
        summary=reduce_group(rows),
        residual_l1_over_change_l1=sum(abs(r['taylor_residual']) for r in rows) / absolute if absolute else None,
        max_abs_residual=max((abs(r['taylor_residual']) for r in rows), default=0.),
        score_pressure=pressures, teacher_original=teacher,
        pre_update_positive_margin=reduce_group([r for r in rows if r['before_margin'] > r['numerical_floor']]),
        zero_crossings=dict(positive_to_negative=sum(r['before_margin'] > r['numerical_floor']
                           and r['after_margin'] < -r['numerical_floor'] for r in rows),
                           negative_to_positive=sum(r['before_margin'] < -r['numerical_floor']
                           and r['after_margin'] > r['numerical_floor'] for r in rows)),
        mean_projections_on_shrinking_pairs={role: {
            group: fmean(r['contributions'][role][group] for r in negative) if negative else None
            for group in ('trunk', 'head')} for role in ('query', 'document')},
        shrinking_queries=len({r['query_id'] for r in negative}))


def phase_result(path):
    receipt = read(path / 'complete.json')
    require(receipt['passed'] is True, 'phase not passed')
    verify(path, receipt['files'], exact=True, exclude=('complete.json',))
    exit_record, tracking = read(path / 'exit.json'), read(path / 'clearml.json')
    require(exit_record['exit_code'] == 0 and exit_record['error'] is None
            and exit_record['owned_group_closed'] is True, 'phase not closed')
    require(exit_record['elapsed_seconds'] < 1800 and exit_record['peak_tree_rss_bytes'] <= 16 * 2 ** 30
            and exit_record['scientific_training_bounds_relaxed'] is False, 'phase resource boundary failed')
    require(tracking['actual_start'] is True and tracking['closed'] is True
            and tracking['outcome'] == 'passed', 'tracking not closed')
    return read(path / 'results.json')


def review(base, run, inventory_path):
    require(sha(run / 'inputs.json') == MANIFEST and sha(inventory_path) == INVENTORY,
            'wrong frozen run or terminal inventory')
    manifest, inventory = read(run / 'inputs.json'), read(inventory_path)
    verify(base, manifest['dependencies'])
    verify(run, manifest['source'])
    verify(run, inventory['files'], exact=True)
    control = read(run / 'controller-exit.json')
    require(control['status'] == 'all_phases_complete' and control['completed'] == [*CASES, 'review'],
            'controller incomplete')
    data = read(run / 'probe-data.json')
    summaries, detailed, max_error, total = {}, {}, 0., 0
    for name in CASES:
        case, result = data['cases'][name], phase_result(run / name)
        require(result['disposable_optimizer_updates'] == 1 and result['scientific_training_updates'] == 0
                and result['locked_test_scored'] is False and result['quality_evaluation'] is False,
                'diagnostic boundary changed')
        trace = read(run / name / 'replay.json')['actual']
        for key in ('gradient_before_clip', 'update_l2'):
            compare(trace[key], case['expected'][key], rtol=2e-5, atol=2e-7)
        require(len(trace['examples']) == len(case['expected']['examples']) == 4, 'replay exposure mismatch')
        for a, b in zip(trace['examples'], case['expected']['examples'], strict=True):
            for key in ('scores', 'score_gradient', 'loss', 'documents', 'objective',
                        'eligible_pairs', 'supervised_positives', 'vjp_replay_exact'):
                compare(a[key], b[key], rtol=2e-5, atol=2e-7)
        rows = [json.loads(s) for s in (run / name / 'margins.jsonl').read_text().splitlines()]
        require(len(rows) == len(case['probes']) == result['probes'], 'missing pair rows')
        require(len({(r['query_id'], r['positive_id'], r['rival_id']) for r in rows}) == len(rows),
                'duplicate pair rows')
        for row, spec in zip(rows, case['probes'], strict=True):
            require(row['case'] == name, 'case attribution changed')
            max_error = max(max_error, audit_row(row, spec, case, trace))
        summary = reduce_summary(rows)
        max_error = max(max_error, compare(summary, result['summary']))
        summaries[name] = summary
        detailed[name] = {key: characterize([r for r in rows if r['surface'] + '/' + r['domain'] == key])
                          for key in summary}
        total += len(rows)
    terminal = phase_result(run / 'review')
    max_error = max(max_error, compare(summaries, terminal['cases']))
    verify(base, manifest['dependencies'])
    verify(run, inventory['files'], exact=True)
    return dict(passed=True, rows=total, files=len(inventory['files']), source_sha256=sha(Path(__file__)),
                manifest_sha256=MANIFEST, terminal_inventory_sha256=INVENTORY,
                independent_reduction_max_error=max_error, cases=detailed,
                inputs_and_outputs_unchanged=True, independent_reduction_implementation=True,
                jacobians_or_model_inference_rerun=False, locked_test_scored=False,
                quality_evaluation=False, native_cost_evaluated=False, overall_goal_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.run.resolve()),
            'output must be new and outside the frozen run')
    result = review(args.research_root, args.run, args.inventory)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('NG75_INDEPENDENT_REVIEW', result['rows'], result['independent_reduction_max_error'])


if __name__ == '__main__':
    main()
