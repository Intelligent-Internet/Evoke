#!/usr/bin/env python3
"""Read-only replay of a sealed NG71 full TRAIN step-zero snapshot.

The caller supplies the independently observed remote complete SHA. Outputs
must be outside the frozen attempt. No encoder inference or updates occur.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

# Frozen source imports must not create bytecode inside sealed evidence.
sys.dont_write_bytecode = True

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(a, b):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('replay field inventory changed')
        return max((compare(a[key], b[key]) for key in a), default=0.)
    if isinstance(a, list):
        if len(a) != len(b):
            raise ValueError('replay vector length changed')
        return max((compare(x, y) for x, y in zip(a, b, strict=True)), default=0.)
    if isinstance(a, float):
        error = abs(a - b)
        if not np.isfinite(a) or not np.isfinite(b) or error > 1e-12:
            raise ValueError('independent replay differs by more than 1e-12')
        return error
    if a != b:
        raise ValueError('replay identity or count changed')
    return 0.


def review(base, root, expected):
    if sha(root / 'complete.json') != expected:
        raise ValueError('snapshot does not match independent remote completion')
    complete = read(root / 'complete.json')
    files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if not complete['passed'] or files != set(complete['files']) | {'complete.json'}:
        raise ValueError('snapshot file inventory changed or completion failed')
    for name, digest in complete['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError('snapshot output changed: ' + name)
    for name, digest in read(root / 'dependencies.json').items():
        if sha(base / name) != digest:
            raise ValueError('frozen dependency changed: ' + name)
    sys.path.insert(0, str(root))
    import ng71_data as data
    import ng71_diagnostics as diagnostics

    config = read(root / 'config.json')
    prepared = data.audit(base, config)
    order = read(root / 'training-order.json')
    rows = [json.loads(line) for line in (root / 'witnesses.jsonl').read_text().splitlines()]
    if (order != prepared['query_order'] or [r['query_id'] for r in rows]
            != prepared['selection']['pilot']):
        raise ValueError('fixed TRAIN selection or exposure order changed')
    results, exited, tracking = [read(root / name) for name in (
        'results.json', 'exit.json', 'clearml.json')]
    if (exited['exit_code'] != 0 or exited['error'] is not None
            or exited['owned_group_closed'] is not True
            or tracking['closed'] is not True or tracking['actual_start'] is not True
            or not results['full_pilot_manifest_prepared']
            or not results['eligibility_gate_passed']
            or results['actual_optimizer_updates'] != 0
            or results['reference_update'] != 0 or results['locked_test_access']):
        raise ValueError('step-zero preparation receipts failed')
    error = max(compare(diagnostics.diagnose(row, config), row['arm_score_diagnostics'])
                for row in rows)
    batches = diagnostics.batch_audit(rows, order, config)
    compare(batches, results['fixed_batch_audit'])
    previous = {r['query_id']: r for r in (json.loads(line) for line in
        (base / 'NG-0071/global-preflight-v1/witnesses.jsonl').read_text().splitlines())}
    for row in rows:
        if row['query_id'] in previous:
            for field in ('pool', 'positive_mask', 'teacher_scores', 'global_ranks',
                          'sources', 'document_text_sha256'):
                compare(row[field], previous[row['query_id']][field])
    if not set(previous) <= {r['query_id'] for r in rows}:
        raise ValueError('previous canary not retained in the full pilot')
    report = {}
    for domain in config['pilot']['domains']:
        subset = [r for r in rows if r['domain'] == domain]
        diag = results['score_space_diagnostics'][domain]
        source = diag['D']['sources']
        pair_mass = sum(v['abs_pair_derivative_mass'] for v in source.values())
        boundary_mass = sum(source[s]['abs_pair_derivative_mass']
                            for s in ('hybrid_head', 'hybrid_boundary'))
        sizes = [len(r['pool']) for r in subset]
        head = results['summaries'][domain]['sources']['hybrid_head']
        report[domain] = dict(
            queries=len(subset), positives=results['summaries'][domain]['positives'],
            supervised=results['summaries'][domain]['supervised_positives'],
            outside100_positive_pairs=sum(rank > 100 for row in subset
                for rank, positive in zip(row['global_ranks'], row['positive_mask']) if positive),
            truncated_positive_pairs=sum(v['truncated'] for row in subset
                                          for v in row['positive_visibility']),
            pool_range=[min(sizes), max(sizes)],
            D_head_boundary_pair_derivative_fraction=boundary_mass / pair_mass,
            D_vs_C_candidate_gradient_l1=diag['D']['score_gradient_l1'] / diag['C']['score_gradient_l1'],
            D_vs_B_candidate_gradient_l1=diag['D']['score_gradient_l1'] / diag['B']['score_gradient_l1'],
            head_cross100_masked_priority_fraction=(
                head['masked_cross_top100_mass'] / head['cross_top100_mass']
                if head['cross_top100_mass'] else None),
            positive_promote_reduce={arm: [diag[arm]['positive_score_promote_count'],
                                          diag[arm]['positive_score_reduce_count']]
                                     for arm in ('A', 'B', 'C', 'D')})
    return dict(
        passed=True, reviewer_source_sha256=sha(Path(__file__)),
        complete_sha256=expected, source_sha256=sha(root / 'source-frozen.json'),
        copied_files=len(files), total_bytes=sum((root / name).stat().st_size for name in files),
        exact_inventory_and_all_output_sha_verified=True, all_dependency_sha_verified=True,
        independent_frozen_selection_and_order_verified=True,
        prior_96_canary_manifests_identical=True,
        all_384_four_arm_score_derivatives_recomputed=True,
        max_cross_host_replay_error=error, fixed_batch_audit=batches,
        exit=exited, tracking=tracking, TRAIN_diagnostic=report,
        remote_source_deleted=False, scientific_training_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--expected-complete', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.snapshot.resolve()) or args.output.exists():
        raise ValueError('review must be a new file outside frozen evidence')
    result = review(args.research_root.resolve(), args.snapshot.resolve(), args.expected_complete)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(args.output), sha256=sha(args.output), result=result)))


if __name__ == '__main__':
    main()
