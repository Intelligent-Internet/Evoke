"""Read-only NG82 paired review; never select another diagnostic profile."""

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

import ng82_diagnostics as study


def recompute(record):
    ranks, gold, top = record['gold_ranks'], record['gold_ids'], record['top100']
    study.require(len(ranks) == len(gold) > 0 and len(set(gold)) == len(gold)
                  and len(set(ranks)) == len(ranks) and len(top) == len(set(top)) == 100,
                  'rank inventory')
    for doc, rank in zip(gold, ranks, strict=True):
        study.require(type(rank) is int and 0 < rank <= 233009, 'rank range')
        study.require((top[rank - 1] == doc) if rank <= 100 else doc not in top,
                      'positive rank/top100 mismatch')
    denominator = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    ndcg = sum(1 / math.log2(r + 1) for r in ranks if r <= 10) / denominator
    recall = sum(r <= 100 for r in ranks) / len(ranks)
    study.require(abs(ndcg - record['ndcg10']) < 1e-12
                  and abs(recall - record['recall100']) < 1e-12, 'metric mismatch')
    return [ndcg, recall]


def review(base, run, output):
    study.require(not output.exists() and output.parent == run.parent, 'fresh review unit')
    files = study.verify(base, run)
    phase_results = {}
    for name in study.PHASES:
        controller = study.campaign.read(run / (name + '-controller-exit.json'))
        study.require(controller['status'] == 'all_phases_complete'
                      and controller['completed'] == [name], 'unfinished lane')
        phase_results[name] = study.audit.sealed(run / name)
    output.mkdir()
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    manifest = dict(input_sha256=study.campaign.io.sha(run / 'inputs.json'),
        source_sha256=study.campaign.io.sha(Path(__file__)),
        source_role='independent_postprocessing_of_predeclared_comparisons',
        phases={p: study.campaign.io.sha(run / p / 'complete.json') for p in study.PHASES},
        optimizer_updates=0, locked_test_access=False, native_cost_evaluated=False)
    study.campaign.io.write(output / 'inputs.json', manifest)
    all_rows, verified = {}, 0
    for phase in study.PHASES:
        rows = study.campaign.rows(run / phase / 'rankings.jsonl')
        study.require(len(rows) == 1536 and len({r['query_id'] for r in rows}) == 1536,
                      'diagnostic population')
        for row in rows:
            for record in row['rankings'].values():
                recompute(record)
                verified += 1
        derived = study.summarize(rows, 'rankings')
        for profile in derived:
            np.testing.assert_allclose(derived[profile]['macro'],
                phase_results[phase]['quality'][profile]['macro'], rtol=0, atol=1e-12)
        all_rows[phase] = rows
    study.require([r['query_id'] for r in all_rows['mask']]
                  == [r['query_id'] for r in all_rows['geometry']], 'cross-lane population')
    mask = phase_results['mask']
    cost_s = mask['costs']['S']['semantic_df_work'] + mask['costs']['S']['lexical_df_work']
    mask_gates = {}
    for name, comparison in mask['versus_S'].items():
        cost = mask['costs'][name]
        ratio = (cost['semantic_df_work'] + cost['lexical_df_work']) / cost_s
        quality = all(ci[0] >= -.005 for ci in comparison['ci95'])
        mask_gates[name] = dict(quality_noninferiority=quality,
            total_df_work_ratio=ratio, logical_work_gate=ratio <= .70,
            eligible_for_next_native_and_unexposed_check=quality and ratio <= .70)
    geometry = phase_results['geometry']
    geometry_gates, vs_pca = {}, {}
    for name, comparison in geometry['versus_dense'].items():
        geometry_gates[name] = dict(
            dense_noninferiority=all(ci[0] >= -.005 for ci in comparison['ci95']))
    for k in study.DIMS:
        for suffix in ('', '_hybrid'):
            name = 'query_weighted-' + str(k) + suffix
            vs_pca[name] = study.paired(all_rows['geometry'], name, 'pca-' + str(k) + suffix)
    # The available top100 supports four predeclared head/boundary pairs.
    pair_positions = study.PAIRS[:4]
    dense_d = np.load(files['documents'], mmap_mode='r')
    dense_q = np.load(files['queries'], mmap_mode='r')
    ids = study.campaign.read(files['selection'])['diagnostic']
    margin_audit = {}
    for name in geometry['fitted']:
        maps = np.load(run / 'geometry' / (name + '.npz'))
        operator = maps['document_map'] @ maps['query_map'].T
        errors, correct, comparable = [], 0, 0
        for row, index in enumerate(ids):
            top = all_rows['geometry'][row]['rankings']['dense']['top100']
            positive, negative = np.array([(top[p], top[n]) for p, n in pair_positions]).T
            difference = dense_d[positive].astype(np.float64) - dense_d[negative].astype(np.float64)
            query = dense_q[index].astype(np.float64)
            exact = difference @ query
            actual = difference @ (operator @ query)
            errors.extend((actual - exact).tolist())
            comparable += int(np.count_nonzero(exact > 1e-12))
            correct += int(np.count_nonzero((exact > 1e-12) & (actual > 0)))
        margin_audit[name] = dict(pair_count=len(errors), comparable_pairs=comparable,
            mean_absolute_error=float(np.mean(np.abs(errors))),
            mean_squared_error=float(np.mean(np.square(errors))),
            teacher_order_fraction=correct / comparable)
    result = dict(verified_rank_records=verified, mask_gates=mask_gates,
        geometry_gates=geometry_gates, query_weighted_versus_pca=vs_pca,
        head_boundary_margin=margin_audit, independent_holdout=False,
        native_cost_evaluated=False, automatic_promotion=False,
        no_winner_selected_for_expansion=True,
        finished_unix=time.time())
    study.campaign.io.write(output / 'results.json', result)
    # Recheck closure hashes after analysis; a changing run invalidates this review.
    for phase, expected in manifest['phases'].items():
        study.require(study.campaign.io.sha(run / phase / 'complete.json') == expected,
                      'phase changed during review')
    study.campaign.io.write(output / 'complete.json', dict(passed=True,
        files={p.name: study.campaign.io.sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.research_root.resolve(strict=True), args.run.resolve(strict=True),
           args.output.resolve())
