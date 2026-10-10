"""Independent metric/closure review of the fixed NG83 comparisons."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

import ng83_audit as audit_study
import ng83_operator as operator
from review_ng82_diagnostics import recompute


prior = audit_study.prior
campaign, require = prior.campaign, prior.require


def check_rows(rows, records, indices, profiles, labels):
    require([r['query_id'] for r in rows] == [records[i]['query_id'] for i in indices],
            'review query order')
    for row, index in zip(rows, indices, strict=True):
        require(row['domain'] == records[index]['subset'] and set(row['rankings']) == set(profiles),
                'review population/profile')
        for record in row['rankings'].values():
            require(record['gold_ids'] == labels[index], 'review positive identity')
            recompute(record)
    return len(rows) * len(profiles)


def review_audit(files, run, results):
    records = campaign.read(files['records'])
    selection = campaign.read(files['selection'])
    labels = campaign.read(files['labels'])
    rows = campaign.rows(run / 'train-replay/rankings.jsonl')
    count = check_rows(rows, records, selection['calibration'], ['dense', *audit_study.PROFILES], labels)
    quality = prior.summarize(rows, 'rankings')
    for name in quality:
        np.testing.assert_allclose(quality[name]['macro'],
            results['train-replay']['quality'][name]['macro'], rtol=0, atol=1e-12)
    deltas = {p: prior.paired(rows, p, 'S') for p in audit_study.PROFILES[1:]}
    losses = {p: float(np.mean([r['pair_loss'][p] for r in rows])) for p in audit_study.PROFILES}
    baseline = losses['S']
    design = campaign.read(files['design'])
    for p in losses:
        np.testing.assert_allclose(losses[p] - baseline,
            design['profiles'][p]['joint_train_loss_delta'], rtol=0, atol=1e-12)
    split = campaign.read(files['split'])
    labels = campaign.read(files['labels'])
    train_docs = {d for i in split['train'] for d in labels[i]}
    diagnostic_docs = {d for i in split['diagnostic'] for d in labels[i]}
    overlap = len(train_docs & diagnostic_docs)
    require(overlap == results['supervision']['positive_document_overlap_train_diagnostic'],
            'supervision document overlap')
    return dict(verified_rank_records=count, train_quality=quality, train_versus_S=deltas,
        pair_loss=losses, in_sample_surrogate_mismatch=(losses['utility-half'] < baseline
            and deltas['utility-half']['ci95'][0][1] < 0),
        cannot_attribute_failure_only_to_query_generalization=True,
        shared_positive_documents=overlap)


def promotion_gate(vs_field, vs_initial):
    return all(c['ci95'][0][0] > 0 and c['ci95'][1][0] >= -.005
               for c in (vs_field, vs_initial))


def review_operator(files, run, results):
    records = campaign.read(files['records'])
    labels = campaign.read(files['labels'])
    selection = campaign.read(files['selection'])
    prior.audit.sealed(files['pca'].parent)
    historical = campaign.rows(files['pca'].parent / 'rankings.jsonl')
    bank = np.load(run / 'prepare/bank.npz')
    expected = bank['query_indices'][bank['training_order']].tolist()
    groups, count = {}, 0
    for arm in operator.WEIGHTS:
        train = results['fit-' + arm]
        require(train['bank_sha256'] == campaign.io.sha(run / 'prepare/bank.npz')
                and train['initial_map_sha256'] == campaign.io.sha(files['pca'])
                and train['updates'] == 3072 and train['distinct_train_queries'] == 12288
                and train['boundary_weight'] == operator.WEIGHTS[arm], 'matched fit identity')
        require(campaign.read(run / ('fit-' + arm) / 'seen-query-indices.json') == expected,
                'unmatched fit exposures')
        updates = campaign.rows(run / ('fit-' + arm) / 'updates.jsonl')
        require([r['step'] for r in updates] == list(range(1, 3073)), 'update trace')
        require(all(np.isfinite([r[k] for k in ('loss', 'field', 'boundary',
                    'unclipped_gradient_norm')]).all() for r in updates), 'nonfinite trace')
        all_rows = campaign.rows(run / ('rank-' + arm) / 'rankings.jsonl')
        require(len(all_rows) == 1920, 'rank population size')
        groups[arm] = {}
        for role in ('calibration', 'diagnostic'):
            rows = [r for r in all_rows if r['role'] == role]
            count += check_rows(rows, records, selection[role],
                ['dense', 'dense_hybrid', 'initial', 'initial_hybrid', arm, arm + '_hybrid'], labels)
            if role == 'diagnostic':
                require(len(rows) == len(historical), 'historical control population')
                for current, previous in zip(rows, historical, strict=True):
                    require(current['query_id'] == previous['query_id'], 'historical query identity')
                    for now, then in (('dense', 'dense'), ('initial', 'pca-256'),
                                      ('initial_hybrid', 'pca-256_hybrid')):
                        for key in ('gold_ids', 'gold_ranks', 'top100'):
                            require(current['rankings'][now][key] == previous['rankings'][then][key],
                                    'historical teacher/PCA replay')
            quality = prior.summarize(rows, 'rankings')
            for name in quality:
                np.testing.assert_allclose(quality[name]['macro'],
                    results['rank-' + arm]['quality'][role][name]['macro'], rtol=0, atol=1e-12)
            groups[arm][role] = rows
    comparisons, quality, geometry = {}, {}, {}
    for role in ('calibration', 'diagnostic'):
        left, right = groups['field'][role], groups['boundary'][role]
        combined = []
        for a, b in zip(left, right, strict=True):
            require(a['query_id'] == b['query_id'], 'arm populations')
            for name in ('dense', 'dense_hybrid', 'initial', 'initial_hybrid'):
                require(a['rankings'][name] == b['rankings'][name], 'different control ranking')
            combined.append(dict(query_id=a['query_id'], domain=a['domain'],
                rankings={**a['rankings'], **b['rankings']}))
        comparisons[role] = {name: prior.paired(combined, 'boundary_hybrid', name)
                            for name in ('field_hybrid', 'initial_hybrid', 'dense', 'dense_hybrid')}
        quality[role] = prior.summarize(combined, 'rankings')
        geometry[role] = {}
        for arm in operator.WEIGHTS:
            metrics = [r['geometry'][arm] for r in groups[arm][role]]
            geometry[role][arm] = dict(
                mean_centered_score_mse=float(np.mean([m['centered_score_mse'] for m in metrics])),
                mean_boundary_margin_mse=float(np.mean([m['boundary_margin_mse'] for m in metrics])),
                order_retention=sum(m['preserved_positive_pairs'] for m in metrics)
                    / sum(m['positive_teacher_pairs'] for m in metrics))
    eligible = promotion_gate(comparisons['diagnostic']['field_hybrid'],
                              comparisons['diagnostic']['initial_hybrid'])
    return dict(verified_rank_records=count, quality=quality, comparisons=comparisons,
        geometry=geometry, eligible_for_small_SAE_transfer=eligible,
        performance_or_sparse_capacity_proven=False)


def review(base, run, output, mode):
    study = audit_study if mode == 'audit' else operator
    require(not output.exists() and output.parent == run.parent, 'fresh NG83 review required')
    files = study.verify(base, run)
    require(campaign.io.sha(Path(study.__file__)) == campaign.read(run / 'inputs.json')[
            'source'][Path(study.__file__).name], 'review imported unfrozen study')
    results = {}
    for phase in study.PHASES:
        controller = campaign.read(run / (phase + '-controller-exit.json'))
        require(controller['status'] == 'all_phases_complete'
                and controller['completed'] == [phase], 'unfinished lane')
        results[phase] = prior.audit.sealed(run / phase)
    output.mkdir()
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    validator = Path(__file__).parent / 'review_ng82_diagnostics.py'
    shutil.copy2(validator, output / validator.name)
    phase_hashes = {p: campaign.io.sha(run / p / 'complete.json') for p in study.PHASES}
    campaign.io.write(output / 'inputs.json', dict(input_sha256=campaign.io.sha(run / 'inputs.json'),
        phases=phase_hashes, review_source_sha256=campaign.io.sha(Path(__file__)),
        metric_validator_sha256=campaign.io.sha(Path(__file__).parent / 'review_ng82_diagnostics.py'),
        mode=mode, locked_test_access=False, optimizer_updates=0))
    result = (review_audit if mode == 'audit' else review_operator)(files, run, results)
    result.update(independent_holdout=False, native_cost_evaluated=False,
                  automatic_promotion=False, finished_unix=time.time())
    campaign.io.write(output / 'results.json', result)
    for phase, digest in phase_hashes.items():
        require(campaign.io.sha(run / phase / 'complete.json') == digest, 'changing reviewed run')
    campaign.io.write(output / 'complete.json', dict(passed=True,
        files={p.name: campaign.io.sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit', 'operator'), required=True)
    args = parser.parse_args()
    review(args.research_root.resolve(strict=True), args.run.resolve(strict=True),
           args.output.resolve(), args.mode)
