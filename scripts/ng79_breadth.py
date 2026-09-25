"""Freeze and prepare one TRAIN-only, matched-exposure D-objective breadth trial.

This entrypoint mines static initial witnesses only. It cannot train, evaluate
DEV/LOCKED_TEST or automatically expand the failed NG77 retention experiment.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

import ng71_data as data
import ng71_diagnostics as diagnostics
import ng71_execution as execution
import ng71_pilot as pipeline
import ng71_preflight as io
import ng71_snapshot as snapshot
from review_ng75_displacement import compare, read, require, sha, verify as verify_files


PARENT = 'NG-0078/boundary-coverage-v1'
PARENT_INPUT = 'adf0862176bb81e40f7bbd994c3ae4d17771c1d4ede86f51dc9f8351fd1c6778'
INVENTORY = 'a6e35948e372529219c306a7c28f5bd929c2f58e9efbb40145585d2846a9d42c'
LOCAL_REVIEW = '79b9051174932d456d92faf123dfd313df083990993d0019d2f55dc98980f665'
DATA_AUDIT = 'NG-0071/data-preflight-v4/results.json'
DATA_SHA = '65b284abcae6e2e208e1e21d13c9d17f11907ba9d55598922eadcbb5b0d32327'
METHOD = dict(protocol='NG79_D_query_breadth_v1', domains=['fever', 'hotpotqa', 'nq'],
              repeated_distinct=384, breadth_distinct=1536, exposures_per_arm=1536,
              updates_per_arm=384, batch_queries=4, seed=79001,
              first_epoch='historical_NG71_first384', witness_reference_update=0,
              initialization='fresh_common_NG3_in_both_arms', shared_prefix_reused=False,
              stage_limit_seconds=5400, training_enabled=False,
              dev_access=False, locked_test_access=False)


def select(queries, provenance, old, prefix, *, per_domain=128, seed=79001):
    """Hash-select breadth without observing quality or teacher agreement."""
    lookup = {r['query_id']: r for r in queries}
    require(len(lookup) == len(queries), 'duplicate query identity')
    domains = METHOD['domains']
    require(len(old['pilot']) == len(old['sentinel']) == 3 * per_domain,
            'historical selection count changed')
    require(len(prefix) == len(set(prefix)) == 3 * per_domain
            and set(prefix) == set(old['pilot']), 'historical prefix changed')
    protected = old['pilot'] + old['sentinel']
    require(len(set(protected)) == len(protected), 'historical surfaces overlap')
    require(all(q in lookup and lookup[q]['split'] == 'TRAIN' for q in protected),
            'non-TRAIN or missing historical identity')
    texts = [data.provenance.normalized(lookup[q]['query']) for q in protected]
    require(len(set(texts)) == len(texts), 'historical query texts overlap')
    require(all(provenance[q] == 'resolved' for q in protected),
            'historical provenance changed')
    for surface in ('pilot', 'sentinel'):
        require(Counter(lookup[q]['subset'] for q in old[surface])
                == Counter({d: per_domain for d in domains}), 'historical domains changed')
    seen = set(texts)
    added = {}
    for domain in domains:
        candidates = sorted((r for r in queries if r['split'] == 'TRAIN'
                             and r['subset'] == domain and provenance.get(r['query_id']) == 'resolved'),
                            key=lambda r: (hashlib.sha256(
                                ('NG-0079/breadth-v1/' + r['query_id']).encode()).hexdigest(),
                                           r['query_id']))
        chosen = []
        for row in candidates:
            text = data.provenance.normalized(row['query'])
            if text not in seen:
                chosen.append(row['query_id'])
                seen.add(text)
            if len(chosen) == 3 * per_domain:
                break
        require(len(chosen) == 3 * per_domain, 'insufficient disjoint resolved TRAIN breadth')
        added[domain] = chosen
    rng = np.random.default_rng(seed)
    repeated, broad = list(prefix), list(prefix)
    for quarter in range(3):
        order = [str(q) for q in rng.permutation(old['pilot'])]
        repeated.extend(order)
        positions = Counter()
        for q in order:
            domain = lookup[q]['subset']
            broad.append(added[domain][quarter * per_domain + positions[domain]])
            positions[domain] += 1
    require([lookup[q]['subset'] for q in repeated] == [lookup[q]['subset'] for q in broad],
            'per-update domain sequence changed')
    require(Counter(repeated) == Counter({q: 4 for q in old['pilot']})
            and len(broad) == len(set(broad)) == 12 * per_domain, 'exposure count changed')
    return dict(repeated=old['pilot'], breadth=old['pilot'] + sum(added.values(), []),
                sentinel=old['sentinel'], added=added, orders={'R': repeated, 'B': broad},
                excluded_protected_query_texts=len(texts), teacher_quality_filter=False)


def config_from_parent(core):
    config = copy.deepcopy(core)
    config.update(experiment='NG-0079', training_enabled=False,
                  status='frozen_static_witness_preparation')
    config['requires'] = ['frozen_NG79_selection', 'global_witness_gate',
                          'reviewed_frozen_training_controller', 'bounded_resources']
    config['arms'] = {'D': core['arms']['D']}
    config['pilot'].update(train_queries=1536, queries_per_domain=512,
                           epochs=1, updates=384, query_order_seed=79001,
                           selection='NG79_protected_sentinel_then_TRAIN_hash_v1')
    config['pilot'].update(global_observation_updates=[0, 384], exposed_dev_queries=0,
                           selection_revision_reason='fixed breadth; protect old sentinel by ID and text')
    config['pilot'].pop('uniform_pair_ablation', None)
    config['witness'].update(refresh_reference='common_NG3_initial', refresh_updates=[0])
    config['witness'].pop('C_D_share_identical_manifests', None)
    config['gates'] = dict(eligible_positive_fraction_min_per_domain=.8,
        B_minus_R_ndcg_point_min=.005, B_minus_R_ndcg_95_lower_strictly_above=0.,
        B_minus_R_recall_95_lower_min=-.002,
        domain_ndcg_point_min_vs_R_and_initial=-.005,
        domain_recall_point_min_vs_R_and_initial=-.005,
        scale_block_doc_nnz_ratio_vs_init_above=1.25,
        scale_block_query_df_mean_ratio_vs_init_above=1.25,
        cost_query_surface='exposed_TRAIN_SENTINEL_384',
        cost_document_surface='complete_233009_corpus', native_cost_claim_allowed=False,
        bootstrap_replicates=10000, bootstrap_seed=79079,
        independent_holdout=False, automatic_scale_allowed=False)
    return config


def prepare_selection(base):
    audit = read(base / DATA_AUDIT)
    require(sha(base / DATA_AUDIT) == DATA_SHA, 'verified all-TRAIN data audit changed')
    require(audit['eligibility_gate_passed'] and not audit['model_inference']
            and not audit['locked_test_access'], 'invalid inherited data audit')
    require(all(audit['inputs'].get(k) == h for k, h in data.ANCHORS.items()),
            'inherited data audit does not bind source/corpus/teacher')
    old = read(base / 'NG-0071/pilot-v1/selection.json')
    require(old == audit['selection'], 'historical pilot differs from audited data')
    order = read(base / 'NG-0071/pilot-step0-v1/training-order.json')
    query_rows = read(base / 'NG-0069/lexical-v1/queries.json')
    require(len(query_rows) == 7872 and all(q['split'] == 'TRAIN' for q in query_rows[:6144]),
            'frozen query split inventory changed')
    records = {r['query_id']: r for r in audit['records']}
    require(len(records) == 6144 and set(records) == {q['query_id'] for q in query_rows[:6144]},
            'all-TRAIN audit coverage changed')
    selected = select(query_rows[:6144], {q: r['provenance_status'] for q, r in records.items()},
                      old, order[:384])
    summaries = {}
    for surface in ('repeated', 'breadth', 'sentinel'):
        summaries[surface] = {}
        for domain in METHOD['domains']:
            group = [records[q] for q in selected[surface] if records[q]['domain'] == domain]
            positive = sum(r['positive_count'] for r in group)
            supervised = sum(r['supervised_positives'] for r in group)
            summaries[surface][domain] = dict(queries=len(group), positives=positive,
                supervised_positives=supervised, eligible_positive_fraction=supervised / positive,
                no_supervision_queries=sum(r['eligible_pairs'] == 0 for r in group),
                positive_origins=dict(sum((Counter(r['positive_origins']) for r in group), Counter())))
    require(all(r['eligible_positive_fraction'] >= .8
                for s in summaries.values() for r in s.values()), 'original-pool eligibility floor failed')
    for arm, order in selected['orders'].items():
        require(not any(not any(records[q]['eligible_pairs'] for q in order[i:i + 4])
                        for i in range(0, len(order), 4)), 'whole unsupervised batch: ' + arm)
    return dict(selection=selected, original_pool_summaries=summaries,
                original_pool_eligibility_passed=True, global_witness_gate_passed=False,
                source_text_validation='reused_SHA_bound_NG71_all6144_source_audit',
                actual_optimizer_updates=0, new_model_inference=False, locked_test_access=False)


def verify(base, run):
    manifest = read(run / 'inputs.json')
    require(manifest['method'] == METHOD, 'frozen breadth contract changed')
    verify_files(base, manifest['dependencies'])
    verify_files(run, manifest['source'])
    config = config_from_parent(read(base / 'NG-0071/pilot-v1/config.json'))
    require(read(run / 'config.json') == config, 'fixed D configuration changed')
    compare(read(run / 'selection-audit.json'), prepare_selection(base), atol=0.)
    return manifest


def freeze(args):
    base, run = args.research_root, args.run
    require(run.parent == base / 'NG-0079', 'fresh NG79 unit required')
    parent = base / PARENT
    require(sha(parent / 'inputs.json') == PARENT_INPUT, 'NG78 parent changed')
    inv, receipt = [parent.with_name(parent.name + suffix) for suffix in
                    ('-remote-inventory.json', '-local-review.json')]
    require(sha(inv) == INVENTORY and sha(receipt) == LOCAL_REVIEW, 'parent mirrored closure changed')
    require(read(receipt)['passed'], 'parent independent review not passed')
    verify_files(parent, read(inv)['files'], exact=True)
    execution.sealed(parent / 'audit')
    manifest = read(parent / 'inputs.json')
    dependencies = dict(manifest['dependencies'])
    dependencies.update({PARENT + '/' + n: h for n, h in read(inv)['files'].items()})
    dependencies.update({str(p.relative_to(base)): sha(p) for p in (inv, receipt)})
    for name, digest in {**data.ANCHORS, DATA_AUDIT: DATA_SHA,
                         **read(base / DATA_AUDIT)['inputs']}.items():
        require(name not in dependencies or dependencies[name] == digest, 'dependency conflict')
        dependencies[name] = digest
    verify_files(base, dependencies)
    prepared = prepare_selection(base)
    root = Path(__file__).resolve().parent
    sources = [root / n for n in manifest['source'] if n.endswith('.py') and not n.startswith('test_')]
    sources += [Path(__file__), root.parent / 'tests/test_ng79_breadth.py',
                root.parent / 'tests/test_ng71_snapshot.py', root.parent / 'tests/test_ng71_diagnostics.py',
                root.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json',
                root.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0079-query-breadth-plan.zh.md']
    run.mkdir(parents=True, exist_ok=False)
    for source in sources:
        shutil.copy2(source, run / source.name)
    io.write(run / 'config.json', config_from_parent(read(base / 'NG-0071/pilot-v1/config.json')))
    io.write(run / 'selection-audit.json', prepared)
    source_files = [p for p in run.iterdir() if p.is_file()]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root.parent, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--',
                    *[str(p.relative_to(root.parent)) for p in sources]], cwd=root.parent, check=True)
    io.write(run / 'inputs.json', dict(method=METHOD, dependencies=dependencies,
                                     source_commit=revision,
                                     source={p.name: sha(p) for p in source_files}))
    verify(base, run)
    print('NG79_FROZEN', sha(run / 'inputs.json'), flush=True)


def prepare(base, run):
    config = read(run / 'config.json')
    selection = read(run / 'selection-audit.json')['selection']
    prepared = dict(eligibility_gate_passed=True, selection={'pilot': selection['breadth']},
                    query_order=selection['orders']['B'])
    result = snapshot.build_prepared(base, config, run / 'witnesses', prepared)
    require(result['eligibility_gate_passed'], 'full global witness eligibility failed')
    with (run / 'witnesses/witnesses.jsonl').open() as stream:
        records = {r['query_id']: r for r in (json.loads(line) for line in stream)}
    with (base / 'NG-0071/pilot-step0-v1/witnesses.jsonl').open() as stream:
        old = [json.loads(line) for line in stream]
    error = 0.
    for record in old:
        actual = records[record['query_id']]
        expected = copy.deepcopy(record)
        expected['arm_score_diagnostics'] = {'D': expected['arm_score_diagnostics']['D']}
        error = max(error, compare(actual, expected))
    repeated_config = copy.deepcopy(config)
    repeated_config['pilot'].update(train_queries=384, epochs=4)
    result['repeated_batch_audit'] = diagnostics.batch_audit(
        [records[q] for q in selection['repeated']], selection['orders']['R'], repeated_config)
    result.update(stage='NG79_TRAIN_static_initial_witness_preparation',
                  historical_384_witness_max_error=error,
                  inherited_original_pool_audit_sha256=DATA_SHA,
                  training_enabled=False, independent_holdout=False,
                  scientific_quality_evaluated=False, native_cost_evaluated=False)
    return result


def worker(args):
    verify(args.research_root, args.run)
    require(sha(Path(__file__)) == read(args.run / 'inputs.json')['source'][Path(__file__).name],
            'worker differs from frozen source')
    from clearml import Task
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0079/static-witnesses-v1',
                     reuse_last_task_id=False, auto_connect_frameworks=False,
                     auto_connect_arg_parser=False, auto_connect_streams=False,
                     auto_resource_monitoring=False)
    output = args.run / 'witnesses'
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    io.write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(method=METHOD, input_sha256=sha(args.run / 'inputs.json')))
        result = prepare(args.research_root, args.run)
        verify(args.research_root, args.run)
        io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException:
        receipt['outcome'] = 'failed'
        raise
    finally:
        task.close()
        receipt['closed'] = True
        io.write(output / 'clearml.json', receipt)


def supervise(args):
    verify(args.research_root, args.run)
    require(sha(Path(__file__)) == read(args.run / 'inputs.json')['source'][Path(__file__).name],
            'controller differs from frozen source')
    args.worker_script, args.gpu = Path(__file__).name, None
    stopped = []
    previous = {s: signal.signal(s, lambda signum, frame: stopped.append(signum))
                for s in (signal.SIGINT, signal.SIGTERM)}
    io.write(args.run / 'controller-start.json', dict(pid=os.getpid(), gpu=None, started_unix=time.time()))
    try:
        pipeline.phase(args, 'witnesses', stopped, cuda=False, limit_seconds=5400)
        verify(args.research_root, args.run)
        execution.sealed(args.run / 'witnesses')
        io.write(args.run / 'controller-exit.json', dict(status='all_phases_complete', finished_unix=time.time()))
    except BaseException as exc:
        io.write(args.run / 'controller-exit.json', dict(status='failed_preserve_attempt',
                 error=f'{type(exc).__name__}: {exc}', finished_unix=time.time()))
        raise
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=('witnesses',))
    args = parser.parse_args()
    if args.mode == 'verify':
        print('NG79_VERIFIED', len(verify(args.research_root, args.run)['dependencies']))
    else:
        globals()[args.mode](args)
