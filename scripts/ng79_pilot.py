"""One frozen query-breadth comparison, using the unchanged D update engine.

The eight training phases stop at384 updates. No endpoint inference, observer,
retention transform, mid-run witness refresh or automatic expansion is enabled.
"""

import argparse
import copy
from pathlib import Path
import shutil
import subprocess

import ng71_pilot as pipeline
import ng71_preflight as io
from review_ng75_displacement import compare, read, require, sha, verify as verify_files


PREPARATION = 'NG-0079/static-witnesses-v1'
PREPARATION_INPUT = 'f6e0430ea6de939f48c76c153f2f45362d4b7a07761a33a1550e2a647d2fe87c'
PREPARATION_INVENTORY = '8cc056c9c5d8f70dd045b1e191292d53abb05e0c31db2c3dbecaf9dd083c49e8'
REVIEW = 'NG-0079/preparation-review-v1'
REVIEW_INPUT = '17cd6f12413cce893a8509196d28677836834ed958a7773dc898e060eff3a9ca'
REVIEW_INVENTORY = 'e2bb37e1adadb3cbbc52e411f6e17df654a7c0fcd79c2e55b0f6ea7888df264b'
PHASES = tuple(f'train-{arm}-{end}' for end in (96, 192, 288, 384) for arm in ('R', 'B'))
PREFIX = dict(model_sha256='92e574071092f08f6288a980ac27a1f0e17b82b0274acac40d3cf8cc228abd49',
              optimizer_fingerprint='f7726f77dcec15d17c888a6b48ff03168d43fae1fd8d428fc888580d9e820540',
              tokens={'query': 6314, 'document': 3953027}, candidate_pairs=30493)


def configuration(core, arm):
    require(arm in ('R', 'B'), 'unknown study arm')
    require(core['arms'] == {'D': {'pool': 'witness', 'objective': 'balanced_soft_pair'}}
            and core['witness']['refresh_updates'] == [0], 'original static D objective changed')
    config = copy.deepcopy(core)
    config.update(training_enabled=True, status='frozen_NG79_matched_breadth_384')
    config['arms'] = {a: copy.deepcopy(core['arms']['D']) for a in ('R', 'B')}
    config['pilot'].update(train_queries=384 if arm == 'R' else 1536,
                           queries_per_domain=128 if arm == 'R' else 512,
                           epochs=4 if arm == 'R' else 1)
    return config


def prefix_check(result):
    require(result['cumulative_steps'] == 96 and result['steps'] == 96
            and result['query_exposures'] == 384 and result['initial_optimizer_fingerprint'] is None,
            'common prefix did not start fresh')
    compare({key: result[key] for key in PREFIX}, PREFIX, atol=0.)


def preparation_ready(base):
    root, reviewed = base / PREPARATION, base / REVIEW
    require(sha(root / 'inputs.json') == PREPARATION_INPUT
            and sha(reviewed / 'inputs.json') == REVIEW_INPUT, 'wrong parent input')
    inventory_path = root.with_name(root.name + '-remote-inventory.json')
    require(sha(inventory_path) == PREPARATION_INVENTORY, 'preparation inventory changed')
    manifest = read(root / 'inputs.json')
    verify_files(base, manifest['dependencies'])
    verify_files(root, manifest['source'])
    verify_files(root, read(inventory_path)['files'], exact=True)
    result = read(root / 'witnesses/results.json')
    exited, tracking = (read(root / 'witnesses' / n) for n in ('exit.json', 'clearml.json'))
    require(read(root / 'witnesses/complete.json')['passed'] and exited['exit_code'] == 0
            and exited['error'] is None and exited['owned_group_closed']
            and tracking['actual_start'] and tracking['closed'] and tracking['outcome'] == 'passed',
            'preparation phase is not safely closed')
    require(result['passed'] and result['eligibility_gate_passed']
            and result['historical_384_witness_max_error'] == 0., 'preparation did not qualify')
    require(read(root / 'controller-exit.json')['status'] == 'all_phases_complete', 'parent still open')
    review_inventory = reviewed.with_name(reviewed.name + '-remote-inventory.json')
    require(sha(review_inventory) == REVIEW_INVENTORY, 'independent review inventory changed')
    mirror = reviewed.with_name(reviewed.name + '-mirror-review.json')
    receipt = read(mirror)
    require(receipt['passed'] and receipt['input_sha256'] == PREPARATION_INPUT
            and receipt['parent_inventory_sha256'] == PREPARATION_INVENTORY
            and receipt['review_input_sha256'] == REVIEW_INPUT
            and receipt['review_inventory_sha256'] == sha(review_inventory), 'Mac mirror review missing')
    verify_files(reviewed, read(review_inventory)['files'], exact=True)
    review_manifest = read(reviewed / 'inputs.json')
    verify_files(reviewed, review_manifest['source'])
    independent = read(reviewed / 'results.json')
    require(independent['passed'] and independent['queries'] == 1536 and independent['positives'] == 2680
            and independent['historical_384_exact'] and independent['max_independent_scalar_error'] <= 1e-12
            and independent['input_sha256'] == PREPARATION_INPUT
            and independent['reviewer_sha256'] == review_manifest['source']['review_ng79_preparation.py'],
            'independent scalar review did not qualify')
    for stage in ('tests', 'review'):
        exited = read(reviewed / f'{stage}-exit.json')
        require(exited['exit_code'] == 0 and exited['error'] is None and exited['owned_group_closed']
                and exited['seconds'] <= 900 and exited['peak_tree_rss_bytes'] <= 2 ** 30,
                'review procedure not safely closed')
    deps = dict(manifest['dependencies'])
    for path, inventory in ((root, read(inventory_path)), (reviewed, read(review_inventory))):
        deps.update({str((path / name).relative_to(base)): digest
                     for name, digest in inventory['files'].items()})
    for p in (inventory_path, review_inventory, mirror):
        deps[str(p.relative_to(base))] = sha(p)
    return deps


def verify(base, run):
    manifest = read(run / 'inputs.json')
    require(manifest['protocol'] == 'NG79_matched_D_breadth_training_v1'
            and manifest['phases'] == list(PHASES) and manifest['stage_limit_seconds'] == 1800
            and manifest['prefix_reference'] == PREFIX and manifest['total_updates_per_arm'] == 384
            and not any(manifest[k] for k in ('quality_evaluation', 'dev_access', 'locked_test_access',
                                              'gpu_observer', 'automatic_expansion')),
            'frozen training graph changed')
    verify_files(base, manifest['dependencies'])
    verify_files(run, manifest['source'])
    core = read(base / PREPARATION / 'config.json')
    require(read(run / 'configs.json') == {a: configuration(core, a) for a in ('R', 'B')},
            'common model, ranking, optimizer or schedule changed')
    require(read(run / 'selection.json') == read(base / PREPARATION / 'selection-audit.json')['selection'],
            'frozen training identities/order changed')
    return manifest


def freeze(base, run):
    require(run.parent == base / 'NG-0079', 'new NG79 unit required')
    deps = preparation_ready(base)
    core = read(base / PREPARATION / 'config.json')
    source = Path(__file__).resolve().parent
    names = [name for name in read(base / PREPARATION / 'inputs.json')['source']
             if name.endswith('.py') and not name.startswith('test_')]
    sources = [source / name for name in names]
    sources += [Path(__file__), source / 'review_ng79_preparation.py', source / 'review_ng77_preparation.py']
    sources = list(dict.fromkeys(sources))
    sources += [source.parent / 'tests' / name for name in (
        'test_ng79_execution.py', 'test_ng79_pilot.py', 'test_ng77_execution.py', 'test_ng77_cuda.py',
        'test_ng71_execution.py', 'test_ng79_preparation_review.py')]
    sources += [source.parent / 'docs/research-sae/reports/ng0001-ng0099' / name for name in (
        'ng0071-ranking-config.json', 'ng0079-query-breadth-plan.zh.md', 'ng0079-training-execution.zh.md')]
    require(len({p.name for p in sources}) == len(sources), 'duplicate frozen source name')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--',
                    *[str(p.relative_to(source.parent)) for p in sources]], cwd=source.parent, check=True)
    run.mkdir(parents=True, exist_ok=False)
    for p in sources:
        shutil.copy2(p, run / p.name)
    io.write(run / 'configs.json', {a: configuration(core, a) for a in ('R', 'B')})
    io.write(run / 'selection.json', read(base / PREPARATION / 'selection-audit.json')['selection'])
    io.write(run / 'inputs.json', dict(protocol='NG79_matched_D_breadth_training_v1', dependencies=deps,
        source={p.name: sha(p) for p in run.iterdir() if p.is_file()}, phases=list(PHASES),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source.parent, text=True).strip(),
        stage_limit_seconds=1800, prefix_reference=PREFIX, total_updates_per_arm=384,
        quality_evaluation=False, dev_access=False, locked_test_access=False,
        gpu_observer=False, automatic_expansion=False))
    verify(base, run)
    print('NG79_TRAINING_FROZEN', sha(run / 'inputs.json'), flush=True)


def worker(args):
    import numpy as np
    import torch
    from clearml import Task
    import ng71_execution as execution

    verify(args.research_root, args.run)
    require(args.phase in PHASES, 'unplanned phase')
    require(sha(Path(__file__)) == read(args.run / 'inputs.json')['source'][Path(__file__).name],
            'worker source is not frozen')
    for phase in PHASES[:PHASES.index(args.phase)]:
        result = execution.sealed(args.run / phase)
        if phase.endswith('-96'):
            prefix_check(result)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    np.random.seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0079/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    io.write(output / 'clearml-start.json', receipt)
    try:
        base, run = args.research_root, args.run
        _, arm, end = args.phase.split('-')
        end = int(end)
        config = read(run / 'configs.json')[arm]
        selection = read(run / 'selection.json')
        selected = set(selection['repeated' if arm == 'R' else 'breadth'])
        records = {r['query_id']: r for r in pipeline.rows(base / PREPARATION / 'witnesses/witnesses.jsonl')
                   if r['query_id'] in selected}
        queries = {q['query_id']: q for q in read(base / 'NG-0069/lexical-v1/queries.json')
                   if q['query_id'] in selected}
        documents = pipeline.rows(base / 'NG-0067/data/documents.jsonl')
        task.connect(dict(config=config, input_sha256=sha(run / 'inputs.json')))
        result = execution.train_fixed_reference_chunk(base, config, output, arm, end - 96, end,
            records, selection['orders'][arm], queries, documents, task,
            previous=run / f'train-{arm}-{end - 96}' if end > 96 else None, stage_limit_seconds=1800)
        if end == 96:
            prefix_check(result)
        result.update(gpu_observer=False, quality_evaluation=False, dev_access=False,
                      locked_test_access=False, static_reference_update=0,
                      common_prefix_reproduced=(end == 96))
        verify(base, run)
        io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        io.write(output / 'clearml.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3))
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    else:
        manifest = verify(args.research_root, args.run)
        require(sha(Path(__file__)) == manifest['source'][Path(__file__).name], 'executing source not frozen')
        if args.mode == 'worker':
            worker(args)
        elif args.mode == 'supervise':
            require(args.gpu is not None, 'explicit GPU required')
            pipeline.supervise_graph(args, PHASES, Path(__file__).name, limit_seconds=1800)
        else:
            print('NG79_TRAINING_VERIFIED', len(manifest['dependencies']))
