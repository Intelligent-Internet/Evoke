"""Frozen, TRAIN-only NG77 endpoint graph after both training processes close.

Only the existing 768 query identities and pre-training gates are used. This
does not train, extend, access DEV/LOCKED_TEST, tune lambda or promote a model.
"""

import argparse
from pathlib import Path
import shutil

import numpy as np

import ng71_execution as execution
import ng71_observation as observation
import ng71_pilot as pipeline
import ng71_preflight as io
import ng77_pilot as training
import review_ng77_training as audit_training
from review_ng75_displacement import require, verify as verify_files


TRAIN = 'NG-0077/matched-retention-96-v1'
MODELS = ('initial', 'dense', 'bm25', 'Z-96', 'K-96')
PHASES = ('audit-training', 'encode-Z-96', 'encode-K-96',
          *(f'rank-{m}' for m in MODELS), 'review')
CACHE = 'NG-0071/pilot-v1/encode-D-96'
CACHE_COMPLETE = 'cd9ecb1f0349e9d0f3b3744dcda82c56da50048b5403e3c9d8003ba52b4df4ea'
CACHE_MODEL = '92e574071092f08f6288a980ac27a1f0e17b82b0274acac40d3cf8cc228abd49'


def ready(base):
    root = base / TRAIN
    require(io.sha(root / 'inputs.json') == audit_training.INPUT, 'wrong training parent')
    m = training.verify(base, root)
    results = observation.training_closed(root, training.PHASES)
    require(execution.read(root / 'controller-exit.json')['status'] == 'all_phases_complete',
            'training controller not closed')
    inventory = root.with_name(root.name + '-remote-inventory.json')
    require(io.sha(inventory) == audit_training.INVENTORY, 'unverified training mirror')
    verify_files(root, execution.read(inventory)['files'], exact=True)
    return m, results, inventory


def cache_for_zero(base, result):
    if result['model_sha256'] != CACHE_MODEL:
        return None
    root = base / CACHE
    require(io.sha(root / 'complete.json') == CACHE_COMPLETE, 'old D96 cache changed')
    cached = execution.sealed(root)
    require(cached['model_sha256'] == CACHE_MODEL, 'old D96 cache model differs')
    return root


def verify(base, run):
    m = execution.read(run / 'inputs.json')
    require(m['protocol'] == 'NG77_train_only_endpoints_v1'
            and m['phases'] == list(PHASES) and m['models'] == list(MODELS)
            and m['gates'] == training.GATES and m['stage_limit_seconds'] == 1800
            and m['training_updates'] == 0 and not m['dev_access'] and not m['locked_test_access'],
            'frozen endpoint contract differs')
    verify_files(base, m['dependencies'])
    verify_files(run, m['source'])
    parent = base / TRAIN
    for name in ('config.json', 'selection.json', 'endpoint-queries.json'):
        require(io.sha(run / name) == io.sha(parent / name), 'pre-training selection/config changed')
    queries = observation.surface_queries(base, execution.read(run / 'selection.json'),
                                          True, include_dev=False)
    require(execution.read(run / 'endpoint-queries.json') == queries, 'query scope changed')
    _, results, _ = ready(base)
    cache = cache_for_zero(base, results['train-Z-96'])
    require(m['Z_document_cache'] == (CACHE if cache else None), 'cache route changed')
    return m


def freeze(base, run):
    m, results, inventory = ready(base)
    dependencies = dict(m['dependencies'])
    parents = [base / TRAIN, base / 'NG-0071/pilot-v1/train-D-96']
    cache = cache_for_zero(base, results['train-Z-96'])
    if cache is not None:
        parents.append(cache)
    for root in parents:
        dependencies.update({str(p.relative_to(base)): io.sha(p)
                             for p in root.rglob('*') if p.is_file()})
    dependencies[str(inventory.relative_to(base))] = io.sha(inventory)
    run.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    names = [p.name for p in source.glob('ng71_*.py')]
    names += ['ng77_endpoints.py', 'ng77_pilot.py', 'ng77_cuda_canary.py', 'ng77_retention.py',
              'ng77_training.py', 'review_ng77_training.py', 'review_ng77_preparation.py',
              'review_ng77_endpoints.py', 'review_ng75_displacement.py', 'review_ng71_pilot.py',
              'audit_ng70_provenance.py', 'analyze_ng66_learning_curves.py']
    for name in names:
        shutil.copy2(source / name, run / name)
    for name in ('test_ng77_training_review.py', 'test_ng77_endpoints.py'):
        shutil.copy2(source.parent / 'tests' / name, run / name)
    # Imported synthetic fixtures are frozen alongside their tests.
    for name in ('test_ng77_retention.py', 'test_ng77_execution.py', 'test_ng77_cuda.py'):
        shutil.copy2(source.parent / 'tests' / name, run / name)
    for name in ('config.json', 'selection.json', 'endpoint-queries.json'):
        shutil.copy2(base / TRAIN / name, run / name)
    plan = source.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0077-endpoint-protocol.zh.md'
    shutil.copy2(plan, run / plan.name)
    io.write(run / 'inputs.json', dict(protocol='NG77_train_only_endpoints_v1',
        dependencies=dependencies, source={p.name: io.sha(p) for p in run.iterdir() if p.is_file()},
        phases=list(PHASES), models=list(MODELS), gates=training.GATES, stage_limit_seconds=1800,
        training_updates=0, dev_access=False, locked_test_access=False, Z_document_cache=CACHE if cache else None))
    verify(base, run)
    print('NG77_ENDPOINTS_FROZEN', io.sha(run / 'inputs.json'), flush=True)


def dispatch(base, run, phase):
    config = execution.read(run / 'config.json')
    parent, output = base / TRAIN, run / phase
    observation.training_closed(parent, training.PHASES)
    if phase == 'audit-training':
        return audit_training.review(base, parent)
    require(execution.sealed(run / 'audit-training')['passed'], 'independent training audit failed')
    if phase.startswith('encode-'):
        model = phase.removeprefix('encode-')
        queries = execution.read(run / 'endpoint-queries.json')
        documents = pipeline.rows(base / 'NG-0067/data/documents.jsonl')
        cache = execution.read(run / 'inputs.json')['Z_document_cache'] if model == 'Z-96' else None
        return execution.encode_checkpoint(base, config, output, parent / ('train-' + model),
            queries, documents, document_cache=base / cache if cache else None, stage_limit_seconds=1800)
    if phase.startswith('rank-'):
        # Both encodings close before looking at any new quality outcome.
        for model in ('Z-96', 'K-96'):
            execution.sealed(run / ('encode-' + model))
        return observation.rank(base, config, run, output, phase.removeprefix('rank-'),
            training_run=parent, training_phases=training.PHASES, train_only=True,
            witness_source=base / 'NG-0071/pilot-step0-v1', stage_limit_seconds=1800)
    if phase == 'review':
        from review_ng77_endpoints import review
        return review(base, run)
    raise ValueError('unapproved endpoint phase')


def worker(args):
    import torch
    from clearml import Task

    verify(args.research_root, args.run)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(71001)
    Task.set_offline(True)
    task = Task.init(project_name='Evoke-NG', task_name='NG-0077/endpoints/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    io.write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(input_sha256=io.sha(args.run / 'inputs.json'), phase=args.phase))
        result = dispatch(args.research_root, args.run, args.phase)
        verify(args.research_root, args.run)
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
    args.research_root, args.run = args.research_root.resolve(strict=True), args.run.resolve()
    require(args.run.parent == args.research_root / 'NG-0077', 'separate NG77 endpoint unit required')
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    else:
        manifest = verify(args.research_root, args.run)
        require(io.sha(Path(__file__)) == manifest['source']['ng77_endpoints.py'], 'executing source not frozen')
        if args.mode == 'worker':
            worker(args)
        elif args.mode == 'supervise':
            require(args.gpu is not None, 'explicit physical GPU required')
            pipeline.supervise_graph(args, PHASES, 'ng77_endpoints.py', limit_seconds=1800)
