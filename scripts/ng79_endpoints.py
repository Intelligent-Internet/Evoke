"""Frozen terminal384 TRAIN-only observation, separate from active training.

No midpoint selection, new training, witness refresh, DEV or LOCKED_TEST use.
Only a closed and fully mirrored training graph can enter this controller.
"""

import argparse
from collections import Counter
from pathlib import Path
import shutil
import subprocess

import ng71_pilot as pipeline
import ng71_preflight as io
import ng79_pilot as training
from review_ng75_displacement import read, require, sha, verify as verify_files


TRAIN = 'NG-0079/matched-breadth-384-v1'
TRAIN_INPUT = 'fab3122d21c270dda659bef84e9e35bcaebe5d5a2ada1b8bc448437f97ef7068'
MODELS = ('initial', 'dense', 'bm25', 'R-384', 'B-384')
SURFACES = ('TRAIN_PILOT', 'TRAIN_SENTINEL', 'TRAIN_ADDED')
PHASES = ('audit-training', 'encode-R-384', 'encode-B-384',
          *(f'rank-{model}' for model in MODELS), 'review')
GATES = dict(ndcg_point_min=.005, ndcg_95_lower_strictly_above=0., recall_95_lower_min=-.002,
             per_domain_ndcg_and_recall_floor_vs_R_and_initial=-.005,
             document_nnz_ratio_vs_initial_max=1.25, sentinel_query_df_ratio_vs_initial_max=1.25,
             bootstrap_replicates=10000, bootstrap_seed=79079)


def surface_queries(rows, selection):
    require(len(rows) == 7872 and len({r['query_id'] for r in rows}) == 7872,
            'frozen lexical universe changed')
    lookup = {q['query_id']: (i, q) for i, q in enumerate(rows)}
    pilot, sentinel = selection['repeated'], selection['sentinel']
    added = [q for q in selection['breadth'] if q not in set(pilot)]
    require(len(pilot) == len(sentinel) == 384 and len(added) == 1152
            and len(set(pilot + sentinel + added)) == 1920, 'cohort overlap or count changed')
    result = []
    # Preserve the old768-query batch prefix; append the newly trained cohort.
    for surface, ids in zip(SURFACES, (pilot, sentinel, added), strict=True):
        require(Counter(lookup[q][1]['subset'] for q in ids)
                == Counter({d: len(ids) // 3 for d in ('fever', 'hotpotqa', 'nq')}), 'domain omitted')
        for q in ids:
            index, row = lookup[q]
            require(row['split'] == 'TRAIN', 'endpoint query is not TRAIN')
            result.append(dict(row, surface=surface, lexical_index=index))
    return result


def ready(base):
    from review_ng79_training import sealed

    parent = base / TRAIN
    require(sha(parent / 'inputs.json') == TRAIN_INPUT, 'training input changed')
    manifest = training.verify(base, parent)
    control = read(parent / 'controller-exit.json')
    require(control['status'] == 'all_phases_complete' and control['completed'] == list(training.PHASES),
            'training controller is not closed')
    for phase in training.PHASES:
        result = sealed(parent / phase)
        _, arm, end = phase.split('-')
        require(result['arm'] == arm and result['cumulative_steps'] == int(end), 'checkpoint graph changed')
    inventory = parent.with_name(parent.name + '-remote-inventory.json')
    verify_files(parent, read(inventory)['files'], exact=True)
    mirror = parent.with_name(parent.name + '-mirror-review.json')
    receipt = read(mirror)
    require(receipt['passed'] and receipt['input_sha256'] == TRAIN_INPUT
            and receipt['remote_inventory_sha256'] == sha(inventory)
            and receipt['files_verified'] == len(read(inventory)['files']), 'full Mac mirror missing')
    return manifest, inventory, mirror


def verify(base, run):
    m = read(run / 'inputs.json')
    require(m['protocol'] == 'NG79_terminal384_TRAIN_endpoints_v1'
            and m['phases'] == list(PHASES) and m['models'] == list(MODELS) and m['gates'] == GATES
            and m['training_input_sha256'] == TRAIN_INPUT and m['stage_limit_seconds'] == 1800
            and m['training_updates'] == 0 and not m['dev_access'] and not m['locked_test_access'],
            'endpoint contract changed')
    verify_files(base, m['dependencies'])
    verify_files(run, m['source'])
    _, inventory, mirror = ready(base)
    require(m['training_inventory_sha256'] == sha(inventory)
            and m['training_mirror_sha256'] == sha(mirror), 'parent closure/mirror changed')
    parent = base / TRAIN
    require(read(run / 'config.json') == read(parent / 'configs.json')['B']
            and read(run / 'selection.json') == read(parent / 'selection.json'), 'fixed selection/config changed')
    require(read(run / 'endpoint-queries.json') == surface_queries(
        read(base / 'NG-0069/lexical-v1/queries.json'), read(run / 'selection.json')), 'endpoint scope changed')
    return m


def freeze(base, run):
    require(run.parent == base / 'NG-0079', 'separate NG79 endpoint unit required')
    m, inventory, mirror = ready(base)
    parent, source = base / TRAIN, Path(__file__).resolve().parent
    deps = dict(m['dependencies'])
    deps.update({str((parent / name).relative_to(base)): digest
                 for name, digest in read(inventory)['files'].items()})
    deps.update({str(p.relative_to(base)): sha(p) for p in (inventory, mirror)})
    names = {n for n in m['source'] if n.endswith('.py')}
    names.update(('ng79_endpoints.py', 'review_ng79_training.py', 'review_ng79_endpoints.py',
                  'test_ng79_endpoints.py', 'test_ng79_training_review.py', 'test_ng79_endpoint_review.py'))
    sources = [(source.parent / 'tests' if n.startswith('test_') else source) / n for n in sorted(names)]
    docs = source.parent / 'docs/research-sae/reports/ng0001-ng0099'
    sources += [docs / n for n in ('ng0071-ranking-config.json', 'ng0079-query-breadth-plan.zh.md',
                                  'ng0079-endpoint-protocol.zh.md')]
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--',
                    *[str(p.relative_to(source.parent)) for p in sources]], cwd=source.parent, check=True)
    run.mkdir(parents=True, exist_ok=False)
    for p in sources:
        shutil.copy2(p, run / p.name)
    selection = read(parent / 'selection.json')
    io.write(run / 'config.json', read(parent / 'configs.json')['B'])
    io.write(run / 'selection.json', selection)
    io.write(run / 'endpoint-queries.json', surface_queries(
        read(base / 'NG-0069/lexical-v1/queries.json'), selection))
    io.write(run / 'inputs.json', dict(protocol='NG79_terminal384_TRAIN_endpoints_v1',
        dependencies=deps, source={p.name: sha(p) for p in run.iterdir() if p.is_file()},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source.parent, text=True).strip(),
        phases=list(PHASES), models=list(MODELS), gates=GATES, stage_limit_seconds=1800,
        training_updates=0, dev_access=False, locked_test_access=False,
        training_input_sha256=TRAIN_INPUT, training_inventory_sha256=sha(inventory),
        training_mirror_sha256=sha(mirror)))
    verify(base, run)
    print('NG79_ENDPOINTS_FROZEN', sha(run / 'inputs.json'), flush=True)


def dispatch(base, run, phase):
    import ng71_execution as execution
    import ng71_observation as observation
    from review_ng79_training import review as review_training

    output, parent = run / phase, base / TRAIN
    manifest = read(run / 'inputs.json')
    if phase == 'audit-training':
        return review_training(base, parent, manifest['training_inventory_sha256'])
    require(execution.sealed(run / 'audit-training')['passed'], 'independent training audit failed')
    config, queries = read(run / 'config.json'), read(run / 'endpoint-queries.json')
    if phase.startswith('encode-'):
        model = phase.removeprefix('encode-')
        require(model in ('R-384', 'B-384'), 'only terminal checkpoints may encode')
        return execution.encode_fixed_terminal_checkpoint(base, config, output, parent / ('train-' + model),
            queries, pipeline.rows(base / 'NG-0067/data/documents.jsonl'), stage_limit_seconds=1800)
    for model in ('R-384', 'B-384'):
        execution.sealed(run / ('encode-' + model))
    if phase.startswith('rank-'):
        return observation.rank_fixed_surface(base, config, run, output, phase.removeprefix('rank-'),
            queries, parent, witness_source=base / training.PREPARATION / 'witnesses',
            stage_limit_seconds=1800)
    if phase == 'review':
        from review_ng79_endpoints import review
        return review(base, run)
    raise ValueError('unplanned endpoint phase')


def worker(args):
    import numpy as np
    import torch
    from clearml import Task

    verify(args.research_root, args.run)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    np.random.seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='Evoke-NG', task_name='NG-0079/endpoints/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    io.write(args.run / args.phase / 'clearml-start.json', receipt)
    try:
        task.connect(dict(input_sha256=sha(args.run / 'inputs.json'), phase=args.phase))
        result = dispatch(args.research_root, args.run, args.phase)
        verify(args.research_root, args.run)
        io.write(args.run / args.phase / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        io.write(args.run / args.phase / 'clearml.json', receipt)


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
        m = verify(args.research_root, args.run)
        require(sha(Path(__file__)) == m['source'][Path(__file__).name], 'executing source not frozen')
        if args.mode == 'worker':
            worker(args)
        elif args.mode == 'supervise':
            require(args.gpu is not None, 'explicit free physical GPU required')
            pipeline.supervise_graph(args, PHASES, Path(__file__).name, limit_seconds=1800)
        else:
            print('NG79_ENDPOINTS_VERIFIED', len(m['dependencies']))
