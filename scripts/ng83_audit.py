"""NG83 frozen-mask TRAIN replay and supervision readiness, without training."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse

import ng82_diagnostics as prior
import ng80_data as data


campaign, old, require = prior.campaign, prior.old, prior.require
PARENT = 'NG-0082/diagnostics-v1'
REVIEW = 'NG-0082/review-v1'
INPUT_SHA = '5f591075a122242f85b126abe3d4d12a62da974424978dcaf70dc360e2605804'
REVIEW_SHA = 'd566c47b45cad82704bff5341aa7b042e5ecd1bf4090b418c70705291fe4fee6'
PHASES = ('train-replay', 'supervision')
PROFILES = ('S', 'utility-half', 'df-half')
PROTOCOL = 'ng0083-ranking-causality-protocol.zh.md'


def contract():
    return dict(protocol='NG83_frozen_mask_audit_v1', phases=list(PHASES),
                profiles=list(PROFILES), optimizer_updates=0, mask_refits=0,
                locked_test_access=False, independent_holdout=False,
                automatic_promotion=False, native_cost_evaluated=False)


def paths(base):
    files = prior.paths(base)
    files.update(masks=base / PARENT / 'mask/masks.npz',
                 design=base / PARENT / 'mask/mask-design.json',
                 pairs=base / PARENT / 'mask/teacher-pairs.json',
                 diagnostic=base / PARENT / 'mask/rankings.jsonl')
    return files


def verify(base, run):
    manifest = campaign.read(run / 'inputs.json')
    require(all(manifest.get(k) == v for k, v in contract().items()), 'NG83 contract changed')
    require(manifest['files'] == {k: str(p.relative_to(base)) for k, p in paths(base).items()},
            'NG83 paths changed')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in manifest[key].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve())
                    and campaign.io.sha(path) == digest, 'changed dependency: ' + name)
    require(manifest['dependencies'][PARENT + '/inputs.json'] == INPUT_SHA
            and manifest['dependencies'][REVIEW + '/results.json'] == REVIEW_SHA,
            'NG82 anchor changed')
    required = set(manifest['files'].values()) | {PARENT + '/inputs.json', REVIEW + '/results.json'}
    require(required <= set(manifest['dependencies']), 'unbound payload')
    return {k: base / v for k, v in manifest['files'].items()}


def freeze(base, run):
    require(run.parent == base / 'NG-0083' and not run.exists(), 'fresh NG83 required')
    parent = base / PARENT
    require(campaign.io.sha(parent / 'inputs.json') == INPUT_SHA
            and campaign.io.sha(base / REVIEW / 'results.json') == REVIEW_SHA,
            'NG82 parent identity')
    prior.verify(base, parent)
    for phase in prior.PHASES:
        prior.audit.sealed(parent / phase)
        require(campaign.read(parent / (phase + '-controller-exit.json'))['status']
                == 'all_phases_complete', 'NG82 unfinished')
    proof = campaign.read(base / REVIEW / 'complete.json')
    require(proof['passed'], 'NG82 review not passed')
    for name, digest in proof['files'].items():
        require(campaign.io.sha(base / REVIEW / name) == digest, 'NG82 review changed')
    run.mkdir(parents=True)
    for name in campaign.read(parent / 'inputs.json')['source']:
        shutil.copy2(parent / name, run / name)
    source = Path(__file__).resolve().parent
    for path in (Path(__file__), source.parent / 'tests/test_ng83_audit.py',
                 source.parent / 'docs/research-sae/reports/ng0001-ng0099' / PROTOCOL):
        shutil.copy2(path, run / path.name)
    files = paths(base)
    anchors = [parent / 'inputs.json', base / REVIEW / 'results.json',
               base / REVIEW / 'complete.json']
    anchors += [parent / p / 'complete.json' for p in prior.PHASES]
    manifest = dict(contract(), files={k: str(p.relative_to(base)) for k, p in files.items()},
        dependencies={str(p.relative_to(base)): campaign.io.sha(p)
                      for p in [*files.values(), *anchors]},
        source={p.name: campaign.io.sha(p) for p in run.iterdir() if p.is_file()})
    campaign.io.write(run / 'inputs.json', manifest)
    verify(base, run)
    print('NG83_FROZEN', campaign.io.sha(run / 'inputs.json'), flush=True)


def mask_inventory(archive, width):
    masks = {p: archive[p] for p in PROFILES}
    require(all(k.dtype == np.bool_ and k.shape == (width,) for k in masks.values())
            and masks['S'].all(), 'invalid frozen mask')
    return masks


def joint_loss(scores, pair, tau):
    require(tau > 0 and np.isfinite(tau), 'invalid temperature')
    margins = scores[pair['positive']] - scores[pair['negative']]
    require(len(margins) == 6 and np.isfinite(margins).all(), 'invalid pair margins')
    return float(np.logaddexp(0, -margins / tau).mean())


def replay(files, output):
    records, labels, selected, _, dd, dq = prior.load_context(files)
    sd = sparse.load_npz(files['sd']).astype(np.float64).tocsc()
    ld = sparse.load_npz(files['ld']).astype(np.float64).tocsc()
    sq, lq = sparse.load_npz(files['scq']), sparse.load_npz(files['lcq'])
    masks = mask_inventory(np.load(files['masks']), sd.shape[1])
    pairs, design = campaign.read(files['pairs']), campaign.read(files['design'])
    require(len(pairs) == len(selected['calibration']) == sq.shape[0] == lq.shape[0],
            'TRAIN alignment')
    documents = dd.astype(np.float64)
    results, started = [], time.monotonic()
    with (output / 'rankings.jsonl').open('x') as stream:
        for row, index in enumerate(selected['calibration']):
            require(pairs[row]['query_id'] == records[index]['query_id'], 'pair/query identity')
            teacher = documents @ dq[index].astype(np.float64)
            order = np.lexsort((np.arange(len(teacher)), -teacher))[:200]
            require(pairs[row]['positive'] == [int(order[p]) for p, _ in prior.PAIRS]
                    and pairs[row]['negative'] == [int(order[n]) for _, n in prior.PAIRS],
                    'frozen teacher pair replay')
            lex, _, lw = old.sparse_scores(ld, lq[row])
            result = dict(query_id=records[index]['query_id'], domain=records[index]['subset'],
                          rankings={'dense': prior.compact_rank(teacher, labels[index])},
                          pair_loss={}, costs={})
            for name, keep in masks.items():
                sem, _, work = old.sparse_scores(sd, prior.masked_query(sq[row], keep))
                # Both cached query channels already contain the fixed 0.9/0.1 weights.
                score = sem + lex
                result['rankings'][name] = prior.compact_rank(score, labels[index])
                result['pair_loss'][name] = joint_loss(score, pairs[row], design['tau'])
                result['costs'][name] = dict(semantic_df_work=work, lexical_df_work=lw)
            results.append(result)
            stream.write(json.dumps(result, allow_nan=False) + '\n')
            if row == 31:
                estimate = 1.5 * (time.monotonic() - started) * len(pairs) / 32 + 300
                campaign.io.write(output / 'canary.json', dict(projected_seconds=estimate))
                require(estimate < 5100, 'replay cannot fit bound')
            if row % 64 == 0:
                print('NG83_TRAIN_REPLAY', row + 1, flush=True)
    losses = {p: float(np.mean([r['pair_loss'][p] for r in results])) for p in PROFILES}
    np.testing.assert_allclose(losses['S'], design['parent_train_loss'], rtol=0, atol=1e-12)
    for p in PROFILES:
        np.testing.assert_allclose(losses[p] - losses['S'],
            design['profiles'][p]['joint_train_loss_delta'], rtol=0, atol=1e-12)
    diagnostic = campaign.rows(files['diagnostic'])
    return dict(quality=prior.summarize(results, 'rankings'), joint_pair_loss=losses,
        versus_S={p: prior.paired(results, p, 'S') for p in PROFILES[1:]},
        diagnostic_versus_S={p: prior.paired(diagnostic, p, 'S') for p in PROFILES[1:]},
        queries=len(results), full_corpus_documents=len(documents),
        mask_refits=0, optimizer_updates=0, independent_holdout=False)


def supervision_inventory(records, labels, selection):
    normalized = data.normalized
    by_text, by_positive = defaultdict(list), defaultdict(set)
    roles = {i: role for role in ('train', 'diagnostic') for i in selection[role]}
    require(len(roles) == sum(len(selection[r]) for r in ('train', 'diagnostic')),
            'TRAIN/diagnostic intersection')
    rows = []
    for index in sorted(roles):
        row, role = records[index], roles[index]
        text = normalized(row['query'])
        digest = hashlib.sha256(text.encode()).hexdigest()
        by_text[digest].append((role, row['query_id']))
        for doc in labels[index]:
            by_positive[doc].add(role)
        first = text.split()[0] if text else ''
        rows.append(dict(query_id=row['query_id'], domain=row['subset'], role=role,
                         normalized_sha256=digest, words=len(text.split()),
                         gold_count=len(labels[index]), first_token=first))
    domains = {}
    for domain in old.DOMAINS:
        domains[domain] = {}
        for role in ('train', 'diagnostic'):
            group = [r for r in rows if r['domain'] == domain and r['role'] == role]
            words = [r['words'] for r in group]
            domains[domain][role] = dict(queries=len(group),
                normalized_unique=len({r['normalized_sha256'] for r in group}),
                word_count_quantiles=np.quantile(words, [0, .5, .9, 1]).tolist(),
                query_initial_tokens=dict(Counter(r['first_token'] for r in group).most_common(20)))
    overlap = sum(len(v) > 1 for v in by_positive.values())
    duplicates = {k: v for k, v in by_text.items() if len(v) > 1}
    return rows, dict(domains=domains, exact_normalized_duplicates=duplicates,
        cross_split_duplicate_groups=sum(len({r for r, _ in v}) > 1 for v in by_text.values()),
        positive_document_overlap_train_diagnostic=overlap,
        independent_holdout=False, source_disjoint=False, topic_disjoint_verified=False,
        near_duplicate_audit_completed=False, first_token_is_intent_label=False,
        human_negative_labels_available=False, new_supervision_downloaded=0)


def supervision(files, output):
    records, labels = campaign.read(files['records']), campaign.read(files['labels'])
    rows, result = supervision_inventory(records, labels, campaign.read(files['split']))
    require(result['cross_split_duplicate_groups'] == 0, 'cross-split normalized duplicate')
    with (output / 'query-inventory.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    return result


def worker(args):
    from clearml import Task
    files = verify(args.research_root, args.run)
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0083/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=old.TRACKING_FRAMEWORKS.copy(),
        auto_connect_arg_parser=False, auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    campaign.io.write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(input_sha256=campaign.io.sha(args.run / 'inputs.json'), phase=args.phase))
        result = (replay if args.phase == 'train-replay' else supervision)(files, output)
        verify(args.research_root, args.run)
        campaign.io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        campaign.io.write(output / 'clearml.json', receipt)


def supervise(args):
    import fcntl
    import os
    import signal

    verify(args.research_root, args.run)
    require(campaign.io.sha(Path(__file__)) == campaign.read(args.run / 'inputs.json')[
            'source']['ng83_audit.py'], 'frozen controller required')
    name, stopped = args.phase, []
    require(name in PHASES and not (args.run / (name + '-controller-start.json')).exists(),
            'fresh lane required')
    handlers = {s: signal.signal(s, lambda *_: stopped.append(True))
                for s in (signal.SIGINT, signal.SIGTERM)}
    args.worker_script = 'ng83_audit.py'
    result = dict(status='failed_preserve_attempt', completed=[])
    try:
        with (args.research_root / '.ng80-cpu.lock').open('a') as shared:
            fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with (args.research_root / ('.ng83-' + name + '.lock')).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                campaign.io.write(args.run / (name + '-controller-start.json'),
                    dict(pid=os.getpid(), started_unix=time.time(), physical_gpu=None))
                campaign.guard.phase(args, name, stopped, cuda=False, limit_seconds=5400)
                result = dict(status='all_phases_complete', completed=[name])
    except BaseException as exc:
        result['error'] = str(exc)
        raise
    finally:
        result['finished_unix'] = time.time()
        campaign.io.write(args.run / (name + '-controller-exit.json'), result)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'worker', 'supervise'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    elif args.mode == 'verify':
        verify(args.research_root, args.run)
    elif args.mode == 'worker':
        require(args.phase in PHASES, 'phase required')
        worker(args)
    else:
        supervise(args)
