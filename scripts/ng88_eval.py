"""NG88 frozen full-background evaluation; Lambda2 physical GPU 0 only."""

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from ng85_data import DOMAINS, read, require, rows, sha, write
from ng85_run import closed
from ng88_math import (PROFILES, audit_scores, check_rank, distribution,
                       normalize, normalized_query, paired, prune, ranked, score)


PARENT = 'NG-0087/clean-base-ab-v1'
PARENT_SHA = '27d5965fc24524ca6ab90c7bcf60d10b931d029aaa9f8357990e646296f7b361'
DATA = 'NG-0085/student-ab-v4'
GPU_UUID = 'GPU-a5759f0b-0e22-e3b7-8243-7945961ba2e8'
ARMS = ('base', 'A', 'B')
CORPUS = 290609
SHARDS = 2
DIMENSIONS = 50265
ALPHAS = (0., .1, .25, .5, .75, .9, 1.)
PHASES = tuple(p for arm in ARMS for p in (
    *(f'encode-{arm}-{i}' for i in range(SHARDS)), f'encode-{arm}-queries',
    *(f'rank-{arm}-{profile}' for profile in (('full',) if arm == 'base' else PROFILES)),
)) + ('baseline-calibrate', 'baseline-rank', 'review')
PHASE_CAP = 5400
CAMPAIGN_CAP = 24 * 3600


def panels(queries, previous):
    require(Counter(q['split'] for q in queries) == dict(fit=16384, validation=2048),
            'only sealed fit/validation inventory admitted')
    require(len({q['query_id'] for q in queries}) == len(queries), 'duplicate query IDs')
    validation = [i for i, q in enumerate(queries) if q['split'] == 'validation']
    calibration = [i for i in previous['endpoint_panel'] if queries[i]['split'] == 'fit']
    require(len(calibration) == len(set(calibration)) == 128
            and Counter(queries[i]['subset'] for i in validation) == dict.fromkeys(DOMAINS, 512)
            and Counter(queries[i]['subset'] for i in calibration) == dict.fromkeys(DOMAINS, 32),
            'balanced fixed panels required')
    return dict(validation=validation, calibration=calibration)


def freeze(args):
    from ng87_run import verify as verify87

    base, run = args.research_root, args.run
    require(run.parent == base / 'NG-0088' and not run.exists(), 'fresh NG88 run required')
    parent = base / PARENT
    require(sha(parent / 'inputs.json') == PARENT_SHA, 'wrong NG87 parent')
    previous = verify87(SimpleNamespace(base=base, parent=base / DATA, out=parent))
    require(read(parent / 'campaign-exit.json')['passed'], 'NG87 campaign incomplete')
    for phase in ('train-A', 'train-B', 'profile-base', 'profile-A', 'profile-B', 'review'):
        closed(parent, phase)
    for phase in ('prepare', 'dense', 'bank'):
        closed(base / DATA, phase)
    config = read(parent / 'config.json')
    selected = panels(rows(base / DATA / 'prepare/queries.jsonl'), previous)
    deps = {}
    for phase in ('prepare', 'dense', 'bank'):
        root = base / DATA / phase
        receipt = read(root / 'complete.json')
        for name in receipt['files']:
            if not name.startswith('tracking-cache/'):
                deps[str((root / name).relative_to(base))] = receipt['files'][name]
        deps[str((root / 'complete.json').relative_to(base))] = sha(root / 'complete.json')
    for arm in ARMS:
        profile = parent / f'profile-{arm}'
        for name in ('queries.jsonl', 'complete.json'):
            deps[str((profile / name).relative_to(base))] = sha(profile / name)
        checkpoint = (base / config['base']['checkpoint'] if arm == 'base'
                      else parent / f'train-{arm}/checkpoint-08192/model')
        for path in checkpoint.rglob('*'):
            if path.is_file():
                deps[str(path.relative_to(base))] = sha(path)
    for path in (base / config['base']['tokenizer']).rglob('*'):
        if path.is_file():
            deps[str(path.relative_to(base))] = sha(path)
    for path in (parent / 'inputs.json', parent / 'campaign-exit.json',
                 parent / 'review/results.json', parent / 'train-A/results.json',
                 parent / 'train-B/results.json',
                 base / config['base']['rms'], base / 'NG-0002/checkpoint_io.py'):
        deps[str(path.relative_to(base))] = sha(path)
    run.mkdir(parents=True)
    for name in previous['source']:
        if name.endswith('.py'):
            shutil.copy2(parent / name, run / name)
    for name in ('ng88_eval.py', 'ng88_math.py', 'ng71_pilot.py',
                 'test_ng88.py', 'ng0088-execution.zh.md'):
        shutil.copy2(Path(__file__).parent / name, run / name)
    write(run / 'config.json', dict(experiment='NG-0088', training_enabled=False,
                                   base=config['base'], optimizer=config['optimizer']))
    write(run / 'selection.json', selected)
    write(run / 'inputs.json', dict(protocol='NG88_full_background_gpu0_v1',
        phases=list(PHASES), parent=PARENT, parent_sha256=PARENT_SHA,
        source={p.name: sha(p) for p in run.iterdir() if p.is_file()}, dependencies=deps,
        gpu=0, gpu_uuid=GPU_UUID, profiles=PROFILES, corpus=CORPUS, shards=SHARDS,
        phase_cap=PHASE_CAP, campaign_cap=CAMPAIGN_CAP, dense_alphas=ALPHAS,
        training=False, locked_test_access=False, production_qualified=False,
        validation_independent=False, automatic_training=False,
        original_text_dense_evaluation='pending_separate_protocol'))
    verify(base, run)
    print('NG88_FROZEN', sha(run / 'inputs.json'), flush=True)


def verify(base, run):
    m = read(run / 'inputs.json')
    require(m['protocol'] == 'NG88_full_background_gpu0_v1'
            and m['phases'] == list(PHASES) and m['gpu'] == 0 and m['gpu_uuid'] == GPU_UUID
            and m['parent'] == PARENT and m['parent_sha256'] == PARENT_SHA
            and m['profiles'] == {k: list(v) for k, v in PROFILES.items()}
            and m['corpus'] == CORPUS and m['shards'] == SHARDS
            and m['phase_cap'] == PHASE_CAP and m['campaign_cap'] == CAMPAIGN_CAP
            and m['dense_alphas'] == list(ALPHAS)
            and not any(m[k] for k in ('training', 'locked_test_access',
                                      'production_qualified', 'automatic_training',
                                      'validation_independent'))
            and m['original_text_dense_evaluation'] == 'pending_separate_protocol',
            'frozen NG88 contract changed')
    for root, entries in ((run, m['source']), (base, m['dependencies'])):
        for name, digest in entries.items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve()) and sha(path) == digest,
                    'changed input: ' + name)
    require(read(run / 'selection.json') == panels(rows(base / DATA / 'prepare/queries.jsonl'),
            read(base / PARENT / 'inputs.json')), 'selection changed')
    return m


def raw_encode(encoder, texts, role):
    import torch

    size = encoder.config['optimizer'][role + '_microbatch']
    limit = encoder.config['base'][role + '_input_tokens']
    result = []
    for start in range(0, len(texts), size):
        tokens = encoder.tokenizer(texts[start:start + size], truncation=False,
            padding=True, return_tensors='pt').to(encoder.device)
        require(tokens.input_ids.shape[1] <= limit, 'unexpected further truncation')
        hidden = encoder.model.base_model(**tokens).last_hidden_state
        values = encoder.model.lm_head(hidden).relu().log1p()
        values = values.masked_fill(~tokens.attention_mask.bool()[..., None], 0).amax(1)
        require(torch.isfinite(values).all() and (values.sum(1) > 0).all(), 'invalid raw code')
        result.append(values)
    return torch.cat(result)


def query_readout(encoder, raw):
    import torch

    # Retain the original microbatch FP32 GEMV operation order, not GEMM.
    size = encoder.config['optimizer']['query_microbatch']
    weight = encoder.config['base']['semantic_weight']
    parts = []
    for values in raw.split(size):
        denominator = values @ encoder.rms
        require(torch.isfinite(denominator).all() and (denominator > 0).all(),
                'invalid query normalization')
        parts.append(weight * values / denominator[:, None])
    return torch.cat(parts)


def eta(folder, start, completed, total, fixed_seconds=0.):
    elapsed = time.monotonic() - start
    estimate = fixed_seconds + elapsed + 1.5 * elapsed / completed * (total - completed) + 180
    write(folder / 'eta.json', dict(completed=completed, total=total,
        elapsed_seconds=elapsed, guarded_phase_seconds=estimate, cap=PHASE_CAP))
    require(estimate < PHASE_CAP, 'phase ETA exceeds frozen budget')


def encode(args, queries, documents, task):
    import torch
    from ng71_training import Encoder, parameter_hash

    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.device_count() == 1,
            'physical GPU 0 only')
    _, arm, part = args.phase.split('-')
    base, run, folder = args.research_root, args.run, args.run / args.phase
    config = read(run / 'config.json')
    require(config['base']['semantic_weight'] == .9, 'fixed semantic weight changed')
    require(len(documents) == CORPUS, 'document universe changed')
    checkpoint = None if arm == 'base' else base / PARENT / f'train-{arm}/checkpoint-08192/model'
    begin = time.monotonic()
    encoder = Encoder(base, config, 'cuda', checkpoint=checkpoint)
    expected = (config['base']['state_sha256'] if arm == 'base' else
                read(base / PARENT / f'train-{arm}/results.json')['final_sha256'])
    require(parameter_hash(encoder.model) == expected, 'encoder state mismatch')
    role = 'query' if part == 'queries' else 'document'
    selection = read(run / 'selection.json')
    ids = (selection['calibration'] + selection['validation'] if role == 'query'
           else list(range(CORPUS * int(part) // SHARDS, CORPUS * (int(part) + 1) // SHARDS)))
    texts = [queries[i]['query'] for i in ids] if role == 'query' else [documents[i]['text'] for i in ids]
    loaded = time.monotonic() - begin
    blocks, normalized, tick = [], [], time.monotonic()
    batch = 8 if role == 'query' else 64
    with torch.inference_mode():
        for start in range(0, len(ids), batch):
            current = texts[start:start + batch]
            raw = raw_encode(encoder, current, role)
            if start == 0:
                original = encoder.encode(current, role)
                candidate = raw if role == 'document' else query_readout(encoder, raw)
                require(torch.equal(candidate, original), 'raw readout differs from frozen encoder')
                write(folder / 'readout-parity.json', dict(exact=True, role=role))
            blocks.append(sparse.csr_matrix(raw.cpu().numpy()))
            if role == 'query':
                value = query_readout(encoder, raw)
                require(torch.isfinite(value).all(), 'invalid query normalization')
                normalized.append(sparse.csr_matrix(value.cpu().numpy()))
            completed = min(start + batch, len(ids))
            if completed == min(512, len(ids)) or completed == len(ids):
                eta(folder, tick, completed, len(ids), loaded)
            if start % (32 * batch) == 0:
                write(folder / 'progress.json', dict(role=role, completed=completed,
                    total=len(ids), seconds=time.monotonic() - tick))
                task.get_logger().report_scalar('encode', role, completed, completed)
                print('NG88_ENCODE', arm, part, completed, len(ids), flush=True)
    matrix = sparse.vstack(blocks, format='csr')
    del blocks
    require(matrix.shape == (len(ids), DIMENSIONS) and matrix.has_canonical_format,
            'encoded matrix shape/format')
    sparse.save_npz(folder / 'raw.npz', matrix)
    if role == 'query':
        sparse.save_npz(folder / 'normalized.npz', sparse.vstack(normalized, format='csr'))
    require(parameter_hash(encoder.model) == expected, 'read-only encoding mutated model')
    write(folder / 'ids.json', ids)
    return dict(passed=True, arm=arm, role=role, rows=len(ids), nnz=int(matrix.nnz),
        nnz_distribution=distribution(np.diff(matrix.indptr)), model_sha256=expected,
        seconds=time.monotonic() - begin, optimizer_updates=0, physical_gpu=0)


def lexical(base):
    root = base / DATA / 'bank'
    return (sparse.load_npz(root / 'bm25.npz').tocsc(),
            sparse.load_npz(root / 'lexical-queries.npz'))


def record_query(index, query, semantic, lex, sd, sq, ld, lq):
    mixed = semantic + lex
    hybrid, pure = ranked(mixed, query['positives']), ranked(semantic, query['positives'])
    check = sorted(set(hybrid['top100'] + pure['top100'] + query['positives']))
    sem_error = audit_scores(sd, sq, semantic, check)
    lex_error = audit_scores(ld, lq, lex, check)
    return dict(query_index=index, query_id=query['query_id'], subset=query['subset'],
        hybrid=hybrid, semantic=pure, selected_score_error=max(sem_error, lex_error),
        semantic_union_documents=int(np.count_nonzero(semantic)),
        combined_union_documents=int(np.count_nonzero(mixed)))


def rank(args, queries, task):
    _, arm, profile = args.phase.split('-')
    base, run, folder = args.research_root, args.run, args.run / args.phase
    started = time.monotonic()
    pieces = []
    for shard in range(SHARDS):
        phase = f'encode-{arm}-{shard}'
        closed(run, phase)
        ids = read(run / phase / 'ids.json')
        require(ids == list(range(CORPUS * shard // SHARDS, CORPUS * (shard + 1) // SHARDS)),
                'corpus shard order mismatch')
        pieces.append(sparse.load_npz(run / phase / 'raw.npz'))
    documents = sparse.vstack(pieces, format='csr')
    del pieces
    gc.collect()
    kq, kd = PROFILES[profile]
    documents = prune(documents, kd)
    nnz_stats = distribution(np.diff(documents.indptr))
    sd = documents.tocsc()
    del documents
    closed(run, f'encode-{arm}-queries')
    query_root = run / f'encode-{arm}-queries'
    ids = read(query_root / 'ids.json')
    require(ids == read(run / 'selection.json')['calibration']
            + read(run / 'selection.json')['validation'], 'query row identity')
    if kq is None:
        sq = sparse.load_npz(query_root / 'normalized.npz')
    else:
        rms = np.load(base / read(run / 'config.json')['base']['rms'])
        sq = normalized_query(sparse.load_npz(query_root / 'raw.npz'), rms, kq)
    positions = {index: i for i, index in enumerate(ids)}
    ld, lq = lexical(base)
    anchors = {r['query_index']: r for r in rows(base / PARENT / f'profile-{arm}/queries.jsonl')
               if r['checkpoint'] == ('base' if arm == 'base' else '8192')}
    chosen = read(run / 'selection.json')['validation']
    loaded, tick, max_anchor = time.monotonic() - started, time.monotonic(), 0.
    with (folder / 'rankings.jsonl').open('x') as stream:
        for count, index in enumerate(chosen, 1):
            query = sq[positions[index]]
            semantic, sw = score(sd, query)
            lex, lw = score(ld, lq[index])
            entry = record_query(index, queries[index], semantic, lex, sd, query, ld, lq[index])
            if profile == 'full':
                prior = anchors[index]
                actual = (semantic + lex)[prior['document_ids']]
                error = float(np.max(np.abs(actual - prior['scores'])))
                require(np.allclose(actual, prior['scores'], rtol=1e-5, atol=1e-5),
                        'candidate raw score parity failed')
                max_anchor = max(max_anchor, error)
            entry.update(semantic_df_work=sw, lexical_df_work=lw, combined_df_work=sw + lw,
                         query_nnz=int(query.nnz))
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
            if count == 16:
                eta(folder, tick, count, len(chosen), loaded)
            if count % 64 == 0:
                stream.flush()
                write(folder / 'progress.json', dict(completed=count, total=len(chosen)))
                task.get_logger().report_scalar('rank', profile, count, count)
                print('NG88_RANK', arm, profile, count, len(chosen), flush=True)
    return dict(passed=True, arm=arm, profile=profile, queries=len(chosen), documents=CORPUS,
        semantic_postings=int(sd.nnz), lexical_postings=int(ld.nnz),
        document_nnz=nnz_stats, df=distribution(np.diff(sd.indptr)),
        max_candidate_anchor_error=max_anchor if profile == 'full' else None,
        native_cost_evaluated=False, full_beir_evaluated=False)


def baseline(args, queries, task):
    base, run, folder = args.research_root, args.run, args.run / args.phase
    calibration = args.phase == 'baseline-calibrate'
    selected = read(run / 'selection.json')['calibration' if calibration else 'validation']
    q = np.load(base / DATA / 'dense/query.npy', mmap_mode='r')
    d = np.load(base / DATA / 'dense/document.npy', mmap_mode='r').astype(np.float64)
    require(d.shape == (CORPUS, 1024) and q.shape == (len(queries), 1024), 'dense row universe')
    ld, lq = lexical(base)
    alpha = None if calibration else closed(run, 'baseline-calibrate')['alpha']
    metrics, tick = [], time.monotonic()
    with (folder / 'rankings.jsonl').open('x') as stream:
        for count, index in enumerate(selected, 1):
            dense = d @ q[index].astype(np.float64)
            lex, work = score(ld, lq[index])
            results = {}
            choices = ALPHAS if calibration else (alpha,)
            for a in choices:
                results[str(a)] = ranked((1 - a) * normalize(dense) + a * normalize(lex),
                                         queries[index]['positives'])
            entry = dict(query_index=index, query_id=queries[index]['query_id'],
                subset=queries[index]['subset'], dense=ranked(dense, queries[index]['positives']),
                bm25=ranked(lex, queries[index]['positives']), mixtures=results,
                lexical_df_work=work)
            check = sorted(set(entry['dense']['top100'] + queries[index]['positives']))
            alternative = np.einsum('ij,j->i', d[check], q[index].astype(np.float64), optimize=False)
            require(np.allclose(dense[check], alternative, rtol=0, atol=1e-12), 'dense score audit')
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
            metrics.append([[results[str(a)]['ndcg10'], results[str(a)]['recall100']] for a in choices])
            if count == 16:
                eta(folder, tick, count, len(selected))
            if count % 64 == 0:
                stream.flush()
                write(folder / 'progress.json', dict(completed=count, total=len(selected)))
                task.get_logger().report_scalar('baseline', 'queries', count, count)
    if calibration:
        means = np.mean(metrics, axis=0)
        best = max(range(len(ALPHAS)), key=lambda i: (*means[i], -ALPHAS[i]))
        alpha = ALPHAS[best]
    return dict(passed=True, alpha=alpha, rows=len(selected), train_only_calibration=True,
        matched_prefix_only=True, original_text_dense_pending=True,
        dense_hybrid_normalization='offline_full_corpus_zscore_not_serving_route')


def review(args, queries):
    run, values, summaries, costs = args.run, {}, {}, {}
    chosen = read(run / 'selection.json')['validation']
    domains = [queries[i]['subset'] for i in chosen]
    for phase in PHASES[:-1]:
        result = closed(run, phase)
        if not phase.startswith('rank-') and phase != 'baseline-rank':
            continue
        records = rows(run / phase / 'rankings.jsonl')
        require([r['query_index'] for r in records] == chosen, 'ranking panel mismatch')
        keys = ('hybrid', 'semantic') if phase.startswith('rank-') else ('dense', 'bm25', 'mixture')
        for key in keys:
            items = [r[key] if key != 'mixture' else next(iter(r['mixtures'].values())) for r in records]
            for record, index in zip(items, chosen, strict=True):
                require(record['gold_ids'] == queries[index]['positives'], 'positive IDs changed')
                check_rank(record, CORPUS)
            name = phase + '/' + key
            values[name] = items
            summaries[name] = {metric: float(np.mean([r[metric] for r in items]))
                               for metric in ('ndcg10', 'recall10', 'recall100')}
            summaries[name]['by_domain'] = {domain: {metric: float(np.mean([
                r[metric] for r, d in zip(items, domains, strict=True) if d == domain]))
                for metric in ('ndcg10', 'recall100')} for domain in DOMAINS}
        if phase.startswith('rank-'):
            costs[phase] = dict(semantic_postings=result['semantic_postings'],
                lexical_postings=result['lexical_postings'], document_nnz=result['document_nnz'],
                **{k: distribution([r[k] for r in records]) for k in (
                    'semantic_df_work', 'combined_df_work', 'query_nnz',
                    'semantic_union_documents', 'combined_union_documents')})
    reference = values['rank-base-full/hybrid']
    comparisons = {name: {metric: paired([r[metric] for r in reference],
        [r[metric] for r in items], domains, 88001 + n)
        for n, metric in enumerate(('ndcg10', 'recall100'))} for name, items in values.items()}
    a = comparisons['rank-A-full/hybrid']
    eligible = (a['ndcg10']['delta'] >= .005
                and a['ndcg10']['descriptive_interval_95'][0] > 0
                and a['recall100']['delta'] >= -.002)
    return dict(passed=True, meaning='completed evidence audit, not product qualification',
        metrics=summaries, costs=costs, comparisons_to_base=comparisons,
        A_full_background_quality_signal=eligible,
        next_action='freeze_bounded_R_P_training' if eligible else 'stop_and_review_transfer',
        automatic_training_started=False, production_qualified=False,
        independent_validation=False, native_cost_evaluated=False, normal_input_dense_pending=True)


def worker(args):
    from clearml import Task

    verify(args.research_root, args.run)
    require(args.phase in PHASES, 'unknown phase')
    if args.phase.startswith('encode-'):
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(88001)
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0088/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    folder = args.run / args.phase
    receipt = dict(task_id=task.id, offline=True, remote_synced=False, closed=False)
    write(folder / 'clearml-start.json', receipt)
    try:
        task.connect(dict(inputs_sha256=sha(args.run / 'inputs.json'), phase=args.phase, gpu_policy=0))
        queries = rows(args.research_root / DATA / 'prepare/queries.jsonl')
        if args.phase.startswith('encode-'):
            result = encode(args, queries, rows(args.research_root / DATA / 'prepare/documents.jsonl'), task)
        elif args.phase.startswith('rank-'):
            result = rank(args, queries, task)
        elif args.phase.startswith('baseline-'):
            result = baseline(args, queries, task)
        else:
            result = review(args, queries)
        write(folder / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        write(folder / 'clearml.json', receipt)


def gpu_contract(gpu, snapshot):
    require(gpu == 0 and snapshot['gpu'] == 0 and snapshot['uuid'] == GPU_UUID,
            'II42 temporarily permits physical GPU 0 only')


def supervise(args):
    import fcntl
    from ng71_pilot import phase
    from ng85_queue import check_resources, gpu_snapshot

    require(args.gpu == 0, 'GPU 0 only; no alternate GPU fallback')
    verify(args.research_root, args.run)
    require(not (args.run / 'controller-start.json').exists(), 'immutable run already started')
    stopped, completed = [], []
    handlers = {s: signal.signal(s, lambda *_: stopped.append(True))
                for s in (signal.SIGTERM, signal.SIGINT)}
    args.worker_script = 'ng88_eval.py'
    start, error = time.monotonic(), None
    with (args.research_root / '.ng-gpu-0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_contract(args.gpu, gpu_snapshot(0))
        check_resources(args.run, 'campaign', 0)
        write(args.run / 'controller-start.json', dict(pid=os.getpid(), gpu=0,
            started_unix=time.time(), inputs_sha256=sha(args.run / 'inputs.json')))
        try:
            for name in PHASES:
                require(not stopped and time.monotonic() - start < CAMPAIGN_CAP,
                        'campaign stopped or time bound exceeded')
                cuda = name.startswith('encode-')
                if cuda:
                    gpu_contract(0, gpu_snapshot(0))
                    check_resources(args.run, name, 0)
                write(args.run / 'progress.json', dict(phase=name, completed=completed,
                    updated_unix=time.time(), gpu=0 if cuda else None))
                phase(args, name, stopped, cuda=cuda,
                      limit_seconds=min(PHASE_CAP, max(1, int(CAMPAIGN_CAP - (time.monotonic() - start)))))
                closed(args.run, name)
                completed.append(name)
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            write(args.run / 'controller-exit.json', dict(passed=error is None,
                completed=completed, error=error, finished_unix=time.time(),
                automatic_training=False, automatic_retry=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0,))
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    if args.mode == 'freeze':
        freeze(args)
    elif args.mode == 'verify':
        verify(args.research_root, args.run)
    elif args.mode == 'supervise':
        supervise(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
