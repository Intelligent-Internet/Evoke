#!/usr/bin/env python3
"""Frozen, bounded breadth training and full-background A/B evaluation."""

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import psutil


SEEDS = (59059, 66061, 66067)
HOST_BYTES = 16 * 1024 ** 3
GPU_BYTES = 20 * 1024 ** 3


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def order(seed, quarter):
    assert seed in SEEDS and quarter in (1, 2, 3, 4)
    values = np.random.default_rng(seed).permutation(6144)
    return values[(quarter - 1) * 1536:quarter * 1536].tolist()


def schedule():
    trains = [f'train-{seed}-{quarter}' for seed in SEEDS
              for quarter in (1, 2, 3, 4)]
    models = ['dense', 'initial'] + [f'{arm}-{seed}' for seed in SEEDS
                                   for arm in ('A', 'B')]
    return trains + ['rank-bm25'] + [f'{action}-{model}' for model in models
                                    for action in ('encode', 'rank')]


def verify(base, run):
    frozen = read(run / 'inputs.json')
    assert sha(Path(__file__)) == frozen['source_sha256'], 'Pipeline source changed'
    assert sha(run / 'protocol.md') == frozen['protocol_sha256']
    assert sha(run / 'research-protocol.md') == frozen['research_protocol_sha256']
    for name, digest in frozen['files'].items():
        assert sha(base / name) == digest, name
    if 'inherited' in frozen:
        inherited = frozen['inherited']
        parent = base / inherited['parent']
        assert sha(run / 'continuation-protocol.md') == inherited['protocol_sha256']
        for name, digest in inherited['control_files'].items():
            assert sha(parent / name) == digest, name
        for name in inherited['phases']:
            link = run / name
            assert link.is_symlink() and link.resolve() == (parent / name).resolve()
    return frozen


def inherit_successes(base, run, parent, frozen):
    """Reference sealed successes without copying or modifying the failed attempt."""
    assert parent != run and parent.parent == run.parent
    old = read(parent / 'inputs.json')
    assert old['files'] == frozen['files'] and old['phases'] == schedule()
    assert old['seeds'] == list(SEEDS) and 'inherited' not in old
    controls = ['inputs.json', 'ng69_pipeline.py', 'protocol.md',
                'research-protocol.md', 'continuation-state.json',
                'rank-dense/exit.json', 'rank-dense/clearml.json',
                'rank-dense/canary.json']
    assert sha(parent / 'ng69_pipeline.py') == old['source_sha256']
    for name, field in (('protocol.md', 'protocol_sha256'),
                        ('research-protocol.md', 'research_protocol_sha256')):
        assert sha(parent / name) == old[field] == frozen[field]
    assert read(parent / 'continuation-state.json')['status'] == 'failed_preserve_attempt'
    failure = read(parent / 'rank-dense/exit.json')
    assert failure['exit_code'] == 1 and failure['owned_group_closed']
    assert read(parent / 'rank-dense/clearml.json')['closed']
    assert not read(parent / 'rank-dense/canary.json')['passed']
    assert not (parent / 'rank-dense/complete.json').exists()
    phases = schedule()[:14]
    assert phases[-2:] == ['rank-bm25', 'encode-dense']
    assert not any((run / name).exists() or (run / name).is_symlink()
                   for name in schedule()), 'Continuation must start empty'
    for name in phases:
        assert not (parent / name).is_symlink()
        completed(parent, name)
        controls.extend(f'{name}/{file}' for file in
                        ('complete.json', 'exit.json', 'clearml.json'))
    inheritance = {
        'parent': str(parent.relative_to(base)), 'phases': phases,
        'control_files': {name: sha(parent / name) for name in controls},
        'protocol_sha256': sha(run / 'continuation-protocol.md'),
        'training_reexecuted': False,
    }
    for name in phases:
        (run / name).symlink_to(os.path.relpath(parent / name, run),
                               target_is_directory=True)
    return inheritance


def completed(run, phase):
    path = run / phase
    exit_record = read(path / 'exit.json')
    assert exit_record['exit_code'] == 0 and exit_record['owned_group_closed']
    assert exit_record['error'] is None
    record = read(path / 'complete.json')
    assert record['passed'] and read(path / 'clearml.json')['closed']
    for name, digest in record['files'].items():
        assert sha(path / name) == digest, name
    return read(path / 'results.json')


def legacy(base):
    for relative in ('NG-0059', 'NG-0065', 'NG-0002',
                     'NG-0059/evaluation-v1', 'NG-0052/cleanup-followup'):
        sys.path.insert(0, str(base / relative))
    import common59 as common
    assert common.BASE.resolve() == base.resolve()
    return common


def validate_example(query, labels, pool, lexical, target):
    assert query['split'] == 'TRAIN'
    assert query['query_id'] == target['query_id']
    assert pool == target['pool'] and len(pool) == len(set(pool))
    assert len(pool) == len(lexical) == len(target['target'])
    assert len(pool) == len(target['positive_mask'])
    assert set(labels) == {d for d, positive in
                           zip(pool, target['positive_mask']) if positive}
    assert labels and all(0 <= d < 233009 for d in pool)
    assert np.isfinite(lexical).all()
    assert np.isfinite(target['target']).all()
    assert min(target['target']) >= 0
    assert abs(sum(target['target']) - 1) < 1e-6


def freeze(args):
    assert not (args.run / 'inputs.json').exists()
    base = args.research_root
    files = {}
    for stage, manifest in (
            ('NG-0069/teacher-v1', 'complete.json'),
            ('NG-0069/lexical-v1', 'complete.json')):
        record = read(base / stage / manifest)
        assert record['passed']
        files[stage + '/' + manifest] = sha(base / stage / manifest)
        for name, digest in record['files'].items():
            assert sha(base / stage / name) == digest
            files[stage + '/' + name] = digest
    audit = read(base / 'NG-0069/teacher-audit-v1.json')
    assert audit['passed'] and audit['queries_audited'] == 6144
    assert audit['complete_sha256'] == files['NG-0069/teacher-v1/complete.json']
    teacher = base / 'NG-0069/teacher-v1'
    assert audit['inputs_sha256'] == sha(teacher / 'inputs.json')
    exited = read(teacher / 'exit.json')
    assert exited['exit_code'] == 0 and exited['owned_group_closed']
    assert exited['error'] is None and read(teacher / 'clearml.json')['closed']
    files['NG-0069/teacher-v1/exit.json'] = sha(teacher / 'exit.json')
    files['NG-0069/teacher-audit-v1.json'] = sha(base / 'NG-0069/teacher-audit-v1.json')
    lexical = base / 'NG-0069/lexical-v1'
    for name, digest in read(lexical / 'inputs.json')['files'].items():
        assert sha(base / name) == digest, name
    queries, labels, pools, scores = [read(lexical / (name + '.json'))
        for name in ('queries', 'labels', 'train-pools', 'train-lexical')]
    targets = read(teacher / 'targets.json')
    assert len(queries) == len(labels) == 7872
    assert len(pools) == len(scores) == len(targets) == 6144
    assert len({q['query_id'] for q in queries}) == 7872
    assert [q['split'] for q in queries] == (
        ['TRAIN'] * 6144 + ['DEV_NEW'] * 1536 + ['DEV_EXPOSED'] * 192)
    for i in range(6144):
        validate_example(queries[i], labels[i], pools[i], scores[i], targets[i])
    refs = read(base / 'NG-0059/dependencies.json')['files']
    for name, digest in refs.items():
        if any(key in name for key in ('checkpoint-096/', '/model/', 'dense-model/',
                                      'mlm-raw-rms.npy', 'checkpoint_io.py',
                                      'lifecycle52.py')):
            relative = os.path.relpath(base / 'NG-0059' / name, base)
            assert sha(base / relative) == digest
            files[relative] = digest
    for name in ('NG-0059/common59.py', 'NG-0059/model59.py',
                 'NG-0059/evaluation-v1/metrics59.py', 'NG-0065/optimizer65.py',
                 'NG-0067/data/documents.jsonl',
                 'NG-0069/teacher-v1/inputs.json'):
        files[name] = sha(base / name)
    for seed in SEEDS:
        parent = base / f'NG-0066/attempt-v2/seed-{seed}/epoch-4/train'
        exited = read(parent / 'exit.json')
        assert exited['exit_code'] == 0 and exited['owned_group_closed']
        record = read(parent / 'complete.json')
        assert record['passed'] and read(parent / 'clearml.json')['closed']
        for name, digest in record['files'].items():
            path = base / 'NG-0066/attempt-v2' / name
            assert sha(path) == digest
            if '/checkpoint/' in name or name.endswith('results.json'):
                files[str(path.relative_to(base))] = digest
    frozen = {
        'files': files, 'source_sha256': sha(Path(__file__)),
        'protocol_sha256': sha(args.run / 'protocol.md'),
        'research_protocol_sha256': sha(args.run / 'research-protocol.md'),
        'seeds': SEEDS, 'phases': schedule(), 'locked_test_scored': False}
    if args.reuse_run is not None:
        frozen['inherited'] = inherit_successes(
            base, args.run, args.reuse_run.resolve(), frozen)
    write(args.run / 'inputs.json', frozen)


def train(args, task):
    import torch
    from model59 import Encoder, backward, optimizer
    from optimizer65 import restore, save

    c = legacy(args.research_root)
    _, seed, quarter = args.phase.split('-')
    seed, quarter = int(seed), int(quarter)
    path = args.run / args.phase
    lexical_root = args.research_root / 'NG-0069/lexical-v1'
    queries = read(lexical_root / 'queries.json')[:6144]
    documents = rows(args.research_root / 'NG-0067/data/documents.jsonl')
    targets = read(args.research_root / 'NG-0069/teacher-v1/targets.json')
    lexical_scores = read(lexical_root / 'train-lexical.json')
    old = f'train-{seed}-{quarter - 1}' if quarter > 1 else None
    previous = completed(args.run, old) if old else None
    source = args.run / old / 'checkpoint' if old else c.MODEL
    encoder = Encoder(False, source)
    initial = c.parameter_hash(encoder.model)
    assert initial == (previous['model_sha256'] if old else c.STATE_SHA)
    assert sum(p.numel() for p in encoder.model.parameters()) == 49648089
    opt = optimizer(encoder.model)
    before_opt = restore(args.run / old / 'optimizer.pt', encoder.model, opt,
                         (quarter - 1) * 384) if old else None
    if old:
        assert before_opt == previous['optimizer_sha256']
    indices = order(seed, quarter)
    write(path / 'query-order.json', indices)
    durations, query_tokens, document_tokens, pairs = [], 0, 0, 0
    with (path / 'progress.jsonl').open('x') as stream:
        for step in range(384):
            tick = time.monotonic()
            opt.zero_grad(set_to_none=True)
            examples = []
            for i in indices[step * 4:step * 4 + 4]:
                target = targets[i]
                assert target['query_id'] == queries[i]['query_id']
                text = [documents[j]['text'] for j in target['pool']]
                query = queries[i]['query']
                qt = len(encoder.tokenizer(query, truncation=True,
                                            max_length=64)['input_ids'])
                dt = sum(len(ids) for ids in encoder.tokenizer(
                    text, truncation=True, max_length=256)['input_ids'])
                query_tokens += qt
                document_tokens += dt
                pairs += len(text)
                result = backward(
                    encoder, query, text,
                    torch.tensor(lexical_scores[i], dtype=torch.float32, device='cuda'),
                    torch.tensor(target['target'], dtype=torch.float32, device='cuda'))
                result.update(query_id=target['query_id'], train_index=i,
                              query_tokens=qt, document_tokens=dt)
                examples.append(result)
            norm = torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.,
                                                  error_if_nonfinite=True)
            assert norm > 0
            before = [p.detach().clone() for p in encoder.model.parameters()]
            opt.step()
            update = sum(float((p.detach() - b).double().square().sum())
                         for p, b in zip(encoder.model.parameters(), before)) ** .5
            del before
            assert update > 0 and np.isfinite(update)
            assert torch.cuda.max_memory_allocated() < GPU_BYTES
            durations.append(time.monotonic() - tick)
            record = {'step': (quarter - 1) * 384 + step + 1,
                      'examples': examples, 'gradient_before_clip': float(norm),
                      'update_l2': update, 'seconds': durations[-1]}
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            task.get_logger().report_scalar('training', 'loss',
                float(np.mean([v['loss'] for v in examples])), record['step'])
            if step == 7:
                prediction = max(durations) * 384 * 1.5 + 180
                write(path / 'canary.json', {'passed': prediction < 1800,
                                            'predicted_seconds': prediction})
                assert prediction < 1800
            if step == 0 or (step + 1) % 32 == 0:
                print(args.phase, step + 1, '/384', flush=True)
    final = c.parameter_hash(encoder.model)
    assert final != initial
    opt_sha = save(path / 'optimizer.pt', encoder.model, opt, quarter * 384)
    encoder.model.save_pretrained(path / 'checkpoint', safe_serialization=True)
    qt = [queries[i]['query'] for i in indices[:2]]
    dt = [documents[j]['text'] for j in targets[indices[0]]['pool'][:4]]
    with torch.no_grad():
        qref = encoder.encode(qt, 'query').cpu()
        dref = encoder.encode(dt, 'document').cpu()
    del encoder, opt
    gc.collect()
    torch.cuda.empty_cache()
    encoder = Encoder(False, path / 'checkpoint')
    opt = optimizer(encoder.model)
    assert c.parameter_hash(encoder.model) == final
    assert restore(path / 'optimizer.pt', encoder.model, opt, quarter * 384) == opt_sha
    with torch.no_grad():
        assert torch.equal(qref, encoder.encode(qt, 'query').cpu())
        assert torch.equal(dref, encoder.encode(dt, 'document').cpu())
    return {'seed': seed, 'quarter': quarter, 'steps': 384,
            'cumulative_steps': quarter * 384, 'query_exposures': 1536,
            'model_sha256': final, 'initial_model_sha256': initial,
            'optimizer_sha256': opt_sha, 'initial_optimizer_sha256': before_opt,
            'checkpoint_optimizer_replay_exact': True,
            'query_tokens': query_tokens, 'document_tokens': document_tokens,
            'candidate_pairs': pairs,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}


def encode(args, task):
    import inspect
    from scipy import sparse
    import torch
    from model59 import Encoder
    from transformers import AutoModel, AutoTokenizer

    c = legacy(args.research_root)
    model_name = args.phase.removeprefix('encode-')
    path = args.run / args.phase
    documents = rows(args.research_root / 'NG-0067/data/documents.jsonl')
    queries = read(args.research_root / 'NG-0069/lexical-v1/queries.json')
    assert len(documents) == 233009 and len(queries) == 7872
    if model_name == 'dense':
        model, info = AutoModel.from_pretrained(
            c.DENSE, trust_remote_code=True, local_files_only=True,
            dtype=torch.float32, attn_implementation='sdpa', output_loading_info=True)
        assert all(not info.get(k) for k in (
            'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
        assert sha(Path(inspect.getfile(type(model)))) == sha(c.DENSE / 'modeling.py')
        model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            c.DENSE, trust_remote_code=True, local_files_only=True)
        expected = read(args.research_root / 'NG-0069/teacher-v1/results.json')[
            'model_state_sha256']

        @torch.no_grad()
        def perform(texts, role):
            tokens = tokenizer(texts, padding=True, truncation=True,
                max_length=64 if role == 'query' else 256,
                return_tensors='pt').to('cuda')
            hidden = model(**tokens, use_cache=False).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            value = (hidden * mask).sum(1) / mask.sum(1)
            assert torch.isfinite(value).all() and (value.norm(dim=1) > 0).all()
            return torch.nn.functional.normalize(value, dim=1).cpu().numpy()
    else:
        if model_name == 'initial':
            source, expected = c.MODEL, c.STATE_SHA
        elif model_name.startswith('A-'):
            seed = int(model_name[2:])
            source = args.research_root / (
                f'NG-0066/attempt-v2/seed-{seed}/epoch-4/train/checkpoint')
            expected = read(source.parent / 'results.json')['model_sha256']
        else:
            seed = int(model_name[2:])
            result = completed(args.run, f'train-{seed}-4')
            source = args.run / f'train-{seed}-4/checkpoint'
            expected = result['model_sha256']
        encoder = Encoder(False, source)
        model = encoder.model

        @torch.no_grad()
        def perform(texts, role):
            return encoder.encode(texts, role).cpu().numpy()

    assert c.parameter_hash(model) == expected
    counts = {}
    for role, records, key in (('document', documents, 'text'),
                                ('query', queries, 'query')):
        if model_name == 'dense':
            teacher = args.research_root / 'NG-0069/teacher-v1'
            if role == 'document':
                known = np.load(teacher / 'document-ids.npy')
                cached = np.load(teacher / 'documents.npy', mmap_mode='r')
            else:
                known = np.arange(6144)
                cached = np.load(teacher / 'train-query.npy', mmap_mode='r')
            matrix = np.empty((len(records), 1024), dtype=np.float32)
            matrix[known] = cached
            todo = np.setdiff1d(np.arange(len(records)), known)
            assert len(known) + len(todo) == len(records)
            for position in range(0, 8, 4 if role == 'document' else 1):
                ids = known[position:position + (4 if role == 'document' else 1)]
                probe = perform([records[i][key] for i in ids], role)
                np.testing.assert_allclose(probe, matrix[ids], rtol=1e-4, atol=2e-5)
            batch = 4 if role == 'document' else 1
            tick = time.monotonic()
            for start in range(0, len(todo), batch):
                ids = todo[start:start + batch]
                matrix[ids] = perform([records[i][key] for i in ids], role)
                assert torch.cuda.max_memory_allocated() < GPU_BYTES
                if start % (batch * 128) == 0:
                    print(args.phase, role, start + len(ids), '/', len(todo), flush=True)
                if start == batch * 15:
                    prediction = (time.monotonic() - tick) * len(todo) / (batch * 16) * 1.5
                    write(path / (role + '-canary.json'), {
                        'passed': prediction < 4800, 'predicted_seconds': prediction})
                    assert prediction < 4800
            assert np.isfinite(matrix).all()
            np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1,
                                       rtol=0, atol=2e-6)
            assert np.array_equal(matrix[known], cached)
            np.save(path / (role + '.npy'), matrix)
            counts[role] = {'bytes': matrix.nbytes, 'shape': matrix.shape,
                            'reused': len(known), 'encoded': len(todo)}
            del matrix, cached
            gc.collect()
            continue
        parts = []
        batch = 96 if role == 'document' else 8
        tick = time.monotonic()
        for start in range(0, len(records), batch):
            value = perform([r[key] for r in records[start:start + batch]], role)
            assert np.isfinite(value).all()
            assert torch.cuda.max_memory_allocated() < GPU_BYTES
            parts.append(sparse.csr_matrix(value))
            if start % (batch * 32) == 0:
                print(args.phase, role, start + len(value), '/', len(records), flush=True)
            if start == batch * 7:
                prediction = (time.monotonic() - tick) * len(records) / (batch * 8) * 1.5
                write(path / (role + '-canary.json'), {
                    'passed': prediction < 4800, 'predicted_seconds': prediction})
                assert prediction < 4800
        matrix = sparse.vstack(parts, format='csr')
        matrix.eliminate_zeros()
        matrix.sort_indices()
        assert matrix.dtype == np.float32 and matrix.has_canonical_format
        assert (np.diff(matrix.indptr) > 0).all() and (matrix.data > 0).all()
        sparse.save_npz(path / (role + '.npz'), matrix)
        counts[role] = {'nnz': matrix.nnz,
            'csr_bytes': matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes}
        del parts, matrix
        gc.collect()
    assert c.parameter_hash(model) == expected
    return {'model': model_name, 'model_sha256': expected, 'counts': counts,
            'documents': len(documents), 'queries': len(queries),
            'locked_test_queries_encoded': 0,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}


def top_independent(scores, k=100):
    k = min(k, len(scores))
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    ids = np.flatnonzero(scores >= threshold)
    return ids[np.lexsort((ids, -scores[ids]))][:k].tolist()


def dense_alternate(documents, vector):
    # Independent float64 reduction without a full elementwise-product temporary.
    return np.einsum('ij,j->i', documents, vector, optimize=False)


def rank(args, task):
    from scipy import sparse
    import metrics59 as metrics

    model_name = args.phase.removeprefix('rank-')
    path = args.run / args.phase
    lexical = args.research_root / 'NG-0069/lexical-v1'
    queries, labels = read(lexical / 'queries.json'), read(lexical / 'labels.json')
    lq, ld = [sparse.load_npz(lexical / f'lexical-{role}.npz').astype(np.float64)
              for role in ('query', 'document')]
    li = ld.T.tocsr()
    if model_name != 'bm25':
        completed(args.run, 'encode-' + model_name)
        folder = args.run / ('encode-' + model_name)
        if model_name == 'dense':
            q, d = [np.load(folder / (role + '.npy'), mmap_mode='r')
                    for role in ('query', 'document')]
            assert q.shape == (7872, 1024) and d.shape == (233009, 1024)
            d = d.astype(np.float64)
        else:
            q, d = [sparse.load_npz(folder / (role + '.npz')).astype(np.float64)
                    for role in ('query', 'document')]
            di = d.T.tocsr()
    records, errors = [], []
    tick = time.monotonic()
    with (path / 'rankings.jsonl').open('x') as stream:
        for i, query in enumerate(queries):
            if model_name == 'dense':
                vector = q[i].astype(np.float64)
                score = d @ vector
                alternate = dense_alternate(d, vector)
            else:
                score = (lq.getrow(i) @ li).toarray().ravel()
                alternate = ld @ lq.getrow(i).toarray().ravel()
                if model_name != 'bm25':
                    score += (q.getrow(i) @ di).toarray().ravel()
                    alternate += d @ q.getrow(i).toarray().ravel()
            assert score.shape == (233009,) and np.isfinite(score).all()
            row = metrics.rank_row(score, labels[i])
            metrics.audit_row(row, labels[i], 233009)
            error = float(abs(alternate - score).max())
            assert error <= 1e-12
            assert row['top100'] == top_independent(alternate)
            ids = np.arange(len(score))
            ranks = [1 + np.count_nonzero(alternate > alternate[g])
                     + np.count_nonzero((alternate == alternate[g]) & (ids < g))
                     for g in labels[i]]
            assert row['gold_ranks'] == ranks
            row.update(query_id=query['query_id'], domain=query['subset'],
                       split=query['split'])
            records.append(row)
            errors.append(error)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            if i % 128 == 0:
                stream.flush()
                print(args.phase, i + 1, '/7872', flush=True)
            if i == 15:
                prediction = (time.monotonic() - tick) * len(queries) / 16 * 1.5
                write(path / 'canary.json', {
                    'passed': prediction < 5100, 'predicted_seconds': prediction})
                assert prediction < 5100
    quality = {}
    for split in ('TRAIN', 'DEV_NEW', 'DEV_EXPOSED'):
        part = [r for r in records if r['split'] == split]
        quality[split] = metrics.summarize(np.array([
            [r['ndcg10'], r['recall100']] for r in part]), [r['domain'] for r in part])
    counts = {}
    if model_name not in ('bm25', 'dense'):
        df = np.bincount(d.indices, minlength=d.shape[1])
        work = q.sign() @ df
        counts['semantic_df_proxy'] = {'mean': float(work.mean()),
            'p95': float(np.quantile(work, .95)),
            'dev_new_mean': float(work[6144:7680].mean())}
    return {'model': model_name, 'quality': quality, 'counts': counts,
            'all_query_score_and_rank_checks_passed': True,
            'max_independent_score_error': max(errors),
            'native_cost_evaluated': False, 'locked_test_scored': False}


def worker(args):
    c = legacy(args.research_root)
    c.setup()
    frozen = verify(args.research_root, args.run)
    assert args.phase not in frozen.get('inherited', {}).get('phases', [])
    from clearml import Task
    path = args.run / args.phase
    Task.set_offline(True)
    disabled = {key: False for key in ('detect_repository', 'hydra', 'pytorch',
        'scikit', 'joblib', 'matplotlib', 'tensorflow', 'tensorboard', 'tfdefines',
        'megengine', 'xgboost', 'catboost', 'fastai', 'lightgbm', 'gradio')}
    task = Task.init(project_name='Evoke-NG', task_name='NG-0069/' + args.phase,
        task_type=Task.TaskTypes.training if args.phase.startswith('train-')
        else Task.TaskTypes.testing, reuse_last_task_id=False,
        auto_connect_frameworks=disabled, auto_connect_streams=False,
        auto_connect_arg_parser=False, auto_resource_monitoring=False)
    receipt = {'task_id': task.id, 'actual_start': True, 'closed': False,
               'offline': True, 'remote_synced': False, 'post_hoc': False}
    write(path / 'clearml.json', receipt)
    task.connect({'inputs_sha256': sha(args.run / 'inputs.json')})
    tick = time.monotonic()
    try:
        action = args.phase.split('-')[0]
        result = {'train': train, 'encode': encode, 'rank': rank}[action](args, task)
        result.update(seconds=time.monotonic() - tick, overall_goal_qualified=False,
                      production_changed=False)
        write(path / 'results.json', result)
    finally:
        task.close()
        receipt['closed'] = True
        write(path / 'clearml.json', receipt)
    verify(args.research_root, args.run)
    write(path / 'complete.json', {'passed': True, 'files': {
        str(p.relative_to(path)): sha(p) for p in path.rglob('*') if p.is_file()
        and p.name not in ('stdout.log', 'complete.json', 'exit.json')}})


def phase(args, name):
    legacy(args.research_root)
    from lifecycle52 import stop_controller
    path = args.run / name
    if (path / 'exit.json').exists():
        completed(args.run, name)
        return
    path.mkdir(exist_ok=True)
    assert not (path / 'started.json').exists(), 'Inspect interrupted phase'
    verify(args.research_root, args.run)
    assert psutil.virtual_memory().available > 24 * 1024 ** 3
    assert psutil.disk_usage(path).free > 40 * 1024 ** 3
    info = subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-gpu=uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert int(info[1]) < 100 and int(info[2]) == 0, 'GPU0 occupied'
    env = dict(os.environ)
    for key in ('CLEARML_TASK_ID', 'TRAINS_TASK_ID', 'CLEARML_PROC_MASTER_ID',
                'TRAINS_PROC_MASTER_ID'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='0', OPENBLAS_NUM_THREADS='4',
        OMP_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', HF_HUB_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', CLEARML_OFFLINE_MODE='1',
        CLEARML_CACHE_DIR=str(args.run / 'tracking-cache'), PYTHONDONTWRITEBYTECODE='1')
    limit = 1800 if name.startswith('train-') else 5400
    write(path / 'started.json', {'started_unix': time.time(), 'limit_seconds': limit})
    process, error, code, peak, closed = None, None, 1, 0, False
    tick = time.monotonic()
    try:
        with (path / 'stdout.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                '--research-root', str(args.research_root), '--run', str(args.run),
                '--mode', 'worker', '--phase', name], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(args.run / 'continuation-state.json', {
                'status': 'running', 'phase': name, 'controller_pid': os.getpid(),
                'worker_pid': process.pid, 'updated_unix': time.time()})
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in
                              [parent] + parent.children(recursive=True))
                    peak = max(peak, rss)
                    assert rss <= HOST_BYTES, 'Host process-tree limit'
                except psutil.NoSuchProcess:
                    pass
                assert time.monotonic() - tick <= limit, 'Phase wall limit'
                time.sleep(1)
            code = process.wait()
            assert code == 0 and read(path / 'complete.json')['passed']
    except BaseException as exc:
        error = repr(exc)
    finally:
        saved = {sig: signal.signal(sig, signal.SIG_IGN)
                 for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            stop_controller(process)
            closed = True
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            write(path / 'exit.json', {'exit_code': 1 if error else code,
                'error': error, 'owned_group_closed': closed,
                'seconds': time.monotonic() - tick, 'peak_tree_rss_bytes': peak})
    assert error is None and code == 0 and closed, error
    completed(args.run, name)


def supervise(args):
    import fcntl

    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with ExitStack() as stack:
        for path in (args.run / 'pipeline.lock', args.research_root / '.ng-gpu-0.lock'):
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify(args.research_root, args.run)
            for name in schedule():
                phase(args, name)
            write(args.run / 'continuation-state.json', {
                'status': 'all_phases_complete', 'updated_unix': time.time(),
                'next_action': 'independent_paired_review_and_artifact_copy',
                'overall_goal_qualified': False, 'locked_test_scored': False})
        except BaseException as exc:
            write(args.run / 'continuation-state.json', {
                'status': 'failed_preserve_attempt', 'error': repr(exc),
                'updated_unix': time.time(), 'automatic_retry_allowed': False})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--mode', choices=('freeze', 'supervise', 'worker'), required=True)
    parser.add_argument('--phase', choices=schedule())
    parser.add_argument('--reuse-run', type=Path,
                        help='Freeze an evaluation-only continuation of sealed successes')
    args = parser.parse_args()
    args.research_root, args.run = args.research_root.resolve(), args.run.resolve()
    assert sys.version_info[:2] == (3, 12) and not sys.flags.optimize
    assert args.run.is_dir()
    assert args.reuse_run is None or args.mode == 'freeze'
    if args.mode == 'worker':
        assert args.phase is not None
    {'freeze': freeze, 'supervise': supervise, 'worker': worker}[args.mode](args)


if __name__ == '__main__':
    main()
