"""Explain the failed NG77 cache comparison without training or relaxing it."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from scipy import sparse

import ng77_cuda_canary as canary
from review_ng75_displacement import require, verify as verify_files


FAILED = 'NG-0077/no-optimizer-canary-v1'
FAILED_SHA = '38c37764dd99f6193ace406e46c64f0f211ff3486d87551c7d2d3bcf61f75560'
TOTAL = 233009


def corpus_groups(pool, total=TOTAL):
    require(pool and len(set(pool)) == len(pool)
            and all(type(i) is int and 0 <= i < total for i in pool), 'invalid pool')
    return [list(range(start, min(start + 4, total)))
            for start in sorted({i // 4 * 4 for i in pool})]


def failed_evidence(base):
    run = base / FAILED
    require(canary.io.sha(run / 'inputs.json') == FAILED_SHA, 'wrong failed attempt')
    manifest = canary.verify(base, run)
    phase = run / 'canary'
    complete = canary.execution.read(phase / 'complete.json')
    verify_files(phase, complete['files'], exact=True, exclude=('complete.json',))
    exited = canary.execution.read(phase / 'exit.json')
    tracking = canary.execution.read(phase / 'clearml.json')
    require(complete['passed'] is False and exited['exit_code'] == 1
            and exited['error'] is None and exited['owned_group_closed']
            and tracking['actual_start'] and tracking['closed']
            and tracking['outcome'] == 'failed', 'failure is not safely sealed')
    require('cached code/support parity failed' in (phase / 'stdout.log').read_text()
            and not (phase / 'results.json').exists(), 'different failed gate')
    return manifest


def array_audit(actual, expected):
    parity = canary.code_parity(actual, expected)
    require(parity['exact_support'], 'context diagnosis requires the recorded equal support')
    delta = abs(actual.data.astype(np.float64) - expected.data.astype(np.float64))
    limit = canary.PARITY['atol'] + canary.PARITY['rtol'] * abs(expected.data.astype(np.float64))
    rows = np.repeat(np.arange(actual.shape[0]), np.diff(actual.indptr))
    bad = delta > limit
    return dict(parity=parity, bad_coordinates=int(bad.sum()),
                bad_by_row=np.bincount(rows[bad], minlength=actual.shape[0]).tolist(),
                max_tolerance_ratio=float(np.max(delta / limit, initial=0)))


def verify(base, run):
    m = canary.execution.read(run / 'inputs.json')
    require(m['protocol'] == 'NG77_batch_context_diagnosis_v1'
            and m['training_enabled'] is False and m['optimizer_updates'] == 0
            and m['quality_evaluation'] is False and m['training_authorized'] is False
            and m['parity'] == canary.PARITY, 'diagnostic contract changed')
    verify_files(base, m['dependencies'])
    verify_files(run, m['source'])
    parent = canary.execution.read(base / FAILED / 'inputs.json')
    p = canary.execution.read(base / FAILED / 'canary/parity-00.json')
    require(m['query_id'] == parent['query_ids'][0] == p['query_id']
            and m['pool'] == p['pool'] and m['corpus_groups'] == corpus_groups(m['pool']),
            'diagnostic identities or contexts changed')
    require(canary.execution.read(run / 'config.json')
            == canary.execution.read(base / FAILED / 'config.json'), 'model configuration changed')
    return m


def freeze(base, run):
    parent = failed_evidence(base)
    dependencies = dict(parent['dependencies'])
    dependencies.update({str(p.relative_to(base)): canary.io.sha(p)
                         for p in (base / FAILED).rglob('*') if p.is_file()})
    for name in ('NG-0059/model59.py', 'NG-0059/common59.py',
                 'NG-0069/evaluation-v2/ng69_pipeline.py'):
        dependencies[name] = canary.io.sha(base / name)
    p = canary.execution.read(base / FAILED / 'canary/parity-00.json')
    require(p['document']['exact_support'] and not p['document']['passed'],
            'not the observed value-only failure')
    run.mkdir(parents=True, exist_ok=False)
    for name in parent['source']:
        shutil.copy2(base / FAILED / name, run / name)
    source = Path(__file__).resolve()
    shutil.copy2(source, run / source.name)
    shutil.copy2(source.parents[1] / 'tests/test_ng77_context.py', run / 'test_ng77_context.py')
    plan = source.parents[1] / 'docs/research-sae/reports/ng0001-ng0099/ng0077-batch-context-diagnostic.zh.md'
    shutil.copy2(plan, run / plan.name)
    canary.io.write(run / 'inputs.json', dict(protocol='NG77_batch_context_diagnosis_v1',
        dependencies=dependencies, source={p.name: canary.io.sha(p) for p in run.iterdir() if p.is_file()},
        query_id=p['query_id'], pool=p['pool'], corpus_groups=corpus_groups(p['pool']),
        training_enabled=False, optimizer_updates=0, quality_evaluation=False,
        training_authorized=False, parity=canary.PARITY, maximum_wall_seconds=1800))
    verify(base, run)
    print('NG77_CONTEXT_FROZEN', canary.io.sha(run / 'inputs.json'), flush=True)


def legacy_forward(encoder, texts):
    tokens = encoder.tokenizer(texts, truncation=True, padding=True,
                               max_length=256, return_tensors='pt').to('cuda')
    hidden = encoder.model.base_model(**tokens).last_hidden_state
    values = encoder.model.lm_head(hidden).relu().log1p()
    return values.masked_fill(~tokens['attention_mask'].bool()[..., None], 0).amax(1)


def diagnose(base, run):
    import torch

    m = verify(base, run)
    output, pool = run / 'context', m['pool']
    needed = {i for group in m['corpus_groups'] for i in group}
    documents = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for i, line in enumerate(stream):
            if i in needed:
                documents[i] = json.loads(line)
    require(i + 1 == TOTAL and set(documents) == needed, 'canonical corpus changed')
    config = canary.execution.read(run / 'config.json')
    records = {r['query_id']: r for r in canary.pipeline.rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')}
    query = next(q for q in canary.execution.read(base / 'NG-0069/lexical-v1/queries.json')
                 if q['query_id'] == m['query_id'])
    canary.execution.verify_text(records[m['query_id']], query, documents)
    encoder = canary.original.Encoder(base, config, device='cuda')
    before = canary.original.parameter_hash(encoder.model)
    rng = [torch.get_rng_state().clone(), *[r.clone() for r in torch.cuda.get_rng_state_all()]]
    lengths = {i: len(encoder.tokenizer(r['text'], truncation=True, max_length=256)['input_ids'])
               for i, r in documents.items()}
    arrays = {}
    for context, groups in (('pool', [pool[i:i + 4] for i in range(0, len(pool), 4)]),
                             ('corpus', m['corpus_groups'])):
        parts, ids = [], []
        for group in groups:
            texts = [documents[i]['text'] for i in group]
            with torch.no_grad():
                value = encoder.encode(texts, 'document')
                legacy = legacy_forward(encoder, texts)
            require(torch.equal(value, legacy), 'current forward differs from original formula in same context')
            parts.append(sparse.csr_matrix(value.cpu().numpy()))
            ids.extend(group)
        matrix = sparse.vstack(parts, format='csr')
        sparse.save_npz(output / f'{context}-documents.npz', matrix)
        canary.io.write(output / f'{context}-ids.json', ids)
        arrays[context] = matrix[[ids.index(i) for i in pool]]
    cached = sparse.load_npz(base / 'NG-0069/evaluation-v2/encode-initial/document.npz')[pool]
    previous = sparse.load_npz(base / FAILED / 'canary/documents-00.npz')
    audits = dict(pool_vs_cache=array_audit(arrays['pool'], cached),
                  corpus_vs_cache=array_audit(arrays['corpus'], cached),
                  pool_vs_failed=array_audit(arrays['pool'], previous))
    contexts, summary = [], Counter()
    for i, doc in enumerate(pool):
        old = list(range(doc // 4 * 4, min(doc // 4 * 4 + 4, TOTAL)))
        new = pool[i // 4 * 4:i // 4 * 4 + 4]
        old_pad, new_pad = max(lengths[j] for j in old), max(lengths[j] for j in new)
        bad = audits['pool_vs_cache']['bad_by_row'][i]
        summary[f'padding_changed_{old_pad != new_pad}/parity_failed_{bad > 0}'] += 1
        contexts.append(dict(document_id=doc, original_group=old, pool_group=new,
                             own_tokens=lengths[doc], old_pad=old_pad, new_pad=new_pad,
                             bad_coordinates=bad))
    canary.io.write(output / 'contexts.json', contexts)
    after_rng = [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]
    require(before == canary.original.parameter_hash(encoder.model)
            and all(p.grad is None for p in encoder.model.parameters())
            and all(torch.equal(a, b) for a, b in zip(rng, after_rng, strict=True)),
            'no-update model/gradient/RNG contract changed')
    return dict(diagnostic_completed=True, audits=audits, context_summary=dict(summary),
        old_formula_bit_exact_in_each_context=True, model_and_rng_unchanged=True,
        model_sha256=before, query_id=m['query_id'], pool_documents=len(pool),
        canonical_context_documents=len(needed), optimizer_updates=0,
        training_authorized=False, quality_evaluation=False, original_canary_still_failed=True,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())


def worker(args):
    import torch
    from clearml import Task

    verify(args.research_root, args.run)
    require(args.phase == 'context', 'only context diagnosis is enabled')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='Evoke-NG', task_name='NG-0077/batch-context-diagnosis',
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, actual_start=True, post_hoc=False, offline=True,
                   remote_synced=False, closed=False)
    output = args.run / args.phase
    canary.io.write(output / 'clearml-start.json', receipt)
    task.connect(dict(manifest_sha256=canary.io.sha(args.run / 'inputs.json'), optimizer_updates=0))
    try:
        result = diagnose(args.research_root, args.run)
        verify(args.research_root, args.run)
        canary.io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        canary.io.write(output / 'clearml.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'supervise', 'worker'))
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--phase')
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    require(args.run.parent == args.research_root / 'NG-0077', 'fresh NG77 diagnostic required')
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
        return
    verify(args.research_root, args.run)
    require(canary.io.sha(Path(__file__)) == canary.execution.read(
        args.run / 'inputs.json')['source'][Path(__file__).name], 'not frozen diagnostic source')
    if args.mode == 'worker':
        worker(args)
        return
    stopped = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda value, frame: stopped.append(value))
    with (args.research_root / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory, busy = canary.pipeline.gpu_state(args.gpu)
        require(memory < 100 and busy == 0, 'GPU occupied; do not evict')
        canary.io.write(args.run / 'controller-start.json', dict(pid=os.getpid(), gpu=args.gpu,
                                                               started_unix=time.time()))
        try:
            args.worker_script = Path(__file__).name
            canary.pipeline.phase(args, 'context', stopped, cuda=True, limit_seconds=1800)
            status = dict(status='diagnostic_complete_no_training', finished_unix=time.time())
        except BaseException as exc:
            status = dict(status='failed_preserve_attempt', error=f'{type(exc).__name__}: {exc}',
                          finished_unix=time.time())
            canary.io.write(args.run / 'controller-exit.json', status)
            raise
        canary.io.write(args.run / 'controller-exit.json', status)


if __name__ == '__main__':
    main()
