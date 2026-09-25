"""Frozen no-optimizer CUDA gate for NG77; never launches scientific training."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from scipy import sparse

import ng71_execution as execution
import ng71_pilot as pipeline
import ng71_preflight as io
import ng71_training as original
from ng77_retention import combined_loss
from ng77_training import RetainedObjective
from review_ng75_displacement import require, verify as verify_files


PARENT = 'NG-0077/preparation-v1'
PARENT_SHA = '8e9861f011bcbff1f1f43cee450748fee88256379a9fc2370f1c1857d16c9ac4'
PROCEDURE_SHA = '6f4ee6940642389cce99c3fb2989c58810f710d6578d92faf6f3900f39cd6ff5'
REVIEW_SHA = 'faff1a4c1ee6f7060b4b9fd9ae11c767a87112853be5625e202be3f32531ad5b'
PARITY = dict(rtol=2e-5, atol=2e-7, exact_support=True,
              keep_noise_relative_to_D_max=.001, vjp_relative_l2_max=1e-5,
              lipschitz_roundoff_atol=1e-12)
DOMAINS = ('fever', 'hotpotqa', 'nq')
CONTEXT_PROTOCOL = 'NG77_context_qualified_cuda_canary_v2'
CONTEXT_RUN = 'NG-0077/batch-context-diagnostic-v1'
CONTEXT_COMPLETE = 'acaf688396078a64519f3f1a4f5949c98309231607a3d799a1e785c5930963bb'
CONTEXT_INPUTS = 'be201ffa3ae5d7a6c6694bdd734a1b4580fd32414645ba0671b3322afb999237'


def corpus_groups(pool, total=233009):
    require(pool and len(pool) == len(set(pool))
            and all(type(i) is int and 0 <= i < total for i in pool), 'invalid context pool')
    return [list(range(start, min(start + 4, total)))
            for start in sorted({i // 4 * 4 for i in pool})]


def accepted_context(base):
    root = base / CONTEXT_RUN
    require(io.sha(root / 'inputs.json') == CONTEXT_INPUTS
            and io.sha(root / 'context/complete.json') == CONTEXT_COMPLETE,
            'context diagnosis identity changed')
    manifest = execution.read(root / 'inputs.json')
    verify_files(base, manifest['dependencies'])
    verify_files(root, manifest['source'])
    result = execution.sealed(root / 'context')
    require(result['model_and_rng_unchanged'] and result['old_formula_bit_exact_in_each_context']
            and result['original_canary_still_failed'] and not result['training_authorized']
            and all(result['audits'][k]['parity']['passed']
                    and result['audits'][k]['parity']['max_abs_error'] == 0
                    for k in ('corpus_vs_cache', 'pool_vs_failed')), 'context diagnosis not qualified')
    return dict(manifest['dependencies']) | {
        str(p.relative_to(base)): io.sha(p) for p in root.rglob('*') if p.is_file()}


def select_canary(selection, order, records):
    require(len(order) == 768 and len(set(order[:384])) == 384
            and set(order[:384]) == set(selection['pilot']) == set(records), 'original pilot order changed')
    require(not set(selection['pilot']) & set(selection['sentinel']), 'TRAIN split overlap')
    chosen, counts = [], Counter()
    for q in order[:384]:
        r = records[q]
        require(r['split'] == 'TRAIN' and r['reference_update'] == 0
                and r['domain'] in DOMAINS, 'wrong canary reference')
        if counts[r['domain']] < 4:
            chosen.append(q)
            counts[r['domain']] += 1
    require(counts == dict.fromkeys(DOMAINS, 4), 'incomplete 12-query canary')
    return chosen


def verify(base, run):
    m = execution.read(run / 'inputs.json')
    require(m['protocol'] in ('NG77_no_optimizer_cuda_canary_v1', CONTEXT_PROTOCOL)
            and m['phases'] == ['canary'] and m['optimizer_updates'] == 0
            and m['training_enabled'] is False and m['parity'] == PARITY
            and m['locked_test_access'] is False, 'canary protocol changed')
    verify_files(base, m['dependencies'])
    verify_files(run, m['source'])
    require(io.sha(base / PARENT / 'inputs.json') == PARENT_SHA, 'wrong CPU preparation')
    records = {r['query_id']: r for r in pipeline.rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')}
    selection = execution.read(base / 'NG-0071/pilot-v1/selection.json')
    order = execution.read(base / 'NG-0071/pilot-v1/training-order.json')
    require(m['query_ids'] == select_canary(selection, order, records), 'canary identities changed')
    if m['protocol'] == CONTEXT_PROTOCOL:
        require(m['corpus_contexts'] == {q: corpus_groups(records[q]['pool']) for q in m['query_ids']}
                and m['comparison_contract'] == 'same_context_cache_and_actual_pool_score_noise_vjp',
                'context comparison or corpus group identity changed')
        require(m['dependencies'].get(f'{CONTEXT_RUN}/context/complete.json') == CONTEXT_COMPLETE,
                'missing accepted context proof')
    config = execution.read(run / 'config.json')
    expected = dict(execution.read(base / 'NG-0071/pilot-v1/config.json'), training_enabled=False)
    require(config == expected, 'base, optimizer or calibration changed')
    return m


def freeze(base, run):
    parent = base / PARENT
    require(io.sha(parent / 'inputs.json') == PARENT_SHA, 'wrong CPU preparation')
    dependencies = accepted_context(base)
    dependencies.update(execution.read(parent / 'inputs.json')['dependencies'])
    for name, digest in (('preparation-v1-procedure-v1', PROCEDURE_SHA),
                         ('preparation-v1-review-v1', REVIEW_SHA)):
        path = base / 'NG-0077' / name
        require(io.sha(path / 'complete.json') == digest, 'wrong CPU completion proof')
        receipt = execution.read(path / 'complete.json')
        require(receipt['passed'], 'CPU gate failed')
        verify_files(path, receipt['files'], exact=True, exclude=('complete.json',))
        dependencies.update({str(p.relative_to(base)): io.sha(p) for p in path.rglob('*') if p.is_file()})
    inventory = execution.read(base / 'NG-0077/preparation-v1-procedure-v1/run-inventory.json')
    verify_files(parent, inventory['files'], exact=True)
    dependencies.update({f'{PARENT}/{name}': digest for name, digest in inventory['files'].items()})
    pilot = execution.read(base / 'NG-0071/pilot-v1/inputs.json')
    dependencies['NG-0071/pilot-v1/config.json'] = pilot['source']['config.json']
    verify_files(base, dependencies)
    records = {r['query_id']: r for r in pipeline.rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')}
    ids = select_canary(execution.read(base / 'NG-0071/pilot-v1/selection.json'),
        execution.read(base / 'NG-0071/pilot-v1/training-order.json'), records)
    run.mkdir(exist_ok=False, parents=True)
    source = Path(__file__).resolve().parent
    names = [p.name for p in source.glob('ng71_*.py')]
    names += ['audit_ng70_provenance.py', 'analyze_ng66_learning_curves.py',
              'ng77_retention.py', 'ng77_training.py', 'ng77_cuda_canary.py', 'review_ng75_displacement.py']
    for name in names:
        shutil.copy2(source / name, run / name)
    for path in (source.parent / 'tests').glob('test_ng77_cuda*.py'):
        shutil.copy2(path, run / path.name)
    plan = source.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0077-cuda-canary-protocol.zh.md'
    shutil.copy2(plan, run / plan.name)
    context_plan = plan.with_name('ng0077-context-qualified-canary.zh.md')
    shutil.copy2(context_plan, run / context_plan.name)
    io.write(run / 'config.json', dict(execution.read(base / 'NG-0071/pilot-v1/config.json'), training_enabled=False))
    io.write(run / 'inputs.json', dict(protocol=CONTEXT_PROTOCOL,
        dependencies=dependencies, source={p.name: io.sha(p) for p in run.iterdir() if p.is_file()},
        phases=['canary'], query_ids=ids, optimizer_updates=0, training_enabled=False,
        locked_test_access=False, parity=PARITY, synthetic_vjp_fixture='first anchor of first query per domain',
        synthetic_fixture_documents=2, synthetic_fixture_margin_increase=1.,
        corpus_contexts={q: corpus_groups(records[q]['pool']) for q in ids},
        comparison_contract='same_context_cache_and_actual_pool_score_noise_vjp',
        maximum_wall_seconds=1800, maximum_gpu_gib=20, maximum_tree_rss_gib=16))
    verify(base, run)
    print('NG77_CANARY_FROZEN', io.sha(run / 'inputs.json'), flush=True)


def code_parity(actual, expected):
    require(actual.shape == expected.shape and actual.has_canonical_format and expected.has_canonical_format,
            'sparse shape/format differs')
    require(all(np.isfinite(m.data).all() and (m.data > 0).all() for m in (actual, expected)), 'invalid sparse values')
    same = np.array_equal(actual.indptr, expected.indptr) and np.array_equal(actual.indices, expected.indices)
    details = dict(passed=False, actual_nnz=actual.nnz, expected_nnz=expected.nnz, exact_support=same)
    if same:
        details['max_abs_error'] = float(np.max(abs(actual.data - expected.data), initial=0))
        details['values_within_tolerance'] = bool(np.allclose(actual.data, expected.data,
            rtol=PARITY['rtol'], atol=PARITY['atol']))
        details['passed'] = details['values_within_tolerance']
    else:
        delta = actual.sign() - expected.sign()
        rows, columns = delta.nonzero()
        details['support_differences'] = [dict(row=int(r), column=int(c),
            actual=float(actual[r, c]), expected=float(expected[r, c])) for r, c in zip(rows[:32], columns[:32])]
        details['total_support_differences'] = len(rows)
    return details


def legacy_documents(encoder, texts):
    """Original NG59 formula in the unchanged candidate microbatch context."""
    tokens = encoder.tokenizer(texts, truncation=True, padding=True,
                               max_length=256, return_tensors='pt').to(encoder.device)
    hidden = encoder.model.base_model(**tokens).last_hidden_state
    values = encoder.model.lm_head(hidden).relu().log1p()
    return values.masked_fill(~tokens['attention_mask'].bool()[..., None], 0).amax(1)


def context_controls(encoder, documents, pool, groups, actual, expected, output, index):
    import torch

    require(groups == corpus_groups(pool), 'wrong cache batch context')
    with torch.no_grad():
        legacy = torch.cat([legacy_documents(encoder, [documents[j]['text'] for j in pool[s:s + 4]])
                            for s in range(0, len(pool), 4)])
    exact = torch.equal(actual, legacy)
    io.write(output / f'formula-{index:02d}.json', dict(pool=pool, bit_exact=exact))
    require(exact, 'actual candidate-pool formula differs from original D')
    ids, parts = [], []
    with torch.no_grad():
        for group in groups:
            values = encoder.encode([documents[j]['text'] for j in group], 'document')
            ids.extend(group)
            parts.append(sparse.csr_matrix(values.cpu().numpy()))
    matrix = sparse.vstack(parts, format='csr')
    sparse.save_npz(output / f'corpus-documents-{index:02d}.npz', matrix)
    io.write(output / f'corpus-ids-{index:02d}.json', ids)
    # All companions participate in identity parity, but never the loss pool.
    parity = code_parity(matrix, expected[ids])
    return dict(document=parity, pool_formula_bit_exact=True,
                cache_context_documents=len(ids), actual_training_pool_unchanged=True,
                cross_context_diagnostic=code_parity(sparse.csr_matrix(actual.cpu().numpy()), expected[pool]))


def score_noise(scores, baseline, pairs, anchors):
    import torch

    x = scores.detach().clone().requires_grad_()
    legacy = original.pair_loss(x, pairs)
    gradient = torch.autograd.grad(legacy, x, retain_graph=True)[0]
    control, keep = combined_loss(legacy, x, anchors, 0.)
    require(control is legacy and torch.equal(gradient, torch.autograd.grad(control, x, retain_graph=True)[0]),
            'zero-coefficient scalar or gradient parity failed')
    kg = torch.autograd.grad(keep, x)[0]
    actual = x.detach().cpu().double().numpy()
    expected = np.asarray(baseline, dtype=np.float64)
    error = float(abs(actual - expected).max())
    np.testing.assert_allclose(actual, expected, rtol=PARITY['rtol'], atol=PARITY['atol'])
    scalar_gradient = np.zeros(len(actual), dtype=np.float64)
    for item in anchors['anchors']:
        p, n = item['indices']
        a0, a = item['baseline_margin'], actual[p] - actual[n]
        if a < a0:
            derivative = item['coefficient'] * (float(np.exp(-np.logaddexp(0., -a)))
                                                - float(np.exp(-np.logaddexp(0., -a0))))
            scalar_gradient[p] += derivative
            scalar_gradient[n] -= derivative
    observed = kg.detach().cpu().double().numpy()
    np.testing.assert_allclose(observed, scalar_gradient, rtol=1e-6, atol=1e-12)
    mass = float(abs(observed).sum())
    require(mass <= error + PARITY['lipschitz_roundoff_atol'], 'initial keep gradient exceeds Lipschitz bound')
    return dict(scores=actual.tolist(), baseline_scores=expected.tolist(),
        max_score_error=error, D_loss=float(legacy.detach()), keep_loss=float(keep.detach()),
        D_score_gradient_l1=float(gradient.abs().sum()), keep_score_gradient_l1=mass,
        zero_coefficient_bit_exact=True, scalar_keep_gradient_max_error=float(abs(observed - scalar_gradient).max()))


def vjp_check(encoder, query, documents, lexical, output, tag):
    import torch

    with torch.no_grad():
        d, q = encoder.encode(documents, 'document'), encoder.encode([query], 'query')
        scores = (d @ q.T).ravel() + d.new_tensor(lexical)
        baseline = float(scores[0] - scores[1]) + 1.
    require(baseline > 0, 'fixed trusted-pair fixture lost its positive reference')
    anchors = dict(candidate_count=2, total_positives=1, student_temperature=1.,
        anchors=[dict(indices=[0, 1], baseline_margin=baseline, coefficient=1.)])
    pairs = dict(indices=[[0, 1]], targets=[.8], coefficients=[.6], candidate_count=2,
        student_temperature=1., supervised_positives=1)
    example = dict(query=query, documents=documents, lexical=lexical, pairs=pairs)
    results = {}
    for coefficient in (0., 1.):
        encoder.model.zero_grad(set_to_none=True)
        direct_objective = RetainedObjective(anchors, coefficient)
        direct = original.backward_query(encoder, **example, accumulation=4, replay=False,
                                        loss_transform=direct_objective)
        gradients = {n: p.grad.detach().cpu().clone() for n, p in encoder.model.named_parameters() if p.grad is not None}
        encoder.model.zero_grad(set_to_none=True)
        replay_objective = RetainedObjective(anchors, coefficient)
        replay = original.backward_query(encoder, **example, accumulation=4, replay=True,
                                        loss_transform=replay_objective)
        require({k: v for k, v in direct.items() if k != 'vjp_replay_exact'}
                == {k: v for k, v in replay.items() if k != 'vjp_replay_exact'},
                'transformed score/loss/VJP trace changed')
        require(direct_objective.last == replay_objective.last and direct_objective.last['keep_loss'] > 0,
                'synthetic active-penalty VJP fixture failed')
        actual = {n: p.grad.detach().cpu() for n, p in encoder.model.named_parameters() if p.grad is not None}
        require(actual.keys() == gradients.keys(), 'shared-parameter gradient coverage changed')
        squared = reference_squared = maximum = 0.
        digest = hashlib.sha256()
        for name in sorted(gradients):
            delta = gradients[name].double() - actual[name].double()
            squared += float(delta.square().sum())
            reference_squared += float(gradients[name].double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
            digest.update(name.encode())
            digest.update(actual[name].contiguous().numpy().tobytes())
        relative = (squared / reference_squared) ** .5 if reference_squared else float('inf')
        require(np.isfinite(relative) and relative <= PARITY['vjp_relative_l2_max'], 'actual VJP gradient gate failed')
        results[str(coefficient)] = dict(direct=direct, objective=direct_objective.last,
            relative_l2=relative, max_abs_error=maximum, gradient_tensors=len(actual),
            replay_gradient_sha256=digest.hexdigest())
        if coefficient == 0:
            encoder.model.zero_grad(set_to_none=True)
            legacy = original.backward_query(encoder, **example, accumulation=4, replay=True)
            require(legacy == replay and all(torch.equal(actual[n], p.grad.detach().cpu())
                    for n, p in encoder.model.named_parameters() if n in actual),
                    'lambda0 actual shared gradient differs from original D path')
            results[str(coefficient)]['original_shared_gradient_bit_exact'] = True
        del gradients, actual
    encoder.model.zero_grad(set_to_none=True)
    io.write(output / f'vjp-{tag}.json', dict(passed=True, artificial_two_document_fixture=True,
        not_a_training_qrel=True, baseline_plus_one=baseline, coefficients=results, optimizer_updates=0))
    return results


def canary(base, run):
    import torch
    import transformers

    m = execution.read(run / 'inputs.json')
    output, config = run / 'canary', execution.read(run / 'config.json')
    all_queries = execution.read(base / 'NG-0069/lexical-v1/queries.json')
    positions = {q['query_id']: i for i, q in enumerate(all_queries)}
    records = {r['query_id']: r for r in pipeline.rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')}
    anchors = execution.read(base / PARENT / 'audit/anchors.json')
    needed = {d for q in m['query_ids'] for d in records[q]['pool']}
    context_qualified = m['protocol'] == CONTEXT_PROTOCOL
    if context_qualified:
        needed.update(j for groups in m['corpus_contexts'].values() for group in groups for j in group)
    documents = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for index, line in enumerate(stream):
            if index in needed:
                documents[index] = json.loads(line)
    require(index + 1 == 233009 and set(documents) == needed, 'document universe changed')
    cached = base / 'NG-0069/evaluation-v2/encode-initial'
    expected_q = sparse.load_npz(cached / 'query.npz')[[positions[q] for q in m['query_ids']]]
    expected_d = sparse.load_npz(cached / 'document.npz')
    encoder = original.Encoder(base, config, device='cuda')
    initial = original.parameter_hash(encoder.model)
    require(not any(module.training for module in encoder.model.modules()), 'eval mode changed')
    rng = [torch.get_rng_state().clone(), *[r.clone() for r in torch.cuda.get_rng_state_all()]]
    results, vjp = [], {}
    started = time.monotonic()
    with (output / 'per-query.jsonl').open('x') as stream:
        for i, identity in enumerate(m['query_ids']):
            r, query = records[identity], all_queries[positions[identity]]
            execution.verify_text(r, query, documents)
            tick = time.monotonic()
            with torch.no_grad():
                q = encoder.encode([query['query']], 'query')
                d = encoder.encode([documents[j]['text'] for j in r['pool']], 'document')
                scores = (d @ q.T).ravel() + d.new_tensor(r['lexical_scores'])
            qa, da = sparse.csr_matrix(q.cpu().numpy()), sparse.csr_matrix(d.cpu().numpy())
            sparse.save_npz(output / f'query-{i:02d}.npz', qa)
            sparse.save_npz(output / f'documents-{i:02d}.npz', da)
            parity = dict(query_id=identity, domain=r['domain'], pool=r['pool'],
                          query=code_parity(qa, expected_q[i]), document=code_parity(da, expected_d[r['pool']]))
            if context_qualified:
                parity.update(context_controls(encoder, documents, r['pool'], m['corpus_contexts'][identity],
                                               d, expected_d, output, i))
            io.write(output / f'parity-{i:02d}.json', parity)
            require(parity['query']['passed'] and parity['document']['passed'], 'cached code/support parity failed')
            row = score_noise(scores, r['hybrid_scores'], original.prepare_pairs(r, config['ranking']), anchors[identity])
            row.update(query_id=identity, domain=r['domain'], seconds=time.monotonic() - tick,
                       documents=len(r['pool']))
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            results.append(row)
            print('NG77_NO_UPDATE', i + 1, len(m['query_ids']), row['max_score_error'], flush=True)
            if i == 0:
                elapsed = time.time() - execution.read(output / 'started.json')['started_unix']
                estimate = elapsed + row['seconds'] * 12 * 1.5 + 360
                io.write(output / 'timing-canary.json', dict(predicted_seconds=estimate, passed=estimate < 1740))
                require(estimate < 1740, 'canary total time forecast exceeds bound')
    D = float(np.mean([r['D_score_gradient_l1'] for r in results]))
    K = float(np.mean([r['keep_score_gradient_l1'] for r in results]))
    require(D > 0 and K <= PARITY['keep_noise_relative_to_D_max'] * D, 'initial retention noise gate failed')
    for identity in m['query_ids']:
        r = records[identity]
        if r['domain'] not in vjp:
            a = anchors[identity]['anchors'][0]
            indices = a['indices']
            vjp[r['domain']] = vjp_check(encoder, all_queries[positions[identity]]['query'],
                [documents[r['pool'][j]]['text'] for j in indices], [r['lexical_scores'][j] for j in indices],
                output, r['domain'])
    require(initial == original.parameter_hash(encoder.model)
            and all(p.grad is None for p in encoder.model.parameters()), 'model state or cleared gradients changed')
    after_rng = [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]
    require(len(rng) == len(after_rng) and all(torch.equal(a, b) for a, b in zip(rng, after_rng)), 'RNG changed')
    require(torch.cuda.max_memory_allocated() < 20 * 2 ** 30, 'GPU allocation exceeded bound')
    return dict(passed=True, optimizer_updates=0, model_sha256=initial, query_ids=m['query_ids'],
        original_pool_queries=12, original_pool_document_exposures=sum(r['documents'] for r in results),
        synthetic_two_document_vjp_fixtures=3, initial_mean_D_gradient_l1=D, initial_mean_keep_gradient_l1=K,
        initial_keep_to_D_ratio=K / D, max_score_error=max(r['max_score_error'] for r in results),
        vjp=vjp, model_and_rng_unchanged=True, body_seconds=time.monotonic() - started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), training_enabled=False,
        context_qualified_comparison=context_qualified, original_failed_attempt_relabelled=False,
        locked_test_scored=False, quality_evaluation=False,
        runtime=dict(torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
                     gpu=torch.cuda.get_device_name(), tf32=False))


def worker(args):
    import torch
    from clearml import Task

    verify(args.research_root, args.run)
    require(io.sha(Path(__file__)) == execution.read(args.run / 'inputs.json')['source']['ng77_cuda_canary.py'],
            'worker differs from frozen source')
    require(args.phase == 'canary', 'only no-optimizer canary is enabled')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0077/no-optimizer-canary', reuse_last_task_id=False,
        auto_connect_frameworks=False, auto_connect_arg_parser=False, auto_connect_streams=False,
        auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, actual_start=True, post_hoc=False, offline=True,
                   remote_synced=False, closed=False)
    output = args.run / args.phase
    io.write(output / 'clearml-start.json', receipt)
    task.connect(dict(manifest_sha256=io.sha(args.run / 'inputs.json'), optimizer_updates=0))
    try:
        result = canary(args.research_root, args.run)
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


def supervise(args):
    verify(args.research_root, args.run)
    require(io.sha(Path(__file__)) == execution.read(args.run / 'inputs.json')['source']['ng77_cuda_canary.py'],
            'controller differs from frozen source')
    stopped = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda value, frame: stopped.append(value))
    with (args.research_root / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory, busy = pipeline.gpu_state(args.gpu)
        require(memory < 100 and busy == 0, 'requested GPU is occupied')
        io.write(args.run / 'controller-start.json', dict(pid=os.getpid(), gpu=args.gpu, started_unix=time.time()))
        try:
            args.worker_script = 'ng77_cuda_canary.py'
            pipeline.phase(args, 'canary', stopped, cuda=True, limit_seconds=1800)
            io.write(args.run / 'controller-exit.json', dict(status='canary_complete_no_training',
                                                          finished_unix=time.time()))
        except BaseException as exc:
            io.write(args.run / 'controller-exit.json', dict(status='failed_preserve_attempt',
                error=f'{type(exc).__name__}: {exc}', finished_unix=time.time()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'worker', 'supervise'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--phase')
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    require(args.run.parent == args.research_root / 'NG-0077', 'fresh NG77 canary unit required')
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    elif args.mode == 'worker':
        worker(args)
    else:
        supervise(args)


if __name__ == '__main__':
    main()
