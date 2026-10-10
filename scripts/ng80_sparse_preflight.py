"""Actual sparse scoring calibration and discarded-update NG80 canaries."""

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse

import ng80_campaign as campaign
import ng80_fit as fit
import ng80_lexical as native
from ng80_data import DOMAINS, require


PHASES = ('encode-reference', 'calibrate', 'train-S-canary', 'train-P-canary')
CONTINUATION_PHASES = PHASES[1:]
REFERENCE = 'NG-0080/sparse-preflight-v1'
LEXICAL_RUNTIME = 'NG-0080/lexical-runtime-v2'
INITIAL = 'b61b0e54e060a49bc6fd5bbab120871fdb4aa67c31483849943c42d1d14c1e9c'
SD = 'NG-0069/evaluation-v2/encode-initial/document.npz'
LD = 'NG-0069/lexical-v1/lexical-document.npz'
EXTRA = {
    SD: '932fcda5f5b261ef66d19d339e822180470d7151962678d6e30b2dae98513ed9',
    LD: 'c7c27a84f5f9fd3568e1c9166240719b652f7267840a8342cffa775f35dc223b',
    'NG-0008/ng8_lexical.py': '8bdffa41d5cff89c1cd6696c6c04ea4f738e51bc66a09e54a6516dc39e66ed86',
    'NG-0008/native/libtext.dylib': '49b9cfbbb6174c65031ca5e9ddd5d875d0aa79e2c8e609949fad136878d67ae5',
    'NG-0009/surfaces/vocabulary.json': '6eee063af51365679d21073a9eeed651dc654782835378c1997ce55b74e73112',
    'NG-0009/surfaces/train-rms.npy': '4d604b632677505fd8bd9efcb8dbcc07c4a3773a92d8b8e1e647bd325c1f843f',
    'NG-0069/teacher-v1/selection.json': '041272b6328e53a26aeb51f660136e0d29a4acb4621dff27340227854b4843dd',
    'NG-0069/lexical-v1/train-pools.json': '504a5e6a120f6a0498e83b8ebfc79423a4c0295b08c67c6fcee6c7866bc26541',
    'NG-0069/lexical-v1/train-lexical.json': '5fa1ebc684455afb18abbccfcf06a650a7ccf51f5df13bf1633e3e31e9eda83f',
}


def teacher_scale(student, teacher, domains):
    require(len(student) == len(teacher) == len(domains) > 0, 'scale alignment')
    rows = []
    for s, t, domain in zip(student, teacher, domains, strict=True):
        s, t = np.asarray(s, dtype=np.float64), np.asarray(t, dtype=np.float64)
        require(s.shape == t.shape and s.ndim == 1 and len(s) > 1
                and np.isfinite(s).all() and np.isfinite(t).all(), 'scale fields')
        s, t = s - s.mean(), t - t.mean()
        rows.append((float(np.mean(s * s)), float(np.mean(t * t)), float(np.mean(s * t)), domain))
    counts = {d: domains.count(d) for d in DOMAINS}
    require(len(set(counts.values())) == 1 and min(counts.values()) > 0
            and sum(counts.values()) == len(domains), 'balanced TRAIN scale required')
    moments = np.asarray([r[:3] for r in rows]).mean(0)
    require(np.isfinite(moments).all() and min(moments[:2]) > 0, 'degenerate scale')
    return dict(scale=float(np.sqrt(moments[0] / moments[1])),
        moments=dict(zip(('student_variance', 'teacher_variance', 'cross_covariance'), moments.tolist())),
        domains={d: np.asarray([r[:3] for r in rows if r[3] == d]).mean(0).tolist() for d in DOMAINS},
        queries=len(rows), loss='mean(center(s / a - t)^2)', loss_prefactor=1.,
        diagnostic_used=False, serving_rms_changed=False)


def fixture(pool, gold):
    require(len(pool) == len(set(pool)) and set(gold).issubset(pool),
            'fixture pool/positive alignment')
    positives = [d for d in pool if d in gold]
    others = [d for d in pool if d not in gold]
    require(positives and len(others) >= 3, 'four-document fixture unavailable')
    return [positives[0], *others[:3]]


def token_exposures(encoder, item):
    result = {}
    for role, texts in (('query', [item['query']]),
                        ('document', item['documents'])):
        encoded = encoder.tokenizer(
            texts, truncation=True,
            max_length=encoder.config['base'][role + '_input_tokens'])
        result[role] = sum(map(len, encoded['input_ids']))
    return result


def chunk_budget(seconds, updates=16):
    require(np.isfinite(seconds) and seconds > 0 and updates == 16, 'canary timing')
    candidates = {n: seconds / updates * n * 1.5 + 300
                  for n in (1024, 512, 256, 128, 64)}
    feasible = [n for n, estimate in candidates.items() if estimate < 5100]
    return dict(predicted_chunk_seconds=candidates,
        largest_safe_chunk=max(feasible, default=None), science_total_updates=3072,
        training_authorized_by_this_receipt=False)


def verify(base, run):
    m = campaign.read(run / 'inputs.json')
    require(m['protocol'] == 'NG80_actual_sparse_preflight_v4'
            and m['phases'] == list(CONTINUATION_PHASES)
            and m['reference_run'] == REFERENCE
            and m['lexical_runtime'] == LEXICAL_RUNTIME
            and m['icu_locale'] == native.ICU_LOCALE
            and m['canary_updates_per_arm'] == 16
            and m['scientific_training_updates'] == 0 and not m['locked_test_access'], 'preflight contract')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in m[key].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve())
                    and campaign.io.sha(path) == digest, 'changed input: ' + name)
    return m


def continuation_dependencies(base):
    reference = base / REFERENCE
    require(campaign.io.sha(reference / 'inputs.json') == native.PREFLIGHT_SHA,
            'reference preflight changed')
    campaign.closed(reference, 'encode-reference')
    deps = {REFERENCE + '/inputs.json': native.PREFLIGHT_SHA}
    proof = reference / 'encode-reference/complete.json'
    deps[str(proof.relative_to(base))] = campaign.io.sha(proof)
    deps.update({str((proof.parent / name).relative_to(base)): digest
                 for name, digest in campaign.read(proof)['files'].items()})
    runtime = base / LEXICAL_RUNTIME
    proof = runtime / 'complete.json'
    closed = campaign.read(proof)
    require(closed['passed'] is True, 'native runtime parity not passed')
    actual = {p.relative_to(runtime).as_posix() for p in runtime.rglob('*')
              if p.is_file() and p != proof}
    require(actual == set(closed['files']), 'native runtime inventory mismatch')
    parity = campaign.read(runtime / 'parity.json')
    require(parity['queries'] == 12288 and parity['exact_token_sequences']
            and parity['options'] == native.OPTIONS
            and not parity['locked_test_access'], 'native lexical parity contract')
    deps[str(proof.relative_to(base))] = campaign.io.sha(proof)
    deps.update({str((runtime / name).relative_to(base)): digest
                 for name, digest in closed['files'].items()})
    return deps


def freeze(base, run):
    parent = base / fit.PARENT
    require(run.parent == base / 'NG-0080'
            and campaign.io.sha(parent / 'inputs.json') == fit.PARENT_SHA, 'preflight parent')
    m = campaign.verify(base, parent)
    deps = {**m['dependencies'], **EXTRA, **continuation_dependencies(base)}
    for phase in campaign.PHASES:
        campaign.closed(parent, phase)
        proof = parent / phase / 'complete.json'
        deps[str(proof.relative_to(base))] = campaign.io.sha(proof)
        for name, digest in campaign.read(proof)['files'].items():
            deps[str((proof.parent / name).relative_to(base))] = digest
    for name, digest in deps.items():
        require(campaign.io.sha(base / name) == digest, 'dependency mismatch: ' + name)
    run.mkdir(exist_ok=False)
    for name in m['source']:
        shutil.copy2(parent / name, run / name)
    source = Path(__file__).resolve().parent
    for name in ('ng80_sparse_preflight.py', 'ng80_lexical.py',
                 'ng80_fit.py', 'ng80_training.py',
                 'ng71_execution.py', 'ng71_diagnostics.py'):
        shutil.copy2(source / name, run / name)
    for path in (source.parent / 'tests/test_ng80_sparse_preflight.py',
                 source.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0080-sparse-preflight.zh.md'):
        shutil.copy2(path, run / path.name)
    config = campaign.read(parent / 'ng0071-ranking-config.json')
    campaign.io.write(run / 'config.json', dict(
        training_enabled=True, canary_only=True,
        base=config['base'], optimizer=config['optimizer']))
    campaign.io.write(run / 'order.json', fit.order(
        campaign.read(parent / 'prepare/queries.json'), campaign.read(parent / 'prepare/selection.json')))
    for name in ('config.json', 'order.json'):
        require(campaign.io.sha(run / name) == campaign.io.sha(base / REFERENCE / name),
                'continuation changed scientific config/order')
    campaign.io.write(run / 'inputs.json', dict(
        protocol='NG80_actual_sparse_preflight_v4', phases=list(CONTINUATION_PHASES), dependencies=deps,
        reference_run=REFERENCE, lexical_runtime=LEXICAL_RUNTIME,
        icu_locale=native.ICU_LOCALE,
        source={p.name: campaign.io.sha(p) for p in run.iterdir() if p.is_file()},
        canary_updates_per_arm=16, scientific_training_updates=0,
        locked_test_access=False, automatic_promotion=False))
    verify(base, run)
    print('NG80_SPARSE_PREFLIGHT_FROZEN', campaign.io.sha(run / 'inputs.json'), flush=True)


def encode_reference(base, run, output):
    import torch
    from ng71_training import Encoder, parameter_hash

    model = Encoder(base, campaign.read(run / 'config.json'), 'cuda')
    before = parameter_hash(model.model)
    require(before == INITIAL, 'reference parent model changed')
    parent = base / fit.PARENT
    queries = campaign.read(parent / 'prepare/queries.json')
    panels = campaign.read(parent / 'prepare/panels.json')
    documents = campaign.rows(base / campaign.DATA / 'documents.jsonl')
    order = campaign.read(run / 'order.json')
    reference = sparse.load_npz(base / SD)
    require(reference.shape == (233009, 50265) and reference.dtype == np.float32
            and reference.has_canonical_format and (reference.data > 0).all(), 'initial sparse corpus')
    eligible = set(d for i in order for d in panels[i])
    targets = sorted(eligible, key=lambda d: hashlib.sha256(
        ('NG80/cache-context-v1/' + str(d)).encode()).hexdigest())[:64]
    groups = sorted(set(d // 4 for d in targets))
    with torch.no_grad():
        for group in groups:
            ids = list(range(group * 4, min(group * 4 + 4, len(documents))))
            actual = model.encode([documents[i]['text'] for i in ids], 'document').cpu().numpy()
            require(np.array_equal(actual, reference[ids].toarray()), 'same-context cache parity')
        parts, tick = [], time.monotonic()
        for start in range(0, len(order), 8):
            ids = order[start:start + 8]
            parts.append(sparse.csr_matrix(model.encode([queries[i]['query'] for i in ids], 'query')
                                          .cpu().numpy()))
            if start == 56:
                campaign.canary(output, 'query', time.monotonic() - tick, 64, len(order))
            if start % 512 == 0:
                print('NG80_SPARSE_REFERENCE', start + len(ids), len(order), flush=True)
    values = sparse.vstack(parts, format='csr')
    values.eliminate_zeros()
    values.sort_indices()
    require(values.has_canonical_format and np.isfinite(values.data).all()
            and (values.data > 0).all() and (np.diff(values.indptr) > 0).all(), 'sparse query output')
    sparse.save_npz(output / 'query.npz', values)
    campaign.io.write(output / 'cache-context.json', dict(targets=targets,
        original_groups=[list(range(g * 4, min(g * 4 + 4, len(documents)))) for g in groups],
        exact_support_and_values=True))
    require(parameter_hash(model.model) == before, 'preparation changed model')
    return dict(queries=len(order), query_nnz=int(values.nnz), model_sha256=before,
        document_cache_sha256=EXTRA[SD], same_context_parity=True, optimizer_updates=0)


def lexical_roundoff(actual, expected, query_nnz):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and actual.ndim == 1
            and np.isfinite(actual).all() and np.isfinite(expected).all()
            and (actual >= 0).all() and (expected >= 0).all()
            and isinstance(query_nnz, int) and query_nnz >= 0, 'lexical roundoff inputs')
    # Positive FP64 RMS sums, scaling and dot products on two platforms.
    # FMA/non-FMA paths need not be bit-identical; bound both rounding paths.
    steps = 4 * query_nnz + 8
    unit = np.finfo(np.float64).eps / 2
    require(steps * unit < .01, 'lexical roundoff operation bound')
    gamma = steps * unit / (1 - steps * unit)
    bound = 2 * gamma * np.maximum(actual, expected)
    error = np.abs(actual - expected)
    require((error <= bound).all(), 'historical TRAIN lexical score mismatch')
    return float(error.max(initial=0.)), float(bound.max(initial=0.))


def cached_lexical_parity(base, queries, lexical, documents):
    train = campaign.rows(base / campaign.DATA / 'train.jsonl')
    selected = campaign.read(base / 'NG-0069/teacher-v1/selection.json')['indices']
    positions = {train[i]['query_id']: row for row, i in enumerate(selected)}
    pools = campaign.read(base / 'NG-0069/lexical-v1/train-pools.json')
    expected = campaign.read(base / 'NG-0069/lexical-v1/train-lexical.json')
    require(len(positions) == len(pools) == len(expected) == 6144, 'legacy TRAIN alignment')
    checked, max_error, max_bound = 0, 0., 0.
    for row, query in enumerate(queries):
        if query['query_id'] not in positions:
            continue
        old = positions[query['query_id']]
        actual = (lexical[row].astype(np.float64) @ documents[pools[old]].T).toarray().ravel()
        error, bound = lexical_roundoff(actual, expected[old], int(lexical[row].nnz))
        max_error, max_bound = max(max_error, error), max(max_bound, bound)
        checked += 1
    require(checked > 0, 'no shared TRAIN lexical references')
    return dict(shared_train_queries=checked, fp64_forward_error_bound_passed=True,
                max_abs_error=max_error, max_allowed_roundoff=max_bound,
                old_score_bit_exact_required=False,
                diagnostic_or_dev_scores_used=False)


def calibrate(base, run, output):
    parent = base / fit.PARENT
    queries = campaign.read(parent / 'prepare/queries.json')
    panels, labels = [campaign.read(parent / f'prepare/{n}.json') for n in ('panels', 'labels')]
    order = campaign.read(run / 'order.json')
    q = sparse.load_npz(base / REFERENCE / 'encode-reference/query.npz').astype(np.float64)
    d = sparse.load_npz(base / SD).astype(np.float64)
    td = np.load(base / campaign.DENSE, mmap_mode='r')
    tq = np.load(parent / 'encode-teacher/query.npy', mmap_mode='r')
    spec = importlib.util.spec_from_file_location('ng80_lexical', base / 'NG-0008/ng8_lexical.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vocab = campaign.read(base / 'NG-0009/surfaces/vocabulary.json')
    rms = np.load(base / 'NG-0009/surfaces/train-rms.npy')[:len(vocab)]
    tokenizer = native.Tokenizer(base / LEXICAL_RUNTIME / 'libtext.so')
    lexical_parity = native.parity(tokenizer, [queries[i] for i in order],
                                  campaign.read(base / LEXICAL_RUNTIME / 'golden.json'))
    counts, _, oov = module.count_matrix([queries[i]['query'] for i in order], tokenizer, vocab)
    lexical = counts.multiply((.1 / module.scale(counts, rms))[:, None]).tocsr()
    ld = sparse.load_npz(base / LD).astype(np.float64)
    historical_parity = cached_lexical_parity(base, [queries[i] for i in order], lexical, ld)
    semantic, teacher, domains, max_error = [], [], [], 0.
    with (output / 'examples.jsonl').open('x') as stream:
        for row, i in enumerate(order):
            pool = panels[i]
            score = (q[row] @ d[pool].T).toarray().ravel()
            alternate = np.asarray(d[pool].multiply(q[row]).sum(1)).ravel()
            error = float(np.max(np.abs(score - alternate)))
            require(error <= 1e-12, 'independent sparse scores')
            max_error = max(max_error, error)
            target = td[pool].astype(np.float64) @ tq[i].astype(np.float64)
            lex = (lexical[row].astype(np.float64) @ ld[pool].T).toarray().ravel()
            semantic.append(score)
            teacher.append(target)
            domains.append(queries[i]['subset'])
            stream.write(json.dumps(dict(query_id=queries[i]['query_id'], query_index=i,
                pool=pool, teacher=target.tolist(), positive=[p in labels[i] for p in pool],
                lexical=lex.tolist(), initial_semantic=score.tolist()), allow_nan=False) + '\n')
    result = teacher_scale(semantic, teacher, domains)
    campaign.io.write(output / 'scale.json', result)
    sparse.save_npz(output / 'lexical-query.npz', lexical)
    return dict(result, lexical_parity=lexical_parity, historical_lexical_parity=historical_parity,
                icu_locale=native.ICU_LOCALE, lexical_query_oov=oov,
                max_independent_sparse_score_error=max_error,
                all_training_positives_preserved=True, scientific_training_updates=0)


def canary(base, run, output, arm):
    import torch
    import ng71_execution as execution
    from ng71_training import Encoder, parameter_hash
    from ng80_training import Readout, backward

    config = campaign.read(run / 'config.json')
    scale = campaign.read(run / 'calibrate/scale.json')['scale']
    queries = campaign.read(base / fit.PARENT / 'prepare/queries.json')
    documents = campaign.rows(base / campaign.DATA / 'documents.jsonl')
    examples = campaign.rows(run / 'calibrate/examples.jsonl')
    model = Readout(Encoder(base, config, 'cuda'), arm)
    optimizer = model.optimizer()
    initial = parameter_hash(model.encoder.model)
    require(initial == INITIAL, 'canary parent model changed')

    def sample(record, small=False):
        pool = record['pool']
        picked = fixture(pool, [p for p, yes in zip(pool, record['positive'], strict=True) if yes]) if small else pool
        positions = [pool.index(p) for p in picked]
        return dict(query=queries[record['query_index']]['query'],
            documents=[documents[p]['text'] for p in picked],
            teacher=[record['teacher'][j] for j in positions],
            positive=[record['positive'][j] for j in positions],
            lexical=[record['lexical'][j] for j in positions], scale=scale)

    errors = {}
    for domain in DOMAINS:
        record = next(r for r in examples if queries[r['query_index']]['subset'] == domain)
        item = sample(record, True)
        optimizer.zero_grad(set_to_none=True)
        backward(model, **item, accumulation=1, replay=False)
        full = [p.grad.clone() for p in model.parameters()]
        optimizer.zero_grad(set_to_none=True)
        backward(model, **item, accumulation=1, replay=True)
        numerator = sum(float((p.grad.double() - r.double()).square().sum())
                        for p, r in zip(model.parameters(), full, strict=True))
        denominator = sum(float(r.double().square().sum()) for r in full)
        require(denominator > 0, 'zero actual sparse fixture gradient')
        errors[domain] = (numerator / denominator) ** .5
        require(errors[domain] <= 1e-5, 'actual sparse VJP')
        del full
    times, candidates, nnz = [], 0, dict(query=0, document=0)
    tokens = dict(query=0, document=0)
    with (output / 'updates.jsonl').open('x') as stream:
        for step in range(16):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            batch = []
            for record in examples[4 * step:4 * (step + 1)]:
                item = sample(record)
                result = backward(model, **item)
                if step == 0:
                    actual = np.asarray(result['scores'])
                    baseline = np.asarray(record['initial_semantic'])
                    if arm == 'P':
                        baseline += record['lexical']
                    np.testing.assert_allclose(actual, baseline, rtol=2e-5, atol=2e-7)
                result.update(query_id=record['query_id'], pool=record['pool'], lexical=record['lexical'])
                candidates += result['documents']
                for role in nnz:
                    nnz[role] += result[role + '_nnz']
                work = token_exposures(model.encoder, item)
                for role in tokens:
                    tokens[role] += work[role]
                result['token_exposures'] = work
                batch.append(result)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            require(float(norm) > 0, 'zero sparse optimizer gradient')
            before = [p.detach().clone() for p in model.parameters()]
            optimizer.step()
            change = sum(float((p.detach().double() - old.double()).square().sum())
                         for p, old in zip(model.parameters(), before, strict=True)) ** .5
            del before
            require(np.isfinite(change) and change > 0, 'zero/nonfinite parameter update')
            times.append(time.monotonic() - tick)
            stream.write(json.dumps(dict(update=step + 1, gradient_norm=float(norm),
                clipped=bool(norm > 1), update_l2=change, seconds=times[-1],
                examples=batch), allow_nan=False) + '\n')
            stream.flush()
            print('NG80_SPARSE_CANARY', arm, step + 1, times[-1], flush=True)
    final = parameter_hash(model.encoder.model)
    require(final != initial, 'canary did not update model')
    model.encoder.model.save_pretrained(output / 'checkpoint')
    optimizer_sha = execution.save_optimizer(output / 'optimizer.pt', model.encoder, optimizer, 16)
    with torch.no_grad():
        probe = model.encode(sample(examples[0], True)['documents'], 'document').cpu()
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = Readout(Encoder(base, config, 'cuda', checkpoint=output / 'checkpoint'), arm)
    restored = reloaded.optimizer()
    require(parameter_hash(reloaded.encoder.model) == final, 'sparse checkpoint reload')
    require(execution.restore_optimizer(output / 'optimizer.pt', reloaded.encoder, restored, 16)
            == optimizer_sha, 'sparse optimizer reload')
    with torch.no_grad():
        require(torch.equal(probe, reloaded.encode(sample(examples[0], True)['documents'], 'document').cpu()),
                'sparse forward reload')
    return dict(arm=arm, initial_model_sha256=initial, model_sha256=final,
        optimizer_fingerprint=optimizer_sha, canary_updates=16, discard_updates=True,
        query_exposures=64, candidate_exposures=candidates, encoded_nnz_exposures=nnz,
        token_exposures=tokens, token_counts_include_replay=False,
        scale=scale, real_vjp_relative_l2=errors, exact_reload=True,
        initial_full_batch_score_parity=True, compute_seconds=sum(times),
        budget=chunk_budget(sum(times)), quality_evaluated=False,
        native_cost_evaluated=False, locked_test_access=False)


def worker(args):
    import torch
    from clearml import Task

    manifest = verify(args.research_root, args.run)
    require(args.phase in manifest['phases'], 'phase outside frozen continuation')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(80080)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0080/sparse-preflight/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output, base, run = args.run / args.phase, args.research_root, args.run
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    campaign.io.write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(input_sha256=campaign.io.sha(run / 'inputs.json'), phase=args.phase))
        if args.phase == 'encode-reference':
            result = encode_reference(base, run, output)
        else:
            campaign.closed(base / REFERENCE, 'encode-reference')
            if args.phase == 'calibrate':
                result = calibrate(base, run, output)
            else:
                campaign.closed(run, 'calibrate')
                result = canary(base, run, output, args.phase.split('-')[1])
        verify(base, run)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'worker', 'verify', 'supervise'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--gpu', type=int, choices=(0, 3))
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    elif args.mode == 'worker':
        require(args.phase in PHASES, 'phase required')
        worker(args)
    else:
        verify(args.research_root, args.run)
        if args.mode == 'supervise':
            require(args.gpu in (0, 3), 'explicit permitted GPU required')
            campaign.guard.supervise_graph(args, CONTINUATION_PHASES, 'ng80_sparse_preflight.py')
