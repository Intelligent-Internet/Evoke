"""Frozen, bounded disposable AdamW-step and TRAIN margin probes."""

import argparse
from collections import Counter
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
import ng71_training as training
import ng74_uniform_control as uniform
import ng75_update_math as math_probe


require = math_probe.require
PARENT = 'NG-0074/uniform-v1'
INPUT_SHA = '7465eca9d3d4decced2ea58d7a483bd0507d8f0f3c8d51ed5b5aed4c616706f2'
REVIEW_SHA = '456592dd5fc1f8c86a16ff3fce80b6e07eb0b5776f734939508a57f03083cf59'
MAC_REVIEW_SHA = '7d9824833a12cf0ff751af9388124ec790e621e9c1051df9dec12eb023fdb1d2'
CASES = ('probe-D-1', 'probe-U-1', 'probe-D-97', 'probe-U-97')
PHASES = (*CASES, 'review')


def teacher_pair(record, positive, rival):
    if record is None or positive not in record['pool'] or rival not in record['pool']:
        return dict(status='unobserved', margin=None)
    values = record.get('teacher_scores', record.get('scores'))
    margin = values[record['pool'].index(positive)] - values[record['pool'].index(rival)]
    return dict(status='agrees' if margin > 0 else 'opposes' if margin < 0 else 'tie', margin=margin)


def verify_files(root, files):
    for name, digest in files.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and io.sha(path) == digest,
                'frozen input changed: ' + name)


def prepare_data(base):
    parent = base / uniform.PARENT
    selection = execution.read(parent / 'selection.json')
    order = execution.read(parent / 'training-order.json')
    initial = {r['query_id']: r for r in pipeline.rows(parent / 'rank-initial/rankings.jsonl')
               if r['split'] == 'TRAIN'}
    sentinels = math_probe.select_sentinel([initial[q] for q in selection['sentinel']])
    sentinel_ids = [r['query_id'] for r in sentinels]
    batch_ids = {start: order[start * 4:(start + 1) * 4] for start in (0, 96)}
    identities = set(sentinel_ids) | {q for ids in batch_ids.values() for q in ids}
    require(not set(sentinel_ids).intersection(selection['pilot']), 'sentinel gradient overlap')
    lexical = base / 'NG-0069/lexical-v1'
    rows = execution.read(lexical / 'queries.json')
    queries = {r['query_id']: dict(r, lexical_index=i) for i, r in enumerate(rows)
               if r['query_id'] in identities}
    require(set(queries) == identities and all(q['split'] == 'TRAIN' for q in queries.values()),
            'TRAIN identity missing or forbidden split')
    labels = execution.read(lexical / 'labels.json')
    witnesses, cases, wanted = {}, {}, set()
    for start, folder in ((0, base / 'NG-0071/pilot-step0-v1'), (96, parent / 'snapshot-96')):
        witnesses[start] = {r['query_id']: r for r in pipeline.rows(folder / 'witnesses.jsonl')}
        for identity in batch_ids[start]:
            wanted.update(witnesses[start][identity]['pool'])
    for query in queries.values():
        row = initial[query['query_id']]
        require(row['gold_ids'] == labels[query['lexical_index']], 'all-positive labels differ')
        wanted.update(d for pair in math_probe.initial_pairs(row) for d in pair)
    documents = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                documents[i] = json.loads(line)
    require(i + 1 == 233009 and set(documents) == wanted, 'canonical corpus mapping incomplete')
    lq, ld = [sparse.load_npz(lexical / f'lexical-{r}.npz').astype(np.float64)
              for r in ('query', 'document')]
    history = {r['query_id']: r for r in execution.read(base / 'NG-0070/provenance-v1/pair-provenance.json')}
    targets = {r['query_id']: r for r in execution.read(base / 'NG-0069/teacher-v1/targets.json')}
    probes = {}
    for identity, query in queries.items():
        lineage = history[identity]
        require(lineage['status'] == 'resolved', 'unresolved probe provenance')
        origins = {r['document_key']: r['provenance'] for r in lineage['pairs']}
        pairs = math_probe.initial_pairs(initial[identity])
        require(set(origins) == {documents[g]['key'] for g in initial[identity]['gold_ids']},
                'positive lineage mapping changed')
        chosen = sorted({d for pair in pairs for d in pair})
        values = (lq.getrow(query['lexical_index']) @ ld[chosen].T).toarray().ravel()
        lex = dict(zip(chosen, values.tolist(), strict=True))
        probes[identity] = [dict(query_id=identity, split='TRAIN', domain=query['subset'],
            surface='TRAIN_SENTINEL' if identity in sentinel_ids else 'TRAIN_CURRENT_BATCH',
            positive_id=p, rival_id=n, origin=origins[documents[p]['key']],
            lexical_margin=lex[p] - lex[n], rivals_are_judged_negative=False,
            teacher_original=teacher_pair(targets.get(identity), p, n),
            evidence_span_visible='unknown_no_span_judgments') for p, n in pairs]
    sentinel_count = sum(len(probes[q]) for q in sentinel_ids)
    require(sentinel_count <= 128, 'sentinel pair bound exceeded before replay')
    for name in CASES:
        _, arm, step = name.split('-')
        start = int(step) - 1
        source = parent if arm == 'D' else base / PARENT
        recorded = pipeline.rows(source / f'train-{arm}-{start + 96}/progress.jsonl')[0]
        require(recorded['step'] == int(step) and recorded['reference_update'] == start,
                'wrong actual optimizer step')
        require([r['query_id'] for r in recorded['examples']] == batch_ids[start], 'batch order changed')
        ids = sentinel_ids + batch_ids[start]
        cases[name] = dict(arm=arm, start=start, batch_ids=batch_ids[start], expected=recorded,
            witnesses=[witnesses[start][q] for q in batch_ids[start]],
            probes=[dict(row, teacher_reference=teacher_pair(witnesses[start].get(q),
                row['positive_id'], row['rival_id'])) for q in ids for row in probes[q]])
    for query in queries.values():
        query['text_sha256'] = hashlib.sha256(query['query'].encode()).hexdigest()
    return dict(queries=queries, documents={str(k): v for k, v in documents.items()},
                document_text_sha256={str(k): hashlib.sha256(v['text'].encode()).hexdigest()
                                      for k, v in documents.items()},
                sentinel_ids=sentinel_ids, sentinel_pairs=sentinel_count, cases=cases,
                new_inference=False, locked_test_access=False)


def validate_data(data):
    require(set(data['cases']) == set(CASES) and data['locked_test_access'] is False,
            'probe case graph changed')
    require(data['new_inference'] is False, 'preparation must not infer')
    require(len(data['sentinel_ids']) == 24 and len(set(data['sentinel_ids'])) == 24,
            'sentinel count changed')
    for identity, query in data['queries'].items():
        require(query['query_id'] == identity and query['split'] == 'TRAIN'
                and query['subset'] in math_probe.DOMAINS
                and hashlib.sha256(query['query'].encode()).hexdigest()
                == query['text_sha256'], 'probe text/split changed')
    require(set(data['documents']) == set(data['document_text_sha256']), 'document hash coverage changed')
    require(Counter(data['queries'][q]['subset'] for q in data['sentinel_ids'])
            == dict.fromkeys(math_probe.DOMAINS, 8), 'sentinel domain balance changed')
    for key, doc in data['documents'].items():
        require(hashlib.sha256(doc['text'].encode()).hexdigest() == data['document_text_sha256'][key],
                'canonical document text changed')
    for name, case in data['cases'].items():
        _, arm, step = name.split('-')
        require(case['arm'] == arm and case['start'] == int(step) - 1
                and case['expected']['step'] == int(step)
                and case['expected']['reference_update'] == case['start'], 'wrong replay step')
        require(len(set(case['batch_ids'])) == 4 and not set(case['batch_ids']).intersection(data['sentinel_ids']),
                'actual batch or no-sentinel-gradient boundary changed')
        require([w['query_id'] for w in case['witnesses']] == case['batch_ids']
                == [e['query_id'] for e in case['expected']['examples']], 'witness/exposure order changed')
        require(all(w['split'] == 'TRAIN' and all(str(d) in data['documents'] for d in w['pool'])
                    for w in case['witnesses']), 'forbidden or incomplete witness')
        rows = case['probes']
        require(len({(r['query_id'], r['positive_id'], r['rival_id']) for r in rows}) == len(rows),
                'duplicate probe identity')
        sentinel = [r for r in rows if r['surface'] == 'TRAIN_SENTINEL']
        require(len(sentinel) == data['sentinel_pairs'] <= 128, 'sentinel denominator changed')
        require(set(r['query_id'] for r in sentinel) == set(data['sentinel_ids']), 'sentinel coverage changed')
        require(all(r['split'] == 'TRAIN' and r['query_id'] in data['queries'] for r in rows),
                'forbidden probe query')
        require(set(r['query_id'] for r in rows) == set(data['sentinel_ids'] + case['batch_ids']),
                'same-batch probe coverage changed')
        for row in rows:
            require(row['domain'] == data['queries'][row['query_id']]['subset']
                    and row['surface'] == ('TRAIN_SENTINEL' if row['query_id'] in data['sentinel_ids']
                                          else 'TRAIN_CURRENT_BATCH'), 'probe classification changed')
            require(row['positive_id'] != row['rival_id']
                    and all(type(row[k]) is int and str(row[k]) in data['documents']
                            for k in ('positive_id', 'rival_id')), 'invalid pair documents')
            require(row['origin'] in ('original_positive', 'added_vs_original')
                    and np.isfinite(row['lexical_margin'])
                    and row['rivals_are_judged_negative'] is False, 'pair evidence changed')


def verify(base, run):
    frozen = execution.read(run / 'inputs.json')
    require(frozen['protocol'] == 'NG75_actual_displacement_v1' and frozen['phases'] == list(PHASES)
            and frozen['scientific_training_updates'] == 0 and frozen['disposable_optimizer_updates'] == 4
            and frozen['locked_test_access'] is False, 'diagnostic contract changed')
    verify_files(base, frozen['dependencies'])
    verify_files(run, frozen['source'])
    validate_data(execution.read(run / 'probe-data.json'))
    return frozen


def freeze(args):
    base, run = args.research_root, args.run
    require(run.parent == base / 'NG-0075', 'fresh NG75 unit required')
    require(io.sha(base / PARENT / 'inputs.json') == INPUT_SHA
            and io.sha(base / PARENT / 'review/complete.json') == REVIEW_SHA, 'NG74 anchor changed')
    old = uniform.verify(base, base / PARENT)
    for phase in uniform.PHASES:
        execution.sealed(base / PARENT / phase)
    qa = base / 'NG-0074/uniform-v1-mac-review-v1'
    require(io.sha(qa / 'complete.json') == MAC_REVIEW_SHA, 'independent Mac review missing')
    receipt = execution.read(qa / 'complete.json')
    verify_files(qa, receipt['files'])
    require(receipt['passed'] and execution.read(qa / 'verification.json')['inputs_and_outputs_unchanged'],
            'NG74 Mac review incomplete')
    dependencies = dict(old['dependencies'])
    dependencies.update({PARENT + '/' + str(p.relative_to(base / PARENT)): io.sha(p)
                         for p in (base / PARENT).rglob('*') if p.is_file()})
    data = prepare_data(base)
    validate_data(data)
    run.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    names = [p.name for p in source.glob('ng71_*.py')]
    names += ['ng74_uniform_control.py', 'review_ng74_uniform_control.py', 'review_ng71_pilot.py',
              'audit_ng70_provenance.py', 'analyze_ng66_learning_curves.py',
              'ng75_update_math.py', 'ng75_update_diagnosis.py']
    for name in names:
        shutil.copy2(source / name, run / name)
    for p in (source.parent / 'tests').glob('*ng75*.py'):
        shutil.copy2(p, run / p.name)
    plan = source.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0075-parameter-update-diagnosis-plan.zh.md'
    shutil.copy2(plan, run / plan.name)
    shutil.copytree(qa, run / 'parent-mac-review')
    io.write(run / 'probe-data.json', data)
    io.write(run / 'inputs.json', dict(protocol='NG75_actual_displacement_v1', phases=list(PHASES),
        dependencies=dependencies, source={str(p.relative_to(run)): io.sha(p)
        for p in run.rglob('*') if p.is_file()}, scientific_training_updates=0,
        disposable_optimizer_updates=4, locked_test_access=False))
    verify(base, run)


def compare_replay(actual, expected):
    for key in ('gradient_before_clip', 'update_l2'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-5, atol=2e-7)
    require(len(actual['examples']) == len(expected['examples']) == 4, 'replay exposure count')
    for a, b in zip(actual['examples'], expected['examples'], strict=True):
        for key in ('scores', 'score_gradient', 'loss'):
            np.testing.assert_allclose(a[key], b[key], rtol=2e-5, atol=2e-7)
        for key in ('documents', 'objective', 'eligible_pairs', 'supervised_positives', 'vjp_replay_exact'):
            require(a[key] == b[key], 'replay trace changed: ' + key)


def score_pressure(spec, case, actual):
    if spec['query_id'] not in case['batch_ids']:
        return None
    j = case['batch_ids'].index(spec['query_id'])
    pool = case['witnesses'][j]['pool']
    if spec['positive_id'] not in pool or spec['rival_id'] not in pool:
        return None
    gradient = actual['examples'][j]['score_gradient']
    return -gradient[pool.index(spec['positive_id'])] + gradient[pool.index(spec['rival_id'])]


def probe_case(base, run, name, task):
    import torch
    data = execution.read(run / 'probe-data.json')
    case = data['cases'][name]
    arm, start = case['arm'], case['start']
    source = base / (uniform.PARENT if arm == 'D' else PARENT)
    config = execution.read(source / 'config.json')
    previous = source / f'train-{arm}-96' if start else None
    prior = execution.sealed(previous) if previous else None
    encoder = training.Encoder(base, config, device='cuda',
        checkpoint=previous / 'checkpoint' if previous else None)
    initial = training.parameter_hash(encoder.model)
    require(initial == (prior['model_sha256'] if prior else config['base']['state_sha256']),
            'wrong replay model')
    optimizer = training.make_optimizer(encoder)
    fingerprint = execution.restore_optimizer(previous / 'optimizer.pt', encoder, optimizer, start) if start else None
    if prior:
        require(fingerprint == prior['optimizer_fingerprint'], 'wrong replay moments')
    named, groups = math_probe.parameter_layout(encoder.model, optimizer)
    before = math_probe.copy_parameters(named)
    documents = {int(k): v for k, v in data['documents'].items()}
    queries = data['queries']
    examples = [execution.example(w, queries[q], documents, config, arm)
                for w, q in zip(case['witnesses'], case['batch_ids'], strict=True)]
    actual = training.optimizer_step(encoder, optimizer, examples)
    compare_replay(actual, case['expected'])
    after = math_probe.copy_parameters(named)
    delta = math_probe.displacement(before, after)
    np.testing.assert_allclose(sum(float(d.double().square().sum()) for d in delta) ** .5,
                               actual['update_l2'], rtol=0, atol=1e-12)
    after_sha = training.parameter_hash(encoder.model)
    io.write(run / name / 'replay.json', dict(initial_model_sha256=initial, after_model_sha256=after_sha,
        initial_optimizer_fingerprint=fingerprint, actual=actual, recorded_step=start + 1,
        recorded_trace_matches=True, full_historical_step_state_sha_available=False))
    del optimizer
    math_probe.restore_parameters(named, before)
    fixed_seconds = time.time() - execution.read(run / name / 'started.json')['started_unix']
    require(fixed_seconds >= 0, 'wall clock moved before phase start')
    timings, rows = [], []
    with (run / name / 'margins.jsonl').open('x') as stream:
        for index, spec in enumerate(case['probes']):
            tick = time.monotonic()
            query = queries[spec['query_id']]
            require(query['split'] == 'TRAIN', 'forbidden inference split')
            texts = [documents[d]['text'] for d in (spec['positive_id'], spec['rival_id'])]
            q = encoder.encode([query['query']], 'query')[0]
            d = encoder.encode(texts, 'document')
            with torch.no_grad():
                q_check = encoder.encode([query['query']], 'query')[0]
                d_check = encoder.encode(texts, 'document')
            require(torch.equal(q.detach(), q_check) and torch.equal(d.detach(), d_check),
                    'no-update forward is not bit-exact')
            projected = math_probe.margin_projection(q, d[0], d[1], named, delta, groups)
            before_margin = projected['semantic_margin'] + spec['lexical_margin']
            del q, d, q_check, d_check
            math_probe.restore_parameters(named, after)
            with torch.no_grad():
                q1 = encoder.encode([query['query']], 'query')[0].double()
                d1 = encoder.encode(texts, 'document').double()
                after_margin = float(q1 @ (d1[0] - d1[1])) + spec['lexical_margin']
            math_probe.restore_parameters(named, before)
            del q1, d1
            row = dict(spec, case=name, **projected,
                **math_probe.describe_change(before_margin, after_margin, projected['predicted'], 0.))
            row['same_batch_score_pressure'] = score_pressure(spec, case, actual)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            rows.append(row)
            timings.append(time.monotonic() - tick)
            if index + 1 in (1, 8):
                predicted = fixed_seconds + max(timings) * len(case['probes']) * 1.5 + 360
                io.write(run / name / f'canary-{index + 1}.json', dict(
                    probes=index + 1, total_probes=len(case['probes']), predicted_seconds=predicted,
                    fixed_setup_seconds=fixed_seconds, slowest_probe_seconds=max(timings),
                    passed=predicted < 1740, includes_closure_reserve_seconds=360))
                require(predicted < 1740, 'probe canary exceeds unchanged phase bound')
            if index == 0 or (index + 1) % 8 == 0:
                print('NG75_PROBE', name, index + 1, len(case['probes']), sum(timings), flush=True)
            task.get_logger().report_scalar('probe', 'finite_change', row['actual_change'], index)
    require(training.parameter_hash(encoder.model) == initial, 'disposable probe did not restore original parameters')
    return dict(passed=True, case=name, probes=len(rows), initial_model_sha256=initial,
        initial_optimizer_fingerprint=fingerprint, disposable_optimizer_updates=1,
        scientific_training_updates=0, no_update_forward_bit_exact=True,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        summary=math_probe.summarize(rows), locked_test_scored=False,
        quality_evaluation=False, native_cost_evaluated=False)


def review(base, run):
    verify(base, run)
    data = execution.read(run / 'probe-data.json')
    summaries = {}
    for name in CASES:
        result = execution.sealed(run / name)
        actual = execution.read(run / name / 'replay.json')
        case = data['cases'][name]
        compare_replay(actual['actual'], case['expected'])
        require(result['passed'] is True and result['case'] == name
                and result['disposable_optimizer_updates'] == 1 and result['scientific_training_updates'] == 0
                and result['no_update_forward_bit_exact'] is True
                and result['locked_test_scored'] is False and result['quality_evaluation'] is False,
                'case diagnostic boundary changed')
        source = base / (uniform.PARENT if case['arm'] == 'D' else PARENT)
        prior = execution.sealed(source / f"train-{case['arm']}-96") if case['start'] else None
        initial_sha = prior['model_sha256'] if prior else execution.read(source / 'config.json')['base']['state_sha256']
        fingerprint = prior['optimizer_fingerprint'] if prior else None
        require(actual['recorded_step'] == case['start'] + 1
                and actual['initial_model_sha256'] == result['initial_model_sha256'] == initial_sha
                and actual['initial_optimizer_fingerprint'] == result['initial_optimizer_fingerprint'] == fingerprint
                and actual['after_model_sha256'] != initial_sha
                and actual['recorded_trace_matches'] is True
                and actual['full_historical_step_state_sha_available'] is False, 'replay state proof changed')
        rows = pipeline.rows(run / name / 'margins.jsonl')
        specs = data['cases'][name]['probes']
        require(len(rows) == len(specs) == result['probes'], 'probe coverage incomplete')
        for row, spec in zip(rows, specs, strict=True):
            require(all(row[k] == value for k, value in spec.items()), 'probe identities/metadata changed')
            expected = math_probe.describe_change(row['before_margin'], row['after_margin'],
                                                  row['predicted'], row['no_update_error'])
            require(all(row[k] == value for k, value in expected.items()), 'finite-change reduction differs')
            require(row['case'] == name and row['no_update_error'] == 0
                    and row['before_margin'] == row['semantic_margin'] + row['lexical_margin']
                    and row['same_batch_score_pressure'] == score_pressure(spec, case, actual['actual']),
                    'probe margin/pressure attribution changed')
            require(all(np.isfinite(row[k]) and row[k] >= 0 for k in
                        ('gradient_chain_max_abs', 'gradient_chain_relative_l2')), 'invalid chain diagnostic')
            np.testing.assert_allclose(sum(sum(v.values()) for v in row['contributions'].values()),
                                       row['predicted'], rtol=2e-5, atol=2e-7)
        summary = math_probe.summarize(rows)
        require(summary == result['summary'], 'summary denominator changed')
        summaries[name] = summary
    return dict(passed=True, cases=summaries, scientific_training_updates=0,
        disposable_optimizer_updates=4, independent_metric_reduction=True,
        jacobians_independently_rerun=False, locked_test_scored=False,
        quality_evaluation=False, causal_share_of_full_training_measured=False,
        automatic_training_authorized=False, overall_goal_qualified=False)


def worker(args):
    import torch
    from clearml import Task
    verify(args.research_root, args.run)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0075/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    try:
        io.write(output / 'clearml-start.json', receipt)
        task.connect(dict(input_sha256=io.sha(args.run / 'inputs.json'), phase=args.phase))
        result = (review(args.research_root, args.run) if args.phase == 'review'
                  else probe_case(args.research_root, args.run, args.phase, task))
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
    import fcntl
    verify(args.research_root, args.run)
    stopped, completed = [], []
    handlers = {s: signal.signal(s, lambda number, frame: stopped.append(number))
                for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (args.research_root / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            memory, utilization = pipeline.gpu_state(args.gpu)
            require(memory < 100 and utilization == 0, 'GPU occupied; do not evict')
            io.write(args.run / 'controller-start.json', dict(pid=os.getpid(), gpu=args.gpu, started_unix=time.time()))
            args.worker_script = Path(__file__).name
            try:
                for name in PHASES:
                    print('NG75_PHASE_START', name, flush=True)
                    pipeline.phase(args, name, stopped, cuda=name != 'review', limit_seconds=1800)
                    completed.append(name)
                    print('NG75_PHASE_COMPLETE', name, flush=True)
                io.write(args.run / 'controller-exit.json', dict(status='all_phases_complete',
                    completed=completed, finished_unix=time.time()))
            except BaseException as exc:
                io.write(args.run / 'controller-exit.json', dict(status='failed_preserve_attempt',
                    completed=completed, error=f'{type(exc).__name__}: {exc}', finished_unix=time.time()))
                raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3))
    args = parser.parse_args()
    args.research_root = args.research_root.resolve(strict=True)
    args.run = args.run.resolve()
    require(args.run.parent == args.research_root / 'NG-0075', 'new NG75 path required')
    if args.mode == 'freeze':
        freeze(args)
    else:
        verify(args.research_root, args.run)
        require(io.sha(Path(__file__)) == execution.read(args.run / 'inputs.json')['source'][Path(__file__).name],
                'controller is not frozen source')
        if args.mode == 'supervise':
            require(args.gpu is not None, 'explicit physical GPU required')
            supervise(args)
        elif args.mode == 'worker':
            require(args.phase is not None, 'explicit worker phase required')
            worker(args)


if __name__ == '__main__':
    main()
