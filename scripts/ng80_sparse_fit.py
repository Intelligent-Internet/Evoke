"""Fixed-budget S/P training after closed actual sparse canaries."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np

import ng80_campaign as campaign
import ng80_fit as dense_fit
import ng80_sparse_preflight as preflight
from ng80_data import require


PREFLIGHT = 'NG-0080/sparse-preflight-v4'
PREFLIGHT_SHA = '4ac4fca9342bf12fc25218f31b5bdc6b64da5e6ce193b9b4c410b6d5c268087b'
PHASES = ('train-1024', 'train-2048', 'train-3072')


def bounds(phase):
    require(phase in PHASES, 'unknown sparse training phase')
    end = int(phase.split('-')[1])
    return end - 1024, end


def closed(run, phase):
    campaign.closed(run, phase)
    folder = run / phase
    proof = campaign.read(folder / 'complete.json')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*')
              if p.is_file() and p != folder / 'complete.json'}
    require(actual == set(proof['files']), 'phase inventory changed')
    require({'results.json', 'exit.json', 'clearml.json'} <= actual,
            'unbound phase receipts')
    for name in actual:
        require((folder / name).resolve().is_relative_to(folder.resolve()),
                'phase path escape')
    return campaign.read(folder / 'results.json')


def common_budget(results):
    require(set(results) == {'S', 'P'}, 'both sparse canaries required')
    budgets = []
    for arm in ('S', 'P'):
        result = results[arm]
        require(result['arm'] == arm and result['discard_updates'] is True
                and result['canary_updates'] == 16
                and result['initial_model_sha256'] == preflight.INITIAL
                and result['exact_reload'] is True
                and result['initial_full_batch_score_parity'] is True
                and set(result['real_vjp_relative_l2']) == set(dense_fit.data.DOMAINS)
                and all(np.isfinite(v) and 0 <= v <= 1e-5
                        for v in result['real_vjp_relative_l2'].values()),
                'actual sparse canary did not qualify')
        budgets.append(preflight.chunk_budget(result['compute_seconds']))
    require(all(b['largest_safe_chunk'] == 1024 for b in budgets),
            'this runner requires the qualified common 1024-update chunk')
    return dict(zip(('S', 'P'), budgets, strict=True))


def verify(base, run):
    manifest = campaign.read(run / 'inputs.json')
    require(manifest['protocol'] == 'NG80_SP_fixed3072_v1'
            and manifest['phases'] == list(PHASES)
            and manifest['arm'] in ('S', 'P')
            and manifest['preflight_input_sha256'] == PREFLIGHT_SHA
            and manifest['initial_model_sha256'] == preflight.INITIAL
            and manifest['total_updates'] == 3072
            and manifest['query_exposures'] == 12288
            and manifest['common_chunk_updates'] == 1024
            and not manifest['locked_test_access'], 'sparse training contract')
    config = campaign.read(run / 'config.json')
    parent = campaign.read(base / PREFLIGHT / 'config.json')
    require(config == dict(parent, canary_only=False, arm=manifest['arm']),
            'science config differs from qualified canary')
    require(campaign.io.sha(run / 'order.json')
            == campaign.io.sha(base / PREFLIGHT / 'order.json'), 'changed order')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in manifest[key].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve())
                    and campaign.io.sha(path) == digest, 'changed input: ' + name)
    return manifest


def freeze(base, run, arm):
    require(arm in ('S', 'P') and run.parent == base / 'NG-0080', 'fresh sparse unit')
    parent = base / PREFLIGHT
    require(campaign.io.sha(parent / 'inputs.json') == PREFLIGHT_SHA, 'preflight changed')
    original = preflight.verify(base, parent)
    deps = {**original['dependencies'], PREFLIGHT + '/inputs.json': PREFLIGHT_SHA}
    for phase in preflight.CONTINUATION_PHASES:
        closed(parent, phase)
        proof = parent / phase / 'complete.json'
        deps[str(proof.relative_to(base))] = campaign.io.sha(proof)
        deps.update({str((proof.parent / name).relative_to(base)): digest
                     for name, digest in campaign.read(proof)['files'].items()})
    budgets = common_budget({a: campaign.read(parent / f'train-{a}-canary/results.json')
                             for a in ('S', 'P')})
    for name, digest in original['source'].items():
        deps[PREFLIGHT + '/' + name] = digest
    source = Path(__file__).resolve().parent
    for name in ('ng80_training.py', 'ng71_training.py', 'ng71_execution.py'):
        require(campaign.io.sha(source / name) == original['source'][name],
                'qualified gradient/optimizer implementation changed')
    run.mkdir(exist_ok=False)
    for name in original['source']:
        if name != 'config.json':
            shutil.copy2(parent / name, run / name)
    shutil.copy2(source / 'ng80_sparse_fit.py', run / 'ng80_sparse_fit.py')
    for path in (source.parent / 'tests/test_ng80_sparse_fit.py',
                 source.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0080-sparse-training.zh.md'):
        shutil.copy2(path, run / path.name)
    campaign.io.write(run / 'config.json', dict(
        campaign.read(parent / 'config.json'), canary_only=False, arm=arm))
    campaign.io.write(run / 'inputs.json', dict(
        protocol='NG80_SP_fixed3072_v1', arm=arm, phases=list(PHASES),
        preflight_input_sha256=PREFLIGHT_SHA, initial_model_sha256=preflight.INITIAL,
        dependencies=deps, source={p.name: campaign.io.sha(p)
                                  for p in run.iterdir() if p.is_file()},
        total_updates=3072, query_exposures=12288, common_chunk_updates=1024,
        canary_budgets=budgets, locked_test_access=False,
        terminal_evaluation_in_this_unit=False, automatic_expansion=False))
    verify(base, run)
    print('NG80_SPARSE_FIT_FROZEN', arm, campaign.io.sha(run / 'inputs.json'), flush=True)


def validate_examples(examples, indices, queries, panels, labels):
    require(len(examples) == len(indices) == len(set(indices)) == 12288,
            'TRAIN exposure count')
    for item, i in zip(examples, indices, strict=True):
        pool = item['pool']
        require(item['query_index'] == i and item['query_id'] == queries[i]['query_id']
                and pool == panels[i] and len(pool) == len(set(pool))
                and item['positive'] == [d in labels[i] for d in pool]
                and set(labels[i]) <= set(pool), 'TRAIN panel identity')
        for key in ('teacher', 'lexical', 'initial_semantic'):
            value = np.asarray(item[key])
            require(value.shape == (len(pool),) and np.isfinite(value).all(),
                    'TRAIN score alignment')


def load_model(base, run, arm, start):
    import ng71_execution as execution
    from ng71_training import Encoder, parameter_hash
    from ng80_training import Readout

    previous = run / f'train-{start}' if start else None
    prior = closed(run, previous.name) if previous else None
    model = Readout(Encoder(base, campaign.read(run / 'config.json'), 'cuda',
                           checkpoint=previous / 'checkpoint' if previous else None), arm)
    optimizer = model.optimizer()
    expected = prior['model_sha256'] if prior else preflight.INITIAL
    require(parameter_hash(model.encoder.model) == expected, 'sparse parent model changed')
    if prior:
        require(prior['arm'] == arm and prior['updates'] == start
                and prior['real_reload_exact'], 'wrong optimizer-chain parent')
        require(execution.restore_optimizer(previous / 'optimizer.pt', model.encoder,
                    optimizer, start) == prior['optimizer_sha256'], 'optimizer chain changed')
    return model, optimizer


def save_and_reload(base, run, output, model, optimizer, end, sample):
    import torch
    import ng71_execution as execution
    from ng71_training import Encoder, parameter_hash
    from ng80_training import Readout

    tick = time.monotonic()
    digest = parameter_hash(model.encoder.model)
    model.encoder.model.save_pretrained(output / 'checkpoint')
    opt = execution.save_optimizer(output / 'optimizer.pt', model.encoder, optimizer, end)
    with torch.no_grad():
        probes = {role: model.encode(texts, role).cpu()
                  for role, texts in sample.items()}
    reloaded = Readout(Encoder(base, campaign.read(run / 'config.json'), 'cuda',
                              checkpoint=output / 'checkpoint'), model.kind)
    restored = reloaded.optimizer()
    require(parameter_hash(reloaded.encoder.model) == digest, 'sparse checkpoint reload')
    require(execution.restore_optimizer(output / 'optimizer.pt', reloaded.encoder,
                restored, end) == opt, 'sparse optimizer reload')
    with torch.no_grad():
        for role, texts in sample.items():
            require(torch.equal(probes[role], reloaded.encode(texts, role).cpu()),
                    'sparse role output reload')
    del restored, reloaded
    gc.collect()
    torch.cuda.empty_cache()
    return dict(model_sha256=digest, optimizer_sha256=opt, updates=end,
                real_reload_exact=True, checkpoint_reload_seconds=time.monotonic() - tick)


def execute(base, run, output, phase):
    import torch
    from ng71_training import parameter_hash
    from ng80_training import backward

    start, end = bounds(phase)
    arm = campaign.read(run / 'inputs.json')['arm']
    parent = base / dense_fit.PARENT
    queries = campaign.read(parent / 'prepare/queries.json')
    panels, labels = [campaign.read(parent / f'prepare/{n}.json') for n in ('panels', 'labels')]
    documents = campaign.rows(base / campaign.DATA / 'documents.jsonl')
    examples = campaign.rows(base / PREFLIGHT / 'calibrate/examples.jsonl')
    validate_examples(examples, campaign.read(run / 'order.json'), queries, panels, labels)
    scale = campaign.read(base / PREFLIGHT / 'calibrate/scale.json')['scale']
    model, optimizer = load_model(base, run, arm, start)
    initial = parameter_hash(model.encoder.model)
    times, losses, candidates = [], [], 0
    tokens, nnz = dict(query=0, document=0), dict(query=0, document=0)
    tick = time.monotonic()
    with (output / 'updates.jsonl').open('x') as stream:
        for step in range(start, end):
            update_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            batch = []
            for record in examples[4 * step:4 * (step + 1)]:
                item = dict(query=queries[record['query_index']]['query'],
                    documents=[documents[d]['text'] for d in record['pool']],
                    teacher=record['teacher'], positive=record['positive'],
                    lexical=record['lexical'], scale=scale)
                result = backward(model, **item)
                if step == 0:
                    baseline = np.asarray(record['initial_semantic'])
                    if arm == 'P':
                        baseline = baseline + record['lexical']
                    np.testing.assert_allclose(result['scores'], baseline, rtol=2e-5, atol=2e-7)
                work = preflight.token_exposures(model.encoder, item)
                result.update(query_id=record['query_id'], query_index=record['query_index'],
                              pool=record['pool'], lexical=record['lexical'], token_exposures=work)
                candidates += result['documents']
                for role in tokens:
                    tokens[role] += work[role]
                    nnz[role] += result[role + '_nnz']
                batch.append(result)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            require(float(norm) > 0, 'zero complete-batch gradient')
            before = [p.detach().clone() for p in model.parameters()]
            optimizer.step()
            change = sum(float((p.detach().double() - old.double()).square().sum())
                         for p, old in zip(model.parameters(), before, strict=True)) ** .5
            del before
            require(np.isfinite(change) and change > 0, 'invalid parameter update')
            times.append(time.monotonic() - update_start)
            losses.append(float(np.mean([r['loss'] for r in batch])))
            stream.write(json.dumps(dict(update=step + 1, gradient_norm=float(norm),
                clipped=bool(norm > 1), update_l2=change, seconds=times[-1],
                loss=losses[-1], examples=batch), allow_nan=False) + '\n')
            if step - start == 15:
                campaign.canary(output, 'updates', time.monotonic() - tick, 16, end - start)
            if (step + 1) % 64 == 0:
                stream.flush()
                print('NG80_SPARSE_FIT', arm, step + 1, '/', end, losses[-1], flush=True)
    require(parameter_hash(model.encoder.model) != initial, 'sparse model did not update')
    first = examples[4 * start]
    sample = dict(query=[queries[first['query_index']]['query']],
                  document=[documents[d]['text'] for d in first['pool'][:4]])
    result = save_and_reload(base, run, output, model, optimizer, end, sample)
    return dict(result, arm=arm, start_update=start, initial_model_sha256=initial,
        query_exposures=4 * (end - start), candidate_exposures=candidates,
        token_exposures=tokens, encoded_nnz_exposures=nnz, scale=scale,
        token_counts_include_replay=False, compute_seconds=sum(times),
        mean_loss=float(np.mean(losses)), training_and_checkpoint_seconds=time.monotonic() - tick,
        canary_updates_reused=False, quality_evaluated=False,
        native_cost_evaluated=False, locked_test_access=False)


def worker(args):
    import torch
    from clearml import Task

    tick = time.monotonic()
    manifest = verify(args.research_root, args.run)
    bounds(args.phase)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(80080)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0080/sparse-' +
        manifest['arm'] + '/' + args.phase, reuse_last_task_id=False,
        auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True,
                   remote_synced=False, closed=False)
    campaign.io.write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(input_sha256=campaign.io.sha(args.run / 'inputs.json'), phase=args.phase))
        result = execute(args.research_root, args.run, output, args.phase)
        verify(args.research_root, args.run)
        campaign.io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        close_start = time.monotonic()
        task.close()
        receipt.update(closed=True, close_seconds=time.monotonic() - close_start,
                       worker_seconds=time.monotonic() - tick)
        campaign.io.write(output / 'clearml.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'worker', 'supervise'))
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--arm', choices=('S', 'P'))
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--gpu', type=int, choices=(0, 3))
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run, args.arm)
    elif args.mode == 'worker':
        worker(args)
    else:
        verify(args.research_root, args.run)
        if args.mode == 'supervise':
            require(args.gpu in (0, 3), 'explicit permitted GPU required')
            campaign.guard.supervise_graph(args, PHASES, 'ng80_sparse_fit.py')
