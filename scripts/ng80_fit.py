"""Matched frozen/full dense-readout adaptation, not a product sparse model."""

import argparse
import gc
import hashlib
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np

import ng80_campaign as campaign
import ng80_data as data


PARENT = 'NG-0080/readout-v1'
PARENT_SHA = 'd8dca9c36ff49047c0fc0acfe0ee9cbcd607537a65596d9013300537f2e8dff1'
PHASES = ('train-canary', 'train-1024', 'train-2048', 'train-3072')


def order(queries, selection):
    groups = []
    for domain in data.DOMAINS:
        ids = [i for i in selection['train'] if queries[i]['subset'] == domain]
        ids.sort(key=lambda i: hashlib.sha256(
            ('NG80/fit-order-v1/' + queries[i]['query_id']).encode()).hexdigest())
        data.require(len(ids) == 4096, 'fixed training population changed')
        groups.append(ids)
    result = [i for group in zip(*groups, strict=True) for i in group]
    data.require(len(result) == len(set(result)) == 12288
                 and not set(result) & set(selection['diagnostic']), 'split/order changed')
    return result


def holder(model):
    import torch

    named = [('encoder__' + n.replace('.', '__'), p)
             for n, p in model.encoder.model.named_parameters() if p.requires_grad]
    named += [('projection__' + n, p) for n, p in model.projection.named_parameters()]
    data.require(len({n for n, _ in named}) == len(named), 'parameter name collision')
    return SimpleNamespace(model=torch.nn.ParameterDict(dict(named)))


def verify(base, run):
    manifest = campaign.read(run / 'inputs.json')
    data.require(manifest['protocol'] == 'NG80_FD_field_v1'
                 and manifest['phases'] == list(PHASES) and manifest['arm'] in ('F', 'D')
                 and manifest['parent_input_sha256'] == PARENT_SHA
                 and manifest['total_updates'] == 3072 and not manifest['locked_test_access'],
                 'training contract changed')
    config = campaign.read(run / 'config.json')
    data.require(config['training_enabled'] is True and config['arm'] == manifest['arm']
                 and config['scoring'] == 'normalized_dense_dot', 'training config changed')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in manifest[key].items():
            path = root / name
            data.require(path.resolve().is_relative_to(root.resolve()), 'path escape')
            data.require(campaign.io.sha(path) == digest, 'changed input: ' + name)
    return manifest


def freeze(base, run, arm):
    parent = base / PARENT
    data.require(arm in ('F', 'D') and run.parent == base / 'NG-0080', 'fresh F/D unit required')
    data.require(campaign.io.sha(parent / 'inputs.json') == PARENT_SHA, 'parent changed')
    original = campaign.verify(base, parent)
    dependencies = dict(original['dependencies'])
    for phase in campaign.PHASES:
        campaign.closed(parent, phase)
        proof = parent / phase / 'complete.json'
        dependencies[str(proof.relative_to(base))] = campaign.io.sha(proof)
        for name, digest in campaign.read(proof)['files'].items():
            dependencies[str((parent / phase / name).relative_to(base))] = digest
    dependencies[PARENT + '/inputs.json'] = PARENT_SHA
    queries = campaign.read(parent / 'prepare/queries.json')
    selected = campaign.read(parent / 'prepare/selection.json')
    run.mkdir(exist_ok=False)
    for name in original['source']:
        shutil.copy2(parent / name, run / name)
    source = Path(__file__).resolve().parent
    for name in ('ng80_fit.py', 'ng80_training.py', 'ng71_execution.py', 'ng71_diagnostics.py'):
        shutil.copy2(source / name, run / name)
    for path in (source.parent / 'tests/test_ng80_training.py',
                 source.parent / 'tests/test_ng80_fit.py',
                 source.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0080-dense-control.zh.md'):
        shutil.copy2(path, run / path.name)
    campaign.io.write(run / 'order.json', order(queries, selected))
    original_config = campaign.read(parent / 'ng0071-ranking-config.json')
    campaign.io.write(run / 'config.json', {
        'experiment': 'NG-0080', 'training_enabled': True, 'arm': arm,
        'scoring': 'normalized_dense_dot', 'initial_pooling': 'masked_mean',
        'base': {**original_config['base'], 'trainable':
                 'dense_readout_only' if arm == 'F' else 'trunk_and_dense_readout'},
        'optimizer': original_config['optimizer']})
    campaign.io.write(run / 'inputs.json', {
        'protocol': 'NG80_FD_field_v1', 'arm': arm, 'phases': list(PHASES),
        'parent_input_sha256': PARENT_SHA, 'dependencies': dependencies,
        'source': {p.name: campaign.io.sha(p) for p in run.iterdir() if p.is_file()},
        'total_updates': 3072, 'query_exposures': 12288, 'locked_test_access': False,
        'terminal_evaluation_in_this_unit': False, 'automatic_expansion': False})
    verify(base, run)
    print('NG80_FIT_FROZEN', arm, campaign.io.sha(run / 'inputs.json'), flush=True)


def load_model(base, run, previous=None):
    import torch
    import ng71_execution as execution
    from ng71_training import Encoder, parameter_hash
    from ng80_training import Readout

    config = campaign.read(run / 'config.json')
    arm = campaign.read(run / 'inputs.json')['arm']
    weights = np.load(base / PARENT / 'diagnose/mean-projection.npy')
    encoder = Encoder(base, config, device='cuda',
                      checkpoint=previous / 'checkpoint' if previous else None)
    model = Readout(encoder, arm, weights)
    optimizer = model.optimizer()
    if previous:
        prior = campaign.read(previous / 'results.json')
        model.projection.load_state_dict(torch.load(previous / 'projection.pt',
                                        weights_only=True, map_location='cuda'), strict=True)
        data.require(parameter_hash(encoder.model) == prior['model_sha256']
                     and parameter_hash(model.projection) == prior['projection_sha256'],
                     'checkpoint state changed')
        value = execution.restore_optimizer(previous / 'optimizer.pt', holder(model),
                                             optimizer, prior['updates'])
        data.require(value == prior['optimizer_sha256'], 'optimizer chain changed')
    return model, optimizer


def save_and_reload(base, run, output, model, optimizer, updates, sample):
    import torch
    import ng71_execution as execution
    from ng71_training import parameter_hash

    model_sha = parameter_hash(model.encoder.model)
    projection_sha = parameter_hash(model.projection)
    model.encoder.model.save_pretrained(output / 'checkpoint')
    torch.save(model.projection.state_dict(), output / 'projection.pt')
    opt_sha = execution.save_optimizer(output / 'optimizer.pt', holder(model), optimizer, updates)
    meta = {'model_sha256': model_sha, 'projection_sha256': projection_sha,
            'optimizer_sha256': opt_sha, 'updates': updates}
    # The reload helper reads this provisional state, never a completion marker.
    campaign.io.write(output / 'reload-state.json', meta)
    with torch.no_grad():
        before = model.encode(sample, 'document').detach().cpu()
    config = campaign.read(run / 'config.json')
    from ng71_training import Encoder
    from ng80_training import Readout
    reloaded = Readout(Encoder(base, config, 'cuda', checkpoint=output / 'checkpoint'),
                       model.kind, np.load(base / PARENT / 'diagnose/mean-projection.npy'))
    reloaded.projection.load_state_dict(torch.load(output / 'projection.pt',
                                       weights_only=True, map_location='cuda'), strict=True)
    restored = reloaded.optimizer()
    data.require(parameter_hash(reloaded.encoder.model) == model_sha
                 and parameter_hash(reloaded.projection) == projection_sha, 'reload changed model')
    data.require(execution.restore_optimizer(output / 'optimizer.pt', holder(reloaded),
                 restored, updates) == opt_sha, 'reload changed optimizer')
    with torch.no_grad():
        data.require(torch.equal(reloaded.encode(sample, 'document').cpu(), before),
                     'reload changed real document codes')
    del restored, reloaded
    gc.collect()
    torch.cuda.empty_cache()
    return meta


def execute(base, run, output, phase, task):
    import torch
    from ng71_training import parameter_hash
    from ng80_training import backward

    parent = base / PARENT
    queries = campaign.read(parent / 'prepare/queries.json')
    panels = campaign.read(parent / 'prepare/panels.json')
    labels = campaign.read(parent / 'prepare/labels.json')
    documents = campaign.rows(base / campaign.DATA / 'documents.jsonl')
    q = np.load(parent / 'encode-teacher/query.npy', mmap_mode='r')
    d = np.load(base / campaign.DENSE, mmap_mode='r')
    indices = campaign.read(run / 'order.json')
    is_canary = phase == 'train-canary'
    end = 1 if is_canary else int(phase.split('-')[1])
    start = 0 if is_canary else end - 1024
    previous = run / f'train-{start}' if start else None
    if not is_canary:
        campaign.closed(run, 'train-canary')
    if previous:
        campaign.closed(run, previous.name)
    model, optimizer = load_model(base, run, previous)
    initial = parameter_hash(model.encoder.model)
    initial_projection = parameter_hash(model.projection)
    tick = time.monotonic()

    def example(i, small=False):
        pool = panels[i][:4] if small else panels[i]
        return {'query': queries[i]['query'], 'documents': [documents[j]['text'] for j in pool],
                'teacher': d[pool].astype(np.float64) @ q[i].astype(np.float64),
                'positive': [j in labels[i] for j in pool], 'lexical': [0.] * len(pool), 'scale': 1.}

    vjp = None
    if is_canary:
        sample = example(indices[0], small=True)
        optimizer.zero_grad(set_to_none=True)
        backward(model, **sample, accumulation=1, replay=False)
        reference = [p.grad.clone() for p in model.parameters()]
        optimizer.zero_grad(set_to_none=True)
        backward(model, **sample, accumulation=1, replay=True)
        numerator = sum(float((p.grad.double() - ref.double()).square().sum())
                        for p, ref in zip(model.parameters(), reference, strict=True))
        denominator = sum(float(ref.double().square().sum()) for ref in reference)
        data.require(denominator > 0, 'zero real-base derivative')
        vjp = (numerator / denominator) ** .5
        data.require(vjp <= 1e-5, 'real-base VJP exceeds existing relative L2 tolerance')
        del reference
    token_work = {'query': 0, 'document': 0}
    candidates, losses = 0, []
    with (output / 'updates.jsonl').open('x') as stream:
        for step in range(start, end):
            optimizer.zero_grad(set_to_none=True)
            records = []
            for i in indices[4 * step:4 * (step + 1)]:
                item = example(i)
                record = backward(model, **item)
                record['query_id'] = queries[i]['query_id']
                records.append(record)
                candidates += len(item['documents'])
                for role, texts in (('query', [item['query']]), ('document', item['documents'])):
                    encoded = model.encoder.tokenizer(texts, truncation=True,
                        max_length=model.encoder.config['base'][role + '_input_tokens'])
                    token_work[role] += sum(map(len, encoded['input_ids']))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            data.require(float(norm) > 0, 'zero complete-batch gradient')
            optimizer.step()
            loss = float(np.mean([r['loss'] for r in records]))
            losses.append(loss)
            stream.write(__import__('json').dumps({'update': step + 1, 'loss': loss,
                         'gradient_norm': float(norm), 'queries': records}, allow_nan=False) + '\n')
            if step - start == 15:
                campaign.canary(output, 'updates', time.monotonic() - tick, 16, end - start)
            if (step + 1) % 64 == 0:
                stream.flush()
                print('NG80_FIT', model.kind, step + 1, '/', end, loss, flush=True)
                task.get_logger().report_scalar('training', 'field_mse', loss, step + 1)
    if model.kind == 'F':
        data.require(parameter_hash(model.encoder.model) == initial, 'frozen parent changed')
    else:
        data.require(parameter_hash(model.encoder.model) != initial, 'full trunk did not update')
    data.require(parameter_hash(model.projection) != initial_projection, 'readout did not update')
    result = save_and_reload(base, run, output, model, optimizer, end,
                             example(indices[4 * start], small=True)['documents'])
    result.update(arm=model.kind, start_update=start, query_exposures=4 * (end - start),
                  initial_model_sha256=initial, initial_projection_sha256=initial_projection,
                  mean_loss=float(np.mean(losses)), vjp_relative_l2=vjp,
                  real_reload_exact=True, canary_updates_discarded=is_canary,
                  token_exposures=token_work, candidate_exposures=candidates,
                  seconds=time.monotonic() - tick, locked_test_access=False,
                  quality_evaluated=False, native_cost_evaluated=False)
    return result


def worker(args):
    import torch
    from clearml import Task

    manifest = verify(args.research_root, args.run)
    data.require(args.phase in PHASES, 'unknown phase')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(80080)
    np.random.seed(80080)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0080/' + manifest['arm'] + '/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    campaign.io.write(output / 'clearml-start.json', receipt)
    task.connect({'input_sha256': campaign.io.sha(args.run / 'inputs.json'), 'arm': manifest['arm']})
    try:
        result = execute(args.research_root, args.run, output, args.phase, task)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'worker', 'supervise', 'verify'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase')
    parser.add_argument('--arm', choices=('F', 'D'))
    parser.add_argument('--gpu', type=int, choices=(0, 3))
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run, args.arm)
    elif args.mode == 'worker':
        worker(args)
    else:
        verify(args.research_root, args.run)
        if args.mode == 'supervise':
            data.require(args.gpu in (0, 3), 'explicit permitted GPU required')
            campaign.guard.supervise_graph(args, PHASES, 'ng80_fit.py')


if __name__ == '__main__':
    main()
