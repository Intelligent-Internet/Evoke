#!/usr/bin/env python3
"""Frozen NG80 preparation and readout diagnostic; no automatic promotion.

Each GPU lane uses the established single-worker process-group supervisor.
Only the declared NG67 TRAIN file is permitted, never DEV/LOCKED queries.
"""

import argparse
from collections import Counter
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time

import ng71_pilot as guard
import ng71_preflight as io


DATA = 'NG-0067/data/'
DENSE = 'NG-0069/evaluation-v2/encode-dense/document.npy'
PHASES = ('prepare', 'encode-teacher', 'encode-hidden', 'diagnose')
LANES = {'prepare': ['prepare'], 'teacher': ['encode-teacher'],
         'hidden': ['encode-hidden'], 'review': ['diagnose']}
ANCHORS = {
    DATA + 'train.jsonl': '01fbe89cbb8c9ba30330033fb4711dcaa780547c0fa394baba045d74d06ea775',
    DATA + 'documents.jsonl': '16a008e102028b2cd04872dc24bfc60e0f2f9e508a563748020df81499a5b961',
    DATA + 'train-pools.json': '807e4deb96f701eeae2d998947a11ca7e944b8253f0471655596e36f0d0c89cc',
    DATA + 'train-labels.json': 'ec681fe479c4f551445bd26d5474d3c0404406df0c0ccfdf47f1579cca1fb0c1',
    DENSE: '7d612369dcf5246a6c1747a2c49f399486a091ba62a8e0b40725b7bff880b16a',
    'NG-0059/dependencies.json': '3421139b6848847c090c4fd3786f01da7209d89ae2c5a2225d567655d51c410c',
}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def verify(base, run):
    manifest = read(run / 'inputs.json')
    require(manifest['protocol'] == 'NG80_readout_information_v1'
            and manifest['phases'] == list(PHASES)
            and manifest['locked_test_access'] is False
            and manifest['optimizer_training_enabled'] is False, 'protocol changed')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in manifest[key].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve()), 'dependency escape')
            require(io.sha(path) == digest, 'changed input: ' + name)
    return manifest


def freeze(base, run):
    require(run.parent == base / 'NG-0080', 'fresh NG80 unit required')
    dependencies = dict(ANCHORS)
    for name, digest in ANCHORS.items():
        require(io.sha(base / name) == digest, 'unverified anchor: ' + name)
    old = read(base / 'NG-0059/dependencies.json')['files']
    prefixes = ('NG-0002/model/', 'NG-0003/', 'NG-0014/dense-model/',
                'NG-0002/checkpoint_io.py', 'NG-0031/encoded/mlm-raw-rms.npy')
    for relative, digest in old.items():
        name = str((base / 'NG-0059' / relative).resolve().relative_to(base.resolve()))
        if name.startswith(prefixes):
            require(io.sha(base / name) == digest, 'model dependency changed')
            dependencies[name] = digest
    old_teacher = 'NG-0059/recovery/teacher/'
    proof = read(base / old_teacher / 'complete.json')
    for name in ('train-query.npy', 'documents.npy'):
        key = old_teacher + name
        digest = proof['files']['recovery/teacher/' + name]
        require(io.sha(base / key) == digest, 'teacher cache changed')
        dependencies[key] = digest
    dependencies[old_teacher + 'complete.json'] = io.sha(base / old_teacher / 'complete.json')
    source = Path(__file__).resolve().parent
    names = ('ng80_campaign.py', 'ng80_data.py', 'ng71_pilot.py',
             'ng71_preflight.py', 'ng71_training.py', 'ng71_ranking.py')
    sources = [source / name for name in names]
    sources += [source.parent / 'tests/test_ng80_campaign.py']
    docs = source.parent / 'docs/research-sae/reports/ng0001-ng0099'
    sources += [docs / 'ng0080-bottleneck-campaign.zh.md', docs / 'ng0071-ranking-config.json']
    run.mkdir(parents=True, exist_ok=False)
    for path in sources:
        shutil.copy2(path, run / path.name)
    io.write(run / 'inputs.json', {
        'protocol': 'NG80_readout_information_v1', 'phases': list(PHASES),
        'dependencies': dependencies,
        'source': {p.name: io.sha(p) for p in run.iterdir() if p.is_file()},
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=source.parent, text=True).strip(),
        'source_snapshot_sha_is_authoritative': True,
        'locked_test_access': False, 'optimizer_training_enabled': False,
        'readout_fit_enabled': True, 'automatic_expansion': False,
        'created_unix': time.time()})
    verify(base, run)
    print('NG80_FROZEN', io.sha(run / 'inputs.json'), flush=True)


def closed(run, name):
    folder = run / name
    result, exited = read(folder / 'complete.json'), read(folder / 'exit.json')
    tracking = read(folder / 'clearml.json')
    require(result['passed'] and exited['exit_code'] == 0 and exited['error'] is None
            and exited['owned_group_closed'] and tracking['actual_start']
            and tracking['closed'] and tracking['outcome'] == 'passed',
            'prerequisite not closed: ' + name)
    for path, digest in result['files'].items():
        require(io.sha(folder / path) == digest, 'phase payload changed')


def prepare(base, run, output):
    import ng80_data as study

    train = rows(base / DATA / 'train.jsonl')
    pools, labels = [read(base / DATA / name) for name in ('train-pools.json', 'train-labels.json')]
    require(len(train) == 13824 and Counter(r['subset'] for r in train)
            == dict.fromkeys(study.DOMAINS, 4608), 'TRAIN population changed')
    selected = study.selection(train)
    candidates, background = study.panels(pools, labels, 233009)
    io.write(output / 'selection.json', selected)
    io.write(output / 'queries.json', [
        {k: r[k] for k in ('query_id', 'query', 'subset')} for r in train])
    io.write(output / 'panels.json', candidates)
    io.write(output / 'labels.json', labels)
    train_docs = sorted({d for i in selected['train'] for d in pools[i]})
    io.write(output / 'training-document-ids.json', train_docs)
    return {'queries': len(train), 'train': len(selected['train']),
            'diagnostic': len(selected['diagnostic']), 'documents': 233009,
            'training_source_documents': len(train_docs), 'shared_background_ids': background,
            'domain_counts': dict(Counter(r['subset'] for r in train)),
            'panel_max': max(map(len, candidates)), 'source_positives_preserved': True,
            'diagnostic_is_independent_holdout': False, 'locked_test_access': False}


def canary(output, role, seconds, completed, total):
    predicted = 1.5 * seconds * total / completed + 300
    io.write(output / (role + '-canary.json'), {
        'completed': completed, 'total': total, 'seconds': seconds,
        'predicted_seconds': predicted, 'passed': predicted < 5100})
    require(predicted < 5100, 'canary exceeds phase budget; preserve and chunk')


def encode_teacher(base, run, output, task):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from ng71_training import parameter_hash

    folder = base / 'NG-0014/dense-model'
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
    model, info = AutoModel.from_pretrained(
        folder, trust_remote_code=True, local_files_only=True, dtype=torch.float32,
        attn_implementation='sdpa', output_loading_info=True)
    require(not any(info.get(k) for k in (
        'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), 'teacher load')
    require(io.sha(Path(inspect.getfile(type(model)))) == io.sha(folder / 'modeling.py'),
            'teacher executable source changed')
    model.eval().cuda()
    before = parameter_hash(model)
    require(before == read(base / 'NG-0059/recovery/teacher/complete.json')['model_state_sha256'],
            'teacher parameter state changed')
    queries = read(run / 'prepare/queries.json')
    result, lengths = [], []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(queries), 16):
            texts = [q['query'] for q in queries[start:start + 16]]
            lengths += [len(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=64,
                               return_tensors='pt').to('cuda')
            hidden = model(**tokens, use_cache=False).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            require(torch.isfinite(pooled).all() and (pooled.norm(dim=1) > 0).all(),
                    'invalid teacher codes')
            result.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
            if start == 48:
                canary(output, 'query', time.monotonic() - started, 64, len(queries))
            if start % 512 == 0:
                print('teacher-query', start + len(texts), '/', len(queries), flush=True)
                task.get_logger().report_scalar('encoding', 'queries', start + len(texts), start)
    values = np.concatenate(result)
    old = np.load(base / 'NG-0059/recovery/teacher/train-query.npy', mmap_mode='r')
    np.testing.assert_allclose(values[:1536], old, rtol=1e-4, atol=2e-5)
    require(parameter_hash(model) == before, 'teacher changed during inference')
    np.save(output / 'query.npy', values)
    np.save(output / 'query-token-lengths.npy', np.asarray(lengths, dtype=np.int32))
    return {'queries': len(values), 'dimensions': values.shape[1],
            'old_query_max_error': float(np.abs(values[:1536] - old).max()),
            'model_state_sha256': before, 'document_sha256': ANCHORS[DENSE],
            'query_truncated': int(np.sum(np.asarray(lengths) > 64)),
            'seconds': time.monotonic() - started, 'locked_test_access': False}


def encode_hidden(base, run, output, task):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from ng71_training import parameter_hash

    config = read(run / 'ng0071-ranking-config.json')['base']
    spec = importlib.util.spec_from_file_location('ng80_checkpoint_io', base / 'NG-0002/checkpoint_io.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load_model(base / config['checkpoint']).cuda()
    before = parameter_hash(model)
    require(before == config['state_sha256'], 'parent checkpoint changed')
    tokenizer = AutoTokenizer.from_pretrained(base / config['tokenizer'], local_files_only=True)
    queries = read(run / 'prepare/queries.json')
    documents = rows(base / DATA / 'documents.jsonl')
    require(len(documents) == 233009, 'corpus changed')
    started, counters = time.monotonic(), {}
    with torch.no_grad():
        for role, texts, limit in (
            ('query', [q['query'] for q in queries], 64),
            ('document', [d['text'] for d in documents], 256)):
            shape = (len(texts), model.config.hidden_size)
            arrays = {key: np.lib.format.open_memmap(output / f'{role}-{key}.npy',
                      mode='w+', dtype=np.float32, shape=shape) for key in ('mean', 'cls')}
            lengths = np.zeros(len(texts), dtype=np.int32)
            tick = time.monotonic()
            for start in range(0, len(texts), 4):
                chunk = texts[start:start + 4]
                lengths[start:start + len(chunk)] = [len(ids) for ids in tokenizer(
                    chunk, truncation=False, verbose=False)['input_ids']]
                tokens = tokenizer(chunk, padding=True, truncation=True, max_length=limit,
                                   return_tensors='pt').to('cuda')
                hidden = model.base_model(**tokens).last_hidden_state
                require(torch.isfinite(hidden).all(), 'nonfinite parent hidden')
                mask = tokens['attention_mask'].unsqueeze(-1)
                mean = (hidden * mask).sum(1) / mask.sum(1)
                arrays['mean'][start:start + len(chunk)] = mean.cpu().numpy()
                arrays['cls'][start:start + len(chunk)] = hidden[:, 0].cpu().numpy()
                if start == 60:
                    canary(output, role, time.monotonic() - tick, 64, len(texts))
                if start % 4096 == 0:
                    print('hidden-' + role, start + len(chunk), '/', len(texts), flush=True)
                    task.get_logger().report_scalar('encoding', role, start + len(chunk), start)
            for array in arrays.values():
                array.flush()
            np.save(output / f'{role}-token-lengths.npy', lengths)
            counters[role] = {'rows': len(texts), 'truncated': int((lengths > limit).sum()),
                              'token_limit': limit, 'seconds': time.monotonic() - tick}
    require(parameter_hash(model) == before, 'parent parameters changed')
    return {'model_state_sha256': before, 'roles': counters,
            'seconds': time.monotonic() - started, 'optimizer_updates': 0,
            'locked_test_access': False, 'pooling': ['masked_mean', 'cls']}


def diagnose(base, run, output):
    import numpy as np
    import ng80_data as study

    selection = read(run / 'prepare/selection.json')
    panels = read(run / 'prepare/panels.json')
    labels = read(run / 'prepare/labels.json')
    queries = read(run / 'prepare/queries.json')
    td = read(run / 'prepare/training-document-ids.json')
    yq = np.load(run / 'encode-teacher/query.npy', mmap_mode='r')
    yd = np.load(base / DENSE, mmap_mode='r')
    require(yq.shape == (13824, 1024) and yd.shape == (233009, 1024), 'teacher dimensions')
    summaries = {}
    for pooling in ('mean', 'cls'):
        hq, hd = [np.load(run / f'encode-hidden/{role}-{pooling}.npy', mmap_mode='r')
                  for role in ('query', 'document')]
        weights = study.ridge(hq[selection['train']], yq[selection['train']], hd[td], yd[td])
        np.save(output / f'{pooling}-projection.npy', weights)
        sq = study.project(hq, weights)
        # Project documents on demand to keep the full-background array out of RAM.
        reduced = {}
        with (output / f'{pooling}-panels.jsonl').open('x') as stream:
            for surface in ('train', 'diagnostic'):
                values = []
                for i in selection[surface]:
                    pool, gold = panels[i], labels[i]
                    sd = study.project(hd[pool], weights)
                    teacher = yd[pool].astype(np.float64) @ yq[i].astype(np.float64)
                    scores = sd @ sq[i]
                    row = study.metrics(scores, teacher, pool, gold)
                    baseline = study.metrics(teacher, teacher, pool, gold)
                    row.update(query_id=queries[i]['query_id'], domain=queries[i]['subset'],
                               surface=surface, teacher_panel_ndcg10=baseline['panel_ndcg10'])
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    values.append(row)
                for domain in ('macro', *study.DOMAINS):
                    subset = [r for r in values if domain == 'macro' or r['domain'] == domain]
                    reduced[surface + '/' + domain] = {k: float(np.mean([r[k] for r in subset]))
                        for k in ('panel_ndcg10', 'teacher_panel_ndcg10',
                                  'teacher_top10_overlap', 'centered_score_mse')}
                    reduced[surface + '/' + domain]['queries'] = len(subset)
        summaries[pooling] = reduced
    return {'readout': summaries, 'ridge': 1e-3, 'role_weights': [.5, .5],
            'fit_query_count': len(selection['train']), 'fit_document_count': len(td),
            'diagnostic_is_independent_holdout': False, 'full_corpus_ranking': False,
            'native_cost_evaluated': False, 'optimizer_training_completed': False,
            'remaining': ['full_trunk_vs_frozen_training', 'sparse_field_vs_qrel_training',
                          'multi_source_task_diversity', 'teacher_context_control',
                          'human_label_review', 'full_background_and_native_cost']}


def worker(args):
    import numpy as np
    import torch
    from clearml import Task

    verify(args.research_root, args.run)
    if args.phase != 'prepare':
        closed(args.run, 'prepare')
    if args.phase == 'diagnose':
        for phase in ('encode-teacher', 'encode-hidden'):
            closed(args.run, phase)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(80080)
    np.random.seed(80080)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0080/' + args.phase,
        reuse_last_task_id=False, auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    output = args.run / args.phase
    receipt = {'task_id': task.id, 'actual_start': True, 'offline': True,
               'remote_synced': False, 'closed': False}
    io.write(output / 'clearml-start.json', receipt)
    task.connect({'input_sha256': io.sha(args.run / 'inputs.json'), 'phase': args.phase})
    try:
        if args.phase == 'prepare':
            result = prepare(args.research_root, args.run, output)
        elif args.phase == 'encode-teacher':
            result = encode_teacher(args.research_root, args.run, output, task)
        elif args.phase == 'encode-hidden':
            result = encode_hidden(args.research_root, args.run, output, task)
        else:
            result = diagnose(args.research_root, args.run, output)
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
    import fcntl
    import signal

    verify(args.research_root, args.run)
    require(args.lane in LANES, 'unknown lane')
    cuda = args.lane in ('teacher', 'hidden')
    require(not cuda or args.gpu in (0, 3), 'only GPUs 0/3 authorized for this wave')
    require(not (args.run / f'{args.lane}-controller-start.json').exists(), 'do not restart lane')
    stopped = []
    handlers = {s: signal.signal(s, lambda *_: stopped.append(True))
                for s in (signal.SIGTERM, signal.SIGINT)}
    lock_path = args.research_root / (f'.ng-gpu-{args.gpu}.lock' if cuda else '.ng80-cpu.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cuda:
            require(guard.gpu_state(args.gpu) == (1, 0)
                    or (guard.gpu_state(args.gpu)[0] < 100 and guard.gpu_state(args.gpu)[1] == 0),
                    'GPU occupied; do not evict')
        io.write(args.run / f'{args.lane}-controller-start.json',
                 {'pid': __import__('os').getpid(), 'gpu': args.gpu if cuda else None,
                  'started_unix': time.time()})
        args.worker_script = 'ng80_campaign.py'
        try:
            for phase in LANES[args.lane]:
                guard.phase(args, phase, stopped, cuda=cuda, limit_seconds=5400)
            io.write(args.run / f'{args.lane}-controller-exit.json',
                     {'status': 'all_phases_complete', 'finished_unix': time.time()})
        except BaseException as exc:
            io.write(args.run / f'{args.lane}-controller-exit.json',
                     {'status': 'failed_preserve_attempt', 'error': str(exc),
                      'finished_unix': time.time()})
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--lane', choices=LANES)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.research_root, args.run)
    elif args.mode == 'verify':
        verify(args.research_root, args.run)
    elif args.mode == 'supervise':
        supervise(args)
    else:
        require(args.phase in PHASES, 'worker phase required')
        worker(args)


if __name__ == '__main__':
    main()
