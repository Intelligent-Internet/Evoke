#!/usr/bin/env python3
"""Prepare TRAIN-only PPLX targets for a matched-exposure breadth study."""

import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


DOMAINS = ('fever', 'hotpotqa', 'nq')
SECONDS = 5400
HOST_BYTES = 16 * 1024 ** 3
GPU_BYTES = 20 * 1024 ** 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def select_indices(train):
    require(len(train) == 13824, 'NG67 TRAIN size changed')
    require(len({r['query_id'] for r in train}) == len(train), 'Duplicate query')
    require(Counter(r['subset'] for r in train[:1536])
            == dict.fromkeys(DOMAINS, 512), 'Old TRAIN population changed')
    chosen = list(range(1536))
    for domain in DOMAINS:
        candidates = [i for i in range(1536, len(train))
                      if train[i]['subset'] == domain]
        require(len(candidates) == 4096, 'New TRAIN domain count changed')
        candidates.sort(key=lambda i: hashlib.sha256(
            ('NG69-breadth-v1:' + train[i]['query_id']).encode()).hexdigest())
        chosen += candidates[:1536]
    require(len(set(chosen)) == 6144, 'Invalid breadth sample')
    return chosen


def input_files(base):
    reference = base / 'NG-0059'
    old = read(reference / 'dependencies.json')['files']
    result = {os.path.relpath(reference / key, base): value
              for key, value in old.items() if '/dense-model/' in key}
    teacher = read(reference / 'recovery/teacher/complete.json')
    for name in ('documents.npy', 'train-query.npy', 'targets.json'):
        key = 'recovery/teacher/' + name
        result['NG-0059/' + key] = teacher['files'][key]
    prepared = read(base / 'NG-0067/prepared.json')
    for name in ('documents.jsonl', 'train.jsonl', 'train-pools.json',
                 'train-labels.json'):
        key = 'data/' + name
        result['NG-0067/' + key] = prepared['files'][key]
    for name in ('NG-0059/dependencies.json',
                 'NG-0059/recovery/teacher/complete.json',
                 'NG-0059/data/documents.jsonl',
                 'NG-0059/data/train.jsonl', 'NG-0067/prepared.json'):
        result[name] = sha(base / name)
    return result


def verify(args):
    record = read(args.run / 'inputs.json')
    require(record['source_sha256'] == sha(Path(__file__)), 'Source changed')
    require(record['protocol_sha256'] == sha(args.run / 'protocol.md'),
            'Protocol changed')
    for name, expected in record['files'].items():
        require(sha(args.research_root / name) == expected,
                'Input changed: ' + name)
    return record


def freeze(args):
    require(not (args.run / 'inputs.json').exists(), 'Already frozen')
    files = input_files(args.research_root)
    for name, expected in files.items():
        require(sha(args.research_root / name) == expected, name)
    write(args.run / 'inputs.json', {
        'files': files, 'source_sha256': sha(Path(__file__)),
        'protocol_sha256': sha(args.run / 'protocol.md'),
        'train_only': True, 'locked_test_query_access': False})


def parameter_hash(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def worker(args):
    import numpy as np
    import torch
    from clearml import Task

    require(sys.version_info[:2] == (3, 12) and not sys.flags.optimize,
            'Python contract')
    require(np.__version__ == '2.2.6'
            and torch.__version__.startswith('2.9.1'), 'Runtime contract')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(69069)
    verify(args)
    require(not (args.run / 'clearml.json').exists(), 'Worker already started')
    Task.set_offline(True)
    task = Task.init(
        project_name='Evoke-NG', task_name='NG-0069/breadth-teacher-preparation',
        task_type=Task.TaskTypes.data_processing, reuse_last_task_id=False,
        auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    tracking = {'task_id': task.id, 'actual_start': True, 'closed': False,
                'offline': True, 'remote_synced': False, 'post_hoc': False,
                'training': False, 'optimizer_steps': 0}
    write(args.run / 'clearml.json', tracking)
    task.connect({'inputs_sha256': sha(args.run / 'inputs.json')})
    try:
        prepare(args, task)
    finally:
        task.close()
        tracking['closed'] = True
        write(args.run / 'clearml.json', tracking)
    verify(args)
    names = ('selection.json', 'document-ids.npy', 'documents.npy',
             'train-query.npy', 'targets.json', 'results.json', 'clearml.json',
             'canary.json')
    write(args.run / 'complete.json', {
        'passed': True, 'files': {n: sha(args.run / n) for n in names},
        'training_started': False, 'locked_test_scored': False})


def prepare(args, task):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    started = time.monotonic()
    data = args.research_root / 'NG-0067/data'
    teacher = args.research_root / 'NG-0059/recovery/teacher'
    dense = args.research_root / 'NG-0014/dense-model'
    train, docs = rows(data / 'train.jsonl'), rows(data / 'documents.jsonl')
    require(train[:1536] == rows(args.research_root / 'NG-0059/data/train.jsonl'),
            'Old TRAIN records changed')
    require(docs[:59111] == rows(args.research_root / 'NG-0059/data/documents.jsonl'),
            'Old document indices changed')
    selected = select_indices(train)
    pools, labels = read(data / 'train-pools.json'), read(data / 'train-labels.json')
    chosen = set(selected)
    needed = sorted({d for i in selected for d in pools[i] if d >= 59111})
    write(args.run / 'selection.json', {
        'indices': selected, 'query_ids': [train[i]['query_id'] for i in selected],
        'old_queries': 1536, 'new_queries': 4608,
        'new_documents_to_encode': len(needed), 'score_based_selection': False})
    tokenizer = AutoTokenizer.from_pretrained(
        dense, trust_remote_code=True, local_files_only=True)
    model, info = AutoModel.from_pretrained(
        dense, trust_remote_code=True, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa', output_loading_info=True)
    require(all(not info.get(k) for k in (
        'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')),
        'Model loading mismatch')
    require(sha(Path(inspect.getfile(type(model)))) == sha(dense / 'modeling.py'),
            'Unexpected executable model implementation')
    model.eval().cuda()
    initial_hash = parameter_hash(model)
    require(initial_hash == read(teacher / 'complete.json')['model_state_sha256'],
            'Teacher parameter identity changed')

    @torch.no_grad()
    def encode(texts, role):
        tokens = tokenizer(texts, padding=True, truncation=True,
                           max_length=64 if role == 'query' else 256,
                           return_tensors='pt').to('cuda')
        hidden = model(**tokens, use_cache=False).last_hidden_state
        mask = tokens['attention_mask'].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        require(torch.isfinite(pooled).all() and (pooled.norm(dim=1) > 0).all(),
                'Invalid teacher encoding')
        require(torch.cuda.max_memory_allocated() < GPU_BYTES, 'GPU budget')
        return torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()

    old_docs = np.load(teacher / 'documents.npy')
    old_queries = np.load(teacher / 'train-query.npy')
    probe = np.concatenate([encode([d['text'] for d in docs[i:i + 4]], 'document')
                            for i in range(0, 64, 4)])
    np.testing.assert_allclose(probe, old_docs[:64], rtol=1e-4, atol=2e-5)
    for i in range(4):
        np.testing.assert_allclose(encode([train[i]['query']], 'query')[0],
                                   old_queries[i], rtol=1e-4, atol=2e-5)
    require(len(needed) >= 64, 'Unexpectedly small new document set')
    sample = [needed[i] for i in np.linspace(0, len(needed) - 1, 64, dtype=int)]
    tick = time.monotonic()
    for start in range(0, 64, 4):
        encode([docs[i]['text'] for i in sample[start:start + 4]], 'document')
    document_seconds = time.monotonic() - tick
    tick = time.monotonic()
    for i in selected[1536:1552]:
        encode([train[i]['query']], 'query')
    query_seconds = time.monotonic() - tick
    prediction = (document_seconds * len(needed) / 64
                  + query_seconds * 4608 / 16) * 2 + 300
    write(args.run / 'canary.json', {
        'passed': prediction < SECONDS - 120,
        'predicted_seconds': prediction,
        'old_document_max_error': float(abs(probe - old_docs[:64]).max()),
        'rtol': 1e-4, 'atol': 2e-5})
    require(prediction < SECONDS - 120, 'Canary predicts resource overrun')
    outputs = [old_docs]
    for start in range(0, len(needed), 4):
        ids = needed[start:start + 4]
        outputs.append(encode([docs[d]['text'] for d in ids], 'document'))
        if start % 1024 == 0:
            write(args.run / 'progress.json', {'phase': 'new_train_documents',
                  'completed': start + len(ids), 'total': len(needed)})
            print('documents', start + len(ids), '/', len(needed), flush=True)
    embeddings = np.concatenate(outputs)
    document_ids = list(range(59111)) + needed
    lookup = {doc: i for i, doc in enumerate(document_ids)}
    queries = [old_queries]
    for count, i in enumerate(selected[1536:], 1):
        queries.append(encode([train[i]['query']], 'query'))
        if count % 128 == 0:
            write(args.run / 'progress.json', {'phase': 'new_train_queries',
                  'completed': count, 'total': 4608})
            task.get_logger().report_scalar('prepare', 'queries', count, count)
            print('queries', count, '/4608', flush=True)
    query_vectors = np.concatenate(queries)
    require(query_vectors.shape == (6144, 1024), 'Query shape')
    require(embeddings.shape == (len(document_ids), 1024), 'Document shape')
    targets = read(teacher / 'targets.json')
    require(len(targets) == 1536, 'Old target count changed')
    for i, target in enumerate(targets):
        require(target['query_id'] == train[i]['query_id']
                and target['pool'] == pools[i]
                and [d for d, p in zip(target['pool'], target['positive_mask'])
                     if p] == labels[i], 'Old target identity changed')
    for j, i in enumerate(selected[1536:], 1536):
        pool = pools[i]
        scores = query_vectors[j].astype(np.float64) @ embeddings[
            [lookup[d] for d in pool]].astype(np.float64).T
        positive = np.isin(pool, labels[i])
        require(positive.sum() == len(labels[i]) > 0, 'Lost positive')
        probs = np.exp((scores - scores.max()) / .04)
        probs /= probs.sum()
        target = .5 * positive / positive.sum() + .5 * probs
        targets.append({'query_id': train[i]['query_id'], 'pool': pool,
                        'positive_mask': positive.tolist(),
                        'scores': scores.tolist(), 'target': target.tolist()})
    require(parameter_hash(model) == initial_hash, 'Frozen teacher mutated')
    np.save(args.run / 'document-ids.npy', np.asarray(document_ids, dtype=np.int64))
    np.save(args.run / 'documents.npy', embeddings)
    np.save(args.run / 'train-query.npy', query_vectors)
    write(args.run / 'targets.json', targets)
    write(args.run / 'results.json', {
        'train_queries': len(chosen), 'new_queries_encoded': 4608,
        'new_documents_encoded': len(needed), 'reused_documents': 59111,
        'reused_train_queries': 1536, 'held_out_queries_encoded': 0,
        'model_state_sha256': initial_hash, 'seconds': time.monotonic() - started,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'training_started': False, 'overall_goal_qualified': False})


def supervise(args):
    import fcntl
    import psutil

    def interrupt(signum, frame):
        raise InterruptedError(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    require(not (args.run / 'started.json').exists(), 'Inspect, do not retry')
    with (args.research_root / '.ng-gpu-0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(args)
        info = subprocess.check_output([
            'nvidia-smi', '--id=0', '--query-gpu=uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip().split(',')
        require(int(info[1]) < 100 and int(info[2]) == 0, 'GPU0 is occupied')
        require(psutil.virtual_memory().available > 24 * 1024 ** 3,
                'Insufficient free host memory')
        require(psutil.disk_usage(args.run).free > 20 * 1024 ** 3, 'Disk space')
        environment = dict(os.environ)
        for key in ('CLEARML_TASK_ID', 'TRAINS_TASK_ID', 'CLEARML_PROC_MASTER_ID',
                    'TRAINS_PROC_MASTER_ID'):
            environment.pop(key, None)
        environment.update(
            CUDA_VISIBLE_DEVICES='0', OPENBLAS_NUM_THREADS='4',
            OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
            CLEARML_OFFLINE_MODE='1',
            CLEARML_CACHE_DIR=str(args.run / 'tracking-cache'))
        write(args.run / 'started.json', {
            'started_unix': time.time(), 'controller_pid': os.getpid(),
            'gpu_uuid': info[0].strip(), 'phase_seconds': SECONDS})
        process, error, code, peak = None, None, 1, 0
        started = time.monotonic()
        try:
            with (args.run / 'stdout.log').open('x') as log:
                process = subprocess.Popen([
                    sys.executable, '-B', str(Path(__file__).resolve()),
                    '--research-root', str(args.research_root),
                    '--run', str(args.run), '--mode', 'worker'],
                    env=environment, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                while process.poll() is None:
                    try:
                        members = [psutil.Process(process.pid)]
                        members += members[0].children(recursive=True)
                        rss = sum(p.memory_info().rss for p in members
                                  if p.is_running())
                        peak = max(peak, rss)
                        require(rss <= HOST_BYTES, 'Host RSS limit')
                    except psutil.NoSuchProcess:
                        pass
                    require(time.monotonic() - started <= SECONDS, 'Wall limit')
                    time.sleep(1)
                code = process.wait()
                require(code == 0, 'Worker failed')
                complete = read(args.run / 'complete.json')
                require(complete['passed'], 'Missing completion receipt')
                for name, expected in complete['files'].items():
                    require(sha(args.run / name) == expected, 'Output changed')
        except BaseException as exc:
            error = repr(exc)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    time.sleep(2)
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            living = []
            if process is not None:
                for member in psutil.process_iter(['pid', 'status']):
                    try:
                        if (member.info['status'] != psutil.STATUS_ZOMBIE
                                and os.getpgid(member.pid) == process.pid):
                            living.append(member.pid)
                    except (ProcessLookupError, psutil.NoSuchProcess):
                        pass
            if living:
                error = (error or '') + f' Owned group remains: {living}'
            write(args.run / 'exit.json', {
                'exit_code': 1 if error else code, 'error': error,
                'seconds': time.monotonic() - started,
                'peak_tree_rss_bytes': peak, 'training_started': False,
                'owned_group_closed': not living})
        require(error is None and code == 0, str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--mode', choices=('freeze', 'supervise', 'worker'),
                        required=True)
    args = parser.parse_args()
    args.research_root = args.research_root.resolve()
    args.run = args.run.resolve()
    require(args.run.is_dir(), 'Create the run and protocol first')
    {'freeze': freeze, 'supervise': supervise, 'worker': worker}[args.mode](args)


if __name__ == '__main__':
    main()
