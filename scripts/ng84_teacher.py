"""NG84 TRAIN-only ranking-teacher admission; no optimizer or test access."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np


DOMAINS = ('fever', 'hotpotqa', 'nq')
PROTOCOL = 'ng0084-task-ranking-execution.zh.md'
MODEL_REVISION = '40e2b98fcbc6d60457c88508bd775fcb8395a5b0'
INPUTS = {
    'NG-0080/readout-v1/prepare/queries.json':
        'bd20960ea03f8f7121ba836dd226a405d32d78254c5e0551688dbe448049f0b4',
    'NG-0080/readout-v1/prepare/selection.json':
        'f445a4746edb117decbf39be921436589008f6b30a4fca916796576d3bf33372',
    'NG-0080/readout-v1/prepare/labels.json':
        'ec681fe479c4f551445bd26d5474d3c0404406df0c0ccfdf47f1579cca1fb0c1',
    'NG-0083/operator-v1/prepare/bank.npz':
        '296d885e72f3145478ed5c8d847d3c1c9613a6379cee23164d4cdb837d93fa32',
    'NG-0067/data/documents.jsonl':
        '16a008e102028b2cd04872dc24bfc60e0f2f9e508a563748020df81499a5b961',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def config():
    return dict(experiment='NG-0084', phase='teacher-admission-v1',
                queries_per_domain=128, seed=84001, bootstrap_samples=10000,
                batch_size=2, input_tokens=512, precision='float32',
                score_token='<extra_id_10>', model_revision=MODEL_REVISION,
                max_seconds=10800, rss_gib=32, vram_gib=20,
                host_available_gib=24, disk_free_gib=40,
                optimizer_updates=0, locked_test_access=False,
                independent_holdout=False, native_cost_evaluated=False,
                training_auto_launch=False)


def select(queries, split, per_domain=128):
    train, diagnostic = split['train'], split['diagnostic']
    require(len(set(train)) == len(train) and not set(train) & set(diagnostic),
            'invalid fit/exposed-diagnostic split')
    require(all(type(i) is int and 0 <= i < len(queries) for i in train),
            'invalid TRAIN index')
    require(len({r['query_id'] for r in queries}) == len(queries), 'duplicate query ID')
    chosen = []
    for domain in DOMAINS:
        ids = [i for i in train if queries[i]['subset'] == domain]
        require(len(ids) >= per_domain, 'insufficient TRAIN queries')
        ids.sort(key=lambda i: hashlib.sha256(
            ('NG84/teacher-admission/' + queries[i]['query_id']).encode()).hexdigest())
        chosen.append(ids[:per_domain])
    # Interleave domains so the timing canary does not measure one domain only.
    return [i for triple in zip(*chosen, strict=True) for i in triple]


def metrics(ids, scores, positives):
    require(len(ids) == len(scores) and len(ids) == len(set(ids)), 'score/ID alignment')
    require(all(type(i) is int and i >= 0 for i in ids), 'invalid document ID')
    require(positives and set(positives) <= set(ids), 'lost known positive')
    require(all(math.isfinite(float(v)) for v in scores), 'nonfinite score')
    order = sorted(zip(ids, scores, strict=True), key=lambda pair: (-pair[1], pair[0]))
    gold = set(positives)
    ranks = [r for r, (doc, _) in enumerate(order, 1) if doc in gold]
    dcg = sum(1 / math.log2(r + 1) for r in ranks if r <= 10)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(10, len(gold)) + 1))
    return dict(ndcg10=dcg / ideal, recall10=sum(r <= 10 for r in ranks) / len(gold),
                mrr=1 / min(ranks), positive_ranks=ranks)


def summarize(rows):
    require(len(rows) == 384 and {r['domain'] for r in rows} == set(DOMAINS),
            'unexpected admission rows/domains')
    require(len({r['query_id'] for r in rows}) == len(rows), 'duplicate result query')
    groups = {d: [r for r in rows if r['domain'] == d] for d in DOMAINS}
    require(all(len(v) == 128 for v in groups.values()), 'incomplete admission surface')
    rng = np.random.default_rng(84001)
    domain_results, resamples = {}, []
    for domain, records in groups.items():
        delta = np.array([r['rankt5']['ndcg10'] - r['pplx']['ndcg10'] for r in records])
        resamples.append(rng.choice(delta, (10000, len(delta))).mean(axis=1))
        domain_results[domain] = dict(queries=len(records),
            pplx_ndcg10=float(np.mean([r['pplx']['ndcg10'] for r in records])),
            rankt5_ndcg10=float(np.mean([r['rankt5']['ndcg10'] for r in records])),
            delta=float(delta.mean()), wins=int((delta > 0).sum()),
            ties=int((delta == 0).sum()), losses=int((delta < 0).sum()))
    delta = float(np.mean([r['delta'] for r in domain_results.values()]))
    lower, upper = np.quantile(np.mean(resamples, axis=0), [.025, .975]).tolist()
    decision = ('admit_conditional_training' if delta >= .005 and lower > 0 else
                'no_useful_signal_on_this_surface' if upper <= 0 else
                'inconclusive_do_not_auto_launch')
    return dict(domains=domain_results, macro_delta=delta, paired_interval=[lower, upper],
                decision=decision, independent_holdout=False,
                uncertainty='stratified query bootstrap; shared documents not independent',
                scope='fixed TRAIN candidate panels with known-positive qrels only',
                native_cost_evaluated=False, optimizer_updates=0)


def freeze(base, run, model):
    require(run.parent == base / 'NG-0084' and not run.exists(), 'fresh NG84 run required')
    for relative, digest in INPUTS.items():
        require(sha(base / relative) == digest, 'changed input: ' + relative)
    queries, split, labels = [read(base / 'NG-0080/readout-v1/prepare' / (n + '.json'))
                              for n in ('queries', 'selection', 'labels')]
    chosen = select(queries, split)
    with np.load(base / 'NG-0083/operator-v1/prepare/bank.npz') as bank:
        indices, offsets = bank['query_indices'], bank['offsets']
        docs, targets = bank['document_ids'], bank['targets']
        require(set(indices.tolist()) == set(split['train']), 'bank is not exact TRAIN')
        require(len(set(indices.tolist())) == len(indices), 'duplicate bank query')
        require(len(offsets) == len(indices) + 1 and offsets[0] == 0
                and offsets[-1] == len(docs) == len(targets)
                and np.all(np.diff(offsets) > 0), 'bank offsets')
        positions = {int(q): p for p, q in enumerate(indices)}
        selected = []
        for i in chosen:
            pos = positions[i]
            lo, hi = offsets[pos:pos + 2]
            ids, scores = docs[lo:hi].tolist(), targets[lo:hi].tolist()
            metrics(ids, scores, labels[i])
            selected.append(dict(queries[i], query_index=i, document_ids=ids,
                                 positives=labels[i], pplx_scores=scores))
    wanted = {i for row in selected for i in row['document_ids']}
    documents = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                row = json.loads(line)
                require(bool(row['text']), 'empty document')
                documents[str(i)] = row
    require(len(documents) == len(wanted), 'missing document')
    model_files = {p.name: sha(p) for p in sorted(model.iterdir()) if p.is_file()}
    expected = read(Path(__file__).with_name('ng84-rankt5-sha256.json'))
    require(model_files == expected, 'archived model transfer/hash mismatch')
    run.mkdir(parents=True)
    source = Path(__file__).resolve().parent
    paths = [Path(__file__), source / 'ng84-rankt5-sha256.json',
             source / 'requirements-ng84-teacher.txt',
             source.parent / 'tests/test_ng84_teacher.py',
             source.parent / 'docs/research-sae/reports/ng0001-ng0099' / PROTOCOL]
    for path in paths:
        shutil.copy2(path, run / path.name)
    write(run / 'queries.json', selected)
    write(run / 'documents.json', documents)
    write(run / 'config.json', config())
    write(run / 'inputs.json', dict(dependencies=INPUTS, model=str(model.resolve()),
        model_files=model_files, source={p.name: sha(p) for p in run.iterdir() if p.is_file()}))
    verify(run)
    print('NG84_FROZEN', sha(run / 'inputs.json'), len(selected), len(wanted), flush=True)


def verify(run):
    manifest = read(run / 'inputs.json')
    require(read(run / 'config.json') == config(), 'NG84 config changed')
    require(manifest['dependencies'] == INPUTS, 'NG84 dependency contract changed')
    for name, digest in manifest['source'].items():
        require(Path(name).name == name and sha(run / name) == digest, 'changed run file: ' + name)
    require(sha(Path(__file__)) == manifest['source']['ng84_teacher.py'], 'unfrozen source')
    model = Path(manifest['model'])
    require(manifest['model_files'] == read(run / 'ng84-rankt5-sha256.json'), 'model manifest')
    for name, digest in manifest['model_files'].items():
        require(Path(name).name == name and sha(model / name) == digest, 'changed model: ' + name)
    return model


def loading_receipt(report):
    fields = ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
    counts = {name: len(report.get(name, [])) for name in fields}
    require(not any(counts.values()), 'incomplete RankT5 checkpoint load')
    return dict(issue_counts=counts, reported_fields=sorted(report))


def worker(run):
    import google.protobuf
    import sentencepiece
    import torch
    import transformers
    from clearml import Task
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    model_path = verify(run)
    out = run / 'teacher'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(84001)
    require(torch.cuda.device_count() == 1, 'exactly one visible GPU required')
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0084/teacher-admission-v1',
        reuse_last_task_id=False, auto_connect_frameworks=False,
        auto_connect_arg_parser=False, auto_connect_streams=False,
        auto_resource_monitoring=False)
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False,
                   closed=False)
    write(out / 'clearml-start.json', receipt)
    try:
        task.connect(dict(config(), input_sha256=sha(run / 'inputs.json')))
        tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
        sp = sentencepiece.SentencePieceProcessor(model_file=str(model_path / 'spiece.model'))
        probes = ['Query: test Document: test', 'Query: evidence? Document: A study of cancer.']
        require(all(tokenizer(t)['input_ids'] == sp.encode(t) + [tokenizer.eos_token_id]
                    for t in probes), 'tokenizer differs from original SentencePiece')
        write(out / 'tokenizer-parity.json', dict(passed=True, probes=len(probes),
            sentencepiece=sentencepiece.__version__, protobuf=google.protobuf.__version__))
        token_id = tokenizer.convert_tokens_to_ids('<extra_id_10>')
        require(token_id != tokenizer.unk_token_id, 'missing RankT5 relevance token')
        model, loading = T5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32, output_loading_info=True)
        loading = loading_receipt(loading)
        model.to('cuda').eval()
        require(next(model.parameters()).dtype == torch.float32, 'wrong teacher precision')
        write(out / 'runtime.json', dict(torch=torch.__version__, transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(), score_token_id=token_id, loading=loading))
        queries, documents = read(run / 'queries.json'), read(run / 'documents.json')
        records, started, batches = [], time.monotonic(), 0
        with (out / 'scores.jsonl').open('x') as stream, torch.inference_mode():
            for i, row in enumerate(queries):
                scores, lengths = [], []
                for start in range(0, len(row['document_ids']), 2):
                    ids = row['document_ids'][start:start + 2]
                    texts = ['Query: ' + row['query'] + ' Document: ' + documents[str(d)]['text']
                             for d in ids]
                    lengths.extend(len(tokenizer(t, truncation=False)['input_ids']) for t in texts)
                    inputs = tokenizer(texts, padding=True, truncation=True,
                                       max_length=512, return_tensors='pt').to('cuda')
                    decoder = torch.full((len(ids), 1), model.config.decoder_start_token_id,
                                         dtype=torch.long, device='cuda')
                    logits = model(**inputs, decoder_input_ids=decoder).logits[:, 0, token_id]
                    require(bool(torch.isfinite(logits).all()), 'nonfinite RankT5 scores')
                    if batches == 0:
                        generated = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                            return_dict_in_generate=True, output_scores=True)
                        reference = generated.scores[0][:, token_id]
                        torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-5)
                        write(out / 'score-parity.json', dict(passed=True,
                            max_abs_error=float((logits - reference).abs().max())))
                    scores.extend(logits.cpu().double().tolist())
                    batches += 1
                result = dict(query_id=row['query_id'], domain=row['subset'],
                    document_ids=row['document_ids'], positives=row['positives'],
                    scores=scores, pplx_scores=row['pplx_scores'], prompt_tokens=lengths,
                    pplx=metrics(row['document_ids'], row['pplx_scores'], row['positives']),
                    rankt5=metrics(row['document_ids'], scores, row['positives']))
                records.append(result)
                stream.write(json.dumps(result, allow_nan=False) + '\n')
                stream.flush()
                elapsed = time.monotonic() - started
                write(out / 'progress.json', dict(completed_queries=i + 1, total_queries=len(queries),
                    completed_pairs=sum(len(r['scores']) for r in records), elapsed_seconds=elapsed,
                    estimated_total_seconds=elapsed * len(queries) / (i + 1)))
                if i == 15:
                    estimate = 1.5 * elapsed * len(queries) / 16 + 600
                    write(out / 'canary.json', dict(projected_seconds_with_margin=estimate))
                    require(estimate <= 10200, 'teacher pass cannot fit 3h bound; preserve canary')
                if (i + 1) % 16 == 0:
                    task.get_logger().report_scalar('teacher', 'queries', i + 1, i + 1)
                    print('NG84_TEACHER_PROGRESS', i + 1, len(queries), round(elapsed), flush=True)
        result = summarize(records)
        result.update(queries=len(records), pairs=sum(len(r['scores']) for r in records),
                      elapsed_seconds=time.monotonic() - started,
                      max_vram_bytes=torch.cuda.max_memory_allocated(),
                      scores_sha256=sha(out / 'scores.jsonl'))
        write(out / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException as exc:
        receipt['outcome'] = 'failed'
        task.mark_failed(status_reason=type(exc).__name__, status_message=str(exc), force=True)
        raise
    finally:
        task.close()
        receipt['closed'] = True
        write(out / 'clearml.json', receipt)


def gpu_state(gpu):
    text = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    return tuple(int(v.strip()) for v in text.strip().split(','))


def supervise(run, gpu):
    import fcntl
    import psutil

    verify(run)
    require(gpu in (0, 1, 2, 3), 'explicit physical GPU required')
    out = run / 'teacher'
    with (run.parent.parent / f'.ng-gpu-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(gpu_state(gpu)[0] < 100 and gpu_state(gpu)[1] == 0, 'GPU occupied; do not evict')
        require(psutil.virtual_memory().available > 24 * 1024 ** 3, 'host memory floor')
        require(shutil.disk_usage(run).free > 40 * 1024 ** 3, 'disk space floor')
        out.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', CLEARML_OFFLINE_MODE='1',
                   CLEARML_CACHE_DIR=str(out / 'tracking-cache'), PYTHONDONTWRITEBYTECODE='1')
        stopped, peak, error = [], 0, None
        handlers = {s: signal.signal(s, lambda v, frame: stopped.append(v))
                    for s in (signal.SIGTERM, signal.SIGINT)}
        began = time.monotonic()
        process = None
        closed = False
        try:
            with (out / 'stdout.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-B', str(run / 'ng84_teacher.py'),
                    'worker', '--run', str(run)], env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                write(out / 'started.json', dict(pid=process.pid, gpu=gpu, started_unix=time.time(),
                                                config=config()))
                while process.poll() is None:
                    require(not stopped, 'stop requested')
                    try:
                        root = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)])
                        peak = max(peak, rss)
                    except psutil.NoSuchProcess:
                        rss = 0
                    require(rss <= 32 * 1024 ** 3, 'owned RSS cap')
                    require(time.monotonic() - began <= 10800, 'three-hour wall cap')
                    require(psutil.virtual_memory().available > 24 * 1024 ** 3, 'host memory floor')
                    require(shutil.disk_usage(run).free > 40 * 1024 ** 3, 'disk space floor')
                    require(gpu_state(gpu)[0] <= 20 * 1024, 'selected GPU 20GiB cap')
                    time.sleep(2)
                require(process.returncode == 0, 'teacher worker failed')
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            if process is not None:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        closed = True
                        break
                    time.sleep(.1)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            receipt = read(out / 'clearml.json') if (out / 'clearml.json').exists() else {}
            passed = (process is not None and process.returncode == 0 and error is None
                      and closed and receipt.get('closed') and receipt.get('outcome') == 'passed')
            write(out / 'exit.json', dict(exit_code=process.returncode if process else None,
                error=error, owned_group_closed=closed, peak_tree_rss_bytes=peak,
                elapsed_seconds=time.monotonic() - began, finished_unix=time.time()))
            write(out / 'complete.json', dict(passed=bool(passed),
                files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
        require(passed, 'admission execution failed; preserve this attempt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker', 'review'))
    parser.add_argument('--research-root', type=Path)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.mode == 'freeze':
        require(args.research_root is not None and args.model is not None, 'freeze inputs required')
        freeze(args.research_root.resolve(), args.run.resolve(), args.model.resolve())
    elif args.mode == 'supervise':
        supervise(args.run, args.gpu)
    elif args.mode == 'worker':
        worker(args.run)
    elif args.mode == 'review':
        verify(args.run)
        proof = read(args.run / 'teacher/complete.json')
        require(proof['passed'], 'execution has not closed successfully')
        for name, digest in proof['files'].items():
            require(sha(args.run / 'teacher' / name) == digest, 'changed teacher artifact')
        rows = [json.loads(line) for line in (args.run / 'teacher/scores.jsonl').read_text().splitlines()]
        expected = {r['query_id']: r for r in read(args.run / 'queries.json')}
        require(set(expected) == {r['query_id'] for r in rows}, 'review query identity mismatch')
        for row in rows:
            original = expected[row['query_id']]
            require(row['domain'] == original['subset'] and all(
                row[k] == original[k] for k in ('document_ids', 'positives', 'pplx_scores')),
                'review input identity mismatch')
            for key, field in (('pplx', 'pplx_scores'), ('rankt5', 'scores')):
                require(row[key] == metrics(row['document_ids'], row[field], row['positives']),
                        'metric replay mismatch')
        print(json.dumps(summarize(rows), indent=2))
    else:
        verify(args.run)


if __name__ == '__main__':
    main()
