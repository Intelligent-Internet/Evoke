"""Bounded FP32 RankT5 batch-2/batch-8 equivalence and throughput probe."""

import argparse
import fcntl
from pathlib import Path
import shutil
import time

import numpy as np

from ng85_data import read, require, rows, sha, write
from ng85_queue import check_resources
from ng85_run import closed, verify
from ng85_targets import teacher_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--length-sorted', action='store_true')
    args = parser.parse_args()
    verify(args.base, args.run)
    for phase in ('prepare', 'bank'):
        closed(args.run, phase)
    args.out.mkdir()
    shutil.copy2(__file__, args.out / Path(__file__).name)
    shutil.copy2(Path(__file__).parent / 'ng85_targets.py', args.out / 'ng85_targets.py')
    write(args.out / 'inputs.json', dict(parent=args.run.name,
        inputs_sha256=sha(args.run / 'inputs.json'), batches=[2, 8], queries=32,
        gpu=args.gpu, maximum_seconds=900, dtype='float32', locked_test_access=False,
        length_sorted=args.length_sorted,
        targets_source_sha256=sha(args.out / 'ng85_targets.py'),
        source_sha256=sha(Path(__file__))))
    with (args.base / f'.ng-gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_resources(args.out, 'teacher_batch_probe', args.gpu)
        execute(args)


def execute(args):
    import psutil
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    require(torch.cuda.device_count() == 1, 'one bound GPU only')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = args.base / 'NG-0084/rankt5-hub'
    for name, digest in read(args.run / 'ng84-rankt5-sha256.json').items():
        require(sha(folder / name) == digest, 'RankT5 file changed')
    tokenizer = T5Tokenizer.from_pretrained(folder, local_files_only=True)
    model, info = T5ForConditionalGeneration.from_pretrained(folder,
        local_files_only=True, dtype=torch.float32, output_loading_info=True)
    require(not any(info.get(k) for k in ('missing_keys', 'unexpected_keys',
        'mismatched_keys', 'error_msgs')), 'model load mismatch')
    model.cuda().eval()
    token_id = tokenizer.convert_tokens_to_ids('<extra_id_10>')
    require(token_id == 32089, 'token contract changed')
    queries = rows(args.run / 'prepare/queries.jsonl')
    documents = rows(args.run / 'prepare/documents.jsonl')
    with np.load(args.run / 'bank/bank.npz') as bank:
        offsets, ids = bank['offsets'], bank['document_ids']
    shard = [i for i, q in enumerate(queries) if q['split'] == 'fit'][::4]
    pair_total = sum(int(offsets[i + 1] - offsets[i]) for i in shard)
    groups = [['Query: ' + queries[i]['query'] + ' Document: ' + documents[j]['text']
               for j in ids[offsets[i]:offsets[i + 1]]] for i in shard[:32]]
    metrics, outputs = {}, {}
    for batch in (2, 8):
        values = []
        started = time.monotonic()
        with torch.inference_mode():
            for number, texts in enumerate(groups):
                restored = np.full(len(texts), np.nan, dtype=np.float64)
                batches = teacher_batches(texts, tokenizer, batch,
                    length_sorted=(batch == 8 and args.length_sorted))
                for positions, batch_texts in batches:
                    require(psutil.virtual_memory().available > 20 * 1024 ** 3,
                            'host memory floor')
                    require(psutil.Process().memory_info().rss < 24 * 1024 ** 3,
                            'owned RSS cap')
                    tokens = tokenizer(batch_texts, truncation=False,
                        padding=True, return_tensors='pt').to('cuda')
                    require(tokens.input_ids.shape[1] <= 512, 'extra truncation')
                    decoder = torch.full((len(tokens.input_ids), 1),
                        model.config.decoder_start_token_id,
                        dtype=torch.long, device='cuda')
                    value = model(**tokens, decoder_input_ids=decoder).logits[:, 0, token_id]
                    require(torch.isfinite(value).all(), 'nonfinite scores')
                    require(torch.cuda.max_memory_allocated() < 22 * 1024 ** 3,
                            'GPU memory cap')
                    restored[positions] = value.cpu().double().numpy()
                require(np.isfinite(restored).all(), 'missing restored score')
                values.extend(restored.tolist())
                write(args.out / 'progress.json', dict(batch=batch, queries=number + 1,
                    pairs=len(values), seconds=time.monotonic() - started))
                print('NG85_BATCH_PROBE', batch, number + 1, len(values), flush=True)
        seconds = time.monotonic() - started
        outputs[batch] = np.asarray(values)
        np.save(args.out / f'batch-{batch}.npy', outputs[batch])
        metrics[batch] = dict(seconds=seconds, pairs=len(values),
            projected_shard_seconds=seconds * pair_total / len(values),
            guarded_seconds=1.5 * seconds * pair_total / len(values) + 600)
    equivalent = bool(np.allclose(outputs[2], outputs[8], rtol=1e-5, atol=1e-5))
    write(args.out / 'results.json', dict(passed=equivalent and
        metrics[8]['guarded_seconds'] < 8 * 3600, equivalent=equivalent,
        max_abs_error=float(np.max(np.abs(outputs[2] - outputs[8]))),
        metrics=metrics, max_vram_bytes=torch.cuda.max_memory_allocated(),
        query_indices=shard[:32], locked_test_access=False))


if __name__ == '__main__':
    main()
