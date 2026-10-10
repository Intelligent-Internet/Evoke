"""Bounded scalar-only CPU/CUDA probe of three closed NG80 P records.

This diagnoses numerical execution. It is not a replacement audit, model
inference, an optimizer update, or permission to alter a frozen gate.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


CASES = {(7, 11716), (2051, 13664), (2723, 12571)}


def main(base, gpu, output):
    assert gpu in (0, 3) and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
    assert not output.exists()
    with (base / f'.ng-gpu-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = subprocess.check_output([
            'nvidia-smi', f'--id={gpu}',
            '--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        memory, use = map(int, raw.strip().split(','))
        assert memory < 100 and use == 0, raw
        import numpy as np
        import torch
        from torch.nn import functional as functional

        torch.set_num_threads(4)
        started = time.monotonic()
        results, hashes = [], {}
        for path in sorted((base / 'NG-0080/sparse-P-v1').glob('train-*/updates.jsonl')):
            complete = json.loads((path.parent / 'complete.json').read_text())
            assert complete['passed']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == complete['files']['updates.jsonl']
            hashes[str(path.relative_to(base))] = digest
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    for record in row['examples']:
                        identity = (row['update'], record['query_index'])
                        if identity not in CASES:
                            continue
                        variants, targets = {}, {}
                        for device in ('cpu', 'cuda'):
                            scores = torch.tensor(record['scores'], dtype=torch.float32,
                                                  device=device, requires_grad=True)
                            teacher = scores.new_tensor(record['teacher'])
                            mask = torch.tensor(record['positive'], dtype=torch.bool, device=device)
                            margin = teacher[mask, None] - teacher[None, ~mask]
                            target = torch.sigmoid(margin / .04)
                            eligible = margin > 0
                            confidence = (2 * target - 1).clamp_min(0)
                            difference = scores[mask, None] - scores[None, ~mask]
                            terms = functional.binary_cross_entropy_with_logits(
                                difference, target, reduction='none')
                            loss = ((terms * confidence * eligible).sum(1)
                                    / eligible.sum(1).clamp_min(1)).mean()
                            (loss / 4).backward()
                            probability = target.detach().cpu().numpy().astype(np.float64)
                            delta = difference.detach().cpu().numpy().astype(np.float64)
                            allowed = eligible.cpu().numpy()
                            weight = np.maximum(2 * probability - 1, 0) * allowed
                            weight /= np.maximum(allowed.sum(1), 1)[:, None] * int(mask.sum())
                            stable = np.sum(weight * (
                                np.logaddexp(0., -delta) + (1 - probability) * delta))
                            targets[device] = probability
                            actual = float(loss.detach())
                            variants[device] = dict(loss=actual,
                                record_loss_error=abs(actual - record['loss']),
                                record_gradient_error=float(np.max(np.abs(
                                    scores.grad.cpu().numpy() - np.asarray(record['score_gradient'])))),
                                fp64_stable_loss_using_this_target=float(stable))
                        results.append(dict(update=identity[0], query_index=identity[1],
                            record_loss=record['loss'], variants=variants,
                            max_target_cpu_cuda_difference=float(np.max(np.abs(
                                targets['cpu'] - targets['cuda'])))))
                    assert time.monotonic() - started < 180, 'scalar probe time bound'
        assert len(results) == len(CASES)
        result = dict(cases=results, source_update_sha256=hashes,
            seconds=time.monotonic() - started, torch=torch.__version__,
            cuda=torch.version.cuda, physical_gpu=gpu, gpu_lock_held=True,
            model_inference=False, optimizer_updates=0, replacement_audit=False)
        with output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 3), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.research_root, args.gpu, args.output)
