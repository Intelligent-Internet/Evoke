#!/usr/bin/env python3
"""Bounded TRAIN-only CPU diagnosis; does not resume the failed pipeline."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import psutil


EXPECTED_SOURCE = '87db295adc3be3bc44113d6e9e49ffef1f2d6b5b100b69fd1af79e9856f43362'
MODES = ('original', 'direct', 'batch16')


def alternative(documents, vector, mode):
    if mode == 'original':
        return np.concatenate([np.sum(documents[j:j + 2048] * vector, axis=1)
                               for j in range(0, len(documents), 2048)])
    assert mode in ('direct', 'batch16')
    return np.einsum('ij,j->i', documents, vector, optimize=False)


def primary(documents, queries, mode):
    if mode == 'batch16':
        return queries @ documents.T
    return np.stack([documents @ query for query in queries])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    assert not out.exists(), 'Never overwrite a diagnostic attempt'
    assert sys.version_info[:2] == (3, 12) and not sys.flags.optimize
    os.environ.update(CLEARML_OFFLINE_MODE='1', CUDA_VISIBLE_DEVICES='',
                      TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    os.environ['CLEARML_CACHE_DIR'] = str(out / 'tracking-cache')
    sys.path.insert(0, str(run))
    import ng69_pipeline as p

    assert p.sha(run / 'ng69_pipeline.py') == EXPECTED_SOURCE
    p.verify(run.parent.parent, run)
    p.completed(run, 'encode-dense')
    failed = p.read(run / 'rank-dense/exit.json')
    assert failed['exit_code'] == 1 and failed['owned_group_closed']
    assert p.read(run / 'rank-dense/clearml.json')['closed']
    assert not p.read(run / 'rank-dense/canary.json')['passed']
    c = p.legacy(run.parent.parent)
    c.setup()
    import metrics59 as metrics
    from clearml import Task

    out.mkdir(parents=True)
    tracked = {p.relative_to(run): p for folder in ('encode-dense', 'rank-dense')
               for p in (run / folder).rglob('*') if p.is_file()}
    frozen = {str(name): p.sha(path) for name, path in tracked.items()}
    p.write(out / 'inputs.json', {
        'files_relative_to_pipeline': frozen, 'source_sha256': p.sha(Path(__file__)),
        'pipeline_inputs_sha256': p.sha(run / 'inputs.json'),
        'modes': MODES, 'repeats': 3, 'cpu_threads': 4, 'wall_limit_seconds': 240,
        'host_rss_limit_bytes': 16 * 1024 ** 3, 'rank_attempt_resumed': False})
    Task.set_offline(True)
    task = Task.init(project_name='II42-NG', task_name='NG-0069/dense-rank-diagnostic',
        task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
        auto_connect_frameworks=False, auto_connect_arg_parser=False,
        auto_connect_streams=False, auto_resource_monitoring=False)
    receipt = {'task_id': task.id, 'actual_start': True, 'closed': False,
               'offline': True, 'remote_synced': False, 'post_hoc': False}
    p.write(out / 'clearml.json', receipt)
    task.connect({'inputs_sha256': p.sha(out / 'inputs.json')})
    started, peak, error = time.monotonic(), 0, None

    def timed_out(signum, frame):
        raise TimeoutError('Diagnostic 240-second wall bound')

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(240)
    try:
        documents = np.load(run / 'encode-dense/document.npy').astype(np.float64)
        queries = np.load(run / 'encode-dense/query.npy', mmap_mode='r')
        targets = p.read(run.parent / 'teacher-v1/targets.json')
        failed_rows = p.rows(run / 'rank-dense/rankings.jsonl')
        assert len(failed_rows) == 16 and all(r['split'] == 'TRAIN' for r in failed_rows)
        assert documents.shape == (233009, 1024) and queries.shape == (7872, 1024)
        groups = [list(range(16)), np.linspace(128, 6143, 16, dtype=int).tolist()]
        assert len(set(sum(groups, []))) == 32
        results, canonical = [], {}
        for repeat in range(3):
            modes = MODES[repeat:] + MODES[:repeat]
            for indices in groups:
                assert all(0 <= i < 6144 for i in indices)
                vectors = np.asarray(queries[indices], dtype=np.float64)
                for mode in modes:
                    tick = time.monotonic()
                    scores = primary(documents, vectors, mode)
                    primary_seconds = time.monotonic() - tick
                    alternate_seconds, verification_seconds, max_error = 0., 0., 0.
                    for column, i in enumerate(indices):
                        before = time.monotonic()
                        other = alternative(documents, vectors[column], mode)
                        alternate_seconds += time.monotonic() - before
                        before = time.monotonic()
                        score = scores[column]
                        difference = float(abs(score - other).max())
                        assert difference <= 1e-12
                        max_error = max(max_error, difference)
                        target = targets[i]
                        gold = [d for d, positive in
                                zip(target['pool'], target['positive_mask']) if positive]
                        row = metrics.rank_row(score, gold)
                        metrics.audit_row(row, gold, len(documents))
                        assert row['top100'] == p.top_independent(other)
                        ids = np.arange(len(documents))
                        ranks = [1 + np.count_nonzero(other > other[g])
                                 + np.count_nonzero((other == other[g]) & (ids < g))
                                 for g in gold]
                        assert row['gold_ranks'] == ranks
                        if i < 16:
                            old = failed_rows[i]
                            assert old['query_id'] == target['query_id']
                            assert old['top100'] == row['top100']
                            assert dict(zip(old['gold_ids'], old['gold_ranks'])) == dict(zip(gold, ranks))
                        identity = (row['top100'], dict(zip(gold, ranks)))
                        if i in canonical:
                            assert identity == canonical[i], 'Rank changed across kernels'
                        else:
                            canonical[i] = identity
                        verification_seconds += time.monotonic() - before
                    seconds = primary_seconds + alternate_seconds + verification_seconds
                    peak = max(peak, psutil.Process().memory_info().rss)
                    assert peak < 16 * 1024 ** 3
                    item = {'repeat': repeat, 'group': indices[0], 'mode': mode,
                            'queries': len(indices), 'seconds': seconds,
                            'primary_seconds': primary_seconds,
                            'alternative_seconds': alternate_seconds,
                            'verification_seconds': verification_seconds,
                            'max_score_error': max_error,
                            'projection_7872_seconds': seconds * 7872 / 16 * 1.5}
                    results.append(item)
                    print(json.dumps(item), flush=True)
        projections = {mode: max(r['projection_7872_seconds'] for r in results
                                  if r['mode'] == mode) for mode in MODES}
        p.write(out / 'results.json', {
            'passed': True, 'measurements': results, 'worst_projection': projections,
            'all_score_and_rank_checks_passed': True, 'train_queries': 32,
            'dev_queries_scored': 0, 'locked_test_scored': False,
            'model_inference_performed': False, 'full_pipeline_resumed': False,
            'original_rank_gate_seconds': 5100, 'numpy_version': np.__version__,
            'thread_environment': {key: os.environ.get(key) for key in
                ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'peak_rss_bytes': peak, 'seconds': time.monotonic() - started})
        for name, digest in frozen.items():
            assert p.sha(run / name) == digest, name
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        signal.alarm(0)
        task.close()
        receipt['closed'] = True
        p.write(out / 'clearml.json', receipt)
        p.write(out / 'exit.json', {'exit_code': 1 if error else 0, 'error': error,
                                  'seconds': time.monotonic() - started})
    p.write(out / 'complete.json', {'passed': True, 'files': {
        str(path.relative_to(out)): p.sha(path) for path in out.rglob('*')
        if path.is_file() and path.name != 'complete.json'}})


if __name__ == '__main__':
    main()
