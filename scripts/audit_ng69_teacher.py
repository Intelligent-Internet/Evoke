#!/usr/bin/env python3
"""Independently verify indexed TRAIN teacher artifacts before optimization."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def independently_recompute(query, documents, positive):
    # Explicit elementwise sum is separate from preparation's matrix product.
    score = np.sum(documents.astype(np.float64)
                   * query.astype(np.float64), axis=1)
    shifted = (score - np.max(score)) * 25
    teacher = np.exp(shifted)
    teacher = teacher / np.sum(teacher)
    require(np.any(positive), 'Missing positives')
    target = positive.astype(np.float64) / (2 * np.sum(positive))
    target += teacher / 2
    return score, target


def audit(base, run):
    exit_record = read(run / 'exit.json')
    tracking = read(run / 'clearml.json')
    require(exit_record['exit_code'] == 0 and exit_record['error'] is None
            and exit_record['owned_group_closed'], 'Phase not cleanly closed')
    require(tracking['actual_start'] and tracking['closed']
            and not tracking['post_hoc'] and not tracking['training'],
            'Invalid tracking receipt')
    inputs = read(run / 'inputs.json')
    require(sha(run / 'prepare_ng69_breadth.py') == inputs['source_sha256']
            and sha(run / 'protocol.md') == inputs['protocol_sha256'],
            'Frozen source or protocol changed')
    for name, digest in inputs['files'].items():
        require(sha(base / name) == digest, 'Input changed: ' + name)
    complete = read(run / 'complete.json')
    require(complete['passed'], 'Incomplete phase')
    for name, digest in complete['files'].items():
        require(sha(run / name) == digest, 'Output changed: ' + name)
    data = base / 'NG-0067/data'
    with (data / 'train.jsonl').open() as stream:
        train = [json.loads(line) for line in stream]
    pools, labels = read(data / 'train-pools.json'), read(data / 'train-labels.json')
    selection = read(run / 'selection.json')
    selected = selection['indices']
    require(selected[:1536] == list(range(1536)) and len(set(selected)) == 6144,
            'Selected cohort changed')
    expected = list(range(1536))
    for domain in ('fever', 'hotpotqa', 'nq'):
        candidates = sorted(
            (i for i in range(1536, len(train)) if train[i]['subset'] == domain),
            key=lambda i: hashlib.sha256(
                ('NG69-breadth-v1:' + train[i]['query_id']).encode()).hexdigest())
        expected.extend(candidates[:1536])
    require(selected == expected, 'Metric-blind selection changed')
    require(selection['query_ids'] == [train[i]['query_id'] for i in selected],
            'Selected query identities changed')
    ids = np.load(run / 'document-ids.npy')
    needed = sorted({d for i in selected for d in pools[i] if d >= 59111})
    require(ids.tolist() == list(range(59111)) + needed,
            'Indexed teacher document mapping changed')
    docs = np.load(run / 'documents.npy', mmap_mode='r')
    queries = np.load(run / 'train-query.npy', mmap_mode='r')
    require(docs.shape == (len(ids), 1024) and queries.shape == (6144, 1024),
            'Embedding shape changed')
    for matrix in (docs, queries):
        require(matrix.dtype == np.float32 and np.isfinite(matrix).all(),
                'Invalid embedding dtype or values')
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1,
                                   rtol=0, atol=2e-6)
    old = base / 'NG-0059/recovery/teacher'
    require(np.array_equal(docs[:59111], np.load(old / 'documents.npy'))
            and np.array_equal(queries[:1536], np.load(old / 'train-query.npy')),
            'Old embedding reuse is not bitexact')
    targets = read(run / 'targets.json')
    require(len(targets) == 6144 and targets[:1536] == read(old / 'targets.json'),
            'Old targets changed or missing new targets')
    lookup = {int(doc): i for i, doc in enumerate(ids)}
    maximum_score_error, maximum_target_error = 0., 0.
    for j, i in enumerate(selected):
        record = targets[j]
        pool = pools[i]
        positive = np.asarray([d in set(labels[i]) for d in pool])
        require(record['query_id'] == train[i]['query_id']
                and record['pool'] == pool
                and record['positive_mask'] == positive.tolist(),
                'Query, pool, or positive mapping changed')
        score, target = independently_recompute(
            queries[j], docs[[lookup[d] for d in pool]], positive)
        np.testing.assert_allclose(score, record['scores'], rtol=0, atol=1e-13)
        np.testing.assert_allclose(target, record['target'], rtol=0, atol=1e-12)
        maximum_score_error = max(maximum_score_error,
                                  float(abs(score - record['scores']).max()))
        maximum_target_error = max(maximum_target_error,
                                   float(abs(target - record['target']).max()))
    return {'passed': True, 'queries_audited': 6144,
            'new_queries': 4608, 'new_documents': len(needed),
            'old_reuse_bitexact': True,
            'maximum_score_error': maximum_score_error,
            'maximum_target_error': maximum_target_error,
            'locked_test_scored': False, 'full_model_reinference': False,
            'audit_source_sha256': sha(Path(__file__)),
            'inputs_sha256': sha(run / 'inputs.json'),
            'complete_sha256': sha(run / 'complete.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Do not overwrite an audit')
    result = audit(args.research_root, args.run)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
