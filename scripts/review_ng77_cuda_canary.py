"""CPU-only mirror, raw cache-context and score audit of the NG77 CUDA gate.

No Torch, model inference, training or threshold changes. Full parameter VJPs
are attested by the sealed CUDA fixture, not independently recomputed here.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse

from review_ng77_preparation import probability, read, require, sha, verify


INPUT_SHA = '362a94b730e511bdb652e31c0bcae27feaaadc6b42c11ef0bed3f08d6cfe31ca'


def check_codes(actual, expected):
    require(actual.shape == expected.shape and actual.has_canonical_format
            and expected.has_canonical_format, 'invalid sparse shape/format')
    require(np.array_equal(actual.indptr, expected.indptr)
            and np.array_equal(actual.indices, expected.indices), 'support differs')
    delta = abs(actual.data.astype(float) - expected.data.astype(float))
    require(np.isfinite(delta).all() and (actual.data > 0).all()
            and (delta <= 2e-7 + 2e-5 * abs(expected.data)).all(), 'same-context value differs')
    return float(delta.max(initial=0))


def scalar_scores(query, documents, lexical):
    require(query.shape[0] == 1 and documents.shape[0] == len(lexical), 'score shape differs')
    q = dict(zip(query.indices.tolist(), map(float, query.data), strict=True))
    return [float(lexical[i]) + math.fsum(
        q.get(int(j), 0.) * float(v) for j, v in zip(
            documents.indices[documents.indptr[i]:documents.indptr[i + 1]],
            documents.data[documents.indptr[i]:documents.indptr[i + 1]], strict=True))
        for i in range(documents.shape[0])]


def review(base, run):
    require(sha(run / 'inputs.json') == INPUT_SHA, 'wrong canary identity')
    frozen = read(run / 'inputs.json')
    inventory = read(run.with_name(run.name + '-remote-inventory.json'))
    require(set(inventory['files']) == {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()},
            'mirror inventory differs')
    verify(run, inventory['files'])
    verify(run, frozen['source'])
    verify(base, frozen['dependencies'])
    output = run / 'canary'
    complete, result = read(output / 'complete.json'), read(output / 'results.json')
    verify(output, complete['files'])
    exited, tracking = read(output / 'exit.json'), read(output / 'clearml.json')
    require(complete['passed'] and result['passed'] and result['optimizer_updates'] == 0
            and not result['training_enabled'] and result['model_and_rng_unchanged']
            and result['context_qualified_comparison'] and not result['original_failed_attempt_relabelled'],
            'canary did not qualify')
    require(exited['exit_code'] == 0 and exited['error'] is None and exited['owned_group_closed']
            and exited['elapsed_seconds'] < 1800 and exited['peak_tree_rss_bytes'] <= 16 * 2**30,
            'phase not safely closed')
    require(tracking['actual_start'] and tracking['closed'] and tracking['outcome'] == 'passed'
            and read(run / 'controller-exit.json')['status'] == 'canary_complete_no_training',
            'tracking/controller not closed')
    queries = read(base / 'NG-0069/lexical-v1/queries.json')
    position = {q['query_id']: i for i, q in enumerate(queries)}
    cached = base / 'NG-0069/evaluation-v2/encode-initial'
    cq, cd = [sparse.load_npz(cached / (role + '.npz')) for role in ('query', 'document')]
    anchors = read(base / 'NG-0077/preparation-v1/audit/anchors.json')
    records = {r['query_id']: r for line in
               (base / 'NG-0071/pilot-step0-v1/witnesses.jsonl').read_text().splitlines()
               if (r := json.loads(line))}
    rows = [json.loads(line) for line in (output / 'per-query.jsonl').read_text().splitlines()]
    require([r['query_id'] for r in rows] == frozen['query_ids'] == result['query_ids'], 'query order differs')
    checked, cache_max, score_max, exposures = [], 0., 0., 0
    for i, row in enumerate(rows):
        identity = row['query_id']
        require(queries[position[identity]]['split'] == 'TRAIN', 'non-TRAIN query')
        reference = records[identity]
        q = sparse.load_npz(output / f'query-{i:02d}.npz')
        d = sparse.load_npz(output / f'documents-{i:02d}.npz')
        corpus = sparse.load_npz(output / f'corpus-documents-{i:02d}.npz')
        ids = read(output / f'corpus-ids-{i:02d}.json')
        require(ids == [j for group in frozen['corpus_contexts'][identity] for j in group], 'context IDs differ')
        formula = read(output / f'formula-{i:02d}.json')
        require(formula['bit_exact'] and formula['pool'] == reference['pool'], 'formula/pool receipt differs')
        cache_max = max(cache_max, check_codes(q, cq[[position[identity]]]), check_codes(corpus, cd[ids]))
        actual = scalar_scores(q, d, reference['lexical_scores'])
        error = np.max(abs(np.asarray(actual) - row['scores']))
        np.testing.assert_allclose(row['scores'], actual, rtol=2e-5, atol=2e-7)
        require(row['baseline_scores'] == reference['hybrid_scores'], 'baseline was changed')
        np.testing.assert_allclose(row['scores'], reference['hybrid_scores'], rtol=2e-5, atol=2e-7)
        gradient = [0.] * len(actual)
        for a in anchors[identity]['anchors']:
            p, n = a['indices']
            margin = row['scores'][p] - row['scores'][n]
            if margin < a['baseline_margin']:
                value = a['coefficient'] * (probability(margin) - probability(a['baseline_margin']))
                gradient[p] += value
                gradient[n] -= value
        mass = math.fsum(map(abs, gradient))
        require(abs(mass - row['keep_score_gradient_l1']) <= 1e-12
                and mass <= row['max_score_error'] + 1e-12
                and row['zero_coefficient_bit_exact'], 'noise contract differs')
        checked.append(dict(query_id=identity, corpus_documents=len(ids), pool_documents=len(actual),
                            scalar_score_error=float(error), keep_gradient_l1=mass))
        score_max = max(score_max, float(error))
        exposures += len(ids)
    D, K = math.fsum(r['D_score_gradient_l1'] for r in rows), math.fsum(r['keep_score_gradient_l1'] for r in rows)
    require(K <= .001 * D and abs(K / D - result['initial_keep_to_D_ratio']) < 1e-15, 'mean noise gate differs')
    for domain in ('fever', 'hotpotqa', 'nq'):
        receipt = read(output / f'vjp-{domain}.json')
        require(receipt['passed'] and receipt['artificial_two_document_fixture'] and receipt['not_a_training_qrel']
                and receipt['coefficients'] == result['vjp'][domain], 'VJP receipt differs')
        for coefficient, values in receipt['coefficients'].items():
            require(values['gradient_tensors'] == 107 and values['relative_l2'] <= 1e-5
                    and values['objective']['keep_loss'] > 0, 'VJP qualification failed')
        require(receipt['coefficients']['0.0']['original_shared_gradient_bit_exact'], 'original D gradient differs')
    return dict(passed=True, input_sha256=INPUT_SHA, canary_complete_sha256=sha(output / 'complete.json'),
        mirror_files=len(inventory['files']), mirror_bytes=inventory['bytes'], queries=checked,
        corpus_document_exposures=exposures, max_same_context_code_error=cache_max,
        max_cpu_scalar_vs_cuda_score_error=score_max, initial_keep_to_D_ratio=K / D,
        new_model_inference=False, optimizer_updates=0, quality_evaluation=False,
        full_parameter_vjp_independently_recomputed=False, sealed_cuda_vjp_receipts_verified=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.research_root, args.run)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))
