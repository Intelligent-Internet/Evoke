"""Independent CSC accumulation and rank/count review of saved midpoint data."""

import argparse
import heapq
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse

import audit_ng76_replay as io


def require(condition, message):
    if not condition:
        raise ValueError(message)


def column_scores(query, documents):
    """Scatter sorted vocabulary columns, never call a sparse matrix product."""
    require(sparse.isspmatrix_csc(documents) and documents.has_canonical_format
            and query.shape == (1, documents.shape[1]) and query.has_canonical_format,
            'noncanonical independent inputs')
    scores = np.zeros(documents.shape[0], dtype=np.float64)
    for feature, weight in zip(query.indices, query.data, strict=True):
        start, end = documents.indptr[feature:feature + 2]
        ids = documents.indices[start:end]
        scores[ids] += documents.data[start:end].astype(np.float64) * float(weight)
    require(np.isfinite(scores).all(), 'independent scores non-finite')
    return scores


def direct_rank(scores, gold):
    require(len(set(gold)) == len(gold) > 0 and all(0 <= i < len(scores) for i in gold), 'invalid gold')
    ids = np.arange(len(scores))
    ranks = [1 + int(np.count_nonzero((scores > scores[p]) | ((scores == scores[p]) & (ids < p))))
             for p in gold]
    top = heapq.nsmallest(100, range(len(scores)), key=lambda i: (-scores[i], i))
    ideal = math.fsum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    return dict(top100=top, gold_ids=gold, gold_ranks=ranks,
                ndcg10=math.fsum(1 / math.log2(r + 1) for r in ranks if r <= 10) / ideal,
                recall100=sum(r <= 100 for r in ranks) / len(gold))


def classify(positive, rival, scores):
    # Rank comparisons use individual scores, independently of margin sign tests.
    ahead = [(-p, positive) < (-n, rival) for p, n in scores]
    margins = [p - n for p, n in scores]
    return dict(ahead=ahead, margins=margins,
        increments=[margins[1] - margins[0], margins[2] - margins[1]],
        endpoint_lost=ahead[0] and not ahead[-1], endpoint_gained=not ahead[0] and ahead[-1],
        lost_already_midpoint=ahead[0] and not ahead[1] and not ahead[-1],
        lost_after_midpoint=ahead[0] and ahead[1] and not ahead[-1],
        regained=ahead[0] and not ahead[1] and ahead[-1],
        transient_gain=not ahead[0] and ahead[1] and not ahead[-1])


def verify_files(root, inventory):
    for name, digest in inventory.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and io.sha(path) == digest,
                'input hash/path changed: ' + name)


def review(base, run, output):
    manifest = json.loads((run / 'inputs.json').read_text())
    require(manifest['protocol'] == 'NG76_authentic_midpoint_v1'
            and manifest['optimizer_updates'] == manifest['document_forwards'] == 0
            and manifest['query_forwards'] == 26 and not manifest['locked_test_access'], 'wrong protocol')
    verify_files(base, manifest['dependencies'])
    verify_files(run, manifest['source'])
    require(io.sha(Path(__file__)) == manifest['source'][Path(__file__).name],
            'executing reviewer is not the frozen source')
    for name in ('encode-query', 'rank'):
        phase = run / name
        complete, exited, tracking = [json.loads((phase / n).read_text()) for n in
                                     ('complete.json', 'exit.json', 'clearml.json')]
        require(complete['passed'] and exited['exit_code'] == 0 and exited['error'] is None
                and exited['owned_group_closed'] and tracking['actual_start']
                and tracking['closed'] and tracking['outcome'] == 'passed', 'phase not closed')
        require({str(p.relative_to(phase)) for p in phase.rglob('*') if p.is_file()}
                == set(complete['files']) | {'complete.json'}, 'phase inventory changed')
        verify_files(phase, complete['files'])
    data = json.loads((run / 'prepared.json').read_text())
    ids = json.loads((run / 'encode-query/query-ids.json').read_text())
    require(ids == manifest['controls'] + data['query_ids'] and len(set(ids)) == 26
            and all(data['queries'][q]['split'] == 'TRAIN' for q in ids[2:]), 'query scope changed')
    q = sparse.load_npz(run / 'encode-query/query.npz')
    old = sparse.load_npz(base / 'NG-0071/pilot-v1/encode-D-96/query.npz')[:2]
    require(np.array_equal(q[:2].indices, old.indices) and np.array_equal(q[:2].indptr, old.indptr),
            'control support mismatch')
    np.testing.assert_allclose(q[:2].data, old.data, rtol=2e-5, atol=2e-7)
    q = q[2:]
    documents = sparse.load_npz(base / 'NG-0071/pilot-v1/encode-D-96/document.npz').tocsc()
    require(documents.shape[0] == 233009 and q.shape[0] == 24, 'global coverage changed')
    lexical = base / 'NG-0069/lexical-v1'
    rows = json.loads((lexical / 'queries.json').read_text())
    positions = {r['query_id']: i for i, r in enumerate(rows)}
    lq = sparse.load_npz(lexical / 'lexical-query.npz')[[positions[i] for i in ids[2:]]]
    ld = sparse.load_npz(lexical / 'lexical-document.npz').tocsc()
    saved = np.load(run / 'rank/scores.npy', mmap_mode='r')
    require(saved.shape == (24, 233009), 'saved full score shape changed')
    rankings = json.loads((run / 'rank/rankings.json').read_text())
    analysis = json.loads((run / 'rank/analysis.json').read_text())
    require(analysis['checkpoints'] == [0, 96, 192]
            and analysis['diagnostic_TRAIN_only'] is True
            and analysis['causal_domain_attribution'] is False
            and analysis['heldout_quality_evaluation'] is False
            and analysis['overall_goal_qualified'] is False, 'scientific scope changed')
    original_fixed = json.loads((run / 'rank/fixed-scores.json').read_text())
    middle, quality, maximum = {}, {}, 0.
    for i, identity in enumerate(ids[2:]):
        scores = column_scores(q.getrow(i), documents) + column_scores(lq.getrow(i), ld)
        maximum = max(maximum, float(np.max(np.abs(scores - saved[i]))))
        np.testing.assert_allclose(scores, saved[i], rtol=1e-12, atol=1e-12)
        ranked = direct_rank(scores, data['gold'][identity])
        for key in ('top100', 'gold_ids', 'gold_ranks'):
            require(ranked[key] == rankings[identity][key], 'independent rank mismatch')
        for key in ('ndcg10', 'recall100'):
            require(abs(ranked[key] - rankings[identity][key]) < 1e-12, 'independent quality mismatch')
        middle[identity] = scores[data['pools'][identity]].tolist()
        np.testing.assert_allclose(middle[identity], original_fixed[identity], rtol=1e-12, atol=1e-12)
        quality[identity] = ranked
    require([r['query_id'] for r in analysis['per_query']] == ids[2:],
            'per-query coverage/order changed')
    for row in analysis['per_query']:
        qid = row['query_id']
        require(row['domain'] == data['queries'][qid]['subset'], 'query domain changed')
        states = [data['endpoint_rankings']['initial'][qid], quality[qid],
                  data['endpoint_rankings']['D-192'][qid]]
        for metric in ('ndcg10', 'recall100'):
            np.testing.assert_allclose(row[metric], [s[metric] for s in states], rtol=0, atol=1e-12)
        require(row['middle_only_top100']
                == sorted(set(quality[qid]['top100']) - set(data['pools'][qid])),
                'per-query middle-only coverage changed')
    seen, domain_rows = set(), {d: [] for d in ('fever', 'hotpotqa', 'nq')}
    for row in analysis['pairs']:
        qid, p, n = row['query_id'], row['positive_id'], row['rival_id']
        require(row['domain'] == data['queries'][qid]['subset'], 'pair domain changed')
        key = qid, p, n
        require(key not in seen and p in data['gold'][qid] and n not in data['gold'][qid], 'invalid pair')
        seen.add(key)
        pool = data['pools'][qid]
        vectors = [data['endpoint_scores']['initial'][qid], middle[qid],
                   data['endpoint_scores']['D-192'][qid]]
        expected = classify(p, n, [(s[pool.index(p)], s[pool.index(n)]) for s in vectors])
        for name in ('margins', 'increments'):
            np.testing.assert_allclose(row[name], expected[name], rtol=1e-12, atol=1e-12)
        for name in set(expected) - {'margins', 'increments'}:
            require(row[name] == expected[name], 'independent crossing mismatch')
        require(abs(math.fsum(expected['increments']) - (expected['margins'][2] - expected['margins'][0]))
                < 1e-10, 'independent telescoping mismatch')
        domain_rows[data['queries'][qid]['subset']].append(expected)
    wanted = {(q, p, n) for q in data['query_ids'] for p in data['gold'][q]
              for n in data['pools'][q] if n not in data['gold'][q]}
    require(seen == wanted and len(seen) == 5226, 'incomplete pair coverage')
    for domain, pairs in domain_rows.items():
        target = analysis['domains'][domain]
        selected = [q for q in data['query_ids'] if data['queries'][q]['subset'] == domain]
        require(len(selected) == target['queries'] == 8 and len(pairs) == target['pairs'], 'domain count changed')
        for key in ('endpoint_lost', 'endpoint_gained', 'lost_already_midpoint',
                    'lost_after_midpoint', 'regained', 'transient_gain'):
            require(sum(p[key] for p in pairs) == target[key], 'domain crossing count changed')
        states = [data['endpoint_rankings']['initial'], quality, data['endpoint_rankings']['D-192']]
        for metric in ('ndcg10', 'recall100'):
            expected = [math.fsum(s[q][metric] for q in selected) / 8 for s in states]
            np.testing.assert_allclose(expected, target[metric], rtol=0, atol=1e-12)
        require(sum(len(set(quality[q]['top100']) - set(data['pools'][q])) for q in selected)
                == target['middle_only_top100'], 'middle-only coverage changed')
    verify_files(base, manifest['dependencies'])
    verify_files(run, manifest['source'])
    output.mkdir(exist_ok=False)
    with (output / 'results.json').open('x') as stream:
        json.dump(dict(passed=True, queries=24, corpus_documents=233009, pairs=5226,
            max_score_error=maximum, domains=analysis['domains'], no_new_inference=True,
            optimizer_updates=0, heldout_quality_evaluation=False, overall_goal_qualified=False,
            input_manifest_sha256=io.sha(run / 'inputs.json'), source_sha256=io.sha(Path(__file__))),
            stream, indent=2, allow_nan=False)
    print((output / 'results.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.research_root.resolve(strict=True), args.run.resolve(strict=True), args.output.resolve())


if __name__ == '__main__':
    main()
