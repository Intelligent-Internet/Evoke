"""Prepare a fixed endpoint-top100 union for TRAIN-only trajectory diagnosis.

No model inference or optimizer work is performed here. The union preserves
endpoint top100 rankings, not unknown intermediate full-corpus rankings.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy import sparse

import review_ng75_displacement as audit


PARENT = 'NG-0075/actual-displacement-v1'
PILOT = 'NG-0071/pilot-v1'
QA_SHA = '0c37c05b083c81037939254f7b3e66fc814893873ac261757f8907fee4e2cf5f'
NG72_SHA = '74f69b38d84456976810a89748b7ea7b2a9f50819471abb8dc527ef86c264671'


def endpoint_pool(initial, final):
    audit.require(initial['query_id'] == final['query_id'] and initial['split'] == final['split'] == 'TRAIN'
                  and initial['gold_ids'] == final['gold_ids'], 'endpoint identity/labels changed')
    audit.require(all(len(r['top100']) == len(set(r['top100'])) == 100 for r in (initial, final)),
                  'incomplete endpoint head')
    return sorted(set(initial['top100'] + final['top100'] + initial['gold_ids']))


def endpoint_rank(pool, scores, row):
    audit.require(len(pool) == len(scores) and np.isfinite(scores).all(), 'invalid endpoint scores')
    order = np.lexsort((pool, -np.asarray(scores)))
    ranking = [pool[i] for i in order]
    audit.require(ranking[:100] == row['top100'], 'endpoint top100 parity failed')
    ranks = [ranking.index(p) + 1 for p in row['gold_ids']]
    for actual, full in zip(ranks, row['gold_ranks'], strict=True):
        audit.require(actual == full if full <= 100 else actual > 100, 'endpoint gold cutoff rank changed')
    return ranks


def coverage(probes, records, initial, final):
    """Descriptive NG75 coverage only, never a probe-selection rule."""
    identities = {r['query_id'] for r in probes}
    result = {}
    for domain in ('fever', 'hotpotqa', 'nq'):
        ids = {q for q in identities if initial[q]['domain'] == domain}
        selected = [r for r in records if r['query_id'] in ids]
        lost = {(r['query_id'], r['positive_id'], n) for r in selected
                for j, n in enumerate(r['competitor_ids']) if r['baseline_ahead'][j] and not r['final_ahead'][j]}
        fixed = {(r['query_id'], r['positive_id'], r['rival_id']) for r in probes if r['query_id'] in ids}
        changes = [final[q]['ndcg10'] - initial[q]['ndcg10'] for q in sorted(ids)]
        result[domain] = dict(queries=len(ids), fixed_pairs=len(fixed),
            full_explanatory_pairs=sum(len(r['competitor_ids']) for r in selected),
            full_lost=len(lost), fixed_lost=len(fixed & lost),
            initial_to_D192_ndcg_changes=dict(Counter(
                'improved' if x > 0 else 'worse' if x < 0 else 'same' for x in changes)),
            ndcg_mean_change=float(np.mean(changes)) if changes else None)
    return result


def prepare(base, run):
    parent = base / PARENT
    audit.require(audit.sha(parent / 'inputs.json') == audit.MANIFEST, 'wrong NG75 manifest')
    old = audit.read(parent / 'inputs.json')
    audit.verify(base, old['dependencies'])
    inventory_path = base / 'NG-0075/actual-displacement-v1-remote-final.json'
    audit.require(audit.sha(inventory_path) == audit.INVENTORY, 'NG75 terminal inventory changed')
    audit.verify(parent, audit.read(inventory_path)['files'], exact=True)
    qa = base / 'NG-0075/actual-displacement-v1-mac-review-v1'
    audit.require(audit.sha(qa / 'complete.json') == QA_SHA and audit.read(qa / 'complete.json')['passed'],
                  'NG75 independent review missing')
    audit.verify(qa, audit.read(qa / 'complete.json')['files'], exact=True, exclude=('complete.json',))
    data = audit.read(parent / 'probe-data.json')
    ids = data['sentinel_ids']
    audit.require(len(ids) == len(set(ids)) == 24, 'fixed sentinel cohort changed')
    rankings = {}
    for model in ('initial', 'D-192'):
        with (base / PILOT / f'rank-{model}/rankings.jsonl').open() as stream:
            rankings[model] = {r['query_id']: r for line in stream
                               if (r := json.loads(line))['query_id'] in ids}
        audit.require(set(rankings[model]) == set(ids), 'missing endpoint queries')
    pools = {q: endpoint_pool(rankings['initial'][q], rankings['D-192'][q]) for q in ids}
    lexical = base / 'NG-0069/lexical-v1'
    qs = audit.read(lexical / 'queries.json')
    pos0 = {r['query_id']: i for i, r in enumerate(qs)}
    q_indices = [pos0[q] for q in ids]
    lq = sparse.load_npz(lexical / 'lexical-query.npz')[q_indices].astype(np.float64)
    ld = sparse.load_npz(lexical / 'lexical-document.npz').astype(np.float64)
    fixed = {q: (lq.getrow(i) @ ld[pools[q]].T).toarray().ravel() for i, q in enumerate(ids)}
    del lq, ld
    scores, cutoff_ranks = {}, {}
    for model, source in (('initial', base / 'NG-0069/evaluation-v2/encode-initial'),
                          ('D-192', base / PILOT / 'encode-D-192')):
        positions = pos0 if model == 'initial' else {
            q: i for i, q in enumerate(audit.read(source / 'query-ids.json'))}
        sq = sparse.load_npz(source / 'query.npz')[[positions[q] for q in ids]].astype(np.float64)
        sd = sparse.load_npz(source / 'document.npz').astype(np.float64)
        audit.require(sd.shape[0] == 233009, 'full endpoint corpus changed')
        scores[model], cutoff_ranks[model] = {}, {}
        for i, q in enumerate(ids):
            values = fixed[q] + (sq.getrow(i) @ sd[pools[q]].T).toarray().ravel()
            cutoff_ranks[model][q] = endpoint_rank(pools[q], values, rankings[model][q])
            scores[model][q] = values.tolist()
        del sq, sd
    needed = {d for values in pools.values() for d in values}
    docs = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for i, line in enumerate(stream):
            if i in needed:
                docs[str(i)] = json.loads(line)
    audit.require(i + 1 == 233009 and len(docs) == len(needed), 'document mapping incomplete')
    old_pairs = [r for r in data['cases']['probe-D-1']['probes'] if r['query_id'] in ids]
    ng72 = base / 'NG-0072/cross-codes-v1/diagnosis'
    audit.require(audit.sha(ng72 / 'complete.json') == NG72_SHA, 'NG72 evidence changed')
    audit.verify(ng72, audit.read(ng72 / 'complete.json')['files'], exact=True, exclude=('complete.json',))
    records = [json.loads(line) for line in (ng72 / 'margins.jsonl').read_text().splitlines()]
    gap = coverage(old_pairs, records, rankings['initial'], rankings['D-192'])
    evidence = dict(query_ids=ids, queries={q: data['queries'][q] for q in ids}, documents=docs,
        document_text_sha256={k: hashlib.sha256(v['text'].encode()).hexdigest() for k, v in docs.items()},
        pools=pools, gold={q: rankings['initial'][q]['gold_ids'] for q in ids},
        lexical_scores={q: v.tolist() for q, v in fixed.items()}, endpoint_scores=scores,
        endpoint_cutoff_ranks=cutoff_ranks, endpoint_rankings=rankings,
        fixed_query_selection='NG75 hash-selected 24 TRAIN sentinel; no outcome filtering',
        rival_selection='all initial top100 union all D192 top100 union all gold',
        intermediate_full_corpus_ranks_available=False, new_inference=False, locked_test_scored=False,
        scientific_training_enabled=False, trajectory_replay_enabled=False,
        observation_steps=list(range(0, 193, 16)), ng75_descriptive_coverage=gap)
    source = Path(__file__).resolve()
    run.mkdir(exist_ok=False, parents=True)
    shutil.copy2(source, run / source.name)
    shutil.copy2(source.parent / 'review_ng75_displacement.py', run / 'review_ng75_displacement.py')
    plan = source.parent.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0076-endpoint-union-trajectory-plan.zh.md'
    shutil.copy2(plan, run / plan.name)
    with (run / 'prepared.json').open('x') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write('\n')
    dependencies = dict(old['dependencies'])
    dependencies[str(inventory_path.relative_to(base))] = audit.sha(inventory_path)
    for folder in (parent, qa, ng72):
        dependencies.update({str(p.relative_to(base)): audit.sha(p) for p in folder.rglob('*') if p.is_file()})
    with (run / 'inputs.json').open('x') as stream:
        json.dump(dict(protocol='NG76_endpoint_union_preparation_v1', dependencies=dependencies,
                       source={str(p.relative_to(run)): audit.sha(p) for p in run.rglob('*')
                               if p.is_file() and p.name != 'inputs.json'},
                       model_inference=False, training_enabled=False), stream, indent=2)
        stream.write('\n')
    frozen = audit.read(run / 'inputs.json')
    audit.verify(base, frozen['dependencies'])
    audit.verify(run, frozen['source'])
    return dict(passed=True, manifest_sha256=audit.sha(run / 'inputs.json'), queries=len(ids),
                unique_documents=len(needed), query_document_codes=sum(map(len, pools.values())),
                gold_rival_pairs=sum(len(evidence['gold'][q]) * (len(pools[q]) - len(evidence['gold'][q])) for q in ids),
                endpoint_top100_and_gold_cutoff_rank_parity=True, new_inference=False,
                training_enabled=False, trajectory_replay_enabled=False, coverage=gap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    base, run = args.research_root.resolve(strict=True), args.run.resolve()
    audit.require(run.parent == base / 'NG-0076', 'fresh NG76 unit required')
    print(json.dumps(prepare(base, run), indent=2))


if __name__ == '__main__':
    main()
