#!/usr/bin/env python3
"""Audit frozen TRAIN supervision and prepare blinded relevance review."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


SEEDS = (59059, 66061, 66067)
DOMAINS = ('fever', 'hotpotqa', 'nq')
STRATA = ('teacher_conflict', 'student_regression', 'control')


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


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def train_prefix(path, queries):
    """Read only the expected TRAIN prefix, not held-out ranking records."""
    result = []
    with path.open() as stream:
        for query in queries:
            row = json.loads(next(stream))
            require(row['split'] == 'TRAIN'
                    and row['query_id'] == query['query_id']
                    and row['domain'] == query['subset'],
                    'TRAIN identity/split mismatch')
            result.append(row)
    return result


def target_stats(record):
    scores = np.asarray(record['scores'], dtype=np.float64)
    positive = np.asarray(record['positive_mask'])
    target = np.asarray(record['target'], dtype=np.float64)
    pool = record['pool']
    require(scores.ndim == 1 and scores.shape == positive.shape
            == target.shape and len(pool) == len(scores)
            and len(set(pool)) == len(pool), 'Invalid target/pool shape')
    require(positive.dtype == bool and positive.any() and (~positive).any(),
            'Expected positives and source-negative candidates')
    require(np.isfinite(scores).all() and np.isfinite(target).all()
            and (target > 0).all(), 'Invalid probabilities/scores')
    teacher = np.exp((scores - scores.max()) / .04)
    teacher /= teacher.sum()
    expected = .5 * positive / positive.sum() + .5 * teacher
    require(np.allclose(expected, target, rtol=1e-12, atol=1e-14),
            'Historical target equation changed')
    pos, neg = scores[positive], scores[~positive]
    pos_target, neg_target = target[positive], target[~positive]
    return {
        'pool_size': len(pool), 'positive_count': int(positive.sum()),
        'teacher_positive_mass': float(teacher[positive].sum()),
        'target_positive_mass': float(target[positive].sum()),
        'teacher_effective_support': float(np.exp(
            -np.sum(teacher * np.log(teacher)))),
        'teacher_top_is_source_negative': bool(neg.max() > pos.max()),
        'teacher_any_positive_below_source_negative':
            bool((neg[:, None] > pos).any()),
        'target_any_positive_below_source_negative':
            bool((neg_target[:, None] > pos_target).any()),
        'teacher_discordant_pairs': int((neg[:, None] > pos).sum()),
        'target_discordant_pairs': int(
            (neg_target[:, None] > pos_target).sum()),
        'positive_negative_pairs': len(pos) * len(neg),
        'target_min_positive_log_gap':
            float(np.log(pos_target.min()) - np.log(neg_target.max())),
    }


def witnesses(rank, pool):
    gold = set(rank['gold_ids'])
    positive_ranks = rank['gold_ranks']
    require(len(gold) == len(positive_ranks) > 0, 'Invalid positives')
    # Only an item above a positive in/near the top ten can change nDCG@10.
    selected = [doc for i, doc in enumerate(rank['top100'][:10], 1)
                if doc not in gold and any(r > i for r in positive_ranks)]
    return {'ids': selected, 'count': len(selected),
            'outside_pool': sum(doc not in pool for doc in selected)}


def summarize(records):
    result = {}
    for domain in ('ALL', *DOMAINS):
        part = [r for r in records
                if domain == 'ALL' or r['domain'] == domain]
        require(bool(part), 'Empty domain')
        counts = ('teacher_top_is_source_negative',
                  'teacher_any_positive_below_source_negative',
                  'target_any_positive_below_source_negative',
                  'teacher_discordant_pairs', 'target_discordant_pairs',
                  'positive_negative_pairs')
        result[domain] = {'queries': len(part)}
        result[domain]['counts'] = {
            key: sum(r[key] for r in part) for key in counts}
        for key in ('pool_size', 'positive_count', 'teacher_positive_mass',
                    'target_positive_mass', 'teacher_effective_support',
                    'target_min_positive_log_gap'):
            values = [r[key] for r in part]
            result[domain][key] = {
                'mean': statistics.mean(values),
                'p10': float(np.quantile(values, .1)),
                'p50': statistics.median(values),
                'p90': float(np.quantile(values, .9))}
        result[domain]['rank_harm_queries'] = sum(
            r['mean_train_ndcg_delta'] < -1e-13 for r in part)
        result[domain]['witness_by_seed'] = {}
        for seed in SEEDS:
            affected = [r['witness_by_seed'][str(seed)] for r in part
                        if r['witness_by_seed'][str(seed)]['count']]
            result[domain]['witness_by_seed'][str(seed)] = {
                'affected_queries': len(affected),
                'witness_count': sum(r['count'] for r in affected),
                'outside_pool_count': sum(r['outside_pool'] for r in affected),
                'mean_outside_fraction_per_affected_query':
                    statistics.mean(r['outside_pool'] / r['count']
                                    for r in affected) if affected else None}
    return result


def select_review(records, per_cell=10):
    selected, cells = [], {}
    for domain in DOMAINS:
        for stratum in STRATA:
            cell = [r for r in records
                    if r['domain'] == domain and r['stratum'] == stratum]
            cell.sort(key=lambda r: digest('NG68-v1:' + r['query_id']))
            selected += cell[:per_cell]
            cells[domain + '/' + stratum] = {
                'available': len(cell), 'selected': min(per_cell, len(cell))}
    return selected, cells


def blind_packet(selected, queries, documents, rankings, dense, bm25):
    packets, private = [], []
    for row in selected:
        i, identity = row['query_index'], row['query_id']
        gold = set(rankings[i]['gold_ids'])
        choices = set(gold)
        for ranking in (rankings[i], dense[i], bm25[i]):
            choices.update([d for d in ranking['top100'] if d not in gold][:3])
        ordered = sorted(choices, key=lambda d: digest(
            'NG68-card-v1:' + identity + ':' + documents[d]['key']))
        cards, mapping = [], []
        for doc in ordered:
            card_id = digest('NG68-blind:' + identity + ':'
                             + documents[doc]['key'])[:24]
            cards.append({'id': card_id, 'text': documents[doc]['text'],
                          'relevance': None, 'evidence_span': None,
                          'reason': None})
            mapping.append({'id': card_id, 'document_index': doc,
                            'document_key': documents[doc]['key'],
                            'source_positive': doc in gold})
        packet_id = digest('NG68-query:' + identity)[:24]
        packets.append({'id': packet_id, 'query': queries[i]['query'],
                        'cards': cards})
        private.append({'id': packet_id, 'query_id': identity,
                        'stratum': row['stratum'], 'domain': row['domain'],
                        'cards': mapping})
    return packets, private


def analyze(reference, run):
    queries = rows(reference / 'data/train.jsonl')
    targets = read(reference / 'recovery/teacher/targets.json')
    labels = read(reference / 'data/positive-labels.json')[:len(queries)]
    require(len(queries) == len(targets) == 1536, 'TRAIN cohort changed')
    paths = [reference / 'data/train.jsonl',
             reference / 'recovery/teacher/targets.json',
             reference / 'data/positive-labels.json',
             reference / 'data/documents.jsonl']
    rankings = {}
    for seed in SEEDS:
        for epoch in (1, 4):
            path = run / f'seed-{seed}/epoch-{epoch}/eval/rankings.jsonl'
            paths.append(path)
            rankings[seed, epoch] = train_prefix(path, queries)
    baseline = {}
    for model in ('pplx', 'bm25'):
        path = reference / f'phases/eval59-quality/{model}.jsonl'
        paths.append(path)
        baseline[model] = train_prefix(path, queries)
    records = []
    for i, (query, target, gold) in enumerate(zip(queries, targets, labels)):
        require(query['query_id'] == target['query_id'], 'Teacher ID mismatch')
        require([d for d, p in zip(target['pool'], target['positive_mask'])
                 if p] == gold, 'Teacher positives changed')
        for values in (*rankings.values(), *baseline.values()):
            require(values[i]['gold_ids'] == gold, 'Rank positives changed')
        stats = target_stats(target)
        delta = statistics.mean(rankings[s, 4][i]['ndcg10']
                                - rankings[s, 1][i]['ndcg10'] for s in SEEDS)
        stratum = ('teacher_conflict' if stats[
            'teacher_any_positive_below_source_negative'] else
            'student_regression' if delta < -1e-13 else 'control')
        records.append({'query_index': i, 'query_id': query['query_id'],
                        'domain': query['subset'], **stats,
                        'mean_train_ndcg_delta': delta, 'stratum': stratum,
                        'witness_by_seed': {str(s): witnesses(
                            rankings[s, 4][i], set(target['pool']))
                            for s in SEEDS}})
    selected, cells = select_review(records)
    packet, private = blind_packet(
        selected, queries, rows(reference / 'data/documents.jsonl'),
        rankings[SEEDS[0], 4], baseline['pplx'], baseline['bm25'])
    result = {'stage': 'NG-0068', 'scope': 'frozen TRAIN only',
              'fresh_inference': False, 'locked_test_scored': False,
              'source_negatives_are_judged_irrelevant': False,
              'positive_label_origin': 'mixed; pair provenance unresolved',
              'summary': summarize(records), 'review_cells': cells,
              'review_queries': len(packet),
              'review_cards': sum(len(r['cards']) for r in packet),
              'human_judgments_collected': 0,
              'input_sha256': {str(p): sha(p) for p in paths}}
    return result, records, packet, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Do not overwrite an existing attempt')
    result, records, packet, private = analyze(args.reference, args.run)
    args.output.mkdir(parents=True)
    for name, value in (('results.json', result), ('per-query.json', records),
                        ('annotation-blind.json', packet),
                        ('annotation-private-key.json', private)):
        with (args.output / name).open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False,
                      allow_nan=False)
            stream.write('\n')
    manifest = {'passed': True, 'source_sha256': sha(Path(__file__)),
                'files': {p.name: sha(p) for p in args.output.iterdir()}}
    with (args.output / 'complete.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
