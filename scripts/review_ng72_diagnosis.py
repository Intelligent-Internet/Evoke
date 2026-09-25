"""Read-only NG72 artifact/metric/margin audit; no full-corpus rerank or inference."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import psutil
from scipy import sparse

import ng72_diagnose as d


INPUT_SHA = '8ca39510880622470e7423f389a34d5dd5b06c68b504d0c47b6a7f5ec467b794'
COMPLETE_SHA = '74f69b38d84456976810a89748b7ea7b2a9f50819471abb8dc527ef86c264671'
TERMS = (*d.MODAL_TERMS, *d.SUPPORT_TERMS, 'total')
STATES = ('lost', 'retained', 'gained', 'stayed_behind')


def equal(actual, expected):
    if isinstance(expected, dict):
        d.require(set(actual) == set(expected), 'dictionary keys differ')
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, list):
        d.require(len(actual) == len(expected), 'array length differs')
        for a, b in zip(actual, expected, strict=True):
            equal(a, b)
    elif isinstance(expected, float):
        d.require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0),
                  'numeric review mismatch')
    else:
        d.require(type(actual) is type(expected) and actual == expected,
                  'value or type mismatch')


def four_products(q0, q1, a, b):
    """Derive effects independently from four coordinate-product score tables."""
    products = {mode: doc.multiply(query).tocsr()
                for mode, query, doc in
                (('00', q0, a), ('10', q1, a), ('01', q0, b), ('11', q1, b))}
    for product in products.values():
        product.eliminate_zeros()
    scores = {mode: np.asarray(p.sum(1)).ravel()
              for mode, p in products.items()}
    old, new = products['00'], products['11']
    common_old = np.asarray(old.multiply(new.astype(bool)).sum(1)).ravel()
    common_new = np.asarray(new.multiply(old.astype(bool)).sum(1)).ravel()
    effects = dict(query=scores['10'] - scores['00'],
                   document=scores['01'] - scores['00'],
                   interaction=scores['11'] - scores['10'] - scores['01'] + scores['00'],
                   support_lost=common_old - scores['00'],
                   support_gained=scores['11'] - common_new,
                   common_support_weight=common_new - common_old,
                   total=scores['11'] - scores['00'])
    return scores, effects


def teacher(record, positive, competitors):
    lookup = {} if record is None else dict(zip(
        record['pool'], record.get('teacher_scores', record.get('scores')), strict=True))
    result = []
    for doc in competitors:
        if positive not in lookup or doc not in lookup:
            result.append(dict(status='unobserved', margin=None, confidence=None))
        else:
            margin = lookup[positive] - lookup[doc]
            result.append(dict(status='agrees' if margin > 0 else 'opposes' if margin < 0 else 'tie',
                               margin=margin, confidence=math.tanh(margin / .08)))
    return result


def balanced(rows, field):
    values = {}
    for row in rows:
        values.setdefault(row['query_id'], []).append(row[field])
    return {term: statistics.mean(statistics.mean(p[term] for p in pieces)
                                  for pieces in values.values())
            for term in TERMS} if values else {}


def aggregate(ranks, margins, audit):
    quality, harms, effects = {}, {}, {}
    for surface in ('TRAIN_PILOT', 'TRAIN_SENTINEL'):
        for domain in d.DOMAINS:
            key = surface + '/' + domain
            selected = {mode: [r for r in rows if r['surface'] == surface
                               and r['domain'] == domain] for mode, rows in ranks.items()}
            d.require(all(len(rows) == 128 for rows in selected.values()), 'unbalanced rank group')
            quality[key] = {mode: {metric: statistics.mean(r[metric] for r in rows)
                                   for metric in ('ndcg10', 'recall100')}
                            for mode, rows in selected.items()}
            harms[key] = {mode: audit.harm(selected['00'], selected[mode]) for mode in ('10', '01', '11')}
            group = [r for r in margins if r['surface'] == surface and r['domain'] == domain]
            effects[key] = {}
            for origin in ('all', 'original_positive', 'added_vs_original'):
                part = [r for r in group if origin == 'all' or r['origin'] == origin]
                effects[key][origin] = dict(queries=len({r['query_id'] for r in part}),
                    positives=len(part), pair_count=sum(r['competitors'] for r in part),
                    query_balanced_mean=balanced(part, 'means'))
            effects[key]['retention'] = {state: dict(
                pairs=sum(r[state + '_count'] for r in group),
                query_balanced_contribution=balanced(group, state)) for state in STATES}
            effects[key]['teacher_status_pair_counts'] = dict(sum(
                (Counter(r['teacher_counts']) for r in group), Counter()))
    return dict(quality=quality, harm=harms, margin_effects=effects)


def review(base, run):
    start, peak = time.monotonic(), 0

    def guard():
        nonlocal peak
        peak = max(peak, psutil.Process().memory_info().rss)
        d.require(peak < 4 * 1024 ** 3 and psutil.virtual_memory().available > 8 * 1024 ** 3
                  and time.monotonic() - start < 900, 'bounded read-only audit stopped')

    guard()
    d.require(d.sha(run / 'inputs.json') == INPUT_SHA
              and d.sha(run / 'diagnosis/complete.json') == COMPLETE_SHA, 'NG72 anchor changed')
    frozen = d.verify(base, run)
    d.require(d.collect_inputs(base) == frozen['dependencies'], 'model/input chain changed')
    _, _, execution = d.parent_modules(base)
    import analyze_ng66_learning_curves as audit
    observed = execution.sealed(run / 'diagnosis')
    for name in ('dev_scored', 'locked_test_scored', 'new_model_inference', 'native_cost_evaluated'):
        d.require(observed[name] is False, 'scope differs: ' + name)
    d.require(observed['training_updates'] == 0 and observed['queries'] == 768
              and observed['ranked_rows'] == 3072
              and observed['frozen_00_11_rank_parity'] is True
              and observed['max_independent_score_error'] <= 1e-12,
              'observation counts or global score checks differ')
    parent, lexical = base / d.PARENT, base / 'NG-0069/lexical-v1'
    queries, labels = d.read(lexical / 'queries.json'), d.read(lexical / 'labels.json')
    indexes = {q['query_id']: i for i, q in enumerate(queries)}
    selection = d.read(parent / 'selection.json')
    ordered = selection['pilot'] + selection['sentinel']
    d.require(len(set(ordered)) == len(ordered) == 768, 'TRAIN coverage differs')
    encoded = d.read(parent / 'encode-D-192/query-ids.json')
    encoded_lookup = {identity: i for i, identity in enumerate(encoded)}
    initial = base / d.INITIAL
    q0, q1 = [sparse.load_npz(folder / 'query.npz')
              for folder in (initial, parent / 'encode-D-192')]
    a, b = [sparse.load_npz(folder / 'document.npz')
            for folder in (initial, parent / 'encode-D-192')]
    lq, ld = [sparse.load_npz(lexical / f'lexical-{role}.npz') for role in ('query', 'document')]
    guard()
    refs = {mode: {r['query_id']: r for r in audit.rows(parent / f'rank-{name}/rankings.jsonl')
                   if r['query_id'] in set(ordered)} for mode, name in (('00', 'initial'), ('11', 'D-192'))}
    source = {r['query_id']: r for r in d.read(base / 'NG-0070/provenance-v1/pair-provenance.json')}
    targets = {r['query_id']: r for r in d.read(base / 'NG-0069/teacher-v1/targets.json')}
    witnesses = {step: {r['query_id']: r for r in audit.rows(folder / 'witnesses.jsonl')}
                 for step, folder in ((0, base / 'NG-0071/pilot-step0-v1'), (96, parent / 'snapshot-96'))}
    wanted = {g for identity in ordered for g in labels[indexes[identity]]}
    keys = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                keys[index] = json.loads(line)['key']
    ranks, means, max_error = {mode: [] for mode in d.MODES}, [], 0.
    rank_rows = iter(audit.rows(run / 'diagnosis/rankings.jsonl'))
    with (run / 'diagnosis/margins.jsonl').open() as margin_stream:
        for number, identity in enumerate(ordered):
            guard()
            index, ei = indexes[identity], encoded_lookup[identity]
            query, gold = queries[index], labels[index]
            surface = 'TRAIN_PILOT' if number < 384 else 'TRAIN_SENTINEL'
            d.require(query['split'] == 'TRAIN', 'non-TRAIN query forbidden')
            records = {}
            for mode in d.MODES:
                row = next(rank_rows)
                equal([row['mode'], row['query_id'], row['domain'], row['surface'], row['split']],
                      [mode, identity, query['subset'], surface, 'TRAIN'])
                audit.audit_row(row, gold, 233009)
                if mode in refs:
                    equal({k: row[k] for k in ('gold_ids', 'gold_ranks', 'top100', 'top100_scores', 'ndcg10', 'recall100')},
                          {k: refs[mode][identity][k] for k in ('gold_ids', 'gold_ranks', 'top100', 'top100_scores', 'ndcg10', 'recall100')})
                records[mode] = row
                ranks[mode].append(row)
            rivals = set()
            for mode in ('00', '11'):
                rivals.update(records[mode]['top100'][:10] + records[mode]['top100'][79:100])
            for reference in (targets[identity], witnesses[0].get(identity), witnesses[96].get(identity)):
                if reference:
                    rivals.update(reference['pool'])
            rivals = sorted(rivals - set(gold))
            selected = sorted(set(rivals) | set(gold) | set().union(*(set(r['top100']) for r in records.values())))
            positions = {doc: i for i, doc in enumerate(selected)}
            x, y = q0.getrow(index).toarray().ravel().astype(float), q1.getrow(ei).toarray().ravel().astype(float)
            scores, terms = four_products(x, y, a[selected].astype(float), b[selected].astype(float))
            lexical_scores = np.asarray(ld[selected].astype(float).multiply(
                lq.getrow(index).toarray().ravel().astype(float)).sum(1)).ravel()
            scores = {mode: value + lexical_scores for mode, value in scores.items()}
            for mode, record in records.items():
                actual = scores[mode][[positions[g] for g in record['top100']]]
                max_error = max(max_error, float(np.max(abs(actual - record['top100_scores']))))
                np.testing.assert_allclose(actual, record['top100_scores'], rtol=0, atol=1e-12)
            origins = {p['document_key']: p['provenance'] for p in source[identity]['pairs']}
            d.require(source[identity]['status'] == 'resolved' and set(origins) == {keys[g] for g in gold},
                      'positive provenance differs')
            for positive in gold:
                row = json.loads(next(margin_stream))
                equal([row['query_id'], row['surface'], row['domain'], row['positive_id'], row['origin'], row['competitor_ids']],
                      [identity, surface, query['subset'], positive, origins[keys[positive]], rivals])
                equal(row['positive_ranks'], {mode: r['gold_ranks'][gold.index(positive)] for mode, r in records.items()})
                p, r = positions[positive], [positions[g] for g in rivals]
                effects = {term: values[p] - values[r] for term, values in terms.items()}
                equal(row['effects'], {term: values.tolist() for term, values in effects.items()})
                before, after = [scores[mode][p] - scores[mode][r] for mode in ('00', '11')]
                equal(row['before_margin'], before.tolist())
                equal(row['after_margin'], after.tolist())
                ahead0 = [v > 0 or v == 0 and positive < doc for v, doc in zip(before, rivals)]
                ahead1 = [v > 0 or v == 0 and positive < doc for v, doc in zip(after, rivals)]
                equal(row['baseline_ahead'], [bool(v) for v in ahead0])
                equal(row['final_ahead'], [bool(v) for v in ahead1])
                teachers = {}
                for name, ref in (('original', targets[identity]), ('step0', witnesses[0].get(identity)),
                                  ('step96', witnesses[96].get(identity))):
                    teachers[name] = teacher(ref, positive, rivals)
                    equal(row['teacher_' + name], teachers[name])
                visibility = next((v for v in witnesses[0].get(identity, {}).get('positive_visibility', [])
                                   if v['document_id'] == positive), None)
                equal(row['visibility'], visibility)
                d.require(row['evidence_span_visible'] == 'unknown_no_span_judgments'
                          and row['competitors_are_judged_negatives'] is False, 'unsupported judgment')
                item = dict(query_id=identity, domain=query['subset'], surface=surface,
                            origin=row['origin'], competitors=len(rivals),
                            means={term: statistics.mean(values) for term, values in effects.items()},
                            teacher_counts=dict(Counter(t['status'] for t in teachers[
                                'step96' if identity in witnesses[96] else 'original'])))
                states = [('retained' if f else 'lost') if old else ('gained' if f else 'stayed_behind')
                          for old, f in zip(ahead0, ahead1)]
                for state in STATES:
                    item[state + '_count'] = states.count(state)
                    item[state] = {term: math.fsum(v for v, s in zip(values, states) if s == state) / len(rivals)
                                   for term, values in effects.items()}
                means.append(item)
            if (number + 1) % 128 == 0:
                print('NG72_REVIEW', number + 1, time.monotonic() - start, flush=True)
        d.require(next(margin_stream, None) is None and next(rank_rows, None) is None, 'extra observation rows')
    summary = aggregate(ranks, means, audit)
    equal(summary, {k: observed[k] for k in summary})
    d.require(len(means) == observed['positive_rows'] == 1371, 'positive coverage differs')
    d.verify(base, run)
    execution.sealed(run / 'diagnosis')
    guard()
    return dict(passed=True, **summary, queries=768, ranked_rows=3072, positive_rows=len(means),
                max_recomputed_head_score_error=max_error, margins_from_frozen_codes=True,
                cross_code_full_corpus_ranks_recomputed=False, stored_rank_contract_audited=True,
                frozen_00_11_rank_parity=True, producer_full_corpus_two_path_checks_passed=True,
                training_updates=0, new_model_inference=False, dev_scored=False, locked_test_scored=False,
                input_sha256=INPUT_SHA, diagnosis_complete_sha256=COMPLETE_SHA,
                reviewer_source_sha256=d.sha(Path(__file__)), peak_rss_bytes=peak,
                elapsed_seconds=time.monotonic() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d.require(not args.output.exists() and args.output.parent.resolve() == args.run.parent.resolve(),
              'new review output must remain outside frozen run')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    d.write(args.output, review(args.research_root, args.run))
