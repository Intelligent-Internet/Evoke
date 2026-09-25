"""Synthetic full-head coverage and TRAIN-only identity checks."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_ng77_retention as preparation


def data():
    record = dict(query_id='q', domain='nq', split='TRAIN',
        pool=[0, 2, 199], original_pool=[0, 199], positive_ids=[0],
        positive_mask=[True, False, False], judged_negative_mask=[False] * 3,
        teacher_scores=[.9, .3, .1], global_ranks=[1, 3, 200],
        hybrid_scores=[200., 198., 1.], total_positives=1, corpus_size=200,
        rank_scope='full_corpus_reference_snapshot',
        positive_visibility=[dict(document_id=0, origin='original_positive',
            truncated=True, supporting_evidence_visible='unknown_no_span_judgments')])
    head = dict(query_id='q', domain='nq', split='TRAIN', gold_ids=[0],
        gold_ranks=[1], top100=list(range(100)),
        top100_scores=list(range(200, 100, -1)))
    teacher = {d: .3 for d in range(100) if d not in record['pool']}
    config = dict(ndcg_cutoff=10, recall_cutoff=100, recall_weight=1.,
        pair_floor=.05, student_temperature=1., teacher_temperature=.04)
    return record, head, teacher, config


def test_missing_head_not_silently_added_to_anchor_or_forward_pool():
    record, head, teacher, config = data()
    before = deepcopy(record)
    anchors, rows, summary = preparation.query_audit(record, head, teacher, config)
    assert record == before
    assert len(anchors['anchors']) == 2 and summary['forward_documents'] == 3
    assert len(rows) == 100
    outside = [r for r in rows if not r['in_D_pool']]
    assert len(outside) == 98 and all(r['trusted'] for r in outside)
    assert all(not r['used_as_anchor'] and not r['original_D_eligible'] for r in outside)
    assert all(r['evidence_visibility'] == 'unknown_no_span_judgments' for r in rows)


@pytest.mark.parametrize('bad', ['head_score', 'head_rank', 'positive', 'split', 'overwrite_teacher'])
def test_unbound_saved_head_rejected(bad):
    record, head, teacher, config = data()
    if bad == 'head_score':
        head['top100_scores'][2] += .01
    elif bad == 'head_rank':
        record['global_ranks'][1] = 4
    elif bad == 'positive':
        head['gold_ids'] = [5]
    elif bad == 'split':
        head['split'] = 'DEV'
    else:
        teacher[0] = .4
    with pytest.raises(ValueError):
        preparation.query_audit(record, head, teacher, config)


def cohort():
    ids = [f'{domain}-{i}' for domain in ('fever', 'hotpotqa', 'nq') for i in range(128)]
    records = [dict(query_id=q, domain=q.split('-')[0], split='TRAIN',
        reference_update=0, independent_score_and_rank_verification_passed=True) for q in ids]
    selection = dict(pilot=ids, sentinel=['sentinel-' + q for q in ids])
    queries = {q: dict(split='TRAIN', subset=q.split('-')[0]) for q in ids}
    return records, selection, ids * 2, queries


@pytest.mark.parametrize('bad', ['sentinel', 'dev', 'locked', 'snapshot', 'duplicate', 'order', 'domain'])
def test_no_cohort_reselection_or_evaluation_query_anchors(bad):
    records, selection, order, queries = cohort()
    preparation.validate_cohort(records, selection, order, queries)
    q = records[0]['query_id']
    if bad == 'sentinel':
        selection['sentinel'][0] = q
    elif bad in ('dev', 'locked'):
        queries[q]['split'] = 'DEV' if bad == 'dev' else 'LOCKED_TEST'
    elif bad == 'snapshot':
        records[0]['reference_update'] = 96
    elif bad == 'duplicate':
        records[1] = records[0]
    elif bad == 'order':
        order[0] = order[1]
    else:
        records[0]['domain'] = 'fever-removed'
    with pytest.raises(ValueError):
        preparation.validate_cohort(records, selection, order, queries)


def test_selected_rankings_reject_duplicate_and_non_train(tmp_path):
    import json
    p = tmp_path / 'rows.jsonl'
    train = dict(query_id='a', split='TRAIN')
    p.write_text(json.dumps(train) + '\n' + json.dumps(dict(query_id='b', split='DEV')) + '\n')
    assert preparation.selected_rankings(p, ['a']) == {'a': train}
    with pytest.raises(ValueError):
        preparation.selected_rankings(p, ['b'])
    p.write_text((json.dumps(train) + '\n') * 2)
    with pytest.raises(ValueError):
        preparation.selected_rankings(p, ['a'])
