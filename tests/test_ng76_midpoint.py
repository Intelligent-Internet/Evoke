"""Synthetic query-only provenance, sparse arithmetic and lifecycle tests."""

import ast
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_pilot as pipeline
import ng76_authentic_midpoint as execution
import ng76_midpoint_math as primary
import review_ng76_midpoint as audit


def payload():
    ids = [f'{domain}-{i}' for domain in ('fever', 'hotpotqa', 'nq') for i in range(8)]
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    data = dict(query_ids=ids, queries={q: dict(query_id=q, query=q, split='TRAIN',
        subset=q.split('-')[0], text_sha256=sha(q)) for q in ids},
        pools={q: [1, 3] for q in ids}, gold={q: [3] for q in ids},
        documents={'1': {'text': 'a'}, '3': {'text': 'b'}},
        document_text_sha256={'1': sha('a'), '3': sha('b')},
        scientific_training_enabled=False, trajectory_replay_enabled=False,
        locked_test_scored=False, observation_steps=list(range(0, 193, 16)))
    controls = ['pilot-0', 'pilot-1']
    queries = [dict(query_id=q, query=q, split='TRAIN', text_sha256=sha(q)) for q in controls + ids]
    return data, queries, controls


@pytest.mark.parametrize('bad', ['locked', 'text', 'order', 'duplicate', 'sentinel'])
def test_query_identity_and_exposure_are_frozen(bad):
    data, queries, controls = payload()
    execution.query_contract(data, queries, controls)
    if bad == 'locked':
        queries[0]['split'] = 'LOCKED_TEST'
    elif bad == 'text':
        queries[0]['query'] = 'changed'
    elif bad == 'order':
        queries[0], queries[1] = queries[1], queries[0]
    elif bad == 'duplicate':
        controls[0] = controls[1]
    else:
        queries[2]['query'] = 'changed'
        queries[2]['text_sha256'] = hashlib.sha256(b'changed').hexdigest()
    with pytest.raises(ValueError):
        execution.query_contract(data, queries, controls)


@pytest.mark.parametrize('bad', ['support', 'shape', 'nonfinite', 'large_error'])
def test_control_parity_cannot_be_passed_by_nearby_but_different_support(bad):
    old = sparse.csr_matrix(np.array([[.5, 0., 1.], [0., .7, 1.]], dtype=np.float32))
    new = old.copy()
    assert execution.compare_codes(new, old) == 0
    if bad == 'support':
        new.indices[0] = 1
    elif bad == 'shape':
        new = new[:1]
    elif bad == 'nonfinite':
        old.data[0] = new.data[0] = np.nan
    else:
        new.data[0] += .1
    with pytest.raises((ValueError, AssertionError)):
        execution.compare_codes(new, old)


def test_independent_column_sum_matches_product_with_empty_columns_rows():
    d = sparse.csr_matrix(np.array([[1., 0., 2., 0.], [0., 0., 0., 0.], [2., 0., 3., 0.]]))
    q = sparse.csr_matrix(np.array([[.4, 2., .3, 7.]]))
    expected = (q @ d.T).toarray().ravel()
    np.testing.assert_allclose(audit.column_scores(q, d.tocsc()), expected, rtol=0, atol=1e-15)


def test_full_rank_independent_gold_count_includes_positives_outside_top100():
    values = np.zeros(200)
    gold = [0, 9, 99, 100, 199]
    a, b = primary.rank(values, gold), audit.direct_rank(values, gold)
    for key in ('top100', 'gold_ids', 'gold_ranks', 'recall100'):
        assert a[key] == b[key]
    assert a['gold_ranks'] == [1, 10, 100, 101, 200]
    assert abs(a['ndcg10'] - b['ndcg10']) < 1e-15
    assert a['recall100'] == .6


@pytest.mark.parametrize('states', [tuple(bool(i & (1 << j)) for j in range(3)) for i in range(8)])
def test_all_crossing_patterns_and_telescoping(states):
    ids = ['fever', 'hotpotqa', 'nq']
    scores = [[0., 1. if flag else -1., 0.] + [-2.] * 98 for flag in states]
    data = dict(query_ids=ids, queries={q: {'subset': q} for q in ids},
        pools={q: list(range(101)) for q in ids}, gold={q: [1] for q in ids},
        endpoint_scores={'initial': {q: scores[0] for q in ids}, 'D-192': {q: scores[2] for q in ids}},
        endpoint_rankings={'initial': {q: primary.rank(scores[0], [1]) for q in ids},
                           'D-192': {q: primary.rank(scores[2], [1]) for q in ids}})
    result = primary.analyze(data, {q: scores[1] for q in ids},
                             {q: primary.rank(scores[1], [1]) for q in ids})
    assert len(result['pairs']) == 300
    for row in result['pairs']:
        expected = audit.classify(1, row['rival_id'], [(s[1], s[row['rival_id']]) for s in scores])
        assert all(row[k] == v for k, v in expected.items())
        assert sum(row['increments']) == row['margins'][2] - row['margins'][0]
    assert not result['overall_goal_qualified'] and not result['causal_domain_attribution']


def test_tie_classification_respects_numeric_document_id():
    assert audit.classify(1, 2, [(0., 0.)] * 3)['ahead'] == [True] * 3
    assert audit.classify(2, 1, [(0., 0.)] * 3)['ahead'] == [False] * 3


def test_execution_has_no_optimizer_or_document_encoder_call():
    tree = ast.parse(Path(execution.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not any(n.func.attr in ('backward', 'optimizer_step', 'train_chunk', 'make_optimizer') for n in calls)
    encodes = [n for n in calls if n.func.attr == 'encode'
               and isinstance(n.func.value, ast.Name) and n.func.value.id == 'encoder']
    assert len(encodes) == 1 and encodes[0].args[1].value == 'query'


@pytest.mark.parametrize('bound', [True, 0, 17, 8.5])
def test_cpu_guard_can_only_tighten_existing_rss_limit(bound):
    with pytest.raises(ValueError, match='RSS limit'):
        pipeline.phase(SimpleNamespace(), 'rank', [], rss_limit_gib=bound)
