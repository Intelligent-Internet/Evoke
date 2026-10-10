"""Analytic counterexamples and numeric contracts for NG82."""

import sys
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng82_diagnostics as study


@pytest.mark.parametrize('weighted', [False, True])
def test_full_rank_projection_preserves_scores_up_to_query_constant(weighted):
    rng = np.random.default_rng(82)
    d, q = rng.normal(size=(80, 12)), rng.normal(size=(30, 12))
    mean, dm, qm, _ = study.projection(d, q, 12, weighted)
    actual = (q @ qm) @ ((d - mean) @ dm).T
    expected = q @ d.T - (q @ mean)[:, None]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)


def test_weighted_projection_keeps_low_variance_but_query_relevant_direction():
    d = np.array([[20., 1.], [-20., 1.], [20., -1.], [-20., -1.]])
    q = np.array([[0., 1.], [0., -1.]])
    mean, dm, qm, _ = study.projection(d, q, 1, True)
    np.testing.assert_allclose((q @ qm) @ ((d - mean) @ dm).T, q @ d.T, atol=1e-12)
    mean, dm, qm, _ = study.projection(d, q, 1, False)
    assert np.max(np.abs((q @ qm) @ ((d - mean) @ dm).T - q @ d.T)) >= 1


def test_rank_deficient_query_covariance_is_regularized():
    rng = np.random.default_rng(820)
    d = rng.normal(size=(20, 8))
    q = np.ones((10, 8))
    _, dm, qm, info = study.projection(d, q, 4, True)
    assert np.isfinite(dm).all() and np.isfinite(qm).all() and info['ridge'] > 0


def test_utility_matches_independent_leave_one_out_and_not_joint_sum():
    c = np.array([[2., 2.], [1., -.5]])
    m = c.sum(1)
    actual = study.utility(sparse.csr_matrix(c), m, .5)
    loss = lambda x: np.logaddexp(0, -x / .5).mean()
    expected = [loss(m - c[:, j]) - loss(m) for j in range(2)]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert abs(actual.sum() - (loss(m - c.sum(1)) - loss(m))) > .1


def test_budget_keeps_unobserved_features_and_reports_discrete_undershoot():
    cost = np.array([0., 7., 3., 1.])
    mask = study.mask_to_budget(cost, [0, 1, 2, 3], 5.)
    assert mask.tolist() == [True, False, True, True]
    assert cost[mask].sum() == 4


@pytest.mark.parametrize('target', [-1., 12., np.nan])
def test_invalid_budget_rejected(target):
    with pytest.raises(ValueError):
        study.mask_to_budget(np.array([2., 3.]), [0, 1], target)


def test_mask_on_query_and_document_is_idempotent_and_equivalent():
    d = sparse.csr_matrix([[1., 2., 3.], [4., 0., 2.]])
    q = sparse.csr_matrix([[.5, 1., 2.]])
    keep = np.array([True, False, True])
    mq = study.masked_query(q, keep)
    left = (d @ mq.T).toarray()
    right = (d.multiply(keep) @ mq.T).toarray()
    np.testing.assert_array_equal(left, right)
    assert q.nnz == 3 and mq.nnz == 2


def test_tie_break_and_all_positive_metrics():
    result = study.compact_rank(np.array([1., 1., 0., 3.]), [0, 2])
    assert result['gold_ranks'] == [2, 4]
    assert result['top100'] == [3, 0, 1, 2]


def test_nonfinite_utility_rejected():
    with pytest.raises(ValueError):
        study.utility(sparse.csr_matrix([[np.nan]]), np.array([1.]), 1.)


def test_manifest_rejects_training_or_locked_test(tmp_path):
    manifest = dict(protocol='NG82_bounded_diagnostics_v1', phases=list(study.PHASES),
        dimensions=list(study.DIMS), teacher_pairs=list(map(list, study.PAIRS)),
        optimizer_updates=0, locked_test_access=True, independent_holdout=False,
        automatic_promotion=False)
    (tmp_path / 'inputs.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='NG82 protocol'):
        study.verify(tmp_path, tmp_path)


def test_manifest_binds_dependency_content(tmp_path, monkeypatch):
    parent = tmp_path / study.PARENT
    (parent / 'review').mkdir(parents=True)
    for name in ('inputs.json', 'review/results.json'):
        (parent / name).write_text('{}')
    payload = tmp_path / 'payload.json'
    payload.write_text('[1]')
    monkeypatch.setattr(study, 'paths', lambda _: {'data': payload})
    monkeypatch.setattr(study, 'INPUT_SHA', study.campaign.io.sha(parent / 'inputs.json'))
    monkeypatch.setattr(study, 'REVIEW_SHA', study.campaign.io.sha(parent / 'review/results.json'))
    run = tmp_path / 'NG-0082/diagnostics-v1'
    run.mkdir(parents=True)
    dependencies = {str(p.relative_to(tmp_path)): study.campaign.io.sha(p)
                    for p in (payload, parent / 'inputs.json', parent / 'review/results.json')}
    manifest = dict(protocol='NG82_bounded_diagnostics_v1', phases=list(study.PHASES),
        dimensions=list(study.DIMS), teacher_pairs=list(map(list, study.PAIRS)),
        optimizer_updates=0, locked_test_access=False, independent_holdout=False,
        automatic_promotion=False, source={}, dependencies=dependencies,
        files={'data': str(payload.relative_to(tmp_path))})
    (run / 'inputs.json').write_text(json.dumps(manifest))
    assert study.verify(tmp_path, run)['data'] == payload
    payload.write_text('[2]')
    with pytest.raises(ValueError, match='changed input'):
        study.verify(tmp_path, run)
