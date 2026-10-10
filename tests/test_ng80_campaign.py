"""Synthetic-only NG80 split, readout and score controls."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/ng80_data.py'
if not SOURCE.exists():
    SOURCE = Path(__file__).with_name('ng80_data.py')
spec = importlib.util.spec_from_file_location('ng80_data', SOURCE)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_selection_is_deterministic_balanced_and_disjoint():
    rows = [{'query_id': f'{d}-{i}', 'query': f'{d} question {i}', 'subset': d}
            for d in study.DOMAINS for i in range(8)]
    result = study.selection(rows, 2)
    assert result == study.selection(rows, 2)
    assert len(result['train']) == 18 and len(result['diagnostic']) == 6
    assert not set(result['train']) & set(result['diagnostic'])
    assert result['diagnostic_is_independent_holdout'] is False


def test_split_rejects_text_leakage():
    rows = [{'query_id': f'{d}-{i}', 'query': 'same query', 'subset': d}
            for d in study.DOMAINS for i in range(8)]
    with pytest.raises(ValueError, match='crosses'):
        study.selection(rows, 2)


def test_panels_keep_every_positive_and_background():
    actual, background = study.panels([[0, 2], [3, 4]], [[0, 2], [4]], 10, 3)
    assert {0, 2} <= set(actual[0])
    assert set(background) <= set(actual[0]) & set(actual[1])
    assert len(actual[0]) == len(set(actual[0]))


@pytest.mark.parametrize('pool,gold', [([0, 0], [0]), ([0], [1]), ([11], [11])])
def test_panels_reject_bad_ids(pool, gold):
    with pytest.raises(ValueError):
        study.panels([pool], [gold], 10)


def test_ridge_matches_independent_augmented_least_squares():
    rng = np.random.default_rng(80)
    xq, xd = rng.normal(size=(7, 3)), rng.normal(size=(13, 3))
    yq, yd = rng.normal(size=(7, 5)), rng.normal(size=(13, 5))
    penalty = .01
    actual = study.ridge(xq, yq, xd, yd, penalty)
    design = np.concatenate([np.column_stack((xq, np.ones(7))) / np.sqrt(14),
                             np.column_stack((xd, np.ones(13))) / np.sqrt(26),
                             np.column_stack((np.eye(3) * np.sqrt(penalty), np.zeros(3)))])
    target = np.concatenate((yq / np.sqrt(14), yd / np.sqrt(26), np.zeros((3, 5))))
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_ridge_role_replication_does_not_change_weights():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(8, 3)), rng.normal(size=(8, 5))
    np.testing.assert_allclose(study.ridge(x, y, x, y),
                               study.ridge(x, y, np.tile(x, (4, 1)), np.tile(y, (4, 1))))


def test_centered_field_is_all_pair_margin_mse():
    rng = np.random.default_rng(3)
    for _ in range(20):
        s, t = rng.normal(size=(2, 17))
        margins = (s[:, None] - s) - (t[:, None] - t)
        assert study.centered_error(s, t) == pytest.approx(np.mean(margins ** 2) / 2)
        assert study.centered_error(s + 3, t) == pytest.approx(study.centered_error(s, t))


def test_metrics_use_stable_ids_and_all_positives():
    result = study.metrics([1., 1., 0.], [0., 1., 0.], [2, 1, 3], [1, 3])
    expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert result['panel_ndcg10'] == pytest.approx(expected)
    assert result['teacher_top10_overlap'] == 1


def test_projection_rejects_zero_or_nonfinite():
    with pytest.raises(ValueError):
        study.project(np.ones((3, 2)), np.zeros((3, 4)))
    with pytest.raises(ValueError):
        study.ridge(np.ones((2, 2)), np.full((2, 3), np.nan),
                    np.ones((2, 2)), np.ones((2, 3)))
