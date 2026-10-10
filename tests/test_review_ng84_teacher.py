"""Independent NG84 metric and frozen-decision checks; no model imports."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    'review_ng84_teacher',
    Path(__file__).resolve().parents[1] / 'scripts/review_ng84_teacher.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_all_positive_metric_and_id_ties():
    result = MODULE.replay([9, 4, 2], [1., 1., 1.], [9, 2])
    assert result['positive_ranks'] == [1, 3]
    assert result['ndcg10'] == pytest.approx(1.5 / (1 + 1 / 1.584962500721156))
    assert result['recall10'] == result['mrr'] == 1
    assert MODULE.replay([2, 9, 4], [1., 1., 1.], [9, 2]) == result


def test_top10_cutoff_and_complete_recall_denominator():
    result = MODULE.replay(list(range(12)), list(range(12)), [0, 11])
    assert result['positive_ranks'] == [1, 12]
    assert result['recall10'] == .5


@pytest.mark.parametrize('ids,scores,positives', [
    ([1, 1], [0., 1.], [1]), ([1, 2], [0., float('nan')], [1]),
    ([1, 2], [0., 1.], [3]), ([1, 2], [0., 1.], []),
])
def test_rejects_invalid_scores_and_lost_positives(ids, scores, positives):
    with pytest.raises(ValueError):
        MODULE.replay(ids, scores, positives)


def test_gate_does_not_accept_small_or_uncertain_gain():
    assert MODULE.decide(.02, [.01, .03]) == 'admit_conditional_training'
    assert MODULE.decide(.004, [.001, .009]) == 'inconclusive_do_not_auto_launch'
    assert MODULE.decide(.02, [-.01, .05]) == 'inconclusive_do_not_auto_launch'
    assert MODULE.decide(-.02, [-.04, -.01]) == 'no_useful_signal_on_this_surface'


def test_hash_verification_rejects_modified_and_escaping_files(tmp_path):
    with pytest.raises(ValueError, match='unsafe'):
        MODULE.verify_files(tmp_path, {'../outside': 'bad'})
    path = tmp_path / 'sample.txt'
    path.write_text('frozen')
    expected = MODULE.digest(path)
    MODULE.verify_files(tmp_path, {'sample.txt': expected})
    path.write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        MODULE.verify_files(tmp_path, {'sample.txt': expected})
