import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_ng79_preparation as review


def row():
    return dict(positive_mask=[True, True, False, False], total_positives=2,
                judged_negative_mask=[False] * 4, global_ranks=[11, 2, 1, 120],
                hybrid_scores=[.1, .2, .3, .4], teacher_scores=[.5, .4, .1, .6])


def config():
    return dict(ndcg_cutoff=10, recall_cutoff=100, pair_floor=.05, recall_weight=1.,
                student_temperature=1., teacher_temperature=.04)


def test_teacher_opposed_is_not_silently_supervised():
    result = review.scalar_objective(row(), config())
    assert result['pair_indices'] == [[0, 2], [1, 2]]
    assert result['eligible_pairs'] == 2
    assert result['score_gradient'][3] == 0
    assert abs(sum(result['score_gradient'])) < 1e-15


def test_all_positive_denominator_and_zero_supervision_preserved():
    value = row()
    value['teacher_scores'] = [.5, 0., .1, .6]
    result = review.scalar_objective(value, config())
    assert result['supervised_positives'] == 1
    assert result['pair_coefficients'][0] < .5
    value['teacher_scores'] = [0., 0., 1., 1.]
    result = review.scalar_objective(value, config())
    assert result['loss'] == 0 and result['score_gradient'] == [0.] * 4


@pytest.mark.parametrize('mutation', ['judge', 'missing_gold', 'duplicate_rank', 'rank_zero'])
def test_invalid_labels_or_global_ranks_fail(mutation):
    value = copy.deepcopy(row())
    if mutation == 'judge':
        value['judged_negative_mask'][2] = True
    elif mutation == 'missing_gold':
        value['total_positives'] = 3
    elif mutation == 'duplicate_rank':
        value['global_ranks'][2] = 11
    else:
        value['global_ranks'][0] = 0
    with pytest.raises(ValueError):
        review.scalar_objective(value, config())
