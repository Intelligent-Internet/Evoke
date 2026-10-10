"""Review decisions must require ranking, recall, and fixed populations."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_ng83 as review


@pytest.mark.parametrize('lower_n,lower_r,passes', [(.001, -.004, True),
    (0., 0., False), (-.001, 0., False), (.001, -.006, False)])
def test_gate_requires_positive_ndcg_ci_and_noninferior_recall(lower_n, lower_r, passes):
    comparison = dict(ci95=[[lower_n, .01], [lower_r, .01]])
    reference = dict(ci95=[[.002, .01], [0., .01]])
    assert review.promotion_gate(comparison, reference) is passes


def test_different_query_order_is_rejected_before_metric_review():
    with pytest.raises(ValueError, match='query order'):
        review.check_rows([{'query_id': 'a'}], [{'query_id': 'b'}], [0], ['S'], [[1]])


def test_wrong_qrel_identity_rejected_before_metric_recomputation():
    rows = [dict(query_id='a', domain='fever', rankings={'S': {'gold_ids': [2]}})]
    with pytest.raises(ValueError, match='positive identity'):
        review.check_rows(rows, [dict(query_id='a', subset='fever')], [0], ['S'], [[1]])
