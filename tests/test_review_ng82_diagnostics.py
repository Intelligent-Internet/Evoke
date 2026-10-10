"""Independent positive-rank receipts must reject inconsistent results."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_ng82_diagnostics as review


def fixture():
    return review.study.compact_rank(np.arange(200, dtype=float), [199, 180, 0])


def test_independent_metric_and_top100_check():
    record = fixture()
    np.testing.assert_allclose(review.recompute(record),
                               [record['ndcg10'], 2 / 3], rtol=0, atol=1e-12)


@pytest.mark.parametrize('change', ['metric', 'top100', 'positive', 'rank'])
def test_corrupt_receipt_is_rejected(change):
    record = copy.deepcopy(fixture())
    if change == 'metric':
        record['ndcg10'] += .01
    elif change == 'top100':
        record['top100'][0], record['top100'][1] = record['top100'][1], record['top100'][0]
    elif change == 'positive':
        record['gold_ids'][-1] = record['gold_ids'][0]
    else:
        record['gold_ranks'][-1] = 0
    with pytest.raises(ValueError):
        review.recompute(record)
