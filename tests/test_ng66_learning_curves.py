"""Small fixtures for offline, label-aware NG66 result review."""

import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / 'scripts/analyze_ng66_learning_curves.py'
SPEC = importlib.util.spec_from_file_location('ng66_review', PATH)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def row():
    return {'query_id': 'q', 'domain': 'nq', 'gold_ids': [0, 100],
            'gold_ranks': [1, 101], 'top100': list(range(100)),
            'top100_scores': list(range(100, 0, -1)),
            'ndcg10': 1 / (1 + 1 / review.math.log2(3)), 'recall100': .5}


class ReviewTests(unittest.TestCase):
    def test_all_positive_metric(self):
        review.audit_row(row(), [0, 100], 120)

    def test_false_metric_rejected(self):
        value = row()
        value['recall100'] = 1
        with self.assertRaises(ValueError):
            review.audit_row(value, [0, 100], 120)

    def test_inconsistent_positive_rejected(self):
        value = row()
        value['gold_ranks'][1] = 50
        with self.assertRaises(ValueError):
            review.audit_row(value, [0, 100], 120)

    def test_wrong_tie_order_rejected(self):
        value = row()
        value['top100_scores'] = [1] * 100
        value['top100'][1:3] = [2, 1]
        with self.assertRaises(ValueError):
            review.audit_row(value, [0, 100], 120)

    def test_harm_is_not_hidden_by_recall_average(self):
        before, after = row(), row()
        after['gold_ranks'] = [101, 1]
        result = review.harm([before], [after])
        self.assertEqual(len(result['dropped_top100']), 1)
        self.assertEqual(len(result['entered_top100']), 1)
        self.assertEqual(result['positive_rank_worsened'], 1)

    def test_pair_identity_rejected(self):
        before, after = row(), row()
        after['query_id'] = 'other'
        with self.assertRaises(ValueError):
            review.harm([before], [after])

    def test_zero_bootstrap(self):
        self.assertEqual(review.paired_interval([0, 0, 0], review.DOMAINS), [0, 0])


if __name__ == '__main__':
    unittest.main()
