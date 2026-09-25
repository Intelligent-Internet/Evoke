"""Independent NG72 audit math and denominator checks."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import ng72_diagnose as producer
import review_ng72_diagnosis as reviewer


class ReviewTests(unittest.TestCase):
    def test_four_product_derivation_matches_dense_algebra(self):
        rng = np.random.default_rng(72)
        for _ in range(20):
            q0, q1 = rng.integers(0, 3, (2, 12)) / 3
            a, b = rng.integers(0, 3, (2, 9, 12)) / 3
            scores, terms = reviewer.four_products(
                q0, q1, sparse.csr_matrix(a), sparse.csr_matrix(b))
            for mode, q, docs in (('00', q0, a), ('10', q1, a),
                                  ('01', q0, b), ('11', q1, b)):
                np.testing.assert_allclose(scores[mode], docs @ q, atol=1e-12)
            expected, total = producer.decomposition(
                q0, q1, sparse.csr_matrix(a), sparse.csr_matrix(b))
            for key in expected:
                np.testing.assert_allclose(terms[key], expected[key], atol=1e-12)
            np.testing.assert_allclose(terms['total'], total, atol=1e-12)

    def test_empty_support_has_zero_effect(self):
        scores, terms = reviewer.four_products(
            np.zeros(4), np.zeros(4), sparse.csr_matrix((3, 4)),
            sparse.csr_matrix(np.ones((3, 4))))
        for array in [*scores.values(), *terms.values()]:
            np.testing.assert_array_equal(array, np.zeros(3))

    def test_teacher_missing_pair_does_not_become_negative(self):
        records = (None, dict(pool=[1, 2, 3], scores=[.4, .2, .4]),
                   dict(pool=[1, 2, 3], teacher_scores=[.4, .2, .4]))
        for record in records:
            actual = reviewer.teacher(record, 1, [2, 3, 9])
            expected = [producer.teacher_pair(record, 1, r) for r in [2, 3, 9]]
            reviewer.equal(actual, expected)

    def test_query_balancing_not_flat_positive_mean(self):
        rows = [dict(query_id='a', value={k: 1. for k in reviewer.TERMS}),
                dict(query_id='a', value={k: 3. for k in reviewer.TERMS}),
                dict(query_id='b', value={k: 10. for k in reviewer.TERMS})]
        actual = reviewer.balanced(rows, 'value')
        self.assertEqual(actual['total'], 6.)
        self.assertEqual(reviewer.balanced([], 'value'), {})

    def test_recursive_compare_rejects_missing_or_changed_values(self):
        reviewer.equal(dict(a=[1, .5, True]), dict(a=[1, .5 + 1e-14, True]))
        for bad in ({}, dict(a=[1, .6, True]), dict(a=[1., .5, True]),
                    dict(a=[1, .5, 1]), dict(a=[1, .5, True, 4])):
            with self.assertRaises(ValueError):
                reviewer.equal(bad, dict(a=[1, .5, True]))


if __name__ == '__main__':
    unittest.main()
