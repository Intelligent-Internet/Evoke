"""Predeclared gate and query-unit uncertainty fixtures; no model inference."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_ng69_breadth as review


class ReviewTests(unittest.TestCase):
    def comparison(self):
        return {'macro': {'ndcg10': .02, 'recall100': 0},
                'ndcg10_ci95': [.01, .03],
                'domains': {d: {'ndcg10': .02} for d in review.DOMAINS}}

    def test_positive_gate_is_not_product_qualification(self):
        gate = review.breadth_gate(self.comparison())
        self.assertTrue(gate['passed'])
        self.assertFalse(gate['overall_goal_qualified'])
        self.assertFalse(gate['native_cost_evaluated'])

    def test_ci_touching_zero_fails(self):
        value = self.comparison()
        value['ndcg10_ci95'][0] = 0
        self.assertFalse(review.breadth_gate(value)['passed'])

    def test_recall_or_one_domain_regression_fails(self):
        value = self.comparison()
        value['macro']['recall100'] = -.006
        self.assertFalse(review.breadth_gate(value)['passed'])
        value = self.comparison()
        value['domains']['nq']['ndcg10'] = -.011
        self.assertFalse(review.breadth_gate(value)['passed'])

    def test_seed_variation_does_not_triple_query_population(self):
        domains = ['fever', 'hotpotqa', 'nq'] * 2
        baseline = np.full((3, 6, 2), .5)
        candidate = baseline.copy()
        candidate[0, :, 0] += .3
        candidate[1, :, 0] -= .3
        result = review.paired(candidate, baseline, domains)
        np.testing.assert_allclose(result['ndcg10_ci95'], [0, 0], atol=1e-15)
        self.assertEqual(result['queries'], 6)

    def test_constant_effect_and_missing_seed(self):
        domains = ['fever', 'hotpotqa', 'nq'] * 2
        baseline = np.full((3, 6, 2), .5)
        candidate = baseline + .02
        result = review.paired(candidate, baseline, domains)
        np.testing.assert_allclose(result['ndcg10_ci95'], [.02, .02])
        with self.assertRaises(ValueError):
            review.paired(candidate[:1], baseline[:1], domains)

    def test_sparse_cost_recomputed_not_trusted_from_json(self):
        encoded = {'counts': {}}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for role, count in (('query', 7872), ('document', 233009)):
                matrix = sparse.csr_matrix((np.ones(count, dtype=np.float32),
                    np.zeros(count, dtype=np.int32), np.arange(count + 1, dtype=np.int32)),
                    shape=(count, 2))
                sparse.save_npz(folder / (role + '.npz'), matrix)
                encoded['counts'][role] = {'nnz': count, 'csr_bytes': sum(
                    a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr))}
            ranked = {'counts': {'semantic_df_proxy': {
                'mean': 233009, 'p95': 233009, 'dev_new_mean': 233009}}}
            result = review.sparse_cost(folder, encoded, ranked)
            self.assertEqual(result['document']['nnz'], 233009)
            ranked['counts']['semantic_df_proxy']['mean'] = 1
            with self.assertRaises(ValueError):
                review.sparse_cost(folder, encoded, ranked)


if __name__ == '__main__':
    unittest.main()
