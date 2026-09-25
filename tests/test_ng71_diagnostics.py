import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_diagnostics as diagnostics


class SnapshotDerivativeTests(unittest.TestCase):
    def setUp(self):
        path = (Path(__file__).resolve().parents[1]
                / 'docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json')
        if not path.exists():
            path = Path(__file__).resolve().parent / 'ng0071-ranking-config.json'
        self.config = json.loads(path.read_text())
        self.row = dict(
            query_id='train-1', domain='nq', pool=[1, 2, 3, 4],
            original_pool=[1, 2, 3], positive_mask=[True, True, False, False],
            judged_negative_mask=[False] * 4, teacher_scores=[.5, .4, .1, .6],
            global_ranks=[101, 90, 130, 1], hybrid_scores=[1., 2., 0., 4.],
            lexical_scores=[.01] * 4, sources={'hybrid_head': [4]},
            total_positives=2, corpus_size=200,
            rank_scope='full_corpus_reference_snapshot')

    def test_four_arm_derivatives_and_pool_control_are_independent(self):
        result = diagnostics.diagnose(self.row, self.config)
        self.assertEqual(result['A']['pool'], result['B']['pool'])
        self.assertEqual(result['C']['pool'], result['D']['pool'])
        self.assertNotEqual(result['A']['pool'], result['C']['pool'])
        for arm in result.values():
            self.assertLessEqual(arm['max_independent_derivative_error'], 1e-12)
            self.assertAlmostEqual(sum(arm['score_gradient']), 0., places=12)
        self.assertEqual(result['D']['sources']['hybrid_head']['eligible_pairs'], 0)
        self.assertEqual(result['D']['score_gradient'][3], 0.)
        self.assertNotEqual(result['C']['score_gradient'][3], 0.)

    def test_D_only_preserves_original_math_and_zero_batch_guard(self):
        expected = diagnostics.diagnose(self.row, self.config)['D']
        config = copy.deepcopy(self.config)
        config['arms'] = {'D': config['arms']['D']}
        result = diagnostics.diagnose(self.row, config)
        self.assertEqual(result, {'D': expected})
        config['pilot'].update(train_queries=1, epochs=4, updates=1)
        row = {**self.row, 'arm_score_diagnostics': result}
        self.assertTrue(diagnostics.batch_audit([row], ['train-1'] * 4, config)['passed'])
        result['D']['eligible_pairs'] = 0
        with self.assertRaisesRegex(ValueError, 'unsupervised optimizer batch'):
            diagnostics.batch_audit([row], ['train-1'] * 4, config)

    def test_unknown_pool_is_not_silently_witness(self):
        config = copy.deepcopy(self.config)
        config['arms']['D']['pool'] = 'typo'
        with self.assertRaisesRegex(ValueError, 'unknown candidate pool'):
            diagnostics.diagnose(self.row, config)

    def test_unjudged_teacher_conflict_is_not_a_hard_negative(self):
        row = copy.deepcopy(self.row)
        row['teacher_scores'] = [.1, .1, .3, .4]
        result = diagnostics.diagnose(row, self.config)['D']
        self.assertEqual(result['eligible_pairs'], 0)
        self.assertEqual(result['loss'], 0.)
        self.assertEqual(result['score_gradient'], [0.] * 4)

    def test_soft_target_can_reduce_an_already_excessive_positive_margin(self):
        row = copy.deepcopy(self.row)
        row['teacher_scores'] = [.101, .101, .1, .1]
        row['hybrid_scores'] = [20., 20., 0., 0.]
        result = diagnostics.diagnose(row, self.config)['D']
        self.assertGreater(result['positive_score_gradient_sum'], 0.)
        self.assertTrue(all(d > 0 for d in result['pair_score_derivatives']))
        self.assertLess(sum(result['pair_coefficients']), .02)

    def test_original_pool_cannot_silently_lose_an_added_positive(self):
        row = copy.deepcopy(self.row)
        row['original_pool'] = [1, 3]
        with self.assertRaisesRegex(ValueError, 'lose a known positive'):
            diagnostics.original_record(row)

    def test_legacy_target_recomputed_with_all_positive_denominator(self):
        row = copy.deepcopy(self.row)
        row['teacher_scores'] = [0.] * 4
        actual = diagnostics.legacy_target(row, self.config['ranking'])
        np.testing.assert_array_equal(actual, [.375, .375, .125, .125])

    def test_duplicate_witness_attribution_is_rejected(self):
        row = copy.deepcopy(self.row)
        row['sources']['duplicate'] = [4]
        with self.assertRaisesRegex(ValueError, 'duplicate negative'):
            diagnostics.diagnose(row, self.config)

    def test_score_nan_or_candidate_local_ranks_fail_closed(self):
        for field, value in (('hybrid_scores', [1., 2., 0., np.nan]),
                             ('rank_scope', 'candidate_local')):
            row = copy.deepcopy(self.row)
            row[field] = value
            with self.assertRaises(ValueError):
                diagnostics.diagnose(row, self.config)

    def test_fixed_batches_retain_zero_query_but_reject_whole_zero_batch(self):
        config = copy.deepcopy(self.config)
        config['pilot'].update(train_queries=4, epochs=2, updates=2)
        rows = []
        for i in range(4):
            row = copy.deepcopy(self.row)
            row['query_id'] = str(i)
            row['arm_score_diagnostics'] = diagnostics.diagnose(row, config)
            rows.append(row)
        rows[0]['arm_score_diagnostics']['B']['eligible_pairs'] = 0
        order = ['0', '1', '2', '3'] * 2
        self.assertTrue(diagnostics.batch_audit(rows, order, config)['passed'])
        for row in rows:
            row['arm_score_diagnostics']['D']['eligible_pairs'] = 0
        with self.assertRaisesRegex(ValueError, 'unsupervised optimizer batch'):
            diagnostics.batch_audit(rows, order, config)

    def test_exposure_manifest_must_not_pad_or_drop_queries(self):
        config = copy.deepcopy(self.config)
        config['pilot'].update(train_queries=1, epochs=2, updates=1)
        row = copy.deepcopy(self.row)
        row['arm_score_diagnostics'] = diagnostics.diagnose(row, config)
        for order in (['train-1'] * 4, ['train-1', 'other']):
            with self.assertRaisesRegex(ValueError, 'exposure manifest'):
                diagnostics.batch_audit([row], order, config)


if __name__ == '__main__':
    unittest.main()
