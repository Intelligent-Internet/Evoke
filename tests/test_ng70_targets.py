"""Check label-floor dilution without interpreting added positives as errors."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_ng70_targets import target_summary


class TargetTests(unittest.TestCase):
    def test_equal_teacher_two_positives_share_label_floor(self):
        row = {'scores': [0., 0., 0.], 'positive_mask': [True, True, False],
               'target': [.25 + 1 / 6, .25 + 1 / 6, 1 / 6]}
        result = target_summary(row, ['original_positive', 'added_vs_original', 'source_negative'])
        self.assertAlmostEqual(result['original_label_floor'], .25)
        self.assertAlmostEqual(result['original_target_mass'], 5 / 12)
        self.assertFalse(result['target_negative_above_any_original'])

    def test_one_original_positive_retains_half_floor(self):
        row = {'scores': [0., 0.], 'positive_mask': [True, False], 'target': [.75, .25]}
        result = target_summary(row, ['original_positive', 'source_negative'])
        self.assertEqual(result['original_label_floor'], .5)
        self.assertFalse(result['has_added'])

    def test_target_drift_is_rejected(self):
        row = {'scores': [0., 0.], 'positive_mask': [True, False], 'target': [.8, .2]}
        with self.assertRaises(AssertionError):
            target_summary(row, ['original_positive', 'source_negative'])

    def test_teacher_strong_negative_can_exceed_diluted_original(self):
        teacher = np.array([0., 0., 1.])
        teacher = np.exp((teacher - teacher.max()) / .04)
        teacher /= teacher.sum()
        row = {'scores': [0., 0., 1.], 'positive_mask': [True, True, False],
               'target': (np.array([.25, .25, 0]) + .5 * teacher).tolist()}
        result = target_summary(row, ['original_positive', 'added_vs_original', 'source_negative'])
        self.assertTrue(result['target_negative_above_any_original'])


if __name__ == '__main__':
    unittest.main()
