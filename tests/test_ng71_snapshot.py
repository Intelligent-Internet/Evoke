import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_snapshot as snapshot


class ScoreAuditTests(unittest.TestCase):
    def test_external_selection_keeps_early_safety_guards(self):
        for prepared, surface, message in (
            ({'eligibility_gate_passed': False}, 'pilot', 'eligibility'),
            ({'eligibility_gate_passed': True}, 'LOCKED_TEST', 'TRAIN'),
            ({'eligibility_gate_passed': True, 'selection': {'pilot': []}}, 'pilot', 'unique'),
            ({'eligibility_gate_passed': True, 'selection': {'pilot': ['a', 'a']}}, 'pilot', 'unique'),
        ):
            with self.assertRaisesRegex(ValueError, message):
                snapshot.build_prepared(None, {}, None, prepared, surface=surface)

    def test_default_build_still_uses_original_data_audit(self):
        prepared = {'selection': 'fixture'}
        with patch.object(snapshot.data, 'audit', return_value=prepared) as audit:
            with patch.object(snapshot, 'build_prepared', return_value='ok') as builder:
                self.assertEqual(snapshot.build('base', 'config', 'out'), 'ok')
        audit.assert_called_once_with('base', 'config')
        builder.assert_called_once_with('base', 'config', 'out', prepared,
                                        surface='canary', reference=None)

    def test_reference_binds_shared_A96_and_exact_TRAIN_query_identity(self):
        result = {'reference_arm': 'A', 'reference_update': 96}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'query-ids.json').write_text('["b", "a"]')
            with patch('ng71_execution.sealed', return_value=result):
                _, positions = snapshot.reference_inputs(root, ['a', 'b'])
                self.assertEqual(positions, {'b': 0, 'a': 1})
                with self.assertRaisesRegex(ValueError, 'shared A96'):
                    snapshot.reference_inputs(root, ['a', 'locked'])
            with patch('ng71_execution.sealed', return_value={**result, 'reference_arm': 'D'}):
                with self.assertRaisesRegex(ValueError, 'shared A96'):
                    snapshot.reference_inputs(root, ['a', 'b'])

    def test_independent_head_and_selected_ranks_include_ties(self):
        scores = np.array([1., 1., 0., -1.])
        self.assertEqual(snapshot.verify_scores(scores, scores.copy(), 4, [0, 1, 3]), 0.)

    def test_in_tolerance_score_change_cannot_change_order(self):
        scores = np.array([1., 1., 0.])
        other = scores.copy()
        other[1] += 1e-13
        with self.assertRaisesRegex(ValueError, 'ordering'):
            snapshot.verify_scores(scores, other, 3)

    def test_selected_tail_rank_is_checked_independently(self):
        scores = -np.arange(200, dtype=np.float64)
        scores[150:152] = -150.
        other = scores.copy()
        other[151] += 1e-13
        with self.assertRaisesRegex(ValueError, 'selected/positive'):
            snapshot.verify_scores(scores, other, 200, [150, 151])

    def test_float32_local_nonfinite_and_error_are_rejected(self):
        good = np.array([1., 0.])
        for a, b, count in ((good.astype('f4'), good, 2),
                            (good, good, 3), (good, np.array([np.nan, 0.]), 2),
                            (good, good + 1e-10, 2)):
            with self.assertRaises(ValueError):
                snapshot.verify_scores(a, b, count)


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        path = (Path(__file__).resolve().parents[1]
                / 'docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json')
        if not path.exists():
            path = Path(__file__).resolve().parent / 'ng0071-ranking-config.json'
        self.config = json.loads(path.read_text())['ranking']
        self.row = dict(pool=[0, 1, 2, 3], original_pool=[0, 1, 2],
                        positive_mask=[True, True, False, False],
                        teacher_scores=[.5, .4, .1, .6], global_ranks=[101, 90, 130, 1],
                        sources={'hybrid_head': [3], 'hybrid_boundary': []},
                        total_positives=2, domain='nq')

    def test_easy_pool_coverage_does_not_hide_masked_head(self):
        summary = snapshot.supervision(self.row, self.config)
        self.assertEqual(summary['original_pool']['supervised_positive_positions'], [0, 1])
        head = summary['hybrid_head']
        self.assertEqual(head['eligible_pairs'], 0)
        self.assertEqual(head['masked_pairs'], 2)
        self.assertGreater(head['cross_top100_mass'], 0)
        self.assertEqual(head['cross_top100_mass'], head['masked_cross_top100_mass'])

    def test_true_positives_never_compared_as_negatives(self):
        bad = copy.deepcopy(self.row)
        bad['sources']['hybrid_head'] = [1]
        with self.assertRaises(ValueError):
            snapshot.supervision(bad, self.config)

    def test_union_coverage_does_not_double_count_positives(self):
        self.row['supervision'] = snapshot.supervision(self.row, self.config)
        report = snapshot.summarize([self.row], ['nq'])['nq']
        self.assertEqual(report['positives'], 2)
        self.assertEqual(report['supervised_positives'], 2)
        self.assertEqual(report['eligible_positive_fraction'], 1)
        self.assertEqual(report['sources']['hybrid_head']['target_quantiles'], [])


if __name__ == '__main__':
    unittest.main()
