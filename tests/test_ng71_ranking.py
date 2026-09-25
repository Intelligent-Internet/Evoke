"""Verify NG71's objective math without training or touching research data."""

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ng71_ranking import binary_metrics, loss_and_gradient, pair_weight


class RankingTests(unittest.TestCase):
    def evaluate(self, **changes):
        args = {
            'scores': np.array([0.2, -0.1, 0.3, 0.7]),
            'positive_mask': np.array([True, True, False, False]),
            'global_ranks': np.array([94, 121, 100, 3]),
            'teacher_scores': np.array([0.9, 0.7, 0.4, 0.8]),
            'judged_negative_mask': np.array([False, False, True, False]),
            'total_positives': 2,
            'corpus_size': 233009,
            'rank_scope': 'full_corpus_reference_snapshot',
        }
        args.update(changes)
        return loss_and_gradient(**args)

    def test_gradient_matches_finite_differences(self):
        scores = np.array([0.2, -0.1, 0.3, 0.7])
        actual = self.evaluate(scores=scores)['gradient']
        for index in range(scores.size):
            step = np.zeros_like(scores)
            step[index] = 1e-6
            numeric = (
                self.evaluate(scores=scores + step)['loss']
                - self.evaluate(scores=scores - step)['loss']
            ) / 2e-6
            self.assertAlmostEqual(actual[index], numeric, places=8)

    def test_translation_invariance(self):
        before = self.evaluate()
        after = self.evaluate(scores=np.array([0.2, -0.1, 0.3, 0.7]) + 12)
        self.assertAlmostEqual(before['loss'], after['loss'])
        np.testing.assert_allclose(before['gradient'], after['gradient'])
        self.assertAlmostEqual(float(before['gradient'].sum()), 0)

    def test_metric_swap_matches_binary_full_corpus_metrics(self):
        before = binary_metrics(np.array([2, 94, 130]), 1000)
        after = binary_metrics(np.array([2, 101, 130]), 1000)
        expected = sum(abs(a - b) for a, b in zip(before, after))
        self.assertAlmostEqual(pair_weight(94, 101, 3, pair_floor=0), expected)
        after = binary_metrics(np.array([2, 3, 130]), 1000)
        expected = sum(abs(a - b) for a, b in zip(before, after))
        self.assertAlmostEqual(pair_weight(94, 3, 3, pair_floor=0), expected)

    def test_perfect_head_does_not_imply_full_recall(self):
        before = binary_metrics(np.array(list(range(1, 11)) + [94]), 1000)
        after = binary_metrics(np.array(list(range(1, 11)) + [102]), 1000)
        self.assertEqual(before[0], 1.0)
        self.assertEqual(after[0], 1.0)
        self.assertEqual(before[1], 1.0)
        self.assertAlmostEqual(after[1], 10 / 11)

    def test_floor_covers_pairs_outside_both_cutoffs(self):
        self.assertEqual(pair_weight(130, 150, 2, pair_floor=0), 0)
        self.assertAlmostEqual(pair_weight(130, 150, 2), 0.025)

    def test_boundary_priority_exceeds_same_side_tail(self):
        self.assertGreater(pair_weight(101, 100, 2), pair_weight(101, 120, 2))

    def test_conflicting_teacher_unjudged_pair_is_unresolved(self):
        result = self.evaluate(judged_negative_mask=np.zeros(4, dtype=bool))
        self.assertEqual(result['unresolved_pairs'], 1)
        self.assertEqual(result['eligible_pairs'], 3)

    def test_teacher_ties_are_not_hard_negatives(self):
        result = self.evaluate(
            teacher_scores=np.zeros(4),
            judged_negative_mask=np.zeros(4, dtype=bool),
        )
        self.assertEqual(result['loss'], 0)
        self.assertEqual(result['supervised_positives'], 0)
        self.assertEqual(result['unresolved_pairs'], 4)
        np.testing.assert_array_equal(result['gradient'], np.zeros(4))

    def test_no_positive_is_used_as_a_negative(self):
        result = self.evaluate(
            positive_mask=np.ones(4, dtype=bool),
            judged_negative_mask=np.zeros(4, dtype=bool),
            total_positives=4,
        )
        self.assertEqual(result['eligible_pairs'], 0)
        self.assertEqual(result['total_positives'], 4)

    def test_missing_supervision_keeps_all_positive_denominator(self):
        result = self.evaluate(
            scores=np.zeros(4),
            teacher_scores=np.array([1.0, 0.0, 0.5, 0.5]),
            judged_negative_mask=np.zeros(4, dtype=bool),
        )
        self.assertEqual(result['supervised_positives'], 1)
        confidence = 2 / (1 + math.exp(-0.5 / 0.04)) - 1
        self.assertAlmostEqual(result['loss'], confidence * math.log(2) / 2)
        self.assertEqual(result['gradient'][1], 0)

    def test_low_teacher_confidence_is_not_normalized_away(self):
        result = self.evaluate(
            scores=np.zeros(4),
            teacher_scores=np.array([1e-6, 1e-6, 0.0, 0.0]),
            judged_negative_mask=np.zeros(4, dtype=bool),
        )
        self.assertEqual(result['eligible_pairs'], 4)
        self.assertLess(result['loss'], 1e-4)
        self.assertLess(float(np.abs(result['gradient']).max()), 1e-6)

    def test_large_logits_are_finite(self):
        result = self.evaluate(scores=np.array([1000, -1000, 1000, -1000]))
        self.assertTrue(math.isfinite(result['loss']))
        self.assertTrue(np.isfinite(result['gradient']).all())

    def test_uniform_pair_ablation_uses_same_targets(self):
        result = self.evaluate(metric_weights=False)
        self.assertEqual(result['eligible_pairs'], self.evaluate()['eligible_pairs'])
        self.assertFalse(np.allclose(result['gradient'], self.evaluate()['gradient']))

    def test_missing_positive_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'all known positives'):
            self.evaluate(total_positives=3)

    def test_local_ranks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'candidate-local'):
            self.evaluate(rank_scope='candidate_pool')

    def test_invalid_inputs_are_rejected(self):
        for changes in (
            {'scores': np.array([np.nan, 0, 1, 2])},
            {'global_ranks': np.array([1, 1, 2, 3])},
            {'global_ranks': np.array([0, 1, 2, 3])},
            {'global_ranks': np.array([1, 2, 3, 233010])},
            {'global_ranks': np.array([1.0, 2.0, 3.0, 4.0])},
            {'positive_mask': np.array([1, 2, 0, 0])},
            {'judged_negative_mask': np.array([True, False, True, False])},
            {'student_temperature': 0},
            {'teacher_temperature': float('inf')},
            {'pair_floor': -1},
            {'recall_weight': float('nan')},
            {'ndcg_cutoff': True},
            {'recall_cutoff': 0},
            {'scores': np.array([1, 2])},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.evaluate(**changes)


class ConfigurationTests(unittest.TestCase):
    def test_pilot_configuration_is_consistent_and_not_a_launch_receipt(self):
        path = ROOT / 'docs/research-sae/reports/ng0001-ng0099'
        config = json.loads((path / 'ng0071-ranking-config.json').read_text())
        self.assertFalse(config['training_enabled'])
        self.assertEqual(config['status'], 'bounded_preparation_scientific_training_disabled')
        self.assertEqual(config['schema_version'], 2)
        self.assertEqual(config['pilot']['selection'],
                         'resolved_provenance_then_TRAIN_stable_hash_v2')
        pilot = config['pilot']
        optimizer = config['optimizer']
        self.assertEqual(
            pilot['train_queries'] * pilot['epochs'],
            pilot['updates'] * optimizer['queries_per_update'],
        )
        self.assertEqual(
            pilot['queries_per_domain'] * len(pilot['domains']),
            pilot['train_queries'],
        )
        witness = config['witness']
        self.assertEqual(sum(row['quota'] for row in witness['sources']), 64)
        self.assertEqual(witness['max_new_documents'], 64)
        self.assertTrue(witness['retain_all_positives'])
        self.assertTrue(witness['C_D_share_identical_manifests'])
        self.assertFalse(pilot['locked_test_access'])
        self.assertIsNone(config['base']['output_posting_cap'])
        self.assertEqual(set(config['arms']), {'A', 'B', 'C', 'D'})
        for name in ('ndcg_cutoff', 'recall_cutoff', 'recall_weight', 'pair_floor'):
            defaults = {'ndcg_cutoff': 10, 'recall_cutoff': 100,
                        'recall_weight': 1.0, 'pair_floor': 0.05}
            self.assertEqual(config['ranking'][name], defaults[name])


if __name__ == '__main__':
    unittest.main()
