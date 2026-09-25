import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng73_gradient_replay as replay


class GradientReplayTests(unittest.TestCase):
    def setUp(self):
        path = (Path(__file__).resolve().parents[1]
                / 'docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json')
        self.config = json.loads(path.read_text())['ranking']
        self.record = dict(
            query_id='q', split='TRAIN', domain='nq', pool=[1, 2, 3],
            original_pool=[1, 2, 3], positive_mask=[True, False, False],
            judged_negative_mask=[False] * 3, teacher_scores=[.2, .199, .3],
            global_ranks=[1, 11, 101], hybrid_scores=[10., 0., 0.],
            lexical_scores=[0.] * 3, sources={}, total_positives=1,
            corpus_size=200, rank_scope='full_corpus_reference_snapshot')
        self.detail = replay.diagnostics.objective(self.record, self.config, 'balanced_soft_pair')
        self.example = dict(query_id='q', scores=self.record['hybrid_scores'],
                            score_gradient=(np.asarray(self.detail['score_gradient']) / 4).tolist(),
                            loss=self.detail['loss'], documents=3,
                            objective='balanced_soft_pair', vjp_replay_exact=True,
                            eligible_pairs=1, supervised_positives=1)
        self.margin = dict(query_id='q', surface='TRAIN_PILOT', positive_id=1,
                           origin='original_positive', competitor_ids=[2, 3, 4],
                           before_margin=[10.] * 3, after_margin=[-1., 1., -1.],
                           baseline_ahead=[True] * 3, final_ahead=[False, True, False])
        self.head = dict(gold_ids=[1], gold_ranks=[1], top100=[1, 2])

    def pair(self):
        return replay.pair_rows(self.record, self.example, self.detail,
                                [self.margin], self.head, 1, 0)[0]

    def test_actual_gradient_normalization_and_loss_replay(self):
        result = replay.replay_example(self.record, self.example, self.config)
        self.assertGreater(result['pair_score_derivatives'][0], 0.)
        changed = copy.deepcopy(self.example)
        changed['score_gradient'][0] += .1
        with self.assertRaises(AssertionError):
            replay.replay_example(self.record, changed, self.config)

    def test_direction_not_confused_with_teacher_agreement(self):
        row = self.pair()
        self.assertEqual(row['state'], ['shrink', 'ineligible', 'absent'])
        self.assertEqual(row['outcome'], ['lost', 'retained', 'lost'])
        self.assertGreater(row['target'][0], .5)
        self.assertIsNone(row['own_derivative'][1])
        self.assertIsNotNone(row['aggregate_margin_pressure'][1])
        self.assertIsNone(row['aggregate_margin_pressure'][2])

    def test_insufficient_teacher_margin_expands_and_hard_target_never_shrinks(self):
        for scores, negative in (([-1., 0., 0.], False), ([10., 0., 0.], True)):
            row = copy.deepcopy(self.record)
            row['hybrid_scores'] = scores
            row['judged_negative_mask'][1] = negative
            result = replay.diagnostics.objective(row, self.config, 'balanced_soft_pair')
            self.assertLess(result['pair_score_derivatives'][0], 0.)

    def test_baseline_boundaries_not_current_candidate_ranks(self):
        row = self.pair()
        self.assertEqual(row['cross_top100'], [False, True, True])
        self.assertEqual(row['cross_top10'], [False, True, True])

    def test_all_supervised_pairs_must_be_in_diagnostic_union(self):
        self.margin['competitor_ids'][0] = 5
        with self.assertRaisesRegex(ValueError, 'misses an actual'):
            self.pair()

    def test_gold_rival_and_missing_positive_fail_closed(self):
        self.margin['competitor_ids'][0] = 1
        with self.assertRaisesRegex(ValueError, 'invalid rivals'):
            self.pair()
        with self.assertRaisesRegex(ValueError, 'positive coverage'):
            replay.pair_rows(self.record, self.example, self.detail, [], self.head, 1, 0)

    def test_sentinel_dev_and_foreign_query_fail_closed(self):
        for field, value in (('split', 'DEV'), ('query_id', 'other')):
            record = dict(self.record, **{field: value})
            with self.assertRaises(ValueError):
                replay.replay_example(record, self.example, self.config)
        self.margin['surface'] = 'TRAIN_SENTINEL'
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            self.pair()

    def test_full_rival_denominator_and_positive_query_balance(self):
        row = self.pair()
        duplicate = copy.deepcopy(row)
        duplicate['step'] = 2
        result = replay.summarize([row, duplicate])['nq']['all']['groups']['lost']
        self.assertEqual(result['pair_exposures'], 4)
        self.assertAlmostEqual(result['full_rival_query_balanced']['shrink'], 1 / 3)
        self.assertAlmostEqual(result['full_rival_query_balanced']['absent'], 1 / 3)
        self.assertEqual(result['state_counts']['ineligible'], 0)

    def test_outcome_partition_and_stationary_tolerance(self):
        self.assertEqual({replay.outcome(a, b) for a in (False, True)
                          for b in (False, True)}, set(replay.GROUPS))
        self.assertEqual(replay.direction(1e-13), 'stationary')
        self.assertEqual(replay.direction(-1e-3), 'expand')
        with self.assertRaises(ValueError):
            replay.direction(float('nan'))

    def test_started_receipt_failure_still_closes_owned_worker(self):
        process = SimpleNamespace(pid=12345, returncode=-15, wait=Mock(return_value=-15))
        original_write = replay.write

        def failing_start(path, value):
            if path.name == 'started.json':
                raise OSError('receipt unavailable')
            original_write(path, value)

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with (patch.object(replay, 'verify'),
                  patch.object(replay, 'write', side_effect=failing_start),
                  patch.object(replay.subprocess, 'Popen', return_value=process),
                  patch.object(replay.os, 'killpg', side_effect=ProcessLookupError) as kill):
                with self.assertRaisesRegex(ValueError, 'replay failed'):
                    replay.supervise(run, run)
            process.wait.assert_called_once_with(timeout=10)
            self.assertEqual(kill.call_count, 2)
            self.assertFalse(replay.read(run / 'audit/complete.json')['passed'])
            self.assertIn('receipt unavailable', replay.read(run / 'audit/exit.json')['error'])


if __name__ == '__main__':
    unittest.main()
