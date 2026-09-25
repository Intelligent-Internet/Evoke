import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_execution as execution
import ng71_observation as observation
import ng71_pilot as pipeline
import review_ng71_pilot as review


class PilotGraphTests(unittest.TestCase):
    def test_all_training_seals_precede_any_terminal_quality(self):
        graph = pipeline.schedule()
        self.assertEqual(len(graph), len(set(graph)))
        last_train = max(graph.index(p) for p in observation.TRAIN_PHASES)
        self.assertLess(graph.index('train-A-96'), graph.index('encode-A-96'))
        self.assertLess(graph.index('encode-A-96'), graph.index('snapshot-96'))
        self.assertLess(graph.index('snapshot-96'), graph.index('train-A-192'))
        self.assertTrue(all(graph.index(p) > last_train for p in graph
                            if p.startswith('rank-') or p.endswith('-192') and p.startswith('encode-')))
        self.assertEqual(graph[-1], 'review')
        self.assertEqual(len(observation.TRAIN_PHASES), 8)

    def test_terminal_barrier_rejects_wrong_count_and_missing_phase(self):
        def result(path):
            _, arm, step = path.name.split('-')
            return dict(arm=arm, cumulative_steps=int(step), steps=96, query_exposures=384)
        with patch.object(execution, 'sealed', side_effect=result):
            self.assertEqual(len(observation.training_closed(Path('/run'))), 8)
        with patch.object(execution, 'sealed', side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                observation.training_closed(Path('/run'))
        with patch.object(execution, 'sealed', return_value=dict(
                arm='A', cumulative_steps=96, steps=96, query_exposures=383)):
            with self.assertRaises(ValueError):
                observation.training_closed(Path('/run'))

    def test_surface_only_exposes_dev_after_terminal_and_never_locked(self):
        rows = [dict(query_id=str(i), query=f'q{i}', split='TRAIN' if i < 6144 else
                     'DEV_NEW' if i < 7680 else 'DEV_EXPOSED') for i in range(7872)]
        selection = dict(pilot=[str(i) for i in range(384)],
                         sentinel=[str(i) for i in range(384, 768)])
        with patch.object(execution, 'read', return_value=rows):
            self.assertEqual(len(observation.surface_queries(Path('/base'), selection, False)), 384)
            result = observation.surface_queries(Path('/base'), selection, True)
            self.assertEqual(len(result), 2304)
            self.assertNotIn('DEV_EXPOSED', {r['split'] for r in result})
            self.assertEqual(sum(r['split'] == 'DEV_NEW' for r in result), 1536)
            rows[0]['split'] = 'LOCKED_TEST'
            with self.assertRaises(ValueError):
                observation.surface_queries(Path('/base'), selection, False)

    def test_overlap_or_duplicate_identity_is_rejected(self):
        rows = [dict(query_id=str(i), split='TRAIN') for i in range(7872)]
        selection = dict(pilot=['0'] * 384, sentinel=[])
        with patch.object(execution, 'read', return_value=rows):
            with self.assertRaises(ValueError):
                observation.surface_queries(Path('/base'), selection, False)

    def test_terminate_owns_group_and_waits_for_real_exit(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                   start_new_session=True)
        self.assertTrue(pipeline.terminate_group(process))
        self.assertIsNotNone(process.poll())

    def test_started_receipt_failure_still_terminates_owned_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            args = SimpleNamespace(run=run, research_root=run, gpu=0)
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     start_new_session=True)
            original = pipeline.io.write

            def write(path, value):
                if path.name == 'started.json':
                    raise OSError('receipt fixture failure')
                original(path, value)

            with patch.object(pipeline.subprocess, 'Popen', return_value=child), \
                    patch.object(pipeline.psutil, 'virtual_memory',
                                 return_value=SimpleNamespace(available=100 * 1024 ** 3)), \
                    patch.object(pipeline.shutil, 'disk_usage',
                                 return_value=SimpleNamespace(free=100 * 1024 ** 3)), \
                    patch.object(pipeline.io, 'write', side_effect=write):
                with self.assertRaisesRegex(ValueError, 'preserve attempt'):
                    pipeline.phase(args, 'review', [])
            self.assertIsNotNone(child.poll())
            self.assertTrue(pipeline.read(run / 'review/exit.json')['owned_group_closed'])
            self.assertFalse(pipeline.read(run / 'review/complete.json')['passed'])


class ObservationTests(unittest.TestCase):
    def test_full_ties_use_numeric_ids_and_all_positives(self):
        score = np.zeros(200, dtype=np.float64)
        gold = [0, 9, 99, 100, 199]
        row, ranks, error = observation.rank_record(score, score.copy(), gold)
        self.assertEqual(row['gold_ranks'], [1, 10, 100, 101, 200])
        self.assertEqual(row['top100'], list(range(100)))
        self.assertEqual(row['recall100'], .6)
        self.assertEqual(error, 0.)
        review.audit.audit_row(row, gold, 200)

    def test_incomplete_or_nonfinite_scores_cannot_produce_quality(self):
        scores = np.zeros(200, dtype=np.float64)
        for other in (scores[:-1], np.full(200, np.nan), scores + .01):
            with self.assertRaises(ValueError):
                observation.rank_record(scores, other, [0])
        with self.assertRaises(ValueError):
            observation.rank_record(scores, scores, [0, 0])

    def test_df_cost_is_actual_document_support_not_sum_of_weights(self):
        d = sparse.csr_matrix(np.array([[1, 0, 2], [3, 0, 4], [0, 5, 0]], dtype=np.float32))
        q = sparse.csr_matrix(np.array([[1, 0, 1], [0, 9, 0]], dtype=np.float32))
        result = observation.sparse_counts(q, d, [{'surface': 'TRAIN_PILOT'}, {'surface': 'DEV_NEW'}])
        self.assertEqual(result['document_nnz'], 5)
        self.assertEqual(result['query_df_by_surface']['TRAIN_PILOT']['mean'], 4.)
        self.assertEqual(result['query_df_by_surface']['DEV_NEW']['mean'], 1.)
        self.assertFalse(result['native_cost_evaluated'])


class PairedReviewerTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = json.loads((root / 'docs/research-sae/reports/ng0001-ng0099'
                                 / 'ng0071-ranking-config.json').read_text())

    def test_bootstrap_is_paired_stratified_and_reproducible(self):
        domains = ['fever'] * 2 + ['hotpotqa'] * 3 + ['nq'] * 4
        delta = np.array([[.01, .002]] * len(domains))
        result = review.paired(delta, domains)
        self.assertAlmostEqual(result['macro']['ndcg10'], .01)
        np.testing.assert_allclose(result['ci95']['ndcg10'], [.01, .01])
        self.assertEqual(result, review.paired(delta, domains))
        self.assertTrue(result['conditional_on_single_initialization_and_order'])

    def test_quality_harm_or_cost_inflation_blocks_automatic_scale(self):
        comparison = review.paired(np.array([[.01, 0.]] * 3), list(review.audit.DOMAINS))
        gate = review.advancement(comparison, 1., 1., self.config['gates'])
        self.assertTrue(gate['quality_passed'])
        self.assertTrue(gate['uniform_pair_ablation_required'])
        self.assertFalse(gate['automatic_scale_authorized'])
        self.assertFalse(gate['overall_goal_qualified'])
        cost = review.advancement(comparison, 1.251, 1., self.config['gates'])
        self.assertFalse(cost['relative_cost_proxy_passed'])
        bad = copy.deepcopy(comparison)
        bad['domains']['nq']['recall100'] = -.006
        self.assertFalse(review.advancement(bad, 1., 1., self.config['gates'])['quality_passed'])

    def test_new_files_in_sealed_phase_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            payload = {'results.json': {'value': 1},
                       'exit.json': dict(exit_code=0, error=None, owned_group_closed=True),
                       'clearml.json': dict(closed=True, actual_start=True)}
            for name, value in payload.items():
                pipeline.io.write(path / name, value)
            pipeline.io.write(path / 'complete.json', dict(
                passed=True, files={p.name: pipeline.io.sha(p) for p in path.iterdir()}))
            self.assertEqual(execution.sealed(path)['value'], 1)
            pipeline.io.write(path / 'unexpected.json', {})
            with self.assertRaisesRegex(ValueError, 'inventory'):
                execution.sealed(path)


if __name__ == '__main__':
    unittest.main()
