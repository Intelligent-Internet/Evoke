import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng74_uniform_control as control
import ng71_diagnostics as diagnostics
import ng71_execution as execution
import ng71_observation as observation
import ng71_training as training


class UniformControlTests(unittest.TestCase):
    def setUp(self):
        path = (Path(__file__).resolve().parents[1]
                / 'docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json')
        self.parent = json.loads(path.read_text())
        self.parent.update(training_enabled=True, status='frozen_four_arm_pilot')
        self.config = control.configuration(self.parent)
        self.record = dict(
            query_id='q', split='TRAIN', domain='nq', pool=[0, 1, 2, 3],
            original_pool=[0, 1, 2, 3], positive_mask=[True, True, False, False],
            judged_negative_mask=[False] * 4, teacher_scores=[.2, .18, .1, .19],
            global_ranks=[1, 101, 2, 99], hybrid_scores=[1., 2., 0., 1.],
            lexical_scores=[.01] * 4, sources={}, total_positives=2,
            corpus_size=200, rank_scope='full_corpus_reference_snapshot',
            query_sha256=hashlib.sha256(b'query').hexdigest(),
            document_text_sha256=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(4)])

    def test_only_arms_and_status_change_parent_not_mutated(self):
        for key, value in self.parent.items():
            if key not in ('arms', 'status'):
                self.assertEqual(value, self.config[key])
        self.assertEqual(self.config['arms'], {'U': {'pool': 'witness', 'objective': 'uniform_soft_pair'}})
        self.assertIn('D', self.parent['arms'])
        self.config['optimizer']['trunk_lr'] = 1.
        self.assertNotEqual(self.config['optimizer']['trunk_lr'], self.parent['optimizer']['trunk_lr'])

    def test_uniform_preserves_targets_eligibility_not_all_coefficients_equal(self):
        m = training.prepare_pairs(self.record, self.config['ranking'])
        u = training.prepare_pairs(self.record, self.config['ranking'], uniform=True)
        self.assertEqual(m['indices'], u['indices'])
        self.assertEqual(m['targets'], u['targets'])
        self.assertEqual(m['supervised_positives'], u['supervised_positives'])
        self.assertEqual(m['unresolved_pairs'], u['unresolved_pairs'])
        self.assertNotEqual(m['coefficients'], u['coefficients'])
        self.assertGreater(len(set(u['coefficients'])), 1)
        self.assertTrue(u['uniform'])
        self.assertFalse(m['uniform'])

    def test_uniform_coefficient_is_confidence_over_pair_count_and_all_positives(self):
        u = training.prepare_pairs(self.record, self.config['ranking'], uniform=True)
        for (p, n), t, c in zip(u['indices'], u['targets'], u['coefficients'], strict=True):
            count = sum(pair[0] == p for pair in u['indices'])
            self.assertAlmostEqual(c, (2 * t - 1) / count / 2, places=14)

    def test_numpy_and_torch_gradient_agree_for_both_objectives(self):
        for objective in ('balanced_soft_pair', 'uniform_soft_pair'):
            result = diagnostics.objective(self.record, self.config['ranking'], objective)
            self.assertLessEqual(result['max_independent_derivative_error'], 1e-12)
            self.assertAlmostEqual(sum(result['score_gradient']), 0., places=12)

    def test_example_dispatches_uniform_without_changing_texts_or_pool(self):
        query = dict(query_id='q', split='TRAIN', query='query')
        documents = [dict(text=str(i)) for i in range(4)]
        u = execution.example(self.record, query, documents, self.config, 'U')
        m = execution.example(self.record, query, documents, self.parent, 'D')
        for key in ('query', 'documents', 'lexical'):
            self.assertEqual(u[key], m[key])
        self.assertTrue(u['pairs']['uniform'])
        self.assertEqual(u['pairs']['indices'], m['pairs']['indices'])

    def test_uniform_zero_supervision_stays_zero(self):
        record = copy.deepcopy(self.record)
        record['teacher_scores'] = [0., 0., 1., 1.]
        result = diagnostics.objective(record, self.config['ranking'], 'uniform_soft_pair')
        self.assertEqual(result['eligible_pairs'], 0)
        self.assertEqual(result['loss'], 0.)
        self.assertEqual(result['score_gradient'], [0.] * 4)

    def test_default_and_explicit_training_barriers(self):
        def sealed(path):
            _, arm, end = path.name.split('-')
            return dict(arm=arm, cumulative_steps=int(end), steps=96, query_exposures=384)
        with patch.object(execution, 'sealed', side_effect=sealed):
            self.assertEqual(len(observation.training_closed(Path('/unused'))), 8)
            result = observation.training_closed(Path('/unused'), control.TRAIN_PHASES)
            self.assertEqual(set(result), set(control.TRAIN_PHASES))
        for phases in ((), ('train-U-96', 'train-U-96')):
            with self.assertRaisesRegex(ValueError, 'unique nonempty'):
                observation.training_closed(Path('/unused'), phases)

    def test_uniform_backward_records_the_actual_objective(self):
        class Toy:
            config = {'optimizer': {'queries_per_update': 4}}

            def __init__(self):
                self.model = torch.nn.Linear(2, 2, bias=False).double()

            def encode(self, texts, role):
                return self.model(torch.tensor([[1., 2.]] * len(texts), dtype=torch.float64))

        pairs = training.prepare_pairs(self.record, self.config['ranking'], uniform=True)
        result = training.backward_query(Toy(), 'q', ['a', 'b', 'c', 'd'], [.01] * 4, pairs=pairs)
        self.assertEqual(result['objective'], 'uniform_soft_pair')
        self.assertTrue(result['vjp_replay_exact'])

    def test_no_midpoint_dev_or_lock_in_execution_graph(self):
        self.assertEqual(control.PHASES, ('train-preflight', 'train-U-96', 'train-U-192',
                                         'encode-U-192', 'rank-U-192', 'review'))
        self.assertLess(control.PHASES.index('train-U-192'), control.PHASES.index('encode-U-192'))


if __name__ == '__main__':
    unittest.main()
