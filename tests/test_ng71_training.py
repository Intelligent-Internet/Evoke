"""Autograd and shared-encoder VJP parity, with no research-data access."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from ng71_training import backward_query, legacy_loss, optimizer_step, pair_loss, prepare_pairs
from ng71_ranking import loss_and_gradient


CONFIG = {'ndcg_cutoff': 10, 'recall_cutoff': 100, 'recall_weight': 1.,
          'pair_floor': .05, 'student_temperature': 1., 'teacher_temperature': .04}


def record():
    return {'positive_mask': [True, False, True, False],
            'judged_negative_mask': [False, True, False, False],
            'global_ranks': [94, 100, 120, 3],
            'teacher_scores': [.5, .3, .4, .45],
            'total_positives': 2, 'corpus_size': 233009,
            'rank_scope': 'full_corpus_reference_snapshot'}


@unittest.skipIf(torch is None, 'PyTorch is required for NG71 training parity')
class TorchTests(unittest.TestCase):
    def test_autograd_matches_independent_numpy_reference(self):
        row = record()
        for uniform in (False, True):
            scores = torch.tensor([.5, .4, .2, .7], dtype=torch.float64,
                                  requires_grad=True)
            pairs = prepare_pairs(row, CONFIG, uniform=uniform)
            loss = pair_loss(scores, pairs)
            loss.backward()
            expected = loss_and_gradient(
                scores.detach().numpy(), np.array(row['positive_mask']),
                np.array(row['global_ranks']), np.array(row['teacher_scores']),
                np.array(row['judged_negative_mask']),
                total_positives=2, corpus_size=233009,
                rank_scope=row['rank_scope'], metric_weights=not uniform, **CONFIG)
            self.assertAlmostEqual(loss.item(), expected['loss'], places=12)
            np.testing.assert_allclose(scores.grad.numpy(), expected['gradient'],
                                       rtol=1e-12, atol=1e-12)

    def test_empty_objective_has_explicit_zero_gradient(self):
        row = record()
        row['teacher_scores'] = [0.] * 4
        row['judged_negative_mask'] = [False] * 4
        scores = torch.randn(4, requires_grad=True)
        loss = pair_loss(scores, prepare_pairs(row, CONFIG))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.equal(scores.grad, torch.zeros(4)))

    def test_malformed_manifest_rejected(self):
        pairs = prepare_pairs(record(), CONFIG)
        for field, value in (
            ('indices', [[0, 0]]), ('targets', [float('nan')]),
            ('coefficients', [-1.]), ('student_temperature', 0),
        ):
            bad = copy.deepcopy(pairs)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                pair_loss(torch.ones(4), bad)

    def test_non_boolean_mask_rejected(self):
        row = record()
        row['positive_mask'] = [1, 0, 1, 0]
        with self.assertRaises(ValueError):
            prepare_pairs(row, CONFIG)

    def test_empty_indices_cannot_hide_bad_pair_data(self):
        pairs = prepare_pairs(record(), CONFIG)
        pairs['indices'] = []
        with self.assertRaisesRegex(ValueError, 'empty indices'):
            pair_loss(torch.ones(4), pairs)

    def test_fractional_pair_indices_rejected(self):
        pairs = prepare_pairs(record(), CONFIG)
        pairs['indices'][0][0] = .5
        with self.assertRaisesRegex(ValueError, 'integers'):
            pair_loss(torch.ones(4), pairs)

    def encoder(self):
        class Toy:
            def __init__(self):
                self.model = torch.nn.Linear(3, 5, dtype=torch.float64)
                with torch.no_grad():
                    self.model.weight.fill_(.2)
                    self.model.bias.copy_(torch.arange(1., 6.))
                self.config = {'optimizer': {'queries_per_update': 4,
                                             'gradient_clip_norm': 1.}}

            def encode(self, texts, role):
                values = self.model(torch.tensor(texts, dtype=torch.float64)).relu().log1p()
                return .9 * values / values.sum(1, keepdim=True) if role == 'query' else values

        return Toy()

    def example(self):
        return {'query': [1., 2., 3.],
                'documents': [[0., 1., 1.], [1., 0., 2.], [2., 3., 4.], [2., 1., 0.]],
                'lexical': [.2, .1, .3, .5], 'pairs': prepare_pairs(record(), CONFIG)}

    def test_shared_encoder_replay_matches_direct_autograd(self):
        direct, replay = self.encoder(), self.encoder()
        left = backward_query(direct, **self.example(), replay=False)
        right = backward_query(replay, **self.example(), replay=True)
        self.assertEqual(left['scores'], right['scores'])
        self.assertEqual(left['score_gradient'], right['score_gradient'])
        for a, b in zip(direct.model.parameters(), replay.model.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)

    def test_legacy_ce_retains_original_formula_and_vjp(self):
        direct, replay = self.encoder(), self.encoder()
        example = self.example()
        example.pop('pairs')
        example['target'] = [.1, .2, .6, .1]
        result = backward_query(direct, **example, replay=False)
        other = backward_query(replay, **example)
        scores = torch.tensor(result['scores'], dtype=torch.float64)
        expected = -(torch.tensor(example['target'], dtype=torch.float64)
                     * scores.log_softmax(-1)).sum()
        self.assertEqual(result['loss'], expected.item())
        self.assertEqual(result['loss'], other['loss'])
        for a, b in zip(direct.model.parameters(), replay.model.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)

    def test_invalid_ce_target_rejected(self):
        for target in ([0., 0.], [-1., 2.], [1.], [float('nan'), 0.]):
            with self.assertRaises(ValueError):
                legacy_loss(torch.ones(2), target)

    def test_mixed_objective_for_one_query_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            backward_query(self.encoder(), **self.example(), target=[.25] * 4)

    def test_accumulation_preserves_shared_encoder_gradient(self):
        once, four = self.encoder(), self.encoder()
        backward_query(once, **self.example(), accumulation=1)
        for _ in range(4):
            backward_query(four, **self.example(), accumulation=4)
        for a, b in zip(once.model.parameters(), four.model.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)

    def test_optimizer_changes_parameters(self):
        encoder = self.encoder()
        optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=1e-4)
        result = optimizer_step(encoder, optimizer, [self.example() for _ in range(4)])
        self.assertGreater(result['update_l2'], 0)
        self.assertTrue(all(r['vjp_replay_exact'] for r in result['examples']))

    def test_all_unsupervised_batch_does_not_apply_weight_decay(self):
        encoder = self.encoder()
        optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=1e-4)
        before = [p.detach().clone() for p in encoder.model.parameters()]
        example = self.example()
        example['pairs']['indices'] = []
        with self.assertRaisesRegex(ValueError, 'no eligible'):
            optimizer_step(encoder, optimizer, [example] * 4)
        for a, b in zip(before, encoder.model.parameters()):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
