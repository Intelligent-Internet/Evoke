import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng75_update_math as probe


class SharedToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(2, 3, bias=False).double()
        self.head = torch.nn.Linear(3, 2, bias=True).double()
        self.head.bias_alias = self.head.bias

    def forward(self, values):
        return self.head(self.trunk(values).tanh())


class UpdateMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(75)
        self.model = SharedToy()
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.trunk.parameters(), 'lr': .002},
            {'params': self.model.head.parameters(), 'lr': .004},
        ], weight_decay=.03)
        self.named, self.groups = probe.parameter_layout(self.model, self.optimizer)
        self.x = torch.tensor([[.4, .2], [1., -.2], [-.3, .5]], dtype=torch.float64)

    def test_shared_roles_chain_rule_and_actual_adamw_step(self):
        for _ in range(2):
            self.optimizer.zero_grad()
            self.model(self.x).square().sum().backward()
            self.optimizer.step()
        before = probe.copy_parameters(self.named)
        moments = copy.deepcopy(self.optimizer.state_dict())
        self.optimizer.zero_grad()
        q, p, n = self.model(self.x)
        loss = -(q @ (p - n))
        loss.backward()
        raw = [p.grad.detach().clone() for _, p in self.named]
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .1)
        self.optimizer.step()
        after = probe.copy_parameters(self.named)
        delta = probe.displacement(before, after)
        self.assertTrue(any(not torch.allclose(d, -g * .002) for d, g in zip(delta, raw)))
        with torch.no_grad():
            q1, p1, n1 = self.model(self.x)
            actual_after = float(q1 @ (p1 - n1))
        probe.restore_parameters(self.named, before)
        q, p, n = self.model(self.x)
        result = probe.margin_projection(q, p, n, self.named, delta, self.groups)
        self.assertLess(result['gradient_chain_max_abs'], 1e-12)
        self.assertNotEqual(sum(result['contributions']['query'].values()), 0)
        self.assertNotEqual(sum(result['contributions']['document'].values()), 0)
        # Central finite differences independently check J dot actual delta.
        eps = 1e-3
        values = []
        for sign in (-1, 1):
            probe.restore_parameters(self.named, [b + sign * eps * d for b, d in zip(before, delta)])
            with torch.no_grad():
                q, p, n = self.model(self.x)
                values.append(float(q @ (p - n)))
        self.assertAlmostEqual((values[1] - values[0]) / (2 * eps), result['predicted'], places=9)
        change = probe.describe_change(result['semantic_margin'], actual_after, result['predicted'], 0.)
        self.assertLess(abs(change['taylor_residual']), 1e-4)
        probe.restore_parameters(self.named, before)
        self.optimizer.load_state_dict(moments)
        self.assertTrue(all(torch.equal(p, b) for (_, p), b in zip(self.named, before)))

    def test_tied_alias_is_counted_once(self):
        self.assertEqual(len(self.named), 3)
        self.assertEqual(self.groups, ['trunk', 'head', 'head'])
        self.optimizer.param_groups[1]['params'].append(self.model.head.bias_alias)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            probe.parameter_layout(self.model, self.optimizer)

    def test_no_update_floor_and_deterministic_forward(self):
        self.assertTrue(torch.equal(self.model(self.x), self.model(self.x)))
        values = probe.copy_parameters(self.named)
        q, p, n = self.model(self.x)
        result = probe.margin_projection(q, p, n, self.named,
                                         probe.displacement(values, values), self.groups)
        self.assertEqual(result['predicted'], 0.)
        change = probe.describe_change(1., 1., 0., 0.)
        self.assertEqual(change['actual_direction'], 'stationary')
        self.assertEqual(change['numerical_floor'], 1e-7)
        self.assertEqual(probe.direction(1e-7, 1e-7), 'stationary')
        self.assertEqual(probe.describe_change(1., 1.00001, 0., 1e-5)['numerical_floor'], .0001)

    def test_malformed_restore_never_partially_changes_parameters(self):
        before = probe.copy_parameters(self.named)
        broken = [v + 1 for v in before]
        broken[-1] = torch.zeros(99, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'shape'):
            probe.restore_parameters(self.named, broken)
        self.assertTrue(all(torch.equal(p, v) for (_, p), v in zip(self.named, before)))

    def test_invalid_displacement_and_role_inputs_fail(self):
        for a, b in (([], []), ([torch.zeros(1)], []),
                     ([torch.zeros(1)], [torch.ones(2)]),
                     ([torch.zeros(1)], [torch.full((1,), float('nan'))])):
            with self.assertRaises(ValueError):
                probe.displacement(a, b)
        q, p, n = self.model(self.x)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            probe.margin_projection(q * float('nan'), p, n, self.named, [], [])
        with self.assertRaises(ValueError):
            probe.describe_change(0., 1., np.inf, 0.)

    def test_all_positives_and_initial_head_rivals(self):
        row = dict(split='TRAIN', gold_ids=[7, 101], top100=list(range(100)))
        self.assertEqual(probe.initial_pairs(row), [(7, 0), (7, 99), (101, 0), (101, 99)])
        self.assertEqual(probe.initial_pairs(dict(row, top100=[7, 2])), [(7, 2), (101, 2)])
        for altered in (dict(row, split='LOCKED_TEST'), dict(row, gold_ids=[7, 7]),
                        dict(row, top100=[7, 101]), dict(row, top100=[True, 4])):
            with self.assertRaises(ValueError):
                probe.initial_pairs(altered)

    def test_selection_ignores_permutation_and_final_outcome(self):
        rows = [dict(query_id=f'{d}-{i}', domain=d, split='TRAIN', surface='TRAIN_SENTINEL')
                for d in probe.DOMAINS for i in range(12)]
        wanted = [r['query_id'] for r in probe.select_sentinel(rows)]
        changed = [dict(r, final_ndcg=1. if i % 2 else 0.) for i, r in enumerate(reversed(rows))]
        self.assertEqual(wanted, [r['query_id'] for r in probe.select_sentinel(changed)])
        for bad in (rows[:-5], rows + [rows[0]], [dict(r, split='DEV_NEW') for r in rows]):
            with self.assertRaises(ValueError):
                probe.select_sentinel(bad)

    def test_summary_preserves_pair_and_query_denominators(self):
        rows = [dict(query_id=q, positive_id=p, origin=origin, domain='nq',
                     surface='TRAIN_SENTINEL', **probe.describe_change(0., change, change, 0.))
                for q, p, origin, change in (
                    ('q1', 1, 'original_positive', 1.), ('q1', 1, 'original_positive', 1.),
                    ('q2', 2, 'added_vs_original', -1.))]
        out = probe.summarize(rows)['TRAIN_SENTINEL/nq']
        self.assertEqual(out['all']['pairs'], 3)
        self.assertEqual(out['all']['queries'], 2)
        self.assertEqual(out['all']['positives'], 2)
        self.assertEqual(out['all']['query_balanced_mean_change'], 0.)
        self.assertAlmostEqual(out['all']['mean_actual_change'], 1 / 3)
        self.assertEqual(out['added_vs_original']['directions']['shrink'], 1)


if __name__ == '__main__':
    unittest.main()
