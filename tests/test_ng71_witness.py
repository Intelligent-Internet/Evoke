import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ng71_witness import full_order, mine


class WitnessTests(unittest.TestCase):
    def setUp(self):
        path = (Path(__file__).resolve().parents[1]
                / 'docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json')
        self.config = json.loads(path.read_text())['witness']
        self.ids = [f'doc-{i:03d}' for i in range(150)]
        self.scores = {k: -np.arange(150, dtype=float) for k in ('hybrid', 'pplx', 'bm25')}

    def run_miner(self, **override):
        args = dict(query_id='q1', original_pool=[149, 148], positive_ids=[149, 147],
                    stable_ids=self.ids, scores=self.scores, config=self.config,
                    corpus_size=150)
        args.update(override)
        return mine(**args)

    def test_full_ranks_and_missing_positive_retained(self):
        row = self.run_miner()
        self.assertEqual(row['pool'][:3], [149, 148, 147])
        self.assertEqual(row['global_ranks'][:3], [150, 149, 148])
        self.assertEqual(sum(row['positive_mask']), 2)
        self.assertLessEqual(row['new_witness_count'], 64)
        self.assertEqual(len(set(row['pool'])), len(row['pool']))
        self.assertFalse(any(row['judged_negative_mask']))

    def test_boundary_is_nearest_100_not_start_of_window(self):
        chosen = self.run_miner()['sources']['hybrid_boundary']
        self.assertEqual(chosen[:5], [99, 98, 100, 97, 101])

    def test_source_shortage_does_not_refill_elsewhere(self):
        chosen = self.run_miner()['sources']
        self.assertEqual(len(chosen['hybrid_head']), 24)
        self.assertEqual(len(chosen['pplx_head']), 8)
        self.assertEqual(len(chosen['bm25_head']), 0)
        self.assertEqual(len(chosen['remaining_hash_sample']), 8)

    def test_positives_never_become_negative_witnesses(self):
        row = self.run_miner(positive_ids=[0, 99, 149])
        for ids in row['sources'].values():
            self.assertFalse(set(ids) & {0, 99, 149})

    def test_ties_use_stable_ids_not_input_positions(self):
        order, ranks = full_order(np.zeros(3), ['z', 'a', 'm'], 3)
        self.assertEqual(order.tolist(), [1, 2, 0])
        self.assertEqual(ranks.tolist(), [3, 1, 2])

    def test_local_score_vector_rejected(self):
        bad = copy.deepcopy(self.scores)
        bad['hybrid'] = bad['hybrid'][:80]
        with self.assertRaisesRegex(ValueError, 'complete'):
            self.run_miner(scores=bad)

    def test_overflow_fails_without_truncating(self):
        bad = dict(self.config, max_total_documents=5)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            self.run_miner(config=bad)

    def test_repeatability_and_nonfinite_rejection(self):
        self.assertEqual(self.run_miner(), self.run_miner())
        bad = copy.deepcopy(self.scores)
        bad['pplx'][13] = np.nan
        with self.assertRaises(ValueError):
            self.run_miner(scores=bad)


if __name__ == '__main__':
    unittest.main()
