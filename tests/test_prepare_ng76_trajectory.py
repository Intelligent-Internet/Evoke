from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_ng76_trajectory as prep


class EndpointUnionTests(unittest.TestCase):
    def test_union_preserves_endpoint_cutoffs_not_full_tail_rank(self):
        a = dict(query_id='q', split='TRAIN', gold_ids=[0, 300], gold_ranks=[1, 301], top100=list(range(100)))
        b = dict(a, top100=list(range(50, 150)), gold_ranks=[151, 301])
        pool = prep.endpoint_pool(a, b)
        self.assertEqual(pool, list(range(150)) + [300])
        scores = -np.asarray(pool, dtype=float)
        ranks = prep.endpoint_rank(pool, scores, a)
        self.assertEqual(ranks, [1, 151])
        with self.assertRaises(ValueError):
            prep.endpoint_rank(pool, scores[::-1], a)
        with self.assertRaises(ValueError):
            prep.endpoint_pool(a, dict(b, split='DEV_NEW'))

    def test_partial_head_cannot_claim_endpoint_parity(self):
        row = dict(query_id='q', split='TRAIN', gold_ids=[0], top100=list(range(99)))
        with self.assertRaises(ValueError):
            prep.endpoint_pool(row, row)

    def test_fixed_easy_rivals_can_miss_actual_lost_comparisons(self):
        initial = {'q': dict(domain='nq', ndcg10=1.)}
        final = {'q': dict(domain='nq', ndcg10=.5)}
        probes = [dict(query_id='q', positive_id=0, rival_id=99)]
        records = [dict(query_id='q', positive_id=0, competitor_ids=[1, 99],
                        baseline_ahead=[True, True], final_ahead=[False, True])]
        result = prep.coverage(probes, records, initial, final)['nq']
        self.assertEqual(result['full_lost'], 1)
        self.assertEqual(result['fixed_lost'], 0)
        self.assertEqual(result['ndcg_mean_change'], -.5)


if __name__ == '__main__':
    unittest.main()
