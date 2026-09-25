import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_data


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = {'pilot': {'queries_per_domain': 2,
                                'preflight_train_queries': 3,
                                'domains': ['fever', 'hotpotqa', 'nq']}}
        self.rows = [dict(query_id=f'{domain}-{i}', subset=domain,
                          query=f'{domain} question {i}')
                     for domain in self.config['pilot']['domains']
                     for i in range(8)]

    def test_selection_is_order_independent_and_disjoint(self):
        first = ng71_data.select(self.rows, self.config)
        self.assertEqual(first, ng71_data.select(self.rows[::-1], self.config))
        self.assertEqual(len(first['pilot']), 6)
        self.assertEqual(len(first['sentinel']), 6)
        self.assertFalse(set(first['pilot']) & set(first['sentinel']))
        self.assertLessEqual(set(first['canary']), set(first['pilot']))

    def test_duplicate_identity_fails_closed(self):
        with self.assertRaises(ValueError):
            ng71_data.select(self.rows + [self.rows[0]], self.config)

    def test_unicode_alias_does_not_permit_fuzzy_or_whitespace_drift(self):
        self.assertTrue(ng71_data.same_source_text('F\u00fatbol', 'Fu\u0301tbol'))
        self.assertFalse(ng71_data.same_source_text('Title  body', 'Title body'))
        self.assertFalse(ng71_data.same_source_text('Title', 'title'))

    def test_duplicate_text_is_not_a_disjoint_sentinel(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row['query'] = 'same text'
        with self.assertRaises(ValueError):
            ng71_data.select(rows, self.config)

    def test_checked_in_pilot_shapes_match_selection_contract(self):
        path = (Path(__file__).resolve().parents[1]
                / 'docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json')
        config = json.loads(path.read_text())['pilot']
        self.assertEqual(config['train_queries'], config['queries_per_domain'] * 3)
        self.assertEqual(config['sentinel_train_queries'], config['train_queries'])
        self.assertEqual(config['preflight_train_queries'] % 3, 0)


if __name__ == '__main__':
    unittest.main()
