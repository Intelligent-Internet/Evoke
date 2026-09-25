import copy
from collections import Counter
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng79_breadth as breadth


class BreadthSelectionTests(unittest.TestCase):
    def setUp(self):
        self.queries = [dict(query_id=f'{d}-{i}', query=f'{d} question {i}',
                             subset=d, split='TRAIN')
                        for d in breadth.METHOD['domains'] for i in range(12)]
        self.provenance = {q['query_id']: 'resolved' for q in self.queries}
        self.old = {s: [f'{d}-{i}' for d in breadth.METHOD['domains'] for i in indices]
                    for s, indices in (('pilot', (0, 1)), ('sentinel', (2, 3)))}
        self.prefix = list(reversed(self.old['pilot']))

    def select(self, queries=None, provenance=None):
        return breadth.select(queries or self.queries, provenance or self.provenance,
                              self.old, self.prefix, per_domain=2)

    def test_equal_exposures_unique_breadth_and_domain_sequence(self):
        value = self.select()
        r, b = value['orders']['R'], value['orders']['B']
        self.assertEqual(r[:6], self.prefix)
        self.assertEqual(b[:6], self.prefix)
        self.assertEqual(Counter(r), Counter({q: 4 for q in self.old['pilot']}))
        self.assertEqual(len(b), len(set(b)))
        self.assertEqual(len(b), 24)
        self.assertEqual(set(b), set(value['breadth']))
        lookup = {q['query_id']: q for q in self.queries}
        self.assertEqual([lookup[q]['subset'] for q in r], [lookup[q]['subset'] for q in b])
        self.assertFalse(set(b).intersection(self.old['sentinel']))
        self.assertFalse(value['teacher_quality_filter'])

    def test_order_invariant_hash_selection_and_teacher_conflict_retained(self):
        rows = copy.deepcopy(self.queries)
        for row in rows:
            row['eligible_pairs'] = 0
            row['previous_ndcg'] = -999
        self.assertEqual(self.select(), self.select(list(reversed(rows))))

    def test_normalized_sentinel_text_cannot_enter_training(self):
        rows = copy.deepcopy(self.queries)
        rows.append(dict(query_id='another-id', query=' fever  question 2 ',
                         subset='fever', split='TRAIN'))
        source = {**self.provenance, 'another-id': 'resolved'}
        self.assertNotIn('another-id', self.select(rows, source)['breadth'])

    def test_nonTRAIN_and_unresolved_are_excluded_without_relabeling(self):
        rows = copy.deepcopy(self.queries)
        for split in ('DEV_NEW', 'DEV_EXPOSED', 'LOCKED_TEST'):
            rows.append(dict(query_id=split, query=split, subset='nq', split=split))
        source = {**self.provenance, **{q['query_id']: 'resolved' for q in rows[-3:]}}
        source['nq-4'] = 'unresolved'
        selected = self.select(rows, source)['breadth']
        self.assertNotIn('nq-4', selected)
        self.assertTrue(all(q not in selected for q in ('DEV_NEW', 'DEV_EXPOSED', 'LOCKED_TEST')))

    def test_duplicate_missing_and_contaminated_history_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.select(self.queries + [self.queries[0]])
        for key, value in (('split', 'DEV_NEW'), ('query', 'fever question 2')):
            rows = copy.deepcopy(self.queries)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                self.select(rows)
        self.prefix[0] = self.prefix[1]
        with self.assertRaisesRegex(ValueError, 'prefix'):
            self.select()

    def test_insufficient_breadth_cannot_silently_shrink_or_backfill(self):
        rows = [q for q in self.queries if q['subset'] != 'nq'
                or int(q['query_id'].split('-')[-1]) < 9]
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            self.select(rows)


class ConfigurationTests(unittest.TestCase):
    def test_only_breadth_schedule_and_evaluation_contract_change(self):
        path = (Path(__file__).resolve().parents[1]
                / 'docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json')
        if not path.exists():
            path = Path(__file__).resolve().parent / 'ng0071-ranking-config.json'
        core = json.loads(path.read_text())
        original = copy.deepcopy(core)
        value = breadth.config_from_parent(core)
        self.assertEqual(core, original)
        for field in ('base', 'ranking', 'optimizer', 'resources'):
            self.assertEqual(value[field], core[field])
        self.assertEqual(value['arms'], {'D': core['arms']['D']})
        self.assertEqual(value['witness']['sources'], core['witness']['sources'])
        self.assertEqual(value['witness']['refresh_updates'], [0])
        self.assertEqual(value['pilot']['train_queries'], 1536)
        self.assertEqual(value['pilot']['updates'], 384)
        self.assertFalse(value['training_enabled'])
        self.assertEqual(value['pilot']['exposed_dev_queries'], 0)
        self.assertEqual(value['gates']['B_minus_R_ndcg_point_min'], .005)
        self.assertFalse(value['gates']['automatic_scale_allowed'])


if __name__ == '__main__':
    unittest.main()
