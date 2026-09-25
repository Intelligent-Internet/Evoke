"""Selection fixtures for a metric-blind, matched-exposure breadth study."""

from collections import Counter
import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / 'scripts/prepare_ng69_breadth.py'
SPEC = importlib.util.spec_from_file_location('ng69_prepare', PATH)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def cohort():
    return [{'query_id': f'{group}-{domain}-{i}', 'subset': domain}
            for group, count in (('old', 512), ('new', 4096))
            for domain in prepare.DOMAINS for i in range(count)]


class BreadthTests(unittest.TestCase):
    def test_old_prefix_and_balanced_unique_queries(self):
        train = cohort()
        indices = prepare.select_indices(train)
        self.assertEqual(indices[:1536], list(range(1536)))
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(Counter(train[i]['subset'] for i in indices),
                         dict.fromkeys(prepare.DOMAINS, 2048))

    def test_new_row_permutation_does_not_change_selected_queries(self):
        train = cohort()
        permuted = train[:1536] + train[1536:][::-1]
        identities = lambda data: [data[i]['query_id']
                                   for i in prepare.select_indices(data)]
        self.assertEqual(identities(train), identities(permuted))

    def test_duplicate_query_rejected(self):
        train = cohort()
        train[-1]['query_id'] = train[0]['query_id']
        with self.assertRaises(ValueError):
            prepare.select_indices(train)

    def test_wrong_domain_quota_rejected(self):
        train = cohort()
        train[-1]['subset'] = 'fever'
        with self.assertRaises(ValueError):
            prepare.select_indices(train)

    def test_missing_query_rejected(self):
        with self.assertRaises(ValueError):
            prepare.select_indices(cohort()[:-1])


if __name__ == '__main__':
    unittest.main()
