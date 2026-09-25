"""Do not infer human correctness from source lineage or unresolved text matches."""

import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / 'scripts/audit_ng70_provenance.py'
SPEC = importlib.util.spec_from_file_location('ng70', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def query(*texts):
    return {'query_id': 'q', 'subset': 'nq', 'candidates': [
        {'key': str(i), 'text': text, 'label': 'positive'} for i, text in enumerate(texts)]}


class ProvenanceTests(unittest.TestCase):
    def test_reused_train_prefix_requires_exact_content(self):
        records = {}
        row = query('original')
        audit.merge_train(records, row)
        audit.merge_train(records, dict(row))
        self.assertEqual(len(records), 1)
        with self.assertRaises(AssertionError):
            audit.merge_train(records, query('changed'))

    def test_original_and_added_are_lineage_not_truth_labels(self):
        result = audit.classify(query('original', 'additional'), [{'pos': ['original']}])
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual([p['provenance'] for p in result['pairs']],
                         ['original_positive', 'added_vs_original'])

    def test_missing_original_positive_prevents_added_label_inference(self):
        result = audit.classify(query('additional'), [{'pos': ['missing']}])
        self.assertEqual(result['status'], 'original_positive_missing')
        self.assertEqual(result['pairs'][0]['provenance'], 'unresolved')

    def test_conflicting_original_rows_stay_ambiguous(self):
        result = audit.classify(query('A', 'B'), [{'pos': ['A']}, {'pos': ['B']}])
        self.assertEqual(result['status'], 'ambiguous_original_query')
        self.assertTrue(all(p['provenance'] == 'unresolved' for p in result['pairs']))

    def test_identical_duplicate_original_rows_are_resolvable(self):
        result = audit.classify(query('A'), [{'pos': ['A']}, {'pos': ['A']}])
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual(result['upstream_rows'], 2)

    def test_unmatched_query_and_empty_original_remain_unresolved(self):
        for originals, status in (([], 'query_not_found'), ([{'pos': []}], 'original_positive_missing')):
            result = audit.classify(query('A'), originals)
            self.assertEqual(result['status'], status)
            self.assertEqual(result['pairs'][0]['provenance'], 'unresolved')

    def test_case_and_punctuation_are_not_collapsed(self):
        self.assertNotEqual(audit.normalized('US'), audit.normalized('us'))
        self.assertNotEqual(audit.normalized('A-B'), audit.normalized('A B'))
        self.assertEqual(audit.normalized('A\n B'), 'A B')

    def test_equivalent_text_never_drops_distinct_document_identities(self):
        result = audit.classify(query('A B', 'A\nB'), [{'pos': ['A B']}])
        self.assertEqual(len(result['pairs']), 2)
        self.assertTrue(all(p['provenance'] == 'original_positive' for p in result['pairs']))


if __name__ == '__main__':
    unittest.main()
