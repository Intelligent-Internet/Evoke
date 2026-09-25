from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_ng71_snapshot as reviewer


class IndependentReviewTests(unittest.TestCase):
    def test_cross_host_rounding_is_bounded_without_weakening_identities(self):
        a = {'score': [1., 2.], 'query': 'a'}
        b = {'score': [1. + 1e-14, 2.], 'query': 'a'}
        self.assertLess(reviewer.compare(a, b), 1e-12)
        with self.assertRaises(ValueError):
            reviewer.compare(a, {**b, 'query': 'b'})

    def test_bad_numbers_vector_lengths_and_fields_are_rejected(self):
        for a, b in (([1.], [1., 2.]), ({'a': 1}, {'b': 1}),
                     (1., 1. + 1e-10), (float('nan'), float('nan'))):
            with self.assertRaises(ValueError):
                reviewer.compare(a, b)

    def test_cli_cannot_write_review_inside_frozen_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ['review', '--research-root', str(root), '--snapshot', str(root),
                    '--expected-complete', 'unused', '--output', str(root / 'result.json')]
            with patch.object(sys, 'argv', argv), patch.object(reviewer, 'review') as run:
                with self.assertRaisesRegex(ValueError, 'outside frozen evidence'):
                    reviewer.main()
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
