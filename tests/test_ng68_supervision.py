"""Fixtures for supervision auditing without consuming held-out records."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / 'scripts/analyze_ng68_supervision.py'
SPEC = importlib.util.spec_from_file_location('ng68_review', PATH)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def target(scores, positive):
    values = np.asarray(scores)
    probs = np.exp((values - values.max()) / .04)
    probs /= probs.sum()
    return {'scores': scores, 'positive_mask': positive,
            'pool': list(range(len(scores))),
            'target': (.5 * np.asarray(positive) / sum(positive)
                       + .5 * probs).tolist()}


class SupervisionTests(unittest.TestCase):
    def test_single_positive_cannot_be_overruled_by_teacher(self):
        row = target([.1, .8, .2], [True, False, False])
        stats = review.target_stats(row)
        self.assertTrue(stats['teacher_top_is_source_negative'])
        self.assertFalse(stats['target_any_positive_below_source_negative'])

    def test_multi_positive_can_be_overruled(self):
        row = target([.1, .2, .8], [True, True, False])
        stats = review.target_stats(row)
        self.assertEqual(stats['target_discordant_pairs'], 2)
        self.assertLess(stats['target_min_positive_log_gap'], 0)

    def test_wrong_target_rejected(self):
        row = target([.8, .1], [True, False])
        row['target'] = [.5, .5]
        with self.assertRaises(ValueError):
            review.target_stats(row)

    def test_no_source_negative_rejected(self):
        with self.assertRaises(ValueError):
            review.target_stats(target([.8, .7], [True, True]))

    def test_witnesses_not_all_top10_nonpositives(self):
        rank = {'gold_ids': [1], 'gold_ranks': [2],
                'top100': list(range(100))}
        stats = review.witnesses(rank, {1})
        self.assertEqual(stats, {'ids': [0], 'count': 1, 'outside_pool': 1})

    def test_partial_strata_do_not_get_synthetic_replacements(self):
        rows = [{'domain': d, 'stratum': 'control', 'query_id': d}
                for d in review.DOMAINS]
        selected, cells = review.select_review(rows)
        self.assertEqual(len(selected), 3)
        self.assertEqual(cells['nq/teacher_conflict']['selected'], 0)

    def test_only_train_prefix_is_parsed(self):
        row = {'query_id': 'q', 'domain': 'nq', 'split': 'TRAIN'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rank.jsonl'
            path.write_text(json.dumps(row) + '\nNOT_JSON_HELD_OUT\n')
            self.assertEqual(review.train_prefix(
                path, [{'query_id': 'q', 'subset': 'nq'}]), [row])

    def test_dev_prefix_is_rejected(self):
        row = {'query_id': 'q', 'domain': 'nq', 'split': 'DEV_EXPOSED'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rank.jsonl'
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                review.train_prefix(path, [{'query_id': 'q', 'subset': 'nq'}])

    def test_packet_blinding_and_all_positives(self):
        selected = [{'query_index': 0, 'query_id': 'q', 'stratum': 'control',
                     'domain': 'nq'}]
        docs = [{'key': str(i), 'text': str(i)} for i in range(12)]
        rank = [{'top100': list(range(12)), 'gold_ids': [9, 10, 11]}]
        packet, key = review.blind_packet(
            selected, [{'query': 'question'}], docs, rank, rank, rank)
        self.assertEqual(len(packet[0]['cards']), 6)
        self.assertEqual(sum(c['source_positive'] for c in key[0]['cards']), 3)
        text = json.dumps(packet)
        for hidden in ('source_positive', 'domain', 'stratum', 'score',
                       'document_index', 'query_id'):
            self.assertNotIn(hidden, text)


if __name__ == '__main__':
    unittest.main()
