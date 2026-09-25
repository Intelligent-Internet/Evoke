import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_execution as execution


class ExecutionBoundaryTests(unittest.TestCase):
    def setUp(self):
        path = (Path(__file__).resolve().parents[1]
                / 'docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json')
        if not path.exists():
            path = Path(__file__).resolve().parent / 'ng0071-ranking-config.json'
        self.config = json.loads(path.read_text())
        self.documents = [{'text': 'positive'}, {'text': 'original'}, {'text': 'witness'}]
        digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
        self.query = dict(query_id='q', query='question', split='TRAIN')
        self.record = dict(
            query_id='q', query_sha256=digest('question'),
            pool=[0, 1, 2], original_pool=[0, 1],
            document_text_sha256=[digest(r['text']) for r in self.documents],
            positive_mask=[True, False, False], judged_negative_mask=[False] * 3,
            total_positives=1, teacher_scores=[.5, .2, .1],
            lexical_scores=[.1, .2, .3], hybrid_scores=[1., 0., 2.],
            global_ranks=[20, 100, 5], corpus_size=200,
            rank_scope='full_corpus_reference_snapshot')

    def test_examples_forward_actual_texts_not_cached_snapshot_codes(self):
        row = execution.example(self.record, self.query, self.documents, self.config, 'D')
        self.assertEqual(row['documents'], ['positive', 'original', 'witness'])
        self.assertEqual(row['lexical'], [.1, .2, .3])
        self.assertNotIn('hybrid_scores', row)
        self.assertNotIn('document_codes', row)
        self.assertIn('pairs', row)
        old = execution.example(self.record, self.query, self.documents, self.config, 'A')
        self.assertEqual(old['documents'], ['positive', 'original'])
        self.assertIn('target', old)

    def test_non_train_or_changed_query_text_is_rejected(self):
        for key, value in (('split', 'LOCKED_TEST'), ('query', 'changed'), ('query_id', 'other')):
            query = {**self.query, key: value}
            with self.assertRaises(ValueError):
                execution.example(self.record, query, self.documents, self.config, 'A')

    def test_changed_canonical_document_text_is_rejected(self):
        documents = copy.deepcopy(self.documents)
        documents[2]['text'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'document text changed'):
            execution.example(self.record, self.query, documents, self.config, 'D')

    def test_disabled_training_fails_before_loading_gpu_or_files(self):
        with patch.object(execution.training, 'Encoder') as encoder:
            with self.assertRaisesRegex(ValueError, 'disabled'):
                execution.train_chunk(None, self.config, None, 'A', 0, 96,
                                      {}, [], {}, [], None)
            encoder.assert_not_called()

    def test_invalid_order_fails_before_loading_gpu(self):
        config = {**self.config, 'training_enabled': True}
        with patch.object(execution.training, 'Encoder') as encoder:
            with self.assertRaisesRegex(ValueError, 'manifest incomplete'):
                execution.train_chunk(None, config, None, 'A', 0, 96,
                                      {}, [], {}, [], None)
            encoder.assert_not_called()

    def test_canary_uses_worst_observed_update_and_headroom(self):
        self.assertEqual(execution.projected_seconds([1., 2.], 96), 468.)
        for sample in ([], [0.], [float('nan')]):
            with self.assertRaises(ValueError):
                execution.projected_seconds(sample, 96)


class OptimizerSealTests(unittest.TestCase):
    def test_optimizer_names_count_and_config_are_bound_and_reload_exact(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        encoder = SimpleNamespace(model=model, device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'optimizer.pt'
            execution.save_optimizer(path, encoder, optimizer, 1)
            fresh = torch.optim.AdamW(model.parameters(), lr=.01)
            execution.restore_optimizer(path, encoder, fresh, 1)
            with self.assertRaisesRegex(ValueError, 'update count'):
                execution.restore_optimizer(path, encoder, fresh, 2)
            changed = torch.optim.AdamW(model.parameters(), lr=.02)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                execution.restore_optimizer(path, encoder, changed, 1)
            with self.assertRaises(FileExistsError):
                execution.save_optimizer(path, encoder, optimizer, 1)
            optimizer.state[model.weight]['step'].fill_(2)
            with self.assertRaisesRegex(ValueError, 'actual step'):
                execution.optimizer_fingerprint(encoder, optimizer, 1)

    def test_complete_requires_receipts_and_every_output_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {'results.json': {'value': 1},
                    'exit.json': dict(exit_code=0, error=None, owned_group_closed=True),
                    'clearml.json': dict(actual_start=True, closed=True)}
            for name, value in data.items():
                (root / name).write_text(json.dumps(value))
            hashes = {name: execution.io.sha(root / name) for name in data}
            (root / 'complete.json').write_text(json.dumps(dict(passed=True, files=hashes)))
            self.assertEqual(execution.sealed(root), {'value': 1})
            (root / 'results.json').write_text('{"value": 2}')
            with self.assertRaisesRegex(ValueError, 'content changed'):
                execution.sealed(root)


if __name__ == '__main__':
    unittest.main()
