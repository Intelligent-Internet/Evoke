"""Pure fixtures for stage ordering, replay slices and exact ranking ties."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load('ng69_pipeline')
lexical = load('prepare_ng69_lexical')


class PipelineTests(unittest.TestCase):
    def test_dense_alternate_preserves_scores_and_tied_ranks(self):
        rng = np.random.default_rng(69069)
        documents = rng.normal(size=(211, 1024))
        documents[100:105] = documents[0]
        vector = rng.normal(size=1024)
        old = np.sum(documents * vector, axis=1)
        actual = pipeline.dense_alternate(documents, vector)
        np.testing.assert_allclose(actual, old, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual, documents @ vector, rtol=0, atol=1e-12)
        self.assertEqual(pipeline.top_independent(actual),
                         pipeline.top_independent(old))

    def inherited_fixture(self, base):
        parent, run = base / 'old', base / 'new'
        parent.mkdir()
        run.mkdir()
        for name in ('protocol.md', 'research-protocol.md'):
            (parent / name).write_text(name)
            (run / name).write_text(name)
        (parent / 'ng69_pipeline.py').write_text('frozen original source')
        (run / 'continuation-protocol.md').write_text('No retraining; same gates')
        frozen = {'files': {}, 'phases': pipeline.schedule(),
                  'seeds': list(pipeline.SEEDS),
                  'source_sha256': pipeline.sha(parent / 'ng69_pipeline.py'),
                  'protocol_sha256': pipeline.sha(parent / 'protocol.md'),
                  'research_protocol_sha256': pipeline.sha(parent / 'research-protocol.md')}
        pipeline.write(parent / 'inputs.json', frozen)
        pipeline.write(parent / 'continuation-state.json', {'status': 'failed_preserve_attempt'})
        failed = parent / 'rank-dense'
        failed.mkdir()
        pipeline.write(failed / 'exit.json', {'exit_code': 1, 'owned_group_closed': True})
        pipeline.write(failed / 'clearml.json', {'closed': True})
        pipeline.write(failed / 'canary.json', {'passed': False})
        for name in pipeline.schedule()[:14]:
            folder = parent / name
            folder.mkdir()
            pipeline.write(folder / 'results.json', {'metric': 1})
            pipeline.write(folder / 'clearml.json', {'closed': True})
            pipeline.write(folder / 'exit.json', {'exit_code': 0,
                'owned_group_closed': True, 'error': None})
            pipeline.write(folder / 'complete.json', {'passed': True, 'files': {
                'results.json': pipeline.sha(folder / 'results.json')}})
        return parent, run, frozen

    def test_inherit_only_fourteen_successes_and_reject_changed_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            parent, run, frozen = self.inherited_fixture(base)
            before = {str(p.relative_to(parent)): pipeline.sha(p)
                      for p in parent.rglob('*') if p.is_file()}
            inherited = pipeline.inherit_successes(base, run, parent, frozen)
            self.assertEqual(len(inherited['phases']), 14)
            self.assertFalse((run / 'rank-dense').exists())
            self.assertEqual(before, {str(p.relative_to(parent)): pipeline.sha(p)
                                      for p in parent.rglob('*') if p.is_file()})
            frozen.update(inherited=inherited,
                          source_sha256=pipeline.sha(Path(pipeline.__file__)))
            pipeline.write(run / 'inputs.json', frozen)
            pipeline.verify(base, run)
            # Resume skips sealed phases without spawning or touching tracking.
            with patch.object(pipeline, 'legacy'), patch.dict(
                    'sys.modules', lifecycle52=type('Lifecycle', (), {'stop_controller': None})):
                args = type('Args', (), {'run': run, 'research_root': base})
                pipeline.phase(args, 'train-59059-1')
            (run / 'train-59059-1').unlink()
            (run / 'train-59059-1').symlink_to(parent / 'train-59059-2')
            with self.assertRaises(AssertionError):
                pipeline.verify(base, run)

    def test_inheritance_rejects_failed_success_or_changed_contract(self):
        for failure in ('failed_training', 'changed_source', 'changed_protocol', 'inputs'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                parent, run, frozen = self.inherited_fixture(base)
                if failure == 'failed_training':
                    pipeline.write(parent / 'train-59059-1/exit.json', {
                        'exit_code': 1, 'owned_group_closed': True, 'error': 'failed'})
                elif failure == 'changed_source':
                    (parent / 'ng69_pipeline.py').write_text('changed')
                elif failure == 'changed_protocol':
                    frozen['protocol_sha256'] = 'changed'
                else:
                    frozen['files'] = {'different': 'input'}
                with self.assertRaises(AssertionError):
                    pipeline.inherit_successes(base, run, parent, frozen)
                self.assertFalse(any((run / name).is_symlink()
                                     for name in pipeline.schedule()))

    def test_training_pool_identity_and_all_positive_contract(self):
        query = {'query_id': 'q1', 'split': 'TRAIN'}
        target = {'query_id': 'q1', 'pool': [2, 3, 5],
                  'positive_mask': [True, False, True], 'target': [.4, .2, .4]}
        pipeline.validate_example(query, [2, 5], [2, 3, 5], [1, 0, 1], target)
        for labels, pool, values in (([2], [2, 3, 5], [1, 0, 1]),
                                     ([2, 5], [2, 5, 3], [1, 0, 1]),
                                     ([2, 5], [2, 3, 5], [1, float('nan'), 1])):
            with self.assertRaises(AssertionError):
                pipeline.validate_example(query, labels, pool, values, target)
        with self.assertRaises(AssertionError):
            pipeline.validate_example(dict(query, split='DEV_NEW'), [2, 5],
                                      [2, 3, 5], [1, 0, 1], target)

    def test_quarters_are_one_permutation_not_repeated_old_population(self):
        for seed in pipeline.SEEDS:
            groups = [pipeline.order(seed, q) for q in range(1, 5)]
            self.assertEqual([len(g) for g in groups], [1536] * 4)
            values = sum(groups, [])
            self.assertEqual(sorted(values), list(range(6144)))
            self.assertEqual(values, np.random.default_rng(seed).permutation(6144).tolist())

    def test_invalid_seed_or_quarter_rejected(self):
        for seed, quarter in ((1, 1), (59059, 0), (59059, 5)):
            with self.assertRaises(AssertionError):
                pipeline.order(seed, quarter)

    def test_training_precedes_all_dev_evaluation(self):
        phases = pipeline.schedule()
        self.assertEqual(len(phases), len(set(phases)))
        self.assertEqual(len(phases), 29)
        self.assertTrue(all(p.startswith('train-') for p in phases[:12]))
        self.assertFalse(any(p.startswith('train-') for p in phases[12:]))
        for model in ('dense', 'initial', 'A-59059', 'B-59059'):
            self.assertLess(phases.index('encode-' + model), phases.index('rank-' + model))

    def test_independent_top_handles_tie_at_100_boundary(self):
        values = np.zeros(150)
        values[130:] = 1
        expected = np.lexsort((np.arange(150), -values))[:100].tolist()
        self.assertEqual(pipeline.top_independent(values), expected)

    def test_independent_top_all_equal_and_short(self):
        self.assertEqual(pipeline.top_independent(np.ones(150)), list(range(100)))
        self.assertEqual(pipeline.top_independent(np.ones(3)), [0, 1, 2])

    def test_independent_top_random(self):
        values = np.random.default_rng(69).normal(size=233009)
        expected = np.lexsort((np.arange(len(values)), -values))[:100].tolist()
        self.assertEqual(pipeline.top_independent(values), expected)

    def test_changed_phase_output_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            phase = run / 'example'
            phase.mkdir()
            pipeline.write(phase / 'results.json', {'metric': 1})
            pipeline.write(phase / 'clearml.json', {'closed': True})
            pipeline.write(phase / 'complete.json', {'passed': True, 'files': {
                'results.json': pipeline.sha(phase / 'results.json')}})
            pipeline.write(phase / 'exit.json', {'exit_code': 0,
                'owned_group_closed': True, 'error': None})
            self.assertEqual(pipeline.completed(run, 'example'), {'metric': 1})
            pipeline.write(phase / 'results.json', {'metric': 2})
            with self.assertRaises(AssertionError):
                pipeline.completed(run, 'example')

    def test_failed_or_unclosed_phase_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            phase = run / 'example'
            phase.mkdir()
            for code, closed in ((1, True), (0, False)):
                pipeline.write(phase / 'exit.json', {'exit_code': code,
                    'owned_group_closed': closed, 'error': None})
                with self.assertRaises(AssertionError):
                    pipeline.completed(run, 'example')

    def test_frozen_bm25_formula_uses_full_length_including_oov(self):
        counts = sparse.csr_matrix([[2, 1], [0, 1]], dtype=np.float32)
        lengths = np.array([7, 4])
        idf = np.array([1.2, .8])
        values = lexical.weight_documents(counts, lengths, idf, 3).toarray()
        expected = np.array([[1.2 * 2 / (2 + 1.5 * (.25 + .75 * 7 / 3)),
                              .8 / (1 + 1.5 * (.25 + .75 * 7 / 3))],
                             [0, .8 / (1 + 1.5 * (.25 + .75 * 4 / 3))]],
                            dtype=np.float32)
        np.testing.assert_array_equal(values, expected)
        self.assertEqual(values.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
