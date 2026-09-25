import copy
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng72_diagnose as d


class CrossCodeTests(unittest.TestCase):
    def test_exact_role_and_support_decomposition(self):
        q0, q1 = np.array([2., 0., 3., 4.]), np.array([1., 5., 0., 4.])
        a = sparse.csr_matrix([[1., 2., 4., 0.], [3., 0., 1., 2.]])
        b = sparse.csr_matrix([[2., 3., 1., 0.], [0., 1., 2., 4.]])
        terms, delta = d.decomposition(q0, q1, a, b)
        np.testing.assert_array_equal(delta, b @ q1 - a @ q0)
        np.testing.assert_array_equal(sum(terms[k] for k in d.MODAL_TERMS), delta)
        np.testing.assert_array_equal(sum(terms[k] for k in d.SUPPORT_TERMS), delta)
        self.assertTrue((terms['support_lost'] <= 0).all())
        self.assertTrue((terms['support_gained'] >= 0).all())
        self.assertGreater(float(abs(terms['interaction']).sum()), 0)

    def test_identity_has_no_effect(self):
        q = np.array([0., 2., 3.])
        doc = sparse.csr_matrix([[1., 2., 3.], [4., 0., 5.]])
        terms, delta = d.decomposition(q, q, doc, doc)
        for values in [*terms.values(), delta]:
            np.testing.assert_array_equal(values, np.zeros(2))

    def test_query_only_drift_has_no_document_or_interaction(self):
        doc = sparse.csr_matrix([[1., 2.]])
        terms, _ = d.decomposition(np.array([1., 0.]), np.array([0., 3.]), doc, doc)
        self.assertEqual(terms['document'][0], 0)
        self.assertEqual(terms['interaction'][0], 0)

    def test_document_only_drift_has_no_query_or_interaction(self):
        q = np.array([1., 2.])
        terms, _ = d.decomposition(q, q, sparse.csr_matrix([[1., 0.]]),
                                   sparse.csr_matrix([[0., 3.]]))
        self.assertEqual(terms['query'][0], 0)
        self.assertEqual(terms['interaction'][0], 0)

    def test_random_margin_identity(self):
        rng = np.random.default_rng(72)
        for _ in range(20):
            a, b = [rng.uniform(size=(7, 13)) for _ in range(2)]
            a[a < .5], b[b < .5] = 0, 0
            q0, q1 = rng.uniform(size=(2, 13))
            terms, delta = d.decomposition(q0, q1, sparse.csr_matrix(a), sparse.csr_matrix(b))
            for i in range(1, 7):
                expected = (b[0] @ q1 - b[i] @ q1) - (a[0] @ q0 - a[i] @ q0)
                self.assertAlmostEqual(sum(terms[k][0] - terms[k][i] for k in d.MODAL_TERMS), expected)
                self.assertAlmostEqual(delta[0] - delta[i], expected)

    def surface(self):
        rows = [dict(query_id=str(i), split='TRAIN', subset=domain)
                for i, domain in enumerate(d.DOMAINS * 2)]
        return rows, dict(pilot=['0', '1', '2'], sentinel=['3', '4', '5'])

    def test_train_only_and_fixed_order(self):
        rows, selection = self.surface()
        result = d.train_surface(rows, selection, 1)
        self.assertEqual([r['lexical_index'] for r in result], list(range(6)))
        self.assertEqual({r['split'] for r in result}, {'TRAIN'})

    def test_dev_and_locked_rejected(self):
        for split in ('DEV_NEW', 'DEV_EXPOSED', 'LOCKED_TEST'):
            rows, selection = self.surface()
            rows[0]['split'] = split
            with self.assertRaises(ValueError):
                d.train_surface(rows, selection, 1)

    def test_duplicate_and_surface_overlap_rejected(self):
        rows, selection = self.surface()
        with self.assertRaises(ValueError):
            d.train_surface(rows + [rows[0]], selection, 1)
        selection['sentinel'][0] = '0'
        with self.assertRaises(ValueError):
            d.train_surface(rows, selection, 1)

    def test_teacher_unobserved_not_negative(self):
        self.assertEqual(d.teacher_pair(None, 1, 2)['status'], 'unobserved')
        record = dict(pool=[1, 2, 3], scores=[.3, .1, .3])
        self.assertEqual(d.teacher_pair(record, 1, 8)['status'], 'unobserved')
        self.assertEqual(d.teacher_pair(record, 1, 2)['status'], 'agrees')
        self.assertEqual(d.teacher_pair(record, 2, 1)['status'], 'opposes')
        self.assertEqual(d.teacher_pair(record, 1, 3)['status'], 'tie')
        self.assertAlmostEqual(d.teacher_pair(record, 1, 2)['confidence'], np.tanh(.2 / .08))

    def test_exact_ties_and_all_gold_are_part_of_parity(self):
        record = dict(gold_ids=[1, 2], gold_ranks=[2, 101], top100=[0, 1],
                      top100_scores=[.5, .5], ndcg10=.4, recall100=.5)
        d.frozen_rank_match(record, record)
        changed = copy.deepcopy(record)
        changed['top100'] = [1, 0]
        with self.assertRaises(ValueError):
            d.frozen_rank_match(changed, record)
        changed = copy.deepcopy(record)
        changed['gold_ranks'][1] = 102
        with self.assertRaises(ValueError):
            d.frozen_rank_match(changed, record)

    def test_manifest_rejects_escape_or_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'run').mkdir()
            d.write(root / 'value.json', {'value': 1})
            frozen = dict(protocol='NG72_TRAIN_cross_codes_v1', training_updates=0,
                          new_model_inference=False, dependencies={'value.json': d.sha(root / 'value.json')},
                          source={})
            d.write(root / 'run/inputs.json', frozen)
            d.verify(root, root / 'run')
            frozen['source'] = {'../value.json': d.sha(root / 'value.json')}
            with patch.object(d, 'read', return_value=frozen):
                with self.assertRaises(ValueError):
                    d.verify(root, root / 'run')
            frozen['source'] = {}
            frozen['dependencies']['value.json'] = '0' * 64
            with patch.object(d, 'read', return_value=frozen):
                with self.assertRaises(ValueError):
                    d.verify(root, root / 'run')

    def test_summary_uses_one_domain_and_balanced_positive_denominators(self):
        ranks, margins = {mode: [] for mode in d.MODES}, []
        for surface in ('TRAIN_PILOT', 'TRAIN_SENTINEL'):
            for domain in d.DOMAINS:
                identity = surface + domain
                for mode in d.MODES:
                    ranks[mode].append(dict(query_id=identity, domain=domain,
                                             surface=surface, ndcg10=.5, recall100=1.))
                for value in (1., 3.):
                    means = {term: value for term in (*d.MODAL_TERMS, *d.SUPPORT_TERMS, 'total')}
                    margins.append(dict(query_id=identity, surface=surface, domain=domain,
                                        origin='original_positive', competitors=2, means=means,
                                        teacher_counts={'agrees': 1, 'unobserved': 1},
                                        retention={k: dict(pairs=0, **means) for k in
                                                   ('lost', 'retained', 'gained', 'stayed_behind')}))
        audit = SimpleNamespace(harm=lambda a, b: {'queries': len(a)})
        result = d.summarize(ranks, margins, audit)
        for group in result['margin_effects'].values():
            self.assertEqual(group['all']['query_balanced_mean']['query'], 2.)
            self.assertEqual(group['all']['queries'], 1)
            self.assertEqual(group['all']['positives'], 2)
            self.assertEqual(group['teacher_status_pair_counts'], {'agrees': 2, 'unobserved': 2})
            self.assertEqual(group['added_vs_original']['query_balanced_mean'], {})

    def test_started_receipt_failure_closes_owned_process(self):
        import ng71_pilot as pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     start_new_session=True)
            original = d.write

            def fail_start(path, value):
                if path.name == 'started.json':
                    raise OSError('injected receipt failure')
                original(path, value)

            try:
                with patch.object(d, 'verify'), patch.object(d, 'resources_ok', return_value=True), \
                        patch.object(d, 'parent_modules', return_value=(pipeline, None, None)), \
                        patch.object(d.subprocess, 'Popen', return_value=child), \
                        patch.object(d, 'write', side_effect=fail_start):
                    with self.assertRaises(ValueError):
                        d.supervise(root, root)
                self.assertIsNotNone(child.poll())
                self.assertTrue(d.read(root / 'diagnosis/exit.json')['owned_group_closed'])
                self.assertFalse(d.read(root / 'diagnosis/complete.json')['passed'])
            finally:
                if child.poll() is None:
                    pipeline.terminate_group(child)


if __name__ == '__main__':
    unittest.main()
