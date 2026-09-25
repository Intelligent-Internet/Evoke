"""Independent dense accumulation and batch equivalence fixtures."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / 'scripts/diagnose_ng69_dense_rank.py'
spec = importlib.util.spec_from_file_location('dense_diagnostic', path)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class DenseDiagnosticTests(unittest.TestCase):
    def test_nonmultiple_chunk_and_all_primary_modes(self):
        rng = np.random.default_rng(69)
        documents = rng.normal(size=(4101, 1024)).astype(np.float32).astype(np.float64)
        queries = rng.normal(size=(16, 1024)).astype(np.float32).astype(np.float64)
        documents /= np.linalg.norm(documents, axis=1)[:, None]
        queries /= np.linalg.norm(queries, axis=1)[:, None]
        reference = diagnostic.primary(documents, queries, 'original')
        for mode in diagnostic.MODES:
            value = diagnostic.primary(documents, queries, mode)
            np.testing.assert_allclose(value, reference, rtol=0, atol=1e-12)
            for i, query in enumerate(queries):
                other = diagnostic.alternative(documents, query, mode)
                np.testing.assert_allclose(other, value[i], rtol=0, atol=1e-12)

    def test_duplicate_vectors_keep_exact_ties(self):
        documents = np.tile(np.linspace(-.1, .1, 1024), (130, 1))
        queries = documents[:16].copy()
        for mode in diagnostic.MODES:
            values = diagnostic.primary(documents, queries, mode)
            np.testing.assert_array_equal(values, np.tile(values[:, :1], (1, 130)))


if __name__ == '__main__':
    unittest.main()
