"""Independent target accumulation fixtures, including multiple positives."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / 'scripts/audit_ng69_teacher.py'
SPEC = importlib.util.spec_from_file_location('ng69_audit', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class TeacherAuditTests(unittest.TestCase):
    def test_independent_dot_and_soft_target(self):
        rng = np.random.default_rng(69069)
        query = rng.standard_normal(1024).astype(np.float32)
        docs = rng.standard_normal((17, 1024)).astype(np.float32)
        query /= np.linalg.norm(query)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        positive = np.arange(17) < 3
        score, target = audit.independently_recompute(query, docs, positive)
        expected_score = query.astype(np.float64) @ docs.astype(np.float64).T
        teacher = np.exp((expected_score - expected_score.max()) / .04)
        teacher /= teacher.sum()
        np.testing.assert_allclose(score, expected_score, rtol=0, atol=1e-13)
        np.testing.assert_allclose(target, .5 * positive / 3 + .5 * teacher,
                                   rtol=0, atol=1e-12)

    def test_missing_positive_fails(self):
        with self.assertRaises(ValueError):
            audit.independently_recompute(np.ones(2), np.ones((3, 2)),
                                          np.zeros(3, dtype=bool))


if __name__ == '__main__':
    unittest.main()
