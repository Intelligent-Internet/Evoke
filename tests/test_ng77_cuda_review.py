"""Independent CPU primitives reject unsupported code equivalence."""

import ast
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_ng77_cuda_canary as review


def test_independent_sparse_scalar_product_and_lexical_term():
    q = sparse.csr_matrix([[0., 2., 3.]])
    d = sparse.csr_matrix([[5., 1., 0.], [0., 4., 2.]])
    assert review.scalar_scores(q, d, [.5, .25]) == [2.5, 14.25]


def test_same_context_support_and_original_value_limits_are_both_required():
    a = sparse.csr_matrix(np.array([[1., 1e-10, 0.]], dtype=np.float32))
    assert review.check_codes(a, a) == 0
    b = sparse.csr_matrix(np.array([[1., 0., 0.]], dtype=np.float32))
    with pytest.raises(ValueError, match='support'):
        review.check_codes(a, b)
    b = a.copy()
    b.data[0] += 1e-3
    with pytest.raises(ValueError, match='value'):
        review.check_codes(b, a)


def test_reviewer_does_not_import_autograd_or_producer_math():
    tree = ast.parse(Path(review.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not set(imports) & {'torch', 'ng77_retention', 'ng77_cuda_canary', 'ng71_training'}
