"""NG77 context-only diagnostics cannot silently loosen a failed cache gate."""

import ast
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng77_context_diagnostic as context


def test_corpus_contexts_reconstruct_original_four_document_groups():
    assert context.corpus_groups([8, 2, 4, 3], total=10) == [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


@pytest.mark.parametrize('pool', [[], [1, 1], [-1], [10], [True], [1.]])
def test_invalid_diagnostic_document_identity_rejected(pool):
    with pytest.raises(ValueError, match='invalid pool'):
        context.corpus_groups(pool, total=10)


def test_value_mismatch_preserves_original_gate_and_records_each_row():
    before = sparse.csr_matrix(np.array([[.001, 0], [.01, .5]], dtype=np.float32))
    after = before.copy()
    after.data[0] += 1e-6
    result = context.array_audit(after, before)
    assert result['parity']['exact_support'] and not result['parity']['passed']
    assert result['bad_coordinates'] == 1 and result['bad_by_row'] == [1, 0]
    assert result['max_tolerance_ratio'] > 1
    assert context.array_audit(before, before)['parity']['passed']


def test_context_change_cannot_hide_a_tiny_support_difference():
    a = sparse.csr_matrix([[1., 1e-20]])
    b = sparse.csr_matrix([[1., 0.]])
    with pytest.raises(ValueError, match='equal support'):
        context.array_audit(a, b)


def test_no_optimizer_backward_or_evaluation_surface_calls():
    tree = ast.parse(Path(context.__file__).read_text())
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
              for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert not called & {'step', 'backward', 'optimizer_step', 'make_optimizer',
                         'train_chunk', 'surface_queries', 'rank', 'encode_checkpoint'}
    assert context.canary.PARITY == dict(rtol=2e-5, atol=2e-7, exact_support=True,
        keep_noise_relative_to_D_max=.001, vjp_relative_l2_max=1e-5,
        lipschitz_roundoff_atol=1e-12)
