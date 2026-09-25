"""Separate same-context identity gates from cross-context diagnostic values."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng77_cuda_canary as canary


class ContextEncoder:
    device = 'cpu'

    def encode(self, texts, role):
        assert role == 'document'
        # Deliberately context-dependent value, without changing support.
        return torch.tensor([[float(t) + max(map(float, texts)) * 1e-4] for t in texts])


def setup_case():
    encoder = ContextEncoder()
    documents = {i: dict(text=str(i + 1)) for i in range(8)}
    pool = [0, 7]
    groups = canary.corpus_groups(pool, total=8)
    actual = encoder.encode([documents[i]['text'] for i in pool], 'document')
    expected = sparse.vstack([sparse.csr_matrix(encoder.encode(
        [documents[i]['text'] for i in group], 'document').numpy()) for group in groups])
    return encoder, documents, pool, groups, actual, expected.tocsr()


def test_same_context_parity_not_obtained_by_widening_values(tmp_path):
    e, docs, pool, groups, actual, expected = setup_case()
    with patch.object(canary, 'legacy_documents', side_effect=lambda e, texts: e.encode(texts, 'document')):
        result = canary.context_controls(e, docs, pool, groups, actual, expected, tmp_path, 0)
    assert result['document']['passed'] and result['document']['max_abs_error'] == 0
    assert result['pool_formula_bit_exact'] and result['cache_context_documents'] == 8
    assert result['cross_context_diagnostic']['max_abs_error'] > 0
    assert not result['cross_context_diagnostic']['passed']
    assert np.array_equal(sparse.load_npz(tmp_path / 'corpus-documents-00.npz').data, expected.data)


def test_cache_mismatch_is_not_hidden_even_for_nonpool_companion(tmp_path):
    e, docs, pool, groups, actual, expected = setup_case()
    expected[1, 0] += 1
    with patch.object(canary, 'legacy_documents', side_effect=lambda e, texts: e.encode(texts, 'document')):
        result = canary.context_controls(e, docs, pool, groups, actual, expected, tmp_path, 0)
    assert not result['document']['passed']


def test_wrong_formula_fails_before_cache_controls(tmp_path):
    e, docs, pool, groups, actual, expected = setup_case()
    with patch.object(canary, 'legacy_documents', return_value=actual + 1):
        with pytest.raises(ValueError, match='formula differs'):
            canary.context_controls(e, docs, pool, groups, actual, expected, tmp_path, 0)
    assert not (tmp_path / 'corpus-documents-00.npz').exists()


def test_corpus_groups_cover_each_original_companion_without_pool_reordering():
    pool = [7, 0, 4]
    assert canary.corpus_groups(pool, total=9) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert pool == [7, 0, 4]
