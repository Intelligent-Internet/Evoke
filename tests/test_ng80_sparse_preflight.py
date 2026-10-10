"""TRAIN-only scale algebra and actual sparse canary contracts."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import ng80_sparse_preflight as preflight
import ng80_training as training
import ng80_lexical as native


def test_native_platform_mismatch_fails_before_loading(tmp_path, monkeypatch):
    library = tmp_path / 'wrong.so'
    library.write_bytes(b'\xcf\xfa\xed\xfe' + b'old Mac binary')
    monkeypatch.setattr(native.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(native.C, 'CDLL', lambda _: pytest.fail('must not load'))
    with pytest.raises(ValueError, match='platform mismatch'):
        native.Tokenizer(library)


def test_native_correct_header_still_requires_loadable_library(tmp_path, monkeypatch):
    library = tmp_path / 'broken.so'
    library.write_bytes(b'\x7fELF' + b'not a shared library')
    monkeypatch.setattr(native.platform, 'system', lambda: 'Linux')
    with pytest.raises(OSError):
        native.Tokenizer(library)


@pytest.mark.parametrize('failure', [False, True])
def test_icu_locale_is_explicit_not_inherited(failure):
    state = {'locale': b'en_CA'}

    def setter(value, status):
        if not failure:
            state['locale'] = value

    def getter():
        return state['locale']

    lib = SimpleNamespace(uloc_setDefault_78=setter, uloc_getDefault_78=getter)
    if failure:
        with pytest.raises(ValueError, match='original ICU locale'):
            native.pin_locale(lib)
    else:
        native.pin_locale(lib)
        assert state['locale'] == b'en_US_POSIX'


def test_native_parity_is_exact_for_ids_text_and_order():
    query = dict(query_id='TRAIN-q', query='Alpha beta alpha')
    tokens = ['alpha', 'beta', 'alpha']
    golden = dict(query_id=query['query_id'], tokens=tokens,
                  text_sha256=native.hashlib.sha256(query['query'].encode()).hexdigest())
    result = native.parity(lambda _: tokens, [query], [golden])
    assert result['exact_token_sequences'] is True
    assert result['locked_test_access'] is False
    for changed in (dict(golden, query_id='wrong'),
                    dict(golden, text_sha256='wrong'),
                    dict(golden, tokens=tokens[::-1] + ['extra'])):
        with pytest.raises(ValueError, match='parity mismatch'):
            native.parity(lambda _: tokens, [query], [changed])
    with pytest.raises(ValueError, match='row count'):
        native.parity(lambda _: tokens, [query], [])


def test_continuation_never_repeats_successful_reference_encode():
    assert preflight.CONTINUATION_PHASES == (
        'calibrate', 'train-S-canary', 'train-P-canary')
    assert preflight.REFERENCE == 'NG-0080/sparse-preflight-v1'
    assert preflight.LEXICAL_RUNTIME == 'NG-0080/lexical-runtime-v2'


def test_historical_lexical_parity_checks_shared_train_scores(monkeypatch, tmp_path):
    rows = [dict(query_id='q' + str(i)) for i in range(6144)]
    values = {'selection.json': dict(indices=list(range(6144))),
              'train-pools.json': [[0, 1]] * 6144,
              'train-lexical.json': [[2., 1.]] * 6144}
    monkeypatch.setattr(preflight.campaign, 'rows', lambda _: rows)
    monkeypatch.setattr(preflight.campaign, 'read', lambda p: values[p.name])
    queries = [rows[0], dict(query_id='new-train')]
    lexical = preflight.sparse.csr_matrix([[2., 1.], [8., 9.]])
    documents = preflight.sparse.eye(2, format='csr')
    result = preflight.cached_lexical_parity(tmp_path, queries, lexical, documents)
    assert result['shared_train_queries'] == 1
    assert result['diagnostic_or_dev_scores_used'] is False
    values['train-lexical.json'][0] = [3., 1.]
    with pytest.raises(ValueError, match='historical TRAIN lexical score mismatch'):
        preflight.cached_lexical_parity(tmp_path, queries, lexical, documents)


def test_lexical_roundoff_accepts_ulp_not_feature_changes():
    value = np.array([.9, .2, 0.])
    adjacent = np.array([np.nextafter(.9, 1.), .2, 0.])
    error, bound = preflight.lexical_roundoff(adjacent, value, 12)
    assert 0 < error < bound < 1e-12
    for changed in ([.900001, .2, 0.], [.9, .2, 1e-30],
                    [.9, -.2, 0.], [.9, np.nan, 0.]):
        with pytest.raises(ValueError):
            preflight.lexical_roundoff(changed, value, 12)


def test_scale_uses_query_variances_not_candidate_count_or_covariance():
    teacher = [[0., 2.], [0., 4., 8.], [1., 2., 3., 4.]]
    student = [np.asarray(t) * -3 + 10 for t in teacher]
    result = preflight.teacher_scale(student, teacher, list(preflight.DOMAINS))
    assert result['scale'] == pytest.approx(3.)
    expected = np.mean([np.var(t) for t in teacher])
    assert result['moments']['teacher_variance'] == pytest.approx(expected)
    assert result['moments']['cross_covariance'] < 0
    assert result['diagnostic_used'] is False
    assert result['serving_rms_changed'] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('student,teacher,domains', [
    ([], [], []),
    ([[1., 2.]] * 2, [[1., 2.]] * 2, list(preflight.DOMAINS[:2])),
    ([[1., 1.]] * 3, [[1., 2.]] * 3, list(preflight.DOMAINS)),
    ([[1., 2.]] * 3, [[1., 1.]] * 3, list(preflight.DOMAINS)),
    ([[1., np.nan]] * 3, [[1., 2.]] * 3, list(preflight.DOMAINS)),
    ([[1., 2.]] * 3, [[1.]] * 3, list(preflight.DOMAINS)),
])
def test_invalid_or_unbalanced_scale_fails(student, teacher, domains):
    with pytest.raises(ValueError):
        preflight.teacher_scale(student, teacher, domains)


def test_normalized_field_loss_has_explicit_inverse_square_prefactor():
    s = torch.tensor([.3, 1.1, -.2], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([.1, .3, .8], dtype=torch.float64)
    scale = 2.3
    actual = training.field_loss(s, t, scale)
    error = s - scale * t
    unnormalized = (error - error.mean()).square().mean()
    torch.testing.assert_close(actual, unnormalized / scale ** 2)
    actual_gradient = torch.autograd.grad(actual, s, retain_graph=True)[0]
    other_gradient = torch.autograd.grad(unnormalized, s)[0]
    torch.testing.assert_close(actual_gradient, other_gradient / scale ** 2)


def test_fixture_preserves_fixed_pool_order():
    assert preflight.fixture([8, 4, 2, 9, 5, 3], [2, 5]) == [2, 8, 4, 9]
    for pool, gold in (([1, 2, 3], [1]), ([1, 1, 2, 3, 4], [1]),
                       ([1, 2, 3, 4], [8])):
        with pytest.raises(ValueError):
            preflight.fixture(pool, gold)


def test_token_work_counts_each_exposure_with_role_limits():
    calls = []

    def tokenize(texts, **kwargs):
        calls.append((texts, kwargs))
        return {'input_ids': [list(range(min(len(t), kwargs['max_length'])))
                              for t in texts]}

    encoder = SimpleNamespace(tokenizer=tokenize, config={'base': {
        'query_input_tokens': 2, 'document_input_tokens': 4}})
    result = preflight.token_exposures(encoder, {
        'query': 'abcdef', 'documents': ['12345', '12345', '12']})
    assert result == {'query': 2, 'document': 10}
    assert calls[0][1] == {'truncation': True, 'max_length': 2}
    assert calls[1][1] == {'truncation': True, 'max_length': 4}


def test_budget_does_not_authorize_training_or_relax_smallest_chunk():
    result = preflight.chunk_budget(32.)
    assert result['largest_safe_chunk'] == 1024
    assert result['training_authorized_by_this_receipt'] is False
    assert preflight.chunk_budget(16000.)['largest_safe_chunk'] is None
    for seconds in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            preflight.chunk_budget(seconds)
    with pytest.raises(ValueError):
        preflight.chunk_budget(32., 15)


class SparseToy:
    def __init__(self, kind):
        self.kind = kind
        self.weight = torch.tensor([[.1, .2], [.5, .3]],
                                   dtype=torch.float64, requires_grad=True)

    def encode(self, rows, role):
        values = torch.as_tensor(rows, dtype=torch.float64) @ self.weight
        return values.relu().log1p()


@pytest.mark.parametrize('kind', ['S', 'P'])
def test_actual_sparse_scoring_replay_includes_lexical_once(kind):
    a, b = SparseToy(kind), SparseToy(kind)
    item = dict(query=[.1, .8], documents=[
        [.2, .7], [.5, .4], [.1, .2], [.4, .9], [.8, .1]],
        teacher=[.9, .4, .8, .2, .3],
        positive=[True, False, False, False, False],
        lexical=[.01, .02, .05, .02, .03], scale=2.3)
    result = training.backward(a, **item, replay=True)
    other = training.backward(b, **item, replay=False)
    torch.testing.assert_close(a.weight.grad, b.weight.grad, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result['score_gradient'], other['score_gradient'])
    semantic = (a.encode(item['documents'], 'document')
                @ a.encode([item['query']], 'query').T).detach().numpy().ravel()
    expected = semantic + item['lexical'] if kind == 'P' else semantic
    np.testing.assert_allclose(result['scores'], expected, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(a.weight.grad.numpy()) > 0
