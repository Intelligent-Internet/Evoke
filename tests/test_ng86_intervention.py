"""NG86 split, single-factor, failure and fixed-analysis contracts."""

from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).parent
sys.path.insert(0, str(SOURCE))
from ng86_intervention import select_panel, support_failure  # noqa: E402
from review_ng86_intervention import summarize  # noqa: E402


def test_panel_is_larger_nested_balanced_and_never_locked():
    from ng85_diagnose import panel_indices

    queries = [dict(query_id=f'{split}/{domain}/{i}', split=split, subset=domain)
        for split in ('fit', 'validation', 'locked_test')
        for domain in ('fever', 'hotpotqa', 'nq', 'fiqa') for i in range(150)]
    panel = select_panel(queries)
    assert len(panel) == len(set(panel)) == 640
    assert set(panel_indices(queries)) <= set(panel)
    assert panel == select_panel(queries)
    assert sum(queries[i]['split'] == 'validation' for i in panel) == 512
    assert not any(queries[i]['split'] == 'locked_test' for i in panel)
    with pytest.raises(ValueError):
        select_panel(queries[:5])


def test_only_finite_nonempty_zero_support_is_expected_failure():
    event = dict(raw_finite=True, denominator_finite=True, denominator=0.,
                 raw_nnz=180, supported_nnz=0, unsupported_nnz=180)
    assert support_failure(event, True)
    assert not support_failure(event, False)
    for key, value in (('raw_finite', False), ('denominator_finite', False),
                       ('denominator', .001), ('raw_nnz', 0),
                       ('supported_nnz', 1), ('unsupported_nnz', 179)):
        assert not support_failure(dict(event, **{key: value}), True)
    assert not support_failure(None, True)


def records():
    output = []
    for arm in ('control', 'fullstd'):
        for i in range(512):
            output.append(dict(arm=arm, checkpoint='2816', split='validation',
                query_index=i, subset=('fever', 'hotpotqa', 'nq', 'fiqa')[i // 128],
                valid=True, candidate_ndcg10=.5 if arm == 'control' else .6,
                candidate_recall10=.8, document_nnz_mean=100,
                query_raw=[dict(unsupported_mass=1., raw_sum=2.)]))
    return output


def test_review_fixed_primary_and_invalid_query_denominator():
    sample = records()
    sample[-1].update(valid=False, candidate_ndcg10=0., candidate_recall10=0.)
    del sample[-1]['document_nnz_mean']
    result = summarize(sample)
    metric = result['metrics']['validation']['fullstd-2816']
    assert metric['invalid'] == 1
    assert metric['candidate_ndcg10'] == pytest.approx(.6 * 511 / 512)
    assert result['paired_validation']['2816']['ndcg_delta'] == pytest.approx(.6 * 511 / 512 - .5)
    assert not result['paired_validation']['3072']['available']
    assert result['primary_step'] == 2816
    sample[-1]['candidate_ndcg10'] = .6
    with pytest.raises(ValueError, match='invalid queries'):
        summarize(sample)


def test_review_rejects_duplicate_and_missing_pairs():
    sample = records()
    sample[-1]['query_index'] = sample[-2]['query_index']
    with pytest.raises(ValueError):
        summarize(sample)
    sample = records()
    sample[-1]['query_index'] = 9999
    with pytest.raises(ValueError, match='paired query identity'):
        summarize(sample)


@pytest.mark.skipif(sys.platform == 'darwin', reason='actual Torch checks run on Lambda2')
@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_single_factor_forward_and_full_derivative(dtype_name):
    import torch
    from ng85_diagnose import full_std_loss, geometry
    from ng85_train import calibrated_loss

    dtype = getattr(torch, dtype_name)
    tolerance = 2e-6 if dtype_name == 'float32' else 1e-12
    target = np.array([.01, .04, .15, .8])
    for values in ([.2, .5, 2., 3.], [1., 1., 1., 1.]):
        s = torch.tensor(values, dtype=dtype, requires_grad=True)
        original, changed = calibrated_loss(s, target), full_std_loss(s, target)
        assert torch.equal(original, changed)
        derivative, = torch.autograd.grad(changed, s)
        assert torch.isfinite(derivative).all()
        centered = s.detach() - s.detach().mean()
        assert float(derivative @ centered) == pytest.approx(0., abs=tolerance)
        expected = geometry(s.detach().double().numpy(), target)
        assert float(derivative.norm()) == pytest.approx(
            expected['full_gradient_norm'], rel=tolerance, abs=tolerance)
