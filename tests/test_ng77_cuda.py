"""Synthetic adapter, canary selection and no-optimizer execution guards."""

import ast
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_training as original
import ng77_cuda_canary as canary
from ng77_training import RetainedObjective


class Encoder:
    def __init__(self):
        self.model = torch.nn.Linear(3, 4, dtype=torch.float64)
        with torch.no_grad():
            self.model.weight.fill_(.2)
            self.model.bias.copy_(torch.arange(1., 5.))
        self.config = dict(optimizer=dict(queries_per_update=4, gradient_clip_norm=1.))

    def encode(self, texts, role):
        values = self.model(torch.tensor(texts, dtype=torch.float64)).relu().log1p()
        return .9 * values / values.sum(1, keepdim=True) if role == 'query' else values


def fixture():
    pairs = dict(indices=[[0, 1]], targets=[.8], coefficients=[.6], candidate_count=2,
                 student_temperature=1., supervised_positives=1)
    example = dict(query=[1., 2., 3.], documents=[[1., 0., 2.], [0., 1., 1.]],
                   lexical=[.1, .2], pairs=pairs)
    anchors = dict(candidate_count=2, total_positives=1, student_temperature=1.,
                   anchors=[dict(indices=[0, 1], baseline_margin=2., coefficient=.7)])
    return example, anchors


def test_default_backward_and_zero_coefficient_keep_exact_shared_gradients():
    a, b = Encoder(), Encoder()
    example, anchors = fixture()
    x = original.backward_query(a, **example)
    objective = RetainedObjective(anchors, 0.)
    y = original.backward_query(b, **example, loss_transform=objective)
    assert x == y and objective.last['keep_loss'] > 0
    for p, q in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.equal(p.grad, q.grad)


@pytest.mark.parametrize('accumulation', [1, 4])
def test_nonzero_transformed_loss_retains_actual_vjp_both_roles(accumulation):
    a, b = Encoder(), Encoder()
    example, anchors = fixture()
    x = original.backward_query(a, **example, replay=False, accumulation=accumulation,
                                 loss_transform=RetainedObjective(anchors, 1.))
    y = original.backward_query(b, **example, replay=True, accumulation=accumulation,
                                 loss_transform=RetainedObjective(anchors, 1.))
    assert x['loss'] == y['loss'] and x['score_gradient'] == y['score_gradient']
    for p, q in zip(a.model.parameters(), b.model.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('bad', ['detached', 'nan', 'vector', 'number'])
def test_loss_transform_must_not_break_the_gradient_chain(bad):
    a = Encoder()
    example, _ = fixture()
    def transform(scores, loss):
        return dict(detached=loss.detach(), nan=loss * float('nan'),
                    vector=scores, number=0.)[bad]
    with pytest.raises(ValueError, match='differentiable scalar'):
        original.backward_query(a, **example, loss_transform=transform)


def test_full_synthetic_real_canary_math_does_not_change_model(tmp_path):
    a = Encoder()
    example, _ = fixture()
    state = deepcopy(a.model.state_dict())
    result = canary.vjp_check(a, example['query'], example['documents'], example['lexical'], tmp_path, 'toy')
    assert result['0.0']['original_shared_gradient_bit_exact']
    assert all(torch.equal(v, a.model.state_dict()[k]) for k, v in state.items())
    assert all(p.grad is None for p in a.model.parameters())


def cohort():
    ids = [f'{domain}-{i}' for i in range(128) for domain in canary.DOMAINS]
    selection = dict(pilot=ids, sentinel=['unused-' + q for q in ids])
    records = {q: dict(query_id=q, split='TRAIN', reference_update=0, domain=q.split('-')[0]) for q in ids}
    return selection, ids * 2, records


def test_fixed_first_four_per_domain_no_quality_reselection():
    selection, order, records = cohort()
    assert canary.select_canary(selection, order, records) == order[:12]
    order[:12] = list(reversed(order[:12]))
    assert canary.select_canary(selection, order, records) == order[:12]


@pytest.mark.parametrize('bad', ['duplicate', 'sentinel', 'locked', 'reference'])
def test_invalid_cohort_cannot_launch(bad):
    selection, order, records = cohort()
    q = order[0]
    if bad == 'duplicate':
        order[1] = q
    elif bad == 'sentinel':
        selection['sentinel'][0] = q
    elif bad == 'locked':
        records[q]['split'] = 'LOCKED_TEST'
    else:
        records[q]['reference_update'] = 96
    with pytest.raises(ValueError):
        canary.select_canary(selection, order, records)


def test_no_update_score_noise_independent_derivative_and_zero_control():
    example, anchors = fixture()
    scores = torch.tensor([2., 0.], dtype=torch.float32)
    r = canary.score_noise(scores, [2. + 1e-7, 0.], example['pairs'], anchors)
    assert r['keep_score_gradient_l1'] == 0 and r['zero_coefficient_bit_exact']
    anchors['anchors'][0]['baseline_margin'] += 1e-7
    r = canary.score_noise(scores, [2. + 1e-7, 0.], example['pairs'], anchors)
    assert 0 < r['keep_score_gradient_l1'] < r['max_score_error']


def test_support_change_is_not_hidden_by_small_weight():
    a = sparse.csr_matrix(np.array([[1., 1e-12, 0.]], dtype=np.float32))
    b = sparse.csr_matrix(np.array([[1., 0., 0.]], dtype=np.float32))
    result = canary.code_parity(a, b)
    assert not result['passed'] and not result['exact_support']
    assert result['total_support_differences'] == 1
    assert canary.code_parity(a, a)['passed']


def test_canary_has_no_optimizer_step_or_heldout_encoder_surface():
    tree = ast.parse(Path(canary.__file__).read_text())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not set(calls) & {'optimizer_step', 'make_optimizer', 'step', 'surface_queries', 'train_chunk'}
