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
import ng71_execution as execution
import ng80_training as training
import review_ng80_sparse_fit as audit


@pytest.mark.parametrize('arm', ['S', 'P'])
@pytest.mark.parametrize('positive', [
    [True, False, False, False], [True, True, False, False],
    [True, True, True, True], [False, False, False, True],
])
def test_independent_scalar_and_derivative(arm, positive):
    scores = torch.tensor([3., .2, -.4, .05], requires_grad=True)
    teacher = [.4, .1, .2, -.3]
    scale = 9.667940425864051
    if arm == 'S':
        loss = training.field_loss(scores, scores.new_tensor(teacher), scale)
    else:
        loss = training.protective_loss(scores, scores.new_tensor(teacher),
                                        torch.tensor(positive))
    (loss / 4).backward()
    record = dict(scores=scores.detach().tolist(), teacher=teacher, documents=4,
        positive=positive, loss=loss.item(), score_gradient=scores.grad.tolist(),
        vjp_replay_exact=True)
    assert audit.scalar(record, teacher, positive, arm, scale) < 2e-8
    with pytest.raises(AssertionError):
        audit.scalar(dict(record, loss=loss.item() + .1), teacher, positive, arm, scale)
    with pytest.raises(AssertionError):
        audit.scalar(dict(record, score_gradient=[.5] * 4), teacher, positive, arm, scale)
    with pytest.raises(ValueError):
        audit.scalar(dict(record, vjp_replay_exact=False), teacher, positive, arm, scale)


def test_sigmoid_extremes_are_finite():
    np.testing.assert_allclose(audit.sigmoid([-1e4, 0, 1e4]), [0, .5, 1])


@pytest.mark.parametrize('margin', [8., 12., 16.])
def test_saturated_target_preserves_fp32_training_contract(margin):
    scores = torch.tensor([margin, 0.], requires_grad=True)
    teacher = [.76, .06]
    positive = [True, False]
    loss = training.protective_loss(scores, scores.new_tensor(teacher), torch.tensor(positive))
    (loss / 4).backward()
    record = dict(scores=scores.detach().tolist(), teacher=teacher, documents=2,
        positive=positive, loss=loss.item(), score_gradient=scores.grad.tolist(),
        vjp_replay_exact=True)
    assert audit.scalar(record, teacher, positive, 'P', 9.667940425864051) < 2e-8
    with pytest.raises(AssertionError):
        audit.scalar(dict(record, loss=record['loss'] + 1e-5), teacher, positive, 'P', 1.)


def test_scientific_audit_rejects_different_target_backend(tmp_path):
    with pytest.raises(ValueError, match='CUDA target arithmetic'):
        audit.review(tmp_path, probability_device='cpu')


class OrderedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z = torch.nn.Parameter(torch.ones(3))
        self.a = torch.nn.Parameter(torch.ones(2))


def test_serialized_optimizer_uses_registration_not_sorted_order(tmp_path):
    model = OrderedModel()
    optimizer = torch.optim.AdamW([{'params': [model.a]}, {'params': [model.z]}])
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    path = tmp_path / 'optimizer.pt'
    digest = execution.save_optimizer(path, SimpleNamespace(model=model), optimizer, 1)
    assert audit.optimizer_audit(path, 1, digest, model) == 2
    saved = torch.load(path, weights_only=True)
    saved['state']['state'][0]['exp_avg'].add_(1)
    bad = tmp_path / 'corrupt.pt'
    torch.save(saved, bad)
    with pytest.raises(ValueError, match='fingerprint'):
        audit.optimizer_audit(bad, 1, digest, model)
    with pytest.raises(ValueError, match='header'):
        audit.optimizer_audit(path, 2, digest, model)


def test_terminal_inputs_are_disjoint_from_canaries():
    assert set(audit.INPUTS) == {'S', 'P'}
    assert len(set(audit.INPUTS.values())) == 2
    assert audit.fit.PHASES == ('train-1024', 'train-2048', 'train-3072')
