"""Independent field derivatives and generic current-document VJP replay."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/ng80_training.py'
if not SOURCE.exists():
    SOURCE = Path(__file__).with_name('ng80_training.py')
spec = importlib.util.spec_from_file_location('ng80_training', SOURCE)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


@pytest.mark.parametrize('scale', [.1, 1., 7.])
def test_field_gradient(scale):
    scores = torch.tensor([.1, .3, .7, .9], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([.2, .4, .1, .9], dtype=torch.float64)
    loss = training.field_loss(scores, teacher, scale)
    loss.backward()
    error = scores.detach().numpy() / scale - teacher.numpy()
    expected = 2 * (error - error.mean()) / len(error) / scale
    np.testing.assert_allclose(scores.grad.numpy(), expected, atol=1e-12)


def test_protective_retains_all_positive_denominators_and_masks_conflicts():
    scores = torch.tensor([.1, .2, .3], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([.9, .1, .3], dtype=torch.float64)
    loss = training.protective_loss(scores, teacher, torch.tensor([True, True, False]))
    loss.backward()
    assert scores.grad[1] == 0
    assert scores.grad[0] < 0 and scores.grad[2] > 0
    target = torch.sigmoid((teacher[0] - teacher[2]) / .04)
    expected = (torch.sigmoid(scores.detach()[0] - scores.detach()[2]) - target) * (2 * target - 1) / 2
    assert float(scores.grad[0]) == pytest.approx(float(expected))


class Toy:
    kind = 'D'

    def __init__(self):
        self.weight = torch.tensor([[.1, .2], [.5, .3]], dtype=torch.float64, requires_grad=True)

    def encode(self, rows, role):
        return torch.as_tensor(rows, dtype=torch.float64) @ self.weight


def test_replay_matches_full_graph_parameter_gradient():
    a, b = Toy(), Toy()
    args = dict(query=[.1, .8], documents=[[.2, .7], [.5, .4], [.1, .2], [.4, .9], [.8, .1]],
                teacher=[.1, .4, .8, .2, .3], positive=[True, False, False, False, False],
                lexical=[0.] * 5, scale=1.)
    result = training.backward(a, **args, replay=True)
    reference = training.backward(b, **args, replay=False)
    torch.testing.assert_close(a.weight.grad, b.weight.grad, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result['score_gradient'], reference['score_gradient'], atol=1e-12)


@pytest.mark.parametrize('scale', [0, -1, float('nan')])
def test_reject_invalid_field_scale(scale):
    with pytest.raises(ValueError):
        training.field_loss(torch.ones(2), torch.ones(2), scale)


def test_all_positive_panel_has_zero_protective_gradient():
    scores = torch.tensor([.1, .2], requires_grad=True)
    loss = training.protective_loss(scores, torch.tensor([.9, .1]),
                                    torch.tensor([True, True]))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(scores.grad, torch.zeros_like(scores))


@pytest.mark.parametrize('bad', [0, -1, 1.5, True])
def test_backward_rejects_invalid_accumulation(bad):
    with pytest.raises(ValueError, match='accumulation'):
        training.backward(Toy(), [1., 2.], [[1., 2.], [2., 1.]],
                          [.1, .2], [True, False], [0., 0.], 1.,
                          accumulation=bad)


class ToyBase(torch.nn.Module):
    def __init__(self, tied):
        super().__init__()
        self.roberta = torch.nn.Linear(2, 2)
        self.lm_head = torch.nn.Linear(2, 2)
        if tied:
            self.lm_head.weight = self.roberta.weight
        self.config = SimpleNamespace(hidden_size=2)

    @property
    def base_model(self):
        return self.roberta


@pytest.mark.parametrize('kind', ['F', 'D', 'S', 'P'])
@pytest.mark.parametrize('tied', [False, True])
def test_optimizer_covers_trainable_parameters_once(kind, tied):
    encoder = SimpleNamespace(model=ToyBase(tied), device='cpu', config={
        'optimizer': {'trunk_lr': 5e-6, 'head_lr': 2e-5, 'betas': [.9, .999],
                      'eps': 1e-8, 'weight_decay': .01}})
    model = training.Readout(encoder, kind, np.zeros((3, 1024)))
    optimizer = model.optimizer()
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters()}
    trunk = list(encoder.model.base_model.parameters())
    assert all(p.requires_grad == (kind != 'F') for p in trunk)
