import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import ng80_fit as fit
import ng80_training as training
import ng71_execution as execution


def test_fixed_order_is_balanced_disjoint_and_deterministic():
    rows = [{'query_id': f'{d}-{i}', 'subset': d}
            for d in fit.data.DOMAINS for i in range(4097)]
    diagnostic = [4096, 8193, 12290]
    selected = {'train': [i for i in range(len(rows)) if i not in diagnostic],
                'diagnostic': diagnostic}
    values = fit.order(rows, selected)
    assert values == fit.order(rows, selected)
    assert len(values) == len(set(values)) == 12288
    for start in range(0, len(values), 3):
        assert [rows[i]['subset'] for i in values[start:start + 3]] == list(fit.data.DOMAINS)


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = torch.nn.Linear(2, 2)
        self.lm_head = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace(hidden_size=2)

    @property
    def base_model(self):
        return self.roberta


@pytest.mark.parametrize('arm', ['F', 'D'])
def test_optimizer_holder_preserves_all_trainable_moments(tmp_path, arm):
    def make():
        encoder = SimpleNamespace(model=Base(), device='cpu', config={
            'optimizer': {'trunk_lr': 5e-6, 'head_lr': 2e-5, 'betas': [.9, .999],
                          'eps': 1e-8, 'weight_decay': .01}})
        model = training.Readout(encoder, arm, np.ones((3, 1024)))
        return model, model.optimizer()

    first, optimizer = make()
    sum(p.square().sum() for p in first.parameters()).backward()
    optimizer.step()
    digest = execution.save_optimizer(tmp_path / 'optimizer.pt', fit.holder(first), optimizer, 1)
    second, restored = make()
    assert execution.restore_optimizer(tmp_path / 'optimizer.pt', fit.holder(second), restored, 1) == digest
    assert {id(p) for p in fit.holder(second).model.parameters()} == {id(p) for p in second.parameters()}
