"""NG85 actual score-gradient tests; execute on the isolated Linux runtime."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).parent
sys.path.insert(0, str(SOURCE))
from ng85_train import calibrated_loss, example  # noqa: E402
from ng85_data import target_distribution  # noqa: E402


def test_actual_gradient_is_calibrated_teacher_kl():
    target = target_distribution([.2, .5, .9])
    scores = torch.tensor([.4, .6, .8], requires_grad=True)
    scale = scores.detach().std(correction=0)
    expected = (torch.softmax(scores.detach() / scale, dim=0)
                - torch.tensor(target, dtype=torch.float32)) / scale
    loss = calibrated_loss(scores, target)
    loss.backward()
    torch.testing.assert_close(scores.grad, expected)


def test_uniform_and_invalid_inputs():
    value = torch.ones(3, requires_grad=True)
    calibrated_loss(value, np.full(3, 1 / 3)).backward()
    assert torch.isfinite(value.grad).all()
    with pytest.raises(ValueError):
        calibrated_loss(value, [1., 1., 1.])


def test_ab_share_documents_lexical_but_not_teacher():
    queries = [dict(query='q', positives=[1])]
    documents = [dict(text='one'), dict(text='two')]
    bank = dict(offsets=np.array([0, 2]), document_ids=np.array([0, 1]),
                pplx=np.array([.9, .1]), lexical=np.array([.1, .2]))
    a = example(0, 'A', queries, documents, bank, {0: np.array([.1, .9])})
    b = example(0, 'B', queries, documents, bank, {0: np.array([.1, .9])})
    assert a['query'] == b['query'] and a['documents'] == b['documents']
    np.testing.assert_array_equal(a['lexical'], b['lexical'])
    assert a['target'][0] > a['target'][1]
    assert b['target'][0] < b['target'][1]
