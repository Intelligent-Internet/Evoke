import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import ng80_endpoints as endpoint
import review_ng80_fit as audit
import ng80_training as training
import ng71_execution as execution
from types import SimpleNamespace


def test_scalar_matches_independent_torch_derivative_and_detects_mutations():
    scores = torch.tensor([.3, .2, -.4], requires_grad=True)
    teacher = np.asarray([.4, .1, -.2], dtype=np.float64)
    loss = training.field_loss(scores, scores.new_tensor(teacher))
    (loss / 4).backward()
    record = dict(scores=scores.detach().tolist(), teacher=teacher.tolist(),
        documents=3, loss=loss.item(), score_gradient=scores.grad.tolist(),
        positive=[True, False, False], vjp_replay_exact=True)
    assert audit.scalar(record, teacher, record['positive']) < 1e-8
    with pytest.raises(AssertionError):
        audit.scalar(dict(record, score_gradient=[0., 0., 0.]), teacher, record['positive'])
    with pytest.raises(ValueError):
        audit.scalar(record, teacher, [False, True, False])


def test_optimizer_independent_fingerprint_and_corruption(tmp_path):
    model = torch.nn.ParameterDict({'z': torch.nn.Parameter(torch.ones(3)),
                                   'a': torch.nn.Parameter(torch.ones(2))})
    optimizer = torch.optim.AdamW([{'params': [model['z']]}, {'params': [model['a']]}])
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    path = tmp_path / 'optimizer.pt'
    digest = execution.save_optimizer(path, SimpleNamespace(model=model), optimizer, 1)
    assert audit.optimizer_audit(path, 1, digest) == 2
    saved = torch.load(path, weights_only=True)
    saved['state']['state'][0]['exp_avg'].add_(1)
    corrupt = tmp_path / 'corrupt.pt'
    torch.save(saved, corrupt)
    with pytest.raises(ValueError, match='fingerprint'):
        audit.optimizer_audit(corrupt, 1, digest)


def test_full_background_all_positive_metrics_and_stable_ties():
    scores = np.zeros(150)
    row = endpoint.rank_row(scores, scores.copy(), [0, 10, 149])
    assert row['top100'] == list(range(100))
    assert row['gold_ranks'] == [1, 11, 150]
    assert row['recall100'] == 2 / 3
    assert row['ndcg10'] == pytest.approx(1 / (1 + 1 / np.log2(3) + .5))
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize('gold', [[], [1, 1], [-1], [150]])
def test_rejects_missing_or_invalid_gold(gold):
    with pytest.raises(ValueError):
        endpoint.rank_row(np.zeros(150), np.zeros(150), gold)


def test_rejects_nonfinite_and_different_reductions():
    with pytest.raises(ValueError):
        endpoint.rank_row(np.zeros(150), np.ones(150), [0])
    with pytest.raises(ValueError):
        endpoint.rank_row(np.full(150, np.nan), np.zeros(150), [0])


def test_fixed_paired_domain_bootstrap():
    domains = np.repeat(endpoint.DOMAINS, 2)
    values = np.tile([.01, -.02], (6, 1))
    result = endpoint.paired(values, domains)
    assert result == endpoint.paired(values, domains)
    np.testing.assert_allclose(result['macro'], [.01, -.02])
    np.testing.assert_allclose(result['ci95'], [[.01, .01], [-.02, -.02]])
    assert result['independent_holdout'] is False
    with pytest.raises(ValueError):
        endpoint.paired(values[:2], domains[:2])


def test_endpoint_has_no_training_and_fixed_lanes():
    assert endpoint.MODELS == ('teacher', 'initial', 'F', 'D')
    assert len(endpoint.PHASES) == len(set(endpoint.PHASES)) == 9
    assert endpoint.LANES['audit'] == ('audit-training',)
    assert endpoint.LANES['adapted'] == ('encode-D',)
    assert endpoint.LANES['ranking'][-1] == 'review'
