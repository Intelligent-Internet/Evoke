"""Single-factor score operator contracts; CPU only."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng83_operator as study


def test_centered_field_equals_all_pair_margin_regression():
    error = torch.tensor([.1, -.3, 2., -1.], dtype=torch.float64)
    pairs = torch.tensor([[0, 1], [2, 3]])
    loss, _, _ = study.relation_loss(error, torch.zeros_like(error), pairs, 0., 1.)
    direct = .5 * ((error[:, None] - error[None, :]) ** 2).mean()
    torch.testing.assert_close(loss, direct)


@pytest.mark.parametrize('weight', [0., .5])
def test_margin_loss_shift_invariance_and_gradcheck(weight):
    source = torch.tensor([.2, .3, -.1, .8], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([.1, .2, .3, -.4], dtype=torch.float64)
    pairs = torch.tensor([[0, 1], [2, 3]])
    fn = lambda score: study.relation_loss(score, target, pairs, weight, .2)[0]
    assert torch.autograd.gradcheck(fn, (source,))
    torch.testing.assert_close(fn(source), fn(source + 13.))


def test_pair_weighting_changes_objective_not_candidate_universe():
    student, target = np.array([0., .1, .8, .9]), np.array([0., 0., 1., 1.])
    pairs = np.array([[0, 1], [2, 3]])
    uniform, field, boundary = study.relation_loss(student, target, pairs, 0., 1.)
    focused, field2, boundary2 = study.relation_loss(student, target, pairs, .5, 1.)
    assert field == field2 and boundary == boundary2
    assert focused == .5 * uniform + .5 * boundary
    assert focused != uniform


def test_candidate_bank_preserves_human_positives_background_and_pairs():
    order = np.arange(200)[::-1]
    bank, pairs = study.candidate_bank(order, [400, 401], [700, 400])
    assert {400, 401, 700} <= set(bank)
    assert len(bank) == len(set(bank))
    assert [[bank[p], bank[n]] for p, n in pairs] == [
        [order[p], order[n]] for p, n in study.prior.PAIRS]


def test_oversized_candidate_bank_rejected():
    with pytest.raises(ValueError):
        study.candidate_bank(np.arange(200), list(range(500)), [0])


def test_export_keeps_both_role_maps_and_centering(tmp_path):
    rng = np.random.default_rng(83)
    d, q = rng.normal(size=(30, 8)), rng.normal(size=8)
    a, b, mean = rng.normal(size=(8, 3)), rng.normal(size=(8, 3)), rng.normal(size=8)
    np.savez(tmp_path / 'maps.npz', mean=mean, query_map=a, document_map=b)
    saved = np.load(tmp_path / 'maps.npz')
    actual = ((d - saved['mean']) @ saved['document_map']) @ (q @ saved['query_map'])
    expected = (d - mean) @ (b @ a.T) @ q
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_matched_configuration_has_only_two_prespecified_weights():
    conf = study.config()
    assert conf['boundary_weights'] == {'field': 0., 'boundary': .5}
    assert conf['train_queries'] == conf['batch_queries'] * conf['updates']
    assert not conf['locked_test_access'] and not conf['automatic_promotion']


def test_config_mutation_rejected_before_reading_data(tmp_path):
    config = study.config()
    config['rank'] = 512
    study.campaign.io.write(tmp_path / 'inputs.json', {'config': config})
    with pytest.raises(ValueError, match='config changed'):
        study.verify(tmp_path, tmp_path)
