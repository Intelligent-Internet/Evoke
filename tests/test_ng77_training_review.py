"""Independent logged-score audit, including accumulation and corruption."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_training as training
import ng77_retention as retention
import review_ng77_training as review
from test_ng77_retention import fixture


def example(coefficient):
    record, config = fixture()
    record.update(query_id='q', split='TRAIN')
    anchors = retention.prepare_anchors(record, config)
    scores = torch.tensor([2., 1., 4., 3., 0.], requires_grad=True)
    pairs = training.prepare_pairs(record, config)
    original = training.pair_loss(scores, pairs)
    total, keep = retention.combined_loss(original, scores, anchors, coefficient)
    (total / 4).backward()
    item = dict(query_id='q', scores=scores.detach().tolist(), score_gradient=scores.grad.tolist(),
        loss=float(total.detach()), documents=len(scores), vjp_replay_exact=True,
        objective='balanced_soft_pair', eligible_pairs=len(pairs['indices']),
        supervised_positives=pairs['supervised_positives'], objective_components=dict(
            original_loss=float(original.detach()), keep_loss=float(keep.detach()),
            retention_coefficient=coefficient))
    return item, record, anchors, config


@pytest.mark.parametrize('coefficient', [0., 1.])
def test_logged_D_plus_keep_matches_independent_decimal_numpy(coefficient):
    result = review.audit_example(*example(coefficient), coefficient)
    assert result['score_gradient_max_error'] < 2e-7
    assert result['keep_loss_error'] < 1e-13


@pytest.mark.parametrize('bad', ['gradient', 'loss', 'keep', 'coefficient', 'split', 'vjp', 'positive'])
def test_corrupt_scientific_receipts_rejected(bad):
    item, record, anchors, config = deepcopy(example(1.))
    if bad == 'gradient':
        item['score_gradient'] = (np.array(item['score_gradient']) * 4).tolist()
    elif bad == 'loss':
        item['loss'] += .1
    elif bad == 'keep':
        item['objective_components']['keep_loss'] += .1
    elif bad == 'coefficient':
        item['objective_components']['retention_coefficient'] = 0.
    elif bad == 'split':
        record['split'] = 'DEV_NEW'
    elif bad == 'vjp':
        item['vjp_replay_exact'] = False
    else:
        record['total_positives'] = 1
    with pytest.raises((ValueError, AssertionError)):
        review.audit_example(item, record, anchors, config, 1.)


@pytest.mark.parametrize('drop', [-1., 0., 1e-12, .00099, .00101, .5, 2.])
def test_retention_reference_small_drop_without_deadband(drop):
    record, config = fixture()
    anchors = retention.prepare_anchors(record, config)
    scores = torch.tensor(record['hybrid_scores'], dtype=torch.float64, requires_grad=True)
    shifted = scores - torch.tensor([drop, 0., 0., 0., 0.])
    loss = retention.retention_loss(shifted, anchors)
    loss.backward()
    scalar, gradient = review.keep_reference(shifted.detach().tolist(), anchors)
    assert scalar == pytest.approx(float(loss.detach()), abs=1e-13, rel=2e-10)
    np.testing.assert_allclose(gradient, scores.grad.numpy(), atol=1e-13, rtol=2e-10)


def test_decimal_reference_retains_high_confidence_gap_without_negative_cancellation():
    anchors = dict(candidate_count=2, student_temperature=1., anchors=[
        dict(indices=[0, 1], baseline_margin=1000., coefficient=.5)])
    loss, gradient = review.keep_reference([999., 0.], anchors)
    assert loss == 0. and np.isfinite(gradient).all()
