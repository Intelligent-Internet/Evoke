from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_training as training
import review_ng79_training as review
from test_ng79_preparation_review import row, config


def fixture():
    record = row()
    record.update(query_id='q', split='TRAIN', reference_update=0, pool=[0, 1, 2, 3],
                  corpus_size=233009, rank_scope='full_corpus_reference_snapshot')
    pairs = training.prepare_pairs(record, config())
    scores = torch.tensor([.7, .3, .9, .8], requires_grad=True)
    loss = training.pair_loss(scores, pairs)
    (loss / 4).backward()
    item = dict(query_id='q', scores=scores.detach().tolist(), score_gradient=scores.grad.tolist(),
        loss=float(loss.detach()), documents=4, vjp_replay_exact=True, objective='balanced_soft_pair',
        eligible_pairs=len(pairs['indices']), supervised_positives=pairs['supervised_positives'])
    return item, record, config()


def test_independent_scalar_matches_actual_torch_D_accumulation():
    assert review.audit_example(*fixture()) < 2e-7


@pytest.mark.parametrize('bad', ['loss', 'gradient', 'retention', 'reference', 'split', 'positive', 'vjp'])
def test_corrupt_actual_training_evidence_fails(bad):
    item, record, cfg = deepcopy(fixture())
    if bad == 'loss': item['loss'] += .1
    elif bad == 'gradient': item['score_gradient'][0] += .1
    elif bad == 'retention': item['objective_components'] = {'retention_coefficient': 1.}
    elif bad == 'reference': record['reference_update'] = 96
    elif bad == 'split': record['split'] = 'DEV_NEW'
    elif bad == 'positive': record['total_positives'] = 1
    else: item['vjp_replay_exact'] = False
    with pytest.raises((ValueError, AssertionError)):
        review.audit_example(item, record, cfg)


@pytest.mark.parametrize('bad', ['arm', 'step', 'model', 'optimizer', 'complete'])
def test_same_arm_optimizer_predecessor_checked_before_loss_reduction(bad):
    result = dict(arm='R', cumulative_steps=192, steps=96, query_exposures=384,
                  checkpoint_optimizer_replay_exact=True, optimizer_sha256='file-sha',
                  static_reference_update=0, gpu_observer=False, quality_evaluation=False,
                  dev_access=False, locked_test_access=False, initial_model_sha256='model',
                  initial_optimizer_fingerprint='moments', previous_complete_sha256='file-sha')
    previous = dict(arm='R', cumulative_steps=96, model_sha256='model', optimizer_fingerprint='moments')
    if bad == 'arm': previous['arm'] = 'B'
    elif bad == 'step': previous['cumulative_steps'] = 192
    elif bad == 'model': previous['model_sha256'] = 'different'
    elif bad == 'optimizer': previous['optimizer_fingerprint'] = 'reset'
    else: result['previous_complete_sha256'] = 'changed'
    with patch.object(review, 'sha', return_value='file-sha'), patch.object(review, 'rows') as records:
        with pytest.raises(ValueError, match='continuation broken'):
            review.audit_phase(Path('/unused/train-R-192'), result, {}, [], {},
                               (Path('/previous'), previous))
        records.assert_not_called()
