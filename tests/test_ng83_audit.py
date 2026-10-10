"""NG83 fixed-input and label-boundary contracts."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng83_audit as study


def test_joint_loss_is_actual_joint_margin_not_utility_sum():
    scores = np.array([2., 1., -.5])
    pair = dict(positive=[0, 0, 1, 1, 0, 1], negative=[1, 2, 2, 0, 1, 2])
    expected = np.mean([np.logaddexp(0, -(scores[p] - scores[n]) / .5)
                        for p, n in zip(pair['positive'], pair['negative'])])
    assert study.joint_loss(scores, pair, .5) == pytest.approx(expected)


@pytest.mark.parametrize('tau', [0., -1., np.nan])
def test_bad_temperature_rejected(tau):
    with pytest.raises(ValueError):
        study.joint_loss(np.ones(3), {}, tau)


def test_frozen_masks_must_be_boolean_and_include_unchanged_S():
    masks = {p: np.ones(4, dtype=bool) for p in study.PROFILES}
    assert study.mask_inventory(masks, 4)['S'].all()
    masks['S'][0] = False
    with pytest.raises(ValueError):
        study.mask_inventory(masks, 4)


def population():
    records, labels = [], []
    for domain in study.old.DOMAINS:
        for role in ('train', 'diagnostic'):
            records.append(dict(query_id=domain + role, subset=domain, query=domain + ' ' + role))
            labels.append([len(records) // 2])
    return records, labels, dict(train=[0, 2, 4], diagnostic=[1, 3, 5])


def test_duplicate_and_document_overlap_are_not_independent_holdout():
    records, labels, split = population()
    records[1]['query'] = '  FEVER   TRAIN '
    labels[1] = labels[0]
    _, result = study.supervision_inventory(records, labels, split)
    assert result['cross_split_duplicate_groups'] == 1
    assert result['positive_document_overlap_train_diagnostic'] >= 1
    assert not result['independent_holdout']
    assert not result['human_negative_labels_available']
    assert not result['first_token_is_intent_label']


def test_split_overlap_rejected():
    records, labels, split = population()
    split['diagnostic'].append(0)
    with pytest.raises(ValueError):
        study.supervision_inventory(records, labels, split)


@pytest.mark.parametrize('key,value', [('optimizer_updates', 1), ('mask_refits', 1),
    ('locked_test_access', True), ('profiles', ['S']), ('automatic_promotion', True)])
def test_changed_contract_rejected(tmp_path, key, value):
    data = study.contract()
    data[key] = value
    study.campaign.io.write(tmp_path / 'inputs.json', data)
    with pytest.raises(ValueError, match='NG83 contract'):
        study.verify(tmp_path, tmp_path)
