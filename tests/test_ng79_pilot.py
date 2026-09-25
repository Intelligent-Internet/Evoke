from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng79_breadth as preparation
import ng79_pilot as pilot


def core():
    path = (Path(__file__).resolve().parents[1]
            / 'docs/research-sae/reports/ng0001-ng0099/ng0071-ranking-config.json')
    if not path.exists():
        path = Path(__file__).resolve().parent / 'ng0071-ranking-config.json'
    return preparation.config_from_parent(json.loads(path.read_text()))


def test_eight_phases_require_both_fresh_prefixes_before_continuation():
    assert len(pilot.PHASES) == 8
    assert pilot.PHASES[:4] == ('train-R-96', 'train-B-96', 'train-R-192', 'train-B-192')
    assert pilot.PHASES[-2:] == ('train-R-384', 'train-B-384')
    assert not any('encode' in p or 'rank' in p for p in pilot.PHASES)


def test_arms_share_every_nonschedule_training_parameter():
    original = core()
    a, b = (pilot.configuration(original, arm) for arm in ('R', 'B'))
    for key in ('base', 'optimizer', 'ranking', 'arms', 'witness', 'gates', 'resources'):
        assert a[key] == b[key]
    assert a['pilot']['train_queries'] * a['pilot']['epochs'] == 1536
    assert b['pilot']['train_queries'] * b['pilot']['epochs'] == 1536
    assert a['pilot']['updates'] == b['pilot']['updates'] == 384
    assert a['witness']['refresh_updates'] == b['witness']['refresh_updates'] == [0]
    assert not original['training_enabled'] and a['training_enabled'] and b['training_enabled']


@pytest.mark.parametrize('bad', ['arm', 'loss', 'reference'])
def test_unfrozen_objective_is_rejected(bad):
    value, arm = core(), 'R'
    if bad == 'arm': arm = 'K'
    elif bad == 'loss': value['arms']['D']['objective'] = 'legacy_CE'
    else: value['witness']['refresh_updates'] = [0, 96]
    with pytest.raises(ValueError):
        pilot.configuration(value, arm)


@pytest.mark.parametrize('bad', ['model_sha256', 'optimizer_fingerprint', 'tokens',
                               'candidate_pairs', 'initial_optimizer_fingerprint'])
def test_prefix_gate_is_exact_and_requires_fresh_initialization(bad):
    result = dict(deepcopy(pilot.PREFIX), cumulative_steps=96, steps=96, query_exposures=384,
                  initial_optimizer_fingerprint=None)
    pilot.prefix_check(result)
    result[bad] = 'changed'
    with pytest.raises(ValueError):
        pilot.prefix_check(result)
