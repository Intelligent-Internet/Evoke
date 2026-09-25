"""Single-factor configuration, TRAIN boundaries and staged graph guards."""

import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng77_pilot as pilot


def test_only_two_copies_of_original_D_with_one_fixed_coefficient_difference():
    parent = dict(training_enabled=True, status='old',
                  arms=dict(D=dict(pool='witness', objective='balanced_soft_pair')),
                  optimizer=dict(lr=1., frozen=True))
    before = deepcopy(parent)
    result = pilot.configuration(parent)
    assert parent == before
    assert result['arms'] == dict(Z=parent['arms']['D'], K=parent['arms']['D'])
    assert result['optimizer'] == parent['optimizer']
    assert pilot.COEFFICIENTS == {'Z': 0., 'K': 1.}
    assert pilot.PHASES == ('train-Z-96', 'train-K-96')
    assert pilot.GATES['primary_surface'] == 'TRAIN_SENTINEL'
    assert not pilot.GATES['automatic_192_updates'] and not pilot.GATES['automatic_dev_access']
    parent['training_enabled'] = False
    with pytest.raises(ValueError):
        pilot.configuration(parent)


def cohort():
    ids = [f'q{i}' for i in range(384)]
    selection = dict(pilot=ids, sentinel=[f's{i}' for i in range(384)])
    records = {q: dict(total_positives=2 if i < 276 else 1,
        positive_ids=[1, 2] if i < 276 else [1], split='TRAIN', reference_update=0)
        for i, q in enumerate(ids)}
    endpoints = [dict(query_id=q, split='TRAIN') for q in ids + selection['sentinel']]
    return selection, ids * 2, records, endpoints


def test_all_660_positives_and_768_disjoint_train_endpoint_identities():
    pilot.validate_cohort(*cohort())


@pytest.mark.parametrize('bad', ['order', 'endpoint', 'split', 'positives', 'reference'])
def test_damaged_identity_contract_rejected(bad):
    selection, order, records, endpoints = cohort()
    if bad == 'order':
        order[0] = order[1]
    elif bad == 'endpoint':
        endpoints[-1] = endpoints[0]
    elif bad == 'split':
        endpoints[-1]['split'] = 'DEV_NEW'
    elif bad == 'positives':
        records[order[0]]['total_positives'] -= 1
    else:
        records[order[0]]['reference_update'] = 96
    with pytest.raises(ValueError):
        pilot.validate_cohort(selection, order, records, endpoints)


@pytest.mark.parametrize('phases', [[], ['x', 'x'], ['../x']])
def test_shared_supervisor_rejects_unsafe_graph_before_lock_or_process(phases):
    with pytest.raises(ValueError, match='safe, unique'):
        pilot.pipeline.supervise_graph(SimpleNamespace(), phases, 'ng77_pilot.py', limit_seconds=1800)


def test_this_stage_has_no_encoding_ranking_or_optimizer_observer():
    tree = ast.parse(Path(pilot.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    attributes = {n.func.attr for n in calls if isinstance(n.func, ast.Attribute)}
    assert not attributes & {'rank', 'encode_checkpoint', 'encode_reference'}
    assert all(k.arg != 'observer' for n in calls for k in n.keywords)
    assert pilot.GATES['bootstrap_seed'] == 71071 and pilot.GATES['bootstrap_replicates'] == 10000
