"""Synthetic failure classification; no model/corpus/locked-test access."""

import ast
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_ng76_replay as audit
import ng76_replay_control as control


def records(count=96):
    example = dict(query_id='q', query_tokens=4, document_tokens=16,
        documents=2, objective='balanced_soft_pair', eligible_pairs=1,
        supervised_positives=1, vjp_replay_exact=True, query_nnz=4,
        document_nnz=12, scores=[1., 2.], score_gradient=[-.1, .1], loss=.3)
    return [dict(step=i + 1, reference_update=0, gradient_before_clip=1., update_l2=.01,
                 examples=[dict(copy.deepcopy(example), query_id=f'{i}-{j}')
                           for j in range(4)]) for i in range(count)]


def endpoint():
    return dict(arm='D', cumulative_steps=96, initial_model_sha256='base',
        initial_optimizer_fingerprint=None, tokens={'query': 1536, 'document': 6144},
        candidate_pairs=768, query_exposures=384, model_sha256='model', optimizer_fingerprint='opt')


def test_audit_separates_tiny_numeric_drift_readout_and_identity():
    expected = records()
    actual = copy.deepcopy(expected[:80])
    actual[68]['update_l2'] += 1e-12
    actual[73]['examples'][0]['scores'][0] += 1e-6
    actual[79]['examples'][0]['document_nnz'] -= 1
    result = audit.audit(actual, expected)
    assert result['steps'] == 80 and result['exposures'] == 320
    assert result['numeric']['update_l2']['first_nonexact_step'] == 69
    assert result['numeric']['scores']['first_nonexact_step'] == 74
    assert result['numeric_tolerance_passed'] and not result['identity_differences']
    assert result['readout_differences'] == [dict(step=80, example=0, query_id='79-0',
        field='document_nnz', actual=11, historical=12)]
    assert not result['complete_trace'] and not result['historical_endpoint_verified']
    assert not result['causal_mechanism_identified'] and not result['overall_goal_qualified']


@pytest.mark.parametrize('mutation', ['empty', 'step', 'reference', 'count', 'shape', 'nan', 'boolean'])
def test_malformed_trace_is_not_reclassified_as_numeric_drift(mutation):
    expected = records(2)
    actual = copy.deepcopy(expected)
    if mutation == 'empty':
        actual = []
    elif mutation == 'step':
        actual[0]['step'] = 2
    elif mutation == 'reference':
        actual[0]['reference_update'] = 96
    elif mutation == 'count':
        actual[0]['examples'].pop()
    elif mutation == 'shape':
        actual[0]['examples'][0]['scores'].pop()
    elif mutation == 'nan':
        actual[0]['update_l2'] = float('nan')
    else:
        actual[0]['update_l2'] = True
    with pytest.raises(ValueError):
        audit.audit(actual, expected)


@pytest.mark.parametrize('mutation', ['model', 'optimizer', 'readout', 'identity', 'numeric'])
def test_completed_control_does_not_waive_historical_gate(mutation):
    expected = records()
    actual = copy.deepcopy(expected)
    result = endpoint()
    if mutation == 'model':
        result['model_sha256'] = 'different'
    elif mutation == 'optimizer':
        result['optimizer_fingerprint'] = 'different'
    elif mutation == 'readout':
        actual[-1]['examples'][0]['document_nnz'] -= 1
    elif mutation == 'identity':
        actual[-1]['examples'][0]['query_id'] = 'wrong'
    else:
        actual[-1]['update_l2'] += .1
    summary = control.summarize(result, endpoint(), actual, expected)
    assert summary['procedure_completed']
    assert not summary['historical_replay_qualified']
    assert not summary['quality_evaluation'] and not summary['overall_goal_qualified']


def test_parity_is_not_new_scientific_qualification():
    result = control.summarize(endpoint(), endpoint(), records(), records())
    assert result['historical_replay_qualified']
    assert result['new_scientific_training_updates'] == 0
    assert not result['overall_goal_qualified']
    with pytest.raises(ValueError, match='incomplete'):
        control.summarize(endpoint(), endpoint(), records(80), records())


def test_control_calls_existing_training_loop_with_observer_explicitly_none():
    tree = ast.parse(Path(control.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'train_chunk']
    assert len(calls) == 1
    hook = next(k.value for k in calls[0].keywords if k.arg == 'observer')
    assert isinstance(hook, ast.Constant) and hook.value is None
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('use_deterministic_algorithms', 'sdpa_kernel') for n in ast.walk(tree))
