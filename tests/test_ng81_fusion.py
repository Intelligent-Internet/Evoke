"""Small CPU contracts for the fixed NG81 fusion comparison."""

import json

import numpy as np
import pytest

import ng81_fusion as study


def test_serving_recovers_historical_weights_exactly():
    rng = np.random.default_rng(81)
    sem, lex = rng.random((2, 1024))
    np.testing.assert_array_equal(study.mix(sem, lex, .1, 'serving'), sem + lex)


@pytest.mark.parametrize('mode', study.NORMS)
def test_endpoints_preserve_channel_rank(mode):
    sem, lex = np.array([.2, .1, .3]), np.array([.1, .4, .2])
    for alpha, source in ((0, sem), (1, lex)):
        mixed = study.mix(sem, lex, alpha, mode)
        np.testing.assert_array_equal(np.argsort(mixed), np.argsort(source))


def test_zscore_positive_affine_invariance():
    sem, lex = np.array([.1, 4., .2]), np.array([7., 8., 2.])
    np.testing.assert_allclose(study.mix(sem, lex, .25, 'zscore'),
        study.mix(9 * sem + 1, 2 * lex - 3, .25, 'zscore'), rtol=0, atol=1e-14)


def test_constant_zscore_channel_is_zero():
    np.testing.assert_array_equal(study.mix([4., 4.], [1., 1.], .1, 'zscore'), [0., 0.])


@pytest.mark.parametrize('alpha', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        study.mix([1., 2.], [2., 1.], alpha, 'serving')


@pytest.mark.parametrize('sem,lex,mode', [
    ([1.], [2.], 'serving'), ([1., 2.], [2.], 'serving'),
    ([1., float('nan')], [2., 1.], 'zscore'),
    ([1., 2.], [2., 1.], 'rrf'),
])
def test_invalid_surface(sem, lex, mode):
    with pytest.raises(ValueError):
        study.mix(sem, lex, .1, mode)


def test_rank_ties_use_document_id_and_all_positives():
    score = np.array([2., 2., 0., 1.])
    result = study.ranked(score, [1, 2])
    assert result['gold_ranks'] == [2, 4]
    assert result['top100'] == [0, 1, 3, 2]
    assert result['recall100'] == 1


def test_full_fusion_can_find_winners_outside_top100_union():
    sem = np.r_[np.full(100, 10.), np.zeros(100), np.full(100, 6.)]
    lex = np.r_[np.zeros(100), np.full(100, 10.), np.full(100, 6.)]
    union = set(np.argsort(-sem)[:100]) | set(np.argsort(-lex)[:100])
    assert 250 not in union
    result = study.ranked(study.mix(.9 * sem, .1 * lex, .5, 'serving'), [250])
    assert result['gold_ranks'] == [51]
    assert result['recall100'] == 1


def test_recall_keeps_all_positive_denominators():
    assert study.metrics([1, 101])[1] == .5


def test_macro_equal_domains_not_query_weighted():
    domains = ['fever', 'fever', 'hotpotqa', 'nq']
    assert set(domains) == set(study.old.DOMAINS)
    values = np.array([[1., 1.], [1., 1.], [0., 0.], [0., 0.]])
    np.testing.assert_allclose(study.macro(values, domains), [1 / 3, 1 / 3])


def calibration_rows():
    rows = []
    for index in range(3):
        candidates = {}
        for mode in study.NORMS:
            candidates[mode] = [dict(alpha=a, gold_ranks=[1], metrics=[1., 1.])
                                for a in study.old.ALPHAS]
        rows.append(dict(query_index=index, candidates=candidates))
    return rows


def test_train_tie_prefers_less_lexical():
    selected = study.select(calibration_rows(), list(study.old.DOMAINS))
    assert all(row['alpha'] == 0 for row in selected.values())


def test_selection_rejects_tampered_metrics():
    rows = calibration_rows()
    rows[0]['candidates']['serving'][0]['metrics'][0] = .5
    with pytest.raises(AssertionError):
        study.select(rows, list(study.old.DOMAINS))


def test_selection_rejects_changed_grid():
    rows = calibration_rows()
    rows[0]['candidates']['zscore'][0]['alpha'] = .01
    with pytest.raises(ValueError):
        study.select(rows, list(study.old.DOMAINS))


@pytest.mark.parametrize('ranks', [[], [1, 1], [0], [1.5]])
def test_invalid_positive_ranks(ranks):
    with pytest.raises(ValueError):
        study.metrics(ranks)


def test_graph_is_evaluation_only():
    assert len(study.PHASES) == 8
    assert study.PHASES[0] == 'encode-calibration'
    assert study.PHASES[-1] == 'review'
    assert not any(p.startswith('train-') for p in study.PHASES)
    assert study.PARENT.endswith('sparse-endpoints-v3')


def test_freeze_refuses_existing_output(tmp_path):
    output = tmp_path / 'NG-0081/fusion-v1'
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match='fresh NG81'):
        study.freeze(tmp_path, output)


def test_freeze_binds_inputs_and_rejects_later_mutation(tmp_path, monkeypatch):
    parent = tmp_path / study.PARENT
    (parent / 'review').mkdir(parents=True)
    for name, value in [('inputs.json', {}), ('review/results.json', {}),
                        ('config.json', {'base': {'semantic_weight': .9}}),
                        ('selection.json', {'calibration': [1], 'diagnostic': [2]})]:
        (parent / name).write_text(json.dumps(value))
    monkeypatch.setattr(study, 'PARENT_SHA', study.campaign.io.sha(parent / 'inputs.json'))
    monkeypatch.setattr(study, 'REVIEW_SHA', study.campaign.io.sha(parent / 'review/results.json'))
    monkeypatch.setattr(study.old, 'verify', lambda *_: dict(dependencies={}, source={}))
    monkeypatch.setattr(study.audit, 'sealed', lambda *_: {})
    run = tmp_path / 'NG-0081/fusion-v1'
    study.freeze(tmp_path, run)
    manifest = study.verify(tmp_path, run)
    assert manifest['source']['ng81_fusion.py'] == study.campaign.io.sha(run / 'ng81_fusion.py')
    assert manifest['optimizer_updates'] == 0
    (parent / 'selection.json').write_text('{}')
    with pytest.raises(ValueError, match='changed input'):
        study.verify(tmp_path, run)


def test_verify_rejects_training_protocol(tmp_path):
    manifest = dict(protocol='NG81_frozen_endpoint_fusion_v1', phases=list(study.PHASES),
        alphas=list(study.old.ALPHAS), normalizations=list(study.NORMS),
        profiles=list(study.PROFILES), optimizer_updates=1, locked_test_access=False,
        independent_holdout=False)
    (tmp_path / 'inputs.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='NG81 protocol'):
        study.verify(tmp_path, tmp_path)
