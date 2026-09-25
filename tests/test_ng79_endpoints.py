from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_execution as execution
import ng71_observation as observation
import ng79_endpoints as endpoint


def surface():
    rows, cohorts = [], []
    for count in (128, 128, 384):
        ids = []
        for domain in ('fever', 'hotpotqa', 'nq'):
            for _ in range(count):
                identity = str(len(rows))
                rows.append(dict(query_id=identity, subset=domain, split='TRAIN', query='text' + identity))
                ids.append(identity)
        cohorts.append(ids)
    while len(rows) < 7872:
        rows.append(dict(query_id=str(len(rows)), subset='nq', split='LOCKED_TEST', query='not selected'))
    pilot, sentinel, added = cohorts
    selection = dict(repeated=pilot, sentinel=sentinel, breadth=pilot + added)
    return rows, selection, endpoint.surface_queries(rows, selection)


def config():
    return dict(pilot=dict(updates=384, corpus_documents=233009), arms=dict(R={}, B={}))


def prior():
    return dict(arm='B', cumulative_steps=384, steps=96, query_exposures=384)


def test_endpoint_is_balanced_train_only_and_preserves_old768_batch_context():
    rows, selection, queries = surface()
    assert Counter(q['surface'] for q in queries) == {
        'TRAIN_PILOT': 384, 'TRAIN_SENTINEL': 384, 'TRAIN_ADDED': 1152}
    old = dict(pilot=selection['repeated'], sentinel=selection['sentinel'])
    with patch.object(execution, 'read', return_value=rows):
        assert queries[:768] == observation.surface_queries(Path('/unused'), old, True, include_dev=False)


@pytest.mark.parametrize('bad', ['count', 'overlap', 'split', 'domain'])
def test_endpoint_selection_fails_closed(bad):
    rows, selection, _ = surface()
    if bad == 'count': selection['breadth'].pop()
    elif bad == 'overlap': selection['sentinel'][0] = selection['repeated'][0]
    elif bad == 'split': rows[0]['split'] = 'DEV_NEW'
    else: rows[0]['subset'] = 'other'
    with pytest.raises(ValueError):
        endpoint.surface_queries(rows, selection)


def test_fixed_encoder_uses_shared_engine_but_forbids_document_cache():
    _, _, queries = surface()
    with patch.object(execution, 'sealed', return_value=prior()), \
            patch.object(execution, '_encode_validated_checkpoint', return_value='ok') as engine:
        assert execution.encode_fixed_terminal_checkpoint(None, config(), None, None, queries,
                                                          [None] * 233009) == 'ok'
    assert engine.call_args.kwargs == dict(document_cache=None, stage_limit_seconds=1800)


@pytest.mark.parametrize('bad', ['step', 'split', 'surface', 'duplicates', 'domain', 'corpus', 'limit'])
def test_fixed_encoding_rejects_unapproved_scope_before_loading_model(bad):
    _, _, queries = surface()
    p, docs, limit = prior(), [None] * 233009, 1800
    if bad == 'step': p['cumulative_steps'] = 192
    elif bad == 'split': queries[0]['split'] = 'LOCKED_TEST'
    elif bad == 'surface': queries[0]['surface'] = 'DEV_NEW'
    elif bad == 'duplicates': queries[0]['query_id'] = queries[1]['query_id']
    elif bad == 'domain': queries[0]['subset'] = 'other'
    elif bad == 'corpus': docs.pop()
    else: limit = 5400
    with patch.object(execution, 'sealed', return_value=p), \
            patch.object(execution, '_encode_validated_checkpoint') as engine:
        with pytest.raises(ValueError):
            execution.encode_fixed_terminal_checkpoint(None, config(), None, None, queries,
                                                        docs, stage_limit_seconds=limit)
        engine.assert_not_called()


def test_legacy_encoder_still_rejects384():
    _, _, queries = surface()
    with patch.object(execution, 'sealed', return_value=prior()), \
            patch.object(execution, '_encode_validated_checkpoint') as engine:
        with pytest.raises(ValueError, match='unapproved checkpoint'):
            execution.encode_checkpoint(None, config(), None, None, queries, [None] * 233009)
        engine.assert_not_called()


def test_fixed_rank_checks_entire_training_graph_before_shared_ranking():
    _, _, queries = surface()
    with patch.object(observation, 'training_closed') as barrier, \
            patch.object(observation, '_rank_surface', return_value='ok') as engine:
        assert observation.rank_fixed_surface(None, config(), None, None, 'B-384', queries,
                                               '/train', witness_source='/witness') == 'ok'
    assert barrier.call_args.args[1] == list(endpoint.training.PHASES)
    assert engine.call_args.args[4:6] == ('B-384', queries)


@pytest.mark.parametrize('model', ['K-96', 'B-192', 'DEV_NEW'])
def test_fixed_rank_rejects_unplanned_model(model):
    _, _, queries = surface()
    with patch.object(observation, 'training_closed'), patch.object(observation, '_rank_surface') as engine:
        with pytest.raises(ValueError):
            observation.rank_fixed_surface(None, config(), None, None, model, queries,
                                           None, witness_source=None)
        engine.assert_not_called()


def test_graph_only_observes_terminal_and_does_not_train_or_select():
    assert endpoint.PHASES[:3] == ('audit-training', 'encode-R-384', 'encode-B-384')
    assert endpoint.PHASES[-1] == 'review'
    assert not any(p.startswith('train-') or '192' in p or '96' in p for p in endpoint.PHASES)
