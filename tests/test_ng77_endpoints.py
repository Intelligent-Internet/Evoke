"""Endpoint graph barriers, unchanged gates and verified document-only reuse."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng77_endpoints as endpoint
import review_ng77_endpoints as review


def comparisons():
    good = dict(macro=dict(ndcg10=.006, recall100=0.),
        ci95=dict(ndcg10=[.001, .01], recall100=[-.001, .001]),
        domains={d: dict(ndcg10=0., recall100=0.) for d in review.metrics.DOMAINS})
    return {'K-96-minus-Z-96': deepcopy(good), 'K-96-minus-initial': deepcopy(good)}


def test_good_exposed_train_gate_does_not_authorize_scale_or_claim_holdout():
    result = review.advancement(comparisons(), 1., 1., endpoint.training.GATES)
    assert result['exploratory_gate_passed']
    assert not any(result[k] for k in ('automatic_scale_authorized', 'overall_goal_qualified',
                                      'native_cost_evaluated', 'independent_holdout'))


@pytest.mark.parametrize('bad', ['mean', 'ndcg_ci', 'recall_ci', 'domain_z', 'domain_initial',
                                 'domain_recall_z', 'domain_recall_initial', 'doc', 'df'])
def test_every_preregistered_gate_blocks_advancement(bad):
    rows, doc, df = comparisons(), 1., 1.
    z, initial = rows.values()
    if bad == 'mean':
        z['macro']['ndcg10'] = .0049
    elif bad == 'ndcg_ci':
        z['ci95']['ndcg10'][0] = 0.
    elif bad == 'recall_ci':
        z['ci95']['recall100'][0] = -.0021
    elif bad.startswith('domain_recall'):
        (z if bad.endswith('_z') else initial)['domains']['hotpotqa']['recall100'] = -.0051
    elif bad.startswith('domain'):
        (z if bad == 'domain_z' else initial)['domains']['nq']['ndcg10'] = -.0051
    elif bad == 'doc':
        doc = 1.251
    else:
        df = 1.251
    assert not review.advancement(rows, doc, df, endpoint.training.GATES)['exploratory_gate_passed']


def test_fixed_graph_contains_no_training_or_dev():
    assert endpoint.PHASES[:3] == ('audit-training', 'encode-Z-96', 'encode-K-96')
    assert len(endpoint.PHASES) == 9 and endpoint.PHASES[-1] == 'review'
    assert not any(p.startswith('train-') for p in endpoint.PHASES)


def test_external_training_barrier_precedes_ranking_reads():
    with patch.object(endpoint.observation, 'training_closed', side_effect=ValueError('not closed')) as gate:
        with pytest.raises(ValueError, match='not closed'):
            endpoint.observation.rank(None, None, Path('/endpoint'), None, 'initial',
                training_run=Path('/training'), training_phases=endpoint.training.PHASES, train_only=True)
        gate.assert_called_once_with(Path('/training'), endpoint.training.PHASES)


def test_dispatch_training_audit_failure_prevents_any_encoding(tmp_path):
    with patch.object(endpoint.execution, 'read', return_value={}), \
            patch.object(endpoint.observation, 'training_closed'), \
            patch.object(endpoint.execution, 'sealed', return_value={'passed': False}), \
            patch.object(endpoint.execution, 'encode_checkpoint') as encode:
        with pytest.raises(ValueError, match='audit failed'):
            endpoint.dispatch(tmp_path, tmp_path, 'encode-Z-96')
        encode.assert_not_called()


@pytest.mark.parametrize('mismatch', [False, True])
def test_document_reuse_never_reuses_query_codes(tmp_path, mismatch):
    output, cache, checkpoint = [tmp_path / n for n in ('output', 'cache', 'checkpoint')]
    output.mkdir()
    cache.mkdir()
    sparse.save_npz(cache / 'document.npz', sparse.csr_matrix(np.ones((3, 4), dtype=np.float32)))
    (cache / 'complete.json').write_text('{}')
    checkpoint.mkdir()
    (checkpoint / 'complete.json').write_text('{}')
    prior = dict(arm='Z', cumulative_steps=96, model_sha256='model')
    cached = dict(model_sha256='bad' if mismatch else 'model', counts=dict(document=dict(rows=3, nnz=12)))
    calls = []
    encoder = SimpleNamespace(model=None, encode=lambda texts, role:
        calls.append((role, list(texts))) or torch.ones((len(texts), 4), dtype=torch.float32))
    queries = [dict(query_id='q1', query='one', split='TRAIN'), dict(query_id='q2', query='two', split='TRAIN')]
    with patch.object(endpoint.execution, 'sealed', side_effect=lambda p: cached if p == cache else prior), \
            patch.object(endpoint.execution.training, 'Encoder', return_value=encoder) as constructor, \
            patch.object(endpoint.execution.training, 'parameter_hash', return_value='model'), \
            patch.object(torch.cuda, 'max_memory_allocated', return_value=0):
        if mismatch:
            with pytest.raises(ValueError, match='does not match'):
                endpoint.execution.encode_checkpoint(None, dict(arms=['Z'], pilot=dict(corpus_documents=3)),
                    output, checkpoint, queries, [{'text': 'd'}] * 3, document_cache=cache)
            constructor.assert_not_called()
        else:
            result = endpoint.execution.encode_checkpoint(None, dict(arms=['Z'], pilot=dict(corpus_documents=3)),
                output, checkpoint, queries, [{'text': 'd'}] * 3, document_cache=cache)
            assert calls == [('query', ['one', 'two'])]
            assert result['counts']['query']['rows'] == 2
            assert endpoint.execution.read(output / 'query-ids.json') == ['q1', 'q2']
            assert (output / 'document.npz').read_bytes() == (cache / 'document.npz').read_bytes()


def test_independent_query_df_and_bytes_audit_rejects_tampered_counters():
    query = sparse.csr_matrix(np.ones((768, 4), dtype=np.float32))
    document = sparse.csr_matrix((np.ones(3, dtype=np.float32), ([0, 1, 2], [0, 1, 1])),
                                  shape=(233009, 4))
    queries = [dict(surface='TRAIN_PILOT' if i < 384 else 'TRAIN_SENTINEL') for i in range(768)]
    counts = endpoint.observation.sparse_counts(query, document, queries)
    with patch.object(review.sparse, 'load_npz', side_effect=lambda p:
                      query if p.name == 'query.npz' else document):
        review.audit_counts(None, Path('/r'), 'K-96', queries, counts)
        counts['query_df_by_surface']['TRAIN_SENTINEL']['mean'] += 1
        with pytest.raises(ValueError, match='query-DF'):
            review.audit_counts(None, Path('/r'), 'K-96', queries, counts)


def test_coordinate_product_review_preserves_duplicate_order_and_float64():
    query = sparse.csr_matrix(np.array([[1e-5, 2., 0., .125]], dtype=np.float32))
    documents = sparse.csr_matrix(np.array([[3., 4., 5., 8.], [7., 0., 3., 16.]], dtype=np.float32))
    got = review.coordinate_scores(query, documents, [1, 0, 1])
    expected = np.array([np.sum(query.toarray().astype(np.float64)[0]
                               * documents.toarray()[i].astype(np.float64)) for i in (1, 0, 1)])
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.float64


def test_head_review_rejects_non_train_before_reading_any_codes():
    query = dict(query_id='q', lexical_index=0, split='LOCKED_TEST')
    with patch.object(review, 'read', return_value=[query] * 768), \
            patch.object(review.sparse, 'load_npz') as load:
        with pytest.raises(ValueError, match='frozen TRAIN'):
            review.check_head_scores(Path('/b'), Path('/r'), lambda: None)
        load.assert_not_called()


def test_mirror_review_rejects_wrong_inventory_before_loading_results():
    with patch.object(review, 'sha', return_value='wrong'), \
            patch.object(review, 'read') as load, \
            patch.object(review.psutil, 'virtual_memory', return_value=SimpleNamespace(available=32 * 1024 ** 3)):
        with pytest.raises(ValueError, match='mirror inventory'):
            review.review_mirror(Path('/b'), Path('/r'))
        load.assert_not_called()


def test_cached_head_score_tampering_is_rejected():
    queries = [dict(query_id=str(i), lexical_index=i, split='TRAIN') for i in range(768)]
    codes = sparse.csr_matrix(np.ones((768, 2), dtype=np.float32))
    records = [dict(query_id=str(i), split='TRAIN', top100=list(range(100)),
                    top100_scores=[4.] * 100) for i in range(768)]
    records[0]['top100_scores'][0] = 4.001
    with patch.object(review, 'read', return_value=queries), \
            patch.object(review, 'rows', return_value=records), \
            patch.object(review.sparse, 'load_npz', return_value=codes):
        with pytest.raises(AssertionError):
            review.check_head_scores(Path('/b'), Path('/r'), lambda: None)
