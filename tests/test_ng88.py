"""NG88 budgets, whole-background ranking, provenance and GPU admission."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

SOURCE = Path(__file__).parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).parent
sys.path.insert(0, str(SOURCE))
from ng85_data import DOMAINS, sha, write  # noqa: E402
import ng88_eval as execution  # noqa: E402
from ng88_math import (PROFILES, audit_scores, check_rank, normalize,  # noqa: E402
                       normalized_query, paired, prune, ranked, score)


def test_topk_positive_budget_ties_and_no_mutation():
    matrix = sparse.csr_matrix(np.array([[3., 3., 3., 1.], [0., 0., 0., 2.]], np.float32))
    actual = prune(matrix, 2)
    np.testing.assert_array_equal(actual.toarray(), [[3., 3., 0., 0.], [0., 0., 0., 2.]])
    assert matrix.nnz == 5 and prune(matrix, None) is matrix
    assert np.diff(prune(matrix, 384).indptr).tolist() == [4, 1]
    assert PROFILES['compact'] == (64, 384)


@pytest.mark.parametrize('k', [0, -1, True, 1.5])
def test_invalid_budget_rejected(k):
    with pytest.raises(ValueError, match='integer support'):
        prune(sparse.csr_matrix([[1.]]), k)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_explicit_zero_or_invalid_posting_rejected(value):
    matrix = sparse.csr_matrix(([value], [0], [0, 1]), shape=(1, 2))
    with pytest.raises(ValueError, match='positive finite'):
        prune(matrix, 1)


def test_query_mask_precedes_fixed_rms_normalization():
    raw = sparse.csr_matrix(np.array([[3., 2., 1.]], np.float32))
    result = normalized_query(raw, np.array([2., 10., 0.], np.float32), 1)
    np.testing.assert_allclose(result.toarray(), [[.45, 0., 0.]], rtol=1e-7)
    assert result[0, 0] != pytest.approx(.9 * 3 / 26)
    with pytest.raises(ValueError, match='masked fixed RMS'):
        normalized_query(raw, np.array([0., 10., 0.], np.float32), 1)
    with pytest.raises(ValueError, match='fixed RMS vector'):
        normalized_query(raw, np.array([-1., 10., 0.], np.float32), 1)


def test_literal_df_work_not_nnz_or_union_and_independent_dot():
    docs = sparse.csr_matrix([[2., 0.], [3., 4.], [5., 0.]], dtype=np.float32)
    query = sparse.csr_matrix([[.5, .25]], dtype=np.float32)
    values, work = score(docs.tocsc(), query)
    np.testing.assert_array_equal(values, [1., 2.5, 2.5])
    assert work == 4 and np.count_nonzero(values) == 3
    assert audit_scores(docs, query, values, [0, 1, 2]) == 0
    values[1] += .001
    with pytest.raises(ValueError, match='score mismatch'):
        audit_scores(docs, query, values, [1])
    # A support budget alone does not guarantee a small matching union.
    _, work = score(prune(docs, 1).tocsc(), sparse.csr_matrix([[1., 0.]]))
    assert work == 2


def test_full_universe_positive_ranks_and_ties():
    values = np.zeros(120)
    values[30] = 2
    record = ranked(values, [30, 0, 119])
    assert record['gold_ranks'] == [1, 2, 120]
    assert record['top100'][:3] == [30, 0, 1]
    assert record['recall100'] == pytest.approx(2 / 3)
    assert record['recall10'] == pytest.approx(2 / 3)
    check_rank(record, 120)
    for field, value in [('recall100', 1.), ('gold_ids', [30, 0, 120]),
                         ('gold_ranks', [1, 3, 120]),
                         ('top100_scores', [0.] * 100)]:
        changed = deepcopy(record)
        changed[field] = value
        with pytest.raises(ValueError):
            check_rank(changed, 120)


def test_full_rank_matches_independent_exhaustive_sort():
    rng = np.random.default_rng(88)
    for _ in range(20):
        values = rng.integers(0, 6, size=130).astype(float)
        positives = rng.choice(130, size=12, replace=False).tolist()
        expected = sorted(range(130), key=lambda i: (-values[i], i))
        record = ranked(values, positives)
        assert record['top100'] == expected[:100]
        assert record['gold_ranks'] == [expected.index(p) + 1 for p in positives]
        check_rank(record, 130)


@pytest.mark.parametrize('positives', [[], [0, 0], [-1], [3]])
def test_bad_positive_identity_rejected(positives):
    with pytest.raises(ValueError):
        ranked(np.array([1., 2., 3.]), positives)


def sample_panels():
    data = [dict(query_id=f'{split}/{domain}/{i}', split=split, subset=domain)
            for split, count in [('fit', 4096), ('validation', 512)]
            for domain in DOMAINS for i in range(count)]
    fit = [i for i, q in enumerate(data) if q['split'] == 'fit'
           and int(q['query_id'].split('/')[-1]) < 32]
    validation = [i for i, q in enumerate(data) if q['split'] == 'validation']
    return data, dict(endpoint_panel=fit + validation)


def test_panels_full_validation_train_only_calibration_no_locked_test():
    data, previous = sample_panels()
    selected = execution.panels(data, previous)
    assert len(selected['calibration']) == 128 and len(selected['validation']) == 2048
    assert all(data[i]['split'] == 'fit' for i in selected['calibration'])
    with pytest.raises(ValueError, match='sealed fit/validation'):
        execution.panels(data + [dict(split='locked_test', query_id='forbidden')], previous)
    data[0]['query_id'] = data[1]['query_id']
    with pytest.raises(ValueError, match='duplicate'):
        execution.panels(data, previous)


def test_gpu0_only_and_fixed_execution_graph():
    execution.gpu_contract(0, dict(gpu=0, uuid=execution.GPU_UUID))
    for gpu, snapshot in [(1, dict(gpu=1, uuid=execution.GPU_UUID)),
                          (0, dict(gpu=0, uuid='another-GPU'))]:
        with pytest.raises(ValueError, match='GPU 0 only'):
            execution.gpu_contract(gpu, snapshot)
    assert len(execution.PHASES) == len(set(execution.PHASES)) == 23
    assert execution.PHASES[-1] == 'review'
    assert not any(p.startswith('train') for p in execution.PHASES)
    assert execution.PHASE_CAP == 5400 and execution.CAMPAIGN_CAP == 86400


def test_paired_macro_domains_and_zero_variance_normalization():
    np.testing.assert_array_equal(normalize(np.ones(3)), np.zeros(3))
    domains = list(DOMAINS) * 4
    result = paired(np.zeros(16), np.full(16, .01), domains)
    assert result['delta'] == pytest.approx(.01)
    np.testing.assert_allclose(result['descriptive_interval_95'], [.01, .01])
    with pytest.raises(ValueError, match='domain missing'):
        paired([0], [1], ['fever'])


def test_manifest_rejects_modified_payload_or_qualification_flag(tmp_path):
    base, run = tmp_path / 'base', tmp_path / 'run'
    data, previous = sample_panels()
    (base / execution.DATA / 'prepare').mkdir(parents=True)
    (base / execution.PARENT).mkdir(parents=True)
    run.mkdir()
    import json
    query_path = base / execution.DATA / 'prepare/queries.jsonl'
    query_path.write_text(''.join(json.dumps(q) + '\n' for q in data))
    write(base / execution.PARENT / 'inputs.json', previous)
    write(run / 'selection.json', execution.panels(data, previous))
    manifest = dict(protocol='NG88_full_background_gpu0_v1', phases=list(execution.PHASES),
        gpu=0, gpu_uuid=execution.GPU_UUID, parent=execution.PARENT,
        parent_sha256=execution.PARENT_SHA, profiles=PROFILES, corpus=execution.CORPUS,
        shards=execution.SHARDS, phase_cap=execution.PHASE_CAP,
        campaign_cap=execution.CAMPAIGN_CAP, dense_alphas=execution.ALPHAS,
        source={'selection.json': sha(run / 'selection.json')}, dependencies={},
        training=False, locked_test_access=False, production_qualified=False,
        automatic_training=False, validation_independent=False,
        original_text_dense_evaluation='pending_separate_protocol')
    write(run / 'inputs.json', manifest)
    execution.verify(base, run)
    manifest['training'] = True
    write(run / 'inputs.json', manifest)
    with pytest.raises(ValueError, match='contract changed'):
        execution.verify(base, run)
    manifest['training'] = False
    write(run / 'inputs.json', manifest)
    write(run / 'selection.json', {})
    with pytest.raises(ValueError, match='changed input'):
        execution.verify(base, run)


@pytest.mark.skipif(sys.platform == 'darwin', reason='Torch execution on Lambda2 only')
def test_query_readout_preserves_frozen_microbatch_operation_order():
    import torch
    torch.manual_seed(88)
    raw = torch.rand(8, 50265)
    encoder = SimpleNamespace(rms=torch.rand(50265), config=dict(
        optimizer=dict(query_microbatch=1), base=dict(semantic_weight=.9)))
    expected = torch.cat([.9 * row / (row @ encoder.rms)[:, None]
                          for row in raw.split(1)])
    assert torch.equal(execution.query_readout(encoder, raw), expected)
    encoder.rms.zero_()
    with pytest.raises(ValueError, match='query normalization'):
        execution.query_readout(encoder, raw)
