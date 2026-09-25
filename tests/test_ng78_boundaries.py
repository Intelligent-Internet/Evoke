"""Boundary accounting is descriptive; positive DCG uses full-corpus ranks."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng78_boundaries as audit
import review_ng78_boundaries as reference


def record():
    return dict(query_id='q', domain='nq', surface='TRAIN_PILOT',
        document_ids=[0, 1, 2, 3, 4], gold_ids=[0, 4],
        origins=['original_positive', 'added_vs_original'],
        scores={'initial': [5., 3., 2., 1., 4.], 'Z-96': [1., 3., 2., 0., 4.],
                'K-96': [2., 1., 3., 0., 4.]},
        heads={'initial': [0, 4, 1, 2, 3], 'Z-96': [4, 1, 2, 0, 3],
               'K-96': [4, 2, 0, 1, 3]},
        gold_ranks={'initial': [1, 2], 'Z-96': [4, 1], 'K-96': [3, 1]},
        pool=[0, 1, 4], anchors=[[0, 1]], teacher_scores=[5., 1., None, None, 4.],
        judged_negative_ids=[])


def test_repaired_persistent_new_are_sets_not_a_net_count():
    row = audit.reductions(record())[0]
    assert row['transition_counts']['10'] == dict(persistent=1, repaired=1, new=0)
    assert row['strict_anchor_loss'] == dict(persistent=0, repaired=1, new=0)
    assert row['coverage_teacher_counts']['10'] == {
        'persistent': {'outside_pool/unobserved': 1},
        'repaired': {'trusted_anchor/agrees': 1}, 'new': {}}
    raw = record()
    raw['scores']['K-96'][3] = 6.
    raw['heads']['K-96'] = [3, 4, 2, 0, 1]
    got = audit.reductions(raw)[0]
    assert got['transition_counts']['10'] == dict(persistent=1, repaired=1, new=1)


def test_strict_margin_tie_can_preserve_id_order():
    raw = record()
    raw['scores']['K-96'][1] = 2.
    row = audit.reductions(raw)[0]
    assert row['strict_anchor_loss'] == dict(persistent=1, repaired=0, new=0)
    assert 1 not in row['boundary_losses']['10']['K-96']


def test_gold_ranks_are_not_inferred_from_small_union():
    raw = record()
    raw['gold_ranks']['K-96'] = [101, 1]
    rows = audit.reductions(raw)
    assert rows[0]['recall_contribution']['K-96'] == 0.
    assert rows[1]['recall_contribution']['K-96'] == .5
    assert rows[0]['dcg_contribution']['K-96'] == 0.
    assert sum(r['dcg_contribution']['initial'] for r in rows) == 1.


def test_positive_positive_reordering_is_not_negative_crossing():
    raw = record()
    raw['scores']['Z-96'] = raw['scores']['K-96'] = [4., 3., 2., 1., 5.]
    raw['heads']['Z-96'] = raw['heads']['K-96'] = [4, 0, 1, 2, 3]
    raw['gold_ranks']['Z-96'] = raw['gold_ranks']['K-96'] = [2, 1]
    rows = audit.reductions(raw)
    assert rows[0]['ndcg_loss_buckets']['K-96'] == 'no_nongold_boundary_crossing'
    assert sum(r['dcg_contribution']['K-96'] for r in rows) == 1.


@pytest.mark.parametrize('scores,judged,status', [
    ({0: 1., 1: 0.}, False, 'agrees'), ({0: 0., 1: 1.}, False, 'opposes'),
    ({0: 1., 1: 1.}, False, 'tie'), ({0: 1.}, False, 'unobserved'),
    ({}, True, 'judged_negative')])
def test_unknown_or_conflicting_teacher_is_not_labeled_negative(scores, judged, status):
    assert audit.teacher_status(0, 1, scores, judged) == status


@pytest.mark.parametrize('rank', [0, -1, 1.5, True])
def test_bad_rank_is_rejected(rank):
    with pytest.raises(ValueError):
        audit.discount(rank)


def test_all_three_heads_gold_and_original_pool_are_retained():
    heads = dict(zip(audit.MODELS, [list(range(i, i + 100)) for i in (0, 100, 200)]))
    assert audit.universe(heads, [399, 400], [500]) == list(range(300)) + [399, 400, 500]
    del heads['Z-96']
    with pytest.raises(ValueError, match='all three'):
        audit.universe(heads, [399], [])


def test_scope_rejects_non_train_duplicate_or_unbalanced_queries():
    queries = [dict(query_id=f'{s}/{d}/{i}', subset=d, surface=s, split='TRAIN')
               for s in audit.SURFACES for d in audit.metrics.DOMAINS for i in range(128)]
    audit.validate_scope(queries)
    for field, value in [('split', 'LOCKED_TEST'), ('subset', 'nq'),
                         ('query_id', queries[1]['query_id'])]:
        bad = deepcopy(queries)
        bad[0][field] = value
        with pytest.raises(ValueError):
            audit.validate_scope(bad)


def test_scores_are_fp64_with_deterministic_id_tie_order():
    query = sparse.csr_matrix([[2., 1.]])
    documents = sparse.csr_matrix([[1., 2.], [2., 0.], [0., 1.]])
    zeros = sparse.csr_matrix((1, 2))
    values, error = audit.check_scores(query, documents, zeros, documents,
                                       [3, 1, 2], [1, 3, 2], [4., 4., 1.])
    assert values.tolist() == [4., 4., 1.] and error == 0.
    with pytest.raises(ValueError, match='head parity'):
        audit.check_scores(query, documents, zeros, documents,
                            [3, 1, 2], [3, 1, 2], [4., 4., 1.])
    with pytest.raises(AssertionError):
        audit.check_scores(query, documents, zeros, documents,
                            [3, 1, 2], [1, 3, 2], [4.01, 4., 1.])


def test_summary_partitions_only_initial_relative_loss_and_keeps_origins():
    data = []
    for s in audit.SURFACES:
        for d in audit.metrics.DOMAINS:
            for i in range(128):
                raw = record()
                raw.update(query_id=f'{s}/{d}/{i}', surface=s, domain=d)
                if s == 'TRAIN_SENTINEL':
                    raw.update(pool=None, anchors=[])
                data.extend(audit.reductions(raw))
    result = audit.summarize(data)
    pilot = result['TRAIN_PILOT/nq']
    assert pilot['positives'] == 256
    assert pilot['origins'] == dict(original_positive=128, added_vs_original=128)
    assert pilot['comparisons']['K-96-minus-Z-96']['gross_positive_dcg_loss_partition'] == {}
    assert pilot['strict_anchor_transitions'] == dict(persistent=0, repaired=128, new=0)
    sentinel = result['TRAIN_SENTINEL/nq']['boundary_transitions']['10']['persistent']
    assert sentinel['coverage_teacher'] == {'not_trained/unobserved': 128}


def test_supervisor_uses_cpu_only_bounded_existing_phase(tmp_path):
    args = SimpleNamespace(research_root=tmp_path, run=tmp_path)
    with patch.object(audit, 'verify'), patch.object(audit, 'sha', return_value='same'), \
            patch.object(audit, 'read', return_value={'source': {'ng78_boundaries.py': 'same'}}), \
            patch.object(audit.io, 'write'), patch.object(audit.execution, 'sealed'), \
            patch.object(audit.pipeline, 'phase') as phase, \
            patch.object(audit.pipeline, 'supervise_graph') as gpu_graph:
        audit.supervise(args)
        assert args.gpu is None
        assert phase.call_args.kwargs == dict(cuda=False, limit_seconds=1800)
        gpu_graph.assert_not_called()


def test_changed_parent_is_rejected_before_any_output(tmp_path):
    args = SimpleNamespace(research_root=tmp_path, run=tmp_path / 'NG-0078/unit')
    with patch.object(audit, 'sha', return_value='bad'):
        with pytest.raises(ValueError, match='NG77 parent changed'):
            audit.freeze(args)
    assert not args.run.exists()


@pytest.mark.parametrize('tie', [False, True])
@pytest.mark.parametrize('sentinel', [False, True])
def test_independent_scalar_reducer_matches_known_fixtures(tie, sentinel):
    raw = record()
    if tie:
        raw['scores']['K-96'][1] = 2.
    if sentinel:
        raw.update(surface='TRAIN_SENTINEL', pool=None, anchors=[])
    audit.compare(audit.reductions(raw), reference.reference(raw))
