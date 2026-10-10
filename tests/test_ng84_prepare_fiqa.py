"""Canonical human-positive provenance, not RLHN relabel promotion."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ng84_prepare_fiqa import canonical_rows, normalized


def qrel(query, doc, score='1'):
    return {'query-id': query, 'corpus-id': doc, 'score': score}


def test_all_original_positives_survive_without_inventing_negatives():
    rows = canonical_rows([qrel('1', 'a'), qrel('1', 'b'), qrel('1', 'a')],
                          [{'_id': '1', 'text': 'A financial question?'}], {'a', 'b', 'c'})
    assert rows[0]['positive_ids'] == ['a', 'b']
    assert rows[0]['judged_negative_ids'] == []
    assert rows[0]['exposure_status'] == 'prior_project_exposure_not_yet_audited'


@pytest.mark.parametrize('queries', [[], [{'_id': '2', 'text': 'Not TRAIN'}],
    [{'_id': '1', 'text': 'TRAIN'}, {'_id': '2', 'text': 'Not TRAIN'}]])
def test_nontrain_or_missing_query_cannot_enter(queries):
    with pytest.raises(ValueError, match='exactly original TRAIN'):
        canonical_rows([qrel('1', 'a')], queries, {'a'})


def test_missing_positive_is_an_error_not_a_dropped_qrel():
    with pytest.raises(ValueError, match='positive absent'):
        canonical_rows([qrel('1', 'missing')], [{'_id': '1', 'text': 'q'}], {'a'})


def test_canonicalization_is_unicode_and_whitespace_stable():
    assert normalized(' A\tCAFE\u0301  ') == normalized('a caf\u00e9')
