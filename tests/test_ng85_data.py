"""CPU-only NG85 contract checks; no model or locked-data access."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    'ng85_data', Path(__file__).parents[1] / 'scripts/ng85_data.py')
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def test_distribution_affine_and_finite():
    np.testing.assert_allclose(DATA.target_distribution([1, 2, 4]),
                               DATA.target_distribution([12, 14, 18]))
    np.testing.assert_allclose(DATA.target_distribution([5, 5]), [.5, .5])
    with pytest.raises(ValueError):
        DATA.target_distribution([1, np.nan])


def test_top_ties_and_negative_scores():
    assert DATA.stable_top([0, 3, 3, 3, -1], 2) == [1, 2]
    assert DATA.stable_top([-3, -2, -1], 2) == [2, 1]


def test_order_complete_balanced_identical_arms():
    queries = [dict(subset=d, split='fit') for d in DATA.DOMAINS for _ in range(8)]
    queries += [dict(subset='fiqa', split='validation')]
    order = DATA.training_order(queries, 85001)
    assert len(order) == 64
    assert sorted(order[:32]) == sorted(order[32:]) == list(range(32))
    assert order == DATA.training_order(queries, 85001)
    assert order != DATA.training_order(queries, 85002)
    for start in range(0, len(order), 4):
        assert [queries[i]['subset'] for i in order[start:start + 4]] == list(DATA.DOMAINS)


def test_literal_prefix_fits_all_encoders():
    def tokenizer(text, **kwargs):
        return dict(input_ids=list(text) + [0, 0])

    text = 'unicode \u03b1\u03b2\u03b3 and words'
    visible, lengths = DATA.prefix_visible(text, [tokenizer, tokenizer], [12, 10])
    assert text.startswith(visible)
    assert all(n <= limit for n, limit in zip(lengths, [12, 10], strict=True))
    with pytest.raises(ValueError):
        DATA.prefix_visible('', [tokenizer], [10])


def test_empty_gold_is_quarantined_without_relabeling():
    documents = [dict(_id='1', title='', text='good'),
                 dict(_id='2', title='', text='  '),
                 dict(_id='3', title='title only', text='')]
    queries = [dict(query_id='a', positive_ids=['1']),
               dict(query_id='b', positive_ids=['1', '2']),
               dict(query_id='c', positive_ids=['3'])]
    kept, eligible, receipt = DATA.fiqa_integrity(documents, queries)
    assert [d['_id'] for d in kept] == ['1', '3']
    assert eligible == {0, 2}
    assert receipt['empty_document_ids'] == ['2']
    assert receipt['excluded_queries'][0]['query_id'] == 'b'
    assert queries[1]['positive_ids'] == ['1', '2']
    with pytest.raises(ValueError, match='missing canonical'):
        DATA.fiqa_integrity(documents, [dict(query_id='x', positive_ids=['4'])])
