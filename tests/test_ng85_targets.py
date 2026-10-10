"""Small exact BM25 and storage-shape controls for the common NG85 bank."""

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from ng85_targets import lexical_corpus, teacher_batches


def test_bm25_corpus_matches_frozen_ng8_algebra():
    documents = [dict(text='a a b'), dict(text='a c'), dict(text='d')]
    matrix, vocabulary, rms = lexical_corpus(documents, str.split)
    assert matrix.shape == (3, 4) and matrix.nnz == 5
    counts = np.zeros((3, 4), dtype=np.float64)
    for i, row in enumerate(documents):
        for word in row['text'].split():
            counts[i, vocabulary[word]] += 1
    lengths = counts.sum(axis=1)
    df = (counts > 0).sum(axis=0)
    idf = np.log1p((3 - df + .5) / (df + .5))
    expected = idf * counts / (counts + 1.5 * (
        .25 + .75 * lengths[:, None] / lengths.mean()))
    np.testing.assert_allclose(matrix.toarray(), expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(rms, np.sqrt((expected ** 2).mean(axis=0)), rtol=1e-6)


def test_length_grouping_restores_original_candidate_order():
    texts = ['aaaaa', 'bb', 'cc', 'd', 'eeeeeeee']

    def tokenizer(values, truncation):
        assert truncation is False
        return {'input_ids': [list(text) for text in values]}

    groups = list(teacher_batches(texts, tokenizer, 2, True))
    assert [ids for ids, _ in groups] == [[3, 1], [2, 0], [4]]
    scores = np.full(len(texts), np.nan)
    for positions, values in groups:
        assert values == [texts[i] for i in positions]
        scores[positions] = [len(text) for text in values]
    np.testing.assert_array_equal(scores, [5, 2, 2, 1, 8])
    assert [ids for ids, _ in teacher_batches(texts, tokenizer, 2)] == [
        [0, 1], [2, 3], [4]]
