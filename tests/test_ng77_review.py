"""Independent review primitives must not call the generating objective."""

import ast
import hashlib
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_ng77_preparation as review


@pytest.mark.parametrize('value', [-1000., -1., 0., 1., 1000.])
def test_scalar_probability_extremes(value):
    p = review.probability(value)
    assert math.isfinite(p) and 0 <= p <= 1
    assert p + review.probability(-value) == pytest.approx(1.)


def test_weight_head_and_cutoff_are_separate_from_positive_mean():
    c = dict(ndcg_cutoff=10, recall_cutoff=100, recall_weight=1., pair_floor=.05)
    assert review.weight(101, 102, 2, c) == .025
    assert review.weight(100, 101, 2, c) == .525
    expected = 1 / (1 + 1 / math.log2(3)) + .5 + .025
    assert review.weight(1, 101, 2, c) == pytest.approx(expected)


def test_inventory_rejects_content_change_and_path_escape(tmp_path):
    inner = tmp_path / 'inner'
    inner.mkdir()
    p = inner / 'a'
    p.write_text('original')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    review.verify(inner, {'a': digest})
    with pytest.raises(ValueError):
        review.verify(inner, {'../inner/a': 'f' * 64})
    outside = tmp_path / 'b'
    outside.write_text('original')
    with pytest.raises(ValueError):
        review.verify(inner, {'../b': digest})
    p.write_text('modified')
    with pytest.raises(ValueError):
        review.verify(inner, {'a': digest})


def test_no_generator_or_model_imports():
    tree = ast.parse(Path(review.__file__).read_text())
    modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names}
    modules |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not modules & {'torch', 'ng77_retention', 'prepare_ng77_retention', 'ng71_training', 'transformers'}
