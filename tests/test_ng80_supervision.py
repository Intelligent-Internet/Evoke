import json
import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import ng80_supervision as audit


def test_star_relations_already_determine_every_relative_score():
    result = audit.connectivity([.9, .3, .2, .1], [True, False, False, False])
    assert result['eligible_pairs'] == 3
    assert result['independent_exact_margin_constraints'] == 3
    assert result['full_field_constraints'] == 3
    assert result['unconstrained_relative_degrees'] == 0


def test_teacher_opposition_disconnects_unjudged_documents():
    result = audit.connectivity([.5, .9, .2, .7], [True, False, False, False])
    assert result['eligible_pairs'] == 1
    assert result['components'] == 3
    assert result['unconstrained_relative_degrees'] == 2


def test_all_positive_mask_does_not_invent_supervision():
    result = audit.connectivity([.9, .2, .1], [True, True, True])
    assert result['eligible_pairs'] == 0
    assert result['components'] == 3


def test_second_moment_and_effective_rank_are_repetition_invariant():
    x = np.asarray([[1., 0.], [0., 1.]])
    a, b = audit.spectrum(x), audit.spectrum(np.repeat(x, 5, axis=0))
    np.testing.assert_allclose(a['eigenvalues'], [.5, .5], atol=1e-12)
    np.testing.assert_allclose(a['eigenvalues'], b['eigenvalues'], atol=1e-12)
    assert a['entropy_rank'] == pytest.approx(2)
    assert a['participation_rank'] == pytest.approx(2)


def test_zero_geometry_is_rejected():
    with pytest.raises(ValueError, match='empty geometry'):
        audit.spectrum(np.zeros((3, 2)))


def test_numpy_input_metrics_have_json_native_output_types():
    result = audit.connectivity(np.asarray([.9, .3, .2]), np.asarray([True, False, False]))
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    geometry = audit.spectrum(np.eye(3))
    assert json.loads(json.dumps(geometry, allow_nan=False)) == geometry
