"""Read-only NG86 continuation: timing, provenance and paired scoring gates."""

from pathlib import Path
import sys

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).parent
sys.path.insert(0, str(SOURCE))
from ng85_data import sha, write  # noqa: E402
from ng86_evaluation import (  # noqa: E402
    checked_phase, compare_rows, projected_seconds, validate_records)


def test_eta_does_not_multiply_model_load_by_query_count():
    assert projected_seconds(40, .2, 16, 2560, 5, 3) == pytest.approx(945.7)
    assert projected_seconds(100, .2, 16, 2560, 5, 3) == pytest.approx(1005.7)
    assert projected_seconds(40, 1, 16, 2560, 5, 3) > 1800
    with pytest.raises(ValueError):
        projected_seconds(40, 0, 16, 2560, 5, 3)


def row():
    return dict(query_index=1, split='fit', subset='fever', document_ids=[2, 3],
                valid=True, scores=[.1, .2], candidate_ndcg10=.5, candidate_recall10=1.)


def test_parity_checks_raw_scores_identity_validity_and_metrics():
    ref = row()
    assert compare_rows(row(), ref) == 0.
    for change in (dict(document_ids=[3, 2]), dict(valid=False), dict(scores=[.1, .3]),
                   dict(scores=[.1, float('nan')]), dict(candidate_ndcg10=.51)):
        with pytest.raises(ValueError):
            compare_rows(dict(ref, **change), ref)


def test_failed_phase_can_be_verified_but_cannot_be_promoted(tmp_path):
    folder = tmp_path / 'profile-control'
    folder.mkdir()
    write(folder / 'exit.json', dict(passed=False, exit_code=1, owned_group_closed=True))
    write(folder / 'complete.json', dict(passed=False, files={'exit.json':sha(folder / 'exit.json')}))
    assert not checked_phase(tmp_path, 'profile-control', passed=False)['passed']
    with pytest.raises(ValueError):
        checked_phase(tmp_path, 'profile-control')
    write(folder / 'extra.json', {})
    with pytest.raises(ValueError, match='inventory'):
        checked_phase(tmp_path, 'profile-control', passed=False)


def test_pairing_rejects_mixed_universes_and_partial_panels():
    control = [dict(row(), arm='control', checkpoint=label)
               for label in ('base', 'start', '2560', '2816')]
    fullstd = [dict(row(), arm='fullstd', checkpoint=label)
               for label in ('2560', '2816', '3072')]
    validate_records(control, fullstd, [1])
    with pytest.raises(ValueError):
        validate_records(control[:-1], fullstd, [1])
    fullstd[-1]['document_ids'] = [2, 4]
    with pytest.raises(ValueError, match='universe'):
        validate_records(control, fullstd, [1])
