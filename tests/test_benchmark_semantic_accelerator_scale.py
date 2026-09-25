from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location(
    'benchmark_semantic_accelerator_scale',
    ROOT / 'scripts' / 'benchmark_semantic_accelerator_scale.py',
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecordingCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...] | None]] = []

    def execute(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
    ) -> None:
        self.calls.append((query, params))


def setting_values(cursor: RecordingCursor) -> dict[str, str]:
    return {
        str(params[0]): str(params[1])
        for _query, params in cursor.calls
        if params is not None
    }


def route(name: str) -> Any:
    return next(candidate for candidate in MODULE.ROUTES if candidate.name == name)


def test_term_budget_route_sets_only_requested_error_budget() -> None:
    cursor = RecordingCursor()

    MODULE.configure_route(cursor, route('term_budget0025'))
    values = setting_values(cursor)

    assert values['evoke.test_query_semantic_error_budget_ratio'] == '0.0025'
    assert values['evoke.test_disable_semantic_bmp'] == 'on'
    assert values['evoke.test_disable_semantic_accelerator'] == 'on'
    assert values['evoke.test_query_max_df_ratio'] == '1'
    assert values['evoke.test_query_semantic_work_target_postings'] == '0'


def test_custom_term_budget_sets_requested_ratio() -> None:
    cursor = RecordingCursor()
    custom = MODULE.Route(
        'term_budget_0p02',
        None,
        16,
        False,
        False,
        False,
        2,
        False,
        0.02,
    )

    MODULE.configure_route(cursor, custom)
    values = setting_values(cursor)

    assert values['evoke.test_query_semantic_error_budget_ratio'] == '0.02'
    assert values['evoke.test_disable_semantic_accelerator'] == 'on'
    assert values['evoke.test_disable_semantic_bmp'] == 'on'


def test_exact_route_resets_approximation_controls() -> None:
    cursor = RecordingCursor()

    MODULE.configure_route(cursor, route('exact'))
    values = setting_values(cursor)

    assert values['evoke.test_query_semantic_error_budget_ratio'] == '0.0'
    assert values['evoke.test_disable_semantic_bmp'] == 'off'
    assert values['evoke.test_query_semantic_impact_floor_ratio'] == '0'
    assert values['evoke.test_query_semantic_min_support_ratio'] == '0'
    assert values['evoke.test_disable_semantic_accelerator'] == 'on'


def test_default_route_resets_all_hidden_route_overrides() -> None:
    cursor = RecordingCursor()

    MODULE.configure_route(cursor, route('default'))

    assert cursor.calls
    assert all(params is None for _query, params in cursor.calls)
    reset_names = {
        query.removeprefix('RESET ')
        for query, _params in cursor.calls
    }
    assert 'evoke.test_disable_semantic_accelerator' in reset_names
    assert 'evoke.test_semantic_accelerator_heap_factor' in reset_names
    assert 'evoke.test_query_semantic_error_budget_ratio' in reset_names


def test_combined_budget_clones_accelerator_route() -> None:
    routes = MODULE.select_routes(
        {'exact', 'cross_term32'},
        [0.01],
        True,
    )

    assert [candidate.name for candidate in routes] == [
        'exact',
        'cross_term32',
        'cross_term32_budget_0p01',
    ]
    combined = routes[-1]
    assert combined.error_budget == 0.01
    assert combined.heap_factor == 0.7
    assert combined.candidate_multiplier == 32
    assert combined.bound_residual
    assert combined.accumulate_residual


def test_combined_budget_reaches_accelerator_and_budget_gucs() -> None:
    cursor = RecordingCursor()
    combined = MODULE.select_routes(
        {'exact', 'cross_term32'},
        [0.01],
        True,
    )[-1]

    MODULE.configure_route(cursor, combined)
    values = setting_values(cursor)

    assert values['evoke.test_disable_semantic_accelerator'] == 'off'
    assert values['evoke.test_query_semantic_error_budget_ratio'] == '0.01'
    assert values['evoke.test_disable_semantic_bmp'] == 'on'
