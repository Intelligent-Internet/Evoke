from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'scripts' / 'benchmark_filtered_route_scale.py'
SCRIPT_TEXT = SCRIPT_PATH.read_text(encoding='utf-8')
SPEC = importlib.util.spec_from_file_location(
    'benchmark_filtered_route_scale',
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_filter_accepts_scale_and_unfiltered() -> None:
    assert MODULE.parse_filter('p006:161') == MODULE.FilterSpec('p006', 161, 0)
    assert MODULE.parse_filter('slice:10:3') == MODULE.FilterSpec('slice', 10, 3)
    assert MODULE.parse_filter('unfiltered') == MODULE.FilterSpec(
        'unfiltered',
        None,
        0,
    )


@pytest.mark.parametrize('value', ('', 'p006', 'p006:0', 'p006:10:10'))
def test_parse_filter_rejects_invalid_specs(value: str) -> None:
    with pytest.raises(Exception):
        MODULE.parse_filter(value)


def test_parse_range_filter() -> None:
    assert MODULE.parse_range_filter('recent:publish_date:20240000') == (
        'recent',
        None,
        0,
        'publish_date',
        20240000,
    )


def test_parse_range_filter_rejects_invalid_filter() -> None:
    with pytest.raises(Exception):
        MODULE.parse_range_filter('recent:publish-date:20240000')


def test_compare_hits_distinguishes_identity_and_score_bits() -> None:
    reference = [(1, '3ff0000000000000'), (2, '4000000000000000')]
    same_ids = [(1, '3ff0000000000001'), (2, '4000000000000000')]
    different_ids = [(1, '3ff0000000000000'), (3, '4000000000000000')]

    assert MODULE.compare_hits(reference, reference) == {
        'rank_and_score_bits_equal': True,
        'identity_equal': True,
        'overlap_at_k': 1.0,
    }
    assert MODULE.compare_hits(reference, same_ids) == {
        'rank_and_score_bits_equal': False,
        'identity_equal': True,
        'overlap_at_k': 1.0,
    }
    assert MODULE.compare_hits(reference, different_ids) == {
        'rank_and_score_bits_equal': False,
        'identity_equal': False,
        'overlap_at_k': 0.5,
    }


def test_compare_hits_supports_route_pairwise_equivalence() -> None:
    direct = [(7, '3ff0000000000000'), (11, '4000000000000000')]
    transpose = list(direct)

    comparison = MODULE.compare_hits(direct, transpose)

    assert comparison['rank_and_score_bits_equal'] is True
    assert comparison['identity_equal'] is True
    assert comparison['overlap_at_k'] == 1.0


def test_hit_digest_tracks_rank_and_score_bits() -> None:
    hits = [(7, '3ff0000000000000'), (11, '4000000000000000')]

    assert MODULE.hit_digest(hits) == MODULE.hit_digest(list(hits))
    assert MODULE.hit_digest(hits) != MODULE.hit_digest(list(reversed(hits)))
    assert MODULE.hit_digest(hits) != MODULE.hit_digest(
        [(7, '3ff0000000000001'), (11, '4000000000000000')]
    )


def test_routes_for_run_can_reuse_direct_after_exact_qualification() -> None:
    assert [route.name for route in MODULE.routes_for_run(False)] == [
        'exact_bmp',
        'auto',
        'direct',
        'transpose',
        'hybrid',
        'bound',
    ]
    assert [route.name for route in MODULE.routes_for_run(True)] == [
        'auto',
        'direct',
        'transpose',
        'hybrid',
        'bound',
    ]


def test_trace_counters_include_bmp_metadata_windows() -> None:
    assert 'posting_block_metadata_reads' in MODULE.TRACE_COUNTERS


def test_bound_route_is_auditable_against_one_candidate_binary() -> None:
    assert 'filtered_forward_bound_bytes' in MODULE.TRACE_COUNTERS
    assert 'filtered_forward_direct_estimated_bytes' in MODULE.TRACE_COUNTERS
    assert 'filtered_forward_transpose_estimated_bytes' in MODULE.TRACE_COUNTERS
    assert 'filtered_forward_bound_estimated_bytes' in MODULE.TRACE_COUNTERS
    assert 'filtered_forward_bound_blocks_scored' in MODULE.TRACE_COUNTERS
    assert '--candidate-library' in SCRIPT_TEXT
    assert 'pg_temp.evoke_filtered_route_query' in SCRIPT_TEXT
    assert 'pg_temp.evoke_filtered_route_status' in SCRIPT_TEXT
    assert 'auto_vs_direct' in SCRIPT_TEXT
    assert 'direct_vs_hybrid' in SCRIPT_TEXT
    assert 'direct_vs_bound' in SCRIPT_TEXT
    assert "'schema': 'evoke_filtered_route_scale_v2'" in SCRIPT_TEXT


def forward_trace(**overrides: object) -> dict[str, object]:
    trace: dict[str, object] = {
        'query_route': 'forward_rows',
        'filtered_forward_direct_estimated_bytes': '100',
        'filtered_forward_transpose_estimated_bytes': '200',
        'filtered_forward_bound_estimated_bytes': '50',
        'filtered_forward_transpose_budget_exceeded': False,
        'semantic_accelerator_forward_direct_rows': True,
        'semantic_accelerator_forward_transposed': False,
        'semantic_accelerator_forward_bounded': False,
    }
    trace.update(overrides)
    return trace


def test_auto_route_audit_accepts_cheapest_direct_executor() -> None:
    audit = MODULE.audit_auto_route(forward_trace())

    assert audit['expected_mode'] == 'direct'
    assert audit['passed'] is True


def test_auto_route_audit_accepts_bound_above_working_set_limit() -> None:
    trace = forward_trace(
        query_route='forward_bound',
        filtered_forward_direct_estimated_bytes='100000000',
        filtered_forward_transpose_estimated_bytes='90000000',
        filtered_forward_bound_estimated_bytes='10000000',
        semantic_accelerator_forward_direct_rows=False,
        semantic_accelerator_forward_bounded=True,
    )

    audit = MODULE.audit_auto_route(trace)

    assert audit['expected_mode'] == 'bound'
    assert audit['passed'] is True


def test_auto_route_audit_rejects_wrong_or_ambiguous_mode() -> None:
    wrong = MODULE.audit_auto_route(
        forward_trace(
            query_route='forward_transpose',
            semantic_accelerator_forward_direct_rows=False,
            semantic_accelerator_forward_transposed=True,
        )
    )
    ambiguous = MODULE.audit_auto_route(
        forward_trace(semantic_accelerator_forward_transposed=True)
    )

    assert wrong['expected_mode'] == 'direct'
    assert wrong['passed'] is False
    assert ambiguous['observed_modes'] == ['direct', 'transpose']
    assert ambiguous['passed'] is False


def test_auto_route_audit_skips_nonforward_routes() -> None:
    audit = MODULE.audit_auto_route({'query_route': 'ranked_prefix'})

    assert audit == {
        'applicable': False,
        'query_route': 'ranked_prefix',
        'passed': True,
        'reason': 'auto route did not select a forward executor',
    }


def test_query_vector_fixture_bypasses_runtime_encoding() -> None:
    class Cursor:
        def __init__(self) -> None:
            self.executed = False

        def execute(self, _query: object) -> None:
            self.executed = True

        def fetchall(self) -> list[tuple[list[int], list[float], str]]:
            return [([7, 11], [0.5, 1.25], 'fixture-signature')]

    cursor = Cursor()
    encoded = MODULE.encode_query(
        cursor,
        'bench.index',
        'fixture query',
        'bench.query_vector',
    )

    assert cursor.executed is True
    assert encoded == ([7, 11], [0.5, 1.25], 'fixture-signature')
    assert '--query-vector-table' in SCRIPT_TEXT


def test_large_filter_sets_remain_server_side() -> None:
    assert 'CREATE TEMP TABLE evoke_filtered_route_sets' in SCRIPT_TEXT
    assert 'FROM pg_temp.evoke_filtered_route_sets' in SCRIPT_TEXT
    assert 'array_agg(ctid ORDER BY ctid)' in SCRIPT_TEXT
    assert 'list[str] | None' not in SCRIPT_TEXT


def test_root_identity_ignores_query_telemetry() -> None:
    base = {
        'generation_id': '1/2/3',
        'contract_signature': 'contract',
        'posting': {'signature': 'posting'},
        'primary': {'manifest_start_block': 42},
        'semantic_accelerator': {
            'source_manifest_id': '2',
            'builder_policy_id': 7,
        },
        'workload_fold': {'shared_query_observations': 0},
    }
    after = {
        **base,
        'workload_fold': {'shared_query_observations': 100},
    }

    assert MODULE.root_identity(base) == MODULE.root_identity(after)
