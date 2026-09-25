from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'benchmark_forward_bound_projection.py'
QUERY_SOURCE = (ROOT / 'src' / 'evoke_page_query.c').read_text(
    encoding='utf-8'
)
QUERY_HEADER = (ROOT / 'src' / 'evoke_page_query.h').read_text(
    encoding='utf-8'
)


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        'benchmark_forward_bound_projection',
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_bound_oracle_is_bounded_and_read_only() -> None:
    assert 'EVOKE_PAGE_QUERY_FORWARD_BOUND_AUDIT_MAX_BYTES' in QUERY_HEADER
    assert 'estimated_work_bytes' in QUERY_SOURCE
    oracle_source = QUERY_SOURCE.split(
        'evoke_page_query_forward_bound_audit_run(',
        maxsplit=1,
    )[1]
    assert 'store_semantic' not in oracle_source
    assert 'publish_semantic' not in oracle_source
    assert '] = {3, 4, 6};' in oracle_source
    script_source = SCRIPT.read_text(encoding='utf-8')
    assert 'evoke_index_generation_readiness_internal_c' in script_source
    assert '--query-vector-table' in script_source
    assert '--ceiling-only' in script_source
    assert 'evoke_test_query_term_suffix_ceiling' in script_source


def test_term_suffix_probe_does_not_scan_forward_rows() -> None:
    probe_source = QUERY_SOURCE.split(
        'evoke_page_query_term_suffix_ceiling_run(',
        maxsplit=1,
    )[1].split(
        'evoke_page_query_forward_bound_audit_run(',
        maxsplit=1,
    )[0]
    assert 'load_semantic_accelerator_forward_directory' in probe_source
    assert 'load_semantic_accelerator_forward(' not in probe_source
    assert 'forward_chunks[' not in probe_source


def test_expand_query_preserves_field_namespaces() -> None:
    module = load_script()
    ids, weights = module.expand_query(
        {'atoms': [1, 4], 'weights': [0.5, 0.25]},
        lexical_dims=3,
        semantic_dims=2,
        field_count=2,
    )
    assert ids == [1, 4, 7, 9]
    assert weights == [0.5, 0.5, 0.25, 0.25]


def test_promotion_gate_requires_exactness_and_structural_reduction() -> None:
    module = load_script()
    passing = {
        'posting_fraction': 0.49,
        'row_byte_fraction': 0.48,
        'projected_bound_bytes': 40,
        'addressable_query_bytes': 10,
        'competitive_row_bytes': 40,
        'allowed_row_bytes': 100,
        'topk_contained': True,
    }
    assert (
        module.gate_level(passing, 'scattered_p025')['promotion_gate']
        is True
    )

    failing = dict(passing, posting_fraction=0.51)
    assert (
        module.gate_level(failing, 'scattered_p025')['promotion_gate']
        is False
    )
    assert (
        module.gate_level(passing, 'scattered_p006')['promotion_gate']
        is False
    )

    expensive = dict(passing, addressable_query_bytes=40)
    assert (
        module.gate_level(expensive, 'scattered_p025')['promotion_gate']
        is False
    )


def test_prefix_ceiling_requires_work_and_byte_reduction() -> None:
    module = load_script()
    audit = {
        'maximum_query_term': 157_163,
        'forward_term_work_total': '1000',
        'forward_term_work_suffix': '149',
        'forward_term_bytes_total': '2000',
        'forward_term_bytes_suffix': '400',
    }
    ceiling = module.prefix_ceiling(audit)
    assert ceiling['global_suffix_work_fraction'] == 0.149
    assert ceiling['global_suffix_byte_fraction'] == 0.2
    assert ceiling['can_reach_15pct_physical_reduction'] is False

    audit['forward_term_work_suffix'] = '150'
    assert (
        module.prefix_ceiling(audit)['can_reach_15pct_physical_reduction']
        is True
    )


def test_filter_shapes_keep_equal_cardinality_but_change_locality() -> None:
    module = load_script()
    document_count = 12_800
    scattered = module.scattered_slots(document_count, 40)
    distributed = module.distributed_run_slots(document_count, 8, 320)
    clustered = module.clustered_slots(document_count, 0.025)

    assert len(scattered) == len(distributed) == len(clustered) == 320
    assert scattered[:3] == [0, 40, 80]
    assert distributed[:9] == list(range(8)) + [320]
    assert clustered[-1] == 319
