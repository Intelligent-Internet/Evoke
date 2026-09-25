from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
CONTROL = (ROOT / 'evoke.control').read_text(encoding='utf-8')


def current_sql() -> str:
    version = re.search(
        r"default_version = '([^']+)'",
        CONTROL,
    )
    assert version is not None
    return (ROOT / 'sql' / f'evoke--{version.group(1)}.sql').read_text(
        encoding='utf-8',
    )


def c_function(name: str, next_name: str) -> str:
    start = SOURCE.index(f'\n{name}(')
    end = SOURCE.index(f'\n{next_name}(', start)
    return SOURCE[start:end]


def test_product_status_uses_bounded_generation_reader() -> None:
    sql = current_sql()

    assert (
        "AS 'MODULE_PATHNAME', "
        "'evoke_index_generation_readiness_internal_c'"
    ) in sql
    assert 'generation := evoke_index_generation_status_internal(' in sql
    assert 'CREATE FUNCTION evoke_index_generation_audit_internal(' in sql
    assert (
        "generation->>'docs_scope' = 'sealed_generation'"
    ) in sql


def test_current_catalog_exposes_explicit_internal_contract() -> None:
    sql = current_sql()

    assert 'CREATE FUNCTION evoke_catalog_contract_internal()' in sql
    assert "SELECT 'evoke_catalog_v1'::text" in sql
    assert (
        'REVOKE EXECUTE ON FUNCTION '
        'evoke_catalog_contract_internal() FROM PUBLIC;'
    ) in sql


def test_bounded_reader_does_not_walk_relation_sized_diagnostics() -> None:
    readiness = c_function(
        'evoke_am_convergent_generation_readiness_datum',
        'evoke_am_convergent_generation_audit_datum',
    )

    assert 'evoke_segment_pages_load_maintenance_manifest(' in readiness
    assert 'evoke_segment_pages_load_document_summary(' in readiness
    assert '"diagnostics_complete\\\":false' in readiness
    for forbidden in (
        'evoke_segment_pages_load_sealed_manifest(',
        'evoke_segment_pages_inventory_reachable(',
        'evoke_segment_pages_count_recyclable_markers(',
        'evoke_segment_pages_load_semantic_accelerator_directory(',
    ):
        assert forbidden not in readiness


def test_bounded_reader_reports_current_scope_from_fixed_header() -> None:
    helper = c_function(
        'evoke_am_read_scope_readiness',
        'evoke_am_convergent_generation_readiness_datum',
    )
    readiness = c_function(
        'evoke_am_convergent_generation_readiness_datum',
        'evoke_am_convergent_generation_audit_datum',
    )
    audit = c_function(
        'evoke_am_convergent_generation_audit_datum',
        'evoke_am_generation_audit_datum',
    )

    assert 'uint8 scope_bytes[EVOKE_SCOPE_HEADER_SIZE]' in helper
    assert 'evoke_segment_pages_read_range(' in helper
    assert 'evoke_scope_header_deserialize(' in helper
    assert 'evoke_scope_header_is_current(&scope_header)' in helper
    assert '"scope_present\\\":%s' in readiness
    assert '"scope_version\\\":%u' in readiness
    assert '"scope_current\\\":%s' in readiness
    assert '"scope_bytes\\\":' in readiness
    assert 'accelerator_state = "stale_scope";' in readiness
    assert '"scope_present\\\":%s' in audit
    assert '"scope_version\\\":%u' in audit
    assert '"scope_current\\\":%s' in audit
    assert 'accelerator_state = "stale_scope";' in audit


def test_bounded_reader_reports_missing_accelerator_as_absent() -> None:
    readiness = c_function(
        'evoke_am_convergent_generation_readiness_datum',
        'evoke_am_convergent_generation_audit_datum',
    )

    assert 'accelerator_present && accelerator_eligible' in readiness
    assert 'accelerator_baseline_current' in readiness
    assert '"ready_baseline_delta"' in readiness
    assert '"candidate_only_baseline_delta"' in readiness
    assert 'accelerator_state = "absent";' in readiness


def test_status_reports_periodic_accelerator_refresh_debt() -> None:
    readiness = c_function(
        'evoke_am_convergent_generation_readiness_datum',
        'evoke_am_convergent_generation_audit_datum',
    )
    audit = c_function(
        'evoke_am_convergent_generation_audit_datum',
        'evoke_am_generation_audit_datum',
    )

    for status_reader in (readiness, audit):
        assert 'evoke_am_convergent_accelerator_refresh_state(' in (
            status_reader
        )
        assert '"compatible\\\":%s' in status_reader
        assert '"refresh_delta_records\\\":' in status_reader
        assert '"refresh_delta_bytes\\\":' in status_reader
        assert '"refresh_record_threshold\\\":' in status_reader
        assert '"refresh_byte_threshold\\\":' in status_reader
        assert '"refresh_max_age_ms\\\":null' in status_reader
        assert '"refresh_due\\\":%s' in status_reader
        assert '"periodic_refresh_eligible\\\":%s' in status_reader
        assert '"periodic_refresh_interval_ms\\\":%d' in status_reader


def test_deep_audit_retains_complete_storage_validation() -> None:
    audit = c_function(
        'evoke_am_convergent_generation_audit_datum',
        'evoke_am_generation_audit_datum',
    )

    for required in (
        'evoke_segment_pages_load_sealed_manifest(',
        'evoke_segment_pages_inventory_reachable(',
        'evoke_segment_pages_count_recyclable_markers(',
        'evoke_segment_pages_load_semantic_accelerator_directory(',
    ):
        assert required in audit
