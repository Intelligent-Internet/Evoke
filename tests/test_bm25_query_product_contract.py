from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AM = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
FILTER = (ROOT / 'src' / 'evoke_filter.c').read_text(encoding='utf-8')
PRELOAD = (ROOT / 'src' / 'evoke_am_preload.c').read_text(encoding='utf-8')
RESIDENT_FOLD = (
    ROOT / 'src' / 'evoke_am_resident_fold.c'
).read_text(encoding='utf-8')
CONTROL_SQL = (ROOT / 'evoke.control').read_text(encoding='utf-8')
VERSION = CONTROL_SQL.split("default_version = '", 1)[1].split("'", 1)[0]
INSTALL_SQL = (ROOT / 'sql' / f'evoke--{VERSION}.sql').read_text(
    encoding='utf-8',
)


def function_body(name: str, next_name: str) -> str:
    start = AM.index(f'\n{name}(')
    end = AM.index(f'\n{next_name}(', start)
    return AM[start:end]


def test_backend_local_bm25_fallback_requires_no_shared_runtime() -> None:
    bounded = function_body(
        'evoke_am_bounded_bm25_snapshot_admitted',
        'evoke_am_require_bounded_bm25_snapshot',
    )
    admission = function_body(
        'evoke_am_no_shared_runtime_bm25_fallback_admitted',
        'evoke_am_query_token_ids_cached',
    )

    assert 'evoke_am_sae_enabled(index_relation)' in bounded
    assert 'evoke_workspace_cache_bytes <= 0' in bounded
    assert 'evoke_am_relation_nblocks(index_relation) *' in bounded
    assert 'relation_bytes <= (uint64) evoke_workspace_cache_bytes' in bounded
    assert 'evoke_am_preload_available()' in admission
    assert 'evoke_am_bounded_bm25_snapshot_admitted(' in admission


def test_weight_mask_snapshot_is_finitely_bounded() -> None:
    requirement = function_body(
        'evoke_am_require_bounded_bm25_snapshot',
        'evoke_am_no_shared_runtime_bm25_fallback_admitted',
    )

    assert 'evoke_am_bounded_bm25_snapshot_admitted(' in requirement
    assert 'ERRCODE_PROGRAM_LIMIT_EXCEEDED' in requirement
    assert 'evoke.workspace_cache_bytes' in requirement
    assert AM.count('"weight_mask"') >= 3


def test_backend_snapshot_cache_reuses_idle_entries_across_indexes() -> None:
    cache_entry = function_body(
        'evoke_am_cache_get_unleased_entry',
        'evoke_am_cache_get_entry',
    )

    assert 'evoke_am_cache_entry **entries;' in AM
    assert 'entry = evoke_am_cache.entries[i];' in cache_entry
    assert 'if (entry->lease_count == 0)' in cache_entry
    assert 'evoke_am_cache_entry_reset(entry);' in cache_entry
    assert 'entry->index_oid = index_oid;' in cache_entry
    assert 'EVOKE_AM_BACKEND_SNAPSHOT_CACHE_LIMIT' in cache_entry


def test_backend_snapshot_budget_is_aggregate_and_releases_peak_entries() -> None:
    release = function_body(
        'evoke_am_cache_release_lease',
        'evoke_am_cache_workspace_bytes',
    )
    budget = function_body(
        'evoke_am_cache_require_snapshot_budget',
        'evoke_am_cache_entry_matches',
    )
    attach = function_body(
        'evoke_am_get_cached_segment_index',
        'evoke_am_get_cached_index_internal',
    )

    assert 'evoke_am_cache_has_other_idle_snapshot(entry)' in release
    assert 'evoke_am_cache_entry_reset(entry);' in release
    assert 'evoke_am_cache_active_snapshot_bytes(target_entry)' in budget
    assert 'relation_bytes > budget_bytes - active_bytes' in budget
    assert 'evoke.workspace_cache_bytes' in budget
    assert 'evoke_am_cache_require_snapshot_budget(' in attach


def test_backend_snapshot_peak_drain_keeps_one_materialized_idle_entry() -> None:
    idle = function_body(
        'evoke_am_cache_has_other_idle_snapshot',
        'evoke_am_cache_release_lease',
    )

    assert 'entry != excluded_entry' in idle
    assert 'entry->lease_count == 0' in idle
    assert 'entry->mcxt != NULL' in idle


def test_public_bm25_searches_prefer_shared_fold_then_page_native() -> None:
    assert AM.count(
        '!evoke_am_no_shared_runtime_bm25_fallback_admitted('
    ) == 3
    assert AM.count('evoke_am_prepare_page_native_search_state(') >= 3


def test_field_aware_diagnostic_uses_common_query_dispatcher() -> None:
    field_aware = function_body(
        'evoke_field_aware_query_tokens',
        'evoke_query_ids',
    )

    assert 'evoke_am_build_field_query(' in field_aware
    assert 'evoke_am_prepare_page_native_search_state(' in field_aware
    assert 'evoke_am_get_cached_segment_index(' not in field_aware
    assert 'evoke_am_get_cached_index(' not in field_aware


def test_root_embedded_resident_fold_is_rebuilt_after_reclamation() -> None:
    start = PRELOAD.index('\nevoke_am_preload_kind_rekey_safe(')
    end = PRELOAD.index(
        '\nevoke_am_preload_kind_uses_base_identity(',
        start,
    )
    rekey_safe = PRELOAD[start:end]
    rekey_start = PRELOAD.index('\nevoke_am_preload_rekey_generation(')
    rekey_end = PRELOAD.index(
        '\nevoke_am_preload_retire_exact(',
        rekey_start,
    )
    rekey = PRELOAD[rekey_start:rekey_end]

    assert 'EVOKE_AM_PRELOAD_RESIDENT_FOLD' not in rekey_safe
    assert 'EVOKE_AM_PRELOAD_UNIFIED_WARM' in rekey_safe
    assert 'EVOKE_AM_PRELOAD_DOCUMENT_LENGTHS' in rekey_safe
    assert 'EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP' in rekey_safe
    assert 'evoke_am_preload_kind_rekey_safe(kind)' in rekey
    assert 'evoke_am_preload_retire_entry_locked(entry);' in rekey
    assert 'entry->meta = *current_meta;' in rekey


def test_obsolete_preload_reservation_cannot_be_republished() -> None:
    publish_start = PRELOAD.index('\nevoke_am_preload_publish_token(')
    publish_end = PRELOAD.index(
        '\nevoke_am_preload_reserve(',
        publish_start,
    )
    publish = PRELOAD[publish_start:publish_end]

    assert 'entry->in_use && !entry->obsolete' in publish
    assert 'return published;' in publish
    assert 'bool\nevoke_am_preload_reservation_commit(' in PRELOAD


def test_preload_eviction_preserves_higher_priority_relations() -> None:
    start = PRELOAD.index(
        '\nevoke_am_preload_entry_relation_evictable('
    )
    end = PRELOAD.index(
        '\nevoke_am_preload_has_evictable_relation_locked(',
        start,
    )
    eviction = PRELOAD[start:end]

    assert 'entry->auto_preload_priority >' in eviction
    assert 'evoke_am_auto_preload_priority(index_relation)' in eviction
    assert eviction.index(
        'evoke_am_preload_entry_same_generation('
    ) < eviction.rindex('entry->auto_preload_priority >')
    same_generation = eviction.split(
        'evoke_am_preload_entry_same_generation(',
        1,
    )[1].split('if (allow_preload_entries', 1)[0]
    assert 'EVOKE_AM_PRELOAD_RESIDENT_FOLD' in same_generation
    assert 'EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP' in same_generation


def test_resident_fold_materialization_has_admission_preflight() -> None:
    prewarm = function_body(
        'evoke_am_prewarm_unified_generation',
        'evoke_am_cache_ensure_capacity',
    )

    preflight = prewarm.index('evoke_am_preload_status_snapshot(')
    materialize = prewarm.index('evoke_segment_pages_load_sealed_snapshot(')
    assert preflight < materialize
    assert 'resident_fold_admitted' in prewarm
    assert 'EVOKE_AM_PRELOAD_RESIDENT_FOLD' in prewarm[preflight:materialize]
    assert 'EVOKE_AM_PRELOAD_ADMISSION_ADMISSIBLE' in prewarm
    assert 'relation_bytes <= (uint64) evoke_prewarm_max_bytes' not in prewarm
    assert 'evoke_am_resident_fold_materialization_fits(' in prewarm
    lock = prewarm.index('evoke_am_try_resident_build_lock(')
    recheck = prewarm.index(
        'EVOKE_AM_PRELOAD_RESIDENT_FOLD',
        lock,
    )
    assert preflight < lock < recheck < materialize
    assert 'evoke_am_resident_build_unlock();' in prewarm
    accelerator_preference = prewarm.index(
        'prefer_semantic_accelerator ='
    )
    metadata_jump = prewarm.index('goto query_metadata;')
    length_projection = prewarm.index(
        'evoke_am_prewarm_page_native_document_lengths('
    )
    marker_check = prewarm.index(
        'EVOKE_AM_PRELOAD_UNIFIED_WARM',
        length_projection,
    )
    assert accelerator_preference < metadata_jump < materialize
    assert materialize < length_projection < marker_check


def test_semantic_accelerator_precedes_resident_fold() -> None:
    preference = function_body(
        'evoke_am_semantic_accelerator_query_preferred',
        'evoke_am_prewarm_unified_generation',
    )
    search = function_body(
        'evoke_am_prepare_page_native_filtered_search_state_at_root',
        'evoke_am_prepare_page_native_search_state_at_root',
    )
    prefer = search.index(
        '!evoke_am_semantic_accelerator_query_preferred('
    )
    resident = search.index(
        'evoke_am_try_shared_resident_fold_search(',
        prefer,
    )

    assert 'evoke_semantic_accelerator_disabled' in preference
    assert 'evoke_am_sae_enabled(index_relation)' in preference
    assert (
        'evoke_segment_manifest_semantic_accelerator_eligible('
        in preference
    )
    assert 'evoke_method_requires_nonoccurrence(' in preference
    assert prefer < resident


def test_resident_fold_materialization_budget_keeps_scratch_headroom() -> None:
    resident = (
        ROOT / 'src' / 'evoke_am_resident_fold.c'
    ).read_text(encoding='utf-8')
    start = resident.index(
        '\nevoke_am_resident_fold_materialization_fits('
    )
    end = resident.index(
        '\nevoke_am_resident_fold_publish(',
        start,
    )
    budget = resident[start:end]

    assert 'physical_bytes / 8' in budget
    assert 'UINT64_C(8) * 1024 * 1024 * 1024' in budget
    assert 'publication target is already charged' in budget
    assert 'relation_bytes <= scratch_bytes' in budget


def test_semantic_readiness_requires_a_current_fast_path() -> None:
    sql = (ROOT / 'sql' / 'evoke--0.2.5.sql').read_text(encoding='utf-8')
    start = sql.index('CREATE FUNCTION evoke_index_status(index_name regclass)')
    end = sql.index('\nCOMMENT ON FUNCTION evoke_index_status(regclass)', start)
    status = sql[start:end]

    assert "'query_usable', query_usable" in status
    assert "'performance_ready', performance_ready" in status
    assert 'query_ready := query_usable;' in status
    assert 'query_ready := performance_ready;' not in status
    assert "'{semantic_accelerator,eligible}'" in status
    assert "'{shared_preload,resident_fold_current}'" in status
    assert "'{shared_preload,query_metadata_warm}'" in status
    assert "'{generation,auto_preload_priority}'" in status
    assert "THEN 'semantic_completion_pending'" in status
    assert "THEN 'semantic_accelerator_not_ready'" in status
    assert "THEN 'query_metadata_not_warm'" in status


def test_resident_fold_fill_failure_falls_back_without_aborting() -> None:
    resident = (
        ROOT / 'src' / 'evoke_am_resident_fold.c'
    ).read_text(encoding='utf-8')
    start = resident.index('\nevoke_am_resident_fold_publish(')
    end = resident.index('\nevoke_am_resident_fold_view_init(', start)
    publish = resident[start:end]

    assert 'filled = evoke_am_resident_fold_fill(' in publish
    assert 'evoke_am_preload_reservation_abort(&reservation);' in publish
    assert 'EVOKE_AM_RESIDENT_FOLD_PUBLISH_FAILED' in publish
    assert 'failed to build evoke resident fold' not in publish


def test_unified_warm_marker_reports_query_metadata_readiness() -> None:
    header = (ROOT / 'src' / 'evoke_am_preload.h').read_text(
        encoding='utf-8',
    )
    prewarm = function_body(
        'evoke_am_prewarm_unified_generation',
        'evoke_am_read_unified_warm_marker',
    )
    semantic_prewarm = function_body(
        'evoke_am_prewarm_semantic_query_metadata',
        'evoke_am_prewarm_convergent_generation_pages',
    )
    runtime = function_body(
        'evoke_index_runtime_state_internal',
        'evoke_index_shared_preload_resident',
    )

    assert 'evoke_am_unified_warm_marker' in header
    assert 'EVOKE_AM_UNIFIED_WARM_MARKER_VERSION UINT16_C(2)' in header
    assert 'EVOKE_AM_UNIFIED_WARM_QUERY_METADATA_COMPLETE' in header
    assert 'EVOKE_AM_UNIFIED_WARM_ACCELERATOR_AUTHORITY' in header
    assert 'uint64 authority_id;' in header
    assert 'evoke_segment_object_ref authority;' in header
    assert 'evoke_am_prewarm_semantic_forward_heads(' in semantic_prewarm
    forward_heads = semantic_prewarm.index(
        'evoke_am_prewarm_semantic_forward_heads('
    )
    value_metadata = semantic_prewarm.index(
        'scope_header.gram_filter_offset - value_metadata_offset'
    )
    assert forward_heads < value_metadata
    assert '(void) evoke_segment_pages_prewarm_range(' in semantic_prewarm
    assert '*pages_warmed < page_budget' in semantic_prewarm
    assert 'evoke_am_prewarm_page_native_document_lengths(' in prewarm
    assert 'query_metadata_loading' in prewarm
    marker_read = prewarm.index('evoke_am_read_unified_warm_marker(')
    tid_projection = prewarm.index(
        'evoke_am_prewarm_page_native_document_tid_lookup('
    )
    length_projection = prewarm.index(
        'evoke_am_prewarm_page_native_document_lengths('
    )
    loading_check = prewarm.index('if (query_metadata_loading)')
    assert tid_projection < length_projection < loading_check < marker_read
    assert (
        'return EVOKE_AM_AUTO_PRELOAD_BUSY;'
        in prewarm[loading_check:marker_read]
    )
    marker_retire = prewarm.index(
        'evoke_am_preload_retire_exact(',
        marker_read,
    )
    marker_reserve = prewarm.index(
        'evoke_am_preload_reserve(',
        marker_retire,
    )
    assert marker_read < marker_retire < marker_reserve
    assert 'page_metadata_complete' in prewarm
    assert 'query_metadata_complete = query_metadata_complete &&' in prewarm
    assert 'marker->pages_warmed = pages_warmed;' in prewarm
    assert 'marker->authority_id = query_authority_id;' in prewarm
    assert 'marker->authority = query_authority;' in prewarm
    assert 'shared_query_warm_marker_valid=%s' in runtime
    assert 'shared_query_metadata_warm=%s' in runtime
    assert 'shared_query_warm_pages=%llu' in runtime
    assert '\\"query_warm_marker_valid\\"' in runtime
    assert '\\"query_metadata_warm\\"' in runtime
    assert '\\"query_warm_pages\\"' in runtime
    assert "'evoke_index_runtime_state_json'" in INSTALL_SQL


def test_manual_preload_retries_an_incomplete_warm_marker() -> None:
    manual = function_body(
        'evoke_index_preload',
        'evoke_am_test_parse_u64_text',
    )

    read_marker = manual.index('evoke_am_read_unified_warm_marker(')
    incomplete = manual.index(
        'if (query_warm_marker_valid && !query_metadata_warm)',
        read_marker,
    )
    retire = manual.index('evoke_am_preload_retire_exact(', incomplete)
    retry = manual.index('evoke_am_prewarm_unified_generation(', retire)
    assert read_marker < incomplete < retire < retry
    assert 'EVOKE_AM_PRELOAD_UNIFIED_WARM' in manual[retire:retry]


def test_query_metadata_is_scoped_to_its_serving_authority() -> None:
    start = PRELOAD.index('\nevoke_am_preload_kind_rekey_safe(')
    end = PRELOAD.index(
        '\nevoke_am_preload_kind_uses_base_identity(',
        start,
    )
    rekey = PRELOAD[start:end]
    base_kind_start = end
    base_kind_end = PRELOAD.index(
        '\nevoke_am_preload_payload_size_valid(',
        base_kind_start,
    )
    base_kinds = PRELOAD[
        base_kind_start:base_kind_end
    ]
    key_hash_start = PRELOAD.index('\nevoke_am_preload_entry_key_hash(')
    key_hash_end = PRELOAD.index(
        '\nevoke_am_preload_entry_has_block(',
        key_hash_start,
    )
    key_hash = PRELOAD[key_hash_start:key_hash_end]
    same_generation_start = PRELOAD.index(
        '\nevoke_am_preload_entry_same_generation('
    )
    same_generation_end = PRELOAD.index(
        '\nevoke_am_preload_entry_matches_current_kind(',
        same_generation_start,
    )
    same_generation = PRELOAD[
        same_generation_start:same_generation_end
    ]

    assert 'EVOKE_AM_PRELOAD_UNIFIED_WARM' in rekey.split('return ', 1)[1]
    assert 'EVOKE_AM_PRELOAD_DOCUMENT_LENGTHS' in rekey
    assert 'EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP' in rekey
    assert 'EVOKE_AM_PRELOAD_UNIFIED_WARM' in base_kinds
    assert 'EVOKE_AM_PRELOAD_DOCUMENT_LENGTHS' in base_kinds
    assert 'EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP' in base_kinds
    assert '!evoke_am_preload_kind_uses_base_identity(kind)' in key_hash
    assert 'evoke_am_preload_kind_uses_base_identity(kind)' in same_generation
    assert 'evoke_am_preload_base_identity_matches(' in same_generation
    assert 'evoke_am_query_metadata_authority authority;' in AM
    assert 'evoke_am_query_metadata_authority_matches(' in AM
    assert 'header->authority' in AM

    marker = function_body(
        'evoke_am_read_unified_warm_marker',
        'evoke_am_cache_ensure_capacity',
    )
    preference = function_body(
        'evoke_am_semantic_accelerator_query_preferred',
        'evoke_am_prewarm_unified_generation',
    )
    assert 'manifest.semantic_accelerator_directory' in preference
    assert (
        'evoke_segment_manifest_semantic_accelerator_baseline_sequence('
        in preference
    )
    assert 'marker->authority_id == expected_authority_id' in marker
    assert 'evoke_segment_object_ref_equal(' in marker
    assert 'evoke_am_preload_lease_retire(&lease);' in marker
    assert 'evoke_am_preload_retire_exact(' not in marker


def test_resident_fold_publication_does_not_rescan_full_payload() -> None:
    source = (ROOT / 'src' / 'evoke_am_resident_fold.c').read_text()
    fill_start = source.index('\nevoke_am_resident_fold_fill(')
    fill_end = source.index(
        '\nevoke_am_resident_fold_publish(',
        fill_start,
    )
    fill = source[fill_start:fill_end]

    assert '#define EVOKE_AM_RESIDENT_FOLD_VERSION 2' in source
    assert 'uint64 reserved;' in source
    assert 'evoke_am_resident_fold_checksum' not in source
    assert 'memset(block, 0, block_size)' not in fill
    assert 'memset(target, 0, sizeof(*target))' in fill
    assert 'evoke_am_resident_fold_block_valid(block, block_size)' in fill


def test_shared_exact_root_fold_precedes_page_native_context_loading() -> None:
    query_path = function_body(
        'evoke_am_prepare_page_native_filtered_search_state_at_root',
        'evoke_am_prepare_page_native_search_state_at_root',
    )
    resident_attempt = query_path.index(
        'evoke_am_try_shared_resident_fold_search('
    )
    page_native_load = query_path.index(
        'evoke_segment_pages_load_query_context('
    )

    assert resident_attempt < page_native_load
    resident_path = function_body(
        'evoke_am_try_shared_resident_fold_search',
        'evoke_am_try_shared_hot_fold_search',
    )
    assert 'root->active_l0.record_count != 0' in resident_path
    assert 'root->pending_l0.record_count != 0' in resident_path
    assert (
        'evoke_am_resident_fold_root_id(&view) != root->root_id'
        in resident_path
    )


def test_resident_fold_identity_tie_order_uses_the_core_null_contract() -> None:
    start = RESIDENT_FOLD.index('\nevoke_am_resident_fold_topk_internal(')
    end = RESIDENT_FOLD.index(
        '\nevoke_am_resident_fold_topk(',
        start,
    )
    topk = RESIDENT_FOLD[start:end]

    assert topk.count(
        'tie_break_order == NULL\n'
        '                        ? 0\n'
        '                        : header->tie_break_order_count'
    ) == 1
    assert topk.count(
        'tie_break_order == NULL\n'
        '                    ? 0\n'
        '                    : header->tie_break_order_count'
    ) == 1


def test_semantic_resident_fold_preload_keeps_residual_projection_ready() -> None:
    prewarm = function_body(
        'evoke_am_prewarm_unified_generation',
        'evoke_am_read_unified_warm_marker',
    )
    resident_success = prewarm.index(
        'publish_result == EVOKE_AM_RESIDENT_FOLD_PUBLISH_DONE'
    )
    resident_return = prewarm.index(
        'return publish_result == EVOKE_AM_RESIDENT_FOLD_PUBLISH_DONE',
        resident_success,
    )
    success_path = prewarm[resident_success:resident_return]

    assert 'evoke_am_sae_enabled(indexRelation)' in success_path
    assert (
        'evoke_am_prewarm_page_native_document_lengths('
        in success_path
    )


def test_filtered_search_never_attempts_term_major_shared_fold() -> None:
    resident = function_body(
        'evoke_am_try_shared_resident_fold_search',
        'evoke_am_try_shared_hot_fold_search',
    )
    query_path = function_body(
        'evoke_am_prepare_page_native_filtered_search_state_at_root',
        'evoke_am_prepare_page_native_search_state_at_root',
    )

    assert 'const evoke_page_query_filter *filter' not in resident
    assert 'evoke_am_resident_fold_topk_filtered(' not in resident
    first_resident = query_path.index(
        'evoke_am_try_shared_resident_fold_search('
    )
    page_topk = query_path.index('evoke_page_query_topk_filtered(')

    assert 'if (filter == NULL && allowed_tid_keys == NULL &&' in query_path
    assert query_path.find(
        'evoke_am_try_shared_resident_fold_search(',
        first_resident + 1,
        page_topk,
    ) == -1


def test_structured_filters_use_bounded_serving_scope_candidates() -> None:
    estimator_start = AM.index('\nevoke_am_filter_prefix_next_probe(')
    helper_start = AM.index('\nevoke_am_try_structured_filter_prefix(')
    estimator = AM[estimator_start:helper_start]
    helper_end = AM.index(
        '\nPG_FUNCTION_INFO_V1(evoke_index_semantic_query_native_internal);',
        helper_start,
    )
    helper = AM[helper_start:helper_end]
    scope_helper_start = AM.index(
        '\nevoke_am_try_structured_scope_filter(',
        helper_start,
    )
    scope_helper = AM[scope_helper_start:helper_end]
    query_start = AM.index('\nevoke_index_semantic_query_native_internal(')
    query_end = AM.index(
        '\nPG_FUNCTION_INFO_V1(evoke_field_aware_query_tokens);',
        query_start,
    )
    query = AM[query_start:query_end]

    assert 'EVOKE_AM_FILTER_PREFIX_FIRST_MAX_K' in helper
    assert 'EVOKE_AM_FILTER_PREFIX_MAX_K' in helper
    assert (
        '#define EVOKE_AM_FILTER_PREFIX_MAX_K ((size_t) 1024)'
        in AM
    )
    assert 'evoke_am_filter_prefix_next_probe(' in helper
    assert 'adaptive_range_prefix = evoke_filter_is_range_only(filters)' in helper
    assert '!adaptive_range_prefix' in helper
    assert 'evoke_am_prepare_page_native_search_state_at_root(' in helper
    assert 'evoke_filter_match_tid_candidates(' in helper
    assert 'matched_count >= requested_k' in helper
    assert (
        'required_probe > EVOKE_AM_FILTER_PREFIX_MAX_K'
        in estimator
    )
    assert '*root_inout = probe_root;' in helper
    assert 'evoke_am_page_native_stats_accumulate(' in helper
    assert '(*attempt_count_out)++;' in helper
    assert '*trace_out = probe_trace;' not in helper
    assert 'EVOKE_AM_STRUCTURED_SCOPE_OVERFETCH_MULTIPLIER' in scope_helper
    assert 'evoke_am_prepare_page_native_filtered_search_state_at_root(' in (
        scope_helper
    )
    assert 'evoke_filter_match_tid_candidates(' in scope_helper
    assert 'candidate_set_complete' not in scope_helper
    assert 'evoke_filter_collect_tid_keys(' not in scope_helper
    assert 'state_out->topk.len = retained;' in scope_helper
    scope_resolution = query.index('evoke_filter_try_scope_bitmap(')
    scope_candidates = query.index('evoke_am_try_structured_scope_filter(')
    bounded_resolution = query.index('EVOKE_AM_FILTER_RESOLVE_MAX_ROWS')
    ranked_probe = query.index('evoke_am_try_structured_filter_prefix(')
    assert scope_resolution < scope_candidates < bounded_resolution
    assert bounded_resolution < ranked_probe
    assert (
        'if (scope_filter_resolved &&\n'
        '                    scope_predicates_fully_resolved &&\n'
        '                    !evoke_semantic_accelerator_disabled &&\n'
        '                    !evoke_test_force_semantic_bmp)'
        in query
    )
    assert 'evoke_am_align_scope_filter_with_root(' in query
    assert 'cleanup->filter.allowed_document_bitmap = NULL;' in query
    assert 'structured_scope_attempted' not in query
    assert 'structured_scope_complete = true;' in query
    assert (
        'scope_trace.resolved_predicate_count ==\n'
        '                        scope_trace.predicate_count'
        in query
    )
    assert (
        'if (scope_filter_resolved &&\n'
        '                    scope_filter_fully_resolved)'
        in query
    )
    assert '!cleanup->tid_filter_active &&' in query
    assert '#define EVOKE_AM_FILTER_RESOLVE_MAX_ROWS UINT64_C(65536)' in AM
    assert 'filter_probe_attempted = true;' in query
    assert 'filter_probe_rows = cleanup->allowed_tid_count;' in query
    assert r'\"filter_probe_attempted\"' in AM
    assert r'\"filter_probe_complete\"' in AM
    assert r'\"filter_probe_rows\"' in AM
    assert 'evoke_am_filter_prefix_has_physical_leverage(&root)' in query
    assert 'EVOKE_AM_FILTER_PREFIX_MIN_CORPUS_FACTOR' in AM
    assert 'structured_prefix_attempts > 0' in query
    assert 'ranked_prefix_probe_attempts' in query
    assert 'ranked_prefix_probe_documents_examined' in query
    assert 'ranked_prefix_probe_postings_examined' in query
    assert 'query->ranked_prefix_probe_attempts' in AM
    assert 'query->ranked_prefix_probe_memory_bytes' in AM
    assert 'stats->ranked_prefix_query_path' in AM


def test_ranked_filter_probe_uses_candidate_tid_scans() -> None:
    start = FILTER.index('\nevoke_filter_match_tid_candidates(')
    end = FILTER.index('\ntypedef struct evoke_filter_scope_reader', start)
    probe = FILTER[start:end]

    assert 'FROM unnest($2) AS candidate(ctid)' in FILTER
    assert 'source.ctid = candidate.ctid' in FILTER
    assert 'construct_array(' in probe
    assert 'SPI_execute_plan(' in probe
    assert 'evoke_filter_tid_key(&candidate_tids[candidate])' in probe


def test_page_native_projection_batches_document_block_reads() -> None:
    resolver = function_body(
        'evoke_am_resolve_page_native_ranked_tids',
        'evoke_am_tid_filter_contains',
    )

    assert 'qsort(' in resolver
    assert 'evoke_segment_pages_load_query_document_block(' in resolver
    assert 'refs[block_end].block_id == block_id' in resolver
    assert 'evoke_am_document_record_tid(' in resolver


def test_readiness_status_does_not_hash_model_artifacts() -> None:
    status_start = INSTALL_SQL.index(
        'CREATE FUNCTION evoke_index_status(index_name regclass)'
    )
    status_end = INSTALL_SQL.index(
        'CREATE FUNCTION evoke_index_audit(index_name regclass)',
        status_start,
    )
    status = INSTALL_SQL[status_start:status_end]
    audit_end = INSTALL_SQL.index('-- Keep direct runtime', status_end)
    audit = INSTALL_SQL[status_end:audit_end]

    assert 'evoke_index_checkout_validate_internal' not in status
    assert "'model_artifacts_valid', NULL" in status
    assert 'evoke_index_checkout_validate_internal' in audit
    assert 'evoke_index_generation_audit_internal' in audit


def test_structured_overlap_exposes_partial_gin_predicate() -> None:
    source = (ROOT / 'src' / 'evoke_filter.c').read_text()

    assert 'cardinality(source.%s) > 0 AND source.%s && ' in source


def test_filtered_bmp_admission_uses_predicate_domain() -> None:
    source = (ROOT / 'src' / 'evoke_page_query.c').read_text()
    start = source.index(
        'if (!evoke_test_force_semantic_bmp &&\n'
        '        !restrict_to_allowed_blocks &&'
    )
    admission = source[start:source.index('{', start)]

    assert '!restrict_to_allowed_blocks' in admission
    assert 'allowed_superblocks[ref->superblock_id] == 0' in source
    assert 'allowed_blocks[block_id] == 0' in source


def test_fragmented_filtered_bmp_uses_adaptive_sequential_metadata_scan(
) -> None:
    source = (ROOT / 'src' / 'evoke_page_query.c').read_text()
    selector_start = source.index(
        'evoke_page_query_bmp_prefer_sequential_super_refs('
    )
    selector_end = source.index(
        '\nstatic evoke_status\n'
        'evoke_page_query_bmp_cache_filtered_super_refs_sequential(',
        selector_start,
    )
    selector = source[selector_start:selector_end]
    sequential_start = selector_end
    sequential_end = source.index(
        '\nstatic evoke_status\n'
        'evoke_page_query_score_semantic_bmp_block(',
        sequential_start,
    )
    sequential = source[sequential_start:sequential_end]

    assert 'allowed_superblock_range_count * binary_search_reads' in selector
    assert 'estimated_selective_reads >= (uint64) super_ref_count' in selector
    assert 'allowed_superblocks[ref->superblock_id] == 0' in sequential
    assert 'semantic_bmp_filtered_sequential_scans++' in sequential


def test_filtered_bmp_streams_fine_bounds_when_super_ref_cache_is_bounded(
) -> None:
    source = (ROOT / 'src' / 'evoke_page_query.c').read_text()

    assert 'direct_flat_filtered = restrict_to_allowed_blocks;' in source
    assert (
        'score_slot_count = use_compact_filtered_scores\n'
        '        ? (uint64) allowed_block_count *\n'
        '            EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS\n'
        '        : document_slot_count;'
    ) in source
    assert 'use_compact_filtered_scores =' in source
    assert 'scores = calloc((size_t) score_slot_count' in source
    assert 'evoke_page_query_bmp_score_block_offset(' in source
    assert 'semantic_run_count * allowed_superblock_count' in source
    assert 'cache_ref_count = Min(cache_ref_count, selective_capacity);' in (
        source
    )
    assert (
        'if (direct_flat_filtered)\n'
        '    {\n'
        '        cache_super_refs = false;'
    ) not in source
    semantic_bounds = source.index(
        'if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)',
        source.index('direct_flat_filtered ='),
    )
    assert source.index(
        'if (direct_flat_filtered)',
        semantic_bounds,
    ) < source.index(
        'if (run_cache->super_refs != NULL)',
        semantic_bounds,
    )
    assert 'needs_flat_fallback = direct_flat_filtered;' in source
    assert 'materialize_filtered_document_lengths =' in source
    assert 'filtered_document_block_offsets[block_id]' in source
    assert 'else if (filtered_document_lengths != NULL)' in source


def test_filtered_bmp_flat_bounds_reuse_packed_relation_pages() -> None:
    source = (ROOT / 'src' / 'evoke_page_query.c').read_text()
    fallback_start = source.index('if (needs_flat_fallback)')
    fallback = source[fallback_start:]

    assert 'packed_window_first_ref = ref_rank' in fallback
    assert (
        'evoke_segment_pages_load_query_bmp_packed_ref_page('
        in fallback
    )
    assert 'stats->posting_block_metadata_reads++' in fallback
    assert 'evoke_semantic_bmp_packed_ref_decode(' in fallback
    assert 'packed_ref.local_block_id' in fallback
    assert 'evoke_segment_pages_load_query_bmp_bound_by_rank(' not in source


def test_filtered_bmp_batches_probe_selected_superblocks_only() -> None:
    source = (ROOT / 'src' / 'evoke_page_query.c').read_text()
    batch_start = source.index(
        'for (size_t batch_start = 0;',
        source.index('evoke_page_query_try_score_semantic_bmp('),
    )
    batch_end = source.index('if (needs_flat_fallback)', batch_start)
    batch = source[batch_start:batch_end]
    cached = batch.split(
        'if (run_cache->super_refs != NULL)',
        1,
    )[1].split('else', 1)[0]

    assert 'batch_index < batch_count' in cached
    assert '!direct_flat_filtered && batch_start <' in batch
    assert 'evoke_page_query_bmp_run_cache_find_super_ref(' in cached
    assert 'super_ref_index < run_cache->super_ref_count' not in cached


def test_filtered_exact_path_prefers_bmp_before_block_cursor_fallback() -> None:
    source = (ROOT / 'src' / 'evoke_page_query.c').read_text()
    scoring_start = source.index(
        'if (result_status == EVOKE_OK && cursor_count > 0)'
    )
    filtered_start = source.index(
        'else if (filter != NULL)',
        scoring_start,
    )
    filtered_end = source.index(
        'else if (projection == NULL)',
        filtered_start,
    )
    filtered_branch = source[filtered_start:filtered_end]

    bmp_position = filtered_branch.find(
        'evoke_page_query_try_score_semantic_bmp('
    )
    cursor_position = filtered_branch.find(
        'evoke_page_query_score_positive_blocks('
    )

    assert bmp_position >= 0
    assert cursor_position > bmp_position
    assert 'if (result_status == EVOKE_OK && !scored)' in filtered_branch


def test_exact_bmp_merges_a_bounded_linked_l0_projection() -> None:
    source = (ROOT / 'src' / 'evoke_page_query.c').read_text()
    bmp_start = source.index('\nevoke_page_query_try_score_semantic_bmp(')
    bmp_admission_end = source.index(
        'needs_liveness =',
        bmp_start,
    )
    bmp_admission = source[bmp_start:bmp_admission_end]
    scoring_start = source.index(
        'if (result_status == EVOKE_OK && cursor_count > 0)'
    )
    projected_start = source.index(
        'if (projection != NULL)',
        scoring_start,
    )
    projected_end = source.index(
        'else if (filter != NULL)',
        projected_start,
    )
    projected_branch = source[projected_start:projected_end]

    assert 'active_l0.record_count' not in bmp_admission
    assert 'pending_l0.record_count' not in bmp_admission
    assert 'cleanup->projection = projection;' in source
    assert 'evoke_page_query_root_document_allowed(' in source
    assert projected_branch.index(
        'evoke_page_query_try_score_semantic_bmp('
    ) < projected_branch.index(
        'evoke_page_query_score_positive_documents('
    )
    assert 'evoke_page_query_offer_projection(' in source


def test_filtered_query_only_attaches_worker_published_tid_directory() -> None:
    source = (ROOT / 'src' / 'evoke_am.c').read_text()
    preload = (ROOT / 'src' / 'evoke_am_preload.c').read_text()
    segments = (ROOT / 'src' / 'evoke_segments.c').read_text()
    pages = (ROOT / 'src' / 'evoke_segment_pages.c').read_text()
    filtered_function = source.split(
        '\nstatic void\n'
        'evoke_am_prepare_page_native_filtered_search_state_at_root(',
        1,
    )[1].split(
        '\nstatic void\nevoke_am_prepare_page_native_filtered_search_state(',
        1,
    )[0]
    filtered = filtered_function.split(
        'if (allowed_tid_keys != NULL)',
        1,
    )[1].split('cleanup->tid_filter.keys =', 1)[0]
    preload_path = source.split(
        '\nstatic evoke_am_auto_preload_attempt\n'
        'evoke_am_prewarm_unified_generation(',
        1,
    )[1].split('\nstatic void\nevoke_am_cache_ensure_capacity', 1)[0]
    coverage_check = source.split(
        '\nstatic bool\n'
        'evoke_am_document_tid_lookup_covers_authority(',
        1,
    )[1].split(
        '\nstatic void\n'
        'evoke_am_attach_page_native_document_tid_lookup(',
        1,
    )[0]
    tid_projection = source.split(
        '\nstatic void\n'
        'evoke_am_attach_page_native_document_tid_lookup(',
        1,
    )[1].split(
        '\nstatic void\n'
        'evoke_am_prewarm_page_native_document_tid_lookup(',
        1,
    )[0]

    assert '&context,\n            true,\n            &lease' in source
    assert '&cleanup->context,\n                    false,' in filtered
    assert 'root.pending_l0.record_count == 0' not in filtered
    assert 'evoke_am_tid_filter_mark_immutable_slot(' in source
    assert 'EVOKE_AM_QUERY_METADATA_ACCELERATOR' in coverage_check
    assert (
        'view->entry_count == context->manifest.visible_document_count'
        in coverage_check
    )
    assert (
        'view->document_slot_count == context->manifest.document_slot_count'
        in coverage_check
    )
    assert (
        'evoke_am_document_tid_lookup_covers_authority(' in filtered
    )
    assert 'pfree(cleanup->tid_filter.durable_lookup_bytes);' in filtered
    assert 'evoke_am_document_tid_lookup_covers_authority(' in tid_projection
    assert 'durable = false;' in tid_projection
    assert (
        'evoke_segment_pages_visit_query_document_records(' in tid_projection
    )
    assert 'evoke_am_preload_kind_uses_base_identity(' in preload
    assert 'kind == EVOKE_AM_PRELOAD_DOCUMENT_TID_LOOKUP' in preload
    stale_mark = source.split(
        '\nstatic void\nevoke_am_tid_filter_mark_immutable_slot(',
        1,
    )[1].split(
        '\nstatic void\nevoke_am_tid_filter_mark_immutable_document(',
        1,
    )[0]
    assert 'lookup_baseline_stale' in stale_mark
    assert 'evoke_segment_pages_load_document_record(' in stale_mark
    assert 'evoke_am_tid_filter_contains(cleanup, &tid)' in stale_mark
    assert 'evoke_segment_manifest_semantic_accelerator_published(' in segments
    tid_loader = pages.split(
        '\nbool\nevoke_segment_pages_load_semantic_accelerator_tid_lookup(',
        1,
    )[1].split(
        '\nvoid\n'
        'evoke_segment_pages_load_semantic_accelerator_forward_directory(',
        1,
    )[0]
    assert (
        'evoke_segment_manifest_semantic_accelerator_published('
        in tid_loader
    )
    assert (
        'evoke_segment_manifest_semantic_accelerator_eligible('
        not in tid_loader
    )
    assert preload_path.index(
        'evoke_am_prewarm_page_native_document_tid_lookup('
    ) < preload_path.index('evoke_segment_pages_load_sealed_snapshot(')


def test_multi_term_query_heat_observation_does_not_reload_term_plans() -> None:
    source = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
    observe = source.split(
        '\nstatic void\nevoke_am_posting_heat_observe_page_native(',
        1,
    )[1].split(
        '\nstatic bool\nevoke_am_posting_heat_candidate_snapshot(',
        1,
    )[0]

    fast_branch = observe.split('if (!exact_term_work)', 1)[1]
    assert fast_branch.index('continue;') < fast_branch.index(
        'evoke_segment_pages_load_query_term_plan('
    )
    assert 'inspect_hot_fold = exact_term_work &&' in observe


def test_filtered_bmp_materializes_only_allowed_document_lengths() -> None:
    source = (ROOT / 'src' / 'evoke_page_query.c').read_text()
    bmp_start = source.index('evoke_page_query_try_score_semantic_bmp(')
    bmp_end = source.index(
        '\nstatic bool\nevoke_page_query_term_at_a_time_budget(',
        bmp_start,
    )
    bmp = source[bmp_start:bmp_end]
    restricted = bmp.index('if (restrict_to_allowed_blocks)')
    unrestricted = bmp.index(
        'else if (!evoke_segment_pages_load_query_document_lengths(',
        restricted,
    )

    assert 'allowed_document_bitmap[byte_index]' in bmp[
        restricted:unrestricted
    ]
    assert 'evoke_page_query_load_document(' in bmp[
        restricted:unrestricted
    ]
    assert 'evoke_segment_pages_visit_query_document_records(' not in bmp[
        restricted:unrestricted
    ]


def test_forward_row_reader_caches_each_chunk_directory() -> None:
    header = (ROOT / 'src' / 'evoke_segment_pages.h').read_text()
    source = (ROOT / 'src' / 'evoke_segment_pages.c').read_text()

    assert 'uint32 *row_offsets;' in header
    assert 'uint32 row_offset_count;' in header
    assert 'reader->row_offsets[offset_index] =' in source
    assert 'row_start = reader->row_offsets[row];' in source
    assert 'row_end = reader->row_offsets[row + UINT32_C(1)];' in source
    assert 'reader->header.row_offsets_offset +' not in source[
        source.index(
            'evoke_segment_pages_score_semantic_accelerator_forward_row_internal('
        ):source.index(
            '\nvoid\nevoke_segment_pages_score_semantic_accelerator_forward_row(',
        )
    ]


def test_forward_row_reader_reuses_page_sized_payload_windows() -> None:
    header = (ROOT / 'src' / 'evoke_segment_pages.h').read_text()
    source = (ROOT / 'src' / 'evoke_segment_pages.c').read_text()

    assert 'Size row_window_offset;' in header
    assert 'Size row_window_size;' in header
    assert 'EVOKE_SEGMENT_FORWARD_ROW_WINDOW_BYTES' in source
    assert 'row_start < reader->row_window_offset' in source
    assert 'row_end > reader->row_window_offset +' in source
    assert 'row_view = reader->row_bytes +' in source


def test_filtered_forward_layouts_have_same_root_diagnostic_controls() -> None:
    header = (ROOT / 'src' / 'evoke_page_query.h').read_text()
    query = (ROOT / 'src' / 'evoke_page_query.c').read_text()
    am = (ROOT / 'src' / 'evoke_am.c').read_text()

    assert 'EVOKE_FILTERED_FORWARD_ROUTE_AUTO' in header
    assert 'EVOKE_FILTERED_FORWARD_ROUTE_DIRECT' in header
    assert 'EVOKE_FILTERED_FORWARD_ROUTE_TRANSPOSE' in header
    assert 'EVOKE_FILTERED_FORWARD_ROUTE_HYBRID' in header
    assert 'EVOKE_FILTERED_FORWARD_ROUTE_BOUND' in header
    assert 'evoke.test_filtered_forward_route' in am
    assert 'DefineCustomEnumVariable(' in am
    assert (
        'evoke_test_filtered_forward_route ==\n'
        '            EVOKE_FILTERED_FORWARD_ROUTE_DIRECT'
        in query
    )
    assert (
        'evoke_test_filtered_forward_route ==\n'
        '                 EVOKE_FILTERED_FORWARD_ROUTE_TRANSPOSE'
        in query
    )
    assert (
        'evoke_test_filtered_forward_route ==\n'
        '                 EVOKE_FILTERED_FORWARD_ROUTE_BOUND'
        in query
    )
    bound_policy_start = query.index(
        'evoke_page_query_prefer_bounded_forward_rows(',
    )
    bound_policy_end = query.index(
        '\nstatic uint64\nevoke_page_query_estimate_direct_forward_work(',
        bound_policy_start,
    )
    bound_policy = query[bound_policy_start:bound_policy_end]

    assert 'if (prefer_direct_rows' in bound_policy
    assert (
        'estimated_transpose_bytes >\n'
        '            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT'
        in bound_policy
    )
    assert (
        'estimated_bound_bytes < estimated_transpose_bytes'
        in bound_policy
    )
    assert 'filter->allowed_document_count' not in bound_policy
    assert 'directory->document_count' not in bound_policy
    assert 'prefer_bounded_rows =\n' in query


def test_accelerator_scratch_is_released_after_query_cancellation() -> None:
    query = (ROOT / 'src' / 'evoke_page_query.c').read_text(
        encoding='utf-8',
    )
    owner_start = query.index(
        'typedef struct evoke_page_query_accelerator_cleanup',
    )
    owner_end = query.index(
        '} evoke_page_query_accelerator_cleanup;',
        owner_start,
    )
    owner = query[owner_start:owner_end]
    release_start = query.index(
        'evoke_page_query_accelerator_cleanup_release(',
        owner_end,
    )
    release_end = query.index(
        '\nstatic void\nevoke_page_query_accelerator_context_reset(',
        release_start,
    )
    release = query[release_start:release_end]
    accelerator_start = query.index(
        'evoke_page_query_try_semantic_accelerator(',
        release_end,
    )
    accelerator_end = query.index(
        '\nevoke_status\nevoke_page_query_positive_topk_filtered(',
        accelerator_start,
    )
    accelerator = query[accelerator_start:accelerator_end]

    assert 'MemoryContextCallback callback;' in owner
    assert 'evoke_topk_accumulator filtered_topk;' in query
    assert 'resources->active = false;' in release
    assert 'evoke_topk_result_free(resources->result_out);' in release
    assert 'evoke_semantic_accelerator_directory_free(' in release
    assert 'evoke_segment_forward_row_reader_reset(' in release
    assert 'evoke_segment_forward_transpose_reader_reset(' in release
    assert 'evoke_topk_accumulator_free(' in release
    assert 'evoke_semantic_accelerator_query_scratch_destroy(' in release
    assert 'free(resources->terms);' in release
    register = accelerator.index('MemoryContextRegisterResetCallback(')
    first_page_load = accelerator.index(
        'evoke_segment_pages_load_semantic_accelerator_',
    )
    assert register < first_page_load
    assert (
        'evoke_page_query_accelerator_cleanup_release(resources, false);'
        in accelerator
    )
    assert 'evoke_semantic_accelerator_topk_many_owned(' in accelerator


def test_cancel_rss_gate_reuses_one_backend_and_checks_plateau() -> None:
    gate = (
        ROOT / 'scripts' / 'qualify_filtered_accelerator_cancel_rss.py'
    ).read_text(encoding='utf-8')

    assert 'pg_cancel_backend(%s)' in gate
    assert 'wait_until_active(control, pid)' in gate
    assert 'except errors.QueryCanceled:' in gate
    assert "cursor.execute('SELECT 1')" in gate
    assert "'/proc/' || %s || '/status'" in gate
    assert "'rss_plateau': True" in gate
    assert "parser.add_argument(\n        '--setup-sql'," in gate
    assert 'for setup_sql in args.setup_sql:' in gate


def test_page_native_fallback_scratch_survives_pg_cancellation() -> None:
    query = (ROOT / 'src' / 'evoke_page_query.c').read_text(
        encoding='utf-8',
    )
    ordered_start = query.index(
        'evoke_page_query_try_score_ordered_blocks(',
    )
    ordered_end = query.index(
        '\nstatic bool\nevoke_page_query_bounded_add(',
        ordered_start,
    )
    ordered = query[ordered_start:ordered_end]
    bmp_start = query.index('evoke_page_query_try_score_semantic_bmp(')
    bmp_end = query.index(
        '\nstatic bool\nevoke_page_query_term_at_a_time_budget(',
        bmp_start,
    )
    bmp = query[bmp_start:bmp_end]
    residual_start = query.index(
        'evoke_page_query_accumulate_residual_candidates_dense(',
    )
    residual_end = query.index(
        '\n#define EVOKE_PAGE_QUERY_EXACT_ROWS_PER_FORWARD_CHUNK',
        residual_start,
    )
    residual = query[residual_start:residual_end]

    assert 'MemoryContextRegisterResetCallback(' in ordered
    assert ordered.index('MemoryContextRegisterResetCallback(') < (
        ordered.index('resources->contributions = calloc(')
    )
    assert 'evoke_page_query_ordered_cleanup_release(resources);' in ordered
    assert 'MemoryContextRegisterResetCallback(' in bmp
    assert bmp.index('MemoryContextRegisterResetCallback(') < bmp.index(
        'scores = calloc('
    )
    assert 'resources->run_caches = run_caches;' in bmp
    assert 'resources->pruning_accumulator = pruning_accumulator;' in bmp
    assert 'evoke_page_query_bmp_cleanup_release(resources);' in bmp
    assert 'evoke_page_query_accelerator_cleanup *resources' in residual
    assert 'merge = &resources->exact_residual_merge;' in residual
    assert 'topk = &resources->exact_residual_topk;' in residual
    assert 'resources->residual_candidate_scores = calloc(' in residual
    assert 'free(resources->residual_candidate_scores);' in residual


def test_filtered_forward_trace_counts_physical_work_once() -> None:
    query = (ROOT / 'src' / 'evoke_page_query.c').read_text(
        encoding='utf-8',
    )
    accelerator_start = query.index(
        'evoke_page_query_try_semantic_accelerator(',
    )
    accelerator_end = query.index(
        '\nevoke_status\nevoke_page_query_positive_topk_filtered(',
        accelerator_start,
    )
    accelerator = query[accelerator_start:accelerator_end]
    transpose_start = query.index(
        'evoke_page_query_score_filtered_forward_transposed(',
    )
    transpose_end = query.index(
        '\nstatic evoke_status\nevoke_page_query_score_filtered_forward(',
        transpose_start,
    )
    transpose = query[transpose_start:transpose_end]

    assert 'score_context->chunk_reads++;' in transpose
    assert 'resources->score_context.chunk_reads = active_chunk_count;' not in (
        accelerator
    )
    assert 'evoke_page_query_topk_accumulator_bytes(' in transpose
    assert 'scores_capacity * sizeof(*scores)' in transpose


def test_sparse_filtered_rows_use_exact_payload_reads() -> None:
    header = (ROOT / 'src' / 'evoke_segment_pages.h').read_text()
    pages = (ROOT / 'src' / 'evoke_segment_pages.c').read_text()
    query = (ROOT / 'src' / 'evoke_page_query.c').read_text()

    assert 'bool exact_row_data_reads;' in header
    assert 'evoke_segment_forward_row_reader_set_exact_data_reads(' in pages
    assert 'reader->exact_row_data_reads' in pages
    assert 'EVOKE_PAGE_QUERY_EXACT_DATA_ROWS_PER_FORWARD_CHUNK' in query
    assert 'allowed_count <=' in query


def test_include_scope_filter_uses_same_root_and_exact_fallback() -> None:
    am = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
    options = (
        ROOT / 'src' / 'evoke_am_options.c'
    ).read_text(encoding='utf-8')
    scope = (ROOT / 'src' / 'evoke_scope_pg.c').read_text(encoding='utf-8')
    filter_source = (
        ROOT / 'src' / 'evoke_filter.c'
    ).read_text(encoding='utf-8')
    accelerator = (
        ROOT / 'src' / 'evoke_am_accelerator.c'
    ).read_text(encoding='utf-8')

    assert 'amroutine->amcaninclude = true;' in am
    assert 'return index_relation->rd_index->indnkeyatts;' in options
    assert 'evoke_scope_build_for_index(' in scope
    assert 'source.document_tids' in accelerator
    scope_build = accelerator.index('evoke_scope_build_for_index(')
    scope_lock = accelerator.rfind(
        'ConditionalLockRelationOid(',
        0,
        scope_build,
    )
    assert scope_lock >= 0
    assert 'scope_heap_locked' in accelerator[scope_lock:scope_build]
    assert 'evoke_filter_try_scope_bitmap(' in am
    assert 'evoke_filter_collect_tid_keys(' in am
    assert 'scope_filter_fully_resolved' in am
    assert (
        'index_relation->rd_index->indnatts ==\n'
        '        index_relation->rd_index->indnkeyatts'
        in filter_source
    )
    assert 'cleanup->tid_filter.scope_bitmap' in am
    assert 'cleanup->tid_filter.scope_document_slot_count' in am
    assert 'resolved_predicate_count == predicate_count' in filter_source
    assert 'resolved_predicate_count == 0' in filter_source
    segment_pages = (
        ROOT / 'src' / 'evoke_segment_pages.c'
    ).read_text(encoding='utf-8')
    assert 'root->active_l0.record_count != 0' in segment_pages
    assert 'root->pending_l0.record_count != 0' in segment_pages
    assert 'evoke_filter_scope_parse(' in filter_source
    assert 'evoke_filter_scope_add_ilike(' in filter_source
    assert 'evoke_scope_ascii_ilike_contains_pattern(' in filter_source
    assert (
        'evoke_scope_ascii_case_insensitive_contains('
        in filter_source
    )
    assert 'DirectFunctionCall2Coll(' in filter_source
    assert 'texticlike' in filter_source
    assert 'candidate = candidate == NULL' in filter_source
    assert 'repalloc(candidate, candidate_size)' in filter_source
    assert 'SET_VARSIZE(candidate, candidate_size);' in filter_source
    assert 'EVOKE_FILTER_SCOPE_ILIKE_MAX_COMPARISONS' in filter_source
    assert (
        'context->manifest.semantic_accelerator_directory.\n'
        '                owner_manifest_id'
    ) in segment_pages


def test_minor_only_fold_entries_are_materialized() -> None:
    pages = (
        ROOT / 'src' / 'evoke_segment_pages.c'
    ).read_text(encoding='utf-8')
    start = pages.index('\nevoke_segment_collect_fold_entries(')
    end = pages.index(
        '\nstatic uint32\nevoke_segment_fold_group_count(',
        start,
    )
    collect = pages[start:end]

    assert 'if (coverage == 0)' not in collect
    assert 'if (coverage != 0)' in collect
    assert 'states[term_id].neutral_minor_coverage != 0' in collect
    assert 'if (entry_count != entry_capacity)' in collect


def test_packed_semantic_plans_preserve_physical_posting_offsets() -> None:
    pages = (
        ROOT / 'src' / 'evoke_segment_pages.c'
    ).read_text(encoding='utf-8')
    header = (
        ROOT / 'src' / 'evoke_segment_pages.h'
    ).read_text(encoding='utf-8')
    am = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')

    assert 'uint64 stable_posting_offset;' in header
    assert 'uint64 source_posting_offset;' in header
    assert 'run->stable_posting_offset = stable_posting_offset;' in pages
    assert 'run->source_posting_offset = source_posting_offset;' in pages
    assert 'run->posting_offset = source_posting_offset;' in pages
    assert 'run.posting_offset = source->source_posting_offset;' in pages
    materialize_start = pages.index(
        '\nevoke_segment_pages_load_query_term_from_plan('
    )
    materialize_end = pages.index(
        '\nvoid\nevoke_segment_pages_load_query_term_block_record(',
        materialize_start,
    )
    materialize = pages[materialize_start:materialize_end]
    assert (
        'source->kind ==\n'
        '                    EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT'
    ) in materialize
    assert (
        'evoke_segment_pages_load_query_term_posting_stream_window('
    ) in materialize
    assert (
        'run->stable_posting_offset !=\n'
        '                stable_extent->posting_offset'
    ) in am
    assert (
        'EVOKE_SEGMENT_QUERY_RUN_SOURCE_PAYLOAD,\n'
        '            run.kind,\n'
        '            stable_extent->posting_offset,\n'
        '            run.posting_offset,'
    ) in pages


def test_scope_ilike_work_limit_is_exposed_in_query_trace() -> None:
    header = (ROOT / 'src' / 'evoke_filter.h').read_text(encoding='utf-8')
    filter_source = (
        ROOT / 'src' / 'evoke_filter.c'
    ).read_text(encoding='utf-8')
    am = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')

    assert 'uint64 ilike_comparisons;' in header
    assert 'uint32 ilike_values;' in header
    assert 'bool ilike_comparison_limit_exceeded;' in header
    assert 'trace->ilike_comparisons' in filter_source
    assert 'trace->ilike_values' in filter_source
    assert 'trace->ilike_comparison_limit_exceeded = true;' in filter_source
    assert '\\"scope_ilike_values\\"' in am
    assert '\\"scope_ilike_values_examined\\"' in am
    assert '\\"scope_ilike_blocks_considered\\"' in am
    assert '\\"scope_ilike_blocks_skipped\\"' in am
    assert '\\"scope_ilike_comparisons\\"' in am
    assert '\\"scope_ilike_comparison_limit_exceeded\\"' in am
    assert '\\"semantic_bmp_query_bytes\\"' in am
    assert '\\"semantic_bmp_super_ref_reads\\"' in am
    assert '\\"semantic_bmp_query_super_ref_count\\"' in am
    assert '\\"semantic_bmp_ref_reads\\"' in am
    assert '\\"filtered_bmp_matching_ref_count\\"' in am
    assert '\\"filtered_bmp_matching_super_ref_count\\"' in am
    assert '\\"semantic_bmp_record_reads\\"' in am
    assert '\\"semantic_bmp_postings_examined\\"' in am
    assert '\\"accelerator_forward_postings_examined\\"' in am
    assert '\\"filtered_bmp_allowed_blocks\\"' in am
    assert '\\"filtered_bmp_allowed_superblocks\\"' in am
    assert '\\"semantic_bmp_direct_flat_filtered\\"' in am
    assert '\\"document_block_reads\\"' in am
    assert '\\"memory_bytes\\"' in am
    assert '\\"semantic_accelerator_query_bytes\\"' in am
    assert '\\"accelerator_owned_index_bytes\\"' in am
    assert '\\"accelerator_membership_bytes\\"' in am
    assert '\\"accelerator_candidate_scratch_bytes\\"' in am
    assert '\\"accelerator_forward_scratch_bytes\\"' in am
    assert '\\"accelerator_residual_scratch_bytes\\"' in am
    assert '\\"accelerator_residual_postings\\"' in am
    assert '\\"accelerator_residual_documents\\"' in am
    assert (
        '\\"semantic_accelerator_residual_dense_accumulation\\"' in am
    )
    assert '\\"visibility_rank_attempts\\"' in am
    assert '\\"ranked_prefix_probe_attempts\\"' in am
    assert '\\"ranked_prefix_probe_documents_examined\\"' in am
    assert '\\"ranked_prefix_probe_postings_examined\\"' in am
    assert '\\"ranked_prefix_probe_memory_bytes\\"' in am
    assert '\\"ranked_prefix_probe_query_term_count\\"' in am
    assert '\\"ranked_prefix_probe_directory_term_count\\"' in am
    assert '\\"ranked_prefix_probe_matched_term_count\\"' in am
    assert '\\"ranked_prefix_probe_fallback\\"' in am
    assert (
        'evoke_am_last_semantic_query_trace.memory_bytes = memory_bytes;'
        in am
    )


def test_query_backends_do_not_build_shared_document_lengths() -> None:
    product = function_body(
        'evoke_am_prepare_page_native_filtered_search_state_at_root',
        'evoke_am_prepare_page_native_search_state',
    )
    prewarm = function_body(
        'evoke_am_prewarm_page_native_document_lengths',
        'evoke_am_page_native_search_cleanup_free',
    )

    calls = product.split(
        'evoke_am_attach_page_native_document_lengths('
    )[1:]
    assert len(calls) == 2
    for call in calls:
        arguments = call.split(');', 1)[0]
        assert arguments.rsplit(',', 2)[1].strip() == 'false'
    assert '&context,\n            true,' in prewarm
    assert 'evoke_am_preload_exact_state(' in prewarm
    assert 'if (resident)' in prewarm
    assert '&context,\n                    false,' in prewarm
    assert '*loading_out = loading && !ready;' in prewarm

    attach = function_body(
        'evoke_am_attach_page_native_document_lengths',
        'evoke_am_prewarm_page_native_document_lengths',
    )
    assert 'context->root.active_l0.record_count' not in attach
    assert 'context->root.pending_l0.record_count' not in attach


def test_page_native_visibility_retries_are_accounted() -> None:
    query_path = function_body(
        'evoke_am_prepare_page_native_filtered_search_state_at_root',
        'evoke_am_prepare_page_native_search_state_at_root',
    )
    accumulator = function_body(
        'evoke_am_page_native_stats_accumulate',
        'evoke_am_page_native_memory_estimate',
    )

    assert 'attempt_stats.visibility_rank_attempts = 1;' in query_path
    assert (
        'total->visibility_rank_attempts = evoke_u32_saturating_add('
        in accumulator
    )
    assert (
        'prefix_trace->query.visibility_rank_attempts'
        in AM
    )


def test_accelerator_memory_tracks_peak_visibility_attempt() -> None:
    accumulator = function_body(
        'evoke_am_page_native_stats_accumulate',
        'evoke_am_page_native_memory_estimate',
    )

    assert (
        'attempt->semantic_accelerator_query_bytes >\n'
        '        total->semantic_accelerator_query_bytes'
        in accumulator
    )
    for field in (
        'semantic_accelerator_query_bytes',
        'accelerator_owned_index_bytes',
        'accelerator_membership_bytes',
        'accelerator_candidate_scratch_bytes',
        'accelerator_forward_scratch_bytes',
        'accelerator_residual_scratch_bytes',
    ):
        assert f'total->{field} =\n            attempt->{field};' in accumulator


def test_query_route_tracks_the_final_visibility_attempt() -> None:
    prepare = function_body(
        'evoke_am_prepare_page_native_filtered_search_state_at_root',
        'evoke_am_prepare_page_native_search_state_at_root',
    )
    trace_finish = function_body(
        'evoke_am_semantic_query_trace_finish',
        'evoke_am_try_structured_filter_prefix',
    )

    assert 'evoke_page_query_stats final_attempt_query;' in AM
    assert 'bool final_attempt_query_valid;' in AM
    assert 'trace_out->final_attempt_query = attempt_stats;' in prepare
    assert 'trace_out->final_attempt_query_valid = true;' in prepare
    assert 'route_query' in trace_finish
    assert (
        'structured_prefix_complete,\n'
        '        structured_scope_complete,\n'
        '        route_query'
    ) in trace_finish


def test_query_route_names_the_bounded_forward_executor() -> None:
    route_name = function_body(
        'evoke_am_semantic_query_route_name',
        'evoke_am_semantic_query_trace_reset',
    )

    bound = route_name.index(
        'if (stats->semantic_accelerator_forward_bounded)'
    )
    direct = route_name.index(
        'if (stats->semantic_accelerator_forward_direct_rows)'
    )
    assert bound < direct
    assert 'return "forward_bound";' in route_name
    assert 'return "scope_filter";' in route_name


def test_bounded_scope_residual_uses_parallel_materialization() -> None:
    am = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
    source = (ROOT / 'src' / 'evoke_filter.c').read_text(encoding='utf-8')

    assert 'EVOKE_FILTER_MATERIALIZE_MAX_ROWS' in source
    assert 'Max(work_mem, 1)' in source
    assert 'result_upper_bound > 0' in source
    assert 'SPI_execute_plan(' in source
    assert 'SPI_cursor_open(' in source
    assert (
        'scope_filter_resolved\n'
        '                                ? cleanup->filter.allowed_document_count'
    ) in am
    assert 'row_limit == 0 && result_upper_bound > 0' in source
    assert 'remaining + UINT64_C(1)' in source
    assert 'count > row_limit' in source
    assert '"%s LIMIT " UINT64_FORMAT' in source


def test_text_filters_match_postgresql_trigram_expression_indexes() -> None:
    source = (ROOT / 'src' / 'evoke_filter.c').read_text(encoding='utf-8')

    assert 'source.%s::text ILIKE (%s #>>' in source
    assert 'source.%s::text ILIKE ANY (ARRAY(SELECT' in source


def test_filtered_zero_score_completion_is_k_bounded_with_l0() -> None:
    page_query = (
        ROOT / 'src' / 'evoke_page_query.c'
    ).read_text(encoding='utf-8')
    start = page_query.index(
        '\nevoke_page_query_complete_filtered_zero_scores('
    )
    end = page_query.index(
        '\nstatic evoke_status\nevoke_page_query_complete_zero_scores(',
        start,
    )
    completion = page_query[start:end]

    assert 'evoke_segment_pages_load_matching_live_born_prefix(' in completion
    assert 'evoke_page_query_filtered_zero_block_scan_is_lower(' in completion
    assert (
        'evoke_page_query_collect_filtered_zero_scores_by_block(' in
        completion
    )
    assert (
        'evoke_page_query_block_has_filtered_zero_candidate(' in page_query
    )
    assert (
        page_query.count(
            'evoke_page_query_block_has_filtered_zero_candidate('
        ) == 3
    )
    zero_filter_start = page_query.index(
        'evoke_page_query_block_has_filtered_zero_candidate('
    )
    zero_filter_end = page_query.index(
        '\nstatic bool\nevoke_page_query_filtered_zero_block_scan_is_lower(',
        zero_filter_start,
    )
    zero_filter = page_query[zero_filter_start:zero_filter_end]
    assert 'evoke_page_query_result_contains(' in zero_filter
    assert 'evoke_page_query_l0_find_document(' in zero_filter
    assert '!projected_document->shadows_immutable' in zero_filter
    assert 'evoke_page_query_keep_bounded_zero_candidate(' in completion
    assert 'sizeof(*candidates) * needed * 2U' in completion
    assert 'filter->allowed_document_count >' not in completion
    assert 'for (uint64 document_slot = 0;' not in completion
    assert 'evoke_segment_pages_load_document_record(' not in completion


def test_query_trace_exposes_zero_completion_cost() -> None:
    am = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
    start = am.index('PG_FUNCTION_INFO_V1(evoke_query_trace_internal);')
    end = am.index('\nstatic void\nevoke_am_direct_search_cleanup_free(', start)
    trace = am[start:end]

    assert '\\"zero_score_documents_added\\"' in trace
    assert '\\"zero_score_cow_objects_loaded\\"' in trace
    assert '\\"zero_score_cow_records_examined\\"' in trace
    assert '\\"zero_score_heap_peak\\"' in trace
    assert '\\"accelerator_directory_term_count\\"' in trace
    assert '\\"accelerator_matched_term_count\\"' in trace
