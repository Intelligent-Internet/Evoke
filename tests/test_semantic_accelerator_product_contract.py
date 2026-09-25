from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AM = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
AM_MUTATION = (ROOT / 'src' / 'evoke_am_mutation.c').read_text(
    encoding='utf-8',
)
AM_SCHEDULER = (ROOT / 'src' / 'evoke_am_scheduler.c').read_text(
    encoding='utf-8',
)
AM_SCHEDULER_HEADER = (
    ROOT / 'src' / 'evoke_am_scheduler.h'
).read_text(encoding='utf-8')
DIRECTORY = (
    ROOT / 'src' / 'evoke_semantic_accelerator_directory.h'
).read_text(encoding='utf-8')
PAGE_QUERY = (ROOT / 'src' / 'evoke_page_query.c').read_text(
    encoding='utf-8',
)
SEGMENTS = (ROOT / 'src' / 'evoke_segments.c').read_text(
    encoding='utf-8',
)
SEGMENT_PAGES = (ROOT / 'src' / 'evoke_segment_pages.c').read_text(
    encoding='utf-8',
)
DIRECTORY_SOURCE = (
    ROOT / 'src' / 'evoke_semantic_accelerator_directory.c'
).read_text(encoding='utf-8')
FORWARD = (ROOT / 'src' / 'evoke_semantic_forward.c').read_text(
    encoding='utf-8',
)
FORWARD_HEADER = (ROOT / 'src' / 'evoke_semantic_forward.h').read_text(
    encoding='utf-8',
)
SCOPE_HEADER = (ROOT / 'src' / 'evoke_scope.h').read_text(encoding='utf-8')
SCOPE_PG = (ROOT / 'src' / 'evoke_scope_pg.c').read_text(encoding='utf-8')
FILTER_SOURCE = (ROOT / 'src' / 'evoke_filter.c').read_text(encoding='utf-8')
ACCELERATOR_PG = (
    ROOT / 'src' / 'evoke_am_accelerator.c'
).read_text(encoding='utf-8')
ACCELERATOR = (
    ROOT / 'src' / 'evoke_semantic_accelerator.c'
).read_text(encoding='utf-8')
ACCELERATOR_HEADER = (
    ROOT / 'src' / 'evoke_semantic_accelerator.h'
).read_text(encoding='utf-8')


def test_promoted_accelerator_policy_is_the_product_default() -> None:
    for required in (
        '#define EVOKE_SEMANTIC_ACCELERATOR_POLICY_SCOPE_FORWARD_INT8',
        'EVOKE_SEMANTIC_ACCELERATOR_POLICY_SCOPE_FORWARD_INT8',
        'EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP UINT32_C(64)',
    ):
        assert required in DIRECTORY
    assert 'POLICY_BLOCK_LOCAL_FORWARD_INT8' not in DIRECTORY
    assert 'POLICY_ADAPTIVE_FORWARD_INT8' not in DIRECTORY
    assert (
        'return policy == EVOKE_SEMANTIC_ACCELERATOR_CURRENT_POLICY;'
        in DIRECTORY_SOURCE
    )

    for required in (
        'bool evoke_semantic_accelerator_disabled = false;',
        'double evoke_semantic_accelerator_heap_factor = 0.7;',
        'int evoke_semantic_accelerator_candidate_multiplier = 64;',
        'bool evoke_semantic_accelerator_bound_residual_candidates = true;',
        'bool evoke_semantic_accelerator_accumulate_residual_candidates = true;',
        'score_context.direct_rows =',
        'evoke_semantic_accelerator_bound_residual_candidates &&',
    ):
        assert required in PAGE_QUERY

    for required in (
        '"evoke.test_disable_semantic_accelerator"',
        '&evoke_semantic_accelerator_disabled',
        '&evoke_semantic_accelerator_heap_factor',
        '&evoke_semantic_accelerator_candidate_multiplier',
        '&evoke_semantic_accelerator_bound_residual_candidates',
        '&evoke_semantic_accelerator_accumulate_residual_candidates',
    ):
        assert required in AM


def test_accelerator_runtime_accepts_only_current_durable_formats() -> None:
    for retired in (
        'EVOKE_ACCELERATOR_DIRECTORY_LEGACY_VERSION',
        'EVOKE_ACCELERATOR_DIRECTORY_SCOPE_VERSION',
        'EVOKE_ACCELERATOR_DIRECTORY_LEGACY_HEADER_SIZE',
    ):
        assert retired not in DIRECTORY_SOURCE

    for retired in (
        'EVOKE_SEMANTIC_FORWARD_LEGACY_VERSION',
        'EVOKE_SEMANTIC_FORWARD_SPARSE_TRANSPOSE_VERSION',
        'EVOKE_SEMANTIC_FORWARD_LEGACY_HEADER_SIZE',
    ):
        assert retired not in FORWARD
        assert retired not in FORWARD_HEADER

    assert (
        'version != EVOKE_ACCELERATOR_DIRECTORY_VERSION'
        in DIRECTORY_SOURCE
    )
    assert (
        'evoke_semantic_accelerator_directory_deserialize_retired('
        in DIRECTORY_SOURCE
    )
    assert (
        'EVOKE_ACCELERATOR_DIRECTORY_RETIREMENT_MIN_VERSION'
        in DIRECTORY_SOURCE
    )
    assert (
        'EVOKE_ACCELERATOR_DIRECTORY_RETIREMENT_MAX_VERSION'
        in DIRECTORY_SOURCE
    )
    assert 'allow_retirement' in DIRECTORY_SOURCE
    assert (
        'evoke_segment_pages_load_retired_semantic_accelerator_directory('
        in SEGMENT_PAGES
    )
    assert 'stale_format' in AM
    assert '!context->semantic_accelerator_compatible' in PAGE_QUERY
    assert 'version != EVOKE_SEMANTIC_FORWARD_VERSION' in FORWARD


def test_filtered_accelerator_prefers_transposed_forward_route() -> None:
    filtered_start = PAGE_QUERY.index(
        'score_context.allowed_forward_chunk_counts ='
    )
    filtered = PAGE_QUERY[
        filtered_start:
        PAGE_QUERY.index(
            'resources->owned_indexes = calloc(',
            filtered_start,
        )
    ]

    assert 'evoke_page_query_score_filtered_forward_transposed(' in filtered
    assert 'evoke_page_query_score_filtered_forward(' in filtered
    assert 'evoke_page_query_score_filtered_forward_transposed(' in (
        filtered.split('evoke_page_query_score_filtered_forward(')[0]
    )
    assert 'evoke_page_query_prefer_direct_forward_rows(' in filtered
    assert 'direct_forward_work' in filtered
    assert 'transpose_forward_work' in filtered
    assert 'direct_forward_estimated_bytes' in filtered
    assert 'transpose_forward_estimated_bytes' in filtered
    assert 'bound_forward_estimated_bytes' in filtered
    assert 'estimated_direct_bytes <= estimated_transpose_bytes' in PAGE_QUERY
    assert 'weighted_direct_work' not in PAGE_QUERY
    assert 'FILTER_TRANSPOSE_NEAR_WORK_RATIO' not in PAGE_QUERY
    assert 'if (!prefer_direct_rows)' in filtered
    assert 'if (prefer_direct_rows || !transposed_available)' in filtered
    assert 'transposed_available && result_out->len < k' not in filtered
    assert 'bounded candidate view' not in filtered


def test_filtered_forward_uses_exact_subset_routes_without_prefix() -> None:
    query_start = PAGE_QUERY.index(
        'evoke_page_query_try_semantic_accelerator('
    )
    filtered = PAGE_QUERY[query_start:]
    directory = filtered.index(
        'evoke_page_query_count_allowed_forward_chunks('
    )
    active_documents = filtered.index(
        'maximum_forward_document_count = Max('
    )
    forward = filtered.index(
        'evoke_page_query_score_filtered_forward_transposed('
    )

    assert directory < active_documents < forward
    assert 'evoke_page_query_score_filtered_forward_bounded(' in filtered
    assert 'evoke_page_query_score_filtered_forward_transposed(' in filtered
    assert 'evoke_page_query_score_filtered_forward(' in filtered
    assert 'evoke_page_query_filter_prefix_' not in PAGE_QUERY
    assert 'evoke_page_query_record_prefix_probe(' not in PAGE_QUERY
    assert 'if (retained == k)' not in filtered[:forward]


def test_filtered_forward_locality_reuses_allowed_bitmap_scan() -> None:
    start = PAGE_QUERY.index(
        'evoke_page_query_count_allowed_forward_chunks('
    )
    end = PAGE_QUERY.index(
        '\nstatic int\nevoke_page_query_compare_accelerator_term(',
        start,
    )
    scanner = PAGE_QUERY[start:end]

    assert 'evoke_page_query_stats *stats' in scanner
    assert scanner.count('for (uint64 byte_index = 0;') == 1
    assert 'if (candidates != 0 && stats != NULL)' in scanner
    for field in (
        'filtered_forward_allowed_b8_blocks',
        'filtered_forward_allowed_b8_runs',
        'filtered_forward_allowed_b8_span',
        'filtered_forward_allowed_b8_first',
        'filtered_forward_allowed_b8_last',
        'filtered_forward_allowed_b64_ranges',
        'filtered_forward_allowed_b512_ranges',
        'filtered_forward_allowed_b4096_ranges',
    ):
        assert f'stats->{field}' in scanner

    am = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
    for field in (
        'filtered_forward_allowed_b8_blocks',
        'filtered_forward_allowed_b8_runs',
        'filtered_forward_allowed_b8_span',
        'filtered_forward_allowed_b8_first',
        'filtered_forward_allowed_b8_last',
        'filtered_forward_allowed_b64_ranges',
        'filtered_forward_allowed_b512_ranges',
        'filtered_forward_allowed_b4096_ranges',
    ):
        assert f'\\"{field}\\"' in am


def test_filtered_transpose_only_offers_allowed_documents() -> None:
    score_start = PAGE_QUERY.index(
        'evoke_page_query_score_filtered_forward_transposed(',
    )
    score_end = PAGE_QUERY.index(
        '\nstatic evoke_status\nevoke_page_query_score_filtered_forward(',
        score_start,
    )
    scorer = PAGE_QUERY[score_start:score_end]

    allowed_check = scorer.index(
        'filter->allowed_document_bitmap[document >> 3]'
    )
    topk_offer = scorer.index('evoke_topk_accumulator_offer(', allowed_check)

    assert allowed_check < topk_offer
    assert 'documents_scored++;' in scorer[allowed_check:topk_offer]

    transposed_start = SEGMENT_PAGES.index(
        'evoke_segment_pages_score_semantic_accelerator_forward_transposed('
    )
    transposed_end = SEGMENT_PAGES.index(
        '\nstatic evoke_status\n'
        'evoke_segment_pages_accumulate_forward_costs(',
        transposed_start,
    )
    transposed = SEGMENT_PAGES[transposed_start:transposed_end]

    assert 'scores_out[document] = (float) reader->scores[document];' in (
        transposed
    )


def test_filtered_direct_rows_buffer_multi_row_chunks() -> None:
    assert (
        '#define EVOKE_PAGE_QUERY_EXACT_ROWS_PER_FORWARD_CHUNK '
        'UINT32_C(1)'
    ) in PAGE_QUERY
    assert (
        '#define EVOKE_PAGE_QUERY_EXACT_DATA_ROWS_PER_FORWARD_CHUNK '
        'UINT32_C(256)'
    ) in PAGE_QUERY
    assert (
        'allowed_count <=\n'
        '                    '
        'EVOKE_PAGE_QUERY_EXACT_ROWS_PER_FORWARD_CHUNK'
    ) in PAGE_QUERY
    assert (
        'EVOKE_PAGE_QUERY_EXACT_DATA_ROWS_PER_FORWARD_CHUNK'
    ) in PAGE_QUERY
    assert (
        '#define EVOKE_SEGMENT_FORWARD_ROW_WINDOW_BYTES ((Size) BLCKSZ)'
    ) in SEGMENT_PAGES


def test_filtered_forward_planner_accounts_for_transpose_lane_work() -> None:
    assert (
        '#define EVOKE_PAGE_QUERY_FILTER_TRANSPOSE_LANE_BUDGET '
        '\\\n    EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT'
    ) in PAGE_QUERY
    assert 'EVOKE_PAGE_QUERY_FILTER_DIRECT_COST_NUMERATOR' not in PAGE_QUERY
    assert 'EVOKE_PAGE_QUERY_FILTER_DIRECT_COST_DENOMINATOR' not in PAGE_QUERY
    planner_start = PAGE_QUERY.index(
        'evoke_page_query_prefer_direct_forward_rows('
    )
    planner_end = PAGE_QUERY.index(
        '\nstatic uint64\nevoke_page_query_estimate_direct_forward_work',
        planner_start,
    )
    planner = PAGE_QUERY[planner_start:planner_end]

    assert 'maximum_forward_document_count' in planner
    assert (
        'sizeof(float) + sizeof(double) + sizeof(float) + sizeof(int8)'
        in planner
    )
    assert 'transpose_lane_bytes >' in planner
    assert 'estimated_direct_bytes <= estimated_transpose_bytes' in planner
    assert 'estimated_direct_bytes' in planner
    assert 'estimated_transpose_bytes' in planner
    assert 'weighted_direct_work' not in planner
    assert 'active_forward_posting_count' not in PAGE_QUERY
    assert 'directory->retained_document_cap' not in planner
    assert 'entry->posting_count' in PAGE_QUERY
    assert 'allowed_document_count' in PAGE_QUERY
    assert 'resources->directory.forward_term_work[' in PAGE_QUERY
    assert 'active_forward_document_count' in PAGE_QUERY
    assert 'transpose_stream_bytes' in PAGE_QUERY
    assert 'forward_object.object_bytes' in PAGE_QUERY
    assert 'EVOKE_PAGE_QUERY_FILTER_TRANSPOSE_STREAM_BUDGET' not in PAGE_QUERY
    planning_charge = PAGE_QUERY.index(
        'resources->score_context.bytes_read = evoke_u64_saturating_add(',
        planner_end,
    )
    route_override = PAGE_QUERY.index(
        'if (!prefer_direct_rows)',
        planning_charge,
    )
    assert planning_charge < route_override
    assert (
        'transpose_forward_work = evoke_u64_saturating_add(\n'
        '                    transpose_forward_work,\n'
        '                    active_forward_document_count'
    ) in PAGE_QUERY
    assert 'evoke_segment_pages_estimate_forward_transpose_work(' not in (
        PAGE_QUERY
    )
    assert 'EVOKE_PAGE_QUERY_EXACT_DATA_ROWS_PER_FORWARD_CHUNK' not in planner

    assert 'evoke_segment_pages_accumulate_forward_costs(' in SEGMENT_PAGES
    assert 'header.transpose_term_offsets_offset' in SEGMENT_PAGES
    assert 'header.transpose_dense_term_count' in SEGMENT_PAGES
    assert 'directory.forward_row_data_bytes' in SEGMENT_PAGES
    assert 'directory.forward_transpose_fixed_bytes' in SEGMENT_PAGES
    assert 'directory.forward_term_bytes' in SEGMENT_PAGES
    assert 'directory.forward_bound_term_bytes' in SEGMENT_PAGES

    am = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
    for field in (
        'filtered_forward_direct_work',
        'filtered_forward_transpose_work',
        'filtered_forward_direct_estimated_bytes',
        'filtered_forward_transpose_estimated_bytes',
        'filtered_forward_bound_estimated_bytes',
        'filtered_forward_transpose_planning_bytes',
        'filtered_forward_transpose_stream_bytes',
        'filtered_forward_transpose_lane_bytes',
        'filtered_forward_transpose_budget_exceeded',
    ):
        assert f'\\"{field}\\"' in am


def test_filtered_cold_rows_use_bounded_same_root_prefetch() -> None:
    for required in (
        '#define EVOKE_PAGE_QUERY_FILTER_PREFETCH_MIN_DOCUMENTS '
        'UINT64_C(1024)',
        '#define EVOKE_PAGE_QUERY_FILTER_PREFETCH_BATCH_DOCUMENTS '
        'UINT32_C(1024)',
        'evoke_segment_pages_prefetch_semantic_accelerator_forward_rows(',
    ):
        assert required in PAGE_QUERY

    prefetch_start = SEGMENT_PAGES.index(
        'evoke_segment_pages_prefetch_semantic_accelerator_forward_rows('
    )
    prefetch_end = SEGMENT_PAGES.index(
        '\nstatic void\n'
        'evoke_segment_pages_score_semantic_accelerator_forward_row_internal(',
        prefetch_start,
    )
    prefetch = SEGMENT_PAGES[prefetch_start:prefetch_end]

    assert 'PrefetchBuffer(' in SEGMENT_PAGES
    for required in (
        'evoke_segment_pages_accelerator_artifact_authority(',
        'evoke_segment_pages_prefetch_query_range(',
        'Resident chunk metadata does not imply resident row payloads.',
    ):
        assert required in prefetch

    assert 'metadata_io_started' not in prefetch
    assert prefetch.count(
        'evoke_segment_pages_prefetch_query_range('
    ) >= 2


def test_residual_completion_prefetches_sorted_forward_rows() -> None:
    sort_at = PAGE_QUERY.index(
        'qsort(\n'
        '                resources->residual_candidate_result.doc_ids,'
    )
    score_at = PAGE_QUERY.index(
        'evoke_page_query_score_forward_document(',
        sort_at,
    )
    completion = PAGE_QUERY[sort_at:score_at]

    assert (
        'evoke_segment_pages_prefetch_semantic_accelerator_forward_rows('
        in completion
    )
    assert 'resources->residual_candidate_result.len' in completion


def test_accelerator_exposes_nonlocal_exit_scratch_ownership() -> None:
    assert 'evoke_semantic_accelerator_query_scratch_create' in (
        ACCELERATOR_HEADER
    )
    assert 'evoke_semantic_accelerator_topk_many_owned' in ACCELERATOR_HEADER
    owned_start = ACCELERATOR.index(
        'evoke_semantic_accelerator_topk_many_owned('
    )
    wrapper_start = ACCELERATOR.index(
        '\nevoke_status\nevoke_semantic_accelerator_topk_many(',
        owned_start,
    )
    owned = ACCELERATOR[owned_start:wrapper_start]
    wrapper = ACCELERATOR[wrapper_start:]

    assert 'evoke_semantic_accelerator_query_scratch_clear(scratch);' in owned
    assert 'evoke_semantic_accelerator_query_scratch_clear_blocks(' in (
        ACCELERATOR
    )
    assert 'evoke_semantic_accelerator_query_scratch_destroy(scratch);' in (
        wrapper
    )


def test_preload_prioritizes_current_filter_pruning_metadata() -> None:
    start = AM.index('\nevoke_am_prewarm_semantic_query_metadata(')
    end = AM.index(
        '\nevoke_am_prewarm_convergent_generation_pages(',
        start,
    )
    semantic_warm = AM[start:end]
    generation_start = end
    generation_end = AM.index(
        '\nevoke_am_rebuild_workload_from_convergent(',
        generation_start,
    )
    generation_warm = AM[generation_start:generation_end]

    assert 'evoke_segment_pages_prewarm_range(' in semantic_warm
    assert '&manifest->semantic_accelerator_directory' in semantic_warm
    assert '&directory.scope_object' in semantic_warm
    assert 'scope_header.gram_filter_offset' in semantic_warm
    assert 'scope_header.gram_filter_size' in semantic_warm
    assert (
        'authority_checksum = directory.source_authority_checksum;'
        in semantic_warm
    )
    assert 'evoke_segment_manifest_authority_checksum(' not in semantic_warm
    targeted = generation_warm.index(
        'evoke_am_prewarm_semantic_query_metadata('
    )
    sequential = generation_warm.index('for (uint64 block_number = 0;')
    fragmented = generation_warm.index(
        'if (pages_warmed < page_budget)',
        sequential,
    )
    fragmented_meta = generation_warm.index(
        'meta_buffer = ReadBufferExtended(',
        fragmented,
    )
    assert targeted < sequential
    assert sequential < fragmented < fragmented_meta
    assert 'page_budget > pages_warmed' in generation_warm
    assert 'page_budget - pages_warmed' in generation_warm
    assert 'Min(\n                    remaining_pages,' in generation_warm


def test_preload_warms_current_root_before_accelerator_maintenance() -> None:
    auto_preload_start = AM.index(
        '\nevoke_am_try_auto_preload_index_oid('
    )
    auto_preload_end = AM.index(
        '\nstatic bool\nevoke_am_select_auto_preload_index(',
        auto_preload_start,
    )
    auto_preload = AM[auto_preload_start:auto_preload_end]
    due = auto_preload.index('evoke_am_convergent_accelerator_due(')
    hint = auto_preload.index('EVOKE_AM_WORK_HINT_MAINTENANCE', due)
    prewarm = auto_preload.index('evoke_am_prewarm_unified_generation(', hint)

    assert due < hint < prewarm
    assert 'EVOKE_AM_AUTO_PRELOAD_DEFERRED_TO_MAINTENANCE' not in AM


def test_resident_preload_limits_runtime_contract_snapshot() -> None:
    helper_start = AM.index(
        '\nevoke_am_require_segment_runtime_contract_bounded_snapshot('
    )
    helper_end = AM.index(
        '\nstatic void\nevoke_am_require_exact_bm25_index_at_meta(',
        helper_start,
    )
    helper = AM[helper_start:helper_end]
    preload_start = AM.index(
        '\nevoke_am_prewarm_unified_generation('
    )
    preload_end = AM.index(
        '\nstatic bool\nevoke_am_read_unified_warm_marker(',
        preload_start,
    )
    preload = AM[preload_start:preload_end]

    assert 'if (!ActiveSnapshotSet())' in helper
    assert 'PushActiveSnapshot(GetLatestSnapshot())' in helper
    assert helper.count('PopActiveSnapshot()') == 2
    contract = preload.index(
        'evoke_am_require_segment_runtime_contract_bounded_snapshot('
    )
    materialize = preload.index(
        'evoke_am_rebuild_document_tie_break_order(',
        contract,
    )
    publish = preload.index('evoke_am_resident_fold_publish(', materialize)
    assert contract < materialize < publish


def test_stale_derived_accelerator_does_not_block_exact_query_or_republish(
) -> None:
    current_start = AM.index('\nevoke_am_semantic_accelerator_current(')
    current_end = AM.index(
        '\nstatic evoke_am_accelerator_maintenance_outcome',
        current_start,
    )
    current = AM[current_start:current_end]
    context_start = SEGMENT_PAGES.index(
        '\nevoke_segment_pages_load_query_context('
    )
    context_end = SEGMENT_PAGES.index(
        '\nstatic void\nevoke_segment_pages_validate_block_contract(',
        context_start,
    )
    context = SEGMENT_PAGES[context_start:context_end]
    compatible_start = SEGMENT_PAGES.index(
        '\nevoke_segment_pages_semantic_accelerator_compatible('
    )
    compatible_end = SEGMENT_PAGES.index(
        '\nbool\nevoke_segment_pages_open_semantic_accelerator_scope(',
        compatible_start,
    )
    compatible = SEGMENT_PAGES[compatible_start:compatible_end]

    assert 'evoke_segment_pages_try_load_semantic_accelerator_summary(' in (
        current
    )
    assert 'current = status == EVOKE_OK &&' in current
    assert 'evoke_segment_pages_semantic_accelerator_compatible(' in context
    assert 'evoke_segment_pages_try_load_semantic_accelerator_summary(' in (
        compatible
    )
    assert 'directory_summary_is_current(&summary)' in compatible
    assert 'directory_summary_has_complete_forward(' in compatible
    assert 'has_scope_columns != has_scope' in compatible
    assert '!evoke_scope_header_is_current(&scope_header)' in compatible


def test_standby_preload_skips_unlogged_indexes() -> None:
    start = AM.index('\nevoke_am_try_auto_preload_index_oid(')
    end = AM.index(
        '\nstatic bool\nevoke_am_select_auto_preload_index(',
        start,
    )
    worker = AM[start:end]
    catalog = AM[end:AM.index(
        '\nstatic bool\nevoke_am_select_maintenance_index(',
        end,
    )]

    assert 'RecoveryInProgress()' in worker
    assert 'RELPERSISTENCE_UNLOGGED' in worker
    assert "OR c.relpersistence <> 'u'" in catalog


def test_scope_freshness_applies_only_to_indexes_with_scope_columns() -> None:
    start = AM.index('\nevoke_am_semantic_accelerator_current(')
    end = AM.index(
        '\nstatic evoke_am_accelerator_maintenance_outcome',
        start,
    )
    current = AM[start:end]

    column_check = current.index(
        'index_relation->rd_index->indnatts >'
    )
    scope_check = current.index(
        'if (current && has_scope_columns)'
    )
    open_scope = current.index(
        'evoke_segment_pages_open_semantic_accelerator_scope('
    )

    assert column_check < scope_check < open_scope
    assert 'evoke_scope_header_is_current(&scope_header)' in current


def test_unqualified_lossy_routes_remain_disabled_by_default() -> None:
    for required in (
        'bool evoke_test_semantic_accelerator_seed_bmp = false;',
        'bool evoke_test_semantic_accelerator_summarize_residual_candidates = '
        'false;',
        'double evoke_test_query_max_df_ratio = 1.0;',
        'double evoke_test_query_semantic_error_budget_ratio = 0.0;',
        'double evoke_test_query_semantic_impact_floor_ratio = 0.0;',
    ):
        assert required in PAGE_QUERY


def test_root_successors_inherit_accelerator_baseline() -> None:
    identity_start = SEGMENTS.index(
        '\nevoke_segment_manifest_build_identity(',
    )
    identity_end = SEGMENTS.index(
        '\nstatic uint64_t\nevoke_manifest_authority_u32(',
        identity_start,
    )
    identity = SEGMENTS[identity_start:identity_end]

    assert 'next_manifest->flags = old_manifest->flags;' in identity
    assert (
        'next_manifest->semantic_accelerator_directory =\n'
        '        old_manifest->semantic_accelerator_directory;'
    ) in identity
    assert (
        'evoke_segment_manifest_semantic_accelerator_baseline_sequence('
    ) in identity


def test_manifest_transitions_preserve_live_accelerator_pages() -> None:
    seal_start = AM.index('\nevoke_am_pending_seal_build(')
    seal_end = AM.index(
        '\nstatic bool\nevoke_am_publish_cow_manifest(',
        seal_start,
    )
    seal = AM[seal_start:seal_end]
    retirement_start = SEGMENT_PAGES.index(
        '\nevoke_segment_pages_prepare_retired_ranges(',
    )
    retirement_end = SEGMENT_PAGES.index(
        '\ntypedef struct evoke_term_cow_page_loader_context',
        retirement_start,
    )
    retirement = SEGMENT_PAGES[retirement_start:retirement_end]

    assert 'EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR);' in seal
    assert (
        'next_manifest->semantic_accelerator_directory =\n'
        '        old_manifest->semantic_accelerator_directory;'
    ) in seal
    assert (
        '!evoke_segment_object_refs_equal(\n'
        '             &old_manifest->semantic_accelerator_directory,\n'
        '             &next_manifest->semantic_accelerator_directory)'
    ) in retirement


def test_stale_accelerator_children_keep_source_authority() -> None:
    helper_start = SEGMENT_PAGES.index(
        '\nevoke_segment_pages_accelerator_artifact_authority(',
    )
    helper_end = SEGMENT_PAGES.index(
        '\nevoke_status\nevoke_segment_pages_try_load_',
        helper_start,
    )
    helper = SEGMENT_PAGES[helper_start:helper_end]
    tid_start = SEGMENT_PAGES.index(
        '\nevoke_segment_pages_load_semantic_accelerator_tid_lookup(',
    )
    tid_end = SEGMENT_PAGES.index(
        '\nvoid\nevoke_segment_pages_read_semantic_accelerator_scope_range(',
        tid_start,
    )
    tid_lookup = SEGMENT_PAGES[tid_start:tid_end]

    assert '*authority_checksum_out = source_authority_checksum;' in helper
    assert 'source_authority_checksum != 0' in SEGMENT_PAGES
    assert 'summary.source_authority_checksum' in tid_lookup
    assert (
        'authority_checksum,\n'
        '            0,\n'
        '            summary.document_count,'
    ) in tid_lookup


def test_stale_scope_uses_a_post_baseline_membership_residual() -> None:
    scope_start = FILTER_SOURCE.index('\nevoke_filter_try_scope_bitmap(')
    scope = FILTER_SOURCE[scope_start:]
    mark_start = AM.index('\nevoke_am_tid_filter_mark_slot(')
    mark_end = AM.index(
        '\nstatic void\nevoke_am_tid_filter_mark_document(',
        mark_start,
    )
    mark = AM[mark_start:mark_end]

    assert 'root->active_l0.record_count > 0' in scope
    assert 'root->pending_l0.record_count > 0' in scope
    assert (
        'evoke_segment_manifest_semantic_accelerator_baseline_sequence('
        in scope
    )
    assert '*fully_resolved_out = false;' in scope
    assert (
        '(uint64) document_slot <\n'
        '            cleanup->tid_filter.scope_document_slot_count'
    ) in mark
    assert 'projected_document->shadows_immutable' in mark
    assert (
        '(uint64) document_slot >=\n'
        '            cleanup->tid_filter.scope_document_slot_count'
    ) not in mark


def test_stale_accelerator_is_a_bounded_approximate_serving_bridge() -> None:
    query_start = PAGE_QUERY.index(
        '\nevoke_page_query_positive_topk_filtered('
    )
    query_end = PAGE_QUERY.index(
        '\nevoke_page_query_positive_topk_projected(',
        query_start,
    )
    query = PAGE_QUERY[query_start:query_end]
    attempt_start = PAGE_QUERY.index(
        '\nevoke_page_query_try_semantic_accelerator(',
    )
    attempt_end = PAGE_QUERY.index(
        '\nstatic size_t\nevoke_page_query_stale_accelerator_k(',
        attempt_start,
    )
    attempt = PAGE_QUERY[attempt_start:attempt_end]
    filter_start = PAGE_QUERY.index(
        '\nstatic void\nevoke_page_query_filter_accelerator_baseline('
    )
    filter_end = PAGE_QUERY.index(
        '\nevoke_page_query_positive_topk_filtered(',
        filter_start,
    )
    stale_filter = PAGE_QUERY[filter_start:filter_end]

    assert 'evoke_page_query_filter_accelerator_baseline(' in query
    stale = query.index('if (!accelerator_is_current)')
    validate = query.index(
        'evoke_page_query_filter_accelerator_baseline(',
        stale,
    )
    complete = query.index(
        'stats.semantic_accelerator_stale_baseline = true;',
        validate,
    )
    direct_return = query.index('return EVOKE_OK;', complete)
    exact_scan = query.index(
        'evoke_segment_pages_load_query_term_plan_after_sequence(',
        direct_return,
    )
    assert stale < validate < complete < direct_return < exact_scan
    assert 'evoke_segment_pages_load_document_record(' in stale_filter
    assert 'record.version.born_sequence > baseline_sequence' in stale_filter
    assert 'projected_document->shadows_immutable' in stale_filter
    assert 'baseline_delta_mode = projection != NULL' in attempt
    assert (
        'if (!baseline_delta_mode)\n'
        '        {\n'
        '            residual_term_count +='
    ) in attempt
    assert (
        'if (!baseline_delta_mode &&\n'
        '        result_out->len < k'
    ) in attempt
    assert 'semantic_accelerator_delta_overlay' not in PAGE_QUERY


def test_stale_accelerator_has_a_query_relative_single_pass_bound() -> None:
    helper_start = PAGE_QUERY.index(
        '\nstatic size_t\nevoke_page_query_stale_accelerator_k('
    )
    helper_end = PAGE_QUERY.index(
        '\nstatic void\nevoke_page_query_filter_accelerator_baseline(',
        helper_start,
    )
    helper = PAGE_QUERY[helper_start:helper_end]
    query_start = PAGE_QUERY.index(
        '\nevoke_page_query_positive_topk_filtered('
    )
    query_end = PAGE_QUERY.index(
        '\nevoke_status\nevoke_page_query_positive_topk(',
        query_start,
    )
    query = PAGE_QUERY[query_start:query_end]

    assert (
        '#define EVOKE_PAGE_QUERY_ACCELERATOR_STALE_OVERFETCH_FLOOR '
        'UINT64_C(64)'
    ) in PAGE_QUERY
    assert (
        '#define EVOKE_PAGE_QUERY_ACCELERATOR_STALE_OVERFETCH_MAX '
        'UINT64_C(4096)'
    ) in PAGE_QUERY
    assert 'overfetch_limit = Max(' in helper
    assert 'overfetch_limit = Min(' in helper
    assert 'changed_documents = Min(changed_documents, overfetch_limit);' in (
        helper
    )
    assert query.count('evoke_page_query_try_semantic_accelerator(') == 1


def test_explicit_maintenance_allows_low_debt_checkpoint() -> None:
    maintain_start = AM.index(
        'PG_FUNCTION_INFO_V1(evoke_maintain_index);'
    )
    maintain_end = AM.index(
        'typedef enum evoke_am_accelerator_maintenance_outcome',
        maintain_start,
    )
    maintain_api = AM[maintain_start:maintain_end]
    try_start = AM.index(
        'PG_FUNCTION_INFO_V1(evoke_try_maintain_index);'
    )
    try_end = AM.index(
        'PG_FUNCTION_INFO_V1(evoke_touch_maintenance);',
        try_start,
    )
    try_api = AM[try_start:try_end]
    explicit_call = ''.join(
        'evoke_am_try_maintain_index_oid('
        'index_oid,false,true,true,false)'.split()
    )

    assert explicit_call in ''.join(maintain_api.split())
    assert explicit_call in ''.join(try_api.split())


def test_periodic_accelerator_refresh_yields_urgent_maintenance_lock() -> None:
    refresh_start = AM.index(
        '\nevoke_am_convergent_accelerator_refresh_state('
    )
    due_start = AM.index('\nevoke_am_convergent_accelerator_due(')
    refresh = AM[refresh_start:due_start]
    due_end = AM.index(
        '\nstatic bool\nevoke_am_convergent_structural_fold_due(',
        due_start,
    )
    due = AM[due_start:due_end]
    candidate_start = AM.index('\nevoke_am_get_background_due_candidate(')
    candidate_end = AM.index(
        '\nstatic int\nevoke_am_maintenance_action_tier(',
        candidate_start,
    )
    candidate = AM[candidate_start:candidate_end]
    maintain_start = AM.index('\nevoke_am_try_maintain_index_oid(')
    maintain_end = AM.index(
        '\nPG_FUNCTION_INFO_V1(evoke_try_maintain_index);',
        maintain_start,
    )
    maintain = AM[maintain_start:maintain_end]
    publish_start = AM.index(
        '\nevoke_am_try_publish_accelerator_baseline('
    )
    publish_end = AM.index(
        '\nstatic text *\nevoke_am_try_accelerator_maintenance_result(',
        publish_start,
    )
    publish = AM[publish_start:publish_end]
    mutation_start = AM_MUTATION.index(
        '\nevoke_am_mutation_publish_cow_manifest('
    )
    mutation_end = AM_MUTATION.index(
        '\nXLogRecPtr\nevoke_am_l0_rotate_active_locked(',
        mutation_start,
    )
    mutation = AM_MUTATION[mutation_start:mutation_end]

    assert 'root->pending_l0.record_count != 0' in refresh
    assert 'state_out->baseline_sequence < manifest->max_sequence' in refresh
    assert 'evoke_am_convergent_accelerator_delta_debt(' in refresh
    assert 'EVOKE_AM_ACCELERATOR_REFRESH_BYTES' in refresh
    assert 'GetCurrentTimestamp' not in refresh
    assert 'root->active_l0' not in refresh
    assert 'state_out->periodic_eligible = state_out->compatible &&' in refresh
    assert 'state_out->due = !state_out->compatible ||' in refresh
    assert 'evoke_am_convergent_accelerator_refresh_state(' in due
    assert 'evoke_am_semantic_accelerator_compatible(' in due
    assert 'allow_periodic_low_debt && refresh.periodic_eligible' in due
    semantic = candidate.index(
        'semantic_due = evoke_am_convergent_semantic_completion_due('
    )
    accelerator = candidate.index(
        'semantic_accelerator_due =',
        semantic,
    )
    active_only = candidate.index(
        'if (root.active_l0.record_count == 0)',
        accelerator,
    )
    assert semantic < accelerator < active_only
    assert 'evoke_am_active_l0_checkpoint_due(&root)' in candidate
    assert 'active_l0_low_debt = root.active_l0.record_count > 0 &&' in candidate
    assert 'due = strict_due ||' in candidate
    assert 'allow_periodic_low_debt && low_debt' in candidate
    assert '!strict_due && allow_periodic_low_debt && low_debt' in candidate
    assert 'candidate_out->action_class = !strict_due' in candidate
    assert 'semantic_accelerator_strict_due_raw' in candidate
    assert 'allow_accelerator_build &&' in candidate
    assert 'due = strict_due || semantic_accelerator_due ||' in candidate
    assert 'candidate_out->allow_accelerator_build =' in candidate
    assert 'candidate_out->prefer_accelerator_build =' in candidate
    assert 'if (!semantic_due)' not in candidate
    completion = maintain.index('evoke_am_complete_convergent_semantic(')
    preferred = maintain.index('if (prefer_accelerator_build &&')
    accelerator_label = maintain.index('accelerator_maintenance:')
    checkpoint = maintain.index('reason=accelerator_root_checkpoint')
    assert 'immutable_root_work_deferred' in maintain
    assert preferred < completion < accelerator_label
    assert '!accelerator_build_in_progress' in maintain[
        preferred:completion
    ]
    assert completion < checkpoint
    assert 'semantic_transition_rotated' not in maintain
    rotate = maintain.index('evoke_am_rotate_convergent_active_l0(')
    accelerator_build_lock = maintain.index(
        'evoke_am_try_accelerator_build_lock(',
        rotate,
    )
    maintenance_unlock = maintain.index(
        'evoke_am_maintenance_unlock(index_oid);',
        accelerator_build_lock,
    )
    reservation_unlock = maintain.index(
        'evoke_am_session_maintenance_unlock(index_oid);',
        accelerator_build_lock,
    )
    accelerator_build = maintain.index(
        'evoke_am_try_accelerator_maintenance_result(',
        maintenance_unlock,
    )
    accelerator_unlock = maintain.index(
        'evoke_am_accelerator_build_unlock(index_oid);',
        accelerator_build,
    )
    assert rotate < accelerator_build_lock
    assert accelerator_build_lock < reservation_unlock < maintenance_unlock
    assert maintenance_unlock < accelerator_build
    assert accelerator_build < accelerator_unlock
    assert maintain.count('evoke_am_try_accelerator_maintenance_result(') == 1
    assert 'release_worker_reservation_for_accelerator' in maintain
    assert 'evoke_am_work_hint_defer_accelerator(index_oid);' in maintain
    assert '!allow_accelerator_build ||' in maintain
    assert (
        'evoke_am_try_maintain_index_oid(\n'
        '        candidate->index_oid,\n'
        '        true'
    ) in AM
    assert (
        '&cow_result,\n'
        '                false,\n'
        '                false,\n'
        '                !defer_fsm_handoff'
    ) in publish
    assert (
        'evoke_am_try_accelerator_maintenance_result(\n'
        '                    index_oid,\n'
        '                    indexRelation'
    ) in maintain[rotate:]
    assert (
        publish.count(
            '!evoke_am_accelerator_baseline_has_visible_documents('
        )
        == 2
    )
    assert 'manifest->visible_document_count > 0' in AM
    assert 'manifest->total_document_length > 0' in AM
    for checked_identity in (
        'latest_root.root_id != build_root->root_id',
        'latest_root.next_segment_id !=',
        '&latest_root.manifest,',
        '&build_root->manifest)',
    ):
        assert checked_identity in mutation
    assert 'next_root = latest_root;' in mutation
    assert 'evoke_segment_read_root_replace_manifest(' in mutation


def test_accelerator_retry_cooldown_is_per_index_and_nonblocking() -> None:
    assert '#define EVOKE_AM_SCHEDULER_VERSION 3' in AM_SCHEDULER
    assert 'TimestampTz accelerator_retry_after;' in AM_SCHEDULER_HEADER
    assert 'evoke_am_scheduler_work_hint_defer_accelerator(' in AM_SCHEDULER
    assert 'hint->accelerator_retry_after = retry_after;' in AM_SCHEDULER
    assert (
        'token->accelerator_retry_after <= GetCurrentTimestamp()' in AM
    )
    assert (
        'evoke_am_work_hint_accelerator_retry_allowed(hint_token)' in AM
    )
    assert 'candidate->allow_accelerator_build' in AM
    assert 'candidate->prefer_accelerator_build' in AM
    assert 'evoke_am_scheduler_work_hint_allow_accelerator_now(' in (
        AM_SCHEDULER
    )
    assert 'evoke_am_work_hint_allow_accelerator_now(index_oid);' in AM


def test_active_ingress_prewarm_keeps_the_readable_baseline_warm() -> None:
    start = AM.index('\nevoke_am_try_auto_preload_index_oid(')
    end = AM.index(
        '\nstatic bool\nevoke_am_select_auto_preload_index(',
        start,
    )
    worker = AM[start:end]
    due = worker.index('evoke_am_convergent_accelerator_due(')

    assert 'root.active_l0.record_count == 0' in worker[:due]
    assert 'root.pending_l0.record_count == 0' in worker[:due]
    generation_start = AM.index(
        '\nevoke_am_prewarm_convergent_generation_pages('
    )
    generation_end = AM.index(
        '\nstatic void\nevoke_am_rebuild_workload_from_convergent(',
        generation_start,
    )
    generation = AM[generation_start:generation_end]
    assert 'evoke_segment_manifest_semantic_accelerator_eligible(' in (
        generation
    )
    assert 'evoke_am_convergent_accelerator_due(' not in generation


def test_term_fold_closes_the_accelerator_baseline_before_advancing() -> None:
    helper_start = AM.index(
        '\nevoke_am_term_fold_boundary_requires_promotion('
    )
    helper_end = AM.index(
        '\ntypedef struct evoke_am_term_structural_fold_cleanup',
        helper_start,
    )
    helper = AM[helper_start:helper_end]
    workload_start = AM.index('\nevoke_am_term_workload_fold_select(')
    workload_end = AM.index(
        '\nevoke_am_term_structural_fold_load_term_extents(',
        workload_start,
    )
    workload = AM[workload_start:workload_end]

    assert 'neutral_minor_fold_coverage >' in helper
    assert 'max_sequence >\n            accelerator_baseline_sequence' in helper
    assert 'evoke_am_term_fold_boundary_requires_promotion(' in workload
    assert AM.count('evoke_am_term_fold_boundary_requires_promotion(') >= 3


def test_retirement_and_reachability_own_every_accelerator_child() -> None:
    retirement_start = SEGMENT_PAGES.index(
        '\nevoke_segment_pages_prepare_retired_ranges(',
    )
    retirement_end = SEGMENT_PAGES.index(
        '\ntypedef struct evoke_term_cow_page_loader_context',
        retirement_start,
    )
    retirement = SEGMENT_PAGES[retirement_start:retirement_end]
    reachability_start = SEGMENT_PAGES.index(
        '\nevoke_segment_pages_inventory_reachable(',
    )
    reachability_end = SEGMENT_PAGES.index(
        '\nstatic void\nevoke_segment_pages_load_sealed_manifest_internal(',
        reachability_start,
    )
    reachability = SEGMENT_PAGES[reachability_start:reachability_end]

    assert (
        'old_manifest->semantic_accelerator_directory.start_block'
        in retirement
    )
    assert 'manifest->semantic_accelerator_directory' in reachability
    for child in (
        'accelerator_directory.terms[',
        'accelerator_directory.forward_chunks[',
        'accelerator_directory.forward_bound_shards[',
        'accelerator_directory.scope_object',
        'accelerator_directory.tid_lookup_object',
    ):
        assert child in retirement
        assert child in reachability
    assert 'evoke_block_range_inventory_add(' in retirement
    assert 'evoke_segment_pages_inventory_add_object_ref(' in reachability


def test_durable_tid_lookup_uses_the_accelerator_root_lifecycle() -> None:
    accelerator = (
        ROOT / 'src' / 'evoke_am_accelerator.c'
    ).read_text(encoding='utf-8')

    for required in (
        'evoke_am_accelerator_build_tid_lookup(',
        'source->document_tids',
        'source.tid_lookup_bytes,',
        'source.tid_lookup_size,',
    ):
        assert required in accelerator
    for required in (
        'EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_TID_LOOKUP',
        'directory.tid_lookup_object',
        'evoke_segment_pages_load_semantic_accelerator_tid_lookup(',
    ):
        assert required in SEGMENT_PAGES
    assert 'evoke_segment_pages_load_semantic_accelerator_tid_lookup(' in AM
    assert 'cleanup->tid_filter.durable_lookup.bytes != NULL' in AM


def test_large_single_accelerator_retirement_is_reclaimable() -> None:
    for required in (
        '#define EVOKE_AM_RECLAIM_RETIRED_RANGE_THRESHOLD UINT32_C(8)',
        '#define EVOKE_AM_RECLAIM_ACTION_BLOCK_BUDGET UINT32_C(8192)',
        '#define EVOKE_AM_RECLAIM_RETIRED_BLOCK_THRESHOLD',
        'evoke_am_retired_reclamation_due(',
        'range_blocks >= EVOKE_AM_RECLAIM_RETIRED_BLOCK_THRESHOLD -',
        'if (!evoke_am_retired_reclamation_due(&old_manifest))',
        'evoke_am_acquire_convergent_reader_fence(',
        'EVOKE_AM_RECLAIM_ACTION_BLOCK_BUDGET,',
        'due = evoke_am_retired_reclamation_due(&manifest);',
    ):
        assert required in AM

    for required in (
        'evoke_segment_pages_prepare_cow_reclaim_ranges(',
        'owner_manifest_id <= manifest->manifest_id',
        'evoke_flag_matches_inherited_object_ref(',
    ):
        assert required in SEGMENT_PAGES or required in SEGMENTS


def test_residual_workspace_adapts_without_exceeding_the_query_budget() -> None:
    for required in (
        'evoke_page_query_accumulate_residual_candidates_exact(',
        'evoke_page_query_accumulate_residual_candidates_dense(',
        'evoke_page_query_accumulate_residual_candidates_merge(',
        'evoke_page_query_prefer_dense_residual_accumulation(',
        'evoke_page_query_residual_merge_heap_push(',
        'evoke_page_query_residual_merge_heap_pop(',
        'merge->cursor_count,',
        'merge = &resources->exact_residual_merge;',
        'residual_merge_peak_bytes',
        '> EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT',
        'dense_work <= merge_work',
        'document_count > UINT32_MAX',
        'if (!residual_merge_available)',
        'evoke_test_semantic_accelerator_summarize_residual_candidates',
    ):
        assert required in PAGE_QUERY

    assert 'float *residual_candidate_scores' in PAGE_QUERY
    assert 'free(resources->residual_candidate_scores);' in PAGE_QUERY
    assert 'EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT / sizeof(float)' not in (
        PAGE_QUERY
    )


def test_residual_lexical_candidates_use_the_product_bm25_score() -> None:
    for required in (
        'residual_lexical_neutral',
        'context->resident_document_lengths == NULL',
        'context->resident_document_length_count !=',
        'evoke_score_tfc(',
        'context->resident_document_lengths[',
        'residual_idf * tfc',
    ):
        assert required in PAGE_QUERY

    assert 'evoke_am_attach_page_native_document_lengths(' in AM
    assert '&cleanup->document_lengths_lease' in AM


def test_fixed_root_filter_reuses_only_same_generation_shared_lengths() -> None:
    for required in (
        'resident_root.root_id == fixed_root->root_id',
        'evoke_segment_object_ref_equal(',
        '&resident_root.manifest,',
        '&fixed_root->manifest',
        'evoke_am_attach_page_native_document_lengths(',
        '&resident_meta,',
        '&cleanup->document_lengths_lease',
    ):
        assert required in AM


def test_scope_format_upgrade_uses_existing_maintenance_lifecycle() -> None:
    assert '#define EVOKE_SCOPE_CURRENT_VERSION 6U' in SCOPE_HEADER
    assert 'uint32_t gram_mask;' in SCOPE_HEADER
    assert 'evoke_scope_header_is_current(&scope_header)' in AM
    assert 'evoke_segment_pages_open_semantic_accelerator_scope(' in AM
    assert 'evoke_filter_scope_value_gram_matches(' in FILTER_SOURCE
    assert 'value_gram_pruning' in FILTER_SOURCE


def test_scope_publication_has_exact_memory_admission() -> None:
    accelerator = (
        ROOT / 'src' / 'evoke_am_accelerator.c'
    ).read_text(encoding='utf-8')

    for required in (
        'scope_output_budget,',
        '&scope_required_size,',
        '*scope_required_bytes_out = scope_required_size;',
        'scope_required_size > scope_output_budget',
        '*memory_blocked_out = true;',
    ):
        assert required in accelerator
    for required in (
        'scope_output_budget = *budget_bytes_out -',
        '*estimated_bytes_out = evoke_u64_saturating_add(',
        'scope_required_bytes',
        'reason_out = "accelerator_scope_memory_budget";',
        '"reason=accelerator_scope_memory_budget"',
    ):
        assert required in AM


def test_accelerator_build_bounds_its_mvcc_horizon() -> None:
    accelerator = (
        ROOT / 'src' / 'evoke_am_accelerator.c'
    ).read_text(encoding='utf-8')
    source_start = accelerator.index(
        '\nevoke_am_accelerator_source_build_query_context('
    )
    source_end = accelerator.index(
        '\nstatic evoke_status\nevoke_am_accelerator_produce_term(',
        source_start,
    )
    source = accelerator[source_start:source_end]
    scope_start = SCOPE_PG.index(
        '\nevoke_scope_pg_collect_document_batch('
    )
    scope = SCOPE_PG[scope_start:]

    assert 'InvalidateCatalogSnapshot();' in source
    assert (
        '"evoke maintenance: prepare semantic query accelerator"'
        in source
    )
    assert 'InvalidateCatalogSnapshot();' in scope
    assert (
        '#define EVOKE_SCOPE_SNAPSHOT_BATCH_DOCUMENTS UINT32_C(256)'
        in SCOPE_PG
    )
    assert 'PushActiveSnapshot(GetLatestSnapshot());' in scope
    assert 'GetActiveSnapshot(),' in scope
    assert 'PopActiveSnapshot();' in scope
    assert 'volatile bool snapshot_pushed = false;' in scope
    assert 'SnapshotAny' not in scope
    assert (
        '"evoke maintenance: prepare semantic query accelerator scope"'
        in scope
    )


def test_scope_publication_releases_external_memory_on_pg_error() -> None:
    start = SCOPE_PG.index('\nevoke_scope_build_for_index(')
    build = SCOPE_PG[start:]

    assert 'writer = palloc0(sizeof(*writer));' in build
    assert 'columns = palloc0(' in build
    assert 'serialized_columns = palloc0(' in build
    assert 'PG_FINALLY();' in build
    assert 'evoke_scope_writer_free(writer);' in build
    assert 'pfree(writer);' in build
    assert 'copy = malloc(' not in SCOPE_PG
    assert 'value->gram_value = malloc(' not in SCOPE_PG


def test_accelerator_build_temporaries_use_pg_memory_contexts() -> None:
    finalize_start = ACCELERATOR_PG.index(
        '\nevoke_am_accelerator_finalize_forward('
    )
    finalize_end = ACCELERATOR_PG.index(
        '\nstatic evoke_status\nevoke_am_accelerator_read_document(',
        finalize_start,
    )
    finalize = ACCELERATOR_PG[finalize_start:finalize_end]
    source_start = ACCELERATOR_PG.index(
        '\nevoke_am_accelerator_source_build_query_context('
    )
    source_end = ACCELERATOR_PG.index(
        '\nstatic evoke_status\nevoke_am_accelerator_produce_term(',
        source_start,
    )
    source = ACCELERATOR_PG[source_start:source_end]
    tid_start = ACCELERATOR_PG.index(
        '\nevoke_am_accelerator_build_tid_lookup('
    )
    tid_end = ACCELERATOR_PG.index(
        '\nbool\nevoke_am_prepare_accelerator_baseline(',
        tid_start,
    )
    tid = ACCELERATOR_PG[tid_start:tid_end]

    for block in (finalize, source, tid):
        assert 'palloc' in block
        assert 'MCXT_ALLOC_HUGE' in block
    for temporary in (
        'selected_map = malloc(',
        'retained_documents = calloc(',
        'document_lengths = calloc(',
        'document_frequencies = calloc(',
        'pairs = malloc(',
        'keys = malloc(',
        'slots = malloc(',
    ):
        assert temporary not in ACCELERATOR_PG


def test_forward_bound_publication_uses_sparse_logical_tapes() -> None:
    flush_start = ACCELERATOR_PG.index(
        '\nevoke_am_accelerator_flush_forward_bound_block('
    )
    start = ACCELERATOR_PG.index(
        '\nevoke_am_accelerator_finalize_forward('
    )
    end = ACCELERATOR_PG.index(
        '\nstatic evoke_status\nevoke_am_accelerator_read_document(',
        start,
    )
    finalize = ACCELERATOR_PG[start:end]

    assert 'LogicalTapeSetCreate(false, NULL, -1)' in finalize
    assert 'LogicalTapeSetCreate(true' not in finalize
    assert 'tuplesort_begin' not in finalize
    assert 'LogicalTapeWrite(' in ACCELERATOR_PG[flush_start:start]
    assert 'evoke_am_accelerator_write_forward_bound_file(' in finalize


def test_optional_accelerator_rejects_obsolete_query_authority() -> None:
    pages_header = (
        ROOT / 'src' / 'evoke_segment_pages.h'
    ).read_text(encoding='utf-8')
    helper = 'evoke_segment_pages_query_authority_format_status'

    assert AM.count(helper) >= 2
    assert 'accelerator_source_reindex_required' in AM
    assert helper in pages_header
    assert helper in SEGMENT_PAGES
    assert 'evoke_segment_pages_read_fold_header' in SEGMENT_PAGES
    assert 'evoke_segment_pages_read_payload_header' in SEGMENT_PAGES
    assert 'term_record.neutral_fold_coverage' in SEGMENT_PAGES
    assert 'term_record.neutral_minor_fold_coverage' in SEGMENT_PAGES
