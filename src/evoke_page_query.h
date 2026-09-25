#ifndef EVOKE_PAGE_QUERY_H
#define EVOKE_PAGE_QUERY_H

#include "postgres.h"

#include "utils/rel.h"

#include "evoke_core.h"
#include "evoke_segment_pages.h"

#define EVOKE_PAGE_QUERY_BMP_DEFAULT_SUPER_BATCH UINT32_C(1)
#define EVOKE_PAGE_QUERY_BMP_SUPER_BATCH UINT32_C(8)

typedef struct evoke_page_query_stats
{
    uint64 materialized_query_bytes;
    uint64 term_at_a_time_score_bytes;
    uint64 term_at_a_time_document_length_bytes;
    uint64 semantic_bmp_query_bytes;
    uint64 posting_block_metadata_reads;
    uint64 posting_block_metadata_cache_bytes;
    uint64 posting_block_reads;
    uint64 document_block_reads;
    uint64 blocks_considered;
    uint64 blocks_scored;
    uint64 blocks_skipped;
    uint64 postings_skipped;
    uint64 postings_examined;
    uint64 semantic_bmp_super_ref_reads;
    uint64 semantic_bmp_query_super_ref_count;
    uint64 semantic_bmp_filtered_sequential_scans;
    uint64 semantic_bmp_ref_reads;
    uint64 filtered_bmp_matching_ref_count;
    uint64 filtered_bmp_matching_super_ref_count;
    uint64 semantic_bmp_record_reads;
    uint64 semantic_bmp_postings_examined;
    uint64 filtered_bmp_allowed_blocks;
    uint64 filtered_bmp_allowed_superblocks;
    uint64 query_df_pruned_postings;
    uint64 query_error_budget_pruned_postings;
    uint64 query_impact_floor_omitted_postings;
    uint64 accelerator_forward_chunk_reads;
    uint64 accelerator_forward_row_reads;
    uint64 accelerator_forward_postings_examined;
    uint64 accelerator_forward_bytes;
    uint64 filtered_forward_sparse_oracle_full_bytes;
    uint64 filtered_forward_sparse_oracle_selected_bytes;
    uint64 filtered_forward_sparse_oracle_ranges;
    uint64 filtered_forward_sparse_oracle_full_pages;
    uint64 filtered_forward_sparse_oracle_selected_pages;
    uint64 filtered_forward_direct_work;
    uint64 filtered_forward_transpose_work;
    uint64 filtered_forward_direct_estimated_bytes;
    uint64 filtered_forward_transpose_estimated_bytes;
    uint64 filtered_forward_bound_estimated_bytes;
    uint64 filtered_forward_allowed_b8_blocks;
    uint64 filtered_forward_allowed_b8_runs;
    uint64 filtered_forward_allowed_b8_span;
    uint64 filtered_forward_allowed_b8_first;
    uint64 filtered_forward_allowed_b8_last;
    uint64 filtered_forward_allowed_b64_ranges;
    uint64 filtered_forward_allowed_b512_ranges;
    uint64 filtered_forward_allowed_b4096_ranges;
    uint64 filtered_forward_transpose_planning_bytes;
    uint64 filtered_forward_transpose_stream_bytes;
    uint64 filtered_forward_transpose_lane_bytes;
    uint64 filtered_forward_bound_bytes;
    uint64 filtered_forward_bound_entries;
    uint64 filtered_forward_bound_blocks_considered;
    uint64 filtered_forward_bound_blocks_scored;
    uint64 semantic_accelerator_query_bytes;
    uint64 accelerator_owned_index_bytes;
    uint64 accelerator_membership_bytes;
    uint64 accelerator_candidate_scratch_bytes;
    uint64 accelerator_forward_scratch_bytes;
    uint64 accelerator_residual_scratch_bytes;
    uint64 accelerator_clusters_opened;
    uint64 accelerator_clusters_skipped;
    uint64 accelerator_document_refs_examined;
    uint64 accelerator_documents_scored;
    uint64 accelerator_residual_postings;
    uint64 accelerator_residual_documents;
    uint64 accelerator_residual_summary_capacity;
    uint64 accelerator_residual_summary_counters;
    uint64 accelerator_residual_summary_replacements;
    uint64 accelerator_baseline_sequence;
    uint64 accelerator_baseline_document_slots;
    uint64 accelerator_stale_candidates_rejected;
    uint64 documents_examined;
    uint64 positive_document_count;
    uint64 zero_score_documents_added;
    uint64 zero_score_cow_objects_loaded;
    uint64 zero_score_cow_records_examined;
    uint64 zero_score_heap_peak;
    uint64 ranked_prefix_probe_documents_examined;
    uint64 ranked_prefix_probe_postings_examined;
    uint64 ranked_prefix_probe_memory_bytes;
    uint32 query_term_count;
    uint32 accelerator_directory_term_count;
    uint32 accelerator_matched_term_count;
    uint32 query_run_count;
    uint32 visibility_rank_attempts;
    uint32 ranked_prefix_probe_attempts;
    uint32 ranked_prefix_probe_query_term_count;
    uint32 ranked_prefix_probe_directory_term_count;
    uint32 ranked_prefix_probe_matched_term_count;
    uint32 query_df_pruned_term_count;
    uint32 query_error_budget_pruned_term_count;
    uint32 max_document_block_records;
    double query_semantic_total_absolute_bound;
    double query_semantic_omitted_absolute_bound;
    double accelerator_residual_summary_total_weight;
    double accelerator_residual_summary_maximum_error;
    float nonoccurrence_base_score;
    bool materialized_query_path;
    bool shared_resident_fold_query_path;
    bool term_at_a_time_query_path;
    bool ordered_block_query_path;
    bool semantic_bmp_attempted;
    bool semantic_bmp_admission_skipped;
    bool semantic_bmp_fallback;
    bool semantic_bmp_query_path;
    bool semantic_bmp_direct_flat_filtered;
    bool semantic_accelerator_attempted;
    bool semantic_accelerator_fallback;
    bool semantic_accelerator_query_path;
    bool semantic_accelerator_block_major;
    bool semantic_accelerator_forward_direct_rows;
    bool semantic_accelerator_forward_transposed;
    bool semantic_accelerator_forward_bounded;
    bool filtered_forward_transpose_budget_exceeded;
    bool filtered_forward_bound_budget_exceeded;
    bool filtered_forward_bound_probe_fallback;
    bool ranked_prefix_query_path;
    bool ranked_prefix_probe_fallback;
    bool semantic_accelerator_residual_candidates_bounded;
    bool semantic_accelerator_residual_candidates_accumulated;
    bool semantic_accelerator_residual_candidates_summarized;
    bool semantic_accelerator_residual_dense_accumulation;
    bool semantic_accelerator_stale_baseline;
    bool positive_topk_complete;
    bool topk_complete;
} evoke_page_query_stats;

typedef enum evoke_filtered_forward_route
{
    EVOKE_FILTERED_FORWARD_ROUTE_AUTO = 0,
    EVOKE_FILTERED_FORWARD_ROUTE_DIRECT,
    EVOKE_FILTERED_FORWARD_ROUTE_TRANSPOSE,
    EVOKE_FILTERED_FORWARD_ROUTE_HYBRID,
    EVOKE_FILTERED_FORWARD_ROUTE_BOUND
} evoke_filtered_forward_route;

extern bool evoke_test_disable_semantic_bmp;
extern bool evoke_test_force_semantic_bmp;
extern bool evoke_test_disable_fused_semantic_taat;
extern int evoke_test_filtered_forward_route;
extern double evoke_test_query_max_df_ratio;
extern int evoke_test_query_semantic_work_target_postings;
extern double evoke_test_query_semantic_error_budget_ratio;
extern double evoke_test_query_semantic_impact_floor_ratio;
extern double evoke_test_query_semantic_min_support_ratio;
extern int evoke_test_query_semantic_bmp_super_batch;
extern bool evoke_semantic_accelerator_disabled;
extern bool evoke_test_semantic_accelerator_seed_bmp;
extern double evoke_semantic_accelerator_heap_factor;
extern int evoke_semantic_accelerator_candidate_multiplier;
extern bool evoke_semantic_accelerator_bound_residual_candidates;
extern bool evoke_semantic_accelerator_accumulate_residual_candidates;
extern bool evoke_test_semantic_accelerator_summarize_residual_candidates;
extern int evoke_test_semantic_accelerator_summary_multiplier;

#define EVOKE_PAGE_QUERY_COST_BASE_LEVEL_COUNT 5
#define EVOKE_PAGE_QUERY_COST_LEVEL_COUNT 13

typedef struct evoke_page_query_cost_level
{
    uint64 block_count;
    uint64 touched_term_blocks;
    uint64 total_postings;
    uint64 competitive_blocks;
    uint64 competitive_postings;
    uint64 competitive_document_slots;
    uint64 bound_bytes;
    uint8 block_shift;
} evoke_page_query_cost_level;

typedef struct evoke_page_query_cost_plan
{
    uint64 nodes_read;
    uint64 sparse_term_hits;
    uint64 dense_term_probes;
    uint16 level_mask;
} evoke_page_query_cost_plan;

#define EVOKE_PAGE_QUERY_COST_MAX_ESSENTIAL_PROJECTIONS 20

typedef struct evoke_page_query_cost_essential_projection
{
    uint32 essential_semantic_terms;
    uint32 residual_semantic_terms;
    uint64 essential_semantic_super_refs;
    uint64 essential_semantic_refs;
    uint64 essential_semantic_postings;
    uint64 competitive_blocks;
    uint64 competitive_postings;
    uint64 competitive_document_slots;
    uint64 missed_full_competitive_blocks;
    float residual_global_cap;
    float residual_global_floor;
    float residual_global_absolute_cap;
    float kth_lower_bound;
    bool kth_lower_bound_safe;
} evoke_page_query_cost_essential_projection;

typedef struct evoke_page_query_cost_audit
{
    evoke_page_query_cost_level levels[EVOKE_PAGE_QUERY_COST_LEVEL_COUNT];
    uint64 document_slot_count;
    uint64 query_postings;
    uint64 hierarchy_nodes_read;
    uint64 hierarchy_sparse_term_hits;
    uint64 hierarchy_dense_term_probes;
    uint64 hierarchy_leaf_blocks;
    uint64 hierarchy_leaf_postings;
    evoke_page_query_cost_plan optimal_sparse_plan;
    evoke_page_query_cost_plan optimal_dense_plan;
    evoke_page_query_cost_essential_projection essential_projections[
        EVOKE_PAGE_QUERY_COST_MAX_ESSENTIAL_PROJECTIONS
    ];
    uint32 query_term_count;
    uint32 query_run_count;
    uint32 mandatory_term_count;
    uint32 projectable_semantic_term_count;
    uint32 essential_projection_count;
    uint64 mandatory_touched_b16_blocks;
    float kth_score;
} evoke_page_query_cost_audit;

#define EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT 3
#define EVOKE_PAGE_QUERY_FORWARD_BOUND_AUDIT_MAX_BYTES \
    (UINT64_C(256) * UINT64_C(1024) * UINT64_C(1024))

typedef struct evoke_page_query_forward_bound_level
{
    uint64 block_count;
    uint64 bound_entries;
    uint64 projected_bound_bytes;
    uint64 addressable_storage_bytes;
    uint64 addressable_query_metadata_bytes;
    uint64 addressable_selected_bound_entries;
    uint64 addressable_selected_bound_bytes;
    uint64 addressable_selected_bound_windows;
    uint64 addressable_selected_window_bytes;
    uint64 addressable_query_bytes;
    uint32 addressable_dense_term_count;
    uint32 addressable_sparse_term_count;
    uint64 published_bound_entries;
    uint64 published_bound_bytes;
    uint64 allowed_blocks;
    uint64 allowed_documents;
    uint64 allowed_postings;
    uint64 allowed_row_bytes;
    uint64 competitive_blocks;
    uint64 competitive_documents;
    uint64 competitive_postings;
    uint64 competitive_row_bytes;
    uint8 block_shift;
    bool topk_contained;
} evoke_page_query_forward_bound_level;

typedef struct evoke_page_query_forward_bound_audit
{
    evoke_page_query_forward_bound_level
        levels[EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT];
    uint64 document_count;
    uint64 allowed_document_count;
    uint64 forward_chunk_count;
    uint64 forward_object_bytes;
    uint64 forward_term_work_total;
    uint64 forward_term_work_suffix;
    uint64 forward_term_bytes_total;
    uint64 forward_term_bytes_suffix;
    uint32 query_term_count;
    uint32 maximum_query_term;
    uint32 topk_count;
    uint32 failure_stage;
    float kth_score;
} evoke_page_query_forward_bound_audit;

typedef struct evoke_page_query_term_suffix_ceiling
{
    uint64 document_count;
    uint64 forward_chunk_count;
    uint64 forward_term_work_total;
    uint64 forward_term_work_suffix;
    uint64 forward_term_bytes_total;
    uint64 forward_term_bytes_suffix;
    uint32 vocab_size;
    uint32 query_term_count;
    uint32 maximum_query_term;
} evoke_page_query_term_suffix_ceiling;

#define EVOKE_PAGE_QUERY_L0_VISIBILITY_INCLUDE UINT8_C(0x01)
#define EVOKE_PAGE_QUERY_L0_VISIBILITY_SNAPSHOT_SENSITIVE UINT8_C(0x02)

typedef struct evoke_page_query_l0_input
{
    const uint32_t *query_ids;
    const char *const *query_tokens;
    const bool *query_prefixes;
    size_t query_len;
    bool numeric_source;
} evoke_page_query_l0_input;

typedef struct evoke_page_query_l0_term_stats
{
    uint32 lexical_document_frequency;
    bool lexical_seen;
    bool semantic_seen;
} evoke_page_query_l0_term_stats;

typedef struct evoke_page_query_l0_contribution
{
    uint32 query_index;
    evoke_posting_extent_kind kind;
    evoke_posting_value value;
} evoke_page_query_l0_contribution;

typedef struct evoke_page_query_l0_document
{
    uint32 document_slot;
    uint32 document_length;
    uint64 born_sequence;
    uint32 heap_block;
    uint16 heap_offset;
    bool live;
    bool shadows_immutable;
    size_t first_contribution;
    size_t contribution_count;
} evoke_page_query_l0_document;

/*
 * Query-local linked-L0 state. Memory is proportional to L0 records plus
 * atoms matching this query, never the sealed corpus or complete L0 payload.
 */
typedef struct evoke_page_query_l0_projection
{
    uint64 visible_document_count;
    uint64 total_document_length;
    evoke_page_query_l0_term_stats *terms;
    size_t term_count;
    evoke_page_query_l0_document *documents;
    size_t document_count;
    evoke_page_query_l0_contribution *contributions;
    size_t contribution_count;
    uint32 visited_record_count;
    uint32 visible_record_count;
    bool snapshot_sensitive;
} evoke_page_query_l0_projection;

typedef void (*evoke_page_query_l0_visibility_resolver)(
    void *context,
    const uint32_t *record_xids,
    uint8_t *visibility_out,
    uint32 record_count
);

void evoke_page_query_l0_projection_init(
    evoke_page_query_l0_projection *projection
);

void evoke_page_query_l0_projection_free(
    evoke_page_query_l0_projection *projection
);

void evoke_page_query_build_l0_projection(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_input *input,
    evoke_page_query_l0_visibility_resolver visibility_resolver,
    void *visibility_context,
    evoke_page_query_l0_projection *projection_out
);

const evoke_page_query_l0_document *
evoke_page_query_l0_find_document(
    const evoke_page_query_l0_projection *projection,
    uint32 document_slot
);

/*
 * Optional exact candidate restriction. The bitmap is indexed by the stable
 * document slot in the pinned root. Corpus statistics remain global; only
 * membership in the ranking competition changes.
 */
typedef struct evoke_page_query_filter
{
    const uint8 *allowed_document_bitmap;
    bool (*allows_document)(void *context, uint32 document_slot);
    void *allows_document_context;
    uint64 document_slot_count;
    uint64 allowed_document_count;
    bool live_membership_verified;
    bool ranked_prefix_already_attempted;
} evoke_page_query_filter;

/*
 * Produce exact positive-score top-k candidates from immutable v3 runs.
 * Memory is proportional to query runs plus k, never corpus cardinality.
 * A false positive_topk_complete means zero-score authority is still needed.
 */
evoke_status evoke_page_query_positive_topk(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
);

evoke_status evoke_page_query_positive_topk_projected(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
);

evoke_status evoke_page_query_positive_topk_filtered(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const evoke_page_query_filter *filter,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
);

/* Exact positive ranking plus bounded born-sequence zero-score completion. */
evoke_status evoke_page_query_topk(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
);

evoke_status evoke_page_query_topk_projected(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
);

evoke_status evoke_page_query_topk_filtered(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const evoke_page_query_filter *filter,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
);

/*
 * Read-only research probe for deciding the physical block-max layout. It
 * derives exact, per-term safe bounds from the authoritative posting payload;
 * no result, cache, root, or lifecycle state is published.
 */
evoke_status evoke_page_query_cost_audit_run(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_page_query_cost_audit *audit_out
);

/*
 * Model a same-root per-term bound projection over the current quantized
 * forward score authority. The probe publishes nothing and is intentionally
 * limited to b8 and b16 until the representation gate is decided.
 */
evoke_status evoke_page_query_forward_bound_audit_run(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_filter *filter,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_page_query_forward_bound_audit *audit_out
);

/*
 * Read only the fixed current-root term work table. This establishes the
 * maximum physical saving available to any term-id prefix representation
 * without scanning forward rows or running a scorer.
 */
evoke_status evoke_page_query_term_suffix_ceiling_run(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const uint32_t *query_ids,
    size_t query_len,
    evoke_page_query_term_suffix_ceiling *ceiling_out
);

#endif
