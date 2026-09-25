#ifndef EVOKE_SEGMENT_PAGES_H
#define EVOKE_SEGMENT_PAGES_H

#include "postgres.h"

#include "port/atomics.h"

#include "utils/rel.h"

#include "evoke_block_ranges.h"
#include "evoke_document_cow.h"
#include "evoke_document_tid_lookup.h"
#include "evoke_lexicon_cow.h"
#include "evoke_semantic_accelerator.h"
#include "evoke_semantic_accelerator_directory.h"
#include "evoke_semantic_bmp.h"
#include "evoke_semantic_forward.h"
#include "evoke_scope.h"
#include "evoke_segments.h"
#include "evoke_term_cow.h"

#define EVOKE_INITIAL_FOLD_TARGET_BYTES ((Size) 64 * 1024 * 1024)

typedef struct evoke_l0_chain_write_result
{
    uint32 head_block;
    uint32 tail_block;
    uint32 page_count;
} evoke_l0_chain_write_result;

typedef struct evoke_segment_cow_result
{
    evoke_segment_object_ref manifest;
    uint32 published_block_high_watermark;
    uint32 reused_block_count;
    evoke_block_range *fsm_handoff_ranges;
    uint32 fsm_handoff_range_count;
    uint32 fsm_handoff_block_count;
    evoke_block_range *staged_write_ranges;
    uint32 staged_write_range_count;
    uint32 staged_write_block_count;
} evoke_segment_cow_result;

typedef struct evoke_semantic_accelerator_term_artifact
{
    uint32 term_id;
    const uint8 *bytes;
    Size size;
} evoke_semantic_accelerator_term_artifact;

/*
 * The producer owns artifact bytes until the next call. This keeps publication
 * memory bounded by one immutable term object instead of the complete corpus.
 */
typedef evoke_status (*evoke_semantic_accelerator_artifact_producer)(
    void *context,
    uint32 artifact_index,
    evoke_semantic_accelerator_term_artifact *artifact_out
);

typedef struct evoke_semantic_forward_artifact
{
    uint32 first_document;
    uint32 document_count;
    const uint8 *bytes;
    Size size;
} evoke_semantic_forward_artifact;

typedef evoke_status (*evoke_semantic_forward_artifact_producer)(
    void *context,
    uint32 artifact_index,
    evoke_semantic_forward_artifact *artifact_out
);

typedef struct evoke_semantic_forward_bound_artifact
{
    uint32 first_term;
    uint32 term_count;
    const uint8 *bytes;
    Size size;
} evoke_semantic_forward_bound_artifact;

typedef evoke_status (*evoke_semantic_forward_bound_artifact_producer)(
    void *context,
    uint32 artifact_index,
    evoke_semantic_forward_bound_artifact *artifact_out
);

/* The producer transfers one initialized bundle to the caller per call. */
typedef evoke_status (*evoke_initial_fold_bundle_producer)(
    void *context,
    evoke_term_fold_bundle *bundle_out,
    bool *done_out
);

typedef enum evoke_segment_cow_write_outcome
{
    EVOKE_SEGMENT_COW_WRITE_WRITTEN = 0,
    EVOKE_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED
} evoke_segment_cow_write_outcome;

typedef struct evoke_segment_page_reuse_arena
{
    evoke_block_range_allocator allocator;
    evoke_block_range_inventory staged_writes;
    uint64 fsm_reused_block_count;
    uint32 source_high_watermark;
    bool reader_fenced;
} evoke_segment_page_reuse_arena;

typedef struct evoke_segment_reachability_inventory
{
    evoke_block_range_inventory blocks;
    uint32 physical_block_count;
    uint32 trailing_unpublished_block_count;
} evoke_segment_reachability_inventory;

typedef struct evoke_l0_stored_record
{
    uint8 *bytes;
    Size size;
    evoke_l0_record_view view;
} evoke_l0_stored_record;

typedef struct evoke_l0_storage_snapshot
{
    evoke_l0_stored_record *records;
    uint32 *page_blocks;
    uint32 record_count;
    uint32 page_count;
    uint64 payload_bytes;
} evoke_l0_storage_snapshot;

typedef struct evoke_l0_visit_stats
{
    uint32 record_count;
    uint32 page_count;
    uint64 payload_bytes;
} evoke_l0_visit_stats;

/* The record view and its payload remain valid only during the callback. */
typedef void (*evoke_l0_record_visitor)(
    void *context,
    const evoke_l0_record_view *record
);

typedef struct evoke_segment_query_extent
{
    evoke_posting_extent view;
    uint32 *document_slots;
    evoke_posting_value *values;
    evoke_posting_block_record *blocks;
} evoke_segment_query_extent;

/*
 * Query-local ownership for one term. Its storage is proportional only to
 * postings opened for that term and is released with the query context.
 */
typedef struct evoke_segment_query_term
{
    uint32 term_id;
    uint32 raw_document_frequency;
    evoke_segment_query_extent *extents;
    uint32 extent_count;
    uint32 extent_capacity;
} evoke_segment_query_term;

#define EVOKE_SEGMENT_QUERY_PAGE_VALIDATION_CACHE_SIZE UINT32_C(16384)

/*
 * A published segment page is immutable while the query root is pinned.
 * Remember pages whose application checksum has already been verified so
 * repeated small range reads do not checksum the same 8 KiB payload again.
 */
typedef struct evoke_segment_query_page_validation_cache
{
    uint32 block_keys[EVOKE_SEGMENT_QUERY_PAGE_VALIDATION_CACHE_SIZE];
    pg_atomic_uint64 *shared_block_words;
    uint32 shared_block_word_count;
} evoke_segment_query_page_validation_cache;

/*
 * Bounded ownership for one foreground v3 query. Immutable corpus data and
 * linked L0 stay in relation pages and PostgreSQL shared buffers. Query-time
 * L0 consumers must stream or project records instead of retaining a complete
 * frontier snapshot in one backend.
 */
typedef struct evoke_segment_query_context
{
    evoke_segment_read_root root;
    evoke_segment_manifest manifest;
    evoke_segment_query_contract query_contract;
    evoke_segment_query_page_validation_cache *page_validation_cache;
    const uint32 *resident_document_lengths;
    uint64 resident_document_length_count;
    bool semantic_accelerator_compatible;
} evoke_segment_query_context;

#define EVOKE_SEGMENT_QUERY_DOCUMENT_NODE_CACHE_SIZE UINT32_C(8)

/*
 * Query-local COW reader. Only immutable radix nodes are retained; leaves and
 * document records continue to stream one fixed-size score block at a time.
 */
typedef struct evoke_segment_query_document_reader
{
    Relation index_relation;
    evoke_document_cow_object root_object;
    evoke_document_cow_object node_cache[
        EVOKE_SEGMENT_QUERY_DOCUMENT_NODE_CACHE_SIZE
    ];
    uint64 node_cache_age[
        EVOKE_SEGMENT_QUERY_DOCUMENT_NODE_CACHE_SIZE
    ];
    uint64 access_clock;
    uint32 published_block_high_watermark;
    uint64 max_owner_manifest_id;
    evoke_segment_query_page_validation_cache *page_validation_cache;
    bool node_cache_valid[
        EVOKE_SEGMENT_QUERY_DOCUMENT_NODE_CACHE_SIZE
    ];
    bool initialized;
} evoke_segment_query_document_reader;

typedef struct evoke_segment_lexicon_prefix_match
{
    uint32 term_id;
    uint32 token_len;
    char *token;
} evoke_segment_lexicon_prefix_match;

#define EVOKE_SEGMENT_QUERY_TERM_FOLD_MAX_RUNS UINT32_C(4)
#define EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS \
    (EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM + \
     EVOKE_SEGMENT_QUERY_TERM_FOLD_MAX_RUNS)
#define EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS \
    (UINT32_C(1) << EVOKE_DEFAULT_POSTING_BLOCK_SHIFT)
#define EVOKE_SEGMENT_QUERY_BLOCK_RECORD_WINDOW UINT32_C(32)
#define EVOKE_SEGMENT_QUERY_POSTING_WINDOW UINT32_C(16384)

typedef enum evoke_segment_query_run_source
{
    EVOKE_SEGMENT_QUERY_RUN_SOURCE_INVALID = 0,
    EVOKE_SEGMENT_QUERY_RUN_SOURCE_PAYLOAD = 1,
    EVOKE_SEGMENT_QUERY_RUN_SOURCE_FOLD = 2
} evoke_segment_query_run_source;

/* One query term's compact block-forward directory inside an object. */
typedef struct evoke_segment_query_bmp_term
{
    uint64 section_offset;
    uint64 section_size;
    uint64 super_refs_offset;
    uint64 refs_offset;
    uint64 block_membership_offset;
    uint64 doc_deltas_offset;
    uint64 impacts_offset;
    uint32 document_count;
    uint32 block_count;
    uint32 superblock_count;
    uint32 record_count;
    uint64 posting_count;
    uint32 first_ref;
    uint32 ref_count;
    uint32 first_super_ref;
    uint32 super_ref_count;
    uint32 first_block_membership_byte;
    uint32 block_membership_bytes;
    uint32 first_doc_byte;
    uint32 first_impact;
    uint32 first_document;
    uint32 doc_delta_width;
    float min_impact;
    float max_impact;
    evoke_semantic_impact_precision impact_precision;
    bool available;
} evoke_segment_query_bmp_term;

/* One fixed-size descriptor for a term-local immutable posting run. */
typedef struct evoke_segment_query_run
{
    evoke_segment_object_ref ref;
    /* Logical offset used by the stable COW term-directory identity. */
    uint64 stable_posting_offset;
    /* Physical offset retained when packed semantic access is zero-based. */
    uint64 source_posting_offset;
    uint64 posting_offset;
    uint64 posting_count;
    uint64 block_offset;
    uint64 indices_offset;
    uint64 values_offset;
    uint64 blocks_offset;
    uint64 document_map_offset;
    uint32 block_count;
    uint32 block_shift;
    uint32 payload_flags;
    uint32 document_id_base;
    uint32 local_document_count;
    evoke_posting_extent_kind kind;
    evoke_segment_query_run_source source;
    evoke_segment_query_bmp_term semantic_bmp;
} evoke_segment_query_run;

/* A term plan is bounded by the format's fixed fold and raw-extent limits. */
typedef struct evoke_segment_query_term_plan
{
    uint32 term_id;
    uint32 raw_document_frequency;
    uint32 run_count;
    evoke_segment_query_run runs[EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS];
} evoke_segment_query_term_plan;

/* One decoded posting block; no member scales beyond the fixed block size. */
typedef struct evoke_segment_query_block
{
    uint64 source_posting_offset;
    evoke_posting_block_record record;
    uint32 document_slots[EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS];
    evoke_posting_value values[EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS];
    evoke_posting_extent view;
} evoke_segment_query_block;

/* Sequential state for streaming one immutable term run exactly once. */
typedef struct evoke_segment_query_posting_cursor
{
    uint64 next_posting_index;
    uint32 previous_document_slot;
    bool have_previous_document;
} evoke_segment_query_posting_cursor;

/* Query-local decoded view over one packed semantic super reference. */
typedef struct evoke_segment_query_bmp_cached_super_ref
{
    uint32 superblock_id;
    uint32 first_ref;
    uint32 first_impact;
    uint16 ref_count;
    uint16 reserved;
    float min_impact;
    float max_impact;
} evoke_segment_query_bmp_cached_super_ref;

/* One global document block aligned to the posting block contract. */
typedef struct evoke_segment_query_document_block
{
    uint32 block_id;
    uint32 first_document_slot;
    uint32 record_count;
    evoke_document_cow_record records[
        EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS
    ];
} evoke_segment_query_document_block;

typedef void (*evoke_segment_document_record_visitor)(
    void *context,
    const evoke_document_cow_record *record
);

void evoke_segment_page_reuse_arena_init(
    evoke_segment_page_reuse_arena *arena
);

void evoke_segment_page_reuse_arena_free(
    evoke_segment_page_reuse_arena *arena
);

/*
 * Build an in-memory allocator from exact unreachable interior ranges. The
 * caller must first fence every reader that could still hold an older root.
 */
evoke_status evoke_segment_page_reuse_arena_build(
    const evoke_block_range_inventory *inventory,
    evoke_segment_page_reuse_arena *arena
);

evoke_status evoke_segment_page_reuse_arena_build_retired(
    const evoke_block_range *ranges,
    size_t range_count,
    uint32_t published_block_high_watermark,
    evoke_segment_page_reuse_arena *arena
);

evoke_status evoke_segment_page_reuse_arena_build_empty_fenced(
    uint32_t published_block_high_watermark,
    evoke_segment_page_reuse_arena *arena
);

void evoke_segment_cow_result_init(
    evoke_segment_cow_result *result
);

void evoke_segment_cow_result_free(
    evoke_segment_cow_result *result
);

/*
 * Decoded immutable portion and checked L0 records of one read root. L0
 * records are retained but are not attached to read_view yet.
 */
typedef struct evoke_segment_storage_snapshot
{
    evoke_segment_read_root root;
    evoke_segment_manifest manifest;
    evoke_segment_query_contract query_contract;
    evoke_document_cow_length_extrema *document_block_extrema;
    size_t document_block_count;
    evoke_term_directory term_directory;
    uint32 *materialized_doc_frequencies;
    evoke_segment_object_ref *lexical_catalog_refs;
    evoke_term_cow_fold_state *term_fold_states;
    evoke_term_fold_bundle *term_fold_bundles;
    uint32 term_fold_bundle_count;
    evoke_term_fold_read_plan *term_fold_plans;
    evoke_segment_payload *payloads;
    evoke_segment_payload_view *payload_views;
    evoke_index index_metadata;
    ItemPointerData *doc_tids;
    uint64 *document_tie_break_keys;
    uint32 *document_tie_break_order;
    uint32 document_tie_break_order_count;
    bool document_tie_break_order_is_identity;
    evoke_segment_read_view read_view;
    evoke_corpus_stats corpus_stats;
    evoke_l0_storage_snapshot l0_snapshot;
    uint32 *l0_indices;
    evoke_posting_value *l0_values;
    uint64 l0_posting_count;
    evoke_posting_block_record **l0_block_allocations;
    size_t l0_block_allocation_count;
    uint32 *retired_document_ids;
    uint32 retired_document_count;
} evoke_segment_storage_snapshot;

/*
 * Append one immutable object as a contiguous page chain. The object is not
 * reachable until its returned reference is published by the metapage.
 */
void evoke_segment_pages_write(
    Relation index_relation,
    evoke_segment_object_kind object_kind,
    uint64 object_id,
    uint64 owner_manifest_id,
    const uint8 *object_bytes,
    Size object_size,
    evoke_segment_object_ref *ref_out
);

/*
 * Read and validate an immutable page chain. The caller owns the returned
 * palloc allocation.
 */
uint8 *evoke_segment_pages_read(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    Size *object_size_out
);

/*
 * Read one checked byte range without materializing the complete immutable
 * object. Every touched page validates its envelope and payload checksum.
 */
void evoke_segment_pages_read_range(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    Size offset,
    Size length,
    uint8 *bytes_out
);

/*
 * Warm only the relation pages covering one immutable object byte range.
 * The caller supplies the published high-water mark and a shared page budget;
 * this function never materializes the object in backend memory.
 */
bool evoke_segment_pages_prewarm_range(
    Relation index_relation,
    const evoke_segment_object_ref *ref,
    Size offset,
    Size length,
    BlockNumber published_block_high_watermark,
    uint64 page_budget,
    uint64 *pages_warmed
);

/*
 * Append the not-yet-bound suffix of one COW term tree in object-id order.
 * Leaves precede parents, so each parent records the actual relation page
 * chains returned for its children. No object is reachable until the caller
 * publishes root_out through a descendant manifest.
 */
void evoke_segment_pages_write_term_cow_objects(
    Relation index_relation,
    evoke_term_cow_tree *tree,
    uint64 first_object_id,
    uint64 owner_manifest_id,
    evoke_segment_object_ref *root_out
);

/*
 * Append the path-copied suffix of one document/version COW tree. Child
 * objects are bound before parents, and the root remains unreachable until
 * its physical reference is published by a manifest.
 */
void evoke_segment_pages_write_document_cow_objects(
    Relation index_relation,
    evoke_document_cow_tree *tree,
    uint64 owner_manifest_id,
    evoke_segment_object_ref *root_out
);

/*
 * Append one complete or path-copied lexical lookup tree. Object ids are
 * owner-local, so inherited children remain identified by owner plus id.
 */
void evoke_segment_pages_write_lexicon_cow_objects(
    Relation index_relation,
    evoke_lexicon_cow_tree *tree,
    uint64 owner_manifest_id,
    evoke_segment_object_ref *root_out
);

/*
 * Resolve normalized lexical bytes through only one checked COW radix path.
 */
bool evoke_segment_pages_lookup_lexicon(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const uint8 *bytes,
    Size bytes_len,
    uint32 *term_id_out
);

/*
 * Enumerate one bounded lexical prefix without retaining a decoded
 * vocabulary. The caller owns the palloc-backed result and token strings.
 */
void evoke_segment_pages_collect_lexicon_prefix(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const uint8 *prefix,
    Size prefix_len,
    Size max_matches,
    Size max_token_bytes,
    evoke_segment_lexicon_prefix_match **matches_out,
    Size *match_count_out
);

void evoke_segment_pages_free_lexicon_prefix_matches(
    evoke_segment_lexicon_prefix_match *matches,
    Size match_count
);

/*
 * Resolve one exact document/version record through a bounded radix path.
 */
void evoke_segment_pages_load_document_record(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint64 document_slot,
    evoke_document_cow_record *record_out
);

/*
 * Read one fixed-size global document block through one bounded COW walk.
 */
void evoke_segment_pages_load_query_document_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_segment_query_document_reader *reader,
    uint32 block_id,
    evoke_segment_query_document_block *block_out
);

/*
 * Visit every immutable document/version record by walking only the
 * authoritative COW document tree. Posting payload objects are never opened.
 */
void evoke_segment_pages_visit_document_records(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_segment_document_record_visitor visitor,
    void *visitor_context
);

/* Query-pinned variant that reuses page and radix validation state. */
void evoke_segment_pages_visit_query_document_records(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_segment_query_document_reader *reader,
    evoke_segment_document_record_visitor visitor,
    void *visitor_context
);

/*
 * Load a complete immutable document-length vector from sealed payload
 * metadata when the manifest proves that every live slot has one version and
 * no retirement history. Mutable or ambiguous generations return false so the
 * caller can use the authoritative document COW tree instead.
 */
bool evoke_segment_pages_load_query_document_lengths(
    Relation index_relation,
    const evoke_segment_query_context *context,
    uint32 *document_lengths
);

/*
 * Read only the oldest root-relative live incarnations needed for exact
 * zero-score tie completion. The COW best-first cursor is bounded by limit and
 * the fixed radix shape; it never materializes the complete document order.
 */
void evoke_segment_pages_load_live_born_prefix(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    size_t limit,
    evoke_document_cow_record *records_out,
    size_t record_capacity,
    size_t *record_count_out,
    evoke_document_cow_born_prefix_stats *stats_out
);

void evoke_segment_pages_load_matching_live_born_prefix(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    size_t limit,
    evoke_document_cow_record_predicate predicate,
    void *predicate_context,
    evoke_document_cow_record *records_out,
    size_t record_capacity,
    size_t *record_count_out,
    evoke_document_cow_born_prefix_stats *stats_out
);

/*
 * Read the exact root summaries without traversing document leaves. These
 * counters are part of the checksummed COW authority and therefore describe
 * the same manifest snapshot used by queries and maintenance.
 */
void evoke_segment_pages_load_document_summary(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint64 *live_document_count_out,
    uint64 *semantic_pending_count_out,
    int64 *earliest_retry_after_out
);

/* Read corpus-wide document length bounds from the COW root summary. */
void evoke_segment_pages_load_document_length_extrema(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_document_cow_length_extrema *extrema_out
);

/*
 * Resolve one contract-aligned global document block from persisted COW
 * subtree summaries. Retired versions remain conservative contributors.
 */
void evoke_segment_pages_load_document_block_extrema(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    uint32 block_id,
    evoke_document_cow_length_extrema *extrema_out
);

/*
 * Find one due semantic item from subtree summaries without scanning slots.
 */
bool evoke_segment_pages_find_actionable_document(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    int64 now,
    evoke_document_cow_record *record_out
);

bool evoke_segment_pages_find_actionable_document_from(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint64 first_document_slot,
    int64 now,
    evoke_document_cow_record *record_out
);

/* Find one root-relative score slot whose prior incarnation is fully drained. */
bool evoke_segment_pages_find_reusable_document_from(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint64 first_document_slot,
    evoke_document_cow_record *record_out
);

/*
 * Resolve one term directly from a physical COW root. The cold path reads only
 * one bounded radix path and validates both page envelopes and object codecs.
 */
void evoke_segment_pages_load_term_cow_record(
    Relation index_relation,
    const evoke_segment_object_ref *root_ref,
    uint32 vocab_size,
    uint32 term_id,
    evoke_term_cow_record *record_out
);

/*
 * Validate current query-authority object headers without materializing
 * postings. Optional derived maintenance must not retry an older physical
 * codec forever; foreground readers still report the complete format error.
 */
evoke_status evoke_segment_pages_query_authority_format_status(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest
);

void evoke_segment_query_term_init(evoke_segment_query_term *term);

void evoke_segment_query_term_free(evoke_segment_query_term *term);

void evoke_segment_query_context_init(
    evoke_segment_query_context *context
);

void evoke_segment_query_context_free(
    evoke_segment_query_context *context
);

/*
 * Open one checked COW-only foreground context. This deliberately does not
 * materialize vocabulary, term/DF, payload, document, tie-order, or score
 * arrays. Full snapshots are reserved for maintenance and test oracles.
 */
void evoke_segment_pages_load_query_context(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_query_context *context_out
);

/* Load and authenticate the derived accelerator bound to this exact root. */
void evoke_segment_pages_load_semantic_accelerator_directory(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory *directory_out
);

/* Load accelerator children only to retire a superseded derived format. */
void evoke_segment_pages_load_retired_semantic_accelerator_directory(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory *directory_out
);

/* Load only the forward-chunk slice needed by exact filtered scoring. */
void evoke_segment_pages_load_semantic_accelerator_forward_directory(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_semantic_accelerator_directory *directory_out
);

/* Read only the authenticated first-page accelerator metadata. */
void evoke_segment_pages_load_semantic_accelerator_summary(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory_summary *summary_out
);

/* Return stale derived metadata as a status so maintenance can replace it. */
evoke_status evoke_segment_pages_try_load_semantic_accelerator_summary(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory_summary *summary_out
);

/* Admit only an accelerator whose query contract matches this index. */
bool evoke_segment_pages_semantic_accelerator_compatible(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_semantic_accelerator_directory_summary *summary_out
);

/* Open the optional same-root scope-posting child without loading its body. */
bool evoke_segment_pages_open_semantic_accelerator_scope(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_segment_object_ref *scope_ref_out,
    evoke_scope_header *scope_header_out
);

/* Load the same-root compact reverse TID directory without scanning COW. */
bool evoke_segment_pages_load_semantic_accelerator_tid_lookup(
    Relation index_relation,
    const evoke_segment_query_context *context,
    uint8 **bytes_out,
    Size *size_out,
    evoke_document_tid_lookup_view *view_out
);

/* Read one authenticated scope-object range under the pinned query root. */
void evoke_segment_pages_read_semantic_accelerator_scope_range(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_object_ref *scope_ref,
    Size offset,
    Size length,
    uint8 *bytes_out
);

/* Load one checked per-term accelerator object from a published directory. */
void evoke_segment_pages_load_semantic_accelerator_term(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_semantic_accelerator_directory *directory,
    uint32 term_id,
    evoke_semantic_accelerator_index *index_out
);

/* Load one exact document-major completion chunk. */
void evoke_segment_pages_load_semantic_accelerator_forward(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_semantic_accelerator_directory *directory,
    uint32 document_id,
    evoke_semantic_forward_chunk *chunk_out
);

typedef struct evoke_segment_forward_bound_reader
{
    evoke_segment_object_ref shard_ref;
    evoke_semantic_forward_bound_summary summary;
    uint8 offsets[
        (EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD + 1U) *
            sizeof(uint64)
    ];
    uint64 authority_checksum;
    bool authority_ready;
    bool shard_ready;
} evoke_segment_forward_bound_reader;

void evoke_segment_forward_bound_reader_init(
    evoke_segment_forward_bound_reader *reader
);

void evoke_segment_pages_visit_semantic_accelerator_forward_bound(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    evoke_segment_forward_bound_reader *reader,
    uint32 term_id,
    evoke_semantic_forward_bound_visitor visitor,
    void *visitor_context,
    uint32 *entry_count_out,
    uint64 *bytes_read_out
);

typedef struct evoke_segment_forward_row_reader
{
    const evoke_semantic_accelerator_forward_entry *entry;
    evoke_semantic_forward_header header;
    uint32 *row_offsets;
    Size row_offset_capacity;
    uint32 row_offset_count;
    uint8 *row_bytes;
    Size row_capacity;
    Size row_window_offset;
    Size row_window_size;
    bool exact_row_reads;
    bool exact_row_data_reads;
    uint64 authority_checksum;
    bool authority_ready;
} evoke_segment_forward_row_reader;

void evoke_segment_forward_row_reader_init(
    evoke_segment_forward_row_reader *reader
);

void evoke_segment_forward_row_reader_reset(
    evoke_segment_forward_row_reader *reader
);

void evoke_segment_forward_row_reader_set_exact_reads(
    evoke_segment_forward_row_reader *reader,
    bool exact_row_reads
);

void evoke_segment_forward_row_reader_set_exact_data_reads(
    evoke_segment_forward_row_reader *reader,
    bool exact_row_data_reads
);

/*
 * Prefetch only the forward-row pages needed by one sorted document batch.
 * The eventual reads still validate every page and row before scoring.
 */
void evoke_segment_pages_prefetch_semantic_accelerator_forward_rows(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    const uint32 *document_ids,
    size_t document_count
);

/* Score one forward row through checked object-range reads. */
void evoke_segment_pages_score_semantic_accelerator_forward_row(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    evoke_segment_forward_row_reader *reader,
    uint32 document_id,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_count,
    bool query_is_sorted,
    float *score_out,
    uint64 *postings_examined_out,
    uint64 *bytes_read_out
);

/*
 * Score one aligned eight-document block from its published row offsets. The
 * payload is fetched with one contiguous object-range read; no forward header
 * or per-chunk offset table is revisited at query time.
 */
void evoke_segment_pages_score_semantic_accelerator_forward_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    evoke_segment_forward_row_reader *reader,
    uint32 block_id,
    uint8 allowed_mask,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_count,
    float scores_out[8],
    uint32 *documents_scored_out,
    uint64 *postings_examined_out,
    uint64 *bytes_read_out
);

typedef struct evoke_segment_forward_transpose_reader
{
    uint8 *scales;
    Size scales_capacity;
    uint8 *dense_term_ids;
    Size dense_term_ids_capacity;
    uint8 *dense_codes;
    Size dense_codes_capacity;
    uint8 *term_offsets;
    Size term_offsets_capacity;
    uint32 *query_ranges;
    Size query_ranges_capacity;
    uint32 *query_dense_indexes;
    Size query_dense_indexes_capacity;
    uint8 *postings;
    Size postings_capacity;
    uint8 *sparse_oracle_pages;
    Size sparse_oracle_pages_capacity;
    uint32 *document_runs;
    Size document_runs_capacity;
    double *scores;
    Size score_capacity;
    uint64 authority_checksum;
    uint64 sparse_oracle_full_bytes;
    uint64 sparse_oracle_selected_bytes;
    uint64 sparse_oracle_ranges;
    uint64 sparse_oracle_full_pages;
    uint64 sparse_oracle_selected_pages;
    bool authority_ready;
} evoke_segment_forward_transpose_reader;

extern bool evoke_test_semantic_accelerator_dense_subrange_reads;
extern bool evoke_test_semantic_accelerator_sparse_block_oracle;

void evoke_segment_forward_transpose_reader_init(
    evoke_segment_forward_transpose_reader *reader
);

void evoke_segment_forward_transpose_reader_reset(
    evoke_segment_forward_transpose_reader *reader
);

/*
 * Score one current v6 forward chunk from its block-local term directory.
 */
void evoke_segment_pages_score_semantic_accelerator_forward_transposed(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    evoke_segment_forward_transpose_reader *reader,
    uint32 forward_chunk_index,
    const uint8 *allowed_document_bitmap,
    Size allowed_document_bitmap_size,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_count,
    float *scores_out,
    Size score_count,
    uint64 *postings_examined_out,
    uint64 *bytes_read_out
);

/* Product form: geometric term objects plus complete exact forward chunks. */
evoke_segment_cow_write_outcome
evoke_segment_pages_write_semantic_accelerator_complete_stream_fork(
    Relation index_relation,
    ForkNumber fork_number,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    uint32 artifact_count,
    evoke_semantic_accelerator_artifact_producer produce_artifact,
    void *producer_context,
    uint32 forward_chunk_count,
    uint32 forward_document_shift,
    evoke_semantic_forward_artifact_producer produce_forward,
    void *forward_context,
    uint32 forward_bound_shard_count,
    evoke_semantic_forward_bound_artifact_producer produce_forward_bound,
    void *forward_bound_context,
    const uint8 *scope_bytes,
    Size scope_size,
    const uint8 *tid_lookup_bytes,
    Size tid_lookup_size,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_manifest *next_manifest,
    evoke_segment_cow_result *result_out
);

/* Resolve only fixed-size run metadata for one immutable query term. */
void evoke_segment_pages_load_query_term_plan(
    Relation index_relation,
    const evoke_segment_query_context *context,
    uint32 term_id,
    evoke_segment_query_term_plan *plan_out
);

void evoke_segment_pages_load_query_term_plan_after_sequence(
    Relation index_relation,
    const evoke_segment_query_context *context,
    uint32 term_id,
    uint64 minimum_sequence,
    evoke_segment_query_term_plan *plan_out
);

/* Materialize one immutable term directly from its checked query plan. */
void evoke_segment_pages_load_query_term_from_plan(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    evoke_segment_query_term *term_out
);

/* Read and validate one posting-block descriptor without its postings. */
void evoke_segment_pages_load_query_term_block_record(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 block_index,
    evoke_posting_block_record *record_out
);

uint32 evoke_segment_pages_load_query_term_block_records(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 first_block_index,
    evoke_posting_block_record *records_out,
    uint32 record_capacity
);

/* Read and validate one fixed-size posting block from a term plan. */
void evoke_segment_pages_load_query_term_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 block_index,
    evoke_segment_query_block *block_out
);

/* Load postings for a descriptor already checked by the block cursor. */
void evoke_segment_pages_load_query_term_block_from_record(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 block_index,
    const evoke_posting_block_record *source_record,
    evoke_segment_query_block *block_out
);

/*
 * Read one bounded, contiguous posting window without materializing a run.
 * Payload-local identifiers are globalized before return.
 */
uint32 evoke_segment_pages_load_query_term_posting_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint64 first_posting_index,
    uint32 *document_slots_out,
    evoke_posting_value *values_out,
    uint32 posting_capacity
);

void evoke_segment_query_posting_cursor_init(
    evoke_segment_query_posting_cursor *cursor
);

/*
 * Stream the next bounded posting window. Packed semantic runs use their
 * canonical term-major document deltas and shared impact array directly.
 */
uint32 evoke_segment_pages_load_query_term_posting_stream_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    evoke_segment_query_posting_cursor *cursor,
    uint32 *document_slots_out,
    evoke_posting_value *values_out,
    uint32 posting_capacity
);

/*
 * Decode one packed semantic window directly into the query score array.
 * The caller supplies bounded scratch storage for document deltas and impacts;
 * no decoded posting window is materialized.
 */
uint32 evoke_segment_pages_accumulate_query_semantic_stream_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    evoke_segment_query_posting_cursor *cursor,
    uint8 *delta_scratch,
    uint8 *impact_scratch,
    uint32 posting_capacity,
    float query_weight,
    float *scores,
    uint64 score_count,
    double absolute_impact_floor,
    uint64 *omitted_postings
);

/* Read compact semantic block references without touching exact impacts. */
uint32 evoke_segment_pages_load_query_bmp_ref_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 first_ref_index,
    evoke_semantic_bmp_ref *refs_out,
    uint32 ref_capacity
);

/* Read one term's optional dense fine-block membership directory. */
uint32 evoke_segment_pages_load_query_bmp_block_membership(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint8 *membership_out,
    uint32 membership_capacity
);

/* Read one exact fine-block bound by its rank in the term-local directory. */
uint32 evoke_segment_pages_load_query_bmp_packed_ref_page(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 relative_ref_index,
    uint8 *refs_out,
    uint32 ref_capacity
);

/* Read coarse semantic superblock references without child block bounds. */
uint32 evoke_segment_pages_load_query_bmp_super_ref_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 first_ref_index,
    evoke_semantic_bmp_super_ref *refs_out,
    uint32 ref_capacity
);

/* Decode coarse bounds and exact-impact offsets for query-local reuse. */
uint32 evoke_segment_pages_load_query_bmp_cached_super_ref_window(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 first_ref_index,
    evoke_segment_query_bmp_cached_super_ref *refs_out,
    uint32 ref_capacity
);

/* Resolve one exact term record from the sole packed semantic section. */
bool evoke_segment_pages_load_query_bmp_record(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 block_id,
    evoke_semantic_bmp_record *record_out,
    float impacts_out[EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS]
);

/* Resolve an exact record through an already decoded super reference. */
bool evoke_segment_pages_load_query_bmp_cached_record(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    const evoke_segment_query_bmp_cached_super_ref *super_ref,
    uint32 block_id,
    evoke_semantic_bmp_record *record_out,
    float impacts_out[EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS]
);

/*
 * Resolve and materialize only one immutable term through checked page-range
 * reads. Linked L0 frontiers are deliberately separate query sources and are
 * not included here. term_out must first be initialized with
 * evoke_segment_query_term_init().
 */
void evoke_segment_pages_load_query_term(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint32 term_id,
    evoke_segment_query_term *term_out
);

/*
 * Check append capacity by resolving only terms present in new_payload.
 */
bool evoke_segment_pages_term_cow_append_has_capacity(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_payload *new_payload
);

/*
 * Resolve one extent-pressure term through persistent subtree maxima.
 */
bool evoke_segment_pages_find_term_extent_pressure(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint32 threshold,
    evoke_term_cow_record *record_out
);

/*
 * Check one already loaded replacement range against term-local fold
 * watermarks. A false result is a valid but currently unsafe structural
 * choice, not payload corruption.
 */
bool evoke_segment_pages_cow_replace_fold_safe(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    uint32 first_segment_index,
    const evoke_segment_payload *payloads,
    uint32 payload_count,
    uint32 *conflict_term_id_out
);

void evoke_segment_storage_snapshot_init(
    evoke_segment_storage_snapshot *snapshot
);

void evoke_segment_storage_snapshot_free(
    evoke_segment_storage_snapshot *snapshot
);

/*
 * Load the immutable manifest and validate its complete COW closure. Use this
 * for diagnostics, scrub, and externally visible integrity checks.
 */
void evoke_segment_pages_load_sealed_manifest(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_manifest *manifest_out
);

/*
 * Load only bounded manifest metadata for COW maintenance. Referenced objects
 * are checked lazily as bounded paths are opened; no vocabulary walk occurs.
 */
void evoke_segment_pages_load_maintenance_manifest(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_manifest *manifest_out
);

/*
 * Load bounded query-contract metadata needed by copy-on-write maintenance.
 */
void evoke_segment_pages_load_query_contract(
    Relation index_relation,
    const evoke_segment_manifest *manifest,
    evoke_segment_query_contract *contract_out
);

/*
 * Attach the complete immutable lexical vocabulary needed by a text query
 * snapshot. Bounded maintenance resolves changed terms through the manifest
 * lexical lookup instead of calling this vocabulary-sized read path.
 */
void evoke_segment_pages_load_query_vocabulary(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_segment_query_contract *contract
);

void evoke_segment_pages_load_term_directory(
    Relation index_relation,
    const evoke_segment_manifest *manifest,
    evoke_term_directory *directory_out
);

/*
 * Reconstruct a disposable flat accelerator from the authoritative
 * root-aware term directory. It is never a publication authority.
 */
void evoke_segment_pages_load_term_accelerator(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_term_directory *directory_out,
    uint32 **doc_frequencies_out,
    evoke_segment_object_ref **lexical_catalog_refs_out,
    evoke_term_cow_fold_state **fold_states_out
);

/*
 * Load only one contiguous immutable payload range for bounded maintenance.
 * The caller releases the returned array with
 * evoke_segment_pages_free_payloads().
 */
void evoke_segment_pages_load_payload_range(
    Relation index_relation,
    const evoke_segment_manifest *manifest,
    uint32 first_segment_index,
    uint32 segment_count,
    evoke_segment_payload **payloads_out
);

/*
 * Load one persistent neutral fold for bounded term-local maintenance. This
 * opens only the referenced immutable object and validates it against the
 * current published root and manifest.
 */
void evoke_segment_pages_load_neutral_fold_bundle(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *ref,
    evoke_term_fold_bundle *bundle_out
);

/* Load one checked, epoch-bound impact fold for shared hot-read publication. */
void evoke_segment_pages_load_impact_fold_bundle(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *ref,
    evoke_term_fold_bundle *bundle_out
);

void evoke_segment_pages_free_payloads(
    evoke_segment_payload *payloads,
    uint32 payload_count
);

/*
 * Read and attach every manifest-owned immutable object. Active and pending
 * L0 frontiers remain described by snapshot->root and must be attached before
 * this snapshot can serve as a complete mutable-index query view.
 */
void evoke_segment_pages_load_sealed_snapshot(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_segment_storage_snapshot *snapshot_out
);

/*
 * Append a complete immutable bundle and return a validated read root ready
 * for one later metapage switch. This function does not publish that root.
 * manifest physical refs and descriptor page fields must initially be zero.
 * The caller must serialize bundle writes with the relation publication lock
 * so the returned high-water mark cannot include another unpublished bundle.
 */
void evoke_segment_pages_write_sealed_bundle(
    Relation index_relation,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    const evoke_segment_payload *payloads,
    size_t payload_count,
    uint64 active_l0_segment_id,
    uint64 next_sequence,
    evoke_segment_read_root *root_out
);

void evoke_segment_pages_write_sealed_bundle_fork(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    const evoke_segment_payload *payloads,
    size_t payload_count,
    uint64 active_l0_segment_id,
    uint64 next_sequence,
    evoke_segment_read_root *root_out
);

/*
 * Publish a rebuild directly as complete, size-bounded term folds. The
 * document COW directory remains the version authority; no initial immutable
 * payload segment is retained. Later mutations use the normal L0/tail path.
 */
void evoke_segment_pages_write_initial_folded_bundle_fork(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    const evoke_segment_payload *payload,
    uint64 active_l0_segment_id,
    uint64 next_sequence,
    evoke_segment_read_root *root_out
);

void evoke_segment_pages_write_streamed_initial_folded_bundle_fork(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_segment_manifest *manifest,
    const evoke_segment_query_contract *query_contract,
    evoke_initial_fold_bundle_producer producer,
    void *producer_context,
    const evoke_document_cow_record *document_records,
    size_t document_record_count,
    uint64 active_l0_segment_id,
    uint64 next_sequence,
    evoke_segment_read_root *root_out
);

/*
 * Append one pending L0 frontier as a descendant immutable segment without
 * reading or rewriting ancestor payloads. The new manifest must contain an
 * exact descriptor prefix copied from old_manifest and one unpublished tail
 * descriptor matching build_root->pending_l0. The exact ancestor-owned query
 * contract is always reused. A nonempty lexical_terms suffix is published as
 * one immutable lexical catalog without reconstructing the prior vocabulary.
 *
 * This function writes unreachable COW objects but does not switch the
 * metapage. Foreground active-L0 writes may interleave physically. The caller
 * must re-read the latest root, verify the pending/build identities are still
 * current, and publish with evoke_segment_read_root_seal_pending().
 */
evoke_segment_cow_write_outcome evoke_segment_pages_write_cow_append(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    const char *const *lexical_terms,
    uint32 lexical_term_count,
    const evoke_segment_payload *new_payload,
    const evoke_l0_storage_snapshot *retired_l0,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
);

/*
 * Replace one contiguous immutable descriptor range with one exact merged
 * payload. Unselected payload objects and the query contract are reused.
 * The returned descendant manifest remains unreachable until the caller
 * performs a latest-root compare-and-publish.
 */
evoke_segment_cow_write_outcome evoke_segment_pages_write_cow_replace(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 first_segment_index,
    uint32 replaced_segment_count,
    const evoke_segment_payload *replaced_payloads,
    uint32 replaced_payload_count,
    const evoke_segment_payload *replacement_payload,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
);

/*
 * Publish a document-authority-only descendant. This is used by VACUUM after
 * PostgreSQL has proved tuple versions globally dead. Immutable segments,
 * postings, folds, query metadata, and linked L0 frontiers are retained.
 */
evoke_segment_cow_write_outcome evoke_segment_pages_write_cow_document_patch(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    const evoke_document_cow_record *updates,
    size_t update_count,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
);

/*
 * Prepare an identity descendant that changes only retirement ownership. The
 * bounded handoff is derived from the ancestor manifest's authenticated
 * retired ranges. No posting, directory, fold, or query-contract object is
 * rewritten; the new manifest retires only the superseded manifest object.
 */
void evoke_segment_pages_write_cow_reclaim(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 reclaim_block_limit,
    evoke_segment_cow_result *result_out
);

/*
 * Publish one exact, term-local neutral fold as a physical-only descendant.
 * The immutable segment set, query contract, statistics, and both linked L0
 * frontiers remain unchanged. The fold bundle and copied COW path are written
 * unreachable; callers must perform a latest-root compare-and-publish before
 * readers can discover the descendant manifest.
 */
evoke_segment_cow_write_outcome evoke_segment_pages_write_cow_neutral_fold(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 term_id,
    evoke_term_cow_neutral_fold_level fold_level,
    const evoke_term_fold_bundle *fold_bundle,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
);

/*
 * Publish one epoch-bound lexical-impact specialization through the same COW
 * manifest transition as a neutral fold. Both active and pending L0 must be
 * empty; otherwise the caller must retain the exact neutral read surface.
 */
evoke_segment_cow_write_outcome evoke_segment_pages_write_cow_impact_fold(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest,
    uint32 term_id,
    const evoke_term_fold_bundle *fold_bundle,
    evoke_segment_page_reuse_arena *reuse_arena,
    evoke_segment_cow_result *result_out
);

/*
 * Append one complete, currently unreachable L0 record chain. The caller must
 * publish the returned chain through the metapage read root before readers can
 * discover it. Each page contains one independently checked fragment frame.
 */
void evoke_segment_pages_write_l0_record_chain(
    Relation index_relation,
    uint64 segment_id,
    uint32 ordinal_base,
    uint64 sequence,
    const uint8 *record_bytes,
    Size record_size,
    evoke_l0_chain_write_result *result_out
);

/*
 * Complete a reader-fenced ownership handoff after the descendant root is
 * published. Each page receives a checked WAL-logged recyclable marker before
 * it becomes an index-FSM hint.
 */
void evoke_segment_pages_publish_fsm_handoff(
    Relation index_relation,
    uint64 owner_manifest_id,
    const evoke_segment_cow_result *result
);

void evoke_segment_pages_abandon_staged_write(
    Relation index_relation,
    uint64 owner_manifest_id,
    const evoke_segment_cow_result *result
);

/* Diagnostic-only scan of unreachable ranges from a checked inventory. */
uint64 evoke_segment_pages_count_recyclable_markers(
    Relation index_relation,
    const evoke_segment_reachability_inventory *inventory
);

void evoke_l0_storage_snapshot_init(
    evoke_l0_storage_snapshot *snapshot
);

void evoke_l0_storage_snapshot_free(
    evoke_l0_storage_snapshot *snapshot
);

/*
 * Read pending then active L0 records from one checked root. Every page,
 * fragment, logical record, sequence, and published frontier is validated.
 */
void evoke_segment_pages_load_l0_snapshot(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_l0_storage_snapshot *snapshot_out
);

/*
 * Load only the immutable pending frontier selected for sealing. Concurrent
 * active records are deliberately excluded.
 */
void evoke_segment_pages_load_pending_l0_snapshot(
    Relation index_relation,
    const evoke_segment_read_root *root,
    evoke_l0_storage_snapshot *snapshot_out
);

/*
 * Stream pending then active L0 records from one checked root. The visitor
 * retains at most one reconstructed record; complete frontier payloads never
 * become backend-owned query state.
 */
void evoke_segment_pages_visit_l0_records(
    Relation index_relation,
    const evoke_segment_read_root *root,
    bool include_pending,
    bool include_active,
    evoke_l0_record_visitor visitor,
    void *visitor_context,
    evoke_l0_visit_stats *stats_out
);

void evoke_segment_reachability_inventory_init(
    evoke_segment_reachability_inventory *inventory
);

void evoke_segment_reachability_inventory_free(
    evoke_segment_reachability_inventory *inventory
);

/*
 * Validate and inventory the exact physical closure reachable from one
 * checked root. This is control-plane accounting, not a second persistent
 * locator or query accelerator.
 */
void evoke_segment_pages_inventory_reachable(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest,
    evoke_segment_reachability_inventory *inventory_out
);

#endif
