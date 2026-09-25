#ifndef EVOKE_DOCUMENT_COW_H
#define EVOKE_DOCUMENT_COW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "evoke_block_ranges.h"
#include "evoke_segments.h"

#define EVOKE_DOCUMENT_COW_LEAF_RECORDS UINT32_C(16)
#define EVOKE_DOCUMENT_COW_RADIX_BITS UINT32_C(6)
#define EVOKE_DOCUMENT_COW_RADIX_FANOUT UINT32_C(64)
#define EVOKE_DOCUMENT_COW_RADIX_LEVELS UINT32_C(5)

typedef enum evoke_document_cow_object_kind
{
    EVOKE_DOCUMENT_COW_OBJECT_INVALID = 0,
    EVOKE_DOCUMENT_COW_OBJECT_NODE = 1,
    EVOKE_DOCUMENT_COW_OBJECT_LEAF = 2
} evoke_document_cow_object_kind;

typedef struct evoke_document_cow_record
{
    evoke_document_version_record version;
    evoke_document_retirement_record retirement;
    evoke_semantic_state_record semantic_state;
    /* Physical references reachable from the current published root. */
    uint64_t lexical_residency;
    uint64_t semantic_residency;
    uint64_t event_residency;
} evoke_document_cow_record;

typedef struct evoke_document_cow_ref
{
    evoke_document_cow_object_kind kind;
    uint32_t start_block;
    uint32_t page_count;
    uint32_t reserved;
    uint64_t object_id;
    uint64_t owner_manifest_id;
    uint64_t object_bytes;
    uint64_t checksum;
    uint64_t first_document_slot;
    uint64_t document_slot_count;
    uint64_t live_document_count;
    uint64_t semantic_pending_count;
    int64_t earliest_retry_after;
    uint64_t bounded_document_count;
    uint32_t min_document_length;
    uint32_t max_document_length;
    uint64_t reusable_document_count;
    uint64_t first_reusable_document_slot;
    /* Minimum born sequence among root-relative live incarnations. */
    uint64_t min_live_born_sequence;
} evoke_document_cow_ref;

typedef struct evoke_document_cow_child
{
    uint16_t slot;
    uint16_t reserved;
    uint32_t reserved2;
    evoke_document_cow_ref ref;
} evoke_document_cow_child;

typedef struct evoke_document_cow_node
{
    uint16_t level;
    uint16_t reserved;
    uint32_t child_count;
    uint64_t prefix;
    evoke_document_cow_child children[
        EVOKE_DOCUMENT_COW_RADIX_FANOUT
    ];
} evoke_document_cow_node;

typedef struct evoke_document_cow_leaf
{
    uint64_t base_document_slot;
    uint32_t record_count;
    uint32_t reserved;
    evoke_document_cow_record records[
        EVOKE_DOCUMENT_COW_LEAF_RECORDS
    ];
} evoke_document_cow_leaf;

typedef struct evoke_document_cow_object
{
    evoke_document_cow_ref ref;
    union
    {
        evoke_document_cow_node node;
        evoke_document_cow_leaf leaf;
    } value;
} evoke_document_cow_object;

typedef struct evoke_document_cow_tree
{
    uint64_t document_slot_count;
    uint64_t next_object_id;
    evoke_document_cow_ref root;
    evoke_document_cow_object *objects;
    size_t object_count;
    size_t object_capacity;
    evoke_block_range *retired_ranges;
    size_t retired_range_count;
    size_t retired_range_capacity;
} evoke_document_cow_tree;

typedef struct evoke_document_cow_update_stats
{
    uint32_t changed_records;
    uint32_t written_nodes;
    uint32_t written_leaves;
    uint64_t written_bytes;
} evoke_document_cow_update_stats;

typedef struct evoke_document_cow_born_prefix_stats
{
    uint64_t objects_loaded;
    uint64_t records_examined;
    uint64_t heap_peak;
} evoke_document_cow_born_prefix_stats;

typedef bool (*evoke_document_cow_record_predicate)(
    void *context,
    const evoke_document_cow_record *record
);

/* Compare the complete root-relative state of one slot incarnation. */
bool evoke_document_cow_records_equal(
    const evoke_document_cow_record *left,
    const evoke_document_cow_record *right
);

bool evoke_document_cow_versions_equal(
    const evoke_document_version_record *left,
    const evoke_document_version_record *right
);

bool evoke_document_cow_record_is_l0_owned(
    const evoke_document_cow_record *record
);

uint64_t evoke_document_cow_record_last_sequence(
    const evoke_document_cow_record *record
);

typedef evoke_document_block_extrema evoke_document_cow_length_extrema;

typedef evoke_status (*evoke_document_cow_object_loader)(
    void *context,
    const evoke_document_cow_ref *ref,
    evoke_document_cow_object *object_out
);

typedef evoke_status (*evoke_document_cow_object_visitor)(
    void *context,
    const evoke_document_cow_object *object
);

void evoke_document_cow_tree_init(evoke_document_cow_tree *tree);
void evoke_document_cow_tree_free(evoke_document_cow_tree *tree);

evoke_status evoke_document_cow_tree_build(
    const evoke_document_cow_record *records,
    uint64_t record_count,
    uint64_t owner_manifest_id,
    evoke_document_cow_tree *tree_out
);

evoke_status evoke_document_cow_tree_validate(
    const evoke_document_cow_tree *tree
);

evoke_status evoke_document_cow_tree_lookup(
    const evoke_document_cow_tree *tree,
    uint64_t document_slot,
    evoke_document_cow_record *record_out
);

evoke_status evoke_document_cow_lookup_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t document_slot,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out
);

/*
 * Read one contiguous document-slot range through a single bounded radix
 * walk. The caller owns records_out, which must hold exactly
 * range_document_slot_count records. No allocation scales with the corpus.
 */
evoke_status evoke_document_cow_read_range_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t first_document_slot,
    uint64_t range_document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *records_out
);

/*
 * Read one conservative document-length range from the same COW tree that
 * owns document versions. Retired versions remain in the summary; aborted
 * slot holes do not. A zero document_count means the requested range contains
 * no scoreable version and must not be used as a lexical pruning bound.
 */
evoke_status evoke_document_cow_length_extrema_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t first_document_slot,
    uint64_t range_document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_length_extrema *extrema_out
);

evoke_status evoke_document_cow_tree_length_extrema(
    const evoke_document_cow_tree *tree,
    uint64_t first_document_slot,
    uint64_t range_document_slot_count,
    evoke_document_cow_length_extrema *extrema_out
);

evoke_status evoke_document_cow_block_extrema_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint32_t block_shift,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_length_extrema **extrema_out,
    size_t *extrema_count_out
);

evoke_status evoke_document_cow_tree_block_extrema(
    const evoke_document_cow_tree *tree,
    uint32_t block_shift,
    evoke_document_cow_length_extrema **extrema_out,
    size_t *extrema_count_out
);

void evoke_document_cow_block_extrema_free(
    evoke_document_cow_length_extrema *extrema
);

/*
 * Validate every object reachable from one published root. This verifies the
 * complete radix shape, contiguous document-slot coverage, immutable object
 * identity, physical storage ranges, checksums, and subtree summaries.
 */
evoke_status evoke_document_cow_validate_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context
);

evoke_status evoke_document_cow_visit_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_object_visitor visitor,
    void *visitor_context
);

/*
 * Read the first live root-relative incarnations in exact born-sequence order.
 * The best-first cursor expands only subtrees that can contain the next
 * result. Its allocation is proportional to limit times the fixed radix
 * fanout/depth, never document_slot_count.
 */
evoke_status evoke_document_cow_collect_live_born_prefix_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    size_t limit,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *records_out,
    size_t record_capacity,
    size_t *record_count_out,
    evoke_document_cow_born_prefix_stats *stats_out
);

/*
 * Read the first matching live incarnations in exact born-sequence order.
 * Rejected records are consumed by the same best-first cursor, so output and
 * scratch allocation remain proportional to limit and the fixed radix shape.
 */
evoke_status evoke_document_cow_collect_live_born_prefix_matching_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    size_t limit,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record_predicate predicate,
    void *predicate_context,
    evoke_document_cow_record *records_out,
    size_t record_capacity,
    size_t *record_count_out,
    evoke_document_cow_born_prefix_stats *stats_out
);

evoke_status evoke_document_cow_find_actionable_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    int64_t now,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out,
    bool *found_out
);

evoke_status evoke_document_cow_find_actionable_external_from(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t first_document_slot,
    int64_t now,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out,
    bool *found_out
);

/*
 * Find the first root-relative score slot whose prior incarnation has no
 * remaining posting or payload-event residency. A zero reusable summary
 * returns without loading any COW object.
 */
evoke_status evoke_document_cow_find_reusable_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out,
    bool *found_out
);

evoke_status evoke_document_cow_find_reusable_external_from(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t first_document_slot,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out,
    bool *found_out
);

evoke_status evoke_document_cow_build_external_patch(
    const evoke_document_cow_ref *old_root,
    uint64_t old_document_slot_count,
    uint64_t next_document_slot_count,
    const evoke_document_cow_record *updates,
    size_t update_count,
    uint64_t owner_manifest_id,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_tree *patch_out,
    evoke_document_cow_update_stats *stats_out
);

evoke_status evoke_document_cow_tree_prepare_object_for_storage(
    evoke_document_cow_tree *tree,
    uint64_t object_id,
    uint8_t **bytes_out,
    size_t *size_out
);

evoke_status evoke_document_cow_tree_bind_object_storage(
    evoke_document_cow_tree *tree,
    uint64_t object_id,
    const evoke_segment_object_ref *storage_ref
);

evoke_status evoke_document_cow_object_bind_storage(
    evoke_document_cow_object *object,
    const evoke_segment_object_ref *storage_ref
);

evoke_status evoke_document_cow_ref_as_segment_object_ref(
    const evoke_document_cow_ref *ref,
    evoke_segment_object_ref *storage_ref_out
);

evoke_status evoke_document_cow_tree_object_serialize(
    const evoke_document_cow_tree *tree,
    const evoke_document_cow_ref *ref,
    uint8_t **bytes_out,
    size_t *size_out
);

evoke_status evoke_document_cow_object_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_document_cow_object *object_out
);

#endif
