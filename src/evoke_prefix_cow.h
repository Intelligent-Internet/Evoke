#ifndef EVOKE_PREFIX_COW_H
#define EVOKE_PREFIX_COW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "evoke_block_ranges.h"
#include "evoke_segments.h"

#define EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES UINT32_C(128)
#define EVOKE_PREFIX_COW_NODE_MAX_CHILDREN UINT32_C(48)
#define EVOKE_PREFIX_COW_MAX_DEPTH UINT32_C(32)

typedef enum evoke_prefix_cow_object_kind
{
    EVOKE_PREFIX_COW_OBJECT_INVALID = 0,
    EVOKE_PREFIX_COW_OBJECT_NODE = 1,
    EVOKE_PREFIX_COW_OBJECT_LEAF = 2
} evoke_prefix_cow_object_kind;

typedef struct evoke_prefix_cow_ref
{
    evoke_prefix_cow_object_kind kind;
    uint32_t start_block;
    uint32_t page_count;
    uint32_t reserved;
    uint64_t object_id;
    uint64_t owner_manifest_id;
    uint64_t object_bytes;
    uint64_t checksum;
    uint64_t blob_checksum;
    uint32_t term_count;
    uint32_t reserved2;
} evoke_prefix_cow_ref;

typedef struct evoke_prefix_cow_key
{
    const uint8_t *bytes;
    uint32_t bytes_len;
    uint32_t term_id;
} evoke_prefix_cow_key;

typedef struct evoke_prefix_cow_entry
{
    uint8_t *bytes;
    uint32_t bytes_len;
    uint32_t term_id;
} evoke_prefix_cow_entry;

typedef struct evoke_prefix_cow_child
{
    evoke_prefix_cow_ref ref;
    uint8_t *max_key;
    uint32_t max_key_len;
} evoke_prefix_cow_child;

typedef struct evoke_prefix_cow_node
{
    uint32_t child_count;
    evoke_prefix_cow_child *children;
} evoke_prefix_cow_node;

typedef struct evoke_prefix_cow_leaf
{
    uint32_t entry_count;
    evoke_prefix_cow_entry *entries;
} evoke_prefix_cow_leaf;

typedef struct evoke_prefix_cow_object
{
    evoke_prefix_cow_ref ref;
    union
    {
        evoke_prefix_cow_node node;
        evoke_prefix_cow_leaf leaf;
    } value;
} evoke_prefix_cow_object;

typedef struct evoke_prefix_cow_tree
{
    uint32_t term_count;
    uint32_t reserved;
    uint64_t next_object_id;
    evoke_prefix_cow_ref root;
    evoke_prefix_cow_object *objects;
    size_t object_count;
    size_t object_capacity;
    evoke_block_range *retired_ranges;
    size_t retired_range_count;
    size_t retired_range_capacity;
} evoke_prefix_cow_tree;

typedef struct evoke_prefix_cow_update_stats
{
    uint32_t changed_terms;
    uint32_t read_nodes;
    uint32_t read_leaves;
    uint32_t written_nodes;
    uint32_t written_leaves;
    uint64_t written_bytes;
} evoke_prefix_cow_update_stats;

typedef evoke_status (*evoke_prefix_cow_object_loader)(
    void *context,
    const evoke_prefix_cow_ref *ref,
    evoke_prefix_cow_object *object_out
);

typedef evoke_status (*evoke_prefix_cow_object_visitor)(
    void *context,
    const evoke_prefix_cow_object *object
);

typedef evoke_status (*evoke_prefix_cow_entry_visitor)(
    void *context,
    const evoke_prefix_cow_entry *entry
);

void evoke_prefix_cow_tree_init(evoke_prefix_cow_tree *tree);
void evoke_prefix_cow_tree_free(evoke_prefix_cow_tree *tree);
void evoke_prefix_cow_object_free(evoke_prefix_cow_object *object);

evoke_status evoke_prefix_cow_tree_build(
    const evoke_prefix_cow_key *keys,
    uint32_t key_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_tree *tree_out
);

evoke_status evoke_prefix_cow_build_external_append_patch(
    const evoke_prefix_cow_ref *old_root,
    uint32_t old_term_count,
    const evoke_prefix_cow_key *keys,
    uint32_t key_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_object_loader loader,
    void *loader_context,
    evoke_prefix_cow_tree *patch_out,
    evoke_prefix_cow_update_stats *stats_out
);

evoke_status evoke_prefix_cow_scan_prefix_external(
    const evoke_prefix_cow_ref *root,
    uint32_t expected_term_count,
    const uint8_t *prefix,
    size_t prefix_len,
    evoke_prefix_cow_object_loader loader,
    void *loader_context,
    evoke_prefix_cow_entry_visitor visitor,
    void *visitor_context
);

evoke_status evoke_prefix_cow_validate_external(
    const evoke_prefix_cow_ref *root,
    uint32_t expected_term_count,
    evoke_prefix_cow_object_loader loader,
    void *loader_context
);

evoke_status evoke_prefix_cow_visit_external(
    const evoke_prefix_cow_ref *root,
    uint32_t expected_term_count,
    evoke_prefix_cow_object_loader loader,
    void *loader_context,
    evoke_prefix_cow_object_visitor visitor,
    void *visitor_context
);

evoke_status evoke_prefix_cow_object_serialize(
    const evoke_prefix_cow_object *object,
    uint8_t **bytes_out,
    size_t *size_out
);

evoke_status evoke_prefix_cow_object_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_prefix_cow_object *object_out
);

evoke_status evoke_prefix_cow_tree_prepare_object_for_storage(
    evoke_prefix_cow_tree *tree,
    uint64_t object_id,
    uint8_t **bytes_out,
    size_t *size_out
);

evoke_status evoke_prefix_cow_object_bind_storage(
    evoke_prefix_cow_object *object,
    const evoke_segment_object_ref *storage_ref
);

evoke_status evoke_prefix_cow_tree_bind_object_storage(
    evoke_prefix_cow_tree *tree,
    uint64_t object_id,
    const evoke_segment_object_ref *storage_ref
);

evoke_status evoke_prefix_cow_ref_as_segment_object_ref(
    const evoke_prefix_cow_ref *ref,
    evoke_segment_object_ref *storage_ref_out
);

#endif
