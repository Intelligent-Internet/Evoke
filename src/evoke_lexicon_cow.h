#ifndef EVOKE_LEXICON_COW_H
#define EVOKE_LEXICON_COW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "evoke_block_ranges.h"
#include "evoke_segments.h"

#define EVOKE_LEXICON_COW_RADIX_BITS UINT32_C(6)
#define EVOKE_LEXICON_COW_RADIX_FANOUT UINT32_C(64)
#define EVOKE_LEXICON_COW_RADIX_LEVELS UINT32_C(11)
/* Keep compact buckets within two payload pages on standard 8 KiB builds. */
#define EVOKE_LEXICON_COW_BUCKET_TARGET_BYTES UINT32_C(16000)

typedef enum evoke_lexicon_cow_object_kind
{
    EVOKE_LEXICON_COW_OBJECT_INVALID = 0,
    EVOKE_LEXICON_COW_OBJECT_NODE = 1,
    EVOKE_LEXICON_COW_OBJECT_BUCKET = 2
} evoke_lexicon_cow_object_kind;

typedef struct evoke_lexicon_cow_ref
{
    evoke_lexicon_cow_object_kind kind;
    uint32_t start_block;
    uint32_t page_count;
    uint32_t reserved;
    uint64_t object_id;
    uint64_t owner_manifest_id;
    uint64_t object_bytes;
    uint64_t checksum;
    uint64_t blob_checksum;
} evoke_lexicon_cow_ref;

typedef struct evoke_lexicon_cow_key
{
    const uint8_t *bytes;
    uint32_t bytes_len;
    uint32_t term_id;
} evoke_lexicon_cow_key;

typedef struct evoke_lexicon_cow_entry
{
    uint8_t *bytes;
    uint32_t bytes_len;
    uint32_t term_id;
    uint64_t hash;
} evoke_lexicon_cow_entry;

typedef struct evoke_lexicon_cow_child
{
    uint16_t slot;
    uint16_t reserved;
    uint32_t term_count;
    evoke_lexicon_cow_ref ref;
} evoke_lexicon_cow_child;

typedef struct evoke_lexicon_cow_node
{
    uint16_t level;
    uint16_t reserved;
    uint32_t child_count;
    uint32_t term_count;
    uint32_t reserved2;
    uint64_t prefix;
    evoke_lexicon_cow_child children[EVOKE_LEXICON_COW_RADIX_FANOUT];
} evoke_lexicon_cow_node;

typedef struct evoke_lexicon_cow_bucket
{
    uint16_t depth;
    uint16_t reserved;
    uint32_t entry_count;
    uint64_t prefix;
    evoke_lexicon_cow_entry *entries;
    /* Deserialized buckets own token bytes in one allocation. */
    uint8_t *owned_bytes;
} evoke_lexicon_cow_bucket;

typedef struct evoke_lexicon_cow_object
{
    evoke_lexicon_cow_ref ref;
    union
    {
        evoke_lexicon_cow_node node;
        evoke_lexicon_cow_bucket bucket;
    } value;
} evoke_lexicon_cow_object;

typedef struct evoke_lexicon_cow_tree
{
    uint64_t hash_seed;
    uint32_t term_count;
    uint32_t reserved;
    uint64_t next_object_id;
    evoke_lexicon_cow_ref root;
    evoke_lexicon_cow_object *objects;
    size_t object_count;
    size_t object_capacity;
    evoke_block_range *retired_ranges;
    size_t retired_range_count;
    size_t retired_range_capacity;
} evoke_lexicon_cow_tree;

typedef struct evoke_lexicon_cow_update_stats
{
    uint32_t changed_terms;
    uint32_t read_nodes;
    uint32_t read_buckets;
    uint32_t written_nodes;
    uint32_t written_buckets;
    uint64_t written_bytes;
} evoke_lexicon_cow_update_stats;

typedef evoke_status (*evoke_lexicon_cow_object_loader)(
    void *context,
    const evoke_lexicon_cow_ref *ref,
    evoke_lexicon_cow_object *object_out
);

typedef evoke_status (*evoke_lexicon_cow_serialized_loader)(
    void *context,
    const evoke_lexicon_cow_ref *ref,
    uint8_t **bytes_out,
    size_t *size_out
);

typedef void (*evoke_lexicon_cow_serialized_releaser)(
    void *context,
    uint8_t *bytes
);

typedef evoke_status (*evoke_lexicon_cow_object_visitor)(
    void *context,
    const evoke_lexicon_cow_object *object
);

typedef evoke_status (*evoke_lexicon_cow_entry_visitor)(
    void *context,
    const evoke_lexicon_cow_entry *entry
);

void evoke_lexicon_cow_tree_init(evoke_lexicon_cow_tree *tree);
void evoke_lexicon_cow_tree_free(evoke_lexicon_cow_tree *tree);
void evoke_lexicon_cow_object_free(evoke_lexicon_cow_object *object);

uint64_t evoke_lexicon_cow_hash(
    const uint8_t *bytes,
    size_t bytes_len,
    uint64_t seed
);

evoke_status evoke_lexicon_cow_tree_build(
    const evoke_lexicon_cow_key *keys,
    uint32_t key_count,
    uint64_t hash_seed,
    uint64_t owner_manifest_id,
    evoke_lexicon_cow_tree *tree_out
);

evoke_status evoke_lexicon_cow_tree_lookup(
    const evoke_lexicon_cow_tree *tree,
    const uint8_t *bytes,
    size_t bytes_len,
    uint32_t *term_id_out,
    bool *found_out
);

evoke_status evoke_lexicon_cow_tree_lookup_at(
    const evoke_lexicon_cow_tree *tree,
    const evoke_lexicon_cow_ref *root,
    const uint8_t *bytes,
    size_t bytes_len,
    uint32_t *term_id_out,
    bool *found_out
);

/*
 * Append a contiguous stable-id suffix by path-copying only buckets addressed
 * by the new keys and their shared radix ancestors. The saved old root remains
 * readable because every existing object is immutable.
 */
evoke_status evoke_lexicon_cow_tree_append(
    evoke_lexicon_cow_tree *tree,
    const evoke_lexicon_cow_ref *old_root,
    uint32_t old_term_count,
    const evoke_lexicon_cow_key *keys,
    uint32_t key_count,
    uint64_t owner_manifest_id,
    evoke_lexicon_cow_update_stats *stats_out
);

evoke_status evoke_lexicon_cow_lookup_external(
    const evoke_lexicon_cow_ref *root,
    uint64_t hash_seed,
    const uint8_t *bytes,
    size_t bytes_len,
    evoke_lexicon_cow_object_loader loader,
    void *loader_context,
    uint32_t *term_id_out,
    bool *found_out
);

/*
 * Resolve one key directly from checked serialized objects. This avoids
 * allocating and copying a complete bucket on the query hot path.
 */
evoke_status evoke_lexicon_cow_lookup_serialized_external(
    const evoke_lexicon_cow_object *root_object,
    uint64_t hash_seed,
    const uint8_t *bytes,
    size_t bytes_len,
    evoke_lexicon_cow_serialized_loader loader,
    evoke_lexicon_cow_serialized_releaser releaser,
    void *loader_context,
    uint32_t *term_id_out,
    bool *found_out
);

evoke_status evoke_lexicon_cow_validate_external(
    const evoke_lexicon_cow_ref *root,
    uint64_t hash_seed,
    uint32_t expected_term_count,
    evoke_lexicon_cow_object_loader loader,
    void *loader_context
);

evoke_status evoke_lexicon_cow_visit_external(
    const evoke_lexicon_cow_ref *root,
    uint64_t hash_seed,
    uint32_t expected_term_count,
    evoke_lexicon_cow_object_loader loader,
    void *loader_context,
    evoke_lexicon_cow_object_visitor visitor,
    void *visitor_context
);

/*
 * Stream a validated external tree one bucket at a time. Unlike the full
 * validation visitor, this path does not allocate vocabulary-sized duplicate
 * tracking and is suitable for bounded foreground prefix expansion.
 */
evoke_status evoke_lexicon_cow_scan_external(
    const evoke_lexicon_cow_ref *root,
    uint64_t hash_seed,
    uint32_t expected_term_count,
    evoke_lexicon_cow_object_loader loader,
    void *loader_context,
    evoke_lexicon_cow_entry_visitor visitor,
    void *visitor_context
);

evoke_status evoke_lexicon_cow_build_external_append_patch(
    const evoke_lexicon_cow_ref *old_root,
    uint64_t hash_seed,
    uint32_t old_term_count,
    const evoke_lexicon_cow_key *keys,
    uint32_t key_count,
    uint64_t owner_manifest_id,
    evoke_lexicon_cow_object_loader loader,
    void *loader_context,
    evoke_lexicon_cow_tree *patch_out,
    evoke_lexicon_cow_update_stats *stats_out
);

evoke_status evoke_lexicon_cow_tree_validate_at(
    const evoke_lexicon_cow_tree *tree,
    const evoke_lexicon_cow_ref *root,
    uint32_t expected_term_count
);

evoke_status evoke_lexicon_cow_tree_object_serialize(
    const evoke_lexicon_cow_tree *tree,
    const evoke_lexicon_cow_ref *ref,
    uint8_t **bytes_out,
    size_t *size_out
);

evoke_status evoke_lexicon_cow_object_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_lexicon_cow_object *object_out
);

evoke_status evoke_lexicon_cow_tree_prepare_object_for_storage(
    evoke_lexicon_cow_tree *tree,
    uint64_t object_id,
    uint8_t **bytes_out,
    size_t *size_out
);

evoke_status evoke_lexicon_cow_tree_bind_object_storage(
    evoke_lexicon_cow_tree *tree,
    uint64_t object_id,
    const evoke_segment_object_ref *storage_ref
);

evoke_status evoke_lexicon_cow_object_bind_storage(
    evoke_lexicon_cow_object *object,
    const evoke_segment_object_ref *storage_ref
);

evoke_status evoke_lexicon_cow_ref_as_segment_object_ref(
    const evoke_lexicon_cow_ref *ref,
    evoke_segment_object_ref *storage_ref_out
);

#endif
