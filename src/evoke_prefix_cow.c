#include "evoke_prefix_cow.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_PREFIX_COW_MAGIC UINT32_C(0x58465032)
#define EVOKE_PREFIX_COW_VERSION UINT16_C(1)
#define EVOKE_PREFIX_COW_HEADER_SIZE 80U
#define EVOKE_PREFIX_COW_LEAF_ENTRY_SIZE 16U
#define EVOKE_PREFIX_COW_NODE_CHILD_SIZE 80U
#define EVOKE_PREFIX_COW_BLOB_CHECKSUM_OFFSET 56U

typedef struct evoke_prefix_cow_span
{
    evoke_prefix_cow_ref ref;
    uint8_t *max_key;
    uint32_t max_key_len;
} evoke_prefix_cow_span;

static evoke_status evoke_prefix_cow_object_encode(
    const evoke_prefix_cow_object *object,
    bool require_bound_children,
    uint8_t **bytes_out,
    size_t *size_out
);

static void
evoke_prefix_cow_write_u16(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t) (value & UINT16_C(0xff));
    dst[1] = (uint8_t) ((value >> 8) & UINT16_C(0xff));
}

static uint16_t
evoke_prefix_cow_read_u16(const uint8_t *src)
{
    return (uint16_t) src[0] |
        (uint16_t) ((uint16_t) src[1] << 8);
}

static void
evoke_prefix_cow_write_u32(uint8_t *dst, uint32_t value)
{
    dst[0] = (uint8_t) (value & UINT32_C(0xff));
    dst[1] = (uint8_t) ((value >> 8) & UINT32_C(0xff));
    dst[2] = (uint8_t) ((value >> 16) & UINT32_C(0xff));
    dst[3] = (uint8_t) ((value >> 24) & UINT32_C(0xff));
}

static uint32_t
evoke_prefix_cow_read_u32(const uint8_t *src)
{
    return (uint32_t) src[0] |
        ((uint32_t) src[1] << 8) |
        ((uint32_t) src[2] << 16) |
        ((uint32_t) src[3] << 24);
}

static void
evoke_prefix_cow_write_u64(uint8_t *dst, uint64_t value)
{
    for (uint32_t index = 0; index < 8; index++)
    {
        dst[index] = (uint8_t) (
            (value >> (index * 8)) & UINT64_C(0xff)
        );
    }
}

static uint64_t
evoke_prefix_cow_read_u64(const uint8_t *src)
{
    uint64_t value = 0;

    for (uint32_t index = 0; index < 8; index++)
    {
        value |= (uint64_t) src[index] << (index * 8);
    }
    return value;
}

static uint64_t
evoke_prefix_cow_checksum_bytes(
    const uint8_t *bytes,
    size_t size,
    size_t zero_offset,
    size_t zero_size
)
{
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value = index >= zero_offset &&
            index < zero_offset + zero_size
            ? 0
            : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static void
evoke_prefix_cow_checksum_u64(uint64_t *checksum, uint64_t value)
{
    for (uint32_t index = 0; index < 8; index++)
    {
        *checksum ^= (uint8_t) (value >> (index * 8));
        *checksum *= UINT64_C(1099511628211);
    }
}

static void
evoke_prefix_cow_checksum_blob(
    uint64_t *checksum,
    const uint8_t *bytes,
    size_t size
)
{
    for (size_t index = 0; index < size; index++)
    {
        *checksum ^= bytes[index];
        *checksum *= UINT64_C(1099511628211);
    }
}

static int
evoke_prefix_cow_compare_bytes(
    const uint8_t *left,
    uint32_t left_len,
    const uint8_t *right,
    uint32_t right_len
)
{
    size_t common = left_len < right_len ? left_len : right_len;
    int compared = common == 0 ? 0 : memcmp(left, right, common);

    if (compared != 0)
    {
        return compared;
    }
    if (left_len < right_len)
    {
        return -1;
    }
    if (left_len > right_len)
    {
        return 1;
    }
    return 0;
}

static int
evoke_prefix_cow_compare_keys(const void *left_ptr, const void *right_ptr)
{
    const evoke_prefix_cow_key *left = left_ptr;
    const evoke_prefix_cow_key *right = right_ptr;

    return evoke_prefix_cow_compare_bytes(
        left->bytes,
        left->bytes_len,
        right->bytes,
        right->bytes_len
    );
}

static bool
evoke_prefix_cow_ref_is_unbound(const evoke_prefix_cow_ref *ref)
{
    return ref != NULL && ref->start_block == 0 && ref->page_count == 0;
}

static bool
evoke_prefix_cow_ref_is_valid(
    const evoke_prefix_cow_ref *ref,
    bool allow_unbound
)
{
    bool unbound;

    if (ref == NULL ||
        (ref->kind != EVOKE_PREFIX_COW_OBJECT_NODE &&
         ref->kind != EVOKE_PREFIX_COW_OBJECT_LEAF) ||
        ref->reserved != 0 || ref->reserved2 != 0 ||
        ref->object_id == 0 || ref->owner_manifest_id == 0 ||
        ref->object_bytes < EVOKE_PREFIX_COW_HEADER_SIZE ||
        ref->checksum == 0 || ref->term_count == 0)
    {
        return false;
    }
    unbound = evoke_prefix_cow_ref_is_unbound(ref);
    if (unbound)
    {
        return allow_unbound;
    }
    return ref->start_block > 0 && ref->page_count > 0 &&
        ref->blob_checksum != 0;
}

static bool
evoke_prefix_cow_refs_equal(
    const evoke_prefix_cow_ref *left,
    const evoke_prefix_cow_ref *right
)
{
    return left != NULL && right != NULL &&
        left->kind == right->kind &&
        left->start_block == right->start_block &&
        left->page_count == right->page_count &&
        left->reserved == right->reserved &&
        left->object_id == right->object_id &&
        left->owner_manifest_id == right->owner_manifest_id &&
        left->object_bytes == right->object_bytes &&
        left->checksum == right->checksum &&
        left->blob_checksum == right->blob_checksum &&
        left->term_count == right->term_count &&
        left->reserved2 == right->reserved2;
}

static void
evoke_prefix_cow_entry_free(evoke_prefix_cow_entry *entry)
{
    if (entry != NULL)
    {
        free(entry->bytes);
        memset(entry, 0, sizeof(*entry));
    }
}

static void
evoke_prefix_cow_child_free(evoke_prefix_cow_child *child)
{
    if (child != NULL)
    {
        free(child->max_key);
        memset(child, 0, sizeof(*child));
    }
}

void
evoke_prefix_cow_object_free(evoke_prefix_cow_object *object)
{
    if (object == NULL)
    {
        return;
    }
    if (object->ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
    {
        for (uint32_t index = 0;
             index < object->value.leaf.entry_count;
             index++)
        {
            evoke_prefix_cow_entry_free(&object->value.leaf.entries[index]);
        }
        free(object->value.leaf.entries);
    }
    else if (object->ref.kind == EVOKE_PREFIX_COW_OBJECT_NODE)
    {
        for (uint32_t index = 0;
             index < object->value.node.child_count;
             index++)
        {
            evoke_prefix_cow_child_free(&object->value.node.children[index]);
        }
        free(object->value.node.children);
    }
    memset(object, 0, sizeof(*object));
}

void
evoke_prefix_cow_tree_init(evoke_prefix_cow_tree *tree)
{
    if (tree != NULL)
    {
        memset(tree, 0, sizeof(*tree));
        tree->next_object_id = 1;
    }
}

void
evoke_prefix_cow_tree_free(evoke_prefix_cow_tree *tree)
{
    if (tree == NULL)
    {
        return;
    }
    for (size_t index = 0; index < tree->object_count; index++)
    {
        evoke_prefix_cow_object_free(&tree->objects[index]);
    }
    free(tree->objects);
    free(tree->retired_ranges);
    memset(tree, 0, sizeof(*tree));
}

static void
evoke_prefix_cow_span_free(evoke_prefix_cow_span *span)
{
    if (span != NULL)
    {
        free(span->max_key);
        memset(span, 0, sizeof(*span));
    }
}

static void
evoke_prefix_cow_spans_free(
    evoke_prefix_cow_span *spans,
    size_t span_count
)
{
    if (spans == NULL)
    {
        return;
    }
    for (size_t index = 0; index < span_count; index++)
    {
        evoke_prefix_cow_span_free(&spans[index]);
    }
    free(spans);
}

static evoke_status
evoke_prefix_cow_copy_bytes(
    const uint8_t *source,
    uint32_t source_len,
    uint8_t **bytes_out
)
{
    uint8_t *copy;

    if (source == NULL || source_len == 0 || bytes_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    copy = malloc(source_len);
    if (copy == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    memcpy(copy, source, source_len);
    *bytes_out = copy;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_object_serialized_size(
    const evoke_prefix_cow_object *object,
    size_t *size_out
)
{
    size_t size = EVOKE_PREFIX_COW_HEADER_SIZE;
    uint32_t item_count;

    if (object == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (object->ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
    {
        item_count = object->value.leaf.entry_count;
        if (item_count == 0 ||
            item_count > EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES ||
            item_count >
                (SIZE_MAX - size) / EVOKE_PREFIX_COW_LEAF_ENTRY_SIZE)
        {
            return EVOKE_ERR_FORMAT;
        }
        size += (size_t) item_count * EVOKE_PREFIX_COW_LEAF_ENTRY_SIZE;
        for (uint32_t index = 0; index < item_count; index++)
        {
            const evoke_prefix_cow_entry *entry =
                &object->value.leaf.entries[index];

            if (entry->bytes == NULL || entry->bytes_len == 0 ||
                entry->bytes_len > SIZE_MAX - size)
            {
                return EVOKE_ERR_FORMAT;
            }
            size += entry->bytes_len;
        }
    }
    else if (object->ref.kind == EVOKE_PREFIX_COW_OBJECT_NODE)
    {
        item_count = object->value.node.child_count;
        if (item_count < 2 ||
            item_count > EVOKE_PREFIX_COW_NODE_MAX_CHILDREN ||
            item_count >
                (SIZE_MAX - size) / EVOKE_PREFIX_COW_NODE_CHILD_SIZE)
        {
            return EVOKE_ERR_FORMAT;
        }
        size += (size_t) item_count * EVOKE_PREFIX_COW_NODE_CHILD_SIZE;
        for (uint32_t index = 0; index < item_count; index++)
        {
            const evoke_prefix_cow_child *child =
                &object->value.node.children[index];

            if (child->max_key == NULL || child->max_key_len == 0 ||
                child->max_key_len > SIZE_MAX - size)
            {
                return EVOKE_ERR_FORMAT;
            }
            size += child->max_key_len;
        }
    }
    else
    {
        return EVOKE_ERR_FORMAT;
    }
    *size_out = size;
    return EVOKE_OK;
}

static uint64_t
evoke_prefix_cow_object_logical_checksum(
    const evoke_prefix_cow_object *object
)
{
    uint64_t checksum = UINT64_C(14695981039346656037);

    evoke_prefix_cow_checksum_u64(&checksum, object->ref.kind);
    evoke_prefix_cow_checksum_u64(&checksum, object->ref.object_id);
    evoke_prefix_cow_checksum_u64(
        &checksum,
        object->ref.owner_manifest_id
    );
    evoke_prefix_cow_checksum_u64(&checksum, object->ref.term_count);
    if (object->ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
    {
        evoke_prefix_cow_checksum_u64(
            &checksum,
            object->value.leaf.entry_count
        );
        for (uint32_t index = 0;
             index < object->value.leaf.entry_count;
             index++)
        {
            const evoke_prefix_cow_entry *entry =
                &object->value.leaf.entries[index];

            evoke_prefix_cow_checksum_u64(&checksum, entry->term_id);
            evoke_prefix_cow_checksum_u64(&checksum, entry->bytes_len);
            evoke_prefix_cow_checksum_blob(
                &checksum,
                entry->bytes,
                entry->bytes_len
            );
        }
    }
    else
    {
        evoke_prefix_cow_checksum_u64(
            &checksum,
            object->value.node.child_count
        );
        for (uint32_t index = 0;
             index < object->value.node.child_count;
             index++)
        {
            const evoke_prefix_cow_child *child =
                &object->value.node.children[index];

            evoke_prefix_cow_checksum_u64(&checksum, child->ref.kind);
            evoke_prefix_cow_checksum_u64(
                &checksum,
                child->ref.object_id
            );
            evoke_prefix_cow_checksum_u64(
                &checksum,
                child->ref.owner_manifest_id
            );
            evoke_prefix_cow_checksum_u64(
                &checksum,
                child->ref.term_count
            );
            evoke_prefix_cow_checksum_u64(
                &checksum,
                child->ref.checksum
            );
            evoke_prefix_cow_checksum_u64(
                &checksum,
                child->max_key_len
            );
            evoke_prefix_cow_checksum_blob(
                &checksum,
                child->max_key,
                child->max_key_len
            );
        }
    }
    return checksum;
}

static evoke_status
evoke_prefix_cow_object_shape_validate(
    const evoke_prefix_cow_object *object,
    bool require_bound_children
)
{
    size_t serialized_size;
    uint64_t term_count = 0;
    evoke_status status;

    if (object == NULL ||
        !evoke_prefix_cow_ref_is_valid(&object->ref, true))
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_prefix_cow_object_serialized_size(
        object,
        &serialized_size
    );
    if (status != EVOKE_OK || object->ref.object_bytes != serialized_size ||
        object->ref.checksum !=
            evoke_prefix_cow_object_logical_checksum(object))
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    if (object->ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
    {
        const evoke_prefix_cow_leaf *leaf = &object->value.leaf;

        if (leaf->entries == NULL ||
            leaf->entry_count != object->ref.term_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t index = 0; index < leaf->entry_count; index++)
        {
            const evoke_prefix_cow_entry *entry = &leaf->entries[index];

            if (entry->bytes == NULL || entry->bytes_len == 0 ||
                (index > 0 && evoke_prefix_cow_compare_bytes(
                    leaf->entries[index - 1].bytes,
                    leaf->entries[index - 1].bytes_len,
                    entry->bytes,
                    entry->bytes_len
                ) >= 0))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        return EVOKE_OK;
    }
    if (object->value.node.children == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t index = 0;
         index < object->value.node.child_count;
         index++)
    {
        const evoke_prefix_cow_child *child =
            &object->value.node.children[index];

        if (!evoke_prefix_cow_ref_is_valid(
                &child->ref,
                !require_bound_children) ||
            child->max_key == NULL || child->max_key_len == 0 ||
            child->ref.owner_manifest_id > object->ref.owner_manifest_id ||
            (index > 0 && evoke_prefix_cow_compare_bytes(
                object->value.node.children[index - 1].max_key,
                object->value.node.children[index - 1].max_key_len,
                child->max_key,
                child->max_key_len
            ) >= 0) ||
            UINT32_MAX - term_count < child->ref.term_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        term_count += child->ref.term_count;
    }
    return term_count == object->ref.term_count
        ? EVOKE_OK
        : EVOKE_ERR_FORMAT;
}

static evoke_status
evoke_prefix_cow_object_refresh(evoke_prefix_cow_object *object)
{
    uint8_t *bytes = NULL;
    size_t serialized_size;
    evoke_status status;

    if (object == NULL || object->ref.object_id == 0 ||
        object->ref.owner_manifest_id == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_prefix_cow_object_serialized_size(
        object,
        &serialized_size
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    object->ref.object_bytes = serialized_size;
    object->ref.checksum = evoke_prefix_cow_object_logical_checksum(object);
    object->ref.blob_checksum = UINT64_C(1);
    status = evoke_prefix_cow_object_shape_validate(object, false);
    if (status == EVOKE_OK)
    {
        status = evoke_prefix_cow_object_encode(
            object,
            false,
            &bytes,
            &serialized_size
        );
    }
    if (status == EVOKE_OK)
    {
        object->ref.blob_checksum = evoke_segment_blob_checksum(
            bytes,
            serialized_size
        );
        if (object->ref.blob_checksum == 0)
        {
            status = EVOKE_ERR_FORMAT;
        }
    }
    free(bytes);
    return status;
}

static evoke_status
evoke_prefix_cow_tree_reserve_object(evoke_prefix_cow_tree *tree)
{
    evoke_prefix_cow_object *objects;
    size_t capacity;

    if (tree->object_count < tree->object_capacity)
    {
        return EVOKE_OK;
    }
    capacity = tree->object_capacity == 0 ? 16 : tree->object_capacity * 2;
    if (capacity < tree->object_capacity ||
        capacity > SIZE_MAX / sizeof(*objects))
    {
        return EVOKE_ERR_RANGE;
    }
    objects = realloc(tree->objects, capacity * sizeof(*objects));
    if (objects == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    memset(
        objects + tree->object_capacity,
        0,
        (capacity - tree->object_capacity) * sizeof(*objects)
    );
    tree->objects = objects;
    tree->object_capacity = capacity;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_tree_add_object(
    evoke_prefix_cow_tree *tree,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_object *object,
    evoke_prefix_cow_update_stats *stats,
    evoke_prefix_cow_ref *ref_out
)
{
    evoke_status status;

    if (tree == NULL || owner_manifest_id == 0 || object == NULL ||
        ref_out == NULL || tree->next_object_id == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_prefix_cow_tree_reserve_object(tree);
    if (status != EVOKE_OK)
    {
        return status;
    }
    object->ref.object_id = tree->next_object_id++;
    object->ref.owner_manifest_id = owner_manifest_id;
    status = evoke_prefix_cow_object_refresh(object);
    if (status != EVOKE_OK)
    {
        return status;
    }
    tree->objects[tree->object_count++] = *object;
    *ref_out = object->ref;
    if (stats != NULL)
    {
        if (object->ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
        {
            stats->written_leaves++;
        }
        else
        {
            stats->written_nodes++;
        }
        stats->written_bytes += object->ref.object_bytes;
    }
    memset(object, 0, sizeof(*object));
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_span_from_object(
    const evoke_prefix_cow_object *object,
    evoke_prefix_cow_span *span_out
)
{
    const uint8_t *max_key;
    uint32_t max_key_len;
    evoke_prefix_cow_span span = {0};
    evoke_status status;

    if (object == NULL || span_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (object->ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
    {
        const evoke_prefix_cow_entry *entry =
            &object->value.leaf.entries[
                object->value.leaf.entry_count - 1
            ];

        max_key = entry->bytes;
        max_key_len = entry->bytes_len;
    }
    else
    {
        const evoke_prefix_cow_child *child =
            &object->value.node.children[
                object->value.node.child_count - 1
            ];

        max_key = child->max_key;
        max_key_len = child->max_key_len;
    }
    span.ref = object->ref;
    status = evoke_prefix_cow_copy_bytes(
        max_key,
        max_key_len,
        &span.max_key
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    span.max_key_len = max_key_len;
    *span_out = span;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_span_copy(
    const evoke_prefix_cow_ref *ref,
    const uint8_t *max_key,
    uint32_t max_key_len,
    evoke_prefix_cow_span *span_out
)
{
    evoke_prefix_cow_span span = {0};
    evoke_status status;

    if (ref == NULL || max_key == NULL || max_key_len == 0 ||
        span_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    span.ref = *ref;
    status = evoke_prefix_cow_copy_bytes(
        max_key,
        max_key_len,
        &span.max_key
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    span.max_key_len = max_key_len;
    *span_out = span;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_copy_and_sort_keys(
    const evoke_prefix_cow_key *keys,
    uint32_t key_count,
    uint32_t first_term_id,
    evoke_prefix_cow_key **sorted_out
)
{
    evoke_prefix_cow_key *sorted;
    bool *seen_ids;

    if (keys == NULL || key_count == 0 || sorted_out == NULL ||
        (uint64_t) first_term_id + key_count > UINT32_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    *sorted_out = NULL;
    sorted = malloc((size_t) key_count * sizeof(*sorted));
    seen_ids = calloc(key_count, sizeof(*seen_ids));
    if (sorted == NULL || seen_ids == NULL)
    {
        free(sorted);
        free(seen_ids);
        return EVOKE_ERR_NOMEM;
    }
    memcpy(sorted, keys, (size_t) key_count * sizeof(*sorted));
    for (uint32_t index = 0; index < key_count; index++)
    {
        uint32_t id_offset;

        if (sorted[index].bytes == NULL || sorted[index].bytes_len == 0 ||
            sorted[index].term_id < first_term_id)
        {
            free(sorted);
            free(seen_ids);
            return EVOKE_ERR_FORMAT;
        }
        id_offset = sorted[index].term_id - first_term_id;
        if (id_offset >= key_count || seen_ids[id_offset])
        {
            free(sorted);
            free(seen_ids);
            return EVOKE_ERR_FORMAT;
        }
        seen_ids[id_offset] = true;
    }
    free(seen_ids);
    qsort(sorted, key_count, sizeof(*sorted), evoke_prefix_cow_compare_keys);
    for (uint32_t index = 1; index < key_count; index++)
    {
        if (evoke_prefix_cow_compare_keys(
                &sorted[index - 1],
                &sorted[index]) == 0)
        {
            free(sorted);
            return EVOKE_ERR_FORMAT;
        }
    }
    *sorted_out = sorted;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_make_leaf_from_keys(
    evoke_prefix_cow_tree *tree,
    const evoke_prefix_cow_key *keys,
    uint32_t key_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_update_stats *stats,
    evoke_prefix_cow_span *span_out
)
{
    evoke_prefix_cow_object object = {0};
    evoke_prefix_cow_ref ref;
    evoke_status status;

    if (tree == NULL || keys == NULL || key_count == 0 ||
        key_count > EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES ||
        span_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    object.ref.kind = EVOKE_PREFIX_COW_OBJECT_LEAF;
    object.ref.term_count = key_count;
    object.value.leaf.entry_count = key_count;
    object.value.leaf.entries = calloc(
        key_count,
        sizeof(*object.value.leaf.entries)
    );
    if (object.value.leaf.entries == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t index = 0; index < key_count; index++)
    {
        evoke_prefix_cow_entry *entry =
            &object.value.leaf.entries[index];

        entry->bytes_len = keys[index].bytes_len;
        entry->term_id = keys[index].term_id;
        status = evoke_prefix_cow_copy_bytes(
            keys[index].bytes,
            keys[index].bytes_len,
            &entry->bytes
        );
        if (status != EVOKE_OK)
        {
            evoke_prefix_cow_object_free(&object);
            return status;
        }
    }
    status = evoke_prefix_cow_tree_add_object(
        tree,
        owner_manifest_id,
        &object,
        stats,
        &ref
    );
    if (status != EVOKE_OK)
    {
        evoke_prefix_cow_object_free(&object);
        return status;
    }
    return evoke_prefix_cow_span_from_object(
        &tree->objects[ref.object_id - 1],
        span_out
    );
}

static evoke_status
evoke_prefix_cow_make_leaf_from_entries(
    evoke_prefix_cow_tree *tree,
    const evoke_prefix_cow_entry *entries,
    uint32_t entry_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_update_stats *stats,
    evoke_prefix_cow_span *span_out
)
{
    evoke_prefix_cow_key *keys;
    evoke_status status;

    if (entries == NULL || entry_count == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    keys = malloc((size_t) entry_count * sizeof(*keys));
    if (keys == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t index = 0; index < entry_count; index++)
    {
        keys[index].bytes = entries[index].bytes;
        keys[index].bytes_len = entries[index].bytes_len;
        keys[index].term_id = entries[index].term_id;
    }
    status = evoke_prefix_cow_make_leaf_from_keys(
        tree,
        keys,
        entry_count,
        owner_manifest_id,
        stats,
        span_out
    );
    free(keys);
    return status;
}

static evoke_status
evoke_prefix_cow_make_node(
    evoke_prefix_cow_tree *tree,
    const evoke_prefix_cow_span *children,
    uint32_t child_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_update_stats *stats,
    evoke_prefix_cow_span *span_out
)
{
    evoke_prefix_cow_object object = {0};
    evoke_prefix_cow_ref ref;
    uint64_t term_count = 0;
    evoke_status status;

    if (tree == NULL || children == NULL || child_count < 2 ||
        child_count > EVOKE_PREFIX_COW_NODE_MAX_CHILDREN ||
        span_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    object.ref.kind = EVOKE_PREFIX_COW_OBJECT_NODE;
    object.value.node.child_count = child_count;
    object.value.node.children = calloc(
        child_count,
        sizeof(*object.value.node.children)
    );
    if (object.value.node.children == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t index = 0; index < child_count; index++)
    {
        evoke_prefix_cow_child *child =
            &object.value.node.children[index];

        if (UINT32_MAX - term_count < children[index].ref.term_count)
        {
            evoke_prefix_cow_object_free(&object);
            return EVOKE_ERR_RANGE;
        }
        child->ref = children[index].ref;
        child->max_key_len = children[index].max_key_len;
        status = evoke_prefix_cow_copy_bytes(
            children[index].max_key,
            children[index].max_key_len,
            &child->max_key
        );
        if (status != EVOKE_OK)
        {
            evoke_prefix_cow_object_free(&object);
            return status;
        }
        term_count += child->ref.term_count;
    }
    object.ref.term_count = (uint32_t) term_count;
    status = evoke_prefix_cow_tree_add_object(
        tree,
        owner_manifest_id,
        &object,
        stats,
        &ref
    );
    if (status != EVOKE_OK)
    {
        evoke_prefix_cow_object_free(&object);
        return status;
    }
    return evoke_prefix_cow_span_from_object(
        &tree->objects[ref.object_id - 1],
        span_out
    );
}

static evoke_status
evoke_prefix_cow_build_parent_level(
    evoke_prefix_cow_tree *tree,
    const evoke_prefix_cow_span *children,
    size_t child_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_update_stats *stats,
    evoke_prefix_cow_span **parents_out,
    size_t *parent_count_out
)
{
    evoke_prefix_cow_span *parents;
    size_t parent_count;
    size_t child_index = 0;
    evoke_status status;

    if (tree == NULL || children == NULL || child_count < 2 ||
        parents_out == NULL || parent_count_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    parent_count = (child_count + EVOKE_PREFIX_COW_NODE_MAX_CHILDREN - 1) /
        EVOKE_PREFIX_COW_NODE_MAX_CHILDREN;
    parents = calloc(parent_count, sizeof(*parents));
    if (parents == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (size_t parent_index = 0;
         parent_index < parent_count;
         parent_index++)
    {
        size_t remaining = child_count - child_index;
        size_t remaining_parents = parent_count - parent_index;
        size_t group_count = remaining / remaining_parents;

        if (group_count > EVOKE_PREFIX_COW_NODE_MAX_CHILDREN)
        {
            group_count = EVOKE_PREFIX_COW_NODE_MAX_CHILDREN;
        }
        if (group_count < 2)
        {
            evoke_prefix_cow_spans_free(parents, parent_count);
            return EVOKE_ERR_FORMAT;
        }
        status = evoke_prefix_cow_make_node(
            tree,
            children + child_index,
            (uint32_t) group_count,
            owner_manifest_id,
            stats,
            &parents[parent_index]
        );
        if (status != EVOKE_OK)
        {
            evoke_prefix_cow_spans_free(parents, parent_count);
            return status;
        }
        child_index += group_count;
    }
    if (child_index != child_count)
    {
        evoke_prefix_cow_spans_free(parents, parent_count);
        return EVOKE_ERR_FORMAT;
    }
    *parents_out = parents;
    *parent_count_out = parent_count;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_finish_root(
    evoke_prefix_cow_tree *tree,
    evoke_prefix_cow_span *spans,
    size_t span_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_update_stats *stats
)
{
    evoke_status status = EVOKE_OK;

    if (tree == NULL || spans == NULL || span_count == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    while (span_count > 1)
    {
        evoke_prefix_cow_span *parents = NULL;
        size_t parent_count = 0;

        status = evoke_prefix_cow_build_parent_level(
            tree,
            spans,
            span_count,
            owner_manifest_id,
            stats,
            &parents,
            &parent_count
        );
        evoke_prefix_cow_spans_free(spans, span_count);
        if (status != EVOKE_OK)
        {
            return status;
        }
        spans = parents;
        span_count = parent_count;
    }
    tree->root = spans[0].ref;
    evoke_prefix_cow_spans_free(spans, 1);
    return EVOKE_OK;
}

evoke_status
evoke_prefix_cow_tree_build(
    const evoke_prefix_cow_key *keys,
    uint32_t key_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_tree *tree_out
)
{
    evoke_prefix_cow_tree tree;
    evoke_prefix_cow_key *sorted = NULL;
    evoke_prefix_cow_span *leaves = NULL;
    size_t leaf_count;
    evoke_status status;

    if (keys == NULL || key_count == 0 || owner_manifest_id == 0 ||
        tree_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_prefix_cow_tree_init(&tree);
    status = evoke_prefix_cow_copy_and_sort_keys(
        keys,
        key_count,
        0,
        &sorted
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    leaf_count = (key_count + EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES - 1) /
        EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES;
    leaves = calloc(leaf_count, sizeof(*leaves));
    if (leaves == NULL)
    {
        free(sorted);
        return EVOKE_ERR_NOMEM;
    }
    for (size_t leaf_index = 0; leaf_index < leaf_count; leaf_index++)
    {
        size_t first = leaf_index * EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES;
        uint32_t count = (uint32_t) (
            key_count - first > EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES
                ? EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES
                : key_count - first
        );

        status = evoke_prefix_cow_make_leaf_from_keys(
            &tree,
            sorted + first,
            count,
            owner_manifest_id,
            NULL,
            &leaves[leaf_index]
        );
        if (status != EVOKE_OK)
        {
            evoke_prefix_cow_spans_free(leaves, leaf_count);
            free(sorted);
            evoke_prefix_cow_tree_free(&tree);
            return status;
        }
    }
    free(sorted);
    tree.term_count = key_count;
    status = evoke_prefix_cow_finish_root(
        &tree,
        leaves,
        leaf_count,
        owner_manifest_id,
        NULL
    );
    if (status != EVOKE_OK)
    {
        evoke_prefix_cow_tree_free(&tree);
        return status;
    }
    evoke_prefix_cow_tree_free(tree_out);
    *tree_out = tree;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_tree_add_retired_ref(
    evoke_prefix_cow_tree *tree,
    const evoke_prefix_cow_ref *ref
)
{
    evoke_block_range *ranges;
    size_t capacity;

    if (tree == NULL || !evoke_prefix_cow_ref_is_valid(ref, false))
    {
        return EVOKE_ERR_INVALID;
    }
    if (tree->retired_range_count == tree->retired_range_capacity)
    {
        capacity = tree->retired_range_capacity == 0
            ? 16
            : tree->retired_range_capacity * 2;
        if (capacity < tree->retired_range_capacity ||
            capacity > SIZE_MAX / sizeof(*ranges))
        {
            return EVOKE_ERR_RANGE;
        }
        ranges = realloc(
            tree->retired_ranges,
            capacity * sizeof(*ranges)
        );
        if (ranges == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        tree->retired_ranges = ranges;
        tree->retired_range_capacity = capacity;
    }
    tree->retired_ranges[tree->retired_range_count].start_block =
        ref->start_block;
    tree->retired_ranges[tree->retired_range_count].block_count =
        ref->page_count;
    tree->retired_range_count++;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_spans_append_move(
    evoke_prefix_cow_span **spans,
    size_t *span_count,
    size_t *span_capacity,
    evoke_prefix_cow_span *source,
    size_t source_count
)
{
    evoke_prefix_cow_span *resized;
    size_t required;
    size_t capacity;

    if (spans == NULL || span_count == NULL || span_capacity == NULL ||
        (source_count > 0 && source == NULL) ||
        source_count > SIZE_MAX - *span_count)
    {
        return EVOKE_ERR_INVALID;
    }
    required = *span_count + source_count;
    if (required > *span_capacity)
    {
        capacity = *span_capacity == 0 ? 16 : *span_capacity;
        while (capacity < required)
        {
            if (capacity > SIZE_MAX / 2)
            {
                return EVOKE_ERR_RANGE;
            }
            capacity *= 2;
        }
        if (capacity > SIZE_MAX / sizeof(*resized))
        {
            return EVOKE_ERR_RANGE;
        }
        resized = realloc(*spans, capacity * sizeof(*resized));
        if (resized == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        memset(
            resized + *span_capacity,
            0,
            (capacity - *span_capacity) * sizeof(*resized)
        );
        *spans = resized;
        *span_capacity = capacity;
    }
    memcpy(
        *spans + *span_count,
        source,
        source_count * sizeof(*source)
    );
    memset(source, 0, source_count * sizeof(*source));
    *span_count = required;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_make_nodes_for_children(
    evoke_prefix_cow_tree *tree,
    evoke_prefix_cow_span *children,
    size_t child_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_update_stats *stats,
    evoke_prefix_cow_span **spans_out,
    size_t *span_count_out
)
{
    evoke_prefix_cow_span *spans;
    evoke_status status;

    if (tree == NULL || children == NULL || child_count == 0 ||
        spans_out == NULL || span_count_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *spans_out = NULL;
    *span_count_out = 0;
    if (child_count == 1)
    {
        spans = calloc(1, sizeof(*spans));
        if (spans == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        spans[0] = children[0];
        memset(&children[0], 0, sizeof(children[0]));
        *spans_out = spans;
        *span_count_out = 1;
        return EVOKE_OK;
    }
    if (child_count <= EVOKE_PREFIX_COW_NODE_MAX_CHILDREN)
    {
        spans = calloc(1, sizeof(*spans));
        if (spans == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        status = evoke_prefix_cow_make_node(
            tree,
            children,
            (uint32_t) child_count,
            owner_manifest_id,
            stats,
            spans
        );
        if (status != EVOKE_OK)
        {
            free(spans);
            return status;
        }
        *spans_out = spans;
        *span_count_out = 1;
        return EVOKE_OK;
    }
    return evoke_prefix_cow_build_parent_level(
        tree,
        children,
        child_count,
        owner_manifest_id,
        stats,
        spans_out,
        span_count_out
    );
}

static evoke_status
evoke_prefix_cow_patch_leaf(
    evoke_prefix_cow_tree *tree,
    const evoke_prefix_cow_leaf *leaf,
    const evoke_prefix_cow_key *keys,
    size_t key_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_update_stats *stats,
    evoke_prefix_cow_span **spans_out,
    size_t *span_count_out
)
{
    evoke_prefix_cow_entry *merged;
    evoke_prefix_cow_span *spans;
    size_t merged_count;
    size_t old_index = 0;
    size_t key_index = 0;
    size_t output_index = 0;
    size_t span_count;
    evoke_status status = EVOKE_OK;

    if (tree == NULL || leaf == NULL || keys == NULL || key_count == 0 ||
        spans_out == NULL || span_count_out == NULL ||
        key_count > SIZE_MAX - leaf->entry_count)
    {
        return EVOKE_ERR_INVALID;
    }
    merged_count = leaf->entry_count + key_count;
    merged = calloc(merged_count, sizeof(*merged));
    if (merged == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    while (old_index < leaf->entry_count || key_index < key_count)
    {
        int compared;

        if (old_index == leaf->entry_count)
        {
            compared = 1;
        }
        else if (key_index == key_count)
        {
            compared = -1;
        }
        else
        {
            compared = evoke_prefix_cow_compare_bytes(
                leaf->entries[old_index].bytes,
                leaf->entries[old_index].bytes_len,
                keys[key_index].bytes,
                keys[key_index].bytes_len
            );
        }
        if (compared == 0)
        {
            free(merged);
            return EVOKE_ERR_FORMAT;
        }
        if (compared < 0)
        {
            merged[output_index++] = leaf->entries[old_index++];
        }
        else
        {
            merged[output_index].bytes = (uint8_t *) keys[key_index].bytes;
            merged[output_index].bytes_len = keys[key_index].bytes_len;
            merged[output_index].term_id = keys[key_index].term_id;
            output_index++;
            key_index++;
        }
    }
    if (output_index != merged_count)
    {
        free(merged);
        return EVOKE_ERR_FORMAT;
    }
    span_count = (
        merged_count + EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES - 1
    ) / EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES;
    spans = calloc(span_count, sizeof(*spans));
    if (spans == NULL)
    {
        free(merged);
        return EVOKE_ERR_NOMEM;
    }
    for (size_t span_index = 0; span_index < span_count; span_index++)
    {
        size_t first = span_index * EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES;
        uint32_t count = (uint32_t) (
            merged_count - first > EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES
                ? EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES
                : merged_count - first
        );

        status = evoke_prefix_cow_make_leaf_from_entries(
            tree,
            merged + first,
            count,
            owner_manifest_id,
            stats,
            &spans[span_index]
        );
        if (status != EVOKE_OK)
        {
            evoke_prefix_cow_spans_free(spans, span_count);
            free(merged);
            return status;
        }
    }
    free(merged);
    *spans_out = spans;
    *span_count_out = span_count;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_patch_subtree(
    evoke_prefix_cow_tree *tree,
    const evoke_prefix_cow_ref *old_ref,
    const evoke_prefix_cow_key *keys,
    size_t key_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_object_loader loader,
    void *loader_context,
    uint32_t depth,
    evoke_prefix_cow_update_stats *stats,
    evoke_prefix_cow_span **spans_out,
    size_t *span_count_out
)
{
    evoke_prefix_cow_object object = {0};
    evoke_prefix_cow_span *combined = NULL;
    size_t combined_count = 0;
    size_t combined_capacity = 0;
    size_t key_index = 0;
    evoke_status status;

    if (tree == NULL || !evoke_prefix_cow_ref_is_valid(old_ref, false) ||
        keys == NULL || key_count == 0 || owner_manifest_id == 0 ||
        loader == NULL || spans_out == NULL || span_count_out == NULL ||
        depth >= EVOKE_PREFIX_COW_MAX_DEPTH)
    {
        return EVOKE_ERR_INVALID;
    }
    *spans_out = NULL;
    *span_count_out = 0;
    status = loader(loader_context, old_ref, &object);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_prefix_cow_object_shape_validate(&object, true);
    if (status != EVOKE_OK || !evoke_prefix_cow_refs_equal(
            &object.ref,
            old_ref))
    {
        evoke_prefix_cow_object_free(&object);
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    if (object.ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
    {
        if (stats != NULL)
        {
            stats->read_leaves++;
        }
        status = evoke_prefix_cow_patch_leaf(
            tree,
            &object.value.leaf,
            keys,
            key_count,
            owner_manifest_id,
            stats,
            spans_out,
            span_count_out
        );
    }
    else
    {
        const evoke_prefix_cow_node *node = &object.value.node;

        if (stats != NULL)
        {
            stats->read_nodes++;
        }
        for (uint32_t child_index = 0;
             status == EVOKE_OK && child_index < node->child_count;
             child_index++)
        {
            const evoke_prefix_cow_child *child =
                &node->children[child_index];
            size_t first_key = key_index;
            evoke_prefix_cow_span *child_spans = NULL;
            size_t child_span_count = 0;

            if (child_index + 1 == node->child_count)
            {
                key_index = key_count;
            }
            else
            {
                while (key_index < key_count &&
                       evoke_prefix_cow_compare_bytes(
                           keys[key_index].bytes,
                           keys[key_index].bytes_len,
                           child->max_key,
                           child->max_key_len
                       ) <= 0)
                {
                    key_index++;
                }
            }
            if (key_index == first_key)
            {
                child_spans = calloc(1, sizeof(*child_spans));
                if (child_spans == NULL)
                {
                    status = EVOKE_ERR_NOMEM;
                    break;
                }
                status = evoke_prefix_cow_span_copy(
                    &child->ref,
                    child->max_key,
                    child->max_key_len,
                    child_spans
                );
                child_span_count = status == EVOKE_OK ? 1 : 0;
            }
            else
            {
                status = evoke_prefix_cow_patch_subtree(
                    tree,
                    &child->ref,
                    keys + first_key,
                    key_index - first_key,
                    owner_manifest_id,
                    loader,
                    loader_context,
                    depth + 1,
                    stats,
                    &child_spans,
                    &child_span_count
                );
            }
            if (status == EVOKE_OK)
            {
                status = evoke_prefix_cow_spans_append_move(
                    &combined,
                    &combined_count,
                    &combined_capacity,
                    child_spans,
                    child_span_count
                );
            }
            evoke_prefix_cow_spans_free(child_spans, child_span_count);
        }
        if (status == EVOKE_OK && key_index != key_count)
        {
            status = EVOKE_ERR_FORMAT;
        }
        if (status == EVOKE_OK)
        {
            status = evoke_prefix_cow_make_nodes_for_children(
                tree,
                combined,
                combined_count,
                owner_manifest_id,
                stats,
                spans_out,
                span_count_out
            );
        }
        evoke_prefix_cow_spans_free(combined, combined_count);
    }
    if (status == EVOKE_OK)
    {
        status = evoke_prefix_cow_tree_add_retired_ref(tree, old_ref);
    }
    evoke_prefix_cow_object_free(&object);
    return status;
}

evoke_status
evoke_prefix_cow_build_external_append_patch(
    const evoke_prefix_cow_ref *old_root,
    uint32_t old_term_count,
    const evoke_prefix_cow_key *keys,
    uint32_t key_count,
    uint64_t owner_manifest_id,
    evoke_prefix_cow_object_loader loader,
    void *loader_context,
    evoke_prefix_cow_tree *patch_out,
    evoke_prefix_cow_update_stats *stats_out
)
{
    evoke_prefix_cow_tree patch;
    evoke_prefix_cow_update_stats stats = {0};
    evoke_prefix_cow_key *sorted = NULL;
    evoke_prefix_cow_span *spans = NULL;
    size_t span_count = 0;
    evoke_status status;

    if (!evoke_prefix_cow_ref_is_valid(old_root, false) ||
        old_term_count == 0 || old_root->term_count != old_term_count ||
        keys == NULL || key_count == 0 ||
        (uint64_t) old_term_count + key_count > UINT32_MAX ||
        owner_manifest_id <= old_root->owner_manifest_id ||
        loader == NULL || patch_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_prefix_cow_tree_init(&patch);
    status = evoke_prefix_cow_copy_and_sort_keys(
        keys,
        key_count,
        old_term_count,
        &sorted
    );
    if (status == EVOKE_OK)
    {
        status = evoke_prefix_cow_patch_subtree(
            &patch,
            old_root,
            sorted,
            key_count,
            owner_manifest_id,
            loader,
            loader_context,
            0,
            &stats,
            &spans,
            &span_count
        );
    }
    free(sorted);
    if (status == EVOKE_OK)
    {
        patch.term_count = old_term_count + key_count;
        status = evoke_prefix_cow_finish_root(
            &patch,
            spans,
            span_count,
            owner_manifest_id,
            &stats
        );
        spans = NULL;
        span_count = 0;
    }
    evoke_prefix_cow_spans_free(spans, span_count);
    if (status != EVOKE_OK || patch.root.term_count != patch.term_count)
    {
        evoke_prefix_cow_tree_free(&patch);
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    stats.changed_terms = key_count;
    evoke_prefix_cow_tree_free(patch_out);
    *patch_out = patch;
    if (stats_out != NULL)
    {
        *stats_out = stats;
    }
    return EVOKE_OK;
}

static void
evoke_prefix_cow_serialize_ref(
    uint8_t *bytes,
    const evoke_prefix_cow_ref *ref
)
{
    evoke_prefix_cow_write_u32(bytes + 0, ref->kind);
    evoke_prefix_cow_write_u32(bytes + 4, ref->start_block);
    evoke_prefix_cow_write_u32(bytes + 8, ref->page_count);
    evoke_prefix_cow_write_u32(bytes + 12, ref->reserved);
    evoke_prefix_cow_write_u64(bytes + 16, ref->object_id);
    evoke_prefix_cow_write_u64(bytes + 24, ref->owner_manifest_id);
    evoke_prefix_cow_write_u64(bytes + 32, ref->object_bytes);
    evoke_prefix_cow_write_u64(bytes + 40, ref->checksum);
    evoke_prefix_cow_write_u64(bytes + 48, ref->blob_checksum);
    evoke_prefix_cow_write_u32(bytes + 56, ref->term_count);
    evoke_prefix_cow_write_u32(bytes + 60, ref->reserved2);
}

static void
evoke_prefix_cow_deserialize_ref(
    const uint8_t *bytes,
    evoke_prefix_cow_ref *ref
)
{
    memset(ref, 0, sizeof(*ref));
    ref->kind = (evoke_prefix_cow_object_kind)
        evoke_prefix_cow_read_u32(bytes + 0);
    ref->start_block = evoke_prefix_cow_read_u32(bytes + 4);
    ref->page_count = evoke_prefix_cow_read_u32(bytes + 8);
    ref->reserved = evoke_prefix_cow_read_u32(bytes + 12);
    ref->object_id = evoke_prefix_cow_read_u64(bytes + 16);
    ref->owner_manifest_id = evoke_prefix_cow_read_u64(bytes + 24);
    ref->object_bytes = evoke_prefix_cow_read_u64(bytes + 32);
    ref->checksum = evoke_prefix_cow_read_u64(bytes + 40);
    ref->blob_checksum = evoke_prefix_cow_read_u64(bytes + 48);
    ref->term_count = evoke_prefix_cow_read_u32(bytes + 56);
    ref->reserved2 = evoke_prefix_cow_read_u32(bytes + 60);
}

static evoke_status
evoke_prefix_cow_object_encode(
    const evoke_prefix_cow_object *object,
    bool require_bound_children,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes;
    size_t size;
    size_t key_offset;
    uint32_t item_count;
    uint32_t descriptor_size;
    uint64_t blob_checksum;
    evoke_status status;

    if (object == NULL || bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_prefix_cow_object_shape_validate(
        object,
        require_bound_children
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_prefix_cow_object_serialized_size(object, &size);
    if (status != EVOKE_OK)
    {
        return status;
    }
    bytes = calloc(size, 1);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    item_count = object->ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF
        ? object->value.leaf.entry_count
        : object->value.node.child_count;
    descriptor_size = object->ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF
        ? EVOKE_PREFIX_COW_LEAF_ENTRY_SIZE
        : EVOKE_PREFIX_COW_NODE_CHILD_SIZE;
    key_offset = EVOKE_PREFIX_COW_HEADER_SIZE +
        (size_t) item_count * descriptor_size;

    evoke_prefix_cow_write_u32(bytes + 0, EVOKE_PREFIX_COW_MAGIC);
    evoke_prefix_cow_write_u16(bytes + 4, EVOKE_PREFIX_COW_VERSION);
    evoke_prefix_cow_write_u16(bytes + 6, EVOKE_PREFIX_COW_HEADER_SIZE);
    evoke_prefix_cow_write_u16(bytes + 8, object->ref.kind);
    evoke_prefix_cow_write_u32(bytes + 12, item_count);
    evoke_prefix_cow_write_u32(bytes + 16, object->ref.term_count);
    evoke_prefix_cow_write_u64(bytes + 24, object->ref.object_id);
    evoke_prefix_cow_write_u64(
        bytes + 32,
        object->ref.owner_manifest_id
    );
    evoke_prefix_cow_write_u64(bytes + 40, size);
    evoke_prefix_cow_write_u64(bytes + 48, object->ref.checksum);
    evoke_prefix_cow_write_u64(bytes + 64, EVOKE_PREFIX_COW_HEADER_SIZE);
    evoke_prefix_cow_write_u64(bytes + 72, size);

    if (object->ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
    {
        for (uint32_t index = 0; index < item_count; index++)
        {
            const evoke_prefix_cow_entry *entry =
                &object->value.leaf.entries[index];
            uint8_t *descriptor = bytes + EVOKE_PREFIX_COW_HEADER_SIZE +
                (size_t) index * EVOKE_PREFIX_COW_LEAF_ENTRY_SIZE;

            evoke_prefix_cow_write_u32(descriptor + 0, entry->term_id);
            evoke_prefix_cow_write_u32(descriptor + 4, entry->bytes_len);
            evoke_prefix_cow_write_u64(descriptor + 8, key_offset);
            memcpy(bytes + key_offset, entry->bytes, entry->bytes_len);
            key_offset += entry->bytes_len;
        }
    }
    else
    {
        for (uint32_t index = 0; index < item_count; index++)
        {
            const evoke_prefix_cow_child *child =
                &object->value.node.children[index];
            uint8_t *descriptor = bytes + EVOKE_PREFIX_COW_HEADER_SIZE +
                (size_t) index * EVOKE_PREFIX_COW_NODE_CHILD_SIZE;

            evoke_prefix_cow_serialize_ref(descriptor, &child->ref);
            evoke_prefix_cow_write_u32(
                descriptor + 64,
                child->max_key_len
            );
            evoke_prefix_cow_write_u64(descriptor + 72, key_offset);
            memcpy(bytes + key_offset, child->max_key, child->max_key_len);
            key_offset += child->max_key_len;
        }
    }
    if (key_offset != size)
    {
        free(bytes);
        return EVOKE_ERR_FORMAT;
    }
    blob_checksum = evoke_prefix_cow_checksum_bytes(
        bytes,
        size,
        EVOKE_PREFIX_COW_BLOB_CHECKSUM_OFFSET,
        sizeof(uint64_t)
    );
    evoke_prefix_cow_write_u64(
        bytes + EVOKE_PREFIX_COW_BLOB_CHECKSUM_OFFSET,
        blob_checksum
    );
    *bytes_out = bytes;
    *size_out = size;
    return EVOKE_OK;
}

evoke_status
evoke_prefix_cow_object_serialize(
    const evoke_prefix_cow_object *object,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    return evoke_prefix_cow_object_encode(
        object,
        true,
        bytes_out,
        size_out
    );
}

evoke_status
evoke_prefix_cow_object_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_prefix_cow_object *object_out
)
{
    evoke_prefix_cow_object object = {0};
    uint32_t item_count;
    uint32_t descriptor_size;
    uint64_t items_offset;
    uint64_t total_size;
    size_t minimum_key_offset;
    size_t previous_key_end;
    evoke_status status;

    if (bytes == NULL || object_out == NULL ||
        size < EVOKE_PREFIX_COW_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_prefix_cow_read_u32(bytes + 0) != EVOKE_PREFIX_COW_MAGIC ||
        evoke_prefix_cow_read_u16(bytes + 4) !=
            EVOKE_PREFIX_COW_VERSION ||
        evoke_prefix_cow_read_u16(bytes + 6) !=
            EVOKE_PREFIX_COW_HEADER_SIZE ||
        evoke_prefix_cow_read_u16(bytes + 10) != 0 ||
        evoke_prefix_cow_read_u32(bytes + 20) != 0 ||
        evoke_prefix_cow_read_u64(bytes + 72) != size ||
        evoke_prefix_cow_read_u64(
            bytes + EVOKE_PREFIX_COW_BLOB_CHECKSUM_OFFSET
        ) != evoke_prefix_cow_checksum_bytes(
            bytes,
            size,
            EVOKE_PREFIX_COW_BLOB_CHECKSUM_OFFSET,
            sizeof(uint64_t)
        ))
    {
        return EVOKE_ERR_FORMAT;
    }
    object.ref.kind = (evoke_prefix_cow_object_kind)
        evoke_prefix_cow_read_u16(bytes + 8);
    item_count = evoke_prefix_cow_read_u32(bytes + 12);
    object.ref.term_count = evoke_prefix_cow_read_u32(bytes + 16);
    object.ref.object_id = evoke_prefix_cow_read_u64(bytes + 24);
    object.ref.owner_manifest_id = evoke_prefix_cow_read_u64(bytes + 32);
    object.ref.object_bytes = evoke_prefix_cow_read_u64(bytes + 40);
    object.ref.checksum = evoke_prefix_cow_read_u64(bytes + 48);
    object.ref.blob_checksum = evoke_segment_blob_checksum(bytes, size);
    items_offset = evoke_prefix_cow_read_u64(bytes + 64);
    total_size = evoke_prefix_cow_read_u64(bytes + 72);
    if (items_offset != EVOKE_PREFIX_COW_HEADER_SIZE ||
        total_size != size || object.ref.object_bytes != size ||
        (object.ref.kind != EVOKE_PREFIX_COW_OBJECT_LEAF &&
         object.ref.kind != EVOKE_PREFIX_COW_OBJECT_NODE))
    {
        return EVOKE_ERR_FORMAT;
    }
    descriptor_size = object.ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF
        ? EVOKE_PREFIX_COW_LEAF_ENTRY_SIZE
        : EVOKE_PREFIX_COW_NODE_CHILD_SIZE;
    if (item_count >
        (SIZE_MAX - EVOKE_PREFIX_COW_HEADER_SIZE) / descriptor_size)
    {
        return EVOKE_ERR_RANGE;
    }
    minimum_key_offset = EVOKE_PREFIX_COW_HEADER_SIZE +
        (size_t) item_count * descriptor_size;
    if (minimum_key_offset > size)
    {
        return EVOKE_ERR_FORMAT;
    }
    previous_key_end = minimum_key_offset;

    if (object.ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
    {
        object.value.leaf.entry_count = item_count;
        object.value.leaf.entries = calloc(
            item_count,
            sizeof(*object.value.leaf.entries)
        );
        if (object.value.leaf.entries == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        for (uint32_t index = 0; index < item_count; index++)
        {
            const uint8_t *descriptor =
                bytes + EVOKE_PREFIX_COW_HEADER_SIZE +
                (size_t) index * EVOKE_PREFIX_COW_LEAF_ENTRY_SIZE;
            evoke_prefix_cow_entry *entry =
                &object.value.leaf.entries[index];
            uint64_t key_offset =
                evoke_prefix_cow_read_u64(descriptor + 8);

            entry->term_id = evoke_prefix_cow_read_u32(descriptor + 0);
            entry->bytes_len = evoke_prefix_cow_read_u32(descriptor + 4);
            if (entry->bytes_len == 0 || key_offset != previous_key_end ||
                key_offset > size || entry->bytes_len > size - key_offset)
            {
                evoke_prefix_cow_object_free(&object);
                return EVOKE_ERR_FORMAT;
            }
            status = evoke_prefix_cow_copy_bytes(
                bytes + key_offset,
                entry->bytes_len,
                &entry->bytes
            );
            if (status != EVOKE_OK)
            {
                evoke_prefix_cow_object_free(&object);
                return status;
            }
            previous_key_end = (size_t) key_offset + entry->bytes_len;
        }
    }
    else
    {
        object.value.node.child_count = item_count;
        object.value.node.children = calloc(
            item_count,
            sizeof(*object.value.node.children)
        );
        if (object.value.node.children == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        for (uint32_t index = 0; index < item_count; index++)
        {
            const uint8_t *descriptor =
                bytes + EVOKE_PREFIX_COW_HEADER_SIZE +
                (size_t) index * EVOKE_PREFIX_COW_NODE_CHILD_SIZE;
            evoke_prefix_cow_child *child =
                &object.value.node.children[index];
            uint64_t key_offset;

            evoke_prefix_cow_deserialize_ref(descriptor, &child->ref);
            child->max_key_len = evoke_prefix_cow_read_u32(
                descriptor + 64
            );
            if (evoke_prefix_cow_read_u32(descriptor + 68) != 0)
            {
                evoke_prefix_cow_object_free(&object);
                return EVOKE_ERR_FORMAT;
            }
            key_offset = evoke_prefix_cow_read_u64(descriptor + 72);
            if (child->max_key_len == 0 || key_offset != previous_key_end ||
                key_offset > size ||
                child->max_key_len > size - key_offset)
            {
                evoke_prefix_cow_object_free(&object);
                return EVOKE_ERR_FORMAT;
            }
            status = evoke_prefix_cow_copy_bytes(
                bytes + key_offset,
                child->max_key_len,
                &child->max_key
            );
            if (status != EVOKE_OK)
            {
                evoke_prefix_cow_object_free(&object);
                return status;
            }
            previous_key_end = (size_t) key_offset + child->max_key_len;
        }
    }
    if (previous_key_end != size)
    {
        evoke_prefix_cow_object_free(&object);
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_prefix_cow_object_shape_validate(&object, true);
    if (status != EVOKE_OK)
    {
        evoke_prefix_cow_object_free(&object);
        return status;
    }
    evoke_prefix_cow_object_free(object_out);
    *object_out = object;
    return EVOKE_OK;
}

evoke_status
evoke_prefix_cow_tree_prepare_object_for_storage(
    evoke_prefix_cow_tree *tree,
    uint64_t object_id,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    evoke_prefix_cow_object *object;
    evoke_prefix_cow_object prepared;
    evoke_status status;

    if (tree == NULL || bytes_out == NULL || size_out == NULL ||
        object_id == 0 || object_id > tree->object_count)
    {
        return EVOKE_ERR_INVALID;
    }
    object = &tree->objects[object_id - 1];
    if (object->ref.object_id != object_id ||
        !evoke_prefix_cow_ref_is_unbound(&object->ref))
    {
        return EVOKE_ERR_FORMAT;
    }
    prepared = *object;
    if (prepared.ref.kind == EVOKE_PREFIX_COW_OBJECT_NODE)
    {
        for (uint32_t index = 0;
             index < prepared.value.node.child_count;
             index++)
        {
            evoke_prefix_cow_child *child =
                &prepared.value.node.children[index];
            const evoke_prefix_cow_object *stored_child;

            if (!evoke_prefix_cow_ref_is_unbound(&child->ref))
            {
                if (!evoke_prefix_cow_ref_is_valid(&child->ref, false))
                {
                    return EVOKE_ERR_FORMAT;
                }
                continue;
            }
            if (child->ref.owner_manifest_id !=
                    object->ref.owner_manifest_id ||
                child->ref.object_id == 0 ||
                child->ref.object_id >= object_id ||
                child->ref.object_id > tree->object_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            stored_child = &tree->objects[child->ref.object_id - 1];
            if (stored_child->ref.kind != child->ref.kind ||
                stored_child->ref.object_id != child->ref.object_id ||
                !evoke_prefix_cow_ref_is_valid(
                    &stored_child->ref,
                    false))
            {
                return EVOKE_ERR_FORMAT;
            }
            child->ref = stored_child->ref;
        }
    }
    status = evoke_prefix_cow_object_refresh(&prepared);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_prefix_cow_object_encode(
        &prepared,
        true,
        bytes_out,
        size_out
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    prepared.ref.blob_checksum = evoke_segment_blob_checksum(
        *bytes_out,
        *size_out
    );
    *object = prepared;
    if (tree->root.owner_manifest_id == object->ref.owner_manifest_id &&
        tree->root.object_id == object_id)
    {
        tree->root = object->ref;
    }
    return EVOKE_OK;
}

evoke_status
evoke_prefix_cow_object_bind_storage(
    evoke_prefix_cow_object *object,
    const evoke_segment_object_ref *storage_ref
)
{
    evoke_prefix_cow_object bound;

    if (object == NULL || storage_ref == NULL ||
        !evoke_prefix_cow_ref_is_unbound(&object->ref) ||
        object->ref.blob_checksum == 0 ||
        storage_ref->object_kind != EVOKE_SEGMENT_OBJECT_PREFIX_LOOKUP ||
        storage_ref->object_id != object->ref.object_id ||
        storage_ref->owner_manifest_id != object->ref.owner_manifest_id ||
        storage_ref->object_bytes != object->ref.object_bytes ||
        storage_ref->object_checksum != object->ref.blob_checksum ||
        evoke_segment_object_ref_validate(
            storage_ref,
            UINT32_MAX
        ) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    bound = *object;
    bound.ref.start_block = storage_ref->start_block;
    bound.ref.page_count = storage_ref->page_count;
    if (!evoke_prefix_cow_ref_is_valid(&bound.ref, false) ||
        evoke_prefix_cow_object_shape_validate(&bound, false) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    *object = bound;
    return EVOKE_OK;
}

evoke_status
evoke_prefix_cow_tree_bind_object_storage(
    evoke_prefix_cow_tree *tree,
    uint64_t object_id,
    const evoke_segment_object_ref *storage_ref
)
{
    evoke_prefix_cow_object *object;
    evoke_status status;

    if (tree == NULL || object_id == 0 ||
        object_id > tree->object_count)
    {
        return EVOKE_ERR_INVALID;
    }
    object = &tree->objects[object_id - 1];
    status = evoke_prefix_cow_object_bind_storage(object, storage_ref);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (tree->root.owner_manifest_id == object->ref.owner_manifest_id &&
        tree->root.object_id == object_id)
    {
        tree->root = object->ref;
    }
    return EVOKE_OK;
}

evoke_status
evoke_prefix_cow_ref_as_segment_object_ref(
    const evoke_prefix_cow_ref *ref,
    evoke_segment_object_ref *storage_ref_out
)
{
    evoke_segment_object_ref storage_ref;

    if (!evoke_prefix_cow_ref_is_valid(ref, false) ||
        storage_ref_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&storage_ref, 0, sizeof(storage_ref));
    storage_ref.object_kind = EVOKE_SEGMENT_OBJECT_PREFIX_LOOKUP;
    storage_ref.start_block = ref->start_block;
    storage_ref.page_count = ref->page_count;
    storage_ref.object_id = ref->object_id;
    storage_ref.owner_manifest_id = ref->owner_manifest_id;
    storage_ref.object_bytes = ref->object_bytes;
    storage_ref.object_checksum = ref->blob_checksum;
    if (evoke_segment_object_ref_validate(
            &storage_ref,
            UINT32_MAX
        ) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    *storage_ref_out = storage_ref;
    return EVOKE_OK;
}

static size_t
evoke_prefix_cow_leaf_lower_bound(
    const evoke_prefix_cow_leaf *leaf,
    const uint8_t *key,
    uint32_t key_len
)
{
    size_t lower = 0;
    size_t upper = leaf->entry_count;

    while (lower < upper)
    {
        size_t middle = lower + (upper - lower) / 2;
        const evoke_prefix_cow_entry *entry = &leaf->entries[middle];

        if (evoke_prefix_cow_compare_bytes(
                entry->bytes,
                entry->bytes_len,
                key,
                key_len
            ) < 0)
        {
            lower = middle + 1;
        }
        else
        {
            upper = middle;
        }
    }
    return lower;
}

static size_t
evoke_prefix_cow_node_lower_bound(
    const evoke_prefix_cow_node *node,
    const uint8_t *key,
    uint32_t key_len
)
{
    size_t lower = 0;
    size_t upper = node->child_count;

    while (lower < upper)
    {
        size_t middle = lower + (upper - lower) / 2;
        const evoke_prefix_cow_child *child = &node->children[middle];

        if (evoke_prefix_cow_compare_bytes(
                child->max_key,
                child->max_key_len,
                key,
                key_len
            ) < 0)
        {
            lower = middle + 1;
        }
        else
        {
            upper = middle;
        }
    }
    return lower;
}

static evoke_status
evoke_prefix_cow_scan_prefix_subtree(
    const evoke_prefix_cow_ref *ref,
    const uint8_t *prefix,
    uint32_t prefix_len,
    const uint8_t *upper_key,
    uint32_t upper_key_len,
    bool has_upper_key,
    evoke_prefix_cow_object_loader loader,
    void *loader_context,
    evoke_prefix_cow_entry_visitor visitor,
    void *visitor_context,
    uint32_t depth
)
{
    evoke_prefix_cow_object object = {0};
    evoke_status status;

    if (depth >= EVOKE_PREFIX_COW_MAX_DEPTH)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = loader(loader_context, ref, &object);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_prefix_cow_object_shape_validate(&object, true);
    if (status != EVOKE_OK || !evoke_prefix_cow_refs_equal(
            &object.ref,
            ref))
    {
        evoke_prefix_cow_object_free(&object);
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    if (object.ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
    {
        size_t index = evoke_prefix_cow_leaf_lower_bound(
            &object.value.leaf,
            prefix,
            prefix_len
        );

        for (; index < object.value.leaf.entry_count; index++)
        {
            const evoke_prefix_cow_entry *entry =
                &object.value.leaf.entries[index];

            if (entry->bytes_len < prefix_len ||
                memcmp(entry->bytes, prefix, prefix_len) != 0)
            {
                break;
            }
            status = visitor(visitor_context, entry);
            if (status != EVOKE_OK)
            {
                break;
            }
        }
    }
    else
    {
        const evoke_prefix_cow_node *node = &object.value.node;
        size_t first = evoke_prefix_cow_node_lower_bound(
            node,
            prefix,
            prefix_len
        );

        for (size_t index = first;
             status == EVOKE_OK && index < node->child_count;
             index++)
        {
            const evoke_prefix_cow_child *child = &node->children[index];

            status = evoke_prefix_cow_scan_prefix_subtree(
                &child->ref,
                prefix,
                prefix_len,
                upper_key,
                upper_key_len,
                has_upper_key,
                loader,
                loader_context,
                visitor,
                visitor_context,
                depth + 1
            );
            if (has_upper_key &&
                evoke_prefix_cow_compare_bytes(
                    child->max_key,
                    child->max_key_len,
                    upper_key,
                    upper_key_len
                ) >= 0)
            {
                break;
            }
        }
    }
    evoke_prefix_cow_object_free(&object);
    return status;
}

evoke_status
evoke_prefix_cow_scan_prefix_external(
    const evoke_prefix_cow_ref *root,
    uint32_t expected_term_count,
    const uint8_t *prefix,
    size_t prefix_len,
    evoke_prefix_cow_object_loader loader,
    void *loader_context,
    evoke_prefix_cow_entry_visitor visitor,
    void *visitor_context
)
{
    uint8_t *upper_key;
    uint32_t upper_key_len = 0;
    bool has_upper_key = false;
    evoke_status status;

    if (!evoke_prefix_cow_ref_is_valid(root, false) ||
        root->term_count != expected_term_count ||
        expected_term_count == 0 || prefix == NULL || prefix_len == 0 ||
        prefix_len > UINT32_MAX || loader == NULL || visitor == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    upper_key = malloc(prefix_len);
    if (upper_key == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    memcpy(upper_key, prefix, prefix_len);
    for (size_t index = prefix_len; index > 0; index--)
    {
        if (upper_key[index - 1] != UINT8_MAX)
        {
            upper_key[index - 1]++;
            upper_key_len = (uint32_t) index;
            has_upper_key = true;
            break;
        }
    }
    status = evoke_prefix_cow_scan_prefix_subtree(
        root,
        prefix,
        (uint32_t) prefix_len,
        upper_key,
        upper_key_len,
        has_upper_key,
        loader,
        loader_context,
        visitor,
        visitor_context,
        0
    );
    free(upper_key);
    return status;
}

typedef struct evoke_prefix_cow_validation_state
{
    bool *seen_term_ids;
    uint32_t expected_term_count;
    evoke_prefix_cow_ref *visited;
    size_t visited_count;
    size_t visited_capacity;
    evoke_prefix_cow_object_loader loader;
    void *loader_context;
} evoke_prefix_cow_validation_state;

static evoke_status
evoke_prefix_cow_validation_mark(
    evoke_prefix_cow_validation_state *state,
    const evoke_prefix_cow_ref *ref
)
{
    evoke_prefix_cow_ref *visited;
    size_t capacity;

    for (size_t index = 0; index < state->visited_count; index++)
    {
        if (evoke_prefix_cow_refs_equal(&state->visited[index], ref))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    if (state->visited_count == state->visited_capacity)
    {
        capacity = state->visited_capacity == 0
            ? 16
            : state->visited_capacity * 2;
        if (capacity < state->visited_capacity ||
            capacity > SIZE_MAX / sizeof(*visited))
        {
            return EVOKE_ERR_RANGE;
        }
        visited = realloc(
            state->visited,
            capacity * sizeof(*visited)
        );
        if (visited == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        state->visited = visited;
        state->visited_capacity = capacity;
    }
    state->visited[state->visited_count++] = *ref;
    return EVOKE_OK;
}

static evoke_status
evoke_prefix_cow_validate_subtree(
    evoke_prefix_cow_validation_state *state,
    const evoke_prefix_cow_ref *ref,
    uint32_t depth,
    uint8_t **min_key_out,
    uint32_t *min_key_len_out,
    uint8_t **max_key_out,
    uint32_t *max_key_len_out,
    uint32_t *term_count_out
)
{
    evoke_prefix_cow_object object = {0};
    uint8_t *min_key = NULL;
    uint8_t *max_key = NULL;
    uint8_t *previous_max = NULL;
    uint32_t min_key_len = 0;
    uint32_t max_key_len = 0;
    uint32_t previous_max_len = 0;
    uint64_t term_count = 0;
    evoke_status status;

    if (depth >= EVOKE_PREFIX_COW_MAX_DEPTH)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_prefix_cow_validation_mark(state, ref);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = state->loader(state->loader_context, ref, &object);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_prefix_cow_object_shape_validate(&object, true);
    if (status != EVOKE_OK || !evoke_prefix_cow_refs_equal(
            &object.ref,
            ref))
    {
        evoke_prefix_cow_object_free(&object);
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    if (object.ref.kind == EVOKE_PREFIX_COW_OBJECT_LEAF)
    {
        const evoke_prefix_cow_leaf *leaf = &object.value.leaf;

        for (uint32_t index = 0; index < leaf->entry_count; index++)
        {
            uint32_t term_id = leaf->entries[index].term_id;

            if (term_id >= state->expected_term_count ||
                state->seen_term_ids[term_id])
            {
                status = EVOKE_ERR_FORMAT;
                goto done;
            }
            state->seen_term_ids[term_id] = true;
        }
        status = evoke_prefix_cow_copy_bytes(
            leaf->entries[0].bytes,
            leaf->entries[0].bytes_len,
            &min_key
        );
        if (status == EVOKE_OK)
        {
            min_key_len = leaf->entries[0].bytes_len;
            status = evoke_prefix_cow_copy_bytes(
                leaf->entries[leaf->entry_count - 1].bytes,
                leaf->entries[leaf->entry_count - 1].bytes_len,
                &max_key
            );
            max_key_len = leaf->entries[leaf->entry_count - 1].bytes_len;
        }
        term_count = leaf->entry_count;
    }
    else
    {
        const evoke_prefix_cow_node *node = &object.value.node;

        for (uint32_t index = 0;
             status == EVOKE_OK && index < node->child_count;
             index++)
        {
            uint8_t *child_min = NULL;
            uint8_t *child_max = NULL;
            uint32_t child_min_len = 0;
            uint32_t child_max_len = 0;
            uint32_t child_terms = 0;

            status = evoke_prefix_cow_validate_subtree(
                state,
                &node->children[index].ref,
                depth + 1,
                &child_min,
                &child_min_len,
                &child_max,
                &child_max_len,
                &child_terms
            );
            if (status == EVOKE_OK &&
                (child_terms != node->children[index].ref.term_count ||
                 child_max_len != node->children[index].max_key_len ||
                 memcmp(
                     child_max,
                     node->children[index].max_key,
                     child_max_len
                 ) != 0 ||
                 (previous_max != NULL &&
                  evoke_prefix_cow_compare_bytes(
                      previous_max,
                      previous_max_len,
                      child_min,
                      child_min_len
                  ) >= 0) ||
                 UINT32_MAX - term_count < child_terms))
            {
                status = EVOKE_ERR_FORMAT;
            }
            if (status == EVOKE_OK && index == 0)
            {
                min_key = child_min;
                min_key_len = child_min_len;
                child_min = NULL;
            }
            if (status == EVOKE_OK)
            {
                free(previous_max);
                previous_max = child_max;
                previous_max_len = child_max_len;
                child_max = NULL;
                term_count += child_terms;
            }
            free(child_min);
            free(child_max);
        }
        if (status == EVOKE_OK)
        {
            max_key = previous_max;
            max_key_len = previous_max_len;
            previous_max = NULL;
        }
    }
done:
    free(previous_max);
    evoke_prefix_cow_object_free(&object);
    if (status != EVOKE_OK || term_count != ref->term_count)
    {
        free(min_key);
        free(max_key);
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    *min_key_out = min_key;
    *min_key_len_out = min_key_len;
    *max_key_out = max_key;
    *max_key_len_out = max_key_len;
    *term_count_out = (uint32_t) term_count;
    return EVOKE_OK;
}

evoke_status
evoke_prefix_cow_validate_external(
    const evoke_prefix_cow_ref *root,
    uint32_t expected_term_count,
    evoke_prefix_cow_object_loader loader,
    void *loader_context
)
{
    evoke_prefix_cow_validation_state state = {0};
    uint8_t *min_key = NULL;
    uint8_t *max_key = NULL;
    uint32_t min_key_len = 0;
    uint32_t max_key_len = 0;
    uint32_t term_count = 0;
    evoke_status status;

    if (!evoke_prefix_cow_ref_is_valid(root, false) ||
        expected_term_count == 0 || root->term_count != expected_term_count ||
        loader == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    state.seen_term_ids = calloc(
        expected_term_count,
        sizeof(*state.seen_term_ids)
    );
    if (state.seen_term_ids == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    state.expected_term_count = expected_term_count;
    state.loader = loader;
    state.loader_context = loader_context;
    status = evoke_prefix_cow_validate_subtree(
        &state,
        root,
        0,
        &min_key,
        &min_key_len,
        &max_key,
        &max_key_len,
        &term_count
    );
    if (status == EVOKE_OK)
    {
        for (uint32_t term_id = 0;
             term_id < expected_term_count;
             term_id++)
        {
            if (!state.seen_term_ids[term_id])
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
        }
    }
    free(min_key);
    free(max_key);
    free(state.seen_term_ids);
    free(state.visited);
    return status;
}

typedef struct evoke_prefix_cow_visit_state
{
    evoke_prefix_cow_object_loader loader;
    void *loader_context;
    evoke_prefix_cow_object_visitor visitor;
    void *visitor_context;
    uint32_t visited_terms;
} evoke_prefix_cow_visit_state;

static evoke_status
evoke_prefix_cow_visit_subtree(
    evoke_prefix_cow_visit_state *state,
    const evoke_prefix_cow_ref *ref,
    uint32_t depth
)
{
    evoke_prefix_cow_object object = {0};
    evoke_status status;

    if (depth >= EVOKE_PREFIX_COW_MAX_DEPTH)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = state->loader(state->loader_context, ref, &object);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_prefix_cow_object_shape_validate(&object, true);
    if (status != EVOKE_OK || !evoke_prefix_cow_refs_equal(
            &object.ref,
            ref))
    {
        evoke_prefix_cow_object_free(&object);
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    status = state->visitor(state->visitor_context, &object);
    if (status == EVOKE_OK &&
        object.ref.kind == EVOKE_PREFIX_COW_OBJECT_NODE)
    {
        for (uint32_t index = 0;
             status == EVOKE_OK && index < object.value.node.child_count;
             index++)
        {
            status = evoke_prefix_cow_visit_subtree(
                state,
                &object.value.node.children[index].ref,
                depth + 1
            );
        }
    }
    else if (status == EVOKE_OK)
    {
        if (UINT32_MAX - state->visited_terms < object.ref.term_count)
        {
            status = EVOKE_ERR_RANGE;
        }
        else
        {
            state->visited_terms += object.ref.term_count;
        }
    }
    evoke_prefix_cow_object_free(&object);
    return status;
}

evoke_status
evoke_prefix_cow_visit_external(
    const evoke_prefix_cow_ref *root,
    uint32_t expected_term_count,
    evoke_prefix_cow_object_loader loader,
    void *loader_context,
    evoke_prefix_cow_object_visitor visitor,
    void *visitor_context
)
{
    evoke_prefix_cow_visit_state state = {0};
    evoke_status status;

    if (!evoke_prefix_cow_ref_is_valid(root, false) ||
        expected_term_count == 0 || root->term_count != expected_term_count ||
        loader == NULL || visitor == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    state.loader = loader;
    state.loader_context = loader_context;
    state.visitor = visitor;
    state.visitor_context = visitor_context;
    status = evoke_prefix_cow_visit_subtree(&state, root, 0);
    if (status == EVOKE_OK && state.visited_terms != expected_term_count)
    {
        status = EVOKE_ERR_FORMAT;
    }
    return status;
}
