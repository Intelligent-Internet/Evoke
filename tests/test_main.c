#include "evoke_block_ranges.h"
#include "evoke_core.h"
#include "evoke_document_cow.h"
#include "evoke_document_tid_lookup.h"
#include "evoke_lexicon_cow.h"
#include "evoke_prefix_cow.h"
#include "evoke_posting_heat.h"
#include "evoke_query.h"
#include "evoke_semantic_accelerator.h"
#include "evoke_semantic_accelerator_builder.h"
#include "evoke_semantic_accelerator_directory.h"
#include "evoke_semantic_bmp.h"
#include "evoke_semantic_forward.h"
#include "evoke_semantic_forward_bound.h"
#include "evoke_semantic_impact_frontier.h"
#include "evoke_scope.h"
#include "evoke_segments.h"
#include "evoke_storage.h"
#include "evoke_term_cow.h"
#include "evoke_text.h"
#include "evoke_weighted_space_saving.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT_TRUE(expr)                                                       \
    do                                                                          \
    {                                                                           \
        if (!(expr))                                                            \
        {                                                                       \
            fprintf(stderr, "assertion failed at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, #expr);                                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define ASSERT_STATUS_OK(expr)                                                  \
    do                                                                          \
    {                                                                           \
        evoke_status _status = (expr);                                     \
        if (_status != EVOKE_OK)                                           \
        {                                                                       \
            fprintf(stderr, "unexpected status at %s:%d: %s (%s)\n",            \
                    __FILE__, __LINE__, #expr, evoke_strerror(_status));   \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static uint32_t
test_read_u32_le(const uint8_t *bytes)
{
    return (uint32_t) bytes[0] |
        ((uint32_t) bytes[1] << 8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static uint64_t
test_read_u64_le(const uint8_t *bytes)
{
    uint64_t value = 0;

    for (uint32_t index = 0; index < 8; index++)
    {
        value |= (uint64_t) bytes[index] << (index * 8);
    }
    return value;
}

static void
test_write_u16_le(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t) value;
    bytes[1] = (uint8_t) (value >> 8);
}

static void
test_write_u32_le(uint8_t *bytes, uint32_t value)
{
    for (uint32_t index = 0; index < 4; index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static void
test_write_u64_le(uint8_t *bytes, uint64_t value)
{
    for (uint32_t index = 0; index < 8; index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static uint64_t
test_term_directory_checksum(const uint8_t *bytes, size_t size)
{
    const size_t checksum_offset = 48;
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= checksum_offset &&
            index < checksum_offset + sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static uint64_t
test_segment_manifest_checksum(const uint8_t *bytes, size_t size)
{
    const size_t checksum_offset = 584;
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value = index >= checksum_offset &&
            index < checksum_offset + sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static uint64_t
test_term_cow_checksum(const uint8_t *bytes, size_t size)
{
    const size_t checksum_offset = 56;
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= checksum_offset &&
            index < checksum_offset + sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static uint64_t
test_semantic_accelerator_directory_checksum(
    const uint8_t *bytes,
    size_t size
)
{
    const size_t checksum_offset = 64;
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= checksum_offset &&
            index < checksum_offset + sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static uint64_t
test_semantic_forward_checksum(const uint8_t *bytes, size_t size)
{
    const size_t checksum_offset = 40;
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= checksum_offset &&
            index < checksum_offset + sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static uint64_t
test_scope_checksum(const uint8_t *bytes, size_t size)
{
    const size_t checksum_offset = 72;
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= checksum_offset &&
            index < checksum_offset + sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static void
assert_float_close(float actual, float expected)
{
    float diff = fabsf(actual - expected);
    if (diff > 1e-5f)
    {
        fprintf(stderr,
                "float mismatch: expected %.8f, got %.8f\n",
                expected,
                actual);
        exit(1);
    }
}

static void
assert_uint32_array(
    const uint32_t *actual,
    const uint32_t *expected,
    size_t len
)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        if (actual[i] != expected[i])
        {
            fprintf(stderr,
                    "uint32 mismatch at %zu: expected %u, got %u\n",
                    i,
                    expected[i],
                    actual[i]);
            exit(1);
        }
    }
}

static void
assert_retired_ranges(
    const evoke_block_range *ranges,
    size_t range_count,
    uint32_t high_watermark
)
{
    ASSERT_TRUE(ranges != NULL);
    ASSERT_TRUE(range_count > 0);
    for (size_t index = 0; index < range_count; index++)
    {
        ASSERT_TRUE(ranges[index].start_block > 0);
        ASSERT_TRUE(ranges[index].block_count > 0);
        ASSERT_TRUE(
            (uint64_t) ranges[index].start_block +
                ranges[index].block_count <= high_watermark
        );
        for (size_t prior = 0; prior < index; prior++)
        {
            ASSERT_TRUE(
                ranges[prior].start_block != ranges[index].start_block ||
                ranges[prior].block_count != ranges[index].block_count
            );
        }
    }
}

static void
assert_uint64_array(
    const uint64_t *actual,
    const uint64_t *expected,
    size_t len
)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        if (actual[i] != expected[i])
        {
            fprintf(stderr,
                    "uint64 mismatch at %zu: expected %llu, got %llu\n",
                    i,
                    (unsigned long long) expected[i],
                    (unsigned long long) actual[i]);
            exit(1);
        }
    }
}

static void
assert_float_array(
    const float *actual,
    const float *expected,
    size_t len
)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        assert_float_close(actual[i], expected[i]);
    }
}

static void
assert_uint32_array_present(
    const uint32_t *actual,
    const uint32_t *expected,
    size_t len
)
{
    ASSERT_TRUE(actual != NULL);
    assert_uint32_array(actual, expected, len);
}

static evoke_doc_ids
make_doc(uint32_t *ids, size_t len)
{
    evoke_doc_ids doc;

    doc.token_ids = ids;
    doc.len = len;
    return doc;
}

static void
assert_index_layout_equal(
    const evoke_index *actual,
    const evoke_index *expected
)
{
    uint32_t i;

    ASSERT_TRUE(actual->num_docs == expected->num_docs);
    ASSERT_TRUE(actual->vocab_size == expected->vocab_size);
    ASSERT_TRUE(actual->data_len == expected->data_len);
    ASSERT_TRUE(actual->has_empty_token == expected->has_empty_token);
    ASSERT_TRUE(actual->empty_token_id == expected->empty_token_id);
    assert_float_array(actual->data, expected->data, expected->data_len);
    assert_uint32_array(actual->indices, expected->indices, expected->data_len);
    assert_uint64_array(
        actual->indptr,
        expected->indptr,
        (size_t) expected->vocab_size + 1
    );
    assert_uint32_array_present(
        actual->term_frequencies,
        expected->term_frequencies,
        expected->data_len
    );
    assert_uint32_array_present(
        actual->doc_lengths,
        expected->doc_lengths,
        expected->num_docs
    );
    assert_uint32_array_present(
        actual->doc_frequencies,
        expected->doc_frequencies,
        expected->vocab_size
    );
    if (expected->nonoccurrence == NULL)
    {
        ASSERT_TRUE(actual->nonoccurrence == NULL);
    }
    else
    {
        assert_float_array(
            actual->nonoccurrence,
            expected->nonoccurrence,
            expected->vocab_size
        );
    }
    if (expected->vocab == NULL)
    {
        ASSERT_TRUE(actual->vocab == NULL);
    }
    else
    {
        ASSERT_TRUE(actual->vocab != NULL);
        for (i = 0; i < expected->vocab_size; i++)
        {
            ASSERT_TRUE(strcmp(actual->vocab[i], expected->vocab[i]) == 0);
        }
    }
}

static void
write_u32_le(uint8_t *dst, uint32_t value)
{
    dst[0] = (uint8_t) (value & 0xFFU);
    dst[1] = (uint8_t) ((value >> 8) & 0xFFU);
    dst[2] = (uint8_t) ((value >> 16) & 0xFFU);
    dst[3] = (uint8_t) ((value >> 24) & 0xFFU);
}

static void
write_u64_le(uint8_t *dst, uint64_t value)
{
    size_t i;

    for (i = 0; i < 8; i++)
    {
        dst[i] = (uint8_t) ((value >> (i * 8)) & 0xFFU);
    }
}

static void
initialize_test_object_ref(
    evoke_segment_object_ref *ref,
    evoke_segment_object_kind kind,
    uint32_t start_block,
    uint32_t page_count,
    uint64_t manifest_id,
    uint64_t object_checksum
)
{
    memset(ref, 0, sizeof(*ref));
    ref->object_kind = kind;
    ref->start_block = start_block;
    ref->page_count = page_count;
    ref->object_id = manifest_id;
    ref->owner_manifest_id = manifest_id;
    ref->object_bytes = 128;
    ref->object_checksum = object_checksum;
}

static void
initialize_test_document_directory(
    evoke_segment_manifest *manifest,
    uint32_t start_block,
    uint64_t object_checksum
)
{
    manifest->flags |=
        EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY;
    initialize_test_object_ref(
        &manifest->document_directory,
        EVOKE_SEGMENT_OBJECT_DOCUMENT_DIRECTORY,
        start_block,
        1,
        manifest->manifest_id,
        object_checksum
    );
}

static void
initialize_test_segment_manifest(evoke_segment_manifest *manifest)
{
    evoke_segment_manifest_init(manifest);
    manifest->flags =
        EVOKE_SEGMENT_MANIFEST_FLAG_SAE |
        EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY |
        EVOKE_SEGMENT_MANIFEST_FLAG_NEUTRAL_FOLD |
        EVOKE_SEGMENT_MANIFEST_FLAG_IMPACT_FOLD |
        EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY |
        EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP |
        EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP;
    manifest->manifest_id = 9;
    manifest->parent_manifest_id = 8;
    manifest->max_sequence = 120;
    manifest->statistics_epoch = 7;
    manifest->visible_document_count = 5;
    manifest->document_slot_count = 5;
    manifest->total_document_length = 12;
    manifest->reclaim_before_sequence = 50;
    manifest->neutral_fold_coverage = 100;
    manifest->impact_fold_coverage = 100;
    manifest->impact_statistics_epoch = 7;
    manifest->vocab_size = 4;
    initialize_test_object_ref(
        &manifest->query_contract,
        EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT,
        24,
        2,
        manifest->manifest_id,
        100
    );
    initialize_test_object_ref(
        &manifest->term_directory,
        EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
        30,
        2,
        manifest->manifest_id,
        101
    );
    initialize_test_object_ref(
        &manifest->neutral_fold,
        EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
        40,
        4,
        manifest->manifest_id,
        102
    );
    initialize_test_object_ref(
        &manifest->impact_fold,
        EVOKE_SEGMENT_OBJECT_IMPACT_FOLD,
        50,
        4,
        manifest->manifest_id,
        103
    );
    initialize_test_object_ref(
        &manifest->document_directory,
        EVOKE_SEGMENT_OBJECT_DOCUMENT_DIRECTORY,
        56,
        2,
        manifest->manifest_id,
        105
    );
    initialize_test_object_ref(
        &manifest->lexicon_lookup,
        EVOKE_SEGMENT_OBJECT_LEXICON_LOOKUP,
        58,
        1,
        manifest->manifest_id,
        106
    );
    initialize_test_object_ref(
        &manifest->prefix_lookup,
        EVOKE_SEGMENT_OBJECT_PREFIX_LOOKUP,
        60,
        1,
        manifest->manifest_id,
        107
    );
    manifest->lexicon_hash_seed = UINT64_C(0x42C0FFEE12345678);
    manifest->segment_count = 2;
    manifest->retired_range_count = 2;
    manifest->retired_ranges = calloc(
        manifest->retired_range_count,
        sizeof(*manifest->retired_ranges)
    );
    manifest->segments = calloc(
        manifest->segment_count,
        sizeof(*manifest->segments)
    );
    manifest->doc_frequencies = calloc(
        manifest->vocab_size,
        sizeof(*manifest->doc_frequencies)
    );
    ASSERT_TRUE(manifest->segments != NULL);
    ASSERT_TRUE(manifest->retired_ranges != NULL);
    ASSERT_TRUE(manifest->doc_frequencies != NULL);

    manifest->retired_ranges[0].start_block = 10;
    manifest->retired_ranges[0].block_count = 2;
    manifest->retired_ranges[1].start_block = 14;
    manifest->retired_ranges[1].block_count = 1;

    manifest->doc_frequencies[0] = 3;
    manifest->doc_frequencies[1] = 2;
    manifest->doc_frequencies[2] = 2;
    manifest->doc_frequencies[3] = 2;
    memset(manifest->contract_hash, 0x42, sizeof(manifest->contract_hash));

    manifest->segments[0].segment_id = 1;
    manifest->segments[0].min_sequence = 1;
    manifest->segments[0].max_sequence = 100;
    manifest->segments[0].posting_count = 7;
    manifest->segments[0].document_count = 4;
    manifest->segments[0].total_document_length = 9;
    manifest->segments[0].first_document_slot = 0;
    manifest->segments[0].document_slot_count = 4;
    manifest->segments[0].payload_checksum = 11;
    manifest->segments[0].start_block = 1;
    manifest->segments[0].block_count = 8;
    manifest->segments[0].size_class = 2;
    manifest->segments[0].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL |
        EVOKE_SEGMENT_FLAG_SEMANTIC;
    manifest->segments[0].payload_bytes = 32768;
    manifest->segments[0].payload_owner_manifest_id =
        manifest->parent_manifest_id;

    manifest->segments[1].segment_id = 2;
    manifest->segments[1].min_sequence = 101;
    manifest->segments[1].max_sequence = 120;
    manifest->segments[1].posting_count = 2;
    manifest->segments[1].retirement_count = 1;
    manifest->segments[1].document_count = 1;
    manifest->segments[1].total_document_length = 3;
    manifest->segments[1].first_document_slot = 4;
    manifest->segments[1].document_slot_count = 1;
    manifest->segments[1].payload_checksum = 12;
    manifest->segments[1].start_block = 20;
    manifest->segments[1].block_count = 2;
    manifest->segments[1].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL |
        EVOKE_SEGMENT_FLAG_PENDING |
        EVOKE_SEGMENT_FLAG_RETIREMENTS;
    manifest->segments[1].payload_bytes = 8192;
    manifest->segments[1].payload_owner_manifest_id =
        manifest->manifest_id;
}

static void
test_block_range_inventory_classifies_physical_debt(void)
{
    const evoke_block_range retired_ranges[] = {
        {5, 3},
        {10, 2}
    };
    evoke_block_range_inventory inventory;
    evoke_block_range_allocator allocator;
    uint32_t start_block = 0;

    evoke_block_range_inventory_init(&inventory);
    evoke_block_range_allocator_init(&allocator);
    ASSERT_STATUS_OK(evoke_block_range_inventory_add(
        &inventory,
        8,
        2
    ));
    ASSERT_STATUS_OK(evoke_block_range_inventory_add(
        &inventory,
        0,
        1
    ));
    ASSERT_STATUS_OK(evoke_block_range_inventory_add(
        &inventory,
        1,
        2
    ));
    ASSERT_STATUS_OK(evoke_block_range_inventory_add(
        &inventory,
        1,
        2
    ));
    ASSERT_STATUS_OK(evoke_block_range_inventory_add(
        &inventory,
        8,
        2
    ));
    ASSERT_STATUS_OK(evoke_block_range_inventory_add(
        &inventory,
        3,
        2
    ));
    ASSERT_STATUS_OK(evoke_block_range_inventory_finalize(
        &inventory,
        12
    ));
    ASSERT_TRUE(inventory.observed_range_count == 6);
    ASSERT_TRUE(inventory.range_count == 2);
    ASSERT_TRUE(inventory.ranges[0].start_block == 0);
    ASSERT_TRUE(inventory.ranges[0].block_count == 5);
    ASSERT_TRUE(inventory.ranges[1].start_block == 8);
    ASSERT_TRUE(inventory.ranges[1].block_count == 2);
    ASSERT_TRUE(inventory.reachable_block_count == 7);
    ASSERT_TRUE(inventory.interior_unreachable_block_count == 5);
    ASSERT_TRUE(inventory.highest_reachable_block_exclusive == 10);
    ASSERT_STATUS_OK(evoke_block_range_allocator_build(
        &inventory,
        &allocator
    ));
    ASSERT_TRUE(allocator.range_count == 2);
    ASSERT_TRUE(allocator.available_block_count == 5);
    ASSERT_TRUE(evoke_block_range_allocator_allocate_best_fit(
        &allocator,
        2,
        &start_block
    ));
    ASSERT_TRUE(start_block == 10);
    ASSERT_TRUE(allocator.available_block_count == 3);
    ASSERT_TRUE(allocator.allocated_block_count == 2);
    ASSERT_TRUE(evoke_block_range_allocator_allocate_best_fit(
        &allocator,
        3,
        &start_block
    ));
    ASSERT_TRUE(start_block == 5);
    ASSERT_TRUE(allocator.range_count == 0);
    ASSERT_TRUE(allocator.available_block_count == 0);
    ASSERT_TRUE(!evoke_block_range_allocator_allocate_best_fit(
        &allocator,
        1,
        &start_block
    ));
    evoke_block_range_allocator_free(&allocator);
    evoke_block_range_inventory_free(&inventory);

    evoke_block_range_allocator_init(&allocator);
    ASSERT_STATUS_OK(evoke_block_range_allocator_build_free(
        retired_ranges,
        sizeof(retired_ranges) / sizeof(retired_ranges[0]),
        12,
        &allocator
    ));
    ASSERT_TRUE(allocator.available_block_count == 5);
    ASSERT_TRUE(evoke_block_range_allocator_allocate_best_fit(
        &allocator,
        2,
        &start_block
    ));
    ASSERT_TRUE(start_block == 10);
    ASSERT_TRUE(evoke_block_range_allocator_allocate_best_fit(
        &allocator,
        3,
        &start_block
    ));
    ASSERT_TRUE(start_block == 5);
    evoke_block_range_allocator_free(&allocator);

    evoke_block_range_inventory_init(&inventory);
    evoke_block_range_allocator_init(&allocator);
    ASSERT_STATUS_OK(evoke_block_range_inventory_add(
        &inventory,
        0,
        4
    ));
    ASSERT_STATUS_OK(evoke_block_range_inventory_add(
        &inventory,
        2,
        4
    ));
    ASSERT_TRUE(evoke_block_range_inventory_finalize(
        &inventory,
        8
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_block_range_allocator_build(
        &inventory,
        &allocator
    ) == EVOKE_ERR_INVALID);
    evoke_block_range_allocator_free(&allocator);
    evoke_block_range_inventory_free(&inventory);

    evoke_block_range_inventory_init(&inventory);
    evoke_block_range_allocator_init(&allocator);
    ASSERT_STATUS_OK(evoke_block_range_inventory_add(
        &inventory,
        0,
        1
    ));
    ASSERT_STATUS_OK(evoke_block_range_inventory_add(
        &inventory,
        9,
        2
    ));
    ASSERT_TRUE(evoke_block_range_inventory_finalize(
        &inventory,
        10
    ) == EVOKE_ERR_RANGE);
    evoke_block_range_allocator_free(&allocator);
    evoke_block_range_inventory_free(&inventory);
}

static void
test_segment_page_header_roundtrip_and_validation(void)
{
    static const uint8_t object_bytes[] = {
        0x10U, 0x20U, 0x30U, 0x40U, 0x50U
    };
    evoke_segment_page_header header;
    evoke_segment_page_header restored;
    uint8_t bytes[EVOKE_SEGMENT_PAGE_HEADER_SIZE];
    uint8_t mutated[EVOKE_SEGMENT_PAGE_HEADER_SIZE];
    uint8_t page_payload[116];
    uint64_t checksum;
    uint32_t page_checksum;
    uint32_t page_count = 0;
    const size_t page_content_bytes = 256;

    checksum = evoke_segment_blob_checksum(
        object_bytes,
        sizeof(object_bytes)
    );
    page_checksum = evoke_segment_page_payload_checksum(
        object_bytes,
        sizeof(object_bytes)
    );
    ASSERT_TRUE(checksum != 0);
    ASSERT_TRUE(checksum != evoke_segment_blob_checksum(
        object_bytes,
        sizeof(object_bytes) - 1
    ));
    ASSERT_STATUS_OK(evoke_segment_page_count_required(
        500,
        page_content_bytes,
        &page_count
    ));
    ASSERT_TRUE(page_count == 3);

    memset(&header, 0, sizeof(header));
    header.object_kind = EVOKE_SEGMENT_OBJECT_PAYLOAD;
    header.object_id = 41;
    header.owner_manifest_id = 9;
    header.object_bytes = 500;
    header.object_checksum = checksum;
    header.payload_checksum = page_checksum;
    header.page_count = page_count;
    header.used_bytes = 192;
    ASSERT_STATUS_OK(evoke_segment_page_header_serialize(
        &header,
        page_content_bytes,
        bytes,
        sizeof(bytes)
    ));
    memset(&restored, 0, sizeof(restored));
    ASSERT_STATUS_OK(evoke_segment_page_header_deserialize(
        bytes,
        sizeof(bytes),
        page_content_bytes,
        &restored
    ));
    ASSERT_TRUE(restored.object_kind == header.object_kind);
    ASSERT_TRUE(restored.object_id == header.object_id);
    ASSERT_TRUE(restored.owner_manifest_id == header.owner_manifest_id);
    ASSERT_TRUE(restored.object_bytes == header.object_bytes);
    ASSERT_TRUE(restored.object_checksum == header.object_checksum);
    ASSERT_TRUE(restored.ordinal == 0);
    ASSERT_TRUE(restored.page_count == 3);
    ASSERT_TRUE(restored.used_bytes == 192);
    ASSERT_TRUE(restored.payload_checksum == page_checksum);

    header.ordinal = 2;
    header.used_bytes = 116;
    ASSERT_STATUS_OK(evoke_segment_page_header_serialize(
        &header,
        page_content_bytes,
        bytes,
        sizeof(bytes)
    ));
    ASSERT_STATUS_OK(evoke_segment_page_header_deserialize(
        bytes,
        sizeof(bytes),
        page_content_bytes,
        &restored
    ));
    ASSERT_TRUE(restored.ordinal == 2);
    ASSERT_TRUE(restored.used_bytes == 116);

    memset(page_payload, 0xA5, sizeof(page_payload));
    restored.payload_checksum = evoke_segment_page_payload_checksum(
        page_payload,
        sizeof(page_payload)
    );
    ASSERT_STATUS_OK(evoke_segment_page_payload_validate(
        &restored,
        page_payload,
        sizeof(page_payload)
    ));
    page_payload[17] ^= 0x1U;
    ASSERT_TRUE(evoke_segment_page_payload_validate(
        &restored,
        page_payload,
        sizeof(page_payload)
    ) == EVOKE_ERR_FORMAT);

    memcpy(mutated, bytes, sizeof(mutated));
    mutated[16] ^= 0x1U;
    ASSERT_TRUE(evoke_segment_page_header_deserialize(
        mutated,
        sizeof(mutated),
        page_content_bytes,
        &restored
    ) == EVOKE_ERR_FORMAT);

    header.used_bytes = 115;
    ASSERT_TRUE(evoke_segment_page_header_serialize(
        &header,
        page_content_bytes,
        bytes,
        sizeof(bytes)
    ) == EVOKE_ERR_FORMAT);
    header.used_bytes = 116;
    header.object_kind = EVOKE_SEGMENT_OBJECT_INVALID;
    ASSERT_TRUE(evoke_segment_page_header_serialize(
        &header,
        page_content_bytes,
        bytes,
        sizeof(bytes)
    ) == EVOKE_ERR_FORMAT);
    header.object_kind = EVOKE_SEGMENT_OBJECT_PAYLOAD;
    header.payload_checksum = 0;
    ASSERT_TRUE(evoke_segment_page_header_serialize(
        &header,
        page_content_bytes,
        bytes,
        sizeof(bytes)
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_segment_page_count_required(
        0,
        page_content_bytes,
        &page_count
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_segment_page_count_required(
        500,
        EVOKE_SEGMENT_PAGE_HEADER_SIZE,
        &page_count
    ) == EVOKE_ERR_FORMAT);
}

static void
test_active_l0_header_and_frontier_validation(void)
{
    static const uint8_t payload[] = {
        0x01U, 0x03U, 0x05U, 0x07U, 0x09U
    };
    evoke_active_l0_page_header header;
    evoke_active_l0_page_header restored;
    evoke_active_l0_frontier frontier;
    uint8_t bytes[EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE];
    uint8_t mutated[EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE];
    const size_t page_content_bytes = 256;

    memset(&header, 0, sizeof(header));
    header.segment_id = 17;
    header.min_sequence = 101;
    header.max_sequence = 105;
    header.payload_checksum = evoke_segment_blob_checksum(
        payload,
        sizeof(payload)
    );
    header.ordinal = 2;
    header.next_block = EVOKE_ACTIVE_L0_NO_NEXT_BLOCK;
    header.frame_count = 3;
    header.used_bytes = sizeof(payload);
    ASSERT_STATUS_OK(evoke_active_l0_page_header_serialize(
        &header,
        page_content_bytes,
        bytes,
        sizeof(bytes)
    ));
    memset(&restored, 0, sizeof(restored));
    ASSERT_STATUS_OK(evoke_active_l0_page_header_deserialize(
        bytes,
        sizeof(bytes),
        page_content_bytes,
        &restored
    ));
    ASSERT_TRUE(restored.segment_id == header.segment_id);
    ASSERT_TRUE(restored.min_sequence == header.min_sequence);
    ASSERT_TRUE(restored.max_sequence == header.max_sequence);
    ASSERT_TRUE(restored.ordinal == header.ordinal);
    ASSERT_TRUE(restored.next_block == header.next_block);
    ASSERT_TRUE(restored.frame_count == header.frame_count);
    ASSERT_TRUE(restored.used_bytes == header.used_bytes);
    ASSERT_STATUS_OK(evoke_active_l0_page_payload_validate(
        &restored,
        payload,
        sizeof(payload)
    ));

    memcpy(mutated, bytes, sizeof(mutated));
    mutated[24] ^= 0x1U;
    ASSERT_TRUE(evoke_active_l0_page_header_deserialize(
        mutated,
        sizeof(mutated),
        page_content_bytes,
        &restored
    ) == EVOKE_ERR_FORMAT);
    mutated[0] = payload[0] ^ 0x1U;
    ASSERT_TRUE(evoke_active_l0_page_payload_validate(
        &header,
        mutated,
        sizeof(payload)
    ) == EVOKE_ERR_FORMAT);

    header.next_block = 0;
    ASSERT_TRUE(evoke_active_l0_page_header_serialize(
        &header,
        page_content_bytes,
        bytes,
        sizeof(bytes)
    ) == EVOKE_ERR_FORMAT);
    header.next_block = EVOKE_ACTIVE_L0_NO_NEXT_BLOCK;
    header.used_bytes = page_content_bytes;
    ASSERT_TRUE(evoke_active_l0_page_header_serialize(
        &header,
        page_content_bytes,
        bytes,
        sizeof(bytes)
    ) == EVOKE_ERR_FORMAT);

    memset(&frontier, 0, sizeof(frontier));
    ASSERT_STATUS_OK(evoke_active_l0_frontier_validate(
        &frontier,
        true
    ));
    ASSERT_TRUE(evoke_active_l0_frontier_validate(
        &frontier,
        false
    ) == EVOKE_ERR_FORMAT);

    frontier.segment_id = 18;
    ASSERT_STATUS_OK(evoke_active_l0_frontier_validate(
        &frontier,
        false
    ));
    frontier.min_sequence = 201;
    frontier.max_sequence = 212;
    frontier.payload_bytes = 4096;
    frontier.head_block = 40;
    frontier.tail_block = 45;
    frontier.page_count = 3;
    frontier.record_count = 12;
    ASSERT_STATUS_OK(evoke_active_l0_frontier_validate(
        &frontier,
        false
    ));
    frontier.page_count = EVOKE_ACTIVE_L0_MAX_PAGES + 1;
    ASSERT_TRUE(evoke_active_l0_frontier_validate(
        &frontier,
        false
    ) == EVOKE_ERR_FORMAT);
    frontier.page_count = 4;
    frontier.record_count = 1;
    frontier.max_sequence = frontier.min_sequence;
    ASSERT_STATUS_OK(evoke_active_l0_frontier_validate(
        &frontier,
        false
    ));
    frontier.record_count = EVOKE_ACTIVE_L0_MAX_RECORDS + 1;
    frontier.max_sequence =
        frontier.min_sequence + frontier.record_count - 1;
    ASSERT_TRUE(evoke_active_l0_frontier_validate(
        &frontier,
        false
    ) == EVOKE_ERR_FORMAT);
}

static void
test_l0_record_and_frame_roundtrip(void)
{
    static const uint8_t fingerprint[
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    ] = {
        0x01U, 0x02U, 0x03U, 0x04U,
        0x05U, 0x06U, 0x07U, 0x08U,
        0x09U, 0x0aU, 0x0bU, 0x0cU,
        0x0dU, 0x0eU, 0x0fU, 0x10U
    };
    static const uint8_t bird[] = "bird";
    static const uint8_t river[] = "river";
    evoke_l0_lexical_atom numeric_atoms[2];
    evoke_l0_lexical_atom text_atoms[2];
    evoke_l0_lexical_atom decoded_atoms[2];
    evoke_l0_lexical_atom restored_atom;
    evoke_l0_semantic_atom semantic_atoms[2];
    evoke_l0_semantic_atom decoded_semantic_atoms[2];
    evoke_l0_semantic_atom restored_semantic_atom;
    evoke_l0_record record;
    evoke_l0_record_view view;
    evoke_l0_frame_header frame;
    evoke_l0_frame_header restored_frame;
    uint8_t frame_bytes[EVOKE_L0_FRAME_HEADER_SIZE];
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    size_t size = 0;

    memset(numeric_atoms, 0, sizeof(numeric_atoms));
    numeric_atoms[0].term_id = 2;
    numeric_atoms[0].term_frequency = 3;
    numeric_atoms[1].term_id = 8;
    numeric_atoms[1].term_frequency = 1;
    memset(&record, 0, sizeof(record));
    record.kind = EVOKE_L0_RECORD_UPSERT;
    record.term_encoding = EVOKE_L0_TERM_ENCODING_NUMERIC;
    record.sequence = 11;
    record.document_slot = 7;
    record.record_xid = 3;
    record.heap_block = 5;
    record.heap_offset = 2;
    record.document_length = 4;
    record.semantic_input_fingerprint = fingerprint;
    record.atoms = numeric_atoms;
    record.atom_count = 2;
    ASSERT_STATUS_OK(evoke_l0_record_serialize(
        &record,
        &bytes,
        &size
    ));
    ASSERT_TRUE(size == EVOKE_L0_RECORD_HEADER_SIZE + 16);
    ASSERT_STATUS_OK(evoke_l0_record_view_parse(bytes, size, &view));
    ASSERT_TRUE(view.kind == record.kind);
    ASSERT_TRUE(view.term_encoding == record.term_encoding);
    ASSERT_TRUE(view.sequence == record.sequence);
    ASSERT_TRUE(view.document_slot == record.document_slot);
    ASSERT_TRUE(view.atom_count == record.atom_count);
    ASSERT_TRUE(memcmp(
        view.semantic_input_fingerprint,
        fingerprint,
        sizeof(fingerprint)
    ) == 0);
    ASSERT_STATUS_OK(evoke_l0_record_view_atom(
        &view,
        1,
        &restored_atom
    ));
    ASSERT_TRUE(restored_atom.term_id == 8);
    ASSERT_TRUE(restored_atom.term_frequency == 1);
    ASSERT_STATUS_OK(evoke_l0_record_view_decode_atoms(
        &view,
        decoded_atoms,
        2
    ));
    ASSERT_TRUE(decoded_atoms[0].term_id == 2);
    ASSERT_TRUE(decoded_atoms[0].term_frequency == 3);
    ASSERT_TRUE(decoded_atoms[1].term_id == 8);

    memset(&frame, 0, sizeof(frame));
    frame.flags = EVOKE_L0_FRAME_FLAG_START | EVOKE_L0_FRAME_FLAG_END;
    frame.sequence = record.sequence;
    frame.record_checksum = evoke_segment_blob_checksum(bytes, size);
    frame.record_bytes = (uint32_t) size;
    frame.fragment_bytes = (uint32_t) size;
    ASSERT_STATUS_OK(evoke_l0_frame_header_serialize(
        &frame,
        frame_bytes,
        sizeof(frame_bytes)
    ));
    ASSERT_STATUS_OK(evoke_l0_frame_header_deserialize(
        frame_bytes,
        sizeof(frame_bytes),
        &restored_frame
    ));
    ASSERT_TRUE(restored_frame.sequence == frame.sequence);
    ASSERT_TRUE(restored_frame.record_checksum == frame.record_checksum);
    ASSERT_TRUE(restored_frame.flags == frame.flags);
    frame.flags = EVOKE_L0_FRAME_FLAG_END;
    ASSERT_TRUE(evoke_l0_frame_header_serialize(
        &frame,
        frame_bytes,
        sizeof(frame_bytes)
    ) == EVOKE_ERR_FORMAT);

    mutated = malloc(size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, size);
    mutated[32] ^= 0x1U;
    ASSERT_TRUE(evoke_l0_record_view_parse(
        mutated,
        size,
        &view
    ) == EVOKE_ERR_FORMAT);
    free(mutated);
    free(bytes);
    mutated = NULL;
    bytes = NULL;

    memset(text_atoms, 0, sizeof(text_atoms));
    text_atoms[0].term_frequency = 2;
    text_atoms[0].term_bytes = bird;
    text_atoms[0].term_bytes_len = sizeof(bird) - 1;
    text_atoms[1].term_frequency = 1;
    text_atoms[1].term_bytes = river;
    text_atoms[1].term_bytes_len = sizeof(river) - 1;
    record.term_encoding = EVOKE_L0_TERM_ENCODING_UTF8;
    record.sequence = 12;
    record.atoms = text_atoms;
    ASSERT_STATUS_OK(evoke_l0_record_serialize(
        &record,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(evoke_l0_record_view_parse(bytes, size, &view));
    ASSERT_STATUS_OK(evoke_l0_record_view_atom(
        &view,
        0,
        &restored_atom
    ));
    ASSERT_TRUE(restored_atom.term_id == 0);
    ASSERT_TRUE(restored_atom.term_frequency == 2);
    ASSERT_TRUE(restored_atom.term_bytes_len == sizeof(bird) - 1);
    ASSERT_TRUE(memcmp(
        restored_atom.term_bytes,
        bird,
        sizeof(bird) - 1
    ) == 0);
    ASSERT_STATUS_OK(evoke_l0_record_view_decode_atoms(
        &view,
        decoded_atoms,
        2
    ));
    ASSERT_TRUE(decoded_atoms[1].term_bytes_len == sizeof(river) - 1);
    ASSERT_TRUE(memcmp(
        decoded_atoms[1].term_bytes,
        river,
        sizeof(river) - 1
    ) == 0);
    free(bytes);
    bytes = NULL;

    text_atoms[0] = text_atoms[1];
    text_atoms[1].term_frequency = 2;
    text_atoms[1].term_bytes = bird;
    text_atoms[1].term_bytes_len = sizeof(bird) - 1;
    ASSERT_TRUE(evoke_l0_record_serialized_size(
        &record,
        &size
    ) == EVOKE_ERR_FORMAT);

    memset(&record, 0, sizeof(record));
    record.kind = EVOKE_L0_RECORD_RETIRE;
    record.term_encoding = EVOKE_L0_TERM_ENCODING_NONE;
    record.sequence = 13;
    record.document_slot = 7;
    record.record_xid = 3;
    record.document_length = 4;
    ASSERT_STATUS_OK(evoke_l0_record_serialize(
        &record,
        &bytes,
        &size
    ));
    ASSERT_TRUE(size == EVOKE_L0_RECORD_HEADER_SIZE);
    ASSERT_STATUS_OK(evoke_l0_record_view_parse(bytes, size, &view));
    ASSERT_TRUE(view.kind == EVOKE_L0_RECORD_RETIRE);
    ASSERT_TRUE(view.atom_count == 0);
    ASSERT_TRUE(view.document_length == 4);
    free(bytes);
    bytes = NULL;

    memset(semantic_atoms, 0, sizeof(semantic_atoms));
    semantic_atoms[0].term_id = 12;
    semantic_atoms[0].impact = 0.75f;
    semantic_atoms[1].term_id = 19;
    semantic_atoms[1].impact = 1.5f;
    memset(&record, 0, sizeof(record));
    record.kind = EVOKE_L0_RECORD_SEMANTIC_COMPLETE;
    record.term_encoding = EVOKE_L0_TERM_ENCODING_NUMERIC;
    record.sequence = 14;
    record.document_slot = 7;
    record.record_xid = 4;
    record.semantic_input_fingerprint = fingerprint;
    record.semantic_atoms = semantic_atoms;
    record.atom_count = 2;
    ASSERT_STATUS_OK(evoke_l0_record_serialize(
        &record,
        &bytes,
        &size
    ));
    ASSERT_TRUE(size == EVOKE_L0_RECORD_HEADER_SIZE + 16);
    ASSERT_STATUS_OK(evoke_l0_record_view_parse(bytes, size, &view));
    ASSERT_TRUE(view.kind == EVOKE_L0_RECORD_SEMANTIC_COMPLETE);
    ASSERT_TRUE(view.document_slot == 7);
    ASSERT_TRUE(view.atom_count == 2);
    ASSERT_TRUE(memcmp(
        view.semantic_input_fingerprint,
        fingerprint,
        sizeof(fingerprint)
    ) == 0);
    ASSERT_STATUS_OK(evoke_l0_record_view_semantic_atom(
        &view,
        1,
        &restored_semantic_atom
    ));
    ASSERT_TRUE(restored_semantic_atom.term_id == 19);
    assert_float_close(restored_semantic_atom.impact, 1.5f);
    ASSERT_STATUS_OK(evoke_l0_record_view_decode_semantic_atoms(
        &view,
        decoded_semantic_atoms,
        2
    ));
    ASSERT_TRUE(decoded_semantic_atoms[0].term_id == 12);
    assert_float_close(decoded_semantic_atoms[0].impact, 0.75f);
    ASSERT_TRUE(evoke_l0_record_view_atom(
        &view,
        0,
        &restored_atom
    ) == EVOKE_ERR_INVALID);
    free(bytes);
    bytes = NULL;

    semantic_atoms[0].impact = 0.0f;
    ASSERT_TRUE(evoke_l0_record_serialized_size(
        &record,
        &size
    ) == EVOKE_ERR_FORMAT);
    semantic_atoms[0].impact = NAN;
    ASSERT_TRUE(evoke_l0_record_serialized_size(
        &record,
        &size
    ) == EVOKE_ERR_FORMAT);
    semantic_atoms[0].impact = 0.75f;
    semantic_atoms[1].term_id = 12;
    ASSERT_TRUE(evoke_l0_record_serialized_size(
        &record,
        &size
    ) == EVOKE_ERR_FORMAT);
    semantic_atoms[1].term_id = 19;

    memset(&record, 0, sizeof(record));
    record.kind = EVOKE_L0_RECORD_SEMANTIC_QUARANTINE;
    record.term_encoding = EVOKE_L0_TERM_ENCODING_NONE;
    record.sequence = 15;
    record.document_slot = 7;
    record.record_xid = 5;
    record.semantic_failure_count = 3;
    record.semantic_error_code = 17;
    record.semantic_retry_after = 12345;
    record.semantic_pending_since = 12000;
    record.semantic_error_hash = UINT64_C(0x123456789abcdef0);
    record.semantic_input_fingerprint = fingerprint;
    ASSERT_STATUS_OK(evoke_l0_record_serialize(
        &record,
        &bytes,
        &size
    ));
    ASSERT_TRUE(size == EVOKE_L0_RECORD_HEADER_SIZE);
    ASSERT_STATUS_OK(evoke_l0_record_view_parse(bytes, size, &view));
    ASSERT_TRUE(view.kind == EVOKE_L0_RECORD_SEMANTIC_QUARANTINE);
    ASSERT_TRUE(view.semantic_failure_count == 3);
    ASSERT_TRUE(view.semantic_error_code == 17);
    ASSERT_TRUE(view.semantic_retry_after == 12345);
    ASSERT_TRUE(view.semantic_pending_since == 12000);
    ASSERT_TRUE(
        view.semantic_error_hash == UINT64_C(0x123456789abcdef0)
    );
    ASSERT_TRUE(memcmp(
        view.semantic_input_fingerprint,
        fingerprint,
        sizeof(fingerprint)
    ) == 0);
    free(bytes);

    record.semantic_retry_after = 11999;
    ASSERT_TRUE(evoke_l0_record_serialized_size(
        &record,
        &size
    ) == EVOKE_ERR_FORMAT);
    record.semantic_retry_after = 12345;
    record.semantic_input_fingerprint = NULL;
    ASSERT_TRUE(evoke_l0_record_serialized_size(
        &record,
        &size
    ) == EVOKE_ERR_FORMAT);
}

static void
test_segment_read_root_validation(void)
{
    static const uint8_t manifest_bytes[] = {
        0x4dU, 0x61U, 0x6eU, 0x69U, 0x66U, 0x65U, 0x73U, 0x74U
    };
    evoke_segment_read_root root;
    evoke_segment_read_root restored;
    uint8_t bytes[EVOKE_SEGMENT_READ_ROOT_SERIALIZED_SIZE];
    uint8_t mutated[EVOKE_SEGMENT_READ_ROOT_SERIALIZED_SIZE];

    memset(&root, 0, sizeof(root));
    memset(&restored, 0, sizeof(restored));
    root.root_id = 9;
    root.next_sequence = 1;
    root.next_document_slot = 7;
    root.reusable_document_slot_cursor = 3;
    root.next_segment_id = 12;
    root.published_block_high_watermark = 100;
    root.manifest.object_kind = EVOKE_SEGMENT_OBJECT_MANIFEST;
    root.manifest.start_block = 10;
    root.manifest.page_count = 2;
    root.manifest.object_id = 9;
    root.manifest.owner_manifest_id = 9;
    root.manifest.object_bytes = sizeof(manifest_bytes);
    root.manifest.object_checksum = evoke_segment_blob_checksum(
        manifest_bytes,
        sizeof(manifest_bytes)
    );
    root.active_l0.segment_id = 11;
    ASSERT_STATUS_OK(evoke_segment_read_root_validate(&root));

    root.pending_l0.segment_id = 10;
    root.pending_l0.min_sequence = 1;
    root.pending_l0.max_sequence = 8;
    root.pending_l0.payload_bytes = 1024;
    root.pending_l0.head_block = 20;
    root.pending_l0.tail_block = 22;
    root.pending_l0.page_count = 2;
    root.pending_l0.record_count = 8;
    root.active_l0.min_sequence = 9;
    root.active_l0.max_sequence = 12;
    root.active_l0.payload_bytes = 2048;
    root.active_l0.head_block = 30;
    root.active_l0.tail_block = 35;
    root.active_l0.page_count = 3;
    root.active_l0.record_count = 4;
    root.next_sequence = 13;
    ASSERT_STATUS_OK(evoke_segment_read_root_validate(&root));
    ASSERT_STATUS_OK(evoke_segment_read_root_serialize(
        &root,
        bytes,
        sizeof(bytes)
    ));
    ASSERT_STATUS_OK(evoke_segment_read_root_deserialize(
        bytes,
        sizeof(bytes),
        &restored
    ));
    ASSERT_TRUE(restored.root_id == root.root_id);
    ASSERT_TRUE(
        restored.manifest.object_checksum ==
        root.manifest.object_checksum
    );
    ASSERT_TRUE(
        restored.active_l0.record_count ==
        root.active_l0.record_count
    );
    ASSERT_TRUE(
        restored.pending_l0.max_sequence ==
        root.pending_l0.max_sequence
    );
    ASSERT_TRUE(
        restored.next_document_slot == root.next_document_slot
    );
    ASSERT_TRUE(
        restored.reusable_document_slot_cursor ==
        root.reusable_document_slot_cursor
    );
    ASSERT_TRUE(restored.next_segment_id == root.next_segment_id);
    memcpy(mutated, bytes, sizeof(mutated));
    mutated[80] ^= 0x1U;
    ASSERT_TRUE(evoke_segment_read_root_deserialize(
        mutated,
        sizeof(mutated),
        &restored
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_segment_read_root_deserialize(
        bytes,
        sizeof(bytes) - 1,
        &restored
    ) == EVOKE_ERR_INVALID);

    root.next_sequence = 12;
    ASSERT_TRUE(evoke_segment_read_root_validate(
        &root
    ) == EVOKE_ERR_FORMAT);
    root.next_sequence = 13;
    root.pending_l0.max_sequence = 9;
    ASSERT_TRUE(evoke_segment_read_root_validate(
        &root
    ) == EVOKE_ERR_FORMAT);
    root.pending_l0.max_sequence = 8;
    root.active_l0.tail_block = 100;
    ASSERT_TRUE(evoke_segment_read_root_validate(
        &root
    ) == EVOKE_ERR_FORMAT);
    root.active_l0.tail_block = 35;
    root.next_segment_id = 11;
    ASSERT_TRUE(evoke_segment_read_root_validate(
        &root
    ) == EVOKE_ERR_FORMAT);
    root.next_segment_id = 12;
    root.reusable_document_slot_cursor = 8;
    ASSERT_TRUE(evoke_segment_read_root_validate(
        &root
    ) == EVOKE_ERR_FORMAT);
    root.reusable_document_slot_cursor = 3;
    root.manifest.owner_manifest_id = 8;
    ASSERT_TRUE(evoke_segment_read_root_validate(
        &root
    ) == EVOKE_ERR_FORMAT);

    root.manifest.owner_manifest_id = root.manifest.object_id;
    memset(&root.pending_l0, 0, sizeof(root.pending_l0));
    ASSERT_STATUS_OK(evoke_segment_read_root_rotate_l0(&root));
    ASSERT_TRUE(root.pending_l0.segment_id == 11);
    ASSERT_TRUE(root.pending_l0.min_sequence == 9);
    ASSERT_TRUE(root.pending_l0.max_sequence == 12);
    ASSERT_TRUE(root.active_l0.segment_id == 12);
    ASSERT_TRUE(root.active_l0.record_count == 0);
    ASSERT_TRUE(root.next_segment_id == 13);
    ASSERT_TRUE(root.next_sequence == 13);
    ASSERT_TRUE(root.next_document_slot == 7);
    ASSERT_TRUE(root.reusable_document_slot_cursor == 3);
    ASSERT_TRUE(
        evoke_segment_read_root_rotate_l0(&root) == EVOKE_ERR_FORMAT
    );

    {
        evoke_active_l0_frontier active;
        evoke_active_l0_frontier pending;
        evoke_segment_object_ref compaction_ref;
        evoke_segment_object_ref manifest_ref;
        uint64_t active_segment_id;
        uint32_t published_block_high_watermark;

        root.active_l0.min_sequence = 13;
        root.active_l0.max_sequence = 14;
        root.active_l0.payload_bytes = 256;
        root.active_l0.head_block = 40;
        root.active_l0.tail_block = 41;
        root.active_l0.page_count = 2;
        root.active_l0.record_count = 2;
        root.next_sequence = 15;
        active = root.active_l0;
        pending = root.pending_l0;
        initialize_test_object_ref(
            &compaction_ref,
            EVOKE_SEGMENT_OBJECT_MANIFEST,
            root.published_block_high_watermark,
            2,
            root.next_segment_id,
            0x1233
        );
        ASSERT_TRUE(
            evoke_segment_read_root_replace_manifest(
                &root,
                &compaction_ref,
                root.published_block_high_watermark - 1
            ) == EVOKE_ERR_FORMAT
        );
        published_block_high_watermark =
            root.published_block_high_watermark + 2;
        ASSERT_STATUS_OK(evoke_segment_read_root_replace_manifest(
            &root,
            &compaction_ref,
            published_block_high_watermark
        ));
        ASSERT_TRUE(root.root_id == compaction_ref.object_id);
        ASSERT_TRUE(
            root.next_segment_id == compaction_ref.object_id + 1
        );
        ASSERT_TRUE(memcmp(
            &root.pending_l0,
            &pending,
            sizeof(pending)
        ) == 0);
        ASSERT_TRUE(memcmp(
            &root.active_l0,
            &active,
            sizeof(active)
        ) == 0);
        ASSERT_TRUE(root.reusable_document_slot_cursor == 3);

        initialize_test_object_ref(
            &manifest_ref,
            EVOKE_SEGMENT_OBJECT_MANIFEST,
            root.published_block_high_watermark,
            2,
            root.next_segment_id,
            0x1234
        );
        ASSERT_TRUE(
            evoke_segment_read_root_seal_pending(
                &root,
                &manifest_ref,
                root.published_block_high_watermark - 1
            ) == EVOKE_ERR_FORMAT
        );

        published_block_high_watermark =
            root.published_block_high_watermark + 3;
        ASSERT_STATUS_OK(evoke_segment_read_root_seal_pending(
            &root,
            &manifest_ref,
            published_block_high_watermark
        ));
        ASSERT_TRUE(root.root_id == manifest_ref.object_id);
        ASSERT_TRUE(root.next_segment_id == manifest_ref.object_id + 1);
        ASSERT_TRUE(root.pending_l0.segment_id == 0);
        ASSERT_TRUE(memcmp(
            &root.active_l0,
            &active,
            sizeof(active)
        ) == 0);
        ASSERT_TRUE(root.reusable_document_slot_cursor == 3);

        manifest_ref.object_id = root.next_segment_id + 1;
        manifest_ref.owner_manifest_id = manifest_ref.object_id;
        ASSERT_TRUE(
            evoke_segment_read_root_seal_pending(
                &root,
                &manifest_ref,
                root.published_block_high_watermark + 2
            ) == EVOKE_ERR_FORMAT
        );

        active_segment_id = root.active_l0.segment_id;
        memset(&root.active_l0, 0, sizeof(root.active_l0));
        root.active_l0.segment_id = active_segment_id;
        initialize_test_object_ref(
            &manifest_ref,
            EVOKE_SEGMENT_OBJECT_MANIFEST,
            root.published_block_high_watermark,
            2,
            root.next_segment_id,
            0x1235
        );
        ASSERT_STATUS_OK(evoke_segment_read_root_replace_manifest(
            &root,
            &manifest_ref,
            root.published_block_high_watermark + 2
        ));
        ASSERT_TRUE(root.reusable_document_slot_cursor == 0);
    }
}

static void
test_segment_manifest_roundtrip_and_validation(void)
{
    evoke_segment_manifest manifest;
    evoke_segment_manifest restored;
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    size_t size = 0;

    initialize_test_segment_manifest(&manifest);
    manifest.segments[1].semantic_state_count = 1;
    manifest.segments[1].flags |= EVOKE_SEGMENT_FLAG_SEMANTIC;
    evoke_segment_manifest_init(&restored);
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));
    ASSERT_STATUS_OK(evoke_segment_manifest_serialize(
        &manifest,
        &bytes,
        &size
    ));
    ASSERT_TRUE(bytes != NULL);
    ASSERT_TRUE(size > 592);
    ASSERT_STATUS_OK(evoke_segment_manifest_deserialize(
        bytes,
        size,
        &restored
    ));
    ASSERT_TRUE(restored.manifest_id == manifest.manifest_id);
    ASSERT_TRUE(restored.parent_manifest_id == manifest.parent_manifest_id);
    ASSERT_TRUE(
        restored.document_slot_count == manifest.document_slot_count
    );
    ASSERT_TRUE(restored.segment_count == manifest.segment_count);
    ASSERT_TRUE(restored.vocab_size == manifest.vocab_size);
    ASSERT_TRUE(memcmp(
        restored.contract_hash,
        manifest.contract_hash,
        sizeof(manifest.contract_hash)
    ) == 0);
    ASSERT_TRUE(restored.segments[1].segment_id == 2);
    ASSERT_TRUE(restored.segments[1].flags == manifest.segments[1].flags);
    ASSERT_TRUE(
        restored.segments[1].semantic_state_count == 1
    );
    ASSERT_TRUE(
        restored.segments[0].payload_owner_manifest_id ==
        manifest.parent_manifest_id
    );
    ASSERT_TRUE(
        restored.term_directory.object_checksum ==
        manifest.term_directory.object_checksum
    );
    ASSERT_TRUE(
        restored.document_directory.object_checksum ==
        manifest.document_directory.object_checksum
    );
    ASSERT_TRUE(
        restored.lexicon_lookup.object_checksum ==
        manifest.lexicon_lookup.object_checksum
    );
    ASSERT_TRUE(
        restored.prefix_lookup.object_checksum ==
        manifest.prefix_lookup.object_checksum
    );
    ASSERT_TRUE(
        restored.lexicon_hash_seed == manifest.lexicon_hash_seed
    );
    ASSERT_TRUE(
        restored.retired_range_count == manifest.retired_range_count
    );
    ASSERT_TRUE(restored.retired_ranges[0].start_block == 10);
    ASSERT_TRUE(restored.retired_ranges[0].block_count == 2);
    ASSERT_TRUE(restored.retired_ranges[1].start_block == 14);
    ASSERT_TRUE(restored.retired_ranges[1].block_count == 1);
    assert_uint32_array(
        restored.doc_frequencies,
        manifest.doc_frequencies,
        manifest.vocab_size
    );

    mutated = malloc(size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, size);
    mutated[size - 1] ^= 0x1U;
    ASSERT_TRUE(evoke_segment_manifest_deserialize(
        mutated,
        size,
        &restored
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_segment_manifest_deserialize(
        bytes,
        size - 1,
        &restored
    ) == EVOKE_ERR_FORMAT);
    memcpy(mutated, bytes, size);
    mutated[4] = 0xFFU;
    ASSERT_TRUE(evoke_segment_manifest_deserialize(
        mutated,
        size,
        &restored
    ) == EVOKE_ERR_FORMAT);

    manifest.segments[1].min_sequence = 100;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.segments[1].min_sequence = 101;
    manifest.segments[1].start_block = 8;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.segments[1].start_block = 20;
    manifest.impact_statistics_epoch = 6;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.impact_statistics_epoch = 7;
    manifest.segments[1].flags &= ~EVOKE_SEGMENT_FLAG_SEALED;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.segments[1].flags |= EVOKE_SEGMENT_FLAG_SEALED;
    manifest.segments[1].payload_owner_manifest_id =
        manifest.manifest_id + 1;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.segments[1].payload_owner_manifest_id =
        manifest.manifest_id;
    manifest.query_contract.object_id = manifest.parent_manifest_id;
    manifest.query_contract.owner_manifest_id =
        manifest.parent_manifest_id;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));
    manifest.query_contract.object_id++;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.query_contract.object_id =
        manifest.query_contract.owner_manifest_id;
    manifest.query_contract.owner_manifest_id = manifest.manifest_id + 1;
    manifest.query_contract.object_id =
        manifest.query_contract.owner_manifest_id;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.query_contract.object_id = manifest.manifest_id;
    manifest.query_contract.owner_manifest_id = manifest.manifest_id;
    manifest.document_directory.object_id = 77;
    manifest.document_directory.owner_manifest_id =
        manifest.parent_manifest_id;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));
    manifest.document_directory.owner_manifest_id =
        manifest.manifest_id + 1;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.document_directory.owner_manifest_id =
        manifest.parent_manifest_id;
    manifest.flags &=
        ~EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.flags |=
        EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY;
    manifest.document_directory.start_block =
        manifest.segments[0].start_block;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.document_directory.start_block = 56;
    manifest.lexicon_lookup.owner_manifest_id =
        manifest.parent_manifest_id;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));
    manifest.lexicon_lookup.owner_manifest_id = manifest.manifest_id + 1;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.lexicon_lookup.owner_manifest_id = manifest.manifest_id;
    manifest.lexicon_hash_seed = 0;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.lexicon_hash_seed = UINT64_C(0x42C0FFEE12345678);
    manifest.flags &= ~EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.flags |= EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP;
    manifest.lexicon_lookup.start_block =
        manifest.segments[0].start_block;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.lexicon_lookup.start_block = 58;
    manifest.prefix_lookup.owner_manifest_id =
        manifest.parent_manifest_id;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));
    manifest.prefix_lookup.owner_manifest_id = manifest.manifest_id + 1;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.prefix_lookup.owner_manifest_id = manifest.manifest_id;
    manifest.flags &= ~EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.flags |= EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP;
    manifest.prefix_lookup.start_block =
        manifest.segments[0].start_block;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.prefix_lookup.start_block = 60;
    manifest.parent_manifest_id = manifest.manifest_id;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.parent_manifest_id = 8;
    manifest.visible_document_count = 4;
    manifest.doc_frequencies[0] = 5;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));
    manifest.doc_frequencies[0] = 6;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.doc_frequencies[0] = 3;
    manifest.visible_document_count = 6;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.visible_document_count = 5;
    manifest.document_slot_count = (uint64_t) UINT32_MAX + 1;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.document_slot_count = 5;
    manifest.retired_ranges[0].start_block = 1;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.retired_ranges[0].start_block = 10;
    manifest.retired_ranges[1].start_block = 12;
    ASSERT_TRUE(evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT);
    manifest.retired_ranges[1].start_block = 14;

    free(bytes);
    free(mutated);
    evoke_segment_manifest_free(&manifest);
    evoke_segment_manifest_free(&restored);
}

static void
test_cow_segment_manifest_roundtrip_and_validation(void)
{
    evoke_segment_manifest manifest;
    evoke_segment_manifest restored;
    uint8_t *bytes = NULL;
    size_t size = 0;

    initialize_test_segment_manifest(&manifest);
    evoke_segment_manifest_init(&restored);
    manifest.flags &= ~EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    manifest.flags |= EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY;
    manifest.term_directory.object_id = 71;
    manifest.term_directory.owner_manifest_id =
        manifest.parent_manifest_id;
    free(manifest.doc_frequencies);
    manifest.doc_frequencies = NULL;

    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));
    ASSERT_STATUS_OK(evoke_segment_manifest_serialize(
        &manifest,
        &bytes,
        &size
    ));
    ASSERT_TRUE(
        size ==
        592 +
        (size_t) manifest.segment_count * 120 +
        (size_t) manifest.retired_range_count * 8
    );
    ASSERT_STATUS_OK(evoke_segment_manifest_deserialize(
        bytes,
        size,
        &restored
    ));
    ASSERT_TRUE(
        (restored.flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) != 0
    );
    ASSERT_TRUE(restored.doc_frequencies == NULL);
    ASSERT_TRUE(
        restored.term_directory.object_id ==
        manifest.term_directory.object_id
    );
    ASSERT_TRUE(
        restored.term_directory.owner_manifest_id ==
        manifest.parent_manifest_id
    );

    manifest.flags |= EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    manifest.flags &= ~EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    manifest.doc_frequencies = calloc(
        manifest.vocab_size,
        sizeof(*manifest.doc_frequencies)
    );
    ASSERT_TRUE(manifest.doc_frequencies != NULL);
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );
    free(manifest.doc_frequencies);
    manifest.doc_frequencies = NULL;
    manifest.term_directory.owner_manifest_id =
        manifest.manifest_id + 1;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&manifest) == EVOKE_ERR_FORMAT
    );

    free(bytes);
    evoke_segment_manifest_free(&manifest);
    evoke_segment_manifest_free(&restored);
}

static void
test_semantic_accelerator_manifest_lifecycle(void)
{
    evoke_segment_manifest manifest;
    evoke_segment_manifest restored;
    evoke_segment_manifest legacy_restored;
    evoke_segment_manifest descendant;
    evoke_segment_read_root root = {0};
    uint8_t *bytes = NULL;
    size_t size = 0;
    uint64_t before_checksum;
    uint64_t after_checksum;

    initialize_test_segment_manifest(&manifest);
    evoke_segment_manifest_init(&restored);
    evoke_segment_manifest_init(&legacy_restored);
    evoke_segment_manifest_init(&descendant);
    manifest.flags &= ~EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    manifest.flags |= EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY;
    free(manifest.doc_frequencies);
    manifest.doc_frequencies = NULL;
    manifest.flags |= EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR;
    initialize_test_object_ref(
        &manifest.semantic_accelerator_directory,
        EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_DIRECTORY,
        62,
        1,
        manifest.manifest_id,
        manifest.manifest_id
    );
    manifest.semantic_accelerator_max_sequence = manifest.max_sequence;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));
    ASSERT_STATUS_OK(evoke_segment_manifest_authority_checksum(
        &manifest,
        &before_checksum
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_serialize(
        &manifest,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_deserialize(
        bytes,
        size,
        &restored
    ));
    ASSERT_TRUE(evoke_segment_object_ref_equal(
        &manifest.semantic_accelerator_directory,
        &restored.semantic_accelerator_directory
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_authority_checksum(
        &restored,
        &after_checksum
    ));
    ASSERT_TRUE(before_checksum == after_checksum);

    test_write_u16_le(
        bytes + 4,
        EVOKE_SEGMENT_MANIFEST_LEGACY_VERSION
    );
    test_write_u64_le(bytes + 576, 0);
    test_write_u64_le(bytes + 584, 0);
    test_write_u64_le(
        bytes + 584,
        test_segment_manifest_checksum(bytes, size)
    );
    ASSERT_STATUS_OK(evoke_segment_manifest_deserialize(
        bytes,
        size,
        &legacy_restored
    ));
    ASSERT_TRUE(
        legacy_restored.semantic_accelerator_max_sequence == 0
    );
    ASSERT_TRUE(
        evoke_segment_manifest_semantic_accelerator_baseline_sequence(
            &legacy_restored
        ) == legacy_restored.max_sequence
    );

    root.root_id = manifest.manifest_id;
    root.next_segment_id = manifest.manifest_id + 1;
    ASSERT_STATUS_OK(evoke_segment_manifest_build_identity(
        &root,
        &manifest,
        &descendant
    ));
    ASSERT_TRUE(
        (descendant.flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR) != 0
    );
    ASSERT_TRUE(evoke_segment_object_ref_equal(
        &manifest.semantic_accelerator_directory,
        &descendant.semantic_accelerator_directory
    ));
    ASSERT_TRUE(
        evoke_segment_manifest_semantic_accelerator_baseline_sequence(
            &descendant
        ) == manifest.max_sequence
    );
    ASSERT_TRUE(descendant.retired_range_count == 0);
    ASSERT_TRUE(descendant.retired_ranges == NULL);
    ASSERT_STATUS_OK(evoke_segment_manifest_authority_checksum(
        &descendant,
        &after_checksum
    ));
    ASSERT_TRUE(before_checksum == after_checksum);

    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&descendant));
    descendant.semantic_accelerator_directory.owner_manifest_id =
        descendant.manifest_id + 1;
    ASSERT_TRUE(
        evoke_segment_manifest_validate(&descendant) == EVOKE_ERR_FORMAT
    );

    ASSERT_TRUE(evoke_segment_manifest_semantic_accelerator_eligible(
        &root,
        &manifest
    ));
    root.active_l0.record_count = 1;
    ASSERT_TRUE(evoke_segment_manifest_semantic_accelerator_eligible(
        &root,
        &manifest
    ));
    root.active_l0.record_count = 0;
    root.pending_l0.record_count = 1;
    ASSERT_TRUE(evoke_segment_manifest_semantic_accelerator_eligible(
        &root,
        &manifest
    ));

    free(bytes);
    evoke_segment_manifest_free(&descendant);
    evoke_segment_manifest_free(&legacy_restored);
    evoke_segment_manifest_free(&restored);
    evoke_segment_manifest_free(&manifest);
}

static void
test_document_tid_lookup_roundtrip(void)
{
    const uint64_t keys[] = {
        UINT64_C(0x00010002),
        UINT64_C(0x00020001),
        UINT64_C(0x00030004)
    };
    const uint32_t slots[] = {3, 0, 4};
    evoke_document_tid_lookup_view view;
    uint8_t *bytes = NULL;
    size_t size = 0;
    uint64_t key = 0;
    uint32_t slot = 0;

    ASSERT_STATUS_OK(evoke_document_tid_lookup_serialize(
        UINT64_C(0x1020304050607080),
        keys,
        slots,
        3,
        5,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(evoke_document_tid_lookup_open(
        bytes,
        size,
        UINT64_C(0x1020304050607080),
        3,
        5,
        &view
    ));
    ASSERT_TRUE(view.entry_count == 3);
    ASSERT_TRUE(view.document_slot_count == 5);
    ASSERT_STATUS_OK(evoke_document_tid_lookup_entry(
        &view,
        1,
        &key,
        &slot
    ));
    ASSERT_TRUE(key == keys[1] && slot == slots[1]);
    ASSERT_TRUE(evoke_document_tid_lookup_find(&view, keys[2], &slot));
    ASSERT_TRUE(slot == slots[2]);
    ASSERT_TRUE(!evoke_document_tid_lookup_find(
        &view,
        UINT64_C(0x00040001),
        &slot
    ));
    ASSERT_TRUE(evoke_document_tid_lookup_open(
        bytes,
        size,
        UINT64_C(0x1020304050607081),
        3,
        5,
        &view
    ) == EVOKE_ERR_FORMAT);
    bytes[size - 1U] ^= UINT8_C(1);
    ASSERT_TRUE(evoke_document_tid_lookup_open(
        bytes,
        size,
        UINT64_C(0x1020304050607080),
        3,
        5,
        &view
    ) == EVOKE_ERR_FORMAT);
    free(bytes);
}

static void
test_semantic_accelerator_directory_roundtrip(void)
{
    evoke_semantic_accelerator_directory directory;
    evoke_semantic_accelerator_directory forward_only;
    evoke_semantic_accelerator_directory retired;
    evoke_semantic_accelerator_directory restored;
    evoke_semantic_accelerator_directory_summary summary;
    uint8_t *bytes = NULL;
    uint8_t *retired_bytes = NULL;
    size_t forward_offset = 0;
    size_t forward_size = 0;
    size_t retired_forward_offset;
    size_t retired_size;
    size_t size = 0;
    uint32_t index;

    evoke_semantic_accelerator_directory_init(&directory);
    evoke_semantic_accelerator_directory_init(&forward_only);
    evoke_semantic_accelerator_directory_init(&retired);
    evoke_semantic_accelerator_directory_init(&restored);
    directory.source_manifest_id = 41;
    directory.source_authority_checksum = UINT64_C(0x1020304050607080);
    directory.owner_manifest_id = 42;
    directory.document_count = 1000;
    directory.vocab_size = 128;
    directory.term_count = 2;
    directory.forward_chunk_count = 2;
    directory.forward_document_shift = 9;
    directory.builder_policy_id =
        EVOKE_SEMANTIC_ACCELERATOR_CURRENT_POLICY;
    directory.retained_document_cap =
        EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP;
    directory.terms = calloc(directory.term_count, sizeof(*directory.terms));
    directory.forward_chunks = calloc(
        directory.forward_chunk_count,
        sizeof(*directory.forward_chunks)
    );
    directory.forward_term_work = calloc(
        directory.vocab_size,
        sizeof(*directory.forward_term_work)
    );
    directory.forward_row_data_bytes = calloc(
        directory.forward_chunk_count,
        sizeof(*directory.forward_row_data_bytes)
    );
    directory.forward_transpose_fixed_bytes = calloc(
        directory.forward_chunk_count,
        sizeof(*directory.forward_transpose_fixed_bytes)
    );
    directory.forward_row_offsets = calloc(
        directory.document_count + directory.forward_chunk_count,
        sizeof(*directory.forward_row_offsets)
    );
    directory.forward_term_bytes = calloc(
        directory.vocab_size,
        sizeof(*directory.forward_term_bytes)
    );
    directory.forward_bound_term_bytes = calloc(
        directory.vocab_size,
        sizeof(*directory.forward_bound_term_bytes)
    );
    directory.forward_bound_shard_count = 2;
    directory.forward_bound_shards = calloc(
        directory.forward_bound_shard_count,
        sizeof(*directory.forward_bound_shards)
    );
    ASSERT_TRUE(
        directory.terms != NULL && directory.forward_chunks != NULL &&
        directory.forward_term_work != NULL &&
        directory.forward_row_data_bytes != NULL &&
        directory.forward_transpose_fixed_bytes != NULL &&
        directory.forward_row_offsets != NULL &&
        directory.forward_term_bytes != NULL &&
        directory.forward_bound_term_bytes != NULL &&
        directory.forward_bound_shards != NULL
    );
    directory.forward_term_work[7] = 1000;
    directory.forward_term_work[29] = 420;
    directory.forward_row_data_bytes[0] = 32000;
    directory.forward_row_data_bytes[1] = 29000;
    directory.forward_transpose_fixed_bytes[0] = 4096;
    directory.forward_transpose_fixed_bytes[1] = 4000;
    directory.forward_term_bytes[7] = 3008;
    directory.forward_term_bytes[29] = 1268;
    directory.forward_bound_term_bytes[7] = 800;
    directory.forward_bound_term_bytes[29] = 360;
    directory.terms[0].term_id = 7;
    initialize_test_object_ref(
        &directory.terms[0].term_object,
        EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_TERM,
        90,
        3,
        directory.terms[0].term_id + 1,
        UINT64_C(0x107)
    );
    directory.terms[0].term_object.owner_manifest_id =
        directory.owner_manifest_id;
    directory.terms[1].term_id = 29;
    initialize_test_object_ref(
        &directory.terms[1].term_object,
        EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_TERM,
        100,
        2,
        directory.terms[1].term_id + 1,
        UINT64_C(0x129)
    );
    directory.terms[1].term_object.owner_manifest_id =
        directory.owner_manifest_id;
    directory.forward_chunks[0].first_document = 0;
    directory.forward_chunks[0].document_count = 512;
    directory.forward_chunks[0].posting_count = 16384;
    directory.forward_chunks[0].row_data_offset = 4096;
    initialize_test_object_ref(
        &directory.forward_chunks[0].forward_object,
        EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD,
        110,
        8,
        1,
        UINT64_C(0x201)
    );
    directory.forward_chunks[0].forward_object.owner_manifest_id =
        directory.owner_manifest_id;
    directory.forward_chunks[0].forward_object.object_bytes = 36096;
    directory.forward_chunks[1].first_document = 512;
    directory.forward_chunks[1].document_count = 488;
    directory.forward_chunks[1].posting_count = 14640;
    directory.forward_chunks[1].row_data_offset = 4096;
    initialize_test_object_ref(
        &directory.forward_chunks[1].forward_object,
        EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD,
        120,
        8,
        2,
        UINT64_C(0x202)
    );
    directory.forward_chunks[1].forward_object.owner_manifest_id =
        directory.owner_manifest_id;
    directory.forward_chunks[1].forward_object.object_bytes = 33096;
    for (index = 0; index <= 512; index++)
    {
        directory.forward_row_offsets[index] =
            (uint32_t) ((uint64_t) index * 32000U / 512U);
    }
    for (index = 0; index <= 488; index++)
    {
        directory.forward_row_offsets[513U + index] =
            (uint32_t) ((uint64_t) index * 29000U / 488U);
    }
    initialize_test_object_ref(
        &directory.scope_object,
        EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_SCOPE,
        130,
        2,
        1,
        UINT64_C(0x301)
    );
    directory.scope_object.owner_manifest_id = directory.owner_manifest_id;
    initialize_test_object_ref(
        &directory.tid_lookup_object,
        EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_TID_LOOKUP,
        140,
        2,
        1,
        UINT64_C(0x302)
    );
    directory.tid_lookup_object.owner_manifest_id =
        directory.owner_manifest_id;
    initialize_test_object_ref(
        &directory.forward_bound_shards[0],
        EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD_BOUND,
        150,
        2,
        1,
        UINT64_C(0x401)
    );
    directory.forward_bound_shards[0].owner_manifest_id =
        directory.owner_manifest_id;
    initialize_test_object_ref(
        &directory.forward_bound_shards[1],
        EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD_BOUND,
        160,
        2,
        2,
        UINT64_C(0x402)
    );
    directory.forward_bound_shards[1].owner_manifest_id =
        directory.owner_manifest_id;

    ASSERT_STATUS_OK(evoke_semantic_accelerator_directory_validate(
        &directory
    ));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_directory_serialize(
        &directory,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(
        evoke_semantic_accelerator_directory_summary_deserialize(
            bytes,
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE,
            &summary
        )
    );
    ASSERT_TRUE(summary.source_manifest_id == 41);
    ASSERT_TRUE(summary.owner_manifest_id == 42);
    ASSERT_TRUE(summary.document_count == 1000);
    ASSERT_TRUE(summary.term_count == 2);
    ASSERT_TRUE(summary.forward_chunk_count == 2);
    ASSERT_TRUE(summary.forward_term_work_count == 128);
    ASSERT_TRUE(summary.forward_chunk_cost_count == 2);
    ASSERT_TRUE(summary.forward_row_offset_count == 1002);
    ASSERT_TRUE(summary.forward_term_bytes_count == 128);
    ASSERT_TRUE(summary.forward_bound_term_bytes_count == 128);
    ASSERT_TRUE(summary.forward_bound_shard_count == 2);
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_summary_has_scope(&summary)
    );
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_summary_has_tid_lookup(
            &summary
        )
    );
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_summary_is_current(&summary)
    );
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_summary_has_complete_forward(
            &summary
        )
    );
    ASSERT_STATUS_OK(
        evoke_semantic_accelerator_directory_forward_slice(
            bytes,
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE,
            &summary,
            &forward_offset,
            &forward_size
        )
    );
    ASSERT_STATUS_OK(
        evoke_semantic_accelerator_forward_directory_deserialize(
            &summary,
            bytes + forward_offset,
            forward_size,
            &forward_only
        )
    );
    ASSERT_TRUE(forward_only.terms == NULL);
    ASSERT_TRUE(forward_only.forward_chunk_count == 2);
    ASSERT_TRUE(forward_only.forward_term_work[7] == 1000);
    ASSERT_TRUE(forward_only.forward_term_work[29] == 420);
    ASSERT_TRUE(forward_only.forward_row_data_bytes[0] == 32000);
    ASSERT_TRUE(forward_only.forward_transpose_fixed_bytes[1] == 4000);
    ASSERT_TRUE(forward_only.forward_term_bytes[7] == 3008);
    ASSERT_TRUE(forward_only.forward_bound_term_bytes[29] == 360);
    ASSERT_TRUE(forward_only.forward_bound_shard_count == 2);
    ASSERT_TRUE(forward_only.forward_bound_shards != NULL);
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_find_forward_bound(
            &forward_only,
            127
        ) == &forward_only.forward_bound_shards[1]
    );
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_find_forward(
            &forward_only,
            700
        ) == &forward_only.forward_chunks[1]
    );
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_summary_deserialize(
            bytes,
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE - 1,
            &summary
        ) == EVOKE_ERR_FORMAT
    );
    ASSERT_STATUS_OK(evoke_semantic_accelerator_directory_deserialize(
        bytes,
        size,
        &restored
    ));
    ASSERT_TRUE(restored.source_manifest_id == 41);
    ASSERT_TRUE(restored.owner_manifest_id == 42);
    ASSERT_TRUE(restored.term_count == 2);
    ASSERT_TRUE(restored.forward_chunk_count == 2);
    ASSERT_TRUE(restored.forward_chunks[0].posting_count == 16384);
    ASSERT_TRUE(restored.forward_chunks[1].posting_count == 14640);
    ASSERT_TRUE(restored.forward_term_work[7] == 1000);
    ASSERT_TRUE(restored.forward_term_work[29] == 420);
    ASSERT_TRUE(restored.forward_row_data_bytes[1] == 29000);
    ASSERT_TRUE(restored.forward_transpose_fixed_bytes[0] == 4096);
    ASSERT_TRUE(restored.forward_term_bytes[29] == 1268);
    ASSERT_TRUE(restored.forward_bound_term_bytes[7] == 800);
    ASSERT_TRUE(restored.forward_bound_shard_count == 2);
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_find_forward_bound(
            &restored,
            127
        ) == &restored.forward_bound_shards[1]
    );
    ASSERT_TRUE(evoke_semantic_accelerator_directory_has_scope(&restored));
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_has_tid_lookup(&restored)
    );
    ASSERT_TRUE(evoke_semantic_accelerator_directory_is_current(&restored));
    ASSERT_TRUE(
        restored.retained_document_cap ==
        EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP
    );
    ASSERT_TRUE(evoke_semantic_accelerator_directory_has_complete_forward(
        &restored
    ));
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_find(&restored, 29) != NULL
    );
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_find(&restored, 8) == NULL
    );
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_find_forward(
            &restored,
            700
        ) == &restored.forward_chunks[1]
    );

    retired_size = EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
        (size_t) directory.term_count * 56U +
        (size_t) directory.forward_chunk_count * 56U;
    retired_bytes = calloc(retired_size, 1U);
    ASSERT_TRUE(retired_bytes != NULL);
    retired_forward_offset =
        EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
        (size_t) directory.term_count * 56U;
    memcpy(
        retired_bytes,
        bytes,
        retired_forward_offset
    );
    test_write_u16_le(retired_bytes + 4, 5);
    test_write_u16_le(retired_bytes + 10, 56);
    test_write_u64_le(retired_bytes + 56, retired_size);
    for (index = 0; index < directory.forward_chunk_count; index++)
    {
        const uint8_t *source =
            bytes + retired_forward_offset + (size_t) index * 64U;
        uint8_t *target =
            retired_bytes + retired_forward_offset + (size_t) index * 56U;

        memcpy(target, source, 8U);
        memcpy(target + 8U, source + 16U, 48U);
    }
    test_write_u64_le(retired_bytes + 64, 0);
    test_write_u64_le(
        retired_bytes + 64,
        test_semantic_accelerator_directory_checksum(
            retired_bytes,
            retired_size
        )
    );
    ASSERT_STATUS_OK(
        evoke_semantic_accelerator_directory_summary_deserialize(
            retired_bytes,
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE,
            &summary
        )
    );
    ASSERT_TRUE(
        !evoke_semantic_accelerator_directory_summary_format_is_current(
            &summary
        )
    );
    ASSERT_TRUE(
        !evoke_semantic_accelerator_directory_summary_is_current(&summary)
    );
    ASSERT_TRUE(evoke_semantic_accelerator_directory_deserialize(
        retired_bytes,
        retired_size,
        &retired
    ) == EVOKE_ERR_FORMAT);
    ASSERT_STATUS_OK(
        evoke_semantic_accelerator_directory_deserialize_retired(
            retired_bytes,
            retired_size,
            &retired
        )
    );
    ASSERT_TRUE(retired.forward_chunk_count == 2);
    ASSERT_TRUE(retired.forward_chunks[0].posting_count == 0);
    ASSERT_TRUE(memcmp(
        &retired.forward_chunks[1].forward_object,
        &directory.forward_chunks[1].forward_object,
        sizeof(directory.forward_chunks[1].forward_object)
    ) == 0);

    free(retired_bytes);
    retired_size = EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
        (size_t) directory.term_count * 56U +
        (size_t) directory.forward_chunk_count * 64U;
    retired_bytes = calloc(retired_size, 1U);
    ASSERT_TRUE(retired_bytes != NULL);
    memcpy(retired_bytes, bytes, retired_size);
    test_write_u16_le(retired_bytes + 4, 6);
    test_write_u64_le(retired_bytes + 56, retired_size);
    for (index = 0; index < directory.forward_chunk_count; index++)
    {
        write_u32_le(
            retired_bytes + retired_forward_offset +
                (size_t) index * 64U + 12U,
            0
        );
    }
    test_write_u64_le(retired_bytes + 64, 0);
    test_write_u64_le(
        retired_bytes + 64,
        test_semantic_accelerator_directory_checksum(
            retired_bytes,
            retired_size
        )
    );
    ASSERT_STATUS_OK(
        evoke_semantic_accelerator_directory_summary_deserialize(
            retired_bytes,
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE,
            &summary
        )
    );
    ASSERT_TRUE(summary.format_version == 6);
    ASSERT_TRUE(summary.forward_entry_size == 64);
    ASSERT_TRUE(summary.forward_term_work_count == 0);
    ASSERT_TRUE(
        !evoke_semantic_accelerator_directory_summary_format_is_current(
            &summary
        )
    );
    ASSERT_TRUE(evoke_semantic_accelerator_directory_deserialize(
        retired_bytes,
        retired_size,
        &retired
    ) == EVOKE_ERR_FORMAT);
    ASSERT_STATUS_OK(
        evoke_semantic_accelerator_directory_deserialize_retired(
            retired_bytes,
            retired_size,
            &retired
        )
    );
    ASSERT_TRUE(retired.forward_chunk_count == 2);
    ASSERT_TRUE(retired.forward_chunks[0].posting_count == 16384);
    ASSERT_TRUE(retired.forward_chunks[1].posting_count == 14640);
    ASSERT_TRUE(retired.forward_term_work == NULL);

    free(retired_bytes);
    retired_size = EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
        (size_t) directory.term_count * 56U +
        (size_t) directory.forward_chunk_count * 64U +
        (size_t) directory.vocab_size * sizeof(uint64_t);
    retired_bytes = malloc(retired_size);
    ASSERT_TRUE(retired_bytes != NULL);
    memcpy(retired_bytes, bytes, retired_size);
    test_write_u16_le(retired_bytes + 4, 7);
    test_write_u64_le(retired_bytes + 56, retired_size);
    for (index = 0; index < directory.forward_chunk_count; index++)
    {
        write_u32_le(
            retired_bytes + retired_forward_offset +
                (size_t) index * 64U + 12U,
            0
        );
    }
    test_write_u64_le(retired_bytes + 64, 0);
    test_write_u64_le(
        retired_bytes + 64,
        test_semantic_accelerator_directory_checksum(
            retired_bytes,
            retired_size
        )
    );
    ASSERT_STATUS_OK(
        evoke_semantic_accelerator_directory_deserialize_retired(
            retired_bytes,
            retired_size,
            &retired
        )
    );
    ASSERT_TRUE(retired.forward_term_work[7] == 1000);
    ASSERT_TRUE(retired.forward_bound_shards == NULL);

    free(retired_bytes);
    retired_size = EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
        (size_t) directory.term_count * 56U +
        (size_t) directory.forward_chunk_count * 64U +
        (size_t) directory.vocab_size * sizeof(uint64_t) +
        (size_t) directory.forward_bound_shard_count * 48U;
    retired_bytes = malloc(retired_size);
    ASSERT_TRUE(retired_bytes != NULL);
    retired_forward_offset =
        EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
        (size_t) directory.term_count * 56U +
        (size_t) directory.forward_chunk_count * 64U +
        (size_t) directory.vocab_size * sizeof(uint64_t);
    memcpy(retired_bytes, bytes, retired_forward_offset);
    memcpy(
        retired_bytes + retired_forward_offset,
        bytes + size -
            (size_t) directory.forward_bound_shard_count * 48U,
        (size_t) directory.forward_bound_shard_count * 48U
    );
    test_write_u16_le(retired_bytes + 4, 8);
    test_write_u64_le(retired_bytes + 56, retired_size);
    for (index = 0; index < directory.forward_chunk_count; index++)
    {
        write_u32_le(
            retired_bytes +
                EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
                (size_t) directory.term_count * 56U +
                (size_t) index * 64U + 12U,
            0
        );
    }
    test_write_u64_le(retired_bytes + 64, 0);
    test_write_u64_le(
        retired_bytes + 64,
        test_semantic_accelerator_directory_checksum(
            retired_bytes,
            retired_size
        )
    );
    ASSERT_STATUS_OK(
        evoke_semantic_accelerator_directory_deserialize_retired(
            retired_bytes,
            retired_size,
            &retired
        )
    );
    ASSERT_TRUE(retired.forward_term_work[29] == 420);
    ASSERT_TRUE(retired.forward_bound_shard_count == 2);
    ASSERT_TRUE(
        evoke_semantic_accelerator_directory_find_forward_bound(
            &retired,
            127
        ) == &retired.forward_bound_shards[1]
    );
    ASSERT_TRUE(retired.forward_row_data_bytes == NULL);
    ASSERT_TRUE(retired.forward_term_bytes == NULL);

    free(retired_bytes);
    {
        size_t term_bytes = (size_t) directory.term_count * 56U;
        size_t forward_bytes =
            (size_t) directory.forward_chunk_count * 64U;
        size_t work_bytes =
            (size_t) directory.vocab_size * sizeof(uint64_t);
        size_t chunk_cost_bytes =
            (size_t) directory.forward_chunk_count * 2U * sizeof(uint64_t);
        size_t row_offset_bytes =
            (size_t) (directory.document_count +
                directory.forward_chunk_count) * sizeof(uint32_t);
        size_t metric_bytes = work_bytes * 2U;
        size_t bound_ref_bytes =
            (size_t) directory.forward_bound_shard_count * 48U;
        size_t prefix_size =
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE + term_bytes +
            forward_bytes + work_bytes + chunk_cost_bytes;

        retired_size = prefix_size + metric_bytes + bound_ref_bytes;
        retired_bytes = malloc(retired_size);
        ASSERT_TRUE(retired_bytes != NULL);
        memcpy(retired_bytes, bytes, prefix_size);
        memcpy(
            retired_bytes + prefix_size,
            bytes + prefix_size + row_offset_bytes,
            metric_bytes + bound_ref_bytes
        );
        for (index = 0; index < directory.forward_chunk_count; index++)
        {
            write_u32_le(
                retired_bytes +
                    EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
                    term_bytes + (size_t) index * 64U + 12U,
                0
            );
        }
    }
    test_write_u16_le(retired_bytes + 4, 9);
    test_write_u64_le(retired_bytes + 56, retired_size);
    test_write_u64_le(retired_bytes + 64, 0);
    test_write_u64_le(
        retired_bytes + 64,
        test_semantic_accelerator_directory_checksum(
            retired_bytes,
            retired_size
        )
    );
    ASSERT_STATUS_OK(
        evoke_semantic_accelerator_directory_deserialize_retired(
            retired_bytes,
            retired_size,
            &retired
        )
    );
    ASSERT_TRUE(retired.forward_bound_shard_count == 2);
    ASSERT_TRUE(retired.forward_row_offsets == NULL);
    ASSERT_TRUE(retired.forward_row_data_bytes == NULL);

    test_write_u16_le(bytes + 4, 4);
    test_write_u64_le(bytes + 64, 0);
    test_write_u64_le(
        bytes + 64,
        test_semantic_accelerator_directory_checksum(bytes, size)
    );
    ASSERT_TRUE(evoke_semantic_accelerator_directory_deserialize(
        bytes,
        size,
        &restored
    ) == EVOKE_ERR_FORMAT);

    test_write_u16_le(bytes + 4, 10);
    test_write_u64_le(bytes + 64, 0);
    test_write_u64_le(
        bytes + 64,
        test_semantic_accelerator_directory_checksum(bytes, size)
    );
    bytes[size - 1] ^= 0x01U;
    ASSERT_TRUE(evoke_semantic_accelerator_directory_deserialize(
        bytes,
        size,
        &restored
    ) == EVOKE_ERR_FORMAT);

    free(retired_bytes);
    free(bytes);
    evoke_semantic_accelerator_directory_free(&forward_only);
    evoke_semantic_accelerator_directory_free(&retired);
    evoke_semantic_accelerator_directory_free(&restored);
    evoke_semantic_accelerator_directory_free(&directory);
}

static void
test_scope_roundtrip_and_validation(void)
{
    const uint32_t date_documents[] = {0, 2};
    const uint32_t ai_documents[] = {1};
    const uint32_t lg_documents[] = {0, 2};
    const evoke_scope_column columns[] = {
        {
            .index_attribute = 3,
            .heap_attribute = 3,
            .type_oid = 23,
            .name = (const uint8_t *) "publish_date",
            .name_size = 12,
            .first_value = 0,
            .value_count = 1
        },
        {
            .index_attribute = 4,
            .heap_attribute = 4,
            .type_oid = 1009,
            .element_type_oid = 25,
            .type_modifier = -1,
            .name = (const uint8_t *) "categories",
            .name_size = 10,
            .first_value = 1,
            .value_count = 2
        }
    };
    evoke_scope_value values[] = {
        {
            .column_index = 0,
            .kind = EVOKE_SCOPE_VALUE_SCALAR,
            .value = (const uint8_t *) "20260400",
            .value_size = 8,
            .documents = date_documents,
            .document_count = 2
        },
        {
            .column_index = 1,
            .kind = EVOKE_SCOPE_VALUE_ARRAY_ELEMENT,
            .value = (const uint8_t *) "cs.AI",
            .value_size = 5,
            .documents = ai_documents,
            .document_count = 1
        },
        {
            .column_index = 1,
            .kind = EVOKE_SCOPE_VALUE_ARRAY_ELEMENT,
            .value = (const uint8_t *) "cs.LG",
            .value_size = 5,
            .documents = lg_documents,
            .document_count = 2
        }
    };
    const uint64_t authority = UINT64_C(0x1020304050607080);
    evoke_scope_header header;
    evoke_scope_column_entry column;
    evoke_scope_value_entry value;
    evoke_scope_writer bounded_writer;
    uint8_t *bytes = NULL;
    uint8_t *corrupted = NULL;
    uint8_t *stale = NULL;
    size_t size = 0;

    ASSERT_STATUS_OK(evoke_scope_serialize(
        authority,
        3,
        columns,
        2,
        values,
        3,
        &bytes,
        &size
    ));
    ASSERT_TRUE(size > EVOKE_SCOPE_HEADER_SIZE);
    evoke_scope_writer_init(&bounded_writer);
    ASSERT_TRUE(evoke_scope_writer_begin(
        &bounded_writer,
        authority,
        3,
        columns,
        2,
        3,
        18,
        5,
        size - 1U
    ) == EVOKE_ERR_NOMEM);
    ASSERT_TRUE(bounded_writer.bytes == NULL);
    ASSERT_TRUE(bounded_writer.total_size == size);
    evoke_scope_writer_free(&bounded_writer);
    ASSERT_STATUS_OK(evoke_scope_validate(bytes, size, authority));
    ASSERT_STATUS_OK(evoke_scope_header_deserialize(
        bytes,
        EVOKE_SCOPE_HEADER_SIZE,
        size,
        authority,
        &header
    ));
    ASSERT_TRUE(header.document_count == 3);
    ASSERT_TRUE(header.column_count == 2);
    ASSERT_TRUE(header.value_count == 3);
    ASSERT_TRUE(header.version == EVOKE_SCOPE_CURRENT_VERSION);
    ASSERT_TRUE(evoke_scope_header_is_current(&header));
    ASSERT_TRUE(header.gram_block_values == EVOKE_SCOPE_GRAM_BLOCK_VALUES);
    ASSERT_TRUE(header.gram_filter_bytes == EVOKE_SCOPE_GRAM_FILTER_BYTES);
    ASSERT_TRUE(header.gram_filter_offset ==
        header.dictionary_offset + header.dictionary_size);
    ASSERT_TRUE(header.gram_filter_size == EVOKE_SCOPE_GRAM_FILTER_BYTES);
    ASSERT_TRUE(header.postings_offset ==
        header.gram_filter_offset + header.gram_filter_size);
    ASSERT_TRUE(header.total_size == size);
    {
        uint32_t bit = evoke_scope_ascii_gram_bit('s', '.', 'a');

    ASSERT_TRUE((bytes[header.gram_filter_offset + bit / 8U] &
            (uint8_t) (1U << (bit % 8U))) != 0);
    }
    free(bytes);
    bytes = NULL;
    values[2].value = (const uint8_t *) "\xe2\x84\xaa";
    values[2].value_size = 3;
    ASSERT_STATUS_OK(evoke_scope_serialize(
        authority,
        3,
        columns,
        2,
        values,
        3,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(evoke_scope_header_deserialize(
        bytes,
        EVOKE_SCOPE_HEADER_SIZE,
        size,
        authority,
        &header
    ));
    for (uint32_t byte = 0; byte < EVOKE_SCOPE_GRAM_FILTER_BYTES; byte++)
    {
        ASSERT_TRUE(bytes[header.gram_filter_offset + byte] == UINT8_MAX);
    }
    free(bytes);
    bytes = NULL;
    values[2].gram_value = (const uint8_t *) "kelvin";
    values[2].gram_value_size = 6;
    ASSERT_STATUS_OK(evoke_scope_serialize(
        authority,
        3,
        columns,
        2,
        values,
        3,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(evoke_scope_header_deserialize(
        bytes,
        EVOKE_SCOPE_HEADER_SIZE,
        size,
        authority,
        &header
    ));
    {
        uint32_t bit = evoke_scope_ascii_gram_bit('k', 'e', 'l');
        bool unsaturated = false;

        ASSERT_TRUE((bytes[header.gram_filter_offset + bit / 8U] &
            (uint8_t) (1U << (bit % 8U))) != 0);
        for (uint32_t byte = 0;
             byte < EVOKE_SCOPE_GRAM_FILTER_BYTES;
             byte++)
        {
            if (bytes[header.gram_filter_offset + byte] != UINT8_MAX)
            {
                unsaturated = true;
                break;
            }
        }
        ASSERT_TRUE(unsaturated);
    }
    values[2].gram_value = NULL;
    values[2].gram_value_size = 0;
    values[2].value = (const uint8_t *) "cs.LG";
    values[2].value_size = 5;
    for (uint16_t retired_version = 2; retired_version <= 5;
         retired_version++)
    {
        stale = malloc(size);
        ASSERT_TRUE(stale != NULL);
        memcpy(stale, bytes, size);
        test_write_u16_le(stale + 4, retired_version);
        test_write_u64_le(stale + 72, test_scope_checksum(stale, size));
        ASSERT_TRUE(
            evoke_scope_validate(stale, size, authority) == EVOKE_ERR_FORMAT
        );
        ASSERT_TRUE(evoke_scope_header_deserialize(
            stale,
            EVOKE_SCOPE_HEADER_SIZE,
            size,
            authority,
            &header
        ) == EVOKE_ERR_FORMAT);
        free(stale);
        stale = NULL;
    }
    ASSERT_STATUS_OK(evoke_scope_header_deserialize(
        bytes,
        EVOKE_SCOPE_HEADER_SIZE,
        size,
        authority,
        &header
    ));
    ASSERT_STATUS_OK(evoke_scope_column_entry_deserialize(
        bytes + EVOKE_SCOPE_HEADER_SIZE,
        2 * EVOKE_SCOPE_COLUMN_ENTRY_SIZE,
        &header,
        1,
        &column
    ));
    ASSERT_TRUE(column.heap_attribute == 4);
    ASSERT_TRUE(column.element_type_oid == 25);
    ASSERT_STATUS_OK(evoke_scope_value_entry_deserialize(
        bytes + EVOKE_SCOPE_HEADER_SIZE +
            2 * EVOKE_SCOPE_COLUMN_ENTRY_SIZE,
        3 * EVOKE_SCOPE_VALUE_ENTRY_SIZE,
        &header,
        2,
        &value
    ));
    ASSERT_TRUE(value.column_index == 1);
    ASSERT_TRUE(value.document_count == 2);
    ASSERT_TRUE((value.gram_mask &
        (UINT32_C(1) <<
            (evoke_scope_ascii_gram_bit('k', 'e', 'l') % 32U))) != 0);
    ASSERT_TRUE(test_read_u32_le(
        bytes + header.postings_offset + value.postings_offset
    ) == 0);
    ASSERT_TRUE(test_read_u32_le(
        bytes + header.postings_offset + value.postings_offset + 4
    ) == 2);

    corrupted = malloc(size);
    ASSERT_TRUE(corrupted != NULL);
    memcpy(corrupted, bytes, size);
    corrupted[size - 1] ^= 1U;
    ASSERT_TRUE(
        evoke_scope_validate(corrupted, size, authority) == EVOKE_ERR_FORMAT
    );
    free(corrupted);
    corrupted = NULL;

    values[1].value = (const uint8_t *) "cs.ZZ";
    values[2].value = (const uint8_t *) "cs.AA";
    ASSERT_TRUE(evoke_scope_serialize(
        authority,
        3,
        columns,
        2,
        values,
        3,
        &corrupted,
        &size
    ) == EVOKE_ERR_FORMAT);

    free(bytes);
}

static void
test_scope_ascii_ilike_contains(void)
{
    static const uint8_t pattern[] = "%Cancer%";
    static const uint8_t value[] = "Lung CANCER Biology";
    static const uint8_t miss[] = "cardiology";
    static const uint8_t non_ascii[] = {'x', UINT8_C(0xe2), UINT8_C(0x84),
        UINT8_C(0xaa)};
    const uint8_t *literal = NULL;
    size_t literal_size = 0;
    bool matches = false;

    ASSERT_TRUE(evoke_scope_ascii_ilike_contains_pattern(
        pattern,
        sizeof(pattern) - 1U,
        &literal,
        &literal_size
    ));
    ASSERT_TRUE(literal_size == 6);
    ASSERT_TRUE(memcmp(literal, "Cancer", literal_size) == 0);
    ASSERT_TRUE(evoke_scope_ascii_case_insensitive_contains(
        value,
        sizeof(value) - 1U,
        literal,
        literal_size,
        &matches
    ));
    ASSERT_TRUE(matches);
    ASSERT_TRUE(evoke_scope_ascii_case_insensitive_contains(
        miss,
        sizeof(miss) - 1U,
        literal,
        literal_size,
        &matches
    ));
    ASSERT_TRUE(!matches);
    ASSERT_TRUE(!evoke_scope_ascii_case_insensitive_contains(
        non_ascii,
        sizeof(non_ascii),
        literal,
        literal_size,
        &matches
    ));
    ASSERT_TRUE(!evoke_scope_ascii_ilike_contains_pattern(
        (const uint8_t *) "Cancer%",
        7,
        &literal,
        &literal_size
    ));
    ASSERT_TRUE(!evoke_scope_ascii_ilike_contains_pattern(
        (const uint8_t *) "%Can_er%",
        8,
        &literal,
        &literal_size
    ));
    ASSERT_TRUE(!evoke_scope_ascii_ilike_contains_pattern(
        (const uint8_t *) "%Can\\%er%",
        9,
        &literal,
        &literal_size
    ));
    ASSERT_TRUE(!evoke_scope_ascii_ilike_contains_pattern(
        (const uint8_t *) "%Can%er%",
        8,
        &literal,
        &literal_size
    ));
    {
        static const uint8_t non_ascii_pattern[] = {
            '%', 'c', UINT8_C(0xc3), UINT8_C(0xa9), '%'
        };

        ASSERT_TRUE(!evoke_scope_ascii_ilike_contains_pattern(
            non_ascii_pattern,
            sizeof(non_ascii_pattern),
            &literal,
            &literal_size
        ));
    }
    ASSERT_TRUE(evoke_scope_ascii_ilike_contains_pattern(
        (const uint8_t *) "%%",
        2,
        &literal,
        &literal_size
    ));
    ASSERT_TRUE(literal_size == 0);
}

static void
test_semantic_forward_chunk_roundtrip(void)
{
    const uint32_t row_offsets[] = {0, 3, 4, 6};
    const uint32_t term_ids[] = {2, 7, 7, 7, 1, 9};
    const double contributions[] = {
        0.5,
        1.0,
        -0.25,
        2.0,
        3.0,
        4.0
    };
    const uint8_t operations[] = {
        EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT,
        EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT,
        EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT,
        EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT,
        EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT,
        EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT
    };
    const uint32_t query_ids[] = {1, 7, 9};
    const float query_weights[] = {1.0f, 2.0f, 0.5f};
    const uint32_t ordered_query_ids[] = {7, 2, 7};
    const float ordered_query_weights[] = {2.0f, 3.0f, -1.0f};
    evoke_semantic_forward_chunk chunk;
    evoke_semantic_forward_chunk restored;
    evoke_semantic_forward_header header;
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    uint8_t allowed_document_bitmap[] = {
        (uint8_t) ((UINT8_C(1) << 0) | (UINT8_C(1) << 2))
    };
    size_t size = 0;
    float score = 0.0f;
    float transposed_scores[3] = {0};
    uint32_t serialized_posting_count = 0;
    uint64_t transposed_postings_examined = 0;
    uint64_t transposed_bytes_read = 0;

    evoke_semantic_forward_chunk_init(&chunk);
    evoke_semantic_forward_chunk_init(&restored);
    chunk.source_authority_checksum = UINT64_C(0x1122334455667788);
    chunk.first_document = 100;
    chunk.document_count = 3;
    chunk.posting_count = 6;
    chunk.vocab_size = 10;
    chunk.row_offsets = (uint32_t *) row_offsets;
    chunk.term_ids = (uint32_t *) term_ids;
    chunk.operations = (uint8_t *) operations;
    chunk.contributions = (double *) contributions;
    ASSERT_STATUS_OK(evoke_semantic_forward_chunk_serialize(
        &chunk,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(evoke_semantic_forward_header_deserialize(
        bytes,
        evoke_semantic_forward_header_size(),
        size,
        chunk.source_authority_checksum,
        &header
    ));
    ASSERT_TRUE(header.first_document == chunk.first_document);
    ASSERT_TRUE(header.document_count == chunk.document_count);
    ASSERT_TRUE(header.posting_count == chunk.posting_count);
    ASSERT_TRUE(header.vocab_size == chunk.vocab_size);
    ASSERT_TRUE(
        header.format_version == EVOKE_SEMANTIC_FORWARD_TRANSPOSE_VERSION
    );
    ASSERT_TRUE(header.transpose_sparse_posting_count == 6);
    ASSERT_TRUE(header.transpose_dense_term_count == 0);
    ASSERT_TRUE(
        header.row_offsets_offset == evoke_semantic_forward_header_size()
    );
    ASSERT_TRUE(
        header.row_data_offset == header.row_offsets_offset +
            ((size_t) chunk.document_count + 1U) * sizeof(uint32_t)
    );
    ASSERT_TRUE(header.row_data_end > header.row_data_offset);
    ASSERT_TRUE(header.transpose_scales_offset == header.row_data_end);
    ASSERT_TRUE(
        header.transpose_postings_offset >
            header.transpose_term_offsets_offset
    );
    ASSERT_TRUE(header.quantization_bits == 8);
    ASSERT_TRUE(size > 158 && size < 256);
    ASSERT_STATUS_OK(evoke_semantic_forward_score_transposed_sorted(
        bytes,
        size,
        chunk.source_authority_checksum,
        NULL,
        0,
        query_ids,
        query_weights,
        3,
        transposed_scores,
        3,
        &transposed_postings_examined,
        &transposed_bytes_read
    ));
    ASSERT_TRUE(transposed_postings_examined == 5);
    ASSERT_TRUE(transposed_bytes_read > 0 && transposed_bytes_read < size);
    for (uint32_t document = 0; document < chunk.document_count; document++)
    {
        uint32_t row_start = test_read_u32_le(
            bytes + header.row_offsets_offset +
                (size_t) document * sizeof(uint32_t)
        );
        uint32_t row_end = test_read_u32_le(
            bytes + header.row_offsets_offset +
                ((size_t) document + 1U) * sizeof(uint32_t)
        );

        ASSERT_STATUS_OK(evoke_semantic_forward_score_serialized_row_sorted(
            bytes + header.row_data_offset + row_start,
            (size_t) row_end - row_start,
            query_ids,
            query_weights,
            3,
            &score,
            &serialized_posting_count
        ));
        ASSERT_TRUE(memcmp(
            &score,
            &transposed_scores[document],
            sizeof(score)
        ) == 0);
    }
    ASSERT_STATUS_OK(evoke_semantic_forward_score_transposed_sorted(
        bytes,
        size,
        chunk.source_authority_checksum,
        allowed_document_bitmap,
        sizeof(allowed_document_bitmap),
        query_ids,
        query_weights,
        3,
        transposed_scores,
        3,
        &transposed_postings_examined,
        &transposed_bytes_read
    ));
    ASSERT_TRUE(transposed_scores[1] == 0.0f);
    ASSERT_STATUS_OK(evoke_semantic_forward_score_serialized_row_sorted(
        bytes + header.row_data_offset +
            test_read_u32_le(bytes + header.row_offsets_offset),
        test_read_u32_le(
            bytes + header.row_offsets_offset + sizeof(uint32_t)
        ) - test_read_u32_le(bytes + header.row_offsets_offset),
        query_ids,
        query_weights,
        3,
        &score,
        &serialized_posting_count
    ));
    ASSERT_TRUE(serialized_posting_count == 3);
    ASSERT_TRUE(fabsf(score - 1.5f) < 1e-2f);
    ASSERT_STATUS_OK(evoke_semantic_forward_chunk_deserialize(
        bytes,
        size,
        chunk.source_authority_checksum,
        &restored
    ));
    ASSERT_STATUS_OK(evoke_semantic_forward_chunk_score(
        &restored,
        100,
        query_ids,
        query_weights,
        3,
        &score
    ));
    ASSERT_TRUE(fabsf(score - 1.5f) < 1e-2f);
    ASSERT_STATUS_OK(evoke_semantic_forward_chunk_score_sorted(
        &restored,
        100,
        query_ids,
        query_weights,
        3,
        &score
    ));
    ASSERT_TRUE(fabsf(score - 1.5f) < 1e-2f);
    ASSERT_STATUS_OK(evoke_semantic_forward_chunk_score(
        &restored,
        101,
        query_ids,
        query_weights,
        3,
        &score
    ));
    ASSERT_TRUE(fabsf(score - 4.0f) < 1e-2f);
    ASSERT_STATUS_OK(evoke_semantic_forward_chunk_score(
        &restored,
        102,
        query_ids,
        query_weights,
        3,
        &score
    ));
    ASSERT_TRUE(fabsf(score - 5.0f) < 1e-2f);
    ASSERT_STATUS_OK(evoke_semantic_forward_chunk_score(
        &restored,
        100,
        ordered_query_ids,
        ordered_query_weights,
        3,
        &score
    ));
    ASSERT_TRUE(fabsf(score - 2.25f) < 1e-2f);
    ASSERT_STATUS_OK(evoke_semantic_forward_score_row_sorted(
        restored.term_ids,
        restored.operations,
        restored.contributions,
        restored.row_offsets[1] - restored.row_offsets[0],
        query_ids,
        query_weights,
        3,
        &score
    ));
    ASSERT_TRUE(fabsf(score - 1.5f) < 1e-2f);
    ASSERT_STATUS_OK(evoke_semantic_forward_score_row(
        restored.term_ids,
        restored.operations,
        restored.contributions,
        restored.row_offsets[1] - restored.row_offsets[0],
        ordered_query_ids,
        ordered_query_weights,
        3,
        &score
    ));
    ASSERT_TRUE(fabsf(score - 2.25f) < 1e-2f);
    mutated = malloc(size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, size);
    mutated[
        header.transpose_postings_offset + sizeof(uint16_t)
    ] ^= UINT8_C(0x01);
    test_write_u64_le(mutated + 40, 0);
    test_write_u64_le(
        mutated + 40,
        test_semantic_forward_checksum(mutated, size)
    );
    ASSERT_TRUE(evoke_semantic_forward_chunk_deserialize(
        mutated,
        size,
        chunk.source_authority_checksum,
        &restored
    ) == EVOKE_ERR_FORMAT);
    test_write_u16_le(bytes + 4, 5);
    ASSERT_TRUE(evoke_semantic_forward_header_deserialize(
        bytes,
        evoke_semantic_forward_header_size(),
        size,
        chunk.source_authority_checksum,
        &header
    ) == EVOKE_ERR_FORMAT);
    test_write_u16_le(bytes + 4, EVOKE_SEMANTIC_FORWARD_TRANSPOSE_VERSION);
    ASSERT_TRUE(evoke_semantic_forward_header_deserialize(
        bytes,
        evoke_semantic_forward_header_size(),
        size - 1U,
        chunk.source_authority_checksum,
        &header
    ) == EVOKE_ERR_FORMAT);
    bytes[size - 1U] ^= UINT8_C(0x01);
    ASSERT_TRUE(evoke_semantic_forward_chunk_deserialize(
        bytes,
        size,
        chunk.source_authority_checksum,
        &restored
    ) == EVOKE_ERR_FORMAT);

    free(mutated);
    free(bytes);
    evoke_semantic_forward_chunk_free(&restored);
    evoke_semantic_forward_chunk_init(&chunk);
}

typedef struct test_semantic_forward_bound_entries
{
    uint32_t block_ids[4];
    float maxima[4];
    uint32_t count;
} test_semantic_forward_bound_entries;

static evoke_status
test_semantic_forward_bound_collect(
    void *context,
    uint32_t block_id,
    float maximum
)
{
    test_semantic_forward_bound_entries *entries = context;

    if (entries == NULL || entries->count >= 4)
    {
        return EVOKE_ERR_RANGE;
    }
    entries->block_ids[entries->count] = block_id;
    entries->maxima[entries->count] = maximum;
    entries->count++;
    return EVOKE_OK;
}

static void
test_semantic_forward_bound_roundtrip(void)
{
    const uint64_t authority = UINT64_C(0x1122334455667788);
    uint64_t offsets[4] = {0};
    uint8_t payload[32] = {0};
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    evoke_semantic_forward_bound_summary summary;
    test_semantic_forward_bound_entries entries = {0};
    size_t payload_size = 0;
    size_t encoded_size = 0;
    size_t object_size = 0;
    size_t term_offset = 0;
    size_t term_size = 0;
    uint32_t entry_count = 0;

    ASSERT_STATUS_OK(evoke_semantic_forward_bound_encode_entry(
        payload + payload_size,
        sizeof(payload) - payload_size,
        0,
        1.5f,
        &encoded_size
    ));
    payload_size += encoded_size;
    ASSERT_STATUS_OK(evoke_semantic_forward_bound_encode_entry(
        payload + payload_size,
        sizeof(payload) - payload_size,
        3,
        2.5f,
        &encoded_size
    ));
    payload_size += encoded_size;
    offsets[1] = payload_size;
    offsets[2] = payload_size;
    ASSERT_STATUS_OK(evoke_semantic_forward_bound_encode_entry(
        payload + payload_size,
        sizeof(payload) - payload_size,
        5,
        0.75f,
        &encoded_size
    ));
    payload_size += encoded_size;
    offsets[3] = payload_size;

    ASSERT_STATUS_OK(evoke_semantic_forward_bound_shard_serialize(
        authority,
        64,
        100,
        64,
        3,
        offsets,
        payload,
        payload_size,
        &bytes,
        &object_size
    ));
    ASSERT_STATUS_OK(evoke_semantic_forward_bound_header_deserialize(
        bytes,
        EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE,
        object_size,
        authority,
        &summary
    ));
    ASSERT_TRUE(summary.first_term == 64);
    ASSERT_TRUE(summary.term_count == 3);
    ASSERT_TRUE(summary.document_count == 64);
    ASSERT_TRUE(summary.vocab_size == 100);
    ASSERT_TRUE(summary.block_shift == 3);

    ASSERT_STATUS_OK(evoke_semantic_forward_bound_term_slice(
        &summary,
        bytes + summary.offsets_offset,
        2U * sizeof(uint64_t),
        64,
        &term_offset,
        &term_size
    ));
    ASSERT_STATUS_OK(evoke_semantic_forward_bound_visit(
        bytes + term_offset,
        term_size,
        8,
        test_semantic_forward_bound_collect,
        &entries,
        &entry_count
    ));
    ASSERT_TRUE(entry_count == 2);
    ASSERT_TRUE(entries.count == 2);
    ASSERT_TRUE(entries.block_ids[0] == 0);
    ASSERT_TRUE(entries.block_ids[1] == 3);
    ASSERT_TRUE(entries.maxima[0] == 1.5f);
    ASSERT_TRUE(entries.maxima[1] == 2.5f);

    ASSERT_STATUS_OK(evoke_semantic_forward_bound_term_slice(
        &summary,
        bytes + summary.offsets_offset + sizeof(uint64_t),
        2U * sizeof(uint64_t),
        65,
        &term_offset,
        &term_size
    ));
    ASSERT_TRUE(term_size == 0);
    ASSERT_STATUS_OK(evoke_semantic_forward_bound_visit(
        bytes + term_offset,
        term_size,
        8,
        test_semantic_forward_bound_collect,
        &entries,
        &entry_count
    ));
    ASSERT_TRUE(entry_count == 0);

    ASSERT_STATUS_OK(evoke_semantic_forward_bound_term_slice(
        &summary,
        bytes + summary.offsets_offset + 2U * sizeof(uint64_t),
        2U * sizeof(uint64_t),
        66,
        &term_offset,
        &term_size
    ));
    memset(&entries, 0, sizeof(entries));
    ASSERT_STATUS_OK(evoke_semantic_forward_bound_visit(
        bytes + term_offset,
        term_size,
        8,
        test_semantic_forward_bound_collect,
        &entries,
        &entry_count
    ));
    ASSERT_TRUE(entry_count == 1);
    ASSERT_TRUE(entries.block_ids[0] == 5);
    ASSERT_TRUE(entries.maxima[0] == 0.75f);

    mutated = malloc(object_size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, object_size);
    mutated[0] ^= UINT8_C(0x01);
    ASSERT_TRUE(evoke_semantic_forward_bound_header_deserialize(
        mutated,
        EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE,
        object_size,
        authority,
        &summary
    ) == EVOKE_ERR_FORMAT);

    memcpy(mutated, bytes, object_size);
    test_write_u64_le(
        mutated + EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE +
            sizeof(uint64_t),
        test_read_u64_le(
            mutated + EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE +
                2U * sizeof(uint64_t)
        ) + 1U
    );
    ASSERT_TRUE(evoke_semantic_forward_bound_term_slice(
        &summary,
        mutated + EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE +
            sizeof(uint64_t),
        2U * sizeof(uint64_t),
        65,
        &term_offset,
        &term_size
    ) == EVOKE_ERR_FORMAT);

    memcpy(mutated, bytes, object_size);
    ASSERT_STATUS_OK(evoke_semantic_forward_bound_term_slice(
        &summary,
        mutated + summary.offsets_offset,
        2U * sizeof(uint64_t),
        64,
        &term_offset,
        &term_size
    ));
    mutated[term_offset + evoke_semantic_forward_bound_entry_size(0)] = 0;
    ASSERT_TRUE(evoke_semantic_forward_bound_visit(
        mutated + term_offset,
        term_size,
        8,
        test_semantic_forward_bound_collect,
        &entries,
        &entry_count
    ) == EVOKE_ERR_FORMAT);

    free(mutated);
    free(bytes);
}

static void
test_semantic_forward_adaptive_dense_lane(void)
{
    const uint32_t row_offsets[] = {0, 2, 2, 3, 3, 4, 4, 5, 6};
    const uint32_t term_ids[] = {3, 9, 3, 3, 3, 9};
    const double contributions[] = {
        1.0, 0.5, 3.0, 5.0, 7.0, -0.5
    };
    const uint8_t operations[] = {
        EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT,
        EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT,
        EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT,
        EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT,
        EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT,
        EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT
    };
    const uint32_t query_ids[] = {3, 9};
    const float query_weights[] = {0.25f, 2.0f};
    uint8_t allowed_document_bitmap[] = {
        (uint8_t) (
            (UINT8_C(1) << 0) |
            (UINT8_C(1) << 3) |
            (UINT8_C(1) << 7)
        )
    };
    evoke_semantic_forward_chunk chunk;
    evoke_semantic_forward_chunk restored;
    evoke_semantic_forward_header header;
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    float scores[8] = {0};
    float row_score = 0.0f;
    uint32_t row_postings = 0;
    uint64_t postings_examined = 0;
    uint64_t bytes_read = 0;
    size_t size = 0;

    evoke_semantic_forward_chunk_init(&chunk);
    evoke_semantic_forward_chunk_init(&restored);
    chunk.source_authority_checksum = UINT64_C(0x8877665544332211);
    chunk.first_document = 200;
    chunk.document_count = 8;
    chunk.posting_count = 6;
    chunk.vocab_size = 10;
    chunk.row_offsets = (uint32_t *) row_offsets;
    chunk.term_ids = (uint32_t *) term_ids;
    chunk.operations = (uint8_t *) operations;
    chunk.contributions = (double *) contributions;

    ASSERT_STATUS_OK(evoke_semantic_forward_chunk_serialize(
        &chunk,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(evoke_semantic_forward_header_deserialize(
        bytes,
        evoke_semantic_forward_header_size(),
        size,
        chunk.source_authority_checksum,
        &header
    ));
    ASSERT_TRUE(header.transpose_dense_term_count == 1);
    ASSERT_TRUE(header.transpose_sparse_posting_count == 2);
    ASSERT_TRUE(
        test_read_u32_le(bytes + header.transpose_dense_term_ids_offset) == 3
    );
    ASSERT_TRUE(
        header.object_size - header.transpose_scales_offset <=
            (size_t) chunk.document_count * sizeof(float) +
            ((size_t) chunk.vocab_size + 1U) * sizeof(uint32_t) +
            (size_t) chunk.posting_count *
                EVOKE_SEMANTIC_FORWARD_TRANSPOSE_ENTRY_SIZE
    );

    ASSERT_STATUS_OK(evoke_semantic_forward_score_transposed_sorted(
        bytes,
        size,
        chunk.source_authority_checksum,
        NULL,
        0,
        query_ids,
        query_weights,
        2,
        scores,
        8,
        &postings_examined,
        &bytes_read
    ));
    ASSERT_TRUE(postings_examined == 10);
    for (uint32_t document = 0; document < chunk.document_count; document++)
    {
        uint32_t row_start = test_read_u32_le(
            bytes + header.row_offsets_offset +
                (size_t) document * sizeof(uint32_t)
        );
        uint32_t row_end = test_read_u32_le(
            bytes + header.row_offsets_offset +
                ((size_t) document + 1U) * sizeof(uint32_t)
        );

        ASSERT_STATUS_OK(evoke_semantic_forward_score_serialized_row_sorted(
            bytes + header.row_data_offset + row_start,
            (size_t) row_end - row_start,
            query_ids,
            query_weights,
            2,
            &row_score,
            &row_postings
        ));
        ASSERT_TRUE(memcmp(
            &row_score,
            &scores[document],
            sizeof(row_score)
        ) == 0);
    }

    ASSERT_STATUS_OK(evoke_semantic_forward_score_transposed_sorted(
        bytes,
        size,
        chunk.source_authority_checksum,
        allowed_document_bitmap,
        sizeof(allowed_document_bitmap),
        query_ids,
        query_weights,
        2,
        scores,
        8,
        &postings_examined,
        &bytes_read
    ));
    ASSERT_TRUE(scores[1] == 0.0f);
    ASSERT_TRUE(scores[2] == 0.0f);
    ASSERT_TRUE(scores[4] == 0.0f);
    ASSERT_TRUE(scores[5] == 0.0f);
    ASSERT_TRUE(scores[6] == 0.0f);

    ASSERT_STATUS_OK(evoke_semantic_forward_chunk_deserialize(
        bytes,
        size,
        chunk.source_authority_checksum,
        &restored
    ));
    mutated = malloc(size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, size);
    mutated[header.transpose_dense_codes_offset + 1U] = UINT8_C(1);
    test_write_u64_le(mutated + 40, 0);
    test_write_u64_le(
        mutated + 40,
        test_semantic_forward_checksum(mutated, size)
    );
    ASSERT_TRUE(evoke_semantic_forward_chunk_deserialize(
        mutated,
        size,
        chunk.source_authority_checksum,
        &restored
    ) == EVOKE_ERR_FORMAT);

    free(mutated);
    free(bytes);
    evoke_semantic_forward_chunk_free(&restored);
    evoke_semantic_forward_chunk_init(&chunk);
}

static void
test_semantic_impact_frontier_roundtrip(void)
{
    const uint64_t authority = UINT64_C(0x123456789abcdef0);
    const uint8_t first_documents[] = {4, 2, 9};
    const double first_impacts[] = {1.5, 1.25, 0.5};
    const uint8_t second_documents[] = {7, 1};
    const double second_impacts[] = {2.0, 0.25};
    uint64_t offsets[3] = {0};
    uint8_t payload[64] = {0};
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    evoke_semantic_impact_frontier_summary summary;
    evoke_semantic_impact_frontier_cursor cursor;
    evoke_semantic_impact_frontier_block block;
    size_t encoded_size = 0;
    size_t object_size = 0;
    size_t payload_size = 0;
    size_t term_offset = 0;
    size_t term_size = 0;
    bool has_block = false;
    uint8_t local_document = 0;
    float impact = 0.0f;
    float rounded = 0.0f;

    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_round_up(
        1.0 + 1e-8,
        &rounded
    ));
    ASSERT_TRUE((double) rounded >= 1.0 + 1e-8);
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_encode_block(
        payload + payload_size,
        sizeof(payload) - payload_size,
        1,
        first_documents,
        first_impacts,
        3,
        EVOKE_SEMANTIC_IMPACT_PRECISION_F32,
        &encoded_size
    ));
    payload_size += encoded_size;
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_encode_block(
        payload + payload_size,
        sizeof(payload) - payload_size,
        3,
        second_documents,
        second_impacts,
        2,
        EVOKE_SEMANTIC_IMPACT_PRECISION_F32,
        &encoded_size
    ));
    payload_size += encoded_size;
    offsets[1] = payload_size;
    offsets[2] = payload_size;

    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_shard_serialize(
        authority,
        2048,
        100,
        64,
        2,
        EVOKE_SEMANTIC_IMPACT_PRECISION_F32,
        offsets,
        payload,
        payload_size,
        &bytes,
        &object_size
    ));
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_header_deserialize(
        bytes,
        EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE,
        object_size,
        authority,
        &summary
    ));
    ASSERT_TRUE(summary.block_shift == 6);
    ASSERT_TRUE(
        summary.impact_precision == EVOKE_SEMANTIC_IMPACT_PRECISION_F32
    );
    ASSERT_TRUE(summary.document_count == 2048);
    ASSERT_TRUE(evoke_semantic_impact_frontier_header_deserialize(
        bytes,
        EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE,
        object_size,
        authority + 1U,
        &summary
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_semantic_impact_frontier_round_up(
        NAN,
        &rounded
    ) == EVOKE_ERR_INVALID);
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_term_slice(
        &summary,
        bytes + summary.offsets_offset,
        3U * sizeof(uint64_t),
        64,
        &term_offset,
        &term_size
    ));
    {
        size_t empty_offset = 0;
        size_t empty_size = 1;

        ASSERT_STATUS_OK(evoke_semantic_impact_frontier_term_slice(
            &summary,
            bytes + summary.offsets_offset,
            3U * sizeof(uint64_t),
            65,
            &empty_offset,
            &empty_size
        ));
        ASSERT_TRUE(empty_offset == summary.payload_offset + payload_size);
        ASSERT_TRUE(empty_size == 0);
        ASSERT_TRUE(evoke_semantic_impact_frontier_term_slice(
            &summary,
            bytes + summary.offsets_offset,
            2U * sizeof(uint64_t),
            65,
            &empty_offset,
            &empty_size
        ) == EVOKE_ERR_RANGE);
    }
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_init(
        &cursor,
        bytes + term_offset,
        term_size,
        8,
        summary.impact_precision
    ));
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_next(
        &cursor,
        &block,
        &has_block
    ));
    ASSERT_TRUE(has_block);
    ASSERT_TRUE(block.block_id == 1);
    ASSERT_TRUE(block.posting_count == 3);
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_block_posting(
        &block,
        1,
        &local_document,
        &impact
    ));
    ASSERT_TRUE(local_document == 2);
    ASSERT_TRUE(impact == 1.25f);
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_next(
        &cursor,
        &block,
        &has_block
    ));
    ASSERT_TRUE(has_block);
    ASSERT_TRUE(block.block_id == 4);
    ASSERT_TRUE(block.posting_count == 2);
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_next(
        &cursor,
        &block,
        &has_block
    ));
    ASSERT_TRUE(!has_block);

    mutated = malloc(term_size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes + term_offset, term_size);
    mutated[8] = UINT8_C(0x00);
    mutated[9] = UINT8_C(0x00);
    mutated[10] = UINT8_C(0x00);
    mutated[11] = UINT8_C(0x40);
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_init(
        &cursor,
        mutated,
        term_size,
        8,
        summary.impact_precision
    ));
    ASSERT_TRUE(evoke_semantic_impact_frontier_cursor_next(
        &cursor,
        &block,
        &has_block
    ) == EVOKE_ERR_FORMAT);

    memcpy(mutated, bytes + term_offset, term_size);
    mutated[evoke_semantic_impact_frontier_block_size(
        1,
        3,
        summary.impact_precision
    )] = 0;
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_init(
        &cursor,
        mutated,
        term_size,
        8,
        summary.impact_precision
    ));
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_next(
        &cursor,
        &block,
        &has_block
    ));
    ASSERT_TRUE(evoke_semantic_impact_frontier_cursor_next(
        &cursor,
        &block,
        &has_block
    ) == EVOKE_ERR_FORMAT);

    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_init(
        &cursor,
        bytes + term_offset,
        term_size - 1U,
        8,
        summary.impact_precision
    ));
    ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_next(
        &cursor,
        &block,
        &has_block
    ));
    ASSERT_TRUE(evoke_semantic_impact_frontier_cursor_next(
        &cursor,
        &block,
        &has_block
    ) == EVOKE_ERR_FORMAT);

    {
        const uint8_t duplicate_documents[] = {1, 1};
        const uint8_t out_of_range_documents[] = {64};
        const double ordered_impacts[] = {1.0, 0.5};
        const double unordered_impacts[] = {0.5, 1.0};

        ASSERT_TRUE(evoke_semantic_impact_frontier_encode_block(
            mutated,
            term_size,
            0,
            duplicate_documents,
            ordered_impacts,
            2,
            EVOKE_SEMANTIC_IMPACT_PRECISION_F32,
            &encoded_size
        ) == EVOKE_ERR_INVALID);
        ASSERT_TRUE(evoke_semantic_impact_frontier_encode_block(
            mutated,
            term_size,
            0,
            second_documents,
            unordered_impacts,
            2,
            EVOKE_SEMANTIC_IMPACT_PRECISION_F32,
            &encoded_size
        ) == EVOKE_ERR_INVALID);
        ASSERT_TRUE(evoke_semantic_impact_frontier_encode_block(
            mutated,
            term_size,
            0,
            out_of_range_documents,
            ordered_impacts,
            1,
            EVOKE_SEMANTIC_IMPACT_PRECISION_F32,
            &encoded_size
        ) == EVOKE_ERR_INVALID);
    }

    free(mutated);
    free(bytes);
}

static void
test_semantic_impact_frontier_precision_profiles(void)
{
    const evoke_semantic_impact_precision precisions[] = {
        EVOKE_SEMANTIC_IMPACT_PRECISION_F32,
        EVOKE_SEMANTIC_IMPACT_PRECISION_FP16,
        EVOKE_SEMANTIC_IMPACT_PRECISION_U8
    };
    const uint8_t documents[] = {3, 9, 1};
    const double impacts[] = {1.503, 1.249, 0.501};

    for (size_t precision_index = 0;
         precision_index < sizeof(precisions) / sizeof(precisions[0]);
         precision_index++)
    {
        evoke_semantic_impact_precision precision =
            precisions[precision_index];
        uint64_t offsets[2] = {0};
        uint8_t payload[64] = {0};
        uint8_t replay[64] = {0};
        uint8_t *bytes = NULL;
        evoke_semantic_impact_frontier_summary summary;
        evoke_semantic_impact_frontier_cursor cursor;
        evoke_semantic_impact_frontier_block block;
        double replay_impacts[3] = {0};
        size_t payload_size = 0;
        size_t replay_size = 0;
        size_t object_size = 0;
        size_t term_offset = 0;
        size_t term_size = 0;
        bool has_block = false;

        ASSERT_STATUS_OK(evoke_semantic_impact_frontier_encode_block(
            payload,
            sizeof(payload),
            2,
            documents,
            impacts,
            3,
            precision,
            &payload_size
        ));
        offsets[1] = payload_size;
        ASSERT_STATUS_OK(evoke_semantic_impact_frontier_shard_serialize(
            UINT64_C(0x1234),
            512,
            64,
            0,
            1,
            precision,
            offsets,
            payload,
            payload_size,
            &bytes,
            &object_size
        ));
        ASSERT_STATUS_OK(evoke_semantic_impact_frontier_header_deserialize(
            bytes,
            EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE,
            object_size,
            UINT64_C(0x1234),
            &summary
        ));
        ASSERT_TRUE(summary.impact_precision == precision);
        ASSERT_STATUS_OK(evoke_semantic_impact_frontier_term_slice(
            &summary,
            bytes + summary.offsets_offset,
            2U * sizeof(uint64_t),
            0,
            &term_offset,
            &term_size
        ));
        ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_init(
            &cursor,
            bytes + term_offset,
            term_size,
            8,
            summary.impact_precision
        ));
        ASSERT_STATUS_OK(evoke_semantic_impact_frontier_cursor_next(
            &cursor,
            &block,
            &has_block
        ));
        ASSERT_TRUE(has_block && block.block_id == 2);
        ASSERT_TRUE(block.posting_count == 3);
        ASSERT_TRUE(
            block.posting_width == 1U +
                evoke_semantic_bmp_impact_width(precision)
        );
        for (uint32_t posting_index = 0; posting_index < 3; posting_index++)
        {
            uint8_t local_document = 0;
            float impact = 0.0f;

            ASSERT_STATUS_OK(evoke_semantic_impact_frontier_block_posting(
                &block,
                posting_index,
                &local_document,
                &impact
            ));
            ASSERT_TRUE(local_document == documents[posting_index]);
            ASSERT_TRUE(impact > 0.0f);
            replay_impacts[posting_index] = impact;
        }
        ASSERT_STATUS_OK(evoke_semantic_impact_frontier_encode_block(
            replay,
            sizeof(replay),
            2,
            documents,
            replay_impacts,
            3,
            precision,
            &replay_size
        ));
        ASSERT_TRUE(replay_size == payload_size);
        ASSERT_TRUE(memcmp(replay, payload, payload_size) == 0);
        free(bytes);
    }

    {
        const uint8_t document = 1;
        const double negative_impact = -0.5;
        uint8_t bytes[8] = {0};
        size_t size = 0;

        ASSERT_TRUE(evoke_semantic_impact_frontier_encode_block(
            bytes,
            sizeof(bytes),
            0,
            &document,
            &negative_impact,
            1,
            EVOKE_SEMANTIC_IMPACT_PRECISION_U8,
            &size
        ) == EVOKE_ERR_INVALID);
    }
}

static void
test_segment_manifest_published_closure(void)
{
    evoke_segment_manifest manifest;
    evoke_segment_object_ref manifest_ref;
    evoke_segment_object_ref payload_ref;

    initialize_test_segment_manifest(&manifest);
    initialize_test_object_ref(
        &manifest_ref,
        EVOKE_SEGMENT_OBJECT_MANIFEST,
        61,
        2,
        manifest.manifest_id,
        104
    );
    ASSERT_STATUS_OK(evoke_segment_manifest_validate_published(
        &manifest,
        &manifest_ref,
        64
    ));
    ASSERT_STATUS_OK(evoke_segment_descriptor_payload_ref(
        &manifest,
        &manifest.segments[1],
        &payload_ref
    ));
    ASSERT_TRUE(
        payload_ref.object_kind == EVOKE_SEGMENT_OBJECT_PAYLOAD
    );
    ASSERT_TRUE(payload_ref.object_id == 2);
    ASSERT_TRUE(payload_ref.owner_manifest_id == manifest.manifest_id);
    ASSERT_TRUE(payload_ref.start_block == 20);
    ASSERT_TRUE(payload_ref.page_count == 2);
    ASSERT_STATUS_OK(evoke_segment_descriptor_payload_ref(
        &manifest,
        &manifest.segments[0],
        &payload_ref
    ));
    ASSERT_TRUE(
        payload_ref.owner_manifest_id == manifest.parent_manifest_id
    );

    ASSERT_TRUE(evoke_segment_manifest_validate_published(
        &manifest,
        &manifest_ref,
        53
    ) == EVOKE_ERR_FORMAT);
    manifest.retired_ranges[1].start_block = 63;
    manifest.retired_ranges[1].block_count = 2;
    ASSERT_TRUE(evoke_segment_manifest_validate_published(
        &manifest,
        &manifest_ref,
        64
    ) == EVOKE_ERR_FORMAT);
    manifest.retired_ranges[1].start_block = 14;
    manifest.retired_ranges[1].block_count = 1;
    manifest_ref.start_block = 10;
    ASSERT_TRUE(evoke_segment_manifest_validate_published(
        &manifest,
        &manifest_ref,
        64
    ) == EVOKE_ERR_FORMAT);
    manifest_ref.start_block = 51;
    ASSERT_TRUE(evoke_segment_manifest_validate_published(
        &manifest,
        &manifest_ref,
        64
    ) == EVOKE_ERR_FORMAT);
    manifest_ref.start_block =
        manifest.document_directory.start_block;
    ASSERT_TRUE(evoke_segment_manifest_validate_published(
        &manifest,
        &manifest_ref,
        64
    ) == EVOKE_ERR_FORMAT);
    manifest_ref.start_block = 60;
    manifest_ref.object_id++;
    ASSERT_TRUE(evoke_segment_manifest_validate_published(
        &manifest,
        &manifest_ref,
        64
    ) == EVOKE_ERR_FORMAT);

    evoke_segment_manifest_free(&manifest);
}

static void
test_segment_query_contract_roundtrip(void)
{
    const char *doc0[] = {"cat", "cat", "feline"};
    const char *doc1[] = {"dog", "friend"};
    const char *doc2[] = {"cat", "bird", "bird"};
    evoke_doc_tokens docs[] = {
        {.tokens = doc0, .len = 3},
        {.tokens = doc1, .len = 2},
        {.tokens = doc2, .len = 3}
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_segment_manifest manifest;
    evoke_segment_query_contract contract;
    evoke_segment_query_contract restored;
    evoke_index index;
    evoke_index metadata_index;
    const char *query_tokens[] = {"cat", "missing"};
    uint32_t *query_ids = NULL;
    size_t query_len = 0;
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    size_t size = 0;

    evoke_index_init(&index);
    evoke_segment_manifest_init(&manifest);
    evoke_segment_query_contract_init(&contract);
    evoke_segment_query_contract_init(&restored);
    evoke_index_init(&metadata_index);
    ASSERT_STATUS_OK(evoke_build_index_from_tokens(
        docs,
        3,
        &params,
        &index
    ));

    manifest.manifest_id = 7;
    manifest.vocab_size = index.vocab_size;
    ASSERT_STATUS_OK(evoke_segment_query_contract_build(
        &index,
        &manifest,
        &contract
    ));
    ASSERT_TRUE(
        (contract.flags &
         EVOKE_QUERY_CONTRACT_FLAG_VOCABULARY) != 0
    );
    ASSERT_TRUE(contract.vocab_size == index.vocab_size);
    ASSERT_TRUE(contract.params.method == params.method);
    ASSERT_TRUE(
        contract.block_shift == EVOKE_DEFAULT_POSTING_BLOCK_SHIFT
    );
    ASSERT_STATUS_OK(evoke_segment_query_contract_serialize(
        &contract,
        &manifest,
        &bytes,
        &size
    ));
    ASSERT_TRUE(bytes != NULL);
    ASSERT_TRUE(size == 96);
    ASSERT_STATUS_OK(evoke_segment_query_contract_deserialize(
        bytes,
        size,
        &manifest,
        &restored
    ));
    ASSERT_TRUE(restored.params.idf_method == params.idf_method);
    ASSERT_TRUE(restored.vocab_size == index.vocab_size);
    ASSERT_TRUE(restored.block_shift == contract.block_shift);
    ASSERT_TRUE(restored.vocab == NULL);
    {
        evoke_segment_manifest descendant = manifest;

        descendant.parent_manifest_id = manifest.manifest_id;
        descendant.manifest_id = manifest.manifest_id + 1;
        initialize_test_object_ref(
            &descendant.query_contract,
            EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT,
            1,
            1,
            manifest.manifest_id,
            100
        );
        ASSERT_STATUS_OK(evoke_segment_query_contract_deserialize(
            bytes,
            size,
            &descendant,
            &restored
        ));
        descendant.vocab_size++;
        ASSERT_STATUS_OK(evoke_segment_query_contract_deserialize(
            bytes,
            size,
            &descendant,
            &restored
        ));
        ASSERT_TRUE(restored.vocab_size == descendant.vocab_size);
        ASSERT_TRUE(restored.vocab == NULL);
    }

    mutated = malloc(size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, size);
    mutated[size - 1] ^= 0x1U;
    ASSERT_TRUE(evoke_segment_query_contract_deserialize(
        mutated,
        size,
        &manifest,
        &restored
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_segment_query_contract_deserialize(
        bytes,
        size - 1,
        &manifest,
        &restored
    ) == EVOKE_ERR_FORMAT);
    ASSERT_STATUS_OK(evoke_segment_query_contract_deserialize(
        bytes,
        size,
        &manifest,
        &restored
    ));
    restored.vocab = calloc(
        restored.vocab_size,
        sizeof(*restored.vocab)
    );
    ASSERT_TRUE(restored.vocab != NULL);
    for (uint32_t term_id = 0;
         term_id < restored.vocab_size;
         term_id++)
    {
        size_t length = strlen(contract.vocab[term_id]);

        restored.vocab[term_id] = malloc(length + 1);
        ASSERT_TRUE(restored.vocab[term_id] != NULL);
        memcpy(
            restored.vocab[term_id],
            contract.vocab[term_id],
            length + 1
        );
    }

    manifest.statistics_epoch = 1;
    manifest.doc_frequencies = calloc(
        manifest.vocab_size,
        sizeof(*manifest.doc_frequencies)
    );
    ASSERT_TRUE(manifest.doc_frequencies != NULL);
    initialize_test_object_ref(
        &manifest.query_contract,
        EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT,
        1,
        1,
        manifest.manifest_id,
        100
    );
    ASSERT_STATUS_OK(evoke_segment_index_metadata_build(
        &restored,
        &manifest,
        manifest.doc_frequencies,
        NULL,
        0,
        &metadata_index
    ));
    ASSERT_TRUE(metadata_index.vocab != restored.vocab);
    ASSERT_TRUE(strcmp(metadata_index.vocab[0], restored.vocab[0]) == 0);
    ASSERT_STATUS_OK(evoke_query_token_ids(
        &metadata_index,
        query_tokens,
        2,
        &query_ids,
        &query_len
    ));
    ASSERT_TRUE(query_len == 1);

    free(bytes);
    free(mutated);
    free(query_ids);
    evoke_index_free(&metadata_index);
    evoke_segment_query_contract_free(&restored);
    evoke_segment_query_contract_free(&contract);
    evoke_segment_manifest_free(&manifest);
    evoke_index_free(&index);
}

static void
test_lexical_catalog_roundtrip_and_validation(void)
{
    const char *terms[] = {
        "cat",
        "",
        "\316\262eta"
    };
    evoke_lexical_catalog catalog;
    evoke_lexical_catalog restored;
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    size_t size = 0;

    evoke_lexical_catalog_init(&catalog);
    evoke_lexical_catalog_init(&restored);
    ASSERT_STATUS_OK(evoke_lexical_catalog_build(
        11,
        20,
        3,
        terms,
        &catalog
    ));
    ASSERT_STATUS_OK(evoke_lexical_catalog_serialize(
        &catalog,
        &bytes,
        &size
    ));
    ASSERT_TRUE(bytes != NULL);
    ASSERT_TRUE(size > 64);
    ASSERT_STATUS_OK(evoke_lexical_catalog_deserialize(
        bytes,
        size,
        11,
        &restored
    ));
    ASSERT_TRUE(restored.owner_manifest_id == 11);
    ASSERT_TRUE(restored.first_term_id == 20);
    ASSERT_TRUE(restored.term_count == 3);
    for (uint32_t term_index = 0; term_index < 3; term_index++)
    {
        ASSERT_TRUE(
            strcmp(restored.terms[term_index], terms[term_index]) == 0
        );
    }
    ASSERT_TRUE(evoke_lexical_catalog_deserialize(
        bytes,
        size,
        12,
        &restored
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_lexical_catalog_deserialize(
        bytes,
        size - 1,
        11,
        &restored
    ) == EVOKE_ERR_FORMAT);

    mutated = malloc(size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, size);
    mutated[size - 1] ^= 0x1U;
    ASSERT_TRUE(evoke_lexical_catalog_deserialize(
        mutated,
        size,
        11,
        &restored
    ) == EVOKE_ERR_FORMAT);
    memcpy(mutated, bytes, size);
    write_u64_le(mutated + 64, 1);
    memset(mutated + 56, 0, sizeof(uint64_t));
    write_u64_le(
        mutated + 56,
        evoke_segment_blob_checksum(mutated, size)
    );
    ASSERT_TRUE(evoke_lexical_catalog_deserialize(
        mutated,
        size,
        11,
        &restored
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_lexical_catalog_build(
        0,
        20,
        3,
        terms,
        &restored
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_lexical_catalog_build(
        11,
        UINT32_MAX,
        1,
        terms,
        &restored
    ) == EVOKE_ERR_FORMAT);

    free(mutated);
    free(bytes);
    evoke_lexical_catalog_free(&restored);
    evoke_lexical_catalog_free(&catalog);
}

static void
test_term_directory_roundtrip_and_validation(void)
{
    evoke_segment_manifest manifest;
    evoke_term_directory directory;
    evoke_term_directory restored;
    uint64_t term_offsets[] = {0, 2, 3, 4, 6};
    evoke_term_extent_descriptor extents[] = {
        {
            .segment_index = 0,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 2
        },
        {
            .segment_index = 1,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 1
        },
        {
            .segment_index = 0,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 2,
            .posting_count = 2
        },
        {
            .segment_index = 0,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 4,
            .posting_count = 2
        },
        {
            .segment_index = 0,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 6,
            .posting_count = 1
        },
        {
            .segment_index = 1,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 1,
            .posting_count = 1
        }
    };
    uint64_t excessive_offsets[] = {0, 9, 9, 9, 9};
    evoke_term_extent_descriptor excessive_extents[9] = {0};
    evoke_segment_object_ref published_directory_ref;
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    size_t size = 0;
    size_t unpublished_size = 0;

    initialize_test_segment_manifest(&manifest);
    evoke_term_directory_init(&directory);
    evoke_term_directory_init(&restored);
    directory.vocab_size = 4;
    directory.extent_count = 6;
    directory.term_offsets = term_offsets;
    directory.extents = extents;

    ASSERT_STATUS_OK(evoke_term_directory_validate(&directory, &manifest));
    published_directory_ref = manifest.term_directory;
    manifest.flags &= ~EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    memset(&manifest.term_directory, 0, sizeof(manifest.term_directory));
    ASSERT_STATUS_OK(evoke_term_directory_serialized_size(
        &directory,
        &manifest,
        &unpublished_size
    ));
    ASSERT_TRUE(unpublished_size > 0);
    ASSERT_TRUE(evoke_term_directory_validate(
        &directory,
        &manifest
    ) == EVOKE_ERR_FORMAT);
    manifest.flags |= EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    manifest.term_directory = published_directory_ref;
    ASSERT_STATUS_OK(evoke_term_directory_serialize(
        &directory,
        &manifest,
        &bytes,
        &size
    ));
    ASSERT_TRUE(bytes != NULL);
    ASSERT_STATUS_OK(evoke_term_directory_deserialize(
        bytes,
        size,
        &manifest,
        &restored
    ));
    ASSERT_TRUE(restored.vocab_size == directory.vocab_size);
    ASSERT_TRUE(restored.extent_count == directory.extent_count);
    assert_uint64_array(
        restored.term_offsets,
        directory.term_offsets,
        (size_t) directory.vocab_size + 1
    );
    ASSERT_TRUE(restored.extents[5].segment_index == 1);
    ASSERT_TRUE(
        restored.extents[5].kind ==
        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL
    );
    ASSERT_TRUE(restored.extents[5].posting_offset == 1);

    evoke_term_directory_free(&restored);
    mutated = malloc(size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, size);
    test_write_u32_le(
        mutated + 20,
        EVOKE_TERM_DIRECTORY_LEGACY_MAX_EXTENTS_PER_TERM
    );
    test_write_u64_le(mutated + 48, 0);
    test_write_u64_le(
        mutated + 48,
        test_term_directory_checksum(mutated, size)
    );
    ASSERT_STATUS_OK(evoke_term_directory_deserialize(
        mutated,
        size,
        &manifest,
        &restored
    ));
    evoke_term_directory_free(&restored);

    memcpy(mutated, bytes, size);
    test_write_u32_le(
        mutated + 20,
        EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM + 1
    );
    test_write_u64_le(mutated + 48, 0);
    test_write_u64_le(
        mutated + 48,
        test_term_directory_checksum(mutated, size)
    );
    ASSERT_TRUE(evoke_term_directory_deserialize(
        mutated,
        size,
        &manifest,
        &restored
    ) == EVOKE_ERR_FORMAT);

    memcpy(mutated, bytes, size);
    mutated[size - 1] ^= 0x1U;
    ASSERT_TRUE(evoke_term_directory_deserialize(
        mutated,
        size,
        &manifest,
        &restored
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_term_directory_deserialize(
        bytes,
        size - 1,
        &manifest,
        &restored
    ) == EVOKE_ERR_FORMAT);

    extents[1].segment_index = 0;
    ASSERT_TRUE(evoke_term_directory_validate(
        &directory,
        &manifest
    ) == EVOKE_ERR_FORMAT);
    extents[1].segment_index = 1;
    extents[1].kind = EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
    ASSERT_TRUE(evoke_term_directory_validate(
        &directory,
        &manifest
    ) == EVOKE_ERR_FORMAT);
    extents[1].kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
    extents[1].kind = (evoke_posting_extent_kind) 99;
    ASSERT_TRUE(evoke_term_directory_validate(
        &directory,
        &manifest
    ) == EVOKE_ERR_FORMAT);
    extents[1].kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
    extents[5].posting_offset = 2;
    ASSERT_TRUE(evoke_term_directory_validate(
        &directory,
        &manifest
    ) == EVOKE_ERR_FORMAT);
    extents[1].segment_index = 2;
    ASSERT_TRUE(evoke_term_directory_validate(
        &directory,
        &manifest
    ) == EVOKE_ERR_FORMAT);
    extents[1].segment_index = 1;

    directory.extent_count = 9;
    directory.term_offsets = excessive_offsets;
    directory.extents = excessive_extents;
    ASSERT_TRUE(evoke_term_directory_validate(
        &directory,
        &manifest
    ) == EVOKE_ERR_FORMAT);

    free(bytes);
    free(mutated);
    evoke_term_directory_free(&restored);
    evoke_segment_manifest_free(&manifest);
}

static void
test_term_fold_bundle_roundtrip_and_validation(void)
{
    evoke_term_fold_run runs[] = {
        {
            .term_id = 2,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .coverage_sequence = 41,
            .posting_offset = 0,
            .posting_count = 2
        },
        {
            .term_id = 2,
            .kind = EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT,
            .coverage_sequence = 41,
            .posting_offset = 2,
            .posting_count = 1
        },
        {
            .term_id = 9,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .coverage_sequence = 44,
            .posting_offset = 3,
            .posting_count = 2
        }
    };
    uint32_t document_slots[] = {1, 7, 3, 4, 8};
    evoke_posting_value values[5] = {0};
    evoke_term_fold_bundle bundle;
    evoke_term_fold_bundle decoded;
    evoke_semantic_bmp_packed_index semantic_bmp;
    evoke_term_fold_disk_header disk_header;
    evoke_term_fold_run disk_run;
    evoke_posting_block_record disk_block;
    uint8_t *bytes = NULL;
    size_t size = 0;
    uint64_t checksum = 0;

    values[0].term_frequency = 2;
    values[1].term_frequency = 1;
    values[2].impact = -0.5f;
    values[3].term_frequency = 3;
    values[4].term_frequency = 1;
    evoke_term_fold_bundle_init(&bundle);
    evoke_term_fold_bundle_init(&decoded);
    evoke_semantic_bmp_packed_index_init(&semantic_bmp);
    bundle.object_kind = EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    bundle.owner_manifest_id = 50;
    bundle.run_count = 3;
    bundle.posting_count = 5;
    bundle.runs = runs;
    bundle.document_slots = document_slots;
    bundle.values = values;

    ASSERT_STATUS_OK(evoke_term_fold_bundle_validate(&bundle));
    ASSERT_STATUS_OK(evoke_term_fold_bundle_serialize(
        &bundle,
        &bytes,
        &size,
        &checksum
    ));
    ASSERT_TRUE(bytes != NULL);
    ASSERT_TRUE(size > 128);
    ASSERT_TRUE(checksum != 0);
    ASSERT_STATUS_OK(evoke_term_fold_disk_header_decode(
        bytes,
        EVOKE_TERM_FOLD_HEADER_SIZE,
        &disk_header
    ));
    ASSERT_TRUE(disk_header.run_count == 3);
    ASSERT_TRUE(disk_header.format_version == 4);
    ASSERT_TRUE(disk_header.posting_count == 5);
    ASSERT_TRUE(disk_header.generic_posting_count == 4);
    ASSERT_TRUE(disk_header.block_count == 2);
    ASSERT_TRUE(disk_header.total_size == size);
    ASSERT_TRUE(
        disk_header.semantic_bmp_version ==
        EVOKE_SEMANTIC_BMP_PACKED_FORMAT_VERSION
    );
    ASSERT_TRUE(disk_header.semantic_bmp_size > 0);
    ASSERT_TRUE(
        disk_header.semantic_bmp_offset +
        disk_header.semantic_bmp_size == size
    );
    test_write_u16_le(bytes + 4, 3);
    ASSERT_TRUE(evoke_term_fold_disk_header_decode(
        bytes,
        EVOKE_TERM_FOLD_HEADER_SIZE,
        &disk_header
    ) == EVOKE_ERR_FORMAT);
    test_write_u16_le(bytes + 4, 4);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_deserialize(
        bytes + disk_header.semantic_bmp_offset,
        (size_t) disk_header.semantic_bmp_size,
        &semantic_bmp
    ));
    ASSERT_TRUE(semantic_bmp.term_count == 1);
    ASSERT_TRUE(semantic_bmp.terms[0].term_id == 2);
    ASSERT_TRUE(semantic_bmp.posting_count == 1);
    ASSERT_TRUE(semantic_bmp.impacts[0] == -0.5f);
    ASSERT_STATUS_OK(evoke_term_fold_run_decode(
        bytes + disk_header.runs_offset,
        EVOKE_TERM_FOLD_RUN_SIZE,
        &disk_run
    ));
    ASSERT_TRUE(disk_run.term_id == 2);
    ASSERT_TRUE(disk_run.posting_count == 2);
    ASSERT_STATUS_OK(evoke_term_fold_run_decode(
        bytes + disk_header.runs_offset + EVOKE_TERM_FOLD_RUN_SIZE,
        EVOKE_TERM_FOLD_RUN_SIZE,
        &disk_run
    ));
    ASSERT_TRUE(
        disk_run.kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
    );
    ASSERT_TRUE(disk_run.posting_offset == 0);
    ASSERT_TRUE(disk_run.block_count == 0);
    ASSERT_STATUS_OK(evoke_posting_block_record_decode(
        bytes + disk_header.blocks_offset,
        EVOKE_POSTING_BLOCK_RECORD_SIZE,
        &disk_block
    ));
    ASSERT_TRUE(disk_block.posting_count == 2);
    ASSERT_TRUE(disk_block.first_document_id == 1);
    ASSERT_STATUS_OK(evoke_term_fold_bundle_deserialize(
        bytes,
        size,
        &decoded
    ));
    ASSERT_TRUE(
        decoded.object_kind == EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD
    );
    ASSERT_TRUE(decoded.owner_manifest_id == 50);
    ASSERT_TRUE(decoded.statistics_epoch == 0);
    ASSERT_TRUE(decoded.run_count == 3);
    ASSERT_TRUE(
        decoded.block_shift == EVOKE_DEFAULT_POSTING_BLOCK_SHIFT
    );
    ASSERT_TRUE(decoded.block_count == 3);
    ASSERT_TRUE(decoded.posting_count == 5);
    ASSERT_TRUE(decoded.runs[1].term_id == 2);
    ASSERT_TRUE(
        decoded.runs[1].kind ==
        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
    );
    ASSERT_TRUE(decoded.runs[2].coverage_sequence == 44);
    ASSERT_TRUE(decoded.runs[0].block_offset == 0);
    ASSERT_TRUE(decoded.runs[0].block_count == 1);
    ASSERT_TRUE(decoded.runs[2].block_offset == 2);
    ASSERT_TRUE(decoded.blocks[0].posting_count == 2);
    ASSERT_TRUE(decoded.blocks[1].kind ==
        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT);
    assert_float_close(decoded.blocks[1].min_impact, -0.5f);
    ASSERT_TRUE(decoded.document_slots[4] == 8);
    ASSERT_TRUE(decoded.values[0].term_frequency == 2);
    assert_float_close(decoded.values[2].impact, -0.5f);
    decoded.blocks[0].max_term_frequency++;
    ASSERT_TRUE(
        evoke_term_fold_bundle_validate(&decoded) == EVOKE_ERR_FORMAT
    );
    decoded.blocks[0].max_term_frequency--;
    ASSERT_STATUS_OK(evoke_term_fold_bundle_validate(&decoded));

    runs[1].coverage_sequence = 42;
    ASSERT_TRUE(
        evoke_term_fold_bundle_validate(&bundle) == EVOKE_ERR_FORMAT
    );
    runs[1].coverage_sequence = 41;
    bundle.object_kind = EVOKE_SEGMENT_OBJECT_IMPACT_FOLD;
    bundle.statistics_epoch = 7;
    ASSERT_TRUE(
        evoke_term_fold_bundle_validate(&bundle) == EVOKE_ERR_FORMAT
    );
    bundle.object_kind = EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    bundle.statistics_epoch = 0;

    bytes[size - 1] ^= UINT8_C(0x01);
    ASSERT_TRUE(
        evoke_term_fold_bundle_deserialize(
            bytes,
            size,
            &decoded
        ) == EVOKE_ERR_FORMAT
    );
    ASSERT_TRUE(decoded.owner_manifest_id == 50);

    free(bytes);
    evoke_semantic_bmp_packed_index_free(&semantic_bmp);
    evoke_term_fold_bundle_free(&decoded);
}

static void
test_document_version_records_validation(void)
{
    evoke_segment_manifest manifest;
    evoke_document_version_record versions[] = {
        {
            .document_slot = 0,
            .born_sequence = 1,
            .record_xid = 0,
            .heap_block = 10,
            .document_length = 3,
            .heap_offset = 1,
            .flags = EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE |
                EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID
        },
        {
            .document_slot = 1,
            .born_sequence = 2,
            .record_xid = 21,
            .heap_block = 10,
            .document_length = 2,
            .heap_offset = 2,
            .flags = EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING
        },
        {
            .document_slot = 2,
            .born_sequence = 3,
            .record_xid = 0,
            .heap_block = 0,
            .document_length = 0,
            .heap_offset = 0,
            .flags = EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE |
                EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID
        },
        {
            .document_slot = 3,
            .born_sequence = 4,
            .record_xid = 22,
            .heap_block = 12,
            .document_length = 1,
            .heap_offset = 1,
            .flags = EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_QUARANTINED
        }
    };
    evoke_document_retirement_record retirement = {
        .document_slot = 0,
        .retirement_sequence = 100,
        .record_xid = 30,
        .document_length = 3
    };

    initialize_test_segment_manifest(&manifest);
    manifest.segments[0].retirement_count = 1;
    ASSERT_STATUS_OK(evoke_document_version_records_validate(
        &manifest,
        &manifest.segments[0],
        versions,
        4,
        &retirement,
        1
    ));

    versions[1].document_slot = 0;
    ASSERT_TRUE(evoke_document_version_records_validate(
        &manifest,
        &manifest.segments[0],
        versions,
        4,
        &retirement,
        1
    ) == EVOKE_ERR_FORMAT);
    versions[1].document_slot = 1;
    versions[1].flags |= EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE;
    ASSERT_TRUE(evoke_document_version_records_validate(
        &manifest,
        &manifest.segments[0],
        versions,
        4,
        &retirement,
        1
    ) == EVOKE_ERR_FORMAT);
    versions[1].flags = EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING;
    versions[1].record_xid = 0;
    ASSERT_TRUE(evoke_document_version_records_validate(
        &manifest,
        &manifest.segments[0],
        versions,
        4,
        &retirement,
        1
    ) == EVOKE_ERR_FORMAT);
    versions[1].record_xid = 21;
    versions[2].heap_offset = 1;
    ASSERT_TRUE(evoke_document_version_records_validate(
        &manifest,
        &manifest.segments[0],
        versions,
        4,
        &retirement,
        1
    ) == EVOKE_ERR_FORMAT);
    versions[2].heap_offset = 0;
    versions[2].semantic_input_fingerprint[0] = 1;
    ASSERT_TRUE(evoke_document_version_records_validate(
        &manifest,
        &manifest.segments[0],
        versions,
        4,
        &retirement,
        1
    ) == EVOKE_ERR_FORMAT);
    versions[2].semantic_input_fingerprint[0] = 0;
    versions[2].flags = EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE;
    ASSERT_TRUE(evoke_document_version_records_validate(
        &manifest,
        &manifest.segments[0],
        versions,
        4,
        &retirement,
        1
    ) == EVOKE_ERR_FORMAT);
    versions[2].flags = EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE |
        EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
    retirement.document_slot = manifest.document_slot_count;
    ASSERT_TRUE(evoke_document_version_records_validate(
        &manifest,
        &manifest.segments[0],
        versions,
        4,
        &retirement,
        1
    ) == EVOKE_ERR_FORMAT);
    retirement.document_slot = 0;
    retirement.reserved2 = 1;
    ASSERT_TRUE(evoke_document_version_records_validate(
        &manifest,
        &manifest.segments[0],
        versions,
        4,
        &retirement,
        1
    ) == EVOKE_ERR_FORMAT);
    retirement.reserved2 = 0;

    evoke_segment_manifest_free(&manifest);
}

static void
test_aborted_slot_hole_metadata(void)
{
    uint32_t live_doc[] = {0};
    evoke_doc_ids docs[] = {
        make_doc(live_doc, 1),
        make_doc(NULL, 0)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_document_version_record versions[2] = {0};
    evoke_segment_manifest manifest;
    evoke_segment_payload payload;
    evoke_segment_payload restored;
    evoke_segment_query_contract contract;
    evoke_index index;
    evoke_index metadata_index;
    uint8_t *bytes = NULL;
    size_t size = 0;
    uint64_t checksum = 0;

    evoke_index_init(&index);
    evoke_index_init(&metadata_index);
    evoke_segment_manifest_init(&manifest);
    evoke_segment_payload_init(&payload);
    evoke_segment_payload_init(&restored);
    evoke_segment_query_contract_init(&contract);

    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        2,
        &params,
        false,
        &index
    ));
    versions[0].document_slot = 0;
    versions[0].born_sequence = 1;
    versions[0].heap_block = 10;
    versions[0].document_length = 1;
    versions[0].heap_offset = 1;
    versions[0].flags = EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
    versions[1].document_slot = 1;
    versions[1].born_sequence = 2;
    versions[1].flags = EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE |
        EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical(
        &index,
        1,
        0,
        versions,
        2,
        &payload
    ));

    manifest.flags = EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    manifest.manifest_id = 1;
    manifest.max_sequence = 2;
    manifest.statistics_epoch = 1;
    manifest.visible_document_count = 1;
    manifest.document_slot_count = 2;
    manifest.total_document_length = 1;
    manifest.vocab_size = index.vocab_size;
    initialize_test_object_ref(
        &manifest.query_contract,
        EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT,
        3,
        1,
        manifest.manifest_id,
        100
    );
    initialize_test_object_ref(
        &manifest.term_directory,
        EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
        2,
        1,
        manifest.manifest_id,
        101
    );
    initialize_test_document_directory(&manifest, 1000, 102);
    manifest.segment_count = 1;
    manifest.segments = calloc(1, sizeof(*manifest.segments));
    manifest.doc_frequencies = calloc(
        index.vocab_size,
        sizeof(*manifest.doc_frequencies)
    );
    ASSERT_TRUE(manifest.segments != NULL);
    ASSERT_TRUE(manifest.doc_frequencies != NULL);
    memcpy(
        manifest.doc_frequencies,
        index.doc_frequencies,
        index.vocab_size * sizeof(*manifest.doc_frequencies)
    );
    manifest.segments[0].segment_id = 1;
    manifest.segments[0].min_sequence = 1;
    manifest.segments[0].max_sequence = 2;
    manifest.segments[0].posting_count = index.data_len;
    manifest.segments[0].document_count = 2;
    manifest.segments[0].total_document_length = 1;
    manifest.segments[0].document_slot_count = 2;
    manifest.segments[0].start_block = 1;
    manifest.segments[0].block_count = 1;
    manifest.segments[0].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL;
    manifest.segments[0].payload_owner_manifest_id =
        manifest.manifest_id;
    ASSERT_STATUS_OK(evoke_segment_payload_serialize(
        &payload,
        &manifest,
        &manifest.segments[0],
        &bytes,
        &size,
        &checksum
    ));
    manifest.segments[0].payload_bytes = size;
    manifest.segments[0].payload_checksum = checksum;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));
    ASSERT_STATUS_OK(evoke_segment_query_contract_build(
        &index,
        &manifest,
        &contract
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_deserialize(
        bytes,
        size,
        &manifest,
        &manifest.segments[0],
        &restored
    ));
    ASSERT_TRUE(
        restored.block_shift == EVOKE_DEFAULT_POSTING_BLOCK_SHIFT
    );
    ASSERT_TRUE(restored.block_count == restored.run_count);
    ASSERT_TRUE(restored.runs[0].block_count == 1);
    ASSERT_TRUE(
        restored.versions[1].flags ==
        (EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE |
         EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID)
    );
    ASSERT_STATUS_OK(evoke_segment_index_metadata_build(
        &contract,
        &manifest,
        manifest.doc_frequencies,
        &restored,
        1,
        &metadata_index
    ));
    ASSERT_TRUE(metadata_index.num_docs == 2);
    ASSERT_TRUE(metadata_index.doc_lengths[0] == 1);
    ASSERT_TRUE(metadata_index.doc_lengths[1] == 0);

    restored.indices[0] = 1;
    ASSERT_TRUE(evoke_segment_index_metadata_build(
        &contract,
        &manifest,
        manifest.doc_frequencies,
        &restored,
        1,
        &metadata_index
    ) == EVOKE_ERR_FORMAT);

    free(bytes);
    evoke_segment_query_contract_free(&contract);
    evoke_segment_payload_free(&restored);
    evoke_segment_payload_free(&payload);
    evoke_segment_manifest_free(&manifest);
    evoke_index_free(&metadata_index);
    evoke_index_free(&index);
}

static void
test_frozen_retirement_ids_build(void)
{
    evoke_segment_manifest manifest;
    evoke_segment_payload payloads[2] = {0};
    evoke_document_retirement_record retirements[2] = {
        {
            .document_slot = 4,
            .retirement_sequence = 50,
            .record_xid = 0,
            .flags = EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID
        },
        {
            .document_slot = 0,
            .retirement_sequence = 110,
            .record_xid = 0,
            .flags = EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID
        }
    };
    uint32_t *document_ids = NULL;
    size_t document_id_count = 0;

    initialize_test_segment_manifest(&manifest);
    manifest.segments[0].flags |= EVOKE_SEGMENT_FLAG_RETIREMENTS;
    manifest.segments[0].retirement_count = 1;
    payloads[0].segment_id = manifest.segments[0].segment_id;
    payloads[0].retirements = &retirements[0];
    payloads[0].retirement_count = 1;
    payloads[1].segment_id = manifest.segments[1].segment_id;
    payloads[1].retirements = &retirements[1];
    payloads[1].retirement_count = 1;

    ASSERT_STATUS_OK(evoke_segment_frozen_retirement_ids_build(
        &manifest,
        payloads,
        2,
        &document_ids,
        &document_id_count
    ));
    ASSERT_TRUE(document_id_count == 2);
    ASSERT_TRUE(document_ids[0] == 0);
    ASSERT_TRUE(document_ids[1] == 4);
    free(document_ids);
    document_ids = NULL;

    retirements[0].document_slot = 0;
    ASSERT_TRUE(evoke_segment_frozen_retirement_ids_build(
        &manifest,
        payloads,
        2,
        &document_ids,
        &document_id_count
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(document_ids == NULL);
    ASSERT_TRUE(document_id_count == 0);

    retirements[0].document_slot = 4;
    retirements[1].flags = 0;
    retirements[1].record_xid = 41;
    ASSERT_TRUE(evoke_segment_frozen_retirement_ids_build(
        &manifest,
        payloads,
        2,
        &document_ids,
        &document_id_count
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(document_ids == NULL);
    ASSERT_TRUE(document_id_count == 0);

    evoke_segment_manifest_free(&manifest);
}

static void
test_segment_payload_roundtrip_and_validation(void)
{
    evoke_segment_manifest manifest;
    evoke_segment_payload payload;
    evoke_segment_payload restored;
    evoke_segment_payload quarantine_restored;
    evoke_segment_term_run runs[] = {
        {
            .term_id = 0,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 1
        },
        {
            .term_id = 3,
            .kind = EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT,
            .posting_offset = 1,
            .posting_count = 1
        }
    };
    uint32_t indices[] = {0, 0};
    evoke_posting_value values[2];
    uint32_t document_map[] = {4};
    evoke_document_version_record version = {
        .document_slot = 4,
        .born_sequence = 101,
        .record_xid = 21,
        .heap_block = 22,
        .document_length = 3,
        .heap_offset = 2,
        .flags = EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING
    };
    evoke_document_retirement_record retirement = {
        .document_slot = 0,
        .retirement_sequence = 102,
        .record_xid = 21,
        .document_length = 3
    };
    evoke_semantic_state_record semantic_state = {
        .document_slot = 4,
        .transition_sequence = 103,
        .record_xid = 21,
        .flags = EVOKE_SEMANTIC_STATE_FLAG_COMPLETE
    };
    evoke_segment_payload_disk_header disk_header;
    evoke_segment_term_run disk_run;
    evoke_posting_block_record disk_block;
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    uint8_t *quarantine_bytes = NULL;
    size_t size = 0;
    size_t quarantine_size = 0;
    uint64_t checksum = 0;
    uint64_t quarantine_checksum = 0;

    memset(values, 0, sizeof(values));
    values[0].term_frequency = 2;
    values[1].impact = 0.375f;
    memset(
        version.semantic_input_fingerprint,
        0x5A,
        sizeof(version.semantic_input_fingerprint)
    );
    memcpy(
        semantic_state.semantic_input_fingerprint,
        version.semantic_input_fingerprint,
        sizeof(semantic_state.semantic_input_fingerprint)
    );

    initialize_test_segment_manifest(&manifest);
    manifest.segments[1].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL |
        EVOKE_SEGMENT_FLAG_SEMANTIC |
        EVOKE_SEGMENT_FLAG_RETIREMENTS;
    manifest.segments[1].semantic_state_count = 1;
    evoke_segment_payload_init(&payload);
    evoke_segment_payload_init(&restored);
    evoke_segment_payload_init(&quarantine_restored);
    payload.segment_id = 2;
    payload.vocab_size = 4;
    payload.flags = EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP;
    payload.local_document_count = 1;
    payload.run_count = 2;
    payload.posting_count = 2;
    payload.runs = runs;
    payload.indices = indices;
    payload.values = values;
    payload.document_id_map = document_map;
    payload.versions = &version;
    payload.version_count = 1;
    payload.retirements = &retirement;
    payload.retirement_count = 1;
    payload.semantic_states = &semantic_state;
    payload.semantic_state_count = 1;

    manifest.segments[1].payload_bytes = 0;
    manifest.segments[1].payload_checksum = 0;
    ASSERT_STATUS_OK(evoke_segment_payload_validate(
        &payload,
        &manifest,
        &manifest.segments[1]
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_serialize(
        &payload,
        &manifest,
        &manifest.segments[1],
        &bytes,
        &size,
        &checksum
    ));
    ASSERT_TRUE(bytes != NULL);
    ASSERT_TRUE(size > 144);
    ASSERT_TRUE(checksum != 0);
    manifest.segments[1].payload_bytes = size;
    manifest.segments[1].payload_checksum = checksum;
    ASSERT_STATUS_OK(evoke_segment_payload_disk_header_decode(
        bytes,
        EVOKE_SEGMENT_PAYLOAD_HEADER_SIZE,
        &manifest,
        &manifest.segments[1],
        &disk_header
    ));
    ASSERT_TRUE(disk_header.segment_id == 2);
    ASSERT_TRUE(disk_header.run_count == 2);
    ASSERT_TRUE(disk_header.payload_version == 7);
    ASSERT_TRUE(disk_header.posting_count == 2);
    ASSERT_TRUE(disk_header.generic_posting_count == 1);
    ASSERT_TRUE(disk_header.block_count == 1);
    ASSERT_TRUE(
        disk_header.semantic_bmp_version ==
        EVOKE_SEMANTIC_BMP_PACKED_FORMAT_VERSION
    );
    ASSERT_TRUE(disk_header.semantic_bmp_size > 0);
    ASSERT_TRUE(disk_header.total_size == size);
    ASSERT_STATUS_OK(evoke_segment_term_run_decode(
        bytes + disk_header.runs_offset + EVOKE_SEGMENT_TERM_RUN_SIZE,
        EVOKE_SEGMENT_TERM_RUN_SIZE,
        &disk_run
    ));
    ASSERT_TRUE(disk_run.term_id == 3);
    ASSERT_TRUE(
        disk_run.kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
    );
    ASSERT_TRUE(disk_run.posting_offset == 0);
    ASSERT_TRUE(disk_run.block_count == 0);
    ASSERT_STATUS_OK(evoke_posting_block_record_decode(
        bytes + disk_header.blocks_offset,
        EVOKE_POSTING_BLOCK_RECORD_SIZE,
        &disk_block
    ));
    ASSERT_TRUE(disk_block.kind ==
        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL);
    ASSERT_STATUS_OK(evoke_segment_payload_deserialize(
        bytes,
        size,
        &manifest,
        &manifest.segments[1],
        &restored
    ));
    ASSERT_TRUE(
        restored.block_shift == EVOKE_DEFAULT_POSTING_BLOCK_SHIFT
    );
    ASSERT_TRUE(restored.block_count == restored.run_count);
    ASSERT_TRUE(restored.runs[1].block_offset == 1);
    ASSERT_TRUE(restored.runs[1].block_count == 1);
    ASSERT_TRUE(restored.segment_id == payload.segment_id);
    ASSERT_TRUE(restored.flags == payload.flags);
    ASSERT_TRUE(restored.run_count == 2);
    ASSERT_TRUE(restored.posting_count == 2);
    ASSERT_TRUE(restored.values[0].term_frequency == 2);
    assert_float_close(restored.values[1].impact, 0.375f);
    ASSERT_TRUE(restored.document_id_map[0] == 4);
    ASSERT_TRUE(restored.versions[0].heap_block == 22);
    ASSERT_TRUE(restored.retirements[0].document_slot == 0);
    ASSERT_TRUE(restored.retirements[0].document_length == 3);
    ASSERT_TRUE(restored.semantic_state_count == 1);
    ASSERT_TRUE(
        restored.semantic_states[0].transition_sequence == 103
    );
    ASSERT_TRUE(
        restored.semantic_states[0].flags ==
        EVOKE_SEMANTIC_STATE_FLAG_COMPLETE
    );

    /* Historical packed segments keep their publication high-watermark. */
    manifest.document_slot_count++;
    manifest.visible_document_count++;
    manifest.total_document_length++;
    ASSERT_STATUS_OK(evoke_segment_payload_deserialize(
        bytes,
        size,
        &manifest,
        &manifest.segments[1],
        &restored
    ));
    manifest.document_slot_count--;
    manifest.visible_document_count--;
    manifest.total_document_length--;

    semantic_state.flags = EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED;
    semantic_state.failure_count = 2;
    semantic_state.error_code = 7;
    semantic_state.pending_since = 100;
    semantic_state.retry_after = 200;
    semantic_state.error_hash = UINT64_C(0x123456789abcdef0);
    payload.run_count = 1;
    payload.posting_count = 1;
    manifest.segments[1].posting_count = 1;
    manifest.segments[1].flags |=
        EVOKE_SEGMENT_FLAG_PENDING | EVOKE_SEGMENT_FLAG_QUARANTINE;
    manifest.segments[1].payload_bytes = 0;
    manifest.segments[1].payload_checksum = 0;
    ASSERT_STATUS_OK(evoke_segment_payload_serialize(
        &payload,
        &manifest,
        &manifest.segments[1],
        &quarantine_bytes,
        &quarantine_size,
        &quarantine_checksum
    ));
    manifest.segments[1].payload_bytes = quarantine_size;
    manifest.segments[1].payload_checksum = quarantine_checksum;
    ASSERT_STATUS_OK(evoke_segment_payload_deserialize(
        quarantine_bytes,
        quarantine_size,
        &manifest,
        &manifest.segments[1],
        &quarantine_restored
    ));
    ASSERT_TRUE(
        quarantine_restored.semantic_states[0].pending_since == 100
    );
    ASSERT_TRUE(
        quarantine_restored.semantic_states[0].retry_after == 200
    );
    ASSERT_TRUE(
        quarantine_restored.semantic_states[0].error_hash ==
        UINT64_C(0x123456789abcdef0)
    );
    quarantine_bytes[quarantine_size - 1] ^= 0x1U;
    ASSERT_TRUE(evoke_segment_payload_deserialize(
        quarantine_bytes,
        quarantine_size,
        &manifest,
        &manifest.segments[1],
        &quarantine_restored
    ) == EVOKE_ERR_FORMAT);
    quarantine_bytes[quarantine_size - 1] ^= 0x1U;
    semantic_state.flags = EVOKE_SEMANTIC_STATE_FLAG_COMPLETE;
    semantic_state.failure_count = 0;
    semantic_state.error_code = 0;
    semantic_state.pending_since = 0;
    semantic_state.retry_after = 0;
    semantic_state.error_hash = 0;
    payload.run_count = 2;
    payload.posting_count = 2;
    manifest.segments[1].posting_count = 2;
    manifest.segments[1].flags &=
        ~(EVOKE_SEGMENT_FLAG_PENDING | EVOKE_SEGMENT_FLAG_QUARANTINE);
    manifest.segments[1].payload_bytes = size;
    manifest.segments[1].payload_checksum = checksum;

    manifest.doc_frequencies = realloc(
        manifest.doc_frequencies,
        5 * sizeof(*manifest.doc_frequencies)
    );
    ASSERT_TRUE(manifest.doc_frequencies != NULL);
    manifest.doc_frequencies[4] = 0;
    manifest.vocab_size = 5;
    ASSERT_STATUS_OK(evoke_segment_payload_disk_header_decode(
        bytes,
        EVOKE_SEGMENT_PAYLOAD_HEADER_SIZE,
        &manifest,
        &manifest.segments[1],
        &disk_header
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_deserialize(
        bytes,
        size,
        &manifest,
        &manifest.segments[1],
        &restored
    ));
    runs[1].term_id = 4;
    ASSERT_TRUE(evoke_segment_payload_validate(
        &payload,
        &manifest,
        &manifest.segments[1]
    ) == EVOKE_ERR_FORMAT);
    runs[1].term_id = 3;

    mutated = malloc(size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, size);
    test_write_u16_le(mutated + 4, 6);
    ASSERT_TRUE(evoke_segment_payload_disk_header_decode(
        mutated,
        EVOKE_SEGMENT_PAYLOAD_HEADER_SIZE,
        &manifest,
        &manifest.segments[1],
        &disk_header
    ) == EVOKE_ERR_FORMAT);
    test_write_u16_le(mutated + 4, 7);
    mutated[size - 1] ^= 0x1U;
    ASSERT_TRUE(evoke_segment_payload_deserialize(
        mutated,
        size,
        &manifest,
        &manifest.segments[1],
        &restored
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_segment_payload_deserialize(
        bytes,
        size - 1,
        &manifest,
        &manifest.segments[1],
        &restored
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(restored.segment_id == 2);

    values[0].term_frequency = 0;
    ASSERT_TRUE(evoke_segment_payload_validate(
        &payload,
        &manifest,
        &manifest.segments[1]
    ) == EVOKE_ERR_FORMAT);
    values[0].term_frequency = 2;
    values[1].impact = NAN;
    ASSERT_TRUE(evoke_segment_payload_validate(
        &payload,
        &manifest,
        &manifest.segments[1]
    ) == EVOKE_ERR_FORMAT);
    values[1].impact = 0.375f;
    semantic_state.flags =
        EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED;
    semantic_state.failure_count = 1;
    semantic_state.error_code = 1;
    semantic_state.retry_after = 1;
    semantic_state.pending_since = 1;
    ASSERT_TRUE(evoke_segment_payload_validate(
        &payload,
        &manifest,
        &manifest.segments[1]
    ) == EVOKE_ERR_FORMAT);
    semantic_state.flags = EVOKE_SEMANTIC_STATE_FLAG_COMPLETE;
    semantic_state.failure_count = 0;
    semantic_state.error_code = 0;
    semantic_state.retry_after = 0;
    semantic_state.pending_since = 0;
    semantic_state.error_hash = 0;
    semantic_state.semantic_input_fingerprint[0] ^= 0x1U;
    ASSERT_TRUE(evoke_segment_payload_validate(
        &payload,
        &manifest,
        &manifest.segments[1]
    ) == EVOKE_ERR_FORMAT);
    semantic_state.semantic_input_fingerprint[0] ^= 0x1U;
    version.document_slot = 3;
    ASSERT_TRUE(evoke_segment_payload_validate(
        &payload,
        &manifest,
        &manifest.segments[1]
    ) == EVOKE_ERR_FORMAT);

    free(bytes);
    free(mutated);
    free(quarantine_bytes);
    evoke_segment_payload_free(&restored);
    evoke_segment_payload_free(&quarantine_restored);
    evoke_segment_manifest_free(&manifest);
}

static void
test_semantic_state_segment_and_merge(void)
{
    evoke_segment_manifest manifest;
    evoke_segment_descriptor merged_descriptor = {0};
    evoke_segment_payload payloads[2];
    evoke_segment_payload merged;
    evoke_segment_payload filtered;
    evoke_segment_payload restored;
    evoke_segment_term_run lexical_run = {
        .term_id = 0,
        .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
        .posting_offset = 0,
        .posting_count = 1
    };
    evoke_segment_term_run semantic_run = {
        .term_id = 1,
        .kind = EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT,
        .posting_offset = 0,
        .posting_count = 1
    };
    uint32_t lexical_index = 0;
    uint32_t semantic_index = 0;
    uint32_t semantic_document_map = 0;
    uint32_t excluded_document_id = 0;
    evoke_posting_value lexical_value = {.term_frequency = 1};
    evoke_posting_value semantic_value = {.impact = 0.5f};
    evoke_document_version_record version = {
        .document_slot = 0,
        .born_sequence = 1,
        .heap_block = 10,
        .document_length = 2,
        .heap_offset = 1,
        .flags =
            EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |
            EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING
    };
    evoke_semantic_state_record semantic_state = {
        .document_slot = 0,
        .transition_sequence = 2,
        .flags =
            EVOKE_SEMANTIC_STATE_FLAG_COMPLETE |
            EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID
    };
    uint8_t *bytes = NULL;
    size_t size = 0;
    uint64_t checksum = 0;

    memset(
        version.semantic_input_fingerprint,
        0x31,
        sizeof(version.semantic_input_fingerprint)
    );
    memcpy(
        semantic_state.semantic_input_fingerprint,
        version.semantic_input_fingerprint,
        sizeof(semantic_state.semantic_input_fingerprint)
    );
    evoke_segment_manifest_init(&manifest);
    evoke_segment_payload_init(&payloads[0]);
    evoke_segment_payload_init(&payloads[1]);
    evoke_segment_payload_init(&merged);
    evoke_segment_payload_init(&filtered);
    evoke_segment_payload_init(&restored);

    manifest.flags =
        EVOKE_SEGMENT_MANIFEST_FLAG_SAE |
        EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    manifest.manifest_id = 5;
    manifest.max_sequence = 2;
    manifest.statistics_epoch = 1;
    manifest.visible_document_count = 1;
    manifest.document_slot_count = 1;
    manifest.total_document_length = 2;
    manifest.vocab_size = 2;
    initialize_test_object_ref(
        &manifest.query_contract,
        EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT,
        10,
        1,
        manifest.manifest_id,
        101
    );
    initialize_test_object_ref(
        &manifest.term_directory,
        EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
        11,
        1,
        manifest.manifest_id,
        102
    );
    initialize_test_document_directory(&manifest, 1000, 103);
    manifest.segment_count = 2;
    manifest.segments = calloc(
        manifest.segment_count,
        sizeof(*manifest.segments)
    );
    manifest.doc_frequencies = calloc(
        manifest.vocab_size,
        sizeof(*manifest.doc_frequencies)
    );
    ASSERT_TRUE(manifest.segments != NULL);
    ASSERT_TRUE(manifest.doc_frequencies != NULL);
    manifest.doc_frequencies[0] = 1;

    manifest.segments[0].segment_id = 1;
    manifest.segments[0].min_sequence = 1;
    manifest.segments[0].max_sequence = 1;
    manifest.segments[0].posting_count = 1;
    manifest.segments[0].document_count = 1;
    manifest.segments[0].total_document_length = 2;
    manifest.segments[0].document_slot_count = 1;
    manifest.segments[0].payload_checksum = 201;
    manifest.segments[0].start_block = 1;
    manifest.segments[0].block_count = 1;
    manifest.segments[0].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL;
    manifest.segments[0].payload_bytes = 100;
    manifest.segments[0].payload_owner_manifest_id =
        manifest.manifest_id;

    manifest.segments[1].segment_id = 2;
    manifest.segments[1].min_sequence = 2;
    manifest.segments[1].max_sequence = 2;
    manifest.segments[1].posting_count = 1;
    manifest.segments[1].semantic_state_count = 1;
    manifest.segments[1].document_slot_count = 1;
    manifest.segments[1].payload_checksum = 202;
    manifest.segments[1].start_block = 2;
    manifest.segments[1].block_count = 1;
    manifest.segments[1].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_SEMANTIC;
    manifest.segments[1].payload_bytes = 100;
    manifest.segments[1].payload_owner_manifest_id =
        manifest.manifest_id;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));

    payloads[0].segment_id = 1;
    payloads[0].vocab_size = 2;
    payloads[0].local_document_count = 1;
    payloads[0].run_count = 1;
    payloads[0].posting_count = 1;
    payloads[0].runs = &lexical_run;
    payloads[0].indices = &lexical_index;
    payloads[0].values = &lexical_value;
    payloads[0].versions = &version;
    payloads[0].version_count = 1;

    payloads[1].segment_id = 2;
    payloads[1].vocab_size = 2;
    payloads[1].flags = EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP;
    payloads[1].local_document_count = 1;
    payloads[1].run_count = 1;
    payloads[1].posting_count = 1;
    payloads[1].runs = &semantic_run;
    payloads[1].indices = &semantic_index;
    payloads[1].values = &semantic_value;
    payloads[1].document_id_map = &semantic_document_map;
    payloads[1].semantic_states = &semantic_state;
    payloads[1].semantic_state_count = 1;

    ASSERT_STATUS_OK(evoke_segment_payload_validate(
        &payloads[0],
        &manifest,
        &manifest.segments[0]
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_validate(
        &payloads[1],
        &manifest,
        &manifest.segments[1]
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_serialize(
        &payloads[1],
        &manifest,
        &manifest.segments[1],
        &bytes,
        &size,
        &checksum
    ));
    manifest.segments[1].payload_bytes = size;
    manifest.segments[1].payload_checksum = checksum;
    ASSERT_STATUS_OK(evoke_segment_payload_deserialize(
        bytes,
        size,
        &manifest,
        &manifest.segments[1],
        &restored
    ));
    ASSERT_TRUE(restored.version_count == 0);
    ASSERT_TRUE(restored.local_document_count == 1);
    ASSERT_TRUE(restored.semantic_state_count == 1);
    ASSERT_TRUE(restored.document_id_map[0] == 0);

    ASSERT_STATUS_OK(evoke_segment_payload_merge(
        &manifest,
        0,
        payloads,
        2,
        3,
        &merged
    ));
    ASSERT_TRUE(merged.local_document_count == 1);
    ASSERT_TRUE(merged.version_count == 1);
    ASSERT_TRUE(merged.semantic_state_count == 1);
    ASSERT_TRUE(merged.run_count == 2);
    ASSERT_TRUE(merged.posting_count == 2);
    ASSERT_TRUE(merged.document_id_map[0] == 0);
    ASSERT_TRUE(
        merged.semantic_states[0].transition_sequence == 2
    );
    ASSERT_STATUS_OK(evoke_segment_payload_merge_excluding(
        &manifest,
        0,
        payloads,
        2,
        4,
        &excluded_document_id,
        1,
        &filtered
    ));
    ASSERT_TRUE(filtered.local_document_count == 0);
    ASSERT_TRUE(filtered.version_count == 0);
    ASSERT_TRUE(filtered.semantic_state_count == 0);
    ASSERT_TRUE(filtered.run_count == 0);
    ASSERT_TRUE(filtered.posting_count == 0);
    ASSERT_TRUE(filtered.document_id_map == NULL);
    merged_descriptor.segment_id = 4;
    merged_descriptor.min_sequence = 1;
    merged_descriptor.max_sequence = 2;
    merged_descriptor.flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_HISTORY_BARRIER;
    ASSERT_STATUS_OK(evoke_segment_payload_validate(
        &filtered,
        &manifest,
        &merged_descriptor
    ));

    merged_descriptor.segment_id = 3;
    merged_descriptor.min_sequence = 1;
    merged_descriptor.max_sequence = 2;
    merged_descriptor.posting_count = 2;
    merged_descriptor.document_count = 1;
    merged_descriptor.total_document_length = 2;
    merged_descriptor.document_slot_count = 1;
    merged_descriptor.semantic_state_count = 1;
    merged_descriptor.flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL |
        EVOKE_SEGMENT_FLAG_SEMANTIC;
    ASSERT_STATUS_OK(evoke_segment_payload_validate(
        &merged,
        &manifest,
        &merged_descriptor
    ));
    merged.semantic_states[0].semantic_input_fingerprint[0] ^= 0x1U;
    ASSERT_TRUE(evoke_segment_payload_validate(
        &merged,
        &manifest,
        &merged_descriptor
    ) == EVOKE_ERR_FORMAT);

    free(bytes);
    evoke_segment_payload_free(&restored);
    evoke_segment_payload_free(&filtered);
    evoke_segment_payload_free(&merged);
    evoke_segment_manifest_free(&manifest);
}

static void
test_contiguous_rebuild_payload_partition(void)
{
    evoke_segment_manifest manifest;
    evoke_segment_payload source;
    evoke_segment_payload *parts = NULL;
    uint32_t part_count = 0;
    uint32_t indices[] = {0, 1, 2, 3, 0, 2};
    evoke_segment_term_run runs[] = {
        {
            .term_id = 0,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 4
        },
        {
            .term_id = 3,
            .kind = EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT,
            .posting_offset = 4,
            .posting_count = 2
        }
    };

    evoke_segment_manifest_init(&manifest);
    evoke_segment_payload_init(&source);
    source.segment_id = 5;
    source.vocab_size = 4;
    source.document_id_base = 10;
    source.local_document_count = 4;
    source.run_count = 2;
    source.posting_count = 6;
    source.version_count = 4;
    source.runs = malloc(sizeof(runs));
    source.indices = malloc(sizeof(indices));
    source.values = calloc(6, sizeof(*source.values));
    source.versions = calloc(4, sizeof(*source.versions));
    ASSERT_TRUE(source.runs != NULL);
    ASSERT_TRUE(source.indices != NULL);
    ASSERT_TRUE(source.values != NULL);
    ASSERT_TRUE(source.versions != NULL);
    memcpy(source.runs, runs, sizeof(runs));
    memcpy(source.indices, indices, sizeof(indices));
    for (uint32_t document_index = 0; document_index < 4; document_index++)
    {
        source.versions[document_index].document_slot = 10 + document_index;
        source.versions[document_index].born_sequence = 11 + document_index;
        source.versions[document_index].document_length = 2;
        source.versions[document_index].heap_offset = 1;
        source.versions[document_index].flags =
            EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |
            EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE;
    }
    for (uint32_t posting_index = 0; posting_index < 4; posting_index++)
    {
        source.values[posting_index].term_frequency = 1;
    }
    source.values[4].impact = 0.5f;
    source.values[5].impact = 0.75f;

    ASSERT_STATUS_OK(evoke_segment_payload_partition_contiguous(
        &source,
        20,
        700,
        &parts,
        &part_count
    ));
    ASSERT_TRUE(source.segment_id == 0);
    ASSERT_TRUE(part_count == 2);
    ASSERT_TRUE(parts[0].segment_id == 20);
    ASSERT_TRUE(parts[0].document_id_base == 10);
    ASSERT_TRUE(parts[0].local_document_count == 2);
    ASSERT_TRUE(parts[0].posting_count == 3);
    ASSERT_TRUE(parts[1].segment_id == 21);
    ASSERT_TRUE(parts[1].document_id_base == 12);
    ASSERT_TRUE(parts[1].local_document_count == 2);
    ASSERT_TRUE(parts[1].posting_count == 3);
    ASSERT_TRUE(parts[0].indices[0] == 0);
    ASSERT_TRUE(parts[0].indices[1] == 1);
    ASSERT_TRUE(parts[1].indices[0] == 0);
    ASSERT_TRUE(parts[1].indices[1] == 1);

    manifest.manifest_id = 19;
    manifest.vocab_size = 4;
    manifest.document_slot_count = 14;
    manifest.segment_count = part_count;
    manifest.segments = calloc(part_count, sizeof(*manifest.segments));
    ASSERT_TRUE(manifest.segments != NULL);
    for (uint32_t part_index = 0; part_index < part_count; part_index++)
    {
        evoke_segment_descriptor *descriptor =
            &manifest.segments[part_index];
        uint8_t *bytes = NULL;
        size_t size = 0;
        uint64_t checksum = 0;

        descriptor->segment_id = parts[part_index].segment_id;
        descriptor->min_sequence =
            parts[part_index].versions[0].born_sequence;
        descriptor->max_sequence =
            parts[part_index].versions[
                parts[part_index].version_count - 1
            ].born_sequence;
        descriptor->first_document_slot =
            parts[part_index].document_id_base;
        descriptor->document_count =
            parts[part_index].local_document_count;
        descriptor->document_slot_count =
            parts[part_index].local_document_count;
        descriptor->posting_count = parts[part_index].posting_count;
        descriptor->total_document_length =
            (uint64_t) parts[part_index].local_document_count * 2;
        descriptor->flags =
            EVOKE_SEGMENT_FLAG_SEALED |
            EVOKE_SEGMENT_FLAG_LEXICAL |
            EVOKE_SEGMENT_FLAG_SEMANTIC;
        ASSERT_STATUS_OK(evoke_segment_payload_validate(
            &parts[part_index],
            &manifest,
            descriptor
        ));
        ASSERT_STATUS_OK(evoke_segment_payload_serialize(
            &parts[part_index],
            &manifest,
            descriptor,
            &bytes,
            &size,
            &checksum
        ));
        ASSERT_TRUE(bytes != NULL);
        ASSERT_TRUE(size <= 700);
        ASSERT_TRUE(checksum != 0);
        free(bytes);
        evoke_segment_payload_free(&parts[part_index]);
    }
    free(parts);
    evoke_segment_manifest_free(&manifest);
}

static void
test_sparse_contiguous_rebuild_payload_partition(void)
{
    evoke_segment_payload source;
    evoke_segment_payload *parts = NULL;
    uint32_t part_count = 0;
    uint32_t sparse_documents[] = {0, 300, 599};

    evoke_segment_payload_init(&source);
    source.segment_id = 30;
    source.vocab_size = 1;
    source.local_document_count = 600;
    source.run_count = 1;
    source.posting_count = 3;
    source.version_count = 600;
    source.runs = calloc(1, sizeof(*source.runs));
    source.indices = malloc(sizeof(sparse_documents));
    source.values = calloc(3, sizeof(*source.values));
    source.versions = calloc(600, sizeof(*source.versions));
    ASSERT_TRUE(source.runs != NULL);
    ASSERT_TRUE(source.indices != NULL);
    ASSERT_TRUE(source.values != NULL);
    ASSERT_TRUE(source.versions != NULL);
    source.runs[0].term_id = 0;
    source.runs[0].kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
    source.runs[0].posting_count = 3;
    memcpy(source.indices, sparse_documents, sizeof(sparse_documents));
    for (uint32_t posting_index = 0; posting_index < 3; posting_index++)
    {
        source.values[posting_index].term_frequency = 1;
    }
    for (uint32_t document_index = 0;
         document_index < source.local_document_count;
         document_index++)
    {
        source.versions[document_index].document_slot = document_index;
        source.versions[document_index].born_sequence = document_index + 1;
        source.versions[document_index].document_length = 1;
        source.versions[document_index].heap_offset = 1;
        source.versions[document_index].flags =
            EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
    }

    ASSERT_STATUS_OK(evoke_segment_payload_partition_contiguous(
        &source,
        31,
        29100,
        &parts,
        &part_count
    ));
    ASSERT_TRUE(part_count == 2);
    ASSERT_TRUE(parts[0].local_document_count < 600);
    ASSERT_TRUE(
        parts[0].local_document_count +
        parts[1].local_document_count == 600
    );
    for (uint32_t part_index = 0; part_index < part_count; part_index++)
    {
        evoke_segment_payload_free(&parts[part_index]);
    }
    free(parts);
}

typedef struct test_semantic_stream_reader
{
    const evoke_segment_semantic_posting *postings;
    size_t posting_count;
    size_t cursor;
} test_semantic_stream_reader;

static evoke_status
test_semantic_stream_read(
    void *context,
    evoke_segment_semantic_posting *posting_out
)
{
    test_semantic_stream_reader *reader = context;

    if (reader == NULL || posting_out == NULL ||
        reader->cursor >= reader->posting_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    *posting_out = reader->postings[reader->cursor++];
    return EVOKE_OK;
}

static evoke_status
test_semantic_stream_rewind(void *context)
{
    test_semantic_stream_reader *reader = context;

    if (reader == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    reader->cursor = 0;
    return EVOKE_OK;
}

static void
assert_semantic_payloads_equal(
    const evoke_segment_payload *left,
    const evoke_segment_payload *right
)
{
    ASSERT_TRUE(left->flags == right->flags);
    ASSERT_TRUE(left->run_count == right->run_count);
    ASSERT_TRUE(left->posting_count == right->posting_count);
    ASSERT_TRUE(left->local_document_count == right->local_document_count);
    ASSERT_TRUE(left->version_count == right->version_count);
    ASSERT_TRUE(
        left->semantic_state_count == right->semantic_state_count
    );
    ASSERT_TRUE(memcmp(
        left->runs,
        right->runs,
        left->run_count * sizeof(*left->runs)
    ) == 0);
    ASSERT_TRUE(memcmp(
        left->indices,
        right->indices,
        (size_t) left->posting_count * sizeof(*left->indices)
    ) == 0);
    ASSERT_TRUE(memcmp(
        left->values,
        right->values,
        (size_t) left->posting_count * sizeof(*left->values)
    ) == 0);
    ASSERT_TRUE(memcmp(
        left->document_id_map,
        right->document_id_map,
        left->local_document_count * sizeof(*left->document_id_map)
    ) == 0);
    ASSERT_TRUE(memcmp(
        left->versions,
        right->versions,
        left->version_count * sizeof(*left->versions)
    ) == 0);
    ASSERT_TRUE(memcmp(
        left->semantic_states,
        right->semantic_states,
        left->semantic_state_count * sizeof(*left->semantic_states)
    ) == 0);
}

static void
test_attach_sorted_semantic_stream_matches_array(void)
{
    uint32_t terms[] = {0, 2};
    evoke_doc_ids document = make_doc(terms, 2);
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_document_version_record version = {
        .document_slot = 4,
        .born_sequence = 4,
        .heap_block = 44,
        .document_length = 2,
        .heap_offset = 1,
        .flags =
            EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |
            EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING
    };
    evoke_segment_semantic_posting unsorted_postings[] = {
        {.term_id = 2, .document_slot = 1, .impact = 0.4f},
        {.term_id = 1, .document_slot = 1, .impact = 0.8f}
    };
    evoke_segment_semantic_posting sorted_postings[] = {
        {.term_id = 1, .document_slot = 1, .impact = 0.8f},
        {.term_id = 2, .document_slot = 1, .impact = 0.4f}
    };
    evoke_segment_semantic_posting invalid_postings[] = {
        {.term_id = 2, .document_slot = 1, .impact = 0.4f},
        {.term_id = 1, .document_slot = 1, .impact = 0.8f}
    };
    evoke_semantic_state_record states[] = {
        {
            .document_slot = 1,
            .transition_sequence = 2,
            .flags =
                EVOKE_SEMANTIC_STATE_FLAG_COMPLETE |
                EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID
        }
    };
    test_semantic_stream_reader sorted_reader = {
        .postings = sorted_postings,
        .posting_count = 2
    };
    test_semantic_stream_reader invalid_reader = {
        .postings = invalid_postings,
        .posting_count = 2
    };
    test_semantic_stream_reader short_reader = {
        .postings = sorted_postings,
        .posting_count = 1
    };
    evoke_index index;
    evoke_segment_payload array_payload;
    evoke_segment_payload stream_payload;
    evoke_segment_payload invalid_payload;
    evoke_segment_payload short_payload;

    evoke_index_init(&index);
    evoke_segment_payload_init(&array_payload);
    evoke_segment_payload_init(&stream_payload);
    evoke_segment_payload_init(&invalid_payload);
    evoke_segment_payload_init(&short_payload);
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        &document,
        1,
        &params,
        false,
        &index
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical(
        &index,
        10,
        4,
        &version,
        1,
        &array_payload
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical(
        &index,
        10,
        4,
        &version,
        1,
        &stream_payload
    ));
    invalid_payload.segment_id = 11;
    invalid_payload.vocab_size = index.vocab_size;
    short_payload.segment_id = 12;
    short_payload.vocab_size = index.vocab_size;

    ASSERT_STATUS_OK(evoke_segment_payload_attach_semantic(
        &array_payload,
        unsorted_postings,
        2,
        states,
        1
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_attach_semantic_sorted_reader(
        &stream_payload,
        2,
        test_semantic_stream_read,
        test_semantic_stream_rewind,
        &sorted_reader,
        states,
        1
    ));
    assert_semantic_payloads_equal(&array_payload, &stream_payload);

    ASSERT_TRUE(evoke_segment_payload_attach_semantic_sorted_reader(
        &invalid_payload,
        2,
        test_semantic_stream_read,
        test_semantic_stream_rewind,
        &invalid_reader,
        states,
        1
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_segment_payload_attach_semantic_sorted_reader(
        &short_payload,
        2,
        test_semantic_stream_read,
        test_semantic_stream_rewind,
        &short_reader,
        states,
        1
    ) == EVOKE_ERR_FORMAT);

    evoke_segment_payload_free(&short_payload);
    evoke_segment_payload_free(&invalid_payload);
    evoke_segment_payload_free(&stream_payload);
    evoke_segment_payload_free(&array_payload);
    evoke_index_free(&index);
}

static void
test_initial_fold_stream_matches_combined_payload(void)
{
    uint32_t terms0[] = {0, 3};
    uint32_t terms1[] = {1, 3};
    uint32_t terms2[] = {0, 2};
    evoke_doc_ids documents[] = {
        make_doc(terms0, 2),
        make_doc(terms1, 2),
        make_doc(terms2, 2)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_document_version_record versions[3] = {0};
    evoke_segment_semantic_posting semantic_postings[] = {
        {.term_id = 0, .document_slot = 1, .impact = 0.2f},
        {.term_id = 1, .document_slot = 0, .impact = 0.7f},
        {.term_id = 1, .document_slot = 2, .impact = 0.4f},
        {.term_id = 3, .document_slot = 2, .impact = 0.9f}
    };
    evoke_segment_semantic_posting duplicate_postings[] = {
        {.term_id = 1, .document_slot = 0, .impact = 0.7f},
        {.term_id = 1, .document_slot = 0, .impact = 0.4f}
    };
    test_semantic_stream_reader payload_reader = {
        .postings = semantic_postings,
        .posting_count = 4
    };
    test_semantic_stream_reader stream_reader = {
        .postings = semantic_postings,
        .posting_count = 4
    };
    test_semantic_stream_reader split_reader = {
        .postings = semantic_postings,
        .posting_count = 4
    };
    test_semantic_stream_reader duplicate_reader = {
        .postings = duplicate_postings,
        .posting_count = 2
    };
    evoke_index index;
    evoke_segment_payload payload;
    evoke_initial_fold_stream *stream = NULL;
    evoke_initial_fold_stream *split_stream = NULL;
    evoke_initial_fold_stream *duplicate_stream = NULL;
    evoke_term_fold_bundle expected;
    evoke_term_fold_bundle actual;
    uint8_t *expected_bytes = NULL;
    uint8_t *actual_bytes = NULL;
    size_t expected_size = 0;
    size_t actual_size = 0;
    uint64_t expected_checksum = 0;
    uint64_t actual_checksum = 0;
    bool done = false;
    uint32_t split_group_count = 0;
    uint32_t split_run_count = 0;

    evoke_index_init(&index);
    evoke_segment_payload_init(&payload);
    evoke_term_fold_bundle_init(&expected);
    evoke_term_fold_bundle_init(&actual);
    for (uint32_t document_slot = 0;
         document_slot < 3;
         document_slot++)
    {
        versions[document_slot].document_slot = document_slot;
        versions[document_slot].born_sequence = document_slot + 1;
        versions[document_slot].heap_block = document_slot + 10;
        versions[document_slot].heap_offset = 1;
        versions[document_slot].document_length = 2;
        versions[document_slot].flags =
            EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |
            EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE;
    }
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        documents,
        3,
        &params,
        false,
        &index
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical(
        &index,
        8,
        0,
        versions,
        3,
        &payload
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_attach_semantic_sorted_reader(
        &payload,
        4,
        test_semantic_stream_read,
        test_semantic_stream_rewind,
        &payload_reader,
        NULL,
        0
    ));
    ASSERT_STATUS_OK(evoke_initial_fold_stream_create(
        &index,
        4,
        test_semantic_stream_read,
        test_semantic_stream_rewind,
        &stream_reader,
        99,
        3,
        SIZE_MAX,
        &stream
    ));
    ASSERT_STATUS_OK(evoke_initial_fold_stream_next(
        stream,
        &actual,
        &done
    ));
    ASSERT_TRUE(!done);
    ASSERT_TRUE(actual.run_count == payload.run_count);
    ASSERT_TRUE(actual.posting_count == payload.posting_count);

    expected.object_kind = EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    expected.owner_manifest_id = 99;
    expected.run_count = payload.run_count;
    expected.posting_count = payload.posting_count;
    expected.runs = calloc(expected.run_count, sizeof(*expected.runs));
    expected.document_slots = payload.indices;
    expected.values = payload.values;
    ASSERT_TRUE(expected.runs != NULL);
    for (uint32_t run_index = 0;
         run_index < expected.run_count;
         run_index++)
    {
        expected.runs[run_index].term_id = payload.runs[run_index].term_id;
        expected.runs[run_index].kind = payload.runs[run_index].kind;
        expected.runs[run_index].coverage_sequence = 3;
        expected.runs[run_index].posting_offset =
            payload.runs[run_index].posting_offset;
        expected.runs[run_index].posting_count =
            payload.runs[run_index].posting_count;
    }
    ASSERT_TRUE(memcmp(
        expected.runs,
        actual.runs,
        expected.run_count * sizeof(*expected.runs)
    ) == 0);
    ASSERT_TRUE(memcmp(
        expected.document_slots,
        actual.document_slots,
        (size_t) expected.posting_count *
            sizeof(*expected.document_slots)
    ) == 0);
    ASSERT_TRUE(memcmp(
        expected.values,
        actual.values,
        (size_t) expected.posting_count * sizeof(*expected.values)
    ) == 0);
    ASSERT_STATUS_OK(evoke_term_fold_bundle_serialize(
        &expected,
        &expected_bytes,
        &expected_size,
        &expected_checksum
    ));
    ASSERT_STATUS_OK(evoke_term_fold_bundle_serialize(
        &actual,
        &actual_bytes,
        &actual_size,
        &actual_checksum
    ));
    ASSERT_TRUE(expected_size == actual_size);
    ASSERT_TRUE(expected_checksum == actual_checksum);
    ASSERT_TRUE(memcmp(expected_bytes, actual_bytes, actual_size) == 0);
    evoke_term_fold_bundle_free(&actual);
    ASSERT_STATUS_OK(evoke_initial_fold_stream_next(
        stream,
        &actual,
        &done
    ));
    ASSERT_TRUE(done);

    ASSERT_STATUS_OK(evoke_initial_fold_stream_create(
        &index,
        4,
        test_semantic_stream_read,
        test_semantic_stream_rewind,
        &split_reader,
        99,
        3,
        EVOKE_TERM_FOLD_HEADER_SIZE + 1,
        &split_stream
    ));
    do
    {
        evoke_term_fold_bundle group;

        evoke_term_fold_bundle_init(&group);
        ASSERT_STATUS_OK(evoke_initial_fold_stream_next(
            split_stream,
            &group,
            &done
        ));
        if (!done)
        {
            split_group_count++;
            split_run_count += group.run_count;
            for (uint32_t run_index = 1;
                 run_index < group.run_count;
                 run_index++)
            {
                ASSERT_TRUE(
                    group.runs[run_index - 1].term_id ==
                        group.runs[run_index].term_id
                );
            }
        }
        evoke_term_fold_bundle_free(&group);
    } while (!done);
    ASSERT_TRUE(split_group_count == index.vocab_size);
    ASSERT_TRUE(split_run_count == payload.run_count);

    ASSERT_STATUS_OK(evoke_initial_fold_stream_create(
        &index,
        2,
        test_semantic_stream_read,
        test_semantic_stream_rewind,
        &duplicate_reader,
        99,
        3,
        SIZE_MAX,
        &duplicate_stream
    ));
    ASSERT_TRUE(evoke_initial_fold_stream_next(
        duplicate_stream,
        &actual,
        &done
    ) == EVOKE_ERR_FORMAT);

    evoke_initial_fold_stream_free(duplicate_stream);
    evoke_initial_fold_stream_free(split_stream);
    evoke_initial_fold_stream_free(stream);
    free(actual_bytes);
    free(expected_bytes);
    free(expected.runs);
    expected.runs = NULL;
    expected.document_slots = NULL;
    expected.values = NULL;
    evoke_segment_payload_free(&payload);
    evoke_index_free(&index);
}

static void
test_attach_semantic_postings_to_lexical_payload(void)
{
    uint32_t terms[] = {0, 2};
    evoke_doc_ids document = make_doc(terms, 2);
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_document_version_record version = {
        .document_slot = 4,
        .born_sequence = 4,
        .heap_block = 44,
        .document_length = 2,
        .heap_offset = 1,
        .flags =
            EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |
            EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING
    };
    evoke_segment_semantic_posting postings[] = {
        {.term_id = 2, .document_slot = 1, .impact = 0.4f},
        {.term_id = 1, .document_slot = 1, .impact = 0.8f}
    };
    evoke_segment_semantic_posting initial_posting = {
        .term_id = 1,
        .document_slot = 4,
        .impact = 0.6f
    };
    evoke_semantic_state_record states[2] = {
        {
            .document_slot = 3,
            .transition_sequence = 3,
            .error_code = 9,
            .retry_after = 100,
            .pending_since = 90,
            .flags =
                EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED |
                EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID,
            .failure_count = 1
        },
        {
            .document_slot = 1,
            .transition_sequence = 2,
            .flags =
                EVOKE_SEMANTIC_STATE_FLAG_COMPLETE |
                EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID
        }
    };
    evoke_semantic_state_record quarantine = {
        .document_slot = 7,
        .transition_sequence = 5,
        .error_code = 11,
        .retry_after = 200,
        .pending_since = 190,
        .flags =
            EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED |
            EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID,
        .failure_count = 2
    };
    evoke_index index;
    evoke_segment_manifest manifest;
    evoke_segment_descriptor descriptor = {0};
    evoke_segment_descriptor retirement_descriptor = {0};
    evoke_segment_descriptor semantic_descriptor = {0};
    evoke_segment_payload payload;
    evoke_segment_payload quarantine_payload;
    evoke_segment_payload initial_payload;
    evoke_segment_payload retirement_payload;
    evoke_segment_payload semantic_only_payload;

    memset(
        version.semantic_input_fingerprint,
        0x41,
        sizeof(version.semantic_input_fingerprint)
    );
    memset(
        states[0].semantic_input_fingerprint,
        0x43,
        sizeof(states[0].semantic_input_fingerprint)
    );
    memset(
        states[1].semantic_input_fingerprint,
        0x42,
        sizeof(states[1].semantic_input_fingerprint)
    );
    memset(
        quarantine.semantic_input_fingerprint,
        0x44,
        sizeof(quarantine.semantic_input_fingerprint)
    );
    evoke_index_init(&index);
    evoke_segment_manifest_init(&manifest);
    evoke_segment_payload_init(&payload);
    evoke_segment_payload_init(&quarantine_payload);
    evoke_segment_payload_init(&initial_payload);
    evoke_segment_payload_init(&retirement_payload);
    evoke_segment_payload_init(&semantic_only_payload);
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        &document,
        1,
        &params,
        false,
        &index
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical(
        &index,
        10,
        4,
        &version,
        1,
        &payload
    ));
    version.flags =
        EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |
        EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE;
    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical(
        &index,
        12,
        4,
        &version,
        1,
        &initial_payload
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_attach_semantic(
        &initial_payload,
        &initial_posting,
        1,
        NULL,
        0
    ));
    ASSERT_TRUE(initial_payload.semantic_state_count == 0);
    ASSERT_TRUE(initial_payload.run_count == 3);
    ASSERT_TRUE(initial_payload.posting_count == 3);
    version.flags =
        EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |
        EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING;
    ASSERT_STATUS_OK(evoke_segment_payload_attach_semantic(
        &payload,
        postings,
        2,
        states,
        2
    ));

    ASSERT_TRUE(
        (payload.flags & EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP) != 0
    );
    ASSERT_TRUE(payload.document_id_base == 0);
    ASSERT_TRUE(payload.local_document_count == 3);
    ASSERT_TRUE(payload.document_id_map[0] == 1);
    ASSERT_TRUE(payload.document_id_map[1] == 3);
    ASSERT_TRUE(payload.document_id_map[2] == 4);
    ASSERT_TRUE(payload.version_count == 1);
    ASSERT_TRUE(payload.semantic_state_count == 2);
    ASSERT_TRUE(payload.semantic_states[0].document_slot == 1);
    ASSERT_TRUE(payload.semantic_states[1].document_slot == 3);
    ASSERT_TRUE(payload.run_count == 4);
    ASSERT_TRUE(payload.posting_count == 4);
    ASSERT_TRUE(payload.runs[0].term_id == 0);
    ASSERT_TRUE(
        payload.runs[0].kind ==
        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL
    );
    ASSERT_TRUE(payload.indices[payload.runs[0].posting_offset] == 2);
    ASSERT_TRUE(
        payload.values[payload.runs[0].posting_offset].
            term_frequency == 1
    );
    ASSERT_TRUE(payload.runs[1].term_id == 1);
    ASSERT_TRUE(
        payload.runs[1].kind ==
        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
    );
    ASSERT_TRUE(payload.indices[payload.runs[1].posting_offset] == 0);
    ASSERT_TRUE(payload.runs[2].term_id == 2);
    ASSERT_TRUE(
        payload.runs[2].kind ==
        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL
    );
    ASSERT_TRUE(payload.indices[payload.runs[2].posting_offset] == 2);
    ASSERT_TRUE(payload.runs[3].term_id == 2);
    ASSERT_TRUE(
        payload.runs[3].kind ==
        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
    );
    ASSERT_TRUE(payload.indices[payload.runs[3].posting_offset] == 0);
    manifest.flags = EVOKE_SEGMENT_MANIFEST_FLAG_SAE;
    manifest.manifest_id = 2;
    manifest.max_sequence = 4;
    manifest.statistics_epoch = 1;
    manifest.visible_document_count = 1;
    manifest.document_slot_count = 5;
    manifest.total_document_length = 2;
    manifest.vocab_size = index.vocab_size;
    descriptor.segment_id = 10;
    descriptor.min_sequence = 2;
    descriptor.max_sequence = 4;
    descriptor.posting_count = payload.posting_count;
    descriptor.document_count = 1;
    descriptor.total_document_length = 2;
    descriptor.document_slot_count = payload.local_document_count;
    descriptor.semantic_state_count = 2;
    descriptor.flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL |
        EVOKE_SEGMENT_FLAG_SEMANTIC |
        EVOKE_SEGMENT_FLAG_PENDING |
        EVOKE_SEGMENT_FLAG_QUARANTINE;
    ASSERT_STATUS_OK(evoke_segment_payload_validate(
        &payload,
        &manifest,
        &descriptor
    ));

    quarantine_payload.segment_id = 11;
    quarantine_payload.vocab_size = index.vocab_size;
    ASSERT_STATUS_OK(evoke_segment_payload_attach_semantic(
        &quarantine_payload,
        NULL,
        0,
        &quarantine,
        1
    ));
    ASSERT_TRUE(quarantine_payload.local_document_count == 1);
    ASSERT_TRUE(quarantine_payload.document_id_map[0] == 7);
    ASSERT_TRUE(quarantine_payload.run_count == 0);
    ASSERT_TRUE(quarantine_payload.posting_count == 0);
    ASSERT_TRUE(quarantine_payload.semantic_state_count == 1);

    semantic_only_payload.segment_id = 13;
    semantic_only_payload.vocab_size = index.vocab_size;
    ASSERT_STATUS_OK(evoke_segment_payload_attach_semantic(
        &semantic_only_payload,
        postings,
        2,
        &states[1],
        1
    ));
    ASSERT_TRUE(semantic_only_payload.local_document_count == 1);
    ASSERT_TRUE(semantic_only_payload.document_id_map[0] == 1);
    ASSERT_TRUE(semantic_only_payload.version_count == 0);
    ASSERT_TRUE(semantic_only_payload.semantic_state_count == 1);
    semantic_descriptor.segment_id = 13;
    semantic_descriptor.min_sequence = 2;
    semantic_descriptor.max_sequence = 2;
    semantic_descriptor.posting_count =
        semantic_only_payload.posting_count;
    semantic_descriptor.semantic_state_count = 1;
    semantic_descriptor.document_slot_count = 1;
    semantic_descriptor.flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_SEMANTIC;
    ASSERT_STATUS_OK(evoke_segment_payload_validate(
        &semantic_only_payload,
        &manifest,
        &semantic_descriptor
    ));

    retirement_payload.segment_id = 14;
    retirement_payload.vocab_size = index.vocab_size;
    retirement_payload.document_id_base = 5;
    ASSERT_STATUS_OK(evoke_segment_payload_attach_semantic(
        &retirement_payload,
        NULL,
        0,
        NULL,
        0
    ));
    ASSERT_TRUE(retirement_payload.document_id_map == NULL);
    ASSERT_TRUE(retirement_payload.flags == 0);
    retirement_payload.retirements = calloc(
        1,
        sizeof(*retirement_payload.retirements)
    );
    ASSERT_TRUE(retirement_payload.retirements != NULL);
    retirement_payload.retirement_count = 1;
    retirement_payload.retirements[0].document_slot = 2;
    retirement_payload.retirements[0].retirement_sequence = 3;
    retirement_payload.retirements[0].document_length = 2;
    retirement_payload.retirements[0].flags =
        EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID;
    retirement_descriptor.segment_id = 14;
    retirement_descriptor.min_sequence = 3;
    retirement_descriptor.max_sequence = 3;
    retirement_descriptor.retirement_count = 1;
    retirement_descriptor.first_document_slot = 5;
    retirement_descriptor.flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_RETIREMENTS;
    ASSERT_STATUS_OK(evoke_segment_payload_validate(
        &retirement_payload,
        &manifest,
        &retirement_descriptor
    ));

    ASSERT_TRUE(evoke_segment_payload_attach_semantic(
        &quarantine_payload,
        postings,
        1,
        NULL,
        0
    ) == EVOKE_ERR_INVALID);

    evoke_segment_payload_free(&initial_payload);
    evoke_segment_payload_free(&semantic_only_payload);
    evoke_segment_payload_free(&retirement_payload);
    evoke_segment_payload_free(&quarantine_payload);
    evoke_segment_payload_free(&payload);
    evoke_segment_manifest_free(&manifest);
    evoke_index_free(&index);
}

static void
test_sparse_lexical_entries_build_canonical_segment(void)
{
    evoke_term_entry entries[] = {
        {.token_id = 999999, .doc_id = 1, .tf = 2},
        {.token_id = 7, .doc_id = 0, .tf = 1},
        {.token_id = 999999, .doc_id = 0, .tf = 1}
    };
    evoke_term_entry duplicate_entries[] = {
        {.token_id = 7, .doc_id = 0, .tf = 1},
        {.token_id = 7, .doc_id = 0, .tf = 1}
    };
    uint32_t document_id_map[] = {4, 101};
    uint32_t unsorted_document_id_map[] = {101, 4};
    evoke_document_version_record versions[2] = {0};
    evoke_segment_payload payload;

    versions[0].document_slot = 100;
    versions[0].born_sequence = 1;
    versions[0].document_length = 1;
    versions[0].flags = EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
    versions[1].document_slot = 101;
    versions[1].born_sequence = 2;
    versions[1].document_length = 3;
    versions[1].flags = EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
    evoke_segment_payload_init(&payload);

    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical_entries(
        entries,
        sizeof(entries) / sizeof(entries[0]),
        1000000,
        9,
        100,
        versions,
        sizeof(versions) / sizeof(versions[0]),
        &payload
    ));
    ASSERT_TRUE(payload.vocab_size == 1000000);
    ASSERT_TRUE(payload.run_count == 2);
    ASSERT_TRUE(payload.posting_count == 3);
    ASSERT_TRUE(payload.runs[0].term_id == 7);
    ASSERT_TRUE(payload.runs[0].posting_offset == 0);
    ASSERT_TRUE(payload.runs[0].posting_count == 1);
    ASSERT_TRUE(payload.runs[1].term_id == 999999);
    ASSERT_TRUE(payload.runs[1].posting_offset == 1);
    ASSERT_TRUE(payload.runs[1].posting_count == 2);
    ASSERT_TRUE(payload.indices[0] == 0);
    ASSERT_TRUE(payload.indices[1] == 0);
    ASSERT_TRUE(payload.indices[2] == 1);
    ASSERT_TRUE(payload.values[2].term_frequency == 2);
    ASSERT_TRUE(payload.version_count == 2);
    ASSERT_TRUE(payload.versions[1].document_slot == 101);
    evoke_segment_payload_free(&payload);

    versions[0].document_slot = document_id_map[0];
    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical_mapped_entries(
        entries,
        sizeof(entries) / sizeof(entries[0]),
        1000000,
        10,
        document_id_map,
        versions,
        sizeof(versions) / sizeof(versions[0]),
        &payload
    ));
    ASSERT_TRUE(
        (payload.flags & EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP) != 0
    );
    ASSERT_TRUE(payload.document_id_base == 0);
    ASSERT_TRUE(payload.local_document_count == 2);
    ASSERT_TRUE(payload.document_id_map[0] == 4);
    ASSERT_TRUE(payload.document_id_map[1] == 101);
    ASSERT_TRUE(payload.versions[0].document_slot == 4);
    ASSERT_TRUE(payload.versions[1].document_slot == 101);
    ASSERT_TRUE(payload.indices[0] == 0);
    ASSERT_TRUE(payload.indices[2] == 1);
    evoke_segment_payload_free(&payload);

    ASSERT_TRUE(evoke_segment_payload_build_lexical_mapped_entries(
        entries,
        sizeof(entries) / sizeof(entries[0]),
        1000000,
        11,
        unsorted_document_id_map,
        versions,
        sizeof(versions) / sizeof(versions[0]),
        &payload
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(payload.run_count == 0);

    ASSERT_TRUE(evoke_segment_payload_build_lexical_entries(
        duplicate_entries,
        sizeof(duplicate_entries) / sizeof(duplicate_entries[0]),
        8,
        10,
        100,
        versions,
        sizeof(versions) / sizeof(versions[0]),
        &payload
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(payload.run_count == 0);
}

static void
test_lexical_index_builds_canonical_segment(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    uint32_t query[] = {0, 2, 3};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_document_version_record versions[4];
    evoke_segment_manifest manifest;
    evoke_segment_payload payload;
    evoke_segment_payload restored;
    evoke_segment_payload_view payload_view;
    evoke_segment_query_contract contract;
    evoke_term_directory directory;
    evoke_segment_read_view read_view;
    evoke_index index;
    evoke_index metadata_index;
    evoke_corpus_stats stats;
    uint8_t *bytes = NULL;
    size_t size = 0;
    uint64_t checksum = 0;
    uint64_t total_document_length = 0;
    float *expected_scores = NULL;
    float *segmented_scores = NULL;
    uint32_t document_id;

    evoke_index_init(&index);
    evoke_segment_manifest_init(&manifest);
    evoke_segment_payload_init(&payload);
    evoke_segment_payload_init(&restored);
    evoke_segment_query_contract_init(&contract);
    evoke_term_directory_init(&directory);
    evoke_segment_read_view_init(&read_view);
    evoke_index_init(&metadata_index);
    memset(versions, 0, sizeof(versions));

    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        4,
        &params,
        false,
        &index
    ));
    for (document_id = 0; document_id < index.num_docs; document_id++)
    {
        versions[document_id].document_slot = document_id;
        versions[document_id].born_sequence = document_id + 1;
        versions[document_id].heap_block = 20 + document_id;
        versions[document_id].heap_offset = 1;
        versions[document_id].document_length =
            index.doc_lengths[document_id];
        versions[document_id].flags =
            EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
        total_document_length += index.doc_lengths[document_id];
    }
    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical(
        &index,
        1,
        0,
        versions,
        4,
        &payload
    ));
    ASSERT_TRUE(payload.run_count == 4);
    ASSERT_TRUE(payload.posting_count == index.data_len);
    ASSERT_TRUE(payload.version_count == index.num_docs);
    ASSERT_TRUE(payload.runs[2].term_id == 2);
    ASSERT_TRUE(
        payload.runs[2].kind ==
        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL
    );
    ASSERT_TRUE(
        payload.values[0].term_frequency ==
        index.term_frequencies[0]
    );

    manifest.flags = EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    manifest.manifest_id = 1;
    manifest.max_sequence = 4;
    manifest.statistics_epoch = 1;
    manifest.visible_document_count = index.num_docs;
    manifest.document_slot_count = index.num_docs;
    manifest.total_document_length = total_document_length;
    manifest.vocab_size = index.vocab_size;
    initialize_test_object_ref(
        &manifest.query_contract,
        EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT,
        3,
        1,
        manifest.manifest_id,
        100
    );
    initialize_test_object_ref(
        &manifest.term_directory,
        EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
        2,
        1,
        manifest.manifest_id,
        101
    );
    initialize_test_document_directory(&manifest, 1000, 102);
    manifest.segment_count = 1;
    manifest.segments = calloc(1, sizeof(*manifest.segments));
    manifest.doc_frequencies = calloc(
        index.vocab_size,
        sizeof(*manifest.doc_frequencies)
    );
    ASSERT_TRUE(manifest.segments != NULL);
    ASSERT_TRUE(manifest.doc_frequencies != NULL);
    memcpy(
        manifest.doc_frequencies,
        index.doc_frequencies,
        index.vocab_size * sizeof(*manifest.doc_frequencies)
    );
    manifest.segments[0].segment_id = 1;
    manifest.segments[0].min_sequence = 1;
    manifest.segments[0].max_sequence = 4;
    manifest.segments[0].posting_count = index.data_len;
    manifest.segments[0].document_count = index.num_docs;
    manifest.segments[0].total_document_length =
        total_document_length;
    manifest.segments[0].document_slot_count = index.num_docs;
    manifest.segments[0].start_block = 1;
    manifest.segments[0].block_count = 1;
    manifest.segments[0].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL;
    manifest.segments[0].payload_owner_manifest_id =
        manifest.manifest_id;

    ASSERT_STATUS_OK(evoke_segment_payload_serialize(
        &payload,
        &manifest,
        &manifest.segments[0],
        &bytes,
        &size,
        &checksum
    ));
    manifest.segments[0].payload_bytes = size;
    manifest.segments[0].payload_checksum = checksum;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));
    ASSERT_STATUS_OK(evoke_segment_query_contract_build(
        &index,
        &manifest,
        &contract
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_deserialize(
        bytes,
        size,
        &manifest,
        &manifest.segments[0],
        &restored
    ));
    ASSERT_TRUE(
        restored.block_shift == EVOKE_DEFAULT_POSTING_BLOCK_SHIFT
    );
    ASSERT_TRUE(restored.block_count == restored.run_count);
    ASSERT_TRUE(restored.runs[2].block_offset == 2);
    ASSERT_TRUE(restored.runs[2].block_count == 1);
    ASSERT_TRUE(restored.blocks[2].kind ==
        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL);
    ASSERT_STATUS_OK(evoke_segment_index_metadata_build(
        &contract,
        &manifest,
        manifest.doc_frequencies,
        &restored,
        1,
        &metadata_index
    ));
    ASSERT_TRUE(metadata_index.data == NULL);
    ASSERT_TRUE(metadata_index.indices == NULL);
    ASSERT_TRUE(metadata_index.indptr == NULL);
    ASSERT_TRUE(metadata_index.term_frequencies == NULL);
    ASSERT_TRUE(metadata_index.num_docs == index.num_docs);
    ASSERT_TRUE(metadata_index.vocab_size == index.vocab_size);
    ASSERT_TRUE(memcmp(
        metadata_index.doc_lengths,
        index.doc_lengths,
        index.num_docs * sizeof(*index.doc_lengths)
    ) == 0);
    ASSERT_TRUE(memcmp(
        metadata_index.doc_frequencies,
        index.doc_frequencies,
        index.vocab_size * sizeof(*index.doc_frequencies)
    ) == 0);
    evoke_segment_payload_as_view(&restored, &payload_view);
    ASSERT_STATUS_OK(evoke_term_directory_build_from_payloads(
        &manifest,
        &payload_view,
        1,
        &directory
    ));
    ASSERT_STATUS_OK(evoke_segment_read_view_build(
        &metadata_index,
        &manifest,
        &directory,
        &payload_view,
        1,
        &read_view
    ));
    ASSERT_TRUE(read_view.terms[2].extents[0].blocks != NULL);
    ASSERT_TRUE(read_view.terms[2].extents[0].block_count == 1);
    ASSERT_TRUE(
        read_view.terms[2].extents[0].block_shift ==
            EVOKE_DEFAULT_POSTING_BLOCK_SHIFT
    );

    memset(&stats, 0, sizeof(stats));
    stats.document_count = manifest.visible_document_count;
    stats.total_document_length = manifest.total_document_length;
    stats.doc_frequencies = manifest.doc_frequencies;
    stats.vocab_size = manifest.vocab_size;
    ASSERT_STATUS_OK(evoke_scores_from_ids_exact_stats(
        &index,
        query,
        3,
        NULL,
        &expected_scores
    ));
    ASSERT_STATUS_OK(evoke_scores_from_ids_mixed(
        &metadata_index,
        &stats,
        read_view.terms,
        read_view.vocab_size,
        query,
        3,
        NULL,
        &segmented_scores
    ));
    assert_float_array(
        segmented_scores,
        expected_scores,
        index.num_docs
    );

    versions[2].document_length++;
    ASSERT_TRUE(evoke_segment_payload_build_lexical(
        &index,
        2,
        0,
        versions,
        4,
        &payload
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(payload.segment_id == 1);
    restored.version_count--;
    ASSERT_TRUE(evoke_segment_index_metadata_build(
        &contract,
        &manifest,
        manifest.doc_frequencies,
        &restored,
        1,
        &metadata_index
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(metadata_index.num_docs == index.num_docs);

    free(bytes);
    free(expected_scores);
    free(segmented_scores);
    evoke_segment_read_view_free(&read_view);
    evoke_term_directory_free(&directory);
    evoke_segment_query_contract_free(&contract);
    evoke_segment_payload_free(&restored);
    evoke_segment_payload_free(&payload);
    evoke_segment_manifest_free(&manifest);
    evoke_index_free(&metadata_index);
    evoke_index_free(&index);
}

static void
test_segment_payload_merge_and_directory_replacement(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_document_version_record versions[4] = {0};
    evoke_segment_manifest old_manifest;
    evoke_segment_manifest next_manifest;
    evoke_segment_payload payloads[2];
    evoke_segment_payload merged;
    evoke_segment_payload_view payload_views[2];
    evoke_segment_payload_view merged_view;
    evoke_term_directory old_directory;
    evoke_term_directory next_directory;
    evoke_index first_index;
    evoke_index second_index;
    evoke_index full_index;
    evoke_segment_term_run *resized_runs;
    evoke_document_version_record *resized_versions;
    evoke_posting_value *resized_values;
    uint32_t *resized_indices;
    uint64_t first_length = 0;
    uint64_t second_length = 0;

    evoke_segment_manifest_init(&old_manifest);
    evoke_segment_manifest_init(&next_manifest);
    evoke_segment_payload_init(&payloads[0]);
    evoke_segment_payload_init(&payloads[1]);
    evoke_segment_payload_init(&merged);
    evoke_term_directory_init(&old_directory);
    evoke_term_directory_init(&next_directory);
    evoke_index_init(&first_index);
    evoke_index_init(&second_index);
    evoke_index_init(&full_index);

    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        2,
        &params,
        false,
        &first_index
    ));
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs + 2,
        2,
        &params,
        false,
        &second_index
    ));
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        4,
        &params,
        false,
        &full_index
    ));
    for (uint32_t document_id = 0; document_id < 4; document_id++)
    {
        evoke_index *source_index =
            document_id < 2 ? &first_index : &second_index;
        uint32_t local_document_id = document_id % 2;

        versions[document_id].document_slot = document_id;
        versions[document_id].born_sequence = document_id + 1;
        versions[document_id].heap_block = 40 + document_id;
        versions[document_id].heap_offset = 1;
        versions[document_id].document_length =
            source_index->doc_lengths[local_document_id];
        versions[document_id].flags =
            EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
        if (document_id == 0 || document_id == 2)
        {
            versions[document_id].flags |=
                EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE;
        }
        if (document_id < 2)
        {
            first_length += versions[document_id].document_length;
        }
        else
        {
            second_length += versions[document_id].document_length;
        }
    }
    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical(
        &first_index,
        1,
        0,
        versions,
        2,
        &payloads[0]
    ));
    ASSERT_STATUS_OK(evoke_segment_payload_build_lexical(
        &second_index,
        2,
        2,
        versions + 2,
        2,
        &payloads[1]
    ));
    for (uint32_t payload_index = 0; payload_index < 2; payload_index++)
    {
        evoke_segment_payload *payload = &payloads[payload_index];
        uint64_t old_posting_count = payload->posting_count;
        uint32_t old_run_count = payload->run_count;

        resized_runs = realloc(
            payload->runs,
            (size_t) (old_run_count + 2) * sizeof(*resized_runs)
        );
        resized_indices = realloc(
            payload->indices,
            (size_t) (old_posting_count + 2) *
                sizeof(*resized_indices)
        );
        resized_values = realloc(
            payload->values,
            (size_t) (old_posting_count + 2) *
                sizeof(*resized_values)
        );
        ASSERT_TRUE(resized_runs != NULL);
        ASSERT_TRUE(resized_indices != NULL);
        ASSERT_TRUE(resized_values != NULL);
        payload->runs = resized_runs;
        payload->indices = resized_indices;
        payload->values = resized_values;
        memset(
            &payload->runs[old_run_count],
            0,
            2 * sizeof(*payload->runs)
        );
        payload->runs[old_run_count].term_id = 3;
        payload->runs[old_run_count].kind =
            EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
        payload->runs[old_run_count].posting_offset =
            old_posting_count;
        payload->runs[old_run_count].posting_count = 1;
        payload->runs[old_run_count + 1].term_id = 3;
        payload->runs[old_run_count + 1].kind =
            EVOKE_POSTING_EXTENT_LEXICAL_IMPACT;
        payload->runs[old_run_count + 1].posting_offset =
            old_posting_count + 1;
        payload->runs[old_run_count + 1].posting_count = 1;
        payload->indices[old_posting_count] = 0;
        payload->indices[old_posting_count + 1] = 0;
        memset(
            &payload->values[old_posting_count],
            0,
            2 * sizeof(*payload->values)
        );
        payload->values[old_posting_count].impact =
            payload_index == 0 ? 0.25f : 0.5f;
        payload->values[old_posting_count + 1].impact =
            payload_index == 0 ? 0.75f : 1.0f;
        payload->run_count += 2;
        payload->posting_count += 2;
        payload->vocab_size = full_index.vocab_size;
    }
    resized_versions = realloc(
        payloads[1].versions,
        3 * sizeof(*resized_versions)
    );
    ASSERT_TRUE(resized_versions != NULL);
    payloads[1].versions = resized_versions;
    memset(&payloads[1].versions[2], 0, sizeof(*payloads[1].versions));
    payloads[1].versions[2].document_slot = 4;
    payloads[1].versions[2].born_sequence = 4;
    payloads[1].versions[2].flags =
        EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE |
        EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
    payloads[1].version_count = 3;
    payloads[1].local_document_count = 3;
    payloads[1].retirements = calloc(
        1,
        sizeof(*payloads[1].retirements)
    );
    ASSERT_TRUE(payloads[1].retirements != NULL);
    payloads[1].retirements[0].document_slot = 0;
    payloads[1].retirements[0].retirement_sequence = 3;
    payloads[1].retirements[0].document_length =
        versions[0].document_length;
    payloads[1].retirements[0].flags =
        EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID;
    payloads[1].retirement_count = 1;

    old_manifest.flags = EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    old_manifest.manifest_id = 10;
    old_manifest.max_sequence = 4;
    old_manifest.statistics_epoch = 1;
    old_manifest.visible_document_count = 4;
    old_manifest.document_slot_count = 5;
    old_manifest.total_document_length = first_length + second_length;
    old_manifest.vocab_size = full_index.vocab_size;
    initialize_test_object_ref(
        &old_manifest.query_contract,
        EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT,
        10,
        1,
        old_manifest.manifest_id,
        100
    );
    initialize_test_object_ref(
        &old_manifest.term_directory,
        EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
        11,
        1,
        old_manifest.manifest_id,
        101
    );
    initialize_test_document_directory(&old_manifest, 1000, 102);
    old_manifest.segment_count = 2;
    old_manifest.segments = calloc(
        old_manifest.segment_count,
        sizeof(*old_manifest.segments)
    );
    old_manifest.doc_frequencies = calloc(
        old_manifest.vocab_size,
        sizeof(*old_manifest.doc_frequencies)
    );
    ASSERT_TRUE(old_manifest.segments != NULL);
    ASSERT_TRUE(old_manifest.doc_frequencies != NULL);
    memcpy(
        old_manifest.doc_frequencies,
        full_index.doc_frequencies,
        old_manifest.vocab_size *
            sizeof(*old_manifest.doc_frequencies)
    );
    old_manifest.segments[0].segment_id = 1;
    old_manifest.segments[0].min_sequence = 1;
    old_manifest.segments[0].max_sequence = 2;
    old_manifest.segments[0].posting_count =
        payloads[0].posting_count;
    old_manifest.segments[0].document_count = 2;
    old_manifest.segments[0].total_document_length = first_length;
    old_manifest.segments[0].first_document_slot = 0;
    old_manifest.segments[0].document_slot_count = 2;
    old_manifest.segments[0].payload_checksum = 200;
    old_manifest.segments[0].start_block = 1;
    old_manifest.segments[0].block_count = 1;
    old_manifest.segments[0].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL |
        EVOKE_SEGMENT_FLAG_SEMANTIC;
    old_manifest.segments[0].payload_bytes = 1000;
    old_manifest.segments[0].payload_owner_manifest_id = 10;
    old_manifest.segments[1].segment_id = 2;
    old_manifest.segments[1].min_sequence = 3;
    old_manifest.segments[1].max_sequence = 4;
    old_manifest.segments[1].posting_count =
        payloads[1].posting_count;
    old_manifest.segments[1].retirement_count = 1;
    old_manifest.segments[1].document_count = 3;
    old_manifest.segments[1].total_document_length = second_length;
    old_manifest.segments[1].first_document_slot = 2;
    old_manifest.segments[1].document_slot_count = 3;
    old_manifest.segments[1].payload_checksum = 201;
    old_manifest.segments[1].start_block = 2;
    old_manifest.segments[1].block_count = 1;
    old_manifest.segments[1].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL |
        EVOKE_SEGMENT_FLAG_SEMANTIC |
        EVOKE_SEGMENT_FLAG_RETIREMENTS;
    old_manifest.segments[1].payload_bytes = 1000;
    old_manifest.segments[1].payload_owner_manifest_id = 10;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&old_manifest));
    evoke_segment_payload_as_view(&payloads[0], &payload_views[0]);
    evoke_segment_payload_as_view(&payloads[1], &payload_views[1]);
    ASSERT_STATUS_OK(evoke_term_directory_build_from_payloads(
        &old_manifest,
        payload_views,
        2,
        &old_directory
    ));

    ASSERT_STATUS_OK(evoke_segment_payload_merge(
        &old_manifest,
        0,
        payloads,
        2,
        11,
        &merged
    ));
    ASSERT_TRUE(merged.run_count == full_index.vocab_size + 2);
    ASSERT_TRUE(merged.posting_count == full_index.data_len + 4);
    ASSERT_TRUE(merged.version_count == 5);
    ASSERT_TRUE(merged.local_document_count == 5);
    ASSERT_TRUE(merged.retirement_count == 1);
    ASSERT_TRUE(memcmp(
        merged.indices,
        full_index.indices,
        (size_t) full_index.data_len * sizeof(*merged.indices)
    ) == 0);
    for (uint64_t posting_index = 0;
         posting_index < full_index.data_len;
         posting_index++)
    {
        ASSERT_TRUE(
            merged.values[posting_index].term_frequency ==
            full_index.term_frequencies[posting_index]
        );
    }
    ASSERT_TRUE(
        merged.versions[4].flags ==
        (EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE |
         EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID)
    );
    ASSERT_TRUE(merged.retirements[0].document_slot == 0);
    ASSERT_TRUE(
        merged.retirements[0].document_length ==
        versions[0].document_length
    );
    ASSERT_TRUE(
        merged.runs[full_index.vocab_size].kind ==
        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
    );
    ASSERT_TRUE(
        merged.runs[full_index.vocab_size].posting_offset ==
        full_index.data_len
    );
    ASSERT_TRUE(
        merged.runs[full_index.vocab_size].posting_count == 2
    );
    ASSERT_TRUE(merged.indices[full_index.data_len] == 0);
    ASSERT_TRUE(merged.indices[full_index.data_len + 1] == 2);
    ASSERT_TRUE(merged.values[full_index.data_len].impact == 0.25f);
    ASSERT_TRUE(
        merged.values[full_index.data_len + 1].impact == 0.5f
    );
    ASSERT_TRUE(
        merged.runs[full_index.vocab_size + 1].kind ==
        EVOKE_POSTING_EXTENT_LEXICAL_IMPACT
    );
    ASSERT_TRUE(
        merged.runs[full_index.vocab_size + 1].posting_offset ==
        full_index.data_len + 2
    );
    ASSERT_TRUE(merged.indices[full_index.data_len + 2] == 0);
    ASSERT_TRUE(merged.indices[full_index.data_len + 3] == 2);
    ASSERT_TRUE(
        merged.values[full_index.data_len + 2].impact == 0.75f
    );
    ASSERT_TRUE(
        merged.values[full_index.data_len + 3].impact == 1.0f
    );

    next_manifest.flags = EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    next_manifest.manifest_id = 11;
    next_manifest.parent_manifest_id = 10;
    next_manifest.max_sequence = old_manifest.max_sequence;
    next_manifest.statistics_epoch = old_manifest.statistics_epoch;
    next_manifest.visible_document_count =
        old_manifest.visible_document_count;
    next_manifest.document_slot_count =
        old_manifest.document_slot_count;
    next_manifest.total_document_length =
        old_manifest.total_document_length;
    next_manifest.vocab_size = old_manifest.vocab_size;
    next_manifest.query_contract = old_manifest.query_contract;
    next_manifest.document_directory =
        old_manifest.document_directory;
    next_manifest.flags |=
        EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY;
    initialize_test_object_ref(
        &next_manifest.term_directory,
        EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
        12,
        1,
        next_manifest.manifest_id,
        102
    );
    next_manifest.segment_count = 1;
    next_manifest.segments = calloc(
        1,
        sizeof(*next_manifest.segments)
    );
    next_manifest.doc_frequencies = calloc(
        next_manifest.vocab_size,
        sizeof(*next_manifest.doc_frequencies)
    );
    ASSERT_TRUE(next_manifest.segments != NULL);
    ASSERT_TRUE(next_manifest.doc_frequencies != NULL);
    memcpy(
        next_manifest.doc_frequencies,
        old_manifest.doc_frequencies,
        next_manifest.vocab_size *
            sizeof(*next_manifest.doc_frequencies)
    );
    memcpy(
        next_manifest.contract_hash,
        old_manifest.contract_hash,
        sizeof(next_manifest.contract_hash)
    );
    next_manifest.segments[0].segment_id = 11;
    next_manifest.segments[0].min_sequence = 1;
    next_manifest.segments[0].max_sequence = 4;
    next_manifest.segments[0].posting_count = merged.posting_count;
    next_manifest.segments[0].retirement_count =
        merged.retirement_count;
    next_manifest.segments[0].document_count = merged.version_count;
    next_manifest.segments[0].total_document_length =
        old_manifest.total_document_length;
    next_manifest.segments[0].first_document_slot = 0;
    next_manifest.segments[0].document_slot_count =
        merged.local_document_count;
    next_manifest.segments[0].payload_checksum = 202;
    next_manifest.segments[0].start_block = 3;
    next_manifest.segments[0].block_count = 1;
    next_manifest.segments[0].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL |
        EVOKE_SEGMENT_FLAG_SEMANTIC |
        EVOKE_SEGMENT_FLAG_RETIREMENTS;
    next_manifest.segments[0].payload_bytes = 2000;
    next_manifest.segments[0].payload_owner_manifest_id = 11;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&next_manifest));
    ASSERT_STATUS_OK(evoke_segment_payload_validate(
        &merged,
        &next_manifest,
        &next_manifest.segments[0]
    ));
    evoke_segment_payload_as_view(&merged, &merged_view);
    ASSERT_STATUS_OK(evoke_term_directory_replace_payloads(
        &old_directory,
        &old_manifest,
        &next_manifest,
        0,
        2,
        &merged_view,
        &next_directory
    ));
    ASSERT_TRUE(
        next_directory.extent_count == full_index.vocab_size + 2
    );
    for (uint32_t term_id = 0;
         term_id < next_directory.vocab_size;
         term_id++)
    {
        uint64_t expected_extent_count = term_id == 3 ? 3 : 1;

        ASSERT_TRUE(
            next_directory.term_offsets[term_id + 1] -
            next_directory.term_offsets[term_id] ==
            expected_extent_count
        );
    }
    ASSERT_TRUE(evoke_segment_size_class(0) == 0);
    ASSERT_TRUE(evoke_segment_size_class(65536) == 0);
    ASSERT_TRUE(evoke_segment_size_class(65537) == 1);
    ASSERT_TRUE(evoke_segment_size_class(131072) == 1);
    ASSERT_TRUE(evoke_segment_size_class(131073) == 2);
    ASSERT_TRUE(evoke_segment_size_class(UINT64_MAX) == 48);

    evoke_index_free(&full_index);
    evoke_index_free(&second_index);
    evoke_index_free(&first_index);
    evoke_term_directory_free(&next_directory);
    evoke_term_directory_free(&old_directory);
    evoke_segment_payload_free(&merged);
    evoke_segment_payload_free(&payloads[1]);
    evoke_segment_payload_free(&payloads[0]);
    evoke_segment_manifest_free(&next_manifest);
    evoke_segment_manifest_free(&old_manifest);
}

static void
test_segment_read_view_matches_expanded_index(void)
{
    uint32_t base_doc0[] = {0, 0, 1};
    uint32_t base_doc1[] = {1, 2};
    uint32_t base_doc2[] = {0, 2, 2};
    uint32_t base_doc3[] = {3};
    uint32_t added_doc[] = {0, 3, 3};
    uint32_t query[] = {0, 2, 3};
    evoke_doc_ids base_docs[] = {
        make_doc(base_doc0, 3),
        make_doc(base_doc1, 2),
        make_doc(base_doc2, 3),
        make_doc(base_doc3, 1)
    };
    evoke_doc_ids delta_docs[] = {
        make_doc(added_doc, 3)
    };
    evoke_doc_ids expanded_docs[] = {
        make_doc(base_doc0, 3),
        make_doc(base_doc1, 2),
        make_doc(base_doc2, 3),
        make_doc(base_doc3, 1),
        make_doc(added_doc, 3)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_segment_term_run base_runs[] = {
        {
            .term_id = 0,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 2
        },
        {
            .term_id = 1,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 2,
            .posting_count = 2
        },
        {
            .term_id = 2,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 4,
            .posting_count = 2
        },
        {
            .term_id = 3,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 6,
            .posting_count = 1
        }
    };
    evoke_segment_term_run delta_runs[] = {
        {
            .term_id = 0,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 1
        },
        {
            .term_id = 3,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 1,
            .posting_count = 1
        }
    };
    evoke_segment_manifest manifest;
    evoke_segment_manifest compacted_manifest;
    evoke_segment_manifest old_manifest;
    evoke_segment_descriptor compacted_segment;
    evoke_term_directory directory;
    evoke_term_directory incremental_directory;
    evoke_term_directory old_directory;
    evoke_term_directory folded_tail_directory;
    evoke_term_directory dual_fold_tail_directory;
    evoke_segment_payload_view payloads[2];
    evoke_segment_read_view view;
    evoke_segment_read_view folded_view;
    evoke_term_fold_bundle fold_bundle;
    evoke_term_fold_bundle minor_fold_bundle;
    evoke_term_fold_bundle promoted_fold_bundle;
    evoke_term_fold_bundle advanced_fold_bundle;
    evoke_term_fold_bundle full_fold_bundle;
    evoke_term_fold_bundle impact_fold_bundle;
    evoke_term_fold_bundle compacted_minor_fold_bundle;
    evoke_term_fold_bundle selected_fold_bundles[3];
    evoke_term_fold_read_plan *fold_plans = NULL;
    evoke_index base;
    evoke_index delta;
    evoke_index expanded;
    evoke_corpus_stats stats;
    evoke_posting_value *base_values = NULL;
    evoke_posting_value *delta_values = NULL;
    float *expected_scores = NULL;
    float *segmented_scores = NULL;
    float *folded_scores = NULL;
    evoke_term_extent_list *original_terms;
    uint32_t tail_payload_segment_indices[] = {1};
    uint32_t doc_id;

    evoke_index_init(&base);
    evoke_index_init(&delta);
    evoke_index_init(&expanded);
    evoke_term_directory_init(&directory);
    evoke_term_directory_init(&incremental_directory);
    evoke_term_directory_init(&old_directory);
    evoke_term_directory_init(&folded_tail_directory);
    evoke_term_directory_init(&dual_fold_tail_directory);
    evoke_segment_read_view_init(&view);
    evoke_segment_read_view_init(&folded_view);
    evoke_term_fold_bundle_init(&fold_bundle);
    evoke_term_fold_bundle_init(&minor_fold_bundle);
    evoke_term_fold_bundle_init(&promoted_fold_bundle);
    evoke_term_fold_bundle_init(&advanced_fold_bundle);
    evoke_term_fold_bundle_init(&full_fold_bundle);
    evoke_term_fold_bundle_init(&impact_fold_bundle);
    evoke_term_fold_bundle_init(&compacted_minor_fold_bundle);
    memset(payloads, 0, sizeof(payloads));

    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        base_docs,
        4,
        &params,
        false,
        &base
    ));
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        delta_docs,
        1,
        &params,
        false,
        &delta
    ));
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        expanded_docs,
        5,
        &params,
        false,
        &expanded
    ));
    initialize_test_segment_manifest(&manifest);
    manifest.visible_document_count = 4;

    base_values = calloc(base.data_len, sizeof(*base_values));
    delta_values = calloc(delta.data_len, sizeof(*delta_values));
    ASSERT_TRUE(base_values != NULL);
    ASSERT_TRUE(delta_values != NULL);
    for (doc_id = 0; doc_id < base.data_len; doc_id++)
    {
        base_values[doc_id].term_frequency =
            base.term_frequencies[doc_id];
    }
    for (doc_id = 0; doc_id < delta.data_len; doc_id++)
    {
        delta_values[doc_id].term_frequency =
            delta.term_frequencies[doc_id];
    }

    payloads[0].segment_id = 1;
    payloads[0].posting_count = base.data_len;
    payloads[0].runs = base_runs;
    payloads[0].run_count = 4;
    payloads[0].values = base_values;
    payloads[0].indices = base.indices;
    payloads[0].local_document_count = base.num_docs;
    payloads[1].segment_id = 2;
    payloads[1].posting_count = delta.data_len;
    payloads[1].runs = delta_runs;
    payloads[1].run_count = 2;
    payloads[1].values = delta_values;
    payloads[1].indices = delta.indices;
    payloads[1].document_id_base = base.num_docs;
    payloads[1].local_document_count = delta.num_docs;

    old_manifest = manifest;
    old_manifest.segment_count = 1;
    old_manifest.max_sequence = manifest.segments[0].max_sequence;
    old_manifest.visible_document_count = base.num_docs;
    old_manifest.document_slot_count = base.num_docs;
    old_manifest.total_document_length = 9;
    ASSERT_STATUS_OK(evoke_term_directory_build_from_payloads(
        &old_manifest,
        payloads,
        1,
        &old_directory
    ));
    manifest.parent_manifest_id = old_manifest.manifest_id;
    manifest.manifest_id++;
    manifest.query_contract.object_id = manifest.manifest_id;
    manifest.query_contract.owner_manifest_id = manifest.manifest_id;
    manifest.term_directory.object_id = manifest.manifest_id;
    manifest.term_directory.owner_manifest_id = manifest.manifest_id;
    manifest.neutral_fold.object_id = manifest.manifest_id;
    manifest.neutral_fold.owner_manifest_id = manifest.manifest_id;
    manifest.impact_fold.object_id = manifest.manifest_id;
    manifest.impact_fold.owner_manifest_id = manifest.manifest_id;
    manifest.segments[1].payload_owner_manifest_id =
        manifest.manifest_id;
    ASSERT_STATUS_OK(evoke_term_directory_append_payload(
        &old_directory,
        &old_manifest,
        &manifest,
        &payloads[1],
        &incremental_directory
    ));
    ASSERT_STATUS_OK(evoke_term_directory_build_from_payloads(
        &manifest,
        payloads,
        2,
        &directory
    ));
    ASSERT_TRUE(
        incremental_directory.vocab_size == directory.vocab_size
    );
    ASSERT_TRUE(
        incremental_directory.extent_count == directory.extent_count
    );
    assert_uint64_array(
        incremental_directory.term_offsets,
        directory.term_offsets,
        (size_t) directory.vocab_size + 1
    );
    ASSERT_TRUE(memcmp(
        incremental_directory.extents,
        directory.extents,
        sizeof(*directory.extents) * directory.extent_count
    ) == 0);
    ASSERT_TRUE(directory.extent_count == 6);
    ASSERT_TRUE(directory.term_offsets[1] == 2);
    ASSERT_TRUE(directory.extents[1].segment_index == 1);
    ASSERT_STATUS_OK(evoke_segment_read_view_build(
        &expanded,
        &manifest,
        &directory,
        payloads,
        2,
        &view
    ));
    ASSERT_TRUE(view.vocab_size == expanded.vocab_size);
    ASSERT_TRUE(view.extent_count == 6);
    ASSERT_TRUE(view.terms[0].len == 2);
    ASSERT_TRUE(view.terms[3].len == 2);
    ASSERT_TRUE(view.terms[0].extents[1].document_id_base == 4);

    memset(&stats, 0, sizeof(stats));
    stats.document_count = expanded.num_docs;
    stats.doc_frequencies = expanded.doc_frequencies;
    stats.vocab_size = expanded.vocab_size;
    for (doc_id = 0; doc_id < expanded.num_docs; doc_id++)
    {
        stats.total_document_length += expanded.doc_lengths[doc_id];
    }
    ASSERT_STATUS_OK(evoke_scores_from_ids_exact_stats(
        &expanded,
        query,
        3,
        NULL,
        &expected_scores
    ));
    ASSERT_STATUS_OK(evoke_scores_from_ids_mixed(
        &expanded,
        &stats,
        view.terms,
        view.vocab_size,
        query,
        3,
        NULL,
        &segmented_scores
    ));
    assert_float_array(
        segmented_scores,
        expected_scores,
        expanded.num_docs
    );

    folded_tail_directory.vocab_size = directory.vocab_size;
    folded_tail_directory.extent_count = directory.extent_count - 1;
    folded_tail_directory.term_offsets = calloc(
        (size_t) folded_tail_directory.vocab_size + 1,
        sizeof(*folded_tail_directory.term_offsets)
    );
    folded_tail_directory.extents = calloc(
        folded_tail_directory.extent_count,
        sizeof(*folded_tail_directory.extents)
    );
    fold_plans = calloc(
        expanded.vocab_size,
        sizeof(*fold_plans)
    );
    ASSERT_TRUE(folded_tail_directory.term_offsets != NULL);
    ASSERT_TRUE(folded_tail_directory.extents != NULL);
    ASSERT_TRUE(fold_plans != NULL);
    for (uint32_t term_id = 0;
         term_id <= folded_tail_directory.vocab_size;
         term_id++)
    {
        folded_tail_directory.term_offsets[term_id] =
            directory.term_offsets[term_id] -
            (directory.term_offsets[term_id] > 0 ? 1 : 0);
    }
    memcpy(
        folded_tail_directory.extents,
        &directory.extents[1],
        (size_t) folded_tail_directory.extent_count *
            sizeof(*folded_tail_directory.extents)
    );
    ASSERT_STATUS_OK(evoke_term_directory_validate(
        &folded_tail_directory,
        &manifest
    ));

    ASSERT_TRUE(evoke_term_fold_bundle_build_neutral_prefix(
        &manifest,
        &directory,
        payloads,
        2,
        manifest.manifest_id,
        0,
        manifest.segments[0].max_sequence - 1,
        &fold_bundle
    ) == EVOKE_ERR_FORMAT);
    ASSERT_STATUS_OK(evoke_term_fold_bundle_build_neutral_prefix(
        &manifest,
        &directory,
        payloads,
        2,
        manifest.manifest_id,
        0,
        manifest.segments[0].max_sequence,
        &fold_bundle
    ));
    ASSERT_TRUE(fold_bundle.run_count == 1);
    ASSERT_TRUE(
        fold_bundle.posting_count == base_runs[0].posting_count
    );
    ASSERT_TRUE(fold_bundle.runs[0].term_id == 0);
    ASSERT_TRUE(
        fold_bundle.runs[0].kind ==
            EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL
    );
    ASSERT_TRUE(memcmp(
        fold_bundle.document_slots,
        &base.indices[base_runs[0].posting_offset],
        (size_t) fold_bundle.posting_count *
            sizeof(*fold_bundle.document_slots)
    ) == 0);
    ASSERT_TRUE(memcmp(
        fold_bundle.values,
        &base_values[base_runs[0].posting_offset],
        (size_t) fold_bundle.posting_count *
            sizeof(*fold_bundle.values)
    ) == 0);
    ASSERT_STATUS_OK(evoke_term_fold_bundle_advance_neutral(
        &manifest,
        0,
        &directory.extents[1],
        1,
        &payloads[1],
        tail_payload_segment_indices,
        1,
        &fold_bundle,
        manifest.segments[0].max_sequence,
        manifest.manifest_id,
        manifest.segments[1].max_sequence,
        &advanced_fold_bundle
    ));
    ASSERT_STATUS_OK(evoke_term_fold_bundle_build_neutral_prefix(
        &manifest,
        &directory,
        payloads,
        2,
        manifest.manifest_id,
        0,
        manifest.segments[1].max_sequence,
        &full_fold_bundle
    ));
    ASSERT_TRUE(
        advanced_fold_bundle.run_count ==
        full_fold_bundle.run_count
    );
    ASSERT_TRUE(
        advanced_fold_bundle.posting_count ==
        full_fold_bundle.posting_count
    );
    ASSERT_TRUE(memcmp(
        advanced_fold_bundle.runs,
        full_fold_bundle.runs,
        (size_t) full_fold_bundle.run_count *
            sizeof(*full_fold_bundle.runs)
    ) == 0);
    ASSERT_TRUE(memcmp(
        advanced_fold_bundle.document_slots,
        full_fold_bundle.document_slots,
        (size_t) full_fold_bundle.posting_count *
            sizeof(*full_fold_bundle.document_slots)
    ) == 0);
    ASSERT_TRUE(memcmp(
        advanced_fold_bundle.values,
        full_fold_bundle.values,
        (size_t) full_fold_bundle.posting_count *
            sizeof(*full_fold_bundle.values)
    ) == 0);
    fold_plans[0].neutral_coverage =
        fold_bundle.runs[0].coverage_sequence;
    fold_plans[0].neutral_run_count = 1;
    ASSERT_STATUS_OK(evoke_segment_read_view_build_folded(
        &expanded,
        &manifest,
        &folded_tail_directory,
        payloads,
        2,
        &fold_bundle,
        1,
        fold_plans,
        expanded.vocab_size,
        &folded_view
    ));
    ASSERT_TRUE(folded_view.terms[0].len == 2);
    ASSERT_TRUE(
        folded_view.terms[0].extents[0].document_id_base == 0
    );
    ASSERT_TRUE(
        folded_view.terms[0].extents[1].document_id_base ==
            base.num_docs
    );
    ASSERT_STATUS_OK(evoke_scores_from_ids_mixed(
        &expanded,
        &stats,
        folded_view.terms,
        folded_view.vocab_size,
        query,
        3,
        NULL,
        &folded_scores
    ));
    assert_float_array(
        folded_scores,
        expected_scores,
        expanded.num_docs
    );
    free(folded_scores);
    folded_scores = NULL;

    dual_fold_tail_directory.vocab_size = directory.vocab_size;
    dual_fold_tail_directory.extent_count =
        directory.extent_count - 2;
    dual_fold_tail_directory.term_offsets = calloc(
        (size_t) dual_fold_tail_directory.vocab_size + 1,
        sizeof(*dual_fold_tail_directory.term_offsets)
    );
    dual_fold_tail_directory.extents = calloc(
        dual_fold_tail_directory.extent_count,
        sizeof(*dual_fold_tail_directory.extents)
    );
    ASSERT_TRUE(dual_fold_tail_directory.term_offsets != NULL);
    ASSERT_TRUE(dual_fold_tail_directory.extents != NULL);
    for (uint32_t term_id = 0;
         term_id <= dual_fold_tail_directory.vocab_size;
         term_id++)
    {
        uint64_t removed = directory.term_offsets[term_id] > 2
            ? 2
            : directory.term_offsets[term_id];

        dual_fold_tail_directory.term_offsets[term_id] =
            directory.term_offsets[term_id] - removed;
    }
    memcpy(
        dual_fold_tail_directory.extents,
        &directory.extents[2],
        (size_t) dual_fold_tail_directory.extent_count *
            sizeof(*dual_fold_tail_directory.extents)
    );
    ASSERT_STATUS_OK(evoke_term_directory_validate(
        &dual_fold_tail_directory,
        &manifest
    ));
    ASSERT_STATUS_OK(evoke_term_fold_bundle_advance_neutral(
        &manifest,
        0,
        &directory.extents[1],
        1,
        &payloads[1],
        tail_payload_segment_indices,
        1,
        NULL,
        manifest.segments[0].max_sequence,
        manifest.manifest_id,
        manifest.segments[1].max_sequence,
        &minor_fold_bundle
    ));
    compacted_manifest = manifest;
    compacted_segment = manifest.segments[0];
    compacted_segment.max_sequence = manifest.max_sequence;
    compacted_manifest.segment_count = 1;
    compacted_manifest.segments = &compacted_segment;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&compacted_manifest));
    {
        evoke_posting_extent compacted_tail_extent = {
            .indices = payloads[1].indices,
            .values = payloads[1].values,
            .document_id_map = payloads[1].document_id_map,
            .len = directory.extents[1].posting_count,
            .document_id_base = payloads[1].document_id_base,
            .local_document_count = payloads[1].local_document_count,
            .kind = directory.extents[1].kind
        };

        ASSERT_TRUE(evoke_term_fold_bundle_advance_neutral_extents(
            &compacted_manifest,
            0,
            &compacted_tail_extent,
            1,
            NULL,
            manifest.segments[0].max_sequence,
            false,
            manifest.manifest_id,
            manifest.max_sequence,
            &compacted_minor_fold_bundle
        ) == EVOKE_ERR_FORMAT);
        ASSERT_STATUS_OK(evoke_term_fold_bundle_advance_neutral_extents(
            &compacted_manifest,
            0,
            &compacted_tail_extent,
            1,
            NULL,
            manifest.segments[0].max_sequence,
            true,
            manifest.manifest_id,
            manifest.max_sequence,
            &compacted_minor_fold_bundle
        ));
    }
    ASSERT_TRUE(
        compacted_minor_fold_bundle.run_count ==
        minor_fold_bundle.run_count
    );
    ASSERT_TRUE(
        compacted_minor_fold_bundle.posting_count ==
        minor_fold_bundle.posting_count
    );
    ASSERT_TRUE(memcmp(
        compacted_minor_fold_bundle.runs,
        minor_fold_bundle.runs,
        (size_t) minor_fold_bundle.run_count *
            sizeof(*minor_fold_bundle.runs)
    ) == 0);
    ASSERT_TRUE(memcmp(
        compacted_minor_fold_bundle.document_slots,
        minor_fold_bundle.document_slots,
        (size_t) minor_fold_bundle.posting_count *
            sizeof(*minor_fold_bundle.document_slots)
    ) == 0);
    ASSERT_TRUE(memcmp(
        compacted_minor_fold_bundle.values,
        minor_fold_bundle.values,
        (size_t) minor_fold_bundle.posting_count *
            sizeof(*minor_fold_bundle.values)
    ) == 0);
    ASSERT_STATUS_OK(evoke_term_fold_bundle_merge_neutral(
        &manifest,
        0,
        &fold_bundle,
        &minor_fold_bundle,
        manifest.manifest_id,
        &promoted_fold_bundle
    ));
    ASSERT_TRUE(
        promoted_fold_bundle.run_count ==
        full_fold_bundle.run_count
    );
    ASSERT_TRUE(
        promoted_fold_bundle.posting_count ==
        full_fold_bundle.posting_count
    );
    ASSERT_TRUE(memcmp(
        promoted_fold_bundle.runs,
        full_fold_bundle.runs,
        (size_t) full_fold_bundle.run_count *
            sizeof(*full_fold_bundle.runs)
    ) == 0);
    ASSERT_TRUE(memcmp(
        promoted_fold_bundle.document_slots,
        full_fold_bundle.document_slots,
        (size_t) full_fold_bundle.posting_count *
            sizeof(*full_fold_bundle.document_slots)
    ) == 0);
    ASSERT_TRUE(memcmp(
        promoted_fold_bundle.values,
        full_fold_bundle.values,
        (size_t) full_fold_bundle.posting_count *
            sizeof(*full_fold_bundle.values)
    ) == 0);
    ASSERT_STATUS_OK(evoke_term_fold_bundle_build_impact(
        &expanded,
        &stats,
        0,
        expanded.doc_frequencies[0],
        &fold_bundle,
        &minor_fold_bundle,
        manifest.manifest_id + 1,
        manifest.statistics_epoch,
        &impact_fold_bundle
    ));
    ASSERT_TRUE(
        impact_fold_bundle.object_kind ==
            EVOKE_SEGMENT_OBJECT_IMPACT_FOLD
    );
    ASSERT_TRUE(
        impact_fold_bundle.statistics_epoch ==
            manifest.statistics_epoch
    );
    ASSERT_TRUE(impact_fold_bundle.run_count == 1);
    ASSERT_TRUE(
        impact_fold_bundle.runs[0].kind ==
            EVOKE_POSTING_EXTENT_LEXICAL_IMPACT
    );
    ASSERT_TRUE(
        impact_fold_bundle.runs[0].coverage_sequence ==
            minor_fold_bundle.runs[0].coverage_sequence
    );
    ASSERT_TRUE(
        impact_fold_bundle.posting_count ==
            promoted_fold_bundle.runs[0].posting_count
    );
    for (uint64_t posting_index = 0;
         posting_index < impact_fold_bundle.posting_count;
         posting_index++)
    {
        uint32_t document_slot =
            impact_fold_bundle.document_slots[posting_index];
        uint32_t term_frequency =
            promoted_fold_bundle.values[
                posting_index
            ].term_frequency;
        double average_document_length =
            (double) stats.total_document_length /
            (double) stats.document_count;
        double idf = evoke_score_idf(
            expanded.params.idf_method,
            (double) expanded.doc_frequencies[0],
            (double) stats.document_count
        );
        double nonoccurrence = idf * evoke_score_tfc(
            expanded.params.method,
            0.0,
            0.0,
            average_document_length,
            expanded.params.k1,
            expanded.params.b,
            expanded.params.delta
        );
        double expected_impact = idf * evoke_score_tfc(
            expanded.params.method,
            (double) term_frequency,
            (double) expanded.doc_lengths[document_slot],
            average_document_length,
            expanded.params.k1,
            expanded.params.b,
            expanded.params.delta
        ) - nonoccurrence;

        ASSERT_TRUE(
            document_slot ==
                promoted_fold_bundle.document_slots[posting_index]
        );
        assert_float_close(
            impact_fold_bundle.values[posting_index].impact,
            (float) expected_impact
        );
    }
    ASSERT_TRUE(
        evoke_term_fold_bundle_build_impact(
            &expanded,
            &stats,
            0,
            expanded.doc_frequencies[0],
            &fold_bundle,
            &minor_fold_bundle,
            manifest.manifest_id,
            manifest.statistics_epoch,
            &impact_fold_bundle
        ) == EVOKE_ERR_FORMAT
    );
    selected_fold_bundles[0] = fold_bundle;
    selected_fold_bundles[1] = minor_fold_bundle;
    fold_plans[0].neutral_minor_coverage =
        minor_fold_bundle.runs[0].coverage_sequence;
    fold_plans[0].neutral_minor_bundle_index = 1;
    fold_plans[0].neutral_minor_run_count = 1;
    ASSERT_STATUS_OK(evoke_segment_read_view_build_folded(
        &expanded,
        &manifest,
        &dual_fold_tail_directory,
        payloads,
        2,
        selected_fold_bundles,
        2,
        fold_plans,
        expanded.vocab_size,
        &folded_view
    ));
    ASSERT_TRUE(folded_view.terms[0].len == 2);
    ASSERT_STATUS_OK(evoke_scores_from_ids_mixed(
        &expanded,
        &stats,
        folded_view.terms,
        folded_view.vocab_size,
        query,
        3,
        NULL,
        &folded_scores
    ));
    assert_float_array(
        folded_scores,
        expected_scores,
        expanded.num_docs
    );
    selected_fold_bundles[2] = impact_fold_bundle;
    selected_fold_bundles[2].owner_manifest_id =
        manifest.manifest_id;
    fold_plans[0].impact_coverage =
        impact_fold_bundle.runs[0].coverage_sequence;
    fold_plans[0].impact_statistics_epoch =
        impact_fold_bundle.statistics_epoch;
    fold_plans[0].impact_bundle_index = 2;
    fold_plans[0].impact_run_count = 1;
    ASSERT_STATUS_OK(evoke_segment_read_view_build_folded(
        &expanded,
        &manifest,
        &dual_fold_tail_directory,
        payloads,
        2,
        selected_fold_bundles,
        3,
        fold_plans,
        expanded.vocab_size,
        &folded_view
    ));
    ASSERT_TRUE(folded_view.terms[0].len == 1);
    ASSERT_TRUE(
        folded_view.terms[0].extents[0].kind ==
            EVOKE_POSTING_EXTENT_LEXICAL_IMPACT
    );
    free(folded_scores);
    folded_scores = NULL;
    ASSERT_STATUS_OK(evoke_scores_from_ids_mixed(
        &expanded,
        &stats,
        folded_view.terms,
        folded_view.vocab_size,
        query,
        3,
        NULL,
        &folded_scores
    ));
    assert_float_array(
        folded_scores,
        expected_scores,
        expanded.num_docs
    );
    fold_plans[0].impact_statistics_epoch++;
    ASSERT_TRUE(evoke_segment_read_view_build_folded(
        &expanded,
        &manifest,
        &dual_fold_tail_directory,
        payloads,
        2,
        selected_fold_bundles,
        3,
        fold_plans,
        expanded.vocab_size,
        &folded_view
    ) == EVOKE_ERR_FORMAT);
    fold_plans[0].impact_statistics_epoch =
        impact_fold_bundle.statistics_epoch;
    fold_plans[0].impact_coverage--;
    ASSERT_TRUE(evoke_segment_read_view_build_folded(
        &expanded,
        &manifest,
        &dual_fold_tail_directory,
        payloads,
        2,
        selected_fold_bundles,
        3,
        fold_plans,
        expanded.vocab_size,
        &folded_view
    ) == EVOKE_ERR_FORMAT);
    fold_plans[0].impact_coverage = 0;
    fold_plans[0].impact_statistics_epoch = 0;
    fold_plans[0].impact_bundle_index = 0;
    fold_plans[0].impact_run_count = 0;
    fold_plans[0].neutral_minor_coverage = 0;
    fold_plans[0].neutral_minor_bundle_index = 0;
    fold_plans[0].neutral_minor_run_count = 0;

    fold_plans[0].neutral_coverage =
        manifest.segments[0].max_sequence - 1;
    fold_bundle.runs[0].coverage_sequence =
        fold_plans[0].neutral_coverage;
    ASSERT_TRUE(evoke_segment_read_view_build_folded(
        &expanded,
        &manifest,
        &folded_tail_directory,
        payloads,
        2,
        &fold_bundle,
        1,
        fold_plans,
        expanded.vocab_size,
        &folded_view
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(folded_view.terms != NULL);
    fold_plans[0].neutral_coverage =
        manifest.segments[0].max_sequence;
    fold_bundle.runs[0].coverage_sequence =
        fold_plans[0].neutral_coverage;

    original_terms = view.terms;
    directory.extents[0].kind =
        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
    ASSERT_TRUE(evoke_segment_read_view_build(
        &expanded,
        &manifest,
        &directory,
        payloads,
        2,
        &view
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(view.terms == original_terms);
    directory.extents[0].kind =
        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
    directory.extents[2].posting_offset = 3;
    ASSERT_TRUE(evoke_segment_read_view_build(
        &expanded,
        &manifest,
        &directory,
        payloads,
        2,
        &view
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(view.terms == original_terms);
    directory.extents[2].posting_offset = 2;
    payloads[1].segment_id = 3;
    ASSERT_TRUE(evoke_segment_read_view_build(
        &expanded,
        &manifest,
        &directory,
        payloads,
        2,
        &view
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(view.terms == original_terms);

    free(expected_scores);
    free(segmented_scores);
    free(folded_scores);
    free(fold_plans);
    free(base_values);
    free(delta_values);
    evoke_term_fold_bundle_free(&full_fold_bundle);
    evoke_term_fold_bundle_free(&advanced_fold_bundle);
    evoke_term_fold_bundle_free(&impact_fold_bundle);
    evoke_term_fold_bundle_free(&minor_fold_bundle);
    evoke_term_fold_bundle_free(&compacted_minor_fold_bundle);
    evoke_term_fold_bundle_free(&promoted_fold_bundle);
    evoke_term_fold_bundle_free(&fold_bundle);
    evoke_segment_read_view_free(&folded_view);
    evoke_segment_read_view_free(&view);
    evoke_term_directory_free(&folded_tail_directory);
    evoke_term_directory_free(&dual_fold_tail_directory);
    evoke_term_directory_free(&old_directory);
    evoke_term_directory_free(&incremental_directory);
    evoke_term_directory_free(&directory);
    evoke_segment_manifest_free(&manifest);
    evoke_index_free(&base);
    evoke_index_free(&delta);
    evoke_index_free(&expanded);
}

static void
initialize_term_cow_test_manifests(
    evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest
)
{
    evoke_segment_manifest_init(old_manifest);
    evoke_segment_manifest_init(next_manifest);

    old_manifest->flags = EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    old_manifest->manifest_id = 10;
    old_manifest->parent_manifest_id = 9;
    old_manifest->max_sequence = 10;
    old_manifest->statistics_epoch = 1;
    old_manifest->visible_document_count = 2;
    old_manifest->document_slot_count = 2;
    old_manifest->total_document_length = 5;
    old_manifest->vocab_size = 80;
    initialize_test_object_ref(
        &old_manifest->query_contract,
        EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT,
        100,
        1,
        old_manifest->manifest_id,
        501
    );
    initialize_test_object_ref(
        &old_manifest->term_directory,
        EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
        110,
        1,
        old_manifest->manifest_id,
        502
    );
    initialize_test_document_directory(old_manifest, 130, 503);
    old_manifest->segment_count = 1;
    old_manifest->segments = calloc(
        old_manifest->segment_count,
        sizeof(*old_manifest->segments)
    );
    old_manifest->doc_frequencies = calloc(
        old_manifest->vocab_size,
        sizeof(*old_manifest->doc_frequencies)
    );
    ASSERT_TRUE(old_manifest->segments != NULL);
    ASSERT_TRUE(old_manifest->doc_frequencies != NULL);
    memset(
        old_manifest->contract_hash,
        0x51,
        sizeof(old_manifest->contract_hash)
    );
    old_manifest->doc_frequencies[1] = 1;
    old_manifest->doc_frequencies[17] = 1;
    old_manifest->doc_frequencies[33] = 1;
    old_manifest->segments[0].segment_id = 100;
    old_manifest->segments[0].min_sequence = 1;
    old_manifest->segments[0].max_sequence = 10;
    old_manifest->segments[0].posting_count = 3;
    old_manifest->segments[0].document_count = 2;
    old_manifest->segments[0].total_document_length = 5;
    old_manifest->segments[0].first_document_slot = 0;
    old_manifest->segments[0].document_slot_count = 2;
    old_manifest->segments[0].payload_checksum = 601;
    old_manifest->segments[0].start_block = 1;
    old_manifest->segments[0].block_count = 1;
    old_manifest->segments[0].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL;
    old_manifest->segments[0].payload_bytes = 256;
    old_manifest->segments[0].payload_owner_manifest_id =
        old_manifest->manifest_id;

    next_manifest->flags = old_manifest->flags;
    next_manifest->manifest_id = 11;
    next_manifest->parent_manifest_id = old_manifest->manifest_id;
    next_manifest->max_sequence = 11;
    next_manifest->statistics_epoch = 2;
    next_manifest->visible_document_count = 3;
    next_manifest->document_slot_count = 3;
    next_manifest->total_document_length = 7;
    next_manifest->vocab_size = old_manifest->vocab_size;
    next_manifest->query_contract = old_manifest->query_contract;
    initialize_test_object_ref(
        &next_manifest->term_directory,
        EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
        120,
        1,
        next_manifest->manifest_id,
        503
    );
    initialize_test_document_directory(next_manifest, 131, 504);
    next_manifest->segment_count = 2;
    next_manifest->segments = calloc(
        next_manifest->segment_count,
        sizeof(*next_manifest->segments)
    );
    next_manifest->doc_frequencies = calloc(
        next_manifest->vocab_size,
        sizeof(*next_manifest->doc_frequencies)
    );
    ASSERT_TRUE(next_manifest->segments != NULL);
    ASSERT_TRUE(next_manifest->doc_frequencies != NULL);
    memcpy(
        next_manifest->contract_hash,
        old_manifest->contract_hash,
        sizeof(next_manifest->contract_hash)
    );
    memcpy(
        next_manifest->segments,
        old_manifest->segments,
        sizeof(*next_manifest->segments)
    );
    memcpy(
        next_manifest->doc_frequencies,
        old_manifest->doc_frequencies,
        (size_t) next_manifest->vocab_size *
            sizeof(*next_manifest->doc_frequencies)
    );
    next_manifest->doc_frequencies[17]++;
    next_manifest->doc_frequencies[50]++;
    next_manifest->doc_frequencies[70]++;
    next_manifest->segments[1].segment_id = 101;
    next_manifest->segments[1].min_sequence = 11;
    next_manifest->segments[1].max_sequence = 11;
    next_manifest->segments[1].posting_count = 3;
    next_manifest->segments[1].document_count = 1;
    next_manifest->segments[1].total_document_length = 2;
    next_manifest->segments[1].first_document_slot = 2;
    next_manifest->segments[1].document_slot_count = 1;
    next_manifest->segments[1].payload_checksum = 602;
    next_manifest->segments[1].start_block = 2;
    next_manifest->segments[1].block_count = 1;
    next_manifest->segments[1].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL;
    next_manifest->segments[1].payload_bytes = 256;
    next_manifest->segments[1].payload_owner_manifest_id =
        next_manifest->manifest_id;

    ASSERT_STATUS_OK(evoke_segment_manifest_validate(old_manifest));
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(next_manifest));
}

typedef struct term_cow_fake_store
{
    const evoke_term_cow_tree *tree;
    uint64_t corrupt_object_id;
    uint32_t load_count;
} term_cow_fake_store;

typedef struct term_cow_composite_store
{
    term_cow_fake_store ancestor;
    term_cow_fake_store patch;
} term_cow_composite_store;

typedef struct cow_visit_counts
{
    uint32_t object_count;
    uint32_t node_count;
    uint32_t leaf_count;
} cow_visit_counts;

static evoke_status
count_term_cow_object(
    void *context,
    const evoke_term_cow_object *object
)
{
    cow_visit_counts *counts = context;

    if (counts == NULL || object == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    counts->object_count++;
    if (object->ref.kind == EVOKE_TERM_COW_OBJECT_NODE)
    {
        counts->node_count++;
    }
    else if (object->ref.kind == EVOKE_TERM_COW_OBJECT_LEAF)
    {
        counts->leaf_count++;
    }
    else
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

#define TERM_COW_TEST_MAX_LAYERS 4

typedef struct term_cow_layered_store
{
    term_cow_fake_store layers[TERM_COW_TEST_MAX_LAYERS];
    size_t layer_count;
} term_cow_layered_store;

static evoke_status
load_term_cow_fake_object(
    void *context,
    const evoke_term_cow_ref *ref,
    evoke_term_cow_object *object_out
)
{
    term_cow_fake_store *store = context;
    evoke_segment_object_ref storage_ref;
    uint8_t *bytes = NULL;
    size_t size = 0;
    evoke_status status;

    if (store == NULL || store->tree == NULL ||
        ref == NULL || object_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    store->load_count++;
    status = evoke_term_cow_tree_object_serialize(
        store->tree,
        ref,
        &bytes,
        &size
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (store->corrupt_object_id == ref->object_id)
    {
        bytes[size - 1] ^= UINT8_C(1);
    }
    status = evoke_term_cow_object_deserialize(
        bytes,
        size,
        object_out
    );
    free(bytes);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_term_cow_ref_as_segment_object_ref(
        ref,
        &storage_ref
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_term_cow_object_bind_storage(
        object_out,
        &storage_ref
    );
}

static evoke_status
load_term_cow_composite_object(
    void *context,
    const evoke_term_cow_ref *ref,
    evoke_term_cow_object *object_out
)
{
    term_cow_composite_store *store = context;

    if (store == NULL || ref == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (store->patch.tree != NULL &&
        ref->owner_manifest_id ==
            store->patch.tree->root.owner_manifest_id)
    {
        return load_term_cow_fake_object(
            &store->patch,
            ref,
            object_out
        );
    }
    return load_term_cow_fake_object(
        &store->ancestor,
        ref,
        object_out
    );
}

static evoke_status
load_term_cow_layered_object(
    void *context,
    const evoke_term_cow_ref *ref,
    evoke_term_cow_object *object_out
)
{
    term_cow_layered_store *store = context;

    if (store == NULL || ref == NULL ||
        store->layer_count > TERM_COW_TEST_MAX_LAYERS)
    {
        return EVOKE_ERR_INVALID;
    }
    for (size_t layer_index = 0;
         layer_index < store->layer_count;
         layer_index++)
    {
        term_cow_fake_store *layer = &store->layers[layer_index];

        if (layer->tree != NULL &&
            layer->tree->root.owner_manifest_id ==
                ref->owner_manifest_id)
        {
            return load_term_cow_fake_object(
                layer,
                ref,
                object_out
            );
        }
    }
    return EVOKE_ERR_FORMAT;
}

static void
clone_segment_manifest_for_test(
    const evoke_segment_manifest *source,
    evoke_segment_manifest *target
)
{
    uint8_t *bytes = NULL;
    size_t size = 0;

    evoke_segment_manifest_init(target);
    ASSERT_STATUS_OK(evoke_segment_manifest_serialize(
        source,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_deserialize(
        bytes,
        size,
        target
    ));
    free(bytes);
}

static void
bind_term_cow_test_tree(
    evoke_term_cow_tree *tree,
    uint64_t owner_manifest_id,
    uint32_t *next_block
)
{
    for (uint64_t object_id = 1;
         object_id <= tree->object_count;
         object_id++)
    {
        evoke_segment_object_ref storage_ref;
        uint8_t *bytes = NULL;
        size_t size = 0;
        uint32_t page_count = 0;

        ASSERT_STATUS_OK(evoke_term_cow_tree_prepare_object_for_storage(
            tree,
            object_id,
            &bytes,
            &size
        ));
        ASSERT_STATUS_OK(evoke_segment_page_count_required(
            size,
            8192,
            &page_count
        ));
        memset(&storage_ref, 0, sizeof(storage_ref));
        storage_ref.object_kind =
            EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY;
        storage_ref.start_block = *next_block;
        storage_ref.page_count = page_count;
        storage_ref.object_id = object_id;
        storage_ref.owner_manifest_id = owner_manifest_id;
        storage_ref.object_bytes = size;
        storage_ref.object_checksum =
            evoke_segment_blob_checksum(bytes, size);
        ASSERT_STATUS_OK(evoke_term_cow_tree_bind_object_storage(
            tree,
            object_id,
            &storage_ref
        ));
        *next_block += page_count;
        free(bytes);
    }
}

static void
test_empty_term_cow_external_closure(void)
{
    evoke_segment_manifest manifest;
    evoke_term_directory empty_directory;
    evoke_term_directory materialized_directory;
    evoke_term_cow_tree tree;
    term_cow_fake_store store;
    evoke_segment_object_ref storage_ref;
    uint32_t *doc_frequencies = NULL;
    uint8_t *bytes = NULL;
    size_t size = 0;

    evoke_segment_manifest_init(&manifest);
    evoke_term_directory_init(&empty_directory);
    evoke_term_directory_init(&materialized_directory);
    evoke_term_cow_tree_init(&tree);
    memset(&store, 0, sizeof(store));

    manifest.flags = EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY;
    manifest.manifest_id = 1;
    manifest.statistics_epoch = 1;
    initialize_test_object_ref(
        &manifest.query_contract,
        EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT,
        1,
        1,
        manifest.manifest_id,
        101
    );
    empty_directory.term_offsets = calloc(
        1,
        sizeof(*empty_directory.term_offsets)
    );
    ASSERT_TRUE(empty_directory.term_offsets != NULL);
    ASSERT_STATUS_OK(evoke_term_cow_tree_build(
        &empty_directory,
        &manifest,
        NULL,
        &tree
    ));
    ASSERT_TRUE(tree.object_count == 1);
    ASSERT_STATUS_OK(evoke_term_cow_tree_prepare_object_for_storage(
        &tree,
        tree.root.object_id,
        &bytes,
        &size
    ));
    initialize_test_object_ref(
        &storage_ref,
        EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
        2,
        1,
        manifest.manifest_id,
        evoke_segment_blob_checksum(bytes, size)
    );
    storage_ref.object_bytes = size;
    ASSERT_STATUS_OK(evoke_term_cow_tree_bind_object_storage(
        &tree,
        tree.root.object_id,
        &storage_ref
    ));
    ASSERT_STATUS_OK(evoke_term_cow_ref_as_segment_object_ref(
        &tree.root,
        &manifest.term_directory
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&manifest));

    store.tree = &tree;
    ASSERT_STATUS_OK(evoke_term_cow_validate_external(
        &tree.root,
        0,
        load_term_cow_fake_object,
        &store
    ));
    ASSERT_TRUE(store.load_count == 1);
    store.load_count = 0;
    ASSERT_STATUS_OK(evoke_term_cow_materialize_external(
        &tree.root,
        &manifest,
        load_term_cow_fake_object,
        &store,
        &materialized_directory,
        &doc_frequencies,
        NULL,
        NULL
    ));
    ASSERT_TRUE(store.load_count == 1);
    ASSERT_TRUE(materialized_directory.vocab_size == 0);
    ASSERT_TRUE(materialized_directory.extent_count == 0);
    ASSERT_TRUE(materialized_directory.term_offsets != NULL);
    ASSERT_TRUE(materialized_directory.term_offsets[0] == 0);
    ASSERT_TRUE(doc_frequencies == NULL);

    free(bytes);
    evoke_term_cow_tree_free(&tree);
    evoke_term_directory_free(&materialized_directory);
    evoke_term_directory_free(&empty_directory);
    evoke_segment_manifest_free(&manifest);
}

static void
test_initial_folded_term_cow_tree(void)
{
    evoke_segment_manifest manifest;
    evoke_term_cow_tree tree;
    evoke_segment_object_ref fold_refs[3] = {{0}};
    evoke_term_cow_record record;

    evoke_segment_manifest_init(&manifest);
    evoke_term_cow_tree_init(&tree);
    manifest.manifest_id = 41;
    manifest.max_sequence = 9;
    manifest.statistics_epoch = 41;
    manifest.visible_document_count = 9;
    manifest.document_slot_count = 9;
    manifest.vocab_size = 3;
    manifest.doc_frequencies = calloc(
        manifest.vocab_size,
        sizeof(*manifest.doc_frequencies)
    );
    ASSERT_TRUE(manifest.doc_frequencies != NULL);
    manifest.doc_frequencies[1] = 2;
    manifest.doc_frequencies[2] = 1;
    initialize_test_object_ref(
        &fold_refs[1],
        EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
        100,
        2,
        manifest.manifest_id,
        901
    );
    fold_refs[2] = fold_refs[1];

    ASSERT_STATUS_OK(evoke_term_cow_tree_build_initial_folds(
        &manifest,
        NULL,
        fold_refs,
        &tree
    ));
    ASSERT_STATUS_OK(evoke_term_cow_tree_validate(&tree));
    ASSERT_STATUS_OK(evoke_term_cow_tree_lookup(&tree, 1, &record));
    ASSERT_TRUE(record.raw_document_frequency == 2);
    ASSERT_TRUE(record.extent_count == 0);
    ASSERT_TRUE(
        (record.flags & EVOKE_TERM_COW_RECORD_FLAG_NEUTRAL_FOLD) != 0
    );
    ASSERT_TRUE(record.neutral_fold_coverage == manifest.max_sequence);
    ASSERT_TRUE(evoke_segment_object_ref_equal(
        &record.neutral_fold,
        &fold_refs[1]
    ));
    ASSERT_STATUS_OK(evoke_term_cow_tree_lookup(&tree, 2, &record));
    ASSERT_TRUE(evoke_segment_object_ref_equal(
        &record.neutral_fold,
        &fold_refs[2]
    ));

    memset(&fold_refs[2], 0, sizeof(fold_refs[2]));
    ASSERT_TRUE(
        evoke_term_cow_tree_build_initial_folds(
            &manifest,
            NULL,
            fold_refs,
            &tree
        ) == EVOKE_ERR_FORMAT
    );

    evoke_term_cow_tree_free(&tree);
    evoke_segment_manifest_free(&manifest);
}

static void
test_term_cow_neutral_fold_patch(void)
{
    evoke_segment_manifest old_manifest;
    evoke_segment_manifest next_manifest;
    evoke_term_directory directory;
    evoke_term_cow_tree ancestor;
    evoke_term_cow_tree patch;
    evoke_term_cow_tree impact_patch;
    evoke_term_cow_tree rejected;
    evoke_term_cow_update_stats stats;
    evoke_term_directory materialized_directory;
    evoke_segment_term_run runs[] = {
        {
            .term_id = 1,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 1
        },
        {
            .term_id = 17,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 1,
            .posting_count = 1
        },
        {
            .term_id = 33,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 2,
            .posting_count = 1
        }
    };
    uint32_t indices[] = {0, 1, 0};
    evoke_posting_value values[] = {
        {.term_frequency = 1},
        {.term_frequency = 1},
        {.term_frequency = 1}
    };
    evoke_segment_payload_view payload = {
        .segment_id = 100,
        .posting_count = 3,
        .runs = runs,
        .run_count = 3,
        .values = values,
        .indices = indices,
        .local_document_count = 2
    };
    term_cow_fake_store ancestor_store;
    term_cow_composite_store composite_store;
    term_cow_layered_store layered_store;
    evoke_segment_object_ref fold_ref;
    evoke_segment_object_ref impact_ref;
    evoke_term_cow_fold_state *fold_states = NULL;
    uint32_t *doc_frequencies = NULL;
    evoke_term_cow_record record;
    uint32_t next_block = 200;

    initialize_term_cow_test_manifests(
        &old_manifest,
        &next_manifest
    );
    next_manifest.max_sequence = old_manifest.max_sequence;
    next_manifest.statistics_epoch = old_manifest.statistics_epoch;
    next_manifest.visible_document_count =
        old_manifest.visible_document_count;
    next_manifest.document_slot_count =
        old_manifest.document_slot_count;
    next_manifest.total_document_length =
        old_manifest.total_document_length;
    next_manifest.segment_count = old_manifest.segment_count;
    memcpy(
        next_manifest.doc_frequencies,
        old_manifest.doc_frequencies,
        (size_t) old_manifest.vocab_size *
            sizeof(*old_manifest.doc_frequencies)
    );
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&next_manifest));
    evoke_term_directory_init(&directory);
    evoke_term_directory_init(&materialized_directory);
    evoke_term_cow_tree_init(&ancestor);
    evoke_term_cow_tree_init(&patch);
    evoke_term_cow_tree_init(&impact_patch);
    evoke_term_cow_tree_init(&rejected);
    memset(&ancestor_store, 0, sizeof(ancestor_store));
    memset(&composite_store, 0, sizeof(composite_store));
    memset(&layered_store, 0, sizeof(layered_store));
    ASSERT_STATUS_OK(evoke_term_directory_build_from_payloads(
        &old_manifest,
        &payload,
        1,
        &directory
    ));
    ASSERT_STATUS_OK(evoke_term_cow_tree_build(
        &directory,
        &old_manifest,
        NULL,
        &ancestor
    ));
    bind_term_cow_test_tree(
        &ancestor,
        old_manifest.manifest_id,
        &next_block
    );
    ancestor_store.tree = &ancestor;
    initialize_test_object_ref(
        &fold_ref,
        EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
        500,
        1,
        next_manifest.manifest_id,
        901
    );

    ASSERT_TRUE(
        evoke_term_cow_build_external_neutral_fold_patch(
            &ancestor.root,
            &old_manifest,
            next_manifest.manifest_id,
            17,
            &fold_ref,
            5,
            EVOKE_TERM_COW_NEUTRAL_FOLD_MAJOR,
            load_term_cow_fake_object,
            &ancestor_store,
            &rejected,
            &stats
        ) == EVOKE_ERR_FORMAT
    );
    ASSERT_TRUE(
        evoke_term_cow_build_external_neutral_fold_patch(
            &ancestor.root,
            &old_manifest,
            next_manifest.manifest_id,
            17,
            &fold_ref,
            10,
            EVOKE_TERM_COW_NEUTRAL_FOLD_MINOR,
            load_term_cow_fake_object,
            &ancestor_store,
            &rejected,
            &stats
        ) == EVOKE_ERR_FORMAT
    );
    ASSERT_STATUS_OK(evoke_term_cow_build_external_neutral_fold_patch(
        &ancestor.root,
        &old_manifest,
        next_manifest.manifest_id,
        17,
        &fold_ref,
        10,
        EVOKE_TERM_COW_NEUTRAL_FOLD_MAJOR,
        load_term_cow_fake_object,
        &ancestor_store,
        &patch,
        &stats
    ));
    ASSERT_TRUE(stats.changed_terms == 1);
    ASSERT_TRUE(stats.changed_leaves == 1);
    ASSERT_TRUE(stats.written_leaves == 1);
    ASSERT_TRUE(
        stats.written_nodes == EVOKE_TERM_COW_RADIX_LEVELS
    );
    bind_term_cow_test_tree(
        &patch,
        next_manifest.manifest_id,
        &next_block
    );
    composite_store.ancestor.tree = &ancestor;
    composite_store.patch.tree = &patch;
    ASSERT_STATUS_OK(evoke_term_cow_validate_external(
        &patch.root,
        old_manifest.vocab_size,
        load_term_cow_composite_object,
        &composite_store
    ));
    ASSERT_STATUS_OK(evoke_term_cow_lookup_external(
        &patch.root,
        old_manifest.vocab_size,
        17,
        load_term_cow_composite_object,
        &composite_store,
        &record
    ));
    ASSERT_TRUE(
        (record.flags &
         EVOKE_TERM_COW_RECORD_FLAG_NEUTRAL_FOLD) != 0
    );
    ASSERT_TRUE(record.neutral_fold_coverage == 10);
    ASSERT_TRUE(memcmp(
        &record.neutral_fold,
        &fold_ref,
        sizeof(fold_ref)
    ) == 0);
    ASSERT_TRUE(record.extent_count == 0);
    ASSERT_TRUE(record.raw_document_frequency == 1);
    ASSERT_STATUS_OK(evoke_term_cow_materialize_external(
        &patch.root,
        &next_manifest,
        load_term_cow_composite_object,
        &composite_store,
        &materialized_directory,
        &doc_frequencies,
        NULL,
        &fold_states
    ));
    ASSERT_TRUE(materialized_directory.extent_count == 2);
    ASSERT_TRUE(
        materialized_directory.term_offsets[18] ==
        materialized_directory.term_offsets[17]
    );
    ASSERT_TRUE(fold_states != NULL);
    ASSERT_TRUE(fold_states[17].neutral_coverage == 10);
    ASSERT_TRUE(memcmp(
        &fold_states[17].neutral_ref,
        &fold_ref,
        sizeof(fold_ref)
    ) == 0);
    initialize_test_object_ref(
        &impact_ref,
        EVOKE_SEGMENT_OBJECT_IMPACT_FOLD,
        700,
        1,
        next_manifest.manifest_id + 1,
        902
    );
    ASSERT_TRUE(
        evoke_term_cow_build_external_impact_fold_patch(
            &patch.root,
            &next_manifest,
            next_manifest.manifest_id + 1,
            17,
            &impact_ref,
            9,
            next_manifest.statistics_epoch,
            load_term_cow_composite_object,
            &composite_store,
            &rejected,
            &stats
        ) == EVOKE_ERR_FORMAT
    );
    ASSERT_TRUE(
        evoke_term_cow_build_external_impact_fold_patch(
            &patch.root,
            &next_manifest,
            next_manifest.manifest_id + 1,
            17,
            &impact_ref,
            10,
            next_manifest.statistics_epoch + 1,
            load_term_cow_composite_object,
            &composite_store,
            &rejected,
            &stats
        ) == EVOKE_ERR_INVALID
    );
    ASSERT_STATUS_OK(evoke_term_cow_build_external_impact_fold_patch(
        &patch.root,
        &next_manifest,
        next_manifest.manifest_id + 1,
        17,
        &impact_ref,
        10,
        next_manifest.statistics_epoch,
        load_term_cow_composite_object,
        &composite_store,
        &impact_patch,
        &stats
    ));
    ASSERT_TRUE(stats.changed_terms == 1);
    ASSERT_TRUE(stats.changed_leaves == 1);
    ASSERT_TRUE(stats.written_leaves == 1);
    ASSERT_TRUE(
        stats.written_nodes == EVOKE_TERM_COW_RADIX_LEVELS
    );
    bind_term_cow_test_tree(
        &impact_patch,
        next_manifest.manifest_id + 1,
        &next_block
    );
    layered_store.layer_count = 3;
    layered_store.layers[0].tree = &impact_patch;
    layered_store.layers[1].tree = &patch;
    layered_store.layers[2].tree = &ancestor;
    ASSERT_STATUS_OK(evoke_term_cow_validate_external(
        &impact_patch.root,
        old_manifest.vocab_size,
        load_term_cow_layered_object,
        &layered_store
    ));
    ASSERT_STATUS_OK(evoke_term_cow_lookup_external(
        &impact_patch.root,
        old_manifest.vocab_size,
        17,
        load_term_cow_layered_object,
        &layered_store,
        &record
    ));
    ASSERT_TRUE(
        (record.flags &
         EVOKE_TERM_COW_RECORD_FLAG_IMPACT_FOLD) != 0
    );
    ASSERT_TRUE(record.impact_fold_coverage == 10);
    ASSERT_TRUE(
        record.impact_statistics_epoch ==
            next_manifest.statistics_epoch
    );
    ASSERT_TRUE(memcmp(
        &record.impact_fold,
        &impact_ref,
        sizeof(impact_ref)
    ) == 0);
    ASSERT_TRUE(record.extent_count == 0);
    ASSERT_TRUE(record.raw_document_frequency == 1);

    free(fold_states);
    free(doc_frequencies);
    evoke_term_cow_tree_free(&rejected);
    evoke_term_cow_tree_free(&impact_patch);
    evoke_term_cow_tree_free(&patch);
    evoke_term_cow_tree_free(&ancestor);
    evoke_term_directory_free(&materialized_directory);
    evoke_term_directory_free(&directory);
    evoke_segment_manifest_free(&next_manifest);
    evoke_segment_manifest_free(&old_manifest);
}

static void
test_term_cow_external_append_patch(void)
{
    evoke_segment_manifest old_manifest;
    evoke_segment_manifest next_manifest;
    evoke_term_directory old_directory;
    evoke_term_directory expected_directory;
    evoke_term_directory materialized_directory;
    evoke_term_cow_tree ancestor;
    evoke_term_cow_tree patch;
    evoke_term_cow_tree catalog_ancestor;
    evoke_term_cow_tree rejected_patch;
    evoke_term_cow_update_stats stats;
    evoke_segment_object_ref catalog_ref;
    evoke_segment_term_run new_runs[] = {
        {
            .term_id = 17,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 1
        },
        {
            .term_id = 50,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 1,
            .posting_count = 1
        },
        {
            .term_id = 70,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 2,
            .posting_count = 1
        }
    };
    uint32_t new_indices[] = {0, 0, 0};
    evoke_posting_value new_values[] = {
        {.term_frequency = 1},
        {.term_frequency = 1},
        {.term_frequency = 1}
    };
    evoke_segment_payload_view new_payload = {
        .segment_id = 101,
        .posting_count = 3,
        .runs = new_runs,
        .run_count = 3,
        .values = new_values,
        .indices = new_indices,
        .document_id_base = 2,
        .local_document_count = 1
    };
    evoke_segment_term_run overflow_runs[
        EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM
    ] = {{0}};
    evoke_segment_payload_view overflow_payload = {
        .runs = overflow_runs,
        .run_count = EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM
    };
    term_cow_fake_store ancestor_store;
    term_cow_composite_store composite_store;
    evoke_term_cow_record pressure_record;
    bool found_pressure = false;
    bool has_capacity = false;
    cow_visit_counts visit_counts = {0};
    uint32_t *materialized_frequencies = NULL;
    uint32_t next_block = 500;
    uint8_t *compact_leaf_bytes = NULL;
    size_t compact_leaf_size = 0;

    initialize_term_cow_test_manifests(
        &old_manifest,
        &next_manifest
    );
    evoke_term_directory_init(&old_directory);
    evoke_term_directory_init(&expected_directory);
    evoke_term_directory_init(&materialized_directory);
    evoke_term_cow_tree_init(&ancestor);
    evoke_term_cow_tree_init(&patch);
    evoke_term_cow_tree_init(&catalog_ancestor);
    evoke_term_cow_tree_init(&rejected_patch);
    memset(&ancestor_store, 0, sizeof(ancestor_store));
    memset(&composite_store, 0, sizeof(composite_store));

    old_directory.vocab_size = old_manifest.vocab_size;
    old_directory.extent_count = 3;
    old_directory.term_offsets = calloc(
        (size_t) old_directory.vocab_size + 1,
        sizeof(*old_directory.term_offsets)
    );
    old_directory.extents = calloc(
        old_directory.extent_count,
        sizeof(*old_directory.extents)
    );
    ASSERT_TRUE(old_directory.term_offsets != NULL);
    ASSERT_TRUE(old_directory.extents != NULL);
    {
        uint32_t extent_cursor = 0;

        for (uint32_t term_id = 0;
             term_id < old_directory.vocab_size;
             term_id++)
        {
            old_directory.term_offsets[term_id] = extent_cursor;
            if (term_id == 1 || term_id == 17 || term_id == 33)
            {
                old_directory.extents[extent_cursor].segment_index = 0;
                old_directory.extents[extent_cursor].kind =
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
                old_directory.extents[extent_cursor].posting_offset =
                    extent_cursor;
                old_directory.extents[extent_cursor].posting_count = 1;
                extent_cursor++;
            }
        }
        old_directory.term_offsets[old_directory.vocab_size] =
            extent_cursor;
    }
    ASSERT_STATUS_OK(evoke_term_directory_validate(
        &old_directory,
        &old_manifest
    ));
    ASSERT_STATUS_OK(evoke_term_cow_tree_build(
        &old_directory,
        &old_manifest,
        NULL,
        &ancestor
    ));
    ASSERT_STATUS_OK(evoke_term_cow_tree_prepare_object_for_storage(
        &ancestor,
        1,
        &compact_leaf_bytes,
        &compact_leaf_size
    ));
    ASSERT_TRUE(compact_leaf_size < 8192);
    free(compact_leaf_bytes);
    bind_term_cow_test_tree(
        &ancestor,
        old_manifest.manifest_id,
        &next_block
    );
    ancestor_store.tree = &ancestor;
    ASSERT_STATUS_OK(evoke_term_cow_append_has_capacity_external(
        &ancestor.root,
        old_manifest.vocab_size,
        &new_payload,
        load_term_cow_fake_object,
        &ancestor_store,
        &has_capacity
    ));
    ASSERT_TRUE(has_capacity);
    for (uint32_t run_index = 0;
         run_index < overflow_payload.run_count;
         run_index++)
    {
        overflow_runs[run_index].term_id = 17;
    }
    ASSERT_STATUS_OK(evoke_term_cow_append_has_capacity_external(
        &ancestor.root,
        old_manifest.vocab_size,
        &overflow_payload,
        load_term_cow_fake_object,
        &ancestor_store,
        &has_capacity
    ));
    ASSERT_TRUE(!has_capacity);
    ancestor_store.load_count = 0;

    ASSERT_STATUS_OK(evoke_term_cow_build_external_append_patch(
        &ancestor.root,
        &old_manifest,
        &next_manifest,
        &new_payload,
        NULL,
        load_term_cow_fake_object,
        &ancestor_store,
        &patch,
        &stats
    ));
    ASSERT_TRUE(stats.changed_terms == 3);
    ASSERT_TRUE(stats.changed_leaves == 3);
    ASSERT_TRUE(stats.written_leaves == 3);
    ASSERT_TRUE(stats.written_nodes == EVOKE_TERM_COW_RADIX_LEVELS);
    ASSERT_TRUE(
        patch.object_count ==
        3 + EVOKE_TERM_COW_RADIX_LEVELS
    );
    ASSERT_TRUE(ancestor_store.load_count <=
        3 * (EVOKE_TERM_COW_RADIX_LEVELS + 1));
    ASSERT_TRUE(patch.root.start_block == 0);
    ASSERT_TRUE(patch.root.page_count == 0);
    ASSERT_TRUE(patch.root.owner_manifest_id == 0);
    assert_retired_ranges(
        patch.retired_ranges,
        patch.retired_range_count,
        next_block
    );

    bind_term_cow_test_tree(
        &patch,
        next_manifest.manifest_id,
        &next_block
    );
    composite_store.ancestor.tree = &ancestor;
    composite_store.patch.tree = &patch;
    ASSERT_STATUS_OK(evoke_term_cow_validate_external(
        &patch.root,
        next_manifest.vocab_size,
        load_term_cow_composite_object,
        &composite_store
    ));
    composite_store.ancestor.load_count = 0;
    composite_store.patch.load_count = 0;
    ASSERT_STATUS_OK(evoke_term_cow_visit_external(
        &patch.root,
        next_manifest.vocab_size,
        load_term_cow_composite_object,
        &composite_store,
        count_term_cow_object,
        &visit_counts
    ));
    ASSERT_TRUE(
        visit_counts.object_count ==
        composite_store.ancestor.load_count +
            composite_store.patch.load_count
    );
    ASSERT_TRUE(visit_counts.node_count > 0);
    ASSERT_TRUE(visit_counts.leaf_count > 0);
    ASSERT_TRUE(composite_store.ancestor.load_count > 0);
    ASSERT_TRUE(composite_store.patch.load_count > 0);
    composite_store.ancestor.load_count = 0;
    composite_store.patch.load_count = 0;
    ASSERT_STATUS_OK(evoke_term_cow_find_extent_pressure_external(
        &patch.root,
        next_manifest.vocab_size,
        2,
        load_term_cow_composite_object,
        &composite_store,
        &pressure_record,
        &found_pressure
    ));
    ASSERT_TRUE(found_pressure);
    ASSERT_TRUE(pressure_record.term_id == 17);
    ASSERT_TRUE(pressure_record.extent_count == 2);
    ASSERT_TRUE(
        composite_store.ancestor.load_count +
            composite_store.patch.load_count <=
        EVOKE_TERM_COW_RADIX_LEVELS + 1
    );
    composite_store.ancestor.load_count = 0;
    composite_store.patch.load_count = 0;
    ASSERT_STATUS_OK(evoke_term_cow_find_extent_pressure_external(
        &patch.root,
        next_manifest.vocab_size,
        3,
        load_term_cow_composite_object,
        &composite_store,
        &pressure_record,
        &found_pressure
    ));
    ASSERT_TRUE(!found_pressure);
    ASSERT_TRUE(
        composite_store.ancestor.load_count +
            composite_store.patch.load_count == 1
    );
    ASSERT_STATUS_OK(evoke_term_directory_append_payload(
        &old_directory,
        &old_manifest,
        &next_manifest,
        &new_payload,
        &expected_directory
    ));
    ASSERT_STATUS_OK(evoke_term_cow_materialize_external(
        &patch.root,
        &next_manifest,
        load_term_cow_composite_object,
        &composite_store,
        &materialized_directory,
        &materialized_frequencies,
        NULL,
        NULL
    ));
    ASSERT_TRUE(
        expected_directory.extent_count ==
        materialized_directory.extent_count
    );
    assert_uint64_array(
        expected_directory.term_offsets,
        materialized_directory.term_offsets,
        (size_t) expected_directory.vocab_size + 1
    );
    ASSERT_TRUE(memcmp(
        expected_directory.extents,
        materialized_directory.extents,
        (size_t) expected_directory.extent_count *
            sizeof(*expected_directory.extents)
    ) == 0);
    assert_uint32_array(
        materialized_frequencies,
        next_manifest.doc_frequencies,
        next_manifest.vocab_size
    );

    initialize_test_object_ref(
        &catalog_ref,
        EVOKE_SEGMENT_OBJECT_LEXICAL_CATALOG,
        450,
        1,
        old_manifest.manifest_id,
        901
    );
    ASSERT_STATUS_OK(evoke_term_cow_tree_build(
        &old_directory,
        &old_manifest,
        &catalog_ref,
        &catalog_ancestor
    ));
    bind_term_cow_test_tree(
        &catalog_ancestor,
        old_manifest.manifest_id,
        &next_block
    );
    ancestor_store.tree = &catalog_ancestor;
    next_manifest.vocab_size++;
    ASSERT_TRUE(evoke_term_cow_build_external_append_patch(
        &catalog_ancestor.root,
        &old_manifest,
        &next_manifest,
        &new_payload,
        NULL,
        load_term_cow_fake_object,
        &ancestor_store,
        &rejected_patch,
        &stats
    ) == EVOKE_ERR_FORMAT);
    next_manifest.vocab_size--;

    free(materialized_frequencies);
    evoke_term_cow_tree_free(&rejected_patch);
    evoke_term_cow_tree_free(&catalog_ancestor);
    evoke_term_cow_tree_free(&patch);
    evoke_term_cow_tree_free(&ancestor);
    evoke_term_directory_free(&materialized_directory);
    evoke_term_directory_free(&expected_directory);
    evoke_term_directory_free(&old_directory);
    evoke_segment_manifest_free(&next_manifest);
    evoke_segment_manifest_free(&old_manifest);
}

static void
test_term_cow_external_replace_patch(void)
{
    evoke_segment_manifest first_manifest;
    evoke_segment_manifest old_manifest;
    evoke_segment_manifest next_manifest;
    evoke_segment_manifest folded_manifest;
    evoke_segment_manifest folded_next_manifest;
    evoke_segment_manifest coverage_only_manifest;
    evoke_segment_manifest straddling_manifest;
    evoke_segment_manifest minor_manifest;
    evoke_segment_manifest promoted_manifest;
    evoke_term_directory first_directory;
    evoke_term_directory old_directory;
    evoke_term_directory expected_directory;
    evoke_term_directory materialized_directory;
    evoke_term_cow_tree ancestor;
    evoke_term_cow_tree patch;
    evoke_term_cow_tree fold_patch;
    evoke_term_cow_tree prefix_fold_patch;
    evoke_term_cow_tree minor_fold_patch;
    evoke_term_cow_tree promoted_fold_patch;
    evoke_term_cow_tree folded_replace_patch;
    evoke_term_cow_tree rejected_patch;
    evoke_term_cow_update_stats stats;
    evoke_segment_term_run first_runs[] = {
        {
            .term_id = 1,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 1
        },
        {
            .term_id = 17,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 1,
            .posting_count = 1
        },
        {
            .term_id = 33,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 2,
            .posting_count = 1
        }
    };
    evoke_segment_term_run second_runs[] = {
        {
            .term_id = 17,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 1
        },
        {
            .term_id = 50,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 1,
            .posting_count = 1
        },
        {
            .term_id = 70,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 2,
            .posting_count = 1
        }
    };
    evoke_segment_term_run replacement_runs[] = {
        {
            .term_id = 1,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 1
        },
        {
            .term_id = 17,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 1,
            .posting_count = 2
        },
        {
            .term_id = 33,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 3,
            .posting_count = 1
        },
        {
            .term_id = 50,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 4,
            .posting_count = 1
        },
        {
            .term_id = 70,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 5,
            .posting_count = 1
        }
    };
    uint32_t first_indices[] = {0, 1, 1};
    uint32_t second_indices[] = {0, 0, 0};
    uint32_t replacement_indices[] = {0, 1, 2, 1, 2, 2};
    evoke_posting_value first_values[3] = {
        {.term_frequency = 1},
        {.term_frequency = 1},
        {.term_frequency = 1}
    };
    evoke_posting_value second_values[3] = {
        {.term_frequency = 1},
        {.term_frequency = 1},
        {.term_frequency = 1}
    };
    evoke_posting_value replacement_values[6] = {
        {.term_frequency = 1},
        {.term_frequency = 1},
        {.term_frequency = 1},
        {.term_frequency = 1},
        {.term_frequency = 1},
        {.term_frequency = 1}
    };
    evoke_segment_payload_view replaced_payloads[2] = {
        {
            .segment_id = 100,
            .posting_count = 3,
            .runs = first_runs,
            .run_count = 3,
            .values = first_values,
            .indices = first_indices,
            .document_id_base = 0,
            .local_document_count = 2
        },
        {
            .segment_id = 101,
            .posting_count = 3,
            .runs = second_runs,
            .run_count = 3,
            .values = second_values,
            .indices = second_indices,
            .document_id_base = 2,
            .local_document_count = 1
        }
    };
    evoke_segment_payload_view replacement_payload = {
        .segment_id = 12,
        .posting_count = 6,
        .runs = replacement_runs,
        .run_count = 5,
        .values = replacement_values,
        .indices = replacement_indices,
        .document_id_base = 0,
        .local_document_count = 3
    };
    term_cow_fake_store ancestor_store;
    term_cow_composite_store composite_store;
    term_cow_composite_store folded_source_store;
    term_cow_composite_store prefix_source_store;
    term_cow_layered_store folded_result_store;
    term_cow_layered_store minor_result_store;
    term_cow_layered_store promoted_result_store;
    evoke_segment_object_ref fold_ref;
    evoke_segment_object_ref prefix_fold_ref;
    evoke_segment_object_ref minor_fold_ref;
    evoke_segment_object_ref promoted_fold_ref;
    evoke_segment_payload_view folded_replacement_payload;
    evoke_segment_payload_view coverage_only_payloads[2];
    evoke_segment_term_run coverage_only_tail_runs[2] = {
        {
            .term_id = 50,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 1
        },
        {
            .term_id = 70,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 1,
            .posting_count = 1
        }
    };
    uint32_t coverage_only_tail_indices[2] = {0, 0};
    evoke_posting_value coverage_only_tail_values[2] = {
        {.term_frequency = 1},
        {.term_frequency = 1}
    };
    evoke_term_cow_record folded_record;
    evoke_term_cow_record minor_record;
    evoke_term_cow_record promoted_record;
    evoke_term_cow_fold_state *fold_states = NULL;
    uint32_t *materialized_frequencies = NULL;
    uint32_t next_block = 700;
    uint32_t conflict_term_id = UINT32_MAX;
    bool fold_safe = false;

    initialize_term_cow_test_manifests(
        &first_manifest,
        &old_manifest
    );
    evoke_segment_manifest_init(&next_manifest);
    evoke_segment_manifest_init(&folded_manifest);
    evoke_segment_manifest_init(&folded_next_manifest);
    evoke_segment_manifest_init(&coverage_only_manifest);
    evoke_segment_manifest_init(&straddling_manifest);
    evoke_segment_manifest_init(&minor_manifest);
    evoke_segment_manifest_init(&promoted_manifest);
    evoke_term_directory_init(&first_directory);
    evoke_term_directory_init(&old_directory);
    evoke_term_directory_init(&expected_directory);
    evoke_term_directory_init(&materialized_directory);
    evoke_term_cow_tree_init(&ancestor);
    evoke_term_cow_tree_init(&patch);
    evoke_term_cow_tree_init(&fold_patch);
    evoke_term_cow_tree_init(&prefix_fold_patch);
    evoke_term_cow_tree_init(&minor_fold_patch);
    evoke_term_cow_tree_init(&promoted_fold_patch);
    evoke_term_cow_tree_init(&folded_replace_patch);
    evoke_term_cow_tree_init(&rejected_patch);
    memset(&ancestor_store, 0, sizeof(ancestor_store));
    memset(&composite_store, 0, sizeof(composite_store));
    memset(&folded_source_store, 0, sizeof(folded_source_store));
    memset(&prefix_source_store, 0, sizeof(prefix_source_store));
    memset(&folded_result_store, 0, sizeof(folded_result_store));
    memset(&minor_result_store, 0, sizeof(minor_result_store));
    memset(&promoted_result_store, 0, sizeof(promoted_result_store));

    first_directory.vocab_size = first_manifest.vocab_size;
    first_directory.extent_count = 3;
    first_directory.term_offsets = calloc(
        (size_t) first_directory.vocab_size + 1,
        sizeof(*first_directory.term_offsets)
    );
    first_directory.extents = calloc(
        first_directory.extent_count,
        sizeof(*first_directory.extents)
    );
    ASSERT_TRUE(first_directory.term_offsets != NULL);
    ASSERT_TRUE(first_directory.extents != NULL);
    {
        uint32_t extent_cursor = 0;

        for (uint32_t term_id = 0;
             term_id < first_directory.vocab_size;
             term_id++)
        {
            first_directory.term_offsets[term_id] = extent_cursor;
            if (term_id == 1 || term_id == 17 || term_id == 33)
            {
                first_directory.extents[extent_cursor].segment_index = 0;
                first_directory.extents[extent_cursor].kind =
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
                first_directory.extents[extent_cursor].posting_offset =
                    extent_cursor;
                first_directory.extents[extent_cursor].posting_count = 1;
                extent_cursor++;
            }
        }
        first_directory.term_offsets[first_directory.vocab_size] =
            extent_cursor;
    }
    ASSERT_STATUS_OK(evoke_term_directory_append_payload(
        &first_directory,
        &first_manifest,
        &old_manifest,
        &replaced_payloads[1],
        &old_directory
    ));
    ASSERT_STATUS_OK(evoke_term_cow_tree_build(
        &old_directory,
        &old_manifest,
        NULL,
        &ancestor
    ));
    bind_term_cow_test_tree(
        &ancestor,
        old_manifest.manifest_id,
        &next_block
    );
    ancestor_store.tree = &ancestor;
    old_manifest.flags &=
        ~EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY;
    old_manifest.flags |=
        EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY;
    free(old_manifest.doc_frequencies);
    old_manifest.doc_frequencies = NULL;
    ASSERT_STATUS_OK(evoke_term_cow_ref_as_segment_object_ref(
        &ancestor.root,
        &old_manifest.term_directory
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&old_manifest));

    next_manifest.flags = 0;
    next_manifest.manifest_id = 12;
    next_manifest.parent_manifest_id = old_manifest.manifest_id;
    next_manifest.max_sequence = old_manifest.max_sequence;
    next_manifest.statistics_epoch = old_manifest.statistics_epoch;
    next_manifest.visible_document_count =
        old_manifest.visible_document_count;
    next_manifest.document_slot_count =
        old_manifest.document_slot_count;
    next_manifest.total_document_length =
        old_manifest.total_document_length;
    next_manifest.vocab_size = old_manifest.vocab_size;
    next_manifest.query_contract = old_manifest.query_contract;
    next_manifest.document_directory =
        old_manifest.document_directory;
    memcpy(
        next_manifest.contract_hash,
        old_manifest.contract_hash,
        sizeof(next_manifest.contract_hash)
    );
    next_manifest.segment_count = 1;
    next_manifest.segments = calloc(
        next_manifest.segment_count,
        sizeof(*next_manifest.segments)
    );
    ASSERT_TRUE(next_manifest.segments != NULL);
    next_manifest.segments[0].segment_id = 12;
    next_manifest.segments[0].min_sequence = 1;
    next_manifest.segments[0].max_sequence = 11;
    next_manifest.segments[0].posting_count = 6;
    next_manifest.segments[0].document_count = 3;
    next_manifest.segments[0].total_document_length = 7;
    next_manifest.segments[0].first_document_slot = 0;
    next_manifest.segments[0].document_slot_count = 3;
    next_manifest.segments[0].flags =
        EVOKE_SEGMENT_FLAG_SEALED |
        EVOKE_SEGMENT_FLAG_LEXICAL;

    ASSERT_STATUS_OK(evoke_term_directory_replace_payloads(
        &old_directory,
        &old_manifest,
        &next_manifest,
        0,
        2,
        &replacement_payload,
        &expected_directory
    ));
    ASSERT_STATUS_OK(evoke_term_cow_build_external_replace_patch(
        &ancestor.root,
        &old_manifest,
        &next_manifest,
        0,
        replaced_payloads,
        2,
        &replacement_payload,
        load_term_cow_fake_object,
        &ancestor_store,
        &patch,
        &stats
    ));
    ASSERT_TRUE(stats.changed_terms == 5);
    ASSERT_TRUE(stats.changed_leaves == 5);
    ASSERT_TRUE(stats.written_leaves == 5);
    ASSERT_TRUE(stats.written_nodes == EVOKE_TERM_COW_RADIX_LEVELS);
    assert_retired_ranges(
        patch.retired_ranges,
        patch.retired_range_count,
        next_block
    );
    bind_term_cow_test_tree(
        &patch,
        next_manifest.manifest_id,
        &next_block
    );
    composite_store.ancestor.tree = &ancestor;
    composite_store.patch.tree = &patch;
    next_manifest.segments[0].payload_checksum = 701;
    next_manifest.segments[0].start_block = 650;
    next_manifest.segments[0].block_count = 1;
    next_manifest.segments[0].payload_bytes = 512;
    next_manifest.segments[0].payload_owner_manifest_id =
        next_manifest.manifest_id;
    next_manifest.flags =
        EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY |
        EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY;
    ASSERT_STATUS_OK(evoke_term_cow_ref_as_segment_object_ref(
        &patch.root,
        &next_manifest.term_directory
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&next_manifest));
    ASSERT_STATUS_OK(evoke_term_cow_validate_external(
        &patch.root,
        next_manifest.vocab_size,
        load_term_cow_composite_object,
        &composite_store
    ));
    ASSERT_STATUS_OK(evoke_term_cow_materialize_external(
        &patch.root,
        &next_manifest,
        load_term_cow_composite_object,
        &composite_store,
        &materialized_directory,
        &materialized_frequencies,
        NULL,
        NULL
    ));
    ASSERT_TRUE(
        expected_directory.extent_count ==
        materialized_directory.extent_count
    );
    assert_uint64_array(
        expected_directory.term_offsets,
        materialized_directory.term_offsets,
        (size_t) expected_directory.vocab_size + 1
    );
    ASSERT_TRUE(memcmp(
        expected_directory.extents,
        materialized_directory.extents,
        (size_t) expected_directory.extent_count *
            sizeof(*expected_directory.extents)
    ) == 0);
    ASSERT_TRUE(materialized_frequencies[1] == 1);
    ASSERT_TRUE(materialized_frequencies[17] == 2);
    ASSERT_TRUE(materialized_frequencies[33] == 1);
    ASSERT_TRUE(materialized_frequencies[50] == 1);
    ASSERT_TRUE(materialized_frequencies[70] == 1);

    initialize_test_object_ref(
        &fold_ref,
        EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
        800,
        1,
        next_manifest.manifest_id,
        801
    );
    ASSERT_STATUS_OK(evoke_term_cow_build_external_neutral_fold_patch(
        &ancestor.root,
        &old_manifest,
        next_manifest.manifest_id,
        17,
        &fold_ref,
        old_manifest.max_sequence,
        EVOKE_TERM_COW_NEUTRAL_FOLD_MAJOR,
        load_term_cow_fake_object,
        &ancestor_store,
        &fold_patch,
        &stats
    ));
    bind_term_cow_test_tree(
        &fold_patch,
        next_manifest.manifest_id,
        &next_block
    );
    clone_segment_manifest_for_test(
        &old_manifest,
        &folded_manifest
    );
    folded_manifest.manifest_id = next_manifest.manifest_id;
    folded_manifest.parent_manifest_id = old_manifest.manifest_id;
    ASSERT_STATUS_OK(evoke_term_cow_ref_as_segment_object_ref(
        &fold_patch.root,
        &folded_manifest.term_directory
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(
        &folded_manifest
    ));

    clone_segment_manifest_for_test(
        &next_manifest,
        &folded_next_manifest
    );
    folded_next_manifest.manifest_id =
        next_manifest.manifest_id + 1;
    folded_next_manifest.parent_manifest_id =
        folded_manifest.manifest_id;
    folded_next_manifest.segments[0].segment_id =
        folded_next_manifest.manifest_id;
    folded_next_manifest.segments[0].payload_owner_manifest_id =
        folded_next_manifest.manifest_id;
    folded_replacement_payload = replacement_payload;
    folded_replacement_payload.segment_id =
        folded_next_manifest.manifest_id;
    folded_source_store.ancestor.tree = &ancestor;
    folded_source_store.patch.tree = &fold_patch;
    ASSERT_STATUS_OK(
        evoke_term_cow_replace_fold_preflight_external(
            &fold_patch.root,
            &folded_manifest,
            0,
            replaced_payloads,
            2,
            load_term_cow_composite_object,
            &folded_source_store,
            &fold_safe,
            &conflict_term_id
        )
    );
    ASSERT_TRUE(fold_safe);
    ASSERT_TRUE(conflict_term_id == UINT32_MAX);

    ASSERT_STATUS_OK(evoke_term_cow_build_external_replace_patch(
        &fold_patch.root,
        &folded_manifest,
        &folded_next_manifest,
        0,
        replaced_payloads,
        2,
        &folded_replacement_payload,
        load_term_cow_composite_object,
        &folded_source_store,
        &folded_replace_patch,
        &stats
    ));
    bind_term_cow_test_tree(
        &folded_replace_patch,
        folded_next_manifest.manifest_id,
        &next_block
    );
    ASSERT_STATUS_OK(evoke_term_cow_ref_as_segment_object_ref(
        &folded_replace_patch.root,
        &folded_next_manifest.term_directory
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(
        &folded_next_manifest
    ));
    folded_result_store.layer_count = 3;
    folded_result_store.layers[0].tree = &folded_replace_patch;
    folded_result_store.layers[1].tree = &fold_patch;
    folded_result_store.layers[2].tree = &ancestor;
    ASSERT_STATUS_OK(evoke_term_cow_validate_external(
        &folded_replace_patch.root,
        folded_next_manifest.vocab_size,
        load_term_cow_layered_object,
        &folded_result_store
    ));
    ASSERT_STATUS_OK(evoke_term_cow_lookup_external(
        &folded_replace_patch.root,
        folded_next_manifest.vocab_size,
        17,
        load_term_cow_layered_object,
        &folded_result_store,
        &folded_record
    ));
    ASSERT_TRUE(folded_record.extent_count == 0);
    ASSERT_TRUE(folded_record.raw_document_frequency == 2);
    ASSERT_TRUE(folded_record.neutral_fold_coverage == 11);
    ASSERT_TRUE(memcmp(
        &folded_record.neutral_fold,
        &fold_ref,
        sizeof(fold_ref)
    ) == 0);

    free(materialized_frequencies);
    materialized_frequencies = NULL;
    evoke_term_directory_free(&materialized_directory);
    evoke_term_directory_init(&materialized_directory);
    ASSERT_STATUS_OK(evoke_term_cow_materialize_external(
        &folded_replace_patch.root,
        &folded_next_manifest,
        load_term_cow_layered_object,
        &folded_result_store,
        &materialized_directory,
        &materialized_frequencies,
        NULL,
        &fold_states
    ));
    ASSERT_TRUE(
        materialized_directory.term_offsets[18] ==
        materialized_directory.term_offsets[17]
    );
    ASSERT_TRUE(materialized_frequencies[17] == 2);
    ASSERT_TRUE(fold_states[17].neutral_coverage == 11);
    ASSERT_TRUE(memcmp(
        &fold_states[17].neutral_ref,
        &fold_ref,
        sizeof(fold_ref)
    ) == 0);

    initialize_test_object_ref(
        &prefix_fold_ref,
        EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
        810,
        1,
        next_manifest.manifest_id,
        811
    );
    ASSERT_STATUS_OK(evoke_term_cow_build_external_neutral_fold_patch(
        &ancestor.root,
        &old_manifest,
        next_manifest.manifest_id,
        17,
        &prefix_fold_ref,
        old_manifest.segments[0].max_sequence,
        EVOKE_TERM_COW_NEUTRAL_FOLD_MAJOR,
        load_term_cow_fake_object,
        &ancestor_store,
        &prefix_fold_patch,
        &stats
    ));
    bind_term_cow_test_tree(
        &prefix_fold_patch,
        next_manifest.manifest_id,
        &next_block
    );
    ASSERT_STATUS_OK(evoke_term_cow_ref_as_segment_object_ref(
        &prefix_fold_patch.root,
        &folded_manifest.term_directory
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(
        &folded_manifest
    ));
    prefix_source_store.ancestor.tree = &ancestor;
    prefix_source_store.patch.tree = &prefix_fold_patch;

    clone_segment_manifest_for_test(
        &folded_manifest,
        &minor_manifest
    );
    minor_manifest.manifest_id = folded_manifest.manifest_id + 1;
    minor_manifest.parent_manifest_id = folded_manifest.manifest_id;
    initialize_test_object_ref(
        &minor_fold_ref,
        EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
        820,
        1,
        minor_manifest.manifest_id,
        821
    );
    ASSERT_STATUS_OK(evoke_term_cow_build_external_neutral_fold_patch(
        &prefix_fold_patch.root,
        &folded_manifest,
        minor_manifest.manifest_id,
        17,
        &minor_fold_ref,
        old_manifest.segments[1].max_sequence,
        EVOKE_TERM_COW_NEUTRAL_FOLD_MINOR,
        load_term_cow_composite_object,
        &prefix_source_store,
        &minor_fold_patch,
        &stats
    ));
    bind_term_cow_test_tree(
        &minor_fold_patch,
        minor_manifest.manifest_id,
        &next_block
    );
    ASSERT_STATUS_OK(evoke_term_cow_ref_as_segment_object_ref(
        &minor_fold_patch.root,
        &minor_manifest.term_directory
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(
        &minor_manifest
    ));
    minor_result_store.layer_count = 3;
    minor_result_store.layers[0].tree = &minor_fold_patch;
    minor_result_store.layers[1].tree = &prefix_fold_patch;
    minor_result_store.layers[2].tree = &ancestor;
    ASSERT_STATUS_OK(evoke_term_cow_validate_external(
        &minor_fold_patch.root,
        minor_manifest.vocab_size,
        load_term_cow_layered_object,
        &minor_result_store
    ));
    ASSERT_STATUS_OK(evoke_term_cow_lookup_external(
        &minor_fold_patch.root,
        minor_manifest.vocab_size,
        17,
        load_term_cow_layered_object,
        &minor_result_store,
        &minor_record
    ));
    ASSERT_TRUE(minor_record.extent_count == 0);
    ASSERT_TRUE(
        (minor_record.flags &
         EVOKE_TERM_COW_RECORD_FLAG_NEUTRAL_MINOR_FOLD) != 0
    );
    ASSERT_TRUE(
        minor_record.neutral_fold_coverage ==
        old_manifest.segments[0].max_sequence
    );
    ASSERT_TRUE(
        minor_record.neutral_minor_fold_coverage ==
        old_manifest.segments[1].max_sequence
    );
    ASSERT_TRUE(memcmp(
        &minor_record.neutral_fold,
        &prefix_fold_ref,
        sizeof(prefix_fold_ref)
    ) == 0);
    ASSERT_TRUE(memcmp(
        &minor_record.neutral_minor_fold,
        &minor_fold_ref,
        sizeof(minor_fold_ref)
    ) == 0);

    /* Later compaction may absorb an inherited fold boundary into a segment. */
    minor_manifest.segments[1].max_sequence++;
    minor_manifest.max_sequence++;
    ASSERT_TRUE(
        minor_manifest.segments[0].max_sequence !=
            minor_record.neutral_minor_fold_coverage &&
        minor_manifest.segments[1].max_sequence !=
            minor_record.neutral_minor_fold_coverage
    );
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(&minor_manifest));
    free(fold_states);
    fold_states = NULL;
    free(materialized_frequencies);
    materialized_frequencies = NULL;
    evoke_term_directory_free(&materialized_directory);
    evoke_term_directory_init(&materialized_directory);
    ASSERT_STATUS_OK(evoke_term_cow_materialize_external(
        &minor_fold_patch.root,
        &minor_manifest,
        load_term_cow_layered_object,
        &minor_result_store,
        &materialized_directory,
        &materialized_frequencies,
        NULL,
        &fold_states
    ));
    ASSERT_TRUE(
        materialized_directory.term_offsets[18] ==
        materialized_directory.term_offsets[17]
    );
    ASSERT_TRUE(
        fold_states[17].neutral_coverage ==
        old_manifest.segments[0].max_sequence
    );
    ASSERT_TRUE(
        fold_states[17].neutral_minor_coverage ==
        old_manifest.segments[1].max_sequence
    );
    ASSERT_TRUE(memcmp(
        &fold_states[17].neutral_minor_ref,
        &minor_fold_ref,
        sizeof(minor_fold_ref)
    ) == 0);

    clone_segment_manifest_for_test(
        &minor_manifest,
        &promoted_manifest
    );
    promoted_manifest.manifest_id = minor_manifest.manifest_id + 1;
    promoted_manifest.parent_manifest_id = minor_manifest.manifest_id;
    initialize_test_object_ref(
        &promoted_fold_ref,
        EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
        830,
        1,
        promoted_manifest.manifest_id,
        831
    );
    ASSERT_STATUS_OK(evoke_term_cow_build_external_neutral_fold_patch(
        &minor_fold_patch.root,
        &minor_manifest,
        promoted_manifest.manifest_id,
        17,
        &promoted_fold_ref,
        minor_record.neutral_minor_fold_coverage,
        EVOKE_TERM_COW_NEUTRAL_FOLD_MAJOR,
        load_term_cow_layered_object,
        &minor_result_store,
        &promoted_fold_patch,
        &stats
    ));
    bind_term_cow_test_tree(
        &promoted_fold_patch,
        promoted_manifest.manifest_id,
        &next_block
    );
    ASSERT_STATUS_OK(evoke_term_cow_ref_as_segment_object_ref(
        &promoted_fold_patch.root,
        &promoted_manifest.term_directory
    ));
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(
        &promoted_manifest
    ));
    promoted_result_store.layer_count = 4;
    promoted_result_store.layers[0].tree = &promoted_fold_patch;
    promoted_result_store.layers[1].tree = &minor_fold_patch;
    promoted_result_store.layers[2].tree = &prefix_fold_patch;
    promoted_result_store.layers[3].tree = &ancestor;
    ASSERT_STATUS_OK(evoke_term_cow_validate_external(
        &promoted_fold_patch.root,
        promoted_manifest.vocab_size,
        load_term_cow_layered_object,
        &promoted_result_store
    ));
    ASSERT_STATUS_OK(evoke_term_cow_lookup_external(
        &promoted_fold_patch.root,
        promoted_manifest.vocab_size,
        17,
        load_term_cow_layered_object,
        &promoted_result_store,
        &promoted_record
    ));
    ASSERT_TRUE(promoted_record.extent_count == 0);
    ASSERT_TRUE(
        (promoted_record.flags &
         EVOKE_TERM_COW_RECORD_FLAG_NEUTRAL_MINOR_FOLD) == 0
    );
    ASSERT_TRUE(
        promoted_record.neutral_fold_coverage ==
        minor_record.neutral_minor_fold_coverage
    );
    ASSERT_TRUE(memcmp(
        &promoted_record.neutral_fold,
        &promoted_fold_ref,
        sizeof(promoted_fold_ref)
    ) == 0);
    ASSERT_TRUE(
        promoted_record.neutral_minor_fold_coverage == 0
    );

    fold_safe = true;
    conflict_term_id = UINT32_MAX;
    ASSERT_STATUS_OK(
        evoke_term_cow_replace_fold_preflight_external(
            &prefix_fold_patch.root,
            &folded_manifest,
            0,
            replaced_payloads,
            2,
            load_term_cow_composite_object,
            &prefix_source_store,
            &fold_safe,
            &conflict_term_id
        )
    );
    ASSERT_TRUE(!fold_safe);
    ASSERT_TRUE(conflict_term_id == 17);

    clone_segment_manifest_for_test(
        &folded_manifest,
        &straddling_manifest
    );
    straddling_manifest.segments[0].max_sequence =
        straddling_manifest.segments[1].min_sequence;
    straddling_manifest.segments[1].min_sequence++;
    straddling_manifest.segments[1].max_sequence++;
    straddling_manifest.max_sequence++;
    ASSERT_STATUS_OK(evoke_segment_manifest_validate(
        &straddling_manifest
    ));
    fold_safe = true;
    conflict_term_id = UINT32_MAX;
    ASSERT_STATUS_OK(
        evoke_term_cow_replace_fold_preflight_external(
            &prefix_fold_patch.root,
            &straddling_manifest,
            0,
            replaced_payloads,
            2,
            load_term_cow_composite_object,
            &prefix_source_store,
            &fold_safe,
            &conflict_term_id
        )
    );
    ASSERT_TRUE(!fold_safe);
    ASSERT_TRUE(conflict_term_id == 17);

    clone_segment_manifest_for_test(
        &folded_manifest,
        &coverage_only_manifest
    );
    coverage_only_manifest.segments[0].max_sequence =
        coverage_only_manifest.segments[1].min_sequence;
    coverage_only_manifest.segments[1].min_sequence++;
    coverage_only_manifest.segments[1].max_sequence++;
    coverage_only_manifest.max_sequence++;
    coverage_only_manifest.segments[1].posting_count = 2;
    coverage_only_payloads[0] = replaced_payloads[0];
    coverage_only_payloads[1] = replaced_payloads[1];
    coverage_only_payloads[1].posting_count = 2;
    coverage_only_payloads[1].runs = coverage_only_tail_runs;
    coverage_only_payloads[1].run_count = 2;
    coverage_only_payloads[1].values = coverage_only_tail_values;
    coverage_only_payloads[1].indices = coverage_only_tail_indices;
    fold_safe = true;
    conflict_term_id = UINT32_MAX;
    ASSERT_STATUS_OK(
        evoke_term_cow_replace_fold_preflight_external(
            &prefix_fold_patch.root,
            &coverage_only_manifest,
            0,
            coverage_only_payloads,
            2,
            load_term_cow_composite_object,
            &prefix_source_store,
            &fold_safe,
            &conflict_term_id
        )
    );
    ASSERT_TRUE(fold_safe);
    ASSERT_TRUE(conflict_term_id == UINT32_MAX);
    ASSERT_TRUE(evoke_term_cow_build_external_replace_patch(
        &prefix_fold_patch.root,
        &folded_manifest,
        &folded_next_manifest,
        0,
        replaced_payloads,
        2,
        &folded_replacement_payload,
        load_term_cow_composite_object,
        &prefix_source_store,
        &rejected_patch,
        &stats
    ) == EVOKE_ERR_FORMAT);

    free(fold_states);
    free(materialized_frequencies);
    evoke_term_cow_tree_free(&rejected_patch);
    evoke_term_cow_tree_free(&folded_replace_patch);
    evoke_term_cow_tree_free(&promoted_fold_patch);
    evoke_term_cow_tree_free(&minor_fold_patch);
    evoke_term_cow_tree_free(&prefix_fold_patch);
    evoke_term_cow_tree_free(&fold_patch);
    evoke_term_cow_tree_free(&patch);
    evoke_term_cow_tree_free(&ancestor);
    evoke_term_directory_free(&materialized_directory);
    evoke_term_directory_free(&expected_directory);
    evoke_term_directory_free(&old_directory);
    evoke_term_directory_free(&first_directory);
    evoke_segment_manifest_free(&folded_next_manifest);
    evoke_segment_manifest_free(&folded_manifest);
    evoke_segment_manifest_free(&coverage_only_manifest);
    evoke_segment_manifest_free(&straddling_manifest);
    evoke_segment_manifest_free(&promoted_manifest);
    evoke_segment_manifest_free(&minor_manifest);
    evoke_segment_manifest_free(&next_manifest);
    evoke_segment_manifest_free(&old_manifest);
    evoke_segment_manifest_free(&first_manifest);
}

static void
test_term_cow_tree_incremental_append(void)
{
    evoke_segment_manifest old_manifest;
    evoke_segment_manifest next_manifest;
    evoke_term_directory old_directory;
    evoke_term_directory expected_directory;
    evoke_term_directory materialized_directory;
    evoke_term_directory external_directory;
    evoke_term_cow_tree tree;
    evoke_term_cow_update_stats stats;
    evoke_term_cow_ref old_root;
    evoke_term_cow_record old_record;
    evoke_term_cow_record next_record;
    evoke_segment_term_run new_runs[] = {
        {
            .term_id = 17,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 0,
            .posting_count = 1
        },
        {
            .term_id = 50,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 1,
            .posting_count = 1
        },
        {
            .term_id = 70,
            .kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
            .posting_offset = 2,
            .posting_count = 1
        }
    };
    uint32_t new_indices[] = {0, 0, 0};
    evoke_posting_value new_values[] = {
        {.term_frequency = 1},
        {.term_frequency = 1},
        {.term_frequency = 1}
    };
    evoke_segment_payload_view new_payload = {
        .segment_id = 101,
        .posting_count = 3,
        .runs = new_runs,
        .run_count = 3,
        .values = new_values,
        .indices = new_indices,
        .document_id_base = 2,
        .local_document_count = 1
    };
    uint32_t *materialized_frequencies = NULL;
    uint32_t *external_frequencies = NULL;
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    size_t size = 0;
    size_t old_object_count;
    uint32_t next_block = 200;
    evoke_term_cow_object restored;
    const evoke_term_cow_ref *leaf_ref = NULL;

    initialize_term_cow_test_manifests(
        &old_manifest,
        &next_manifest
    );
    evoke_term_directory_init(&old_directory);
    evoke_term_directory_init(&expected_directory);
    evoke_term_directory_init(&materialized_directory);
    evoke_term_directory_init(&external_directory);
    evoke_term_cow_tree_init(&tree);
    old_directory.vocab_size = old_manifest.vocab_size;
    old_directory.extent_count = 3;
    old_directory.term_offsets = calloc(
        (size_t) old_directory.vocab_size + 1,
        sizeof(*old_directory.term_offsets)
    );
    old_directory.extents = calloc(
        old_directory.extent_count,
        sizeof(*old_directory.extents)
    );
    ASSERT_TRUE(old_directory.term_offsets != NULL);
    ASSERT_TRUE(old_directory.extents != NULL);
    {
        uint32_t extent_cursor = 0;

        for (uint32_t term_id = 0;
             term_id < old_directory.vocab_size;
             term_id++)
        {
            old_directory.term_offsets[term_id] = extent_cursor;
            if (term_id == 1 || term_id == 17 || term_id == 33)
            {
                old_directory.extents[extent_cursor].segment_index = 0;
                old_directory.extents[extent_cursor].kind =
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
                old_directory.extents[extent_cursor].posting_offset =
                    extent_cursor;
                old_directory.extents[extent_cursor].posting_count = 1;
                extent_cursor++;
            }
        }
        old_directory.term_offsets[old_directory.vocab_size] =
            extent_cursor;
        ASSERT_TRUE(extent_cursor == old_directory.extent_count);
    }
    ASSERT_STATUS_OK(evoke_term_directory_validate(
        &old_directory,
        &old_manifest
    ));
    ASSERT_STATUS_OK(evoke_term_cow_tree_build(
        &old_directory,
        &old_manifest,
        NULL,
        &tree
    ));
    ASSERT_STATUS_OK(evoke_term_cow_tree_validate(&tree));
    old_root = tree.root;
    old_object_count = tree.object_count;
    ASSERT_STATUS_OK(evoke_term_cow_tree_lookup(
        &tree,
        17,
        &old_record
    ));
    ASSERT_TRUE(old_record.raw_document_frequency == 1);
    ASSERT_TRUE(old_record.extent_count == 1);
    ASSERT_TRUE(old_record.extents[0].segment_id == 100);
    ASSERT_STATUS_OK(evoke_term_cow_tree_lookup(
        &tree,
        50,
        &old_record
    ));
    ASSERT_TRUE(old_record.raw_document_frequency == 0);
    ASSERT_TRUE(old_record.extent_count == 0);

    ASSERT_STATUS_OK(evoke_term_cow_tree_append_payload(
        &tree,
        &old_manifest,
        &next_manifest,
        &new_payload,
        &stats
    ));
    ASSERT_TRUE(stats.changed_terms == 3);
    ASSERT_TRUE(stats.changed_leaves == 3);
    ASSERT_TRUE(stats.written_leaves == 3);
    ASSERT_TRUE(
        stats.written_nodes ==
        3 * EVOKE_TERM_COW_RADIX_LEVELS
    );
    ASSERT_TRUE(tree.object_count == old_object_count + 18);
    ASSERT_TRUE(stats.written_bytes > 0);
    ASSERT_TRUE(tree.root.object_id != old_root.object_id);
    ASSERT_STATUS_OK(evoke_term_cow_tree_validate(&tree));

    ASSERT_STATUS_OK(evoke_term_cow_tree_lookup_at(
        &tree,
        &old_root,
        old_manifest.vocab_size,
        17,
        &old_record
    ));
    ASSERT_TRUE(old_record.raw_document_frequency == 1);
    ASSERT_TRUE(old_record.extent_count == 1);
    ASSERT_STATUS_OK(evoke_term_cow_tree_lookup_at(
        &tree,
        &old_root,
        old_manifest.vocab_size,
        50,
        &old_record
    ));
    ASSERT_TRUE(old_record.raw_document_frequency == 0);
    ASSERT_TRUE(old_record.extent_count == 0);

    ASSERT_STATUS_OK(evoke_term_cow_tree_lookup(
        &tree,
        17,
        &next_record
    ));
    ASSERT_TRUE(next_record.raw_document_frequency == 2);
    ASSERT_TRUE(next_record.extent_count == 2);
    ASSERT_TRUE(next_record.extents[0].segment_id == 100);
    ASSERT_TRUE(next_record.extents[1].segment_id == 101);
    ASSERT_STATUS_OK(evoke_term_cow_tree_lookup(
        &tree,
        50,
        &next_record
    ));
    ASSERT_TRUE(next_record.raw_document_frequency == 1);
    ASSERT_TRUE(next_record.extent_count == 1);
    ASSERT_TRUE(next_record.extents[0].segment_id == 101);

    ASSERT_STATUS_OK(evoke_term_directory_append_payload(
        &old_directory,
        &old_manifest,
        &next_manifest,
        &new_payload,
        &expected_directory
    ));
    ASSERT_STATUS_OK(evoke_term_cow_tree_materialize_flat(
        &tree,
        &next_manifest,
        &materialized_directory,
        &materialized_frequencies
    ));
    ASSERT_TRUE(
        expected_directory.vocab_size ==
        materialized_directory.vocab_size
    );
    ASSERT_TRUE(
        expected_directory.extent_count ==
        materialized_directory.extent_count
    );
    assert_uint64_array(
        expected_directory.term_offsets,
        materialized_directory.term_offsets,
        (size_t) expected_directory.vocab_size + 1
    );
    ASSERT_TRUE(memcmp(
        expected_directory.extents,
        materialized_directory.extents,
        (size_t) expected_directory.extent_count *
            sizeof(*expected_directory.extents)
    ) == 0);
    assert_uint32_array(
        materialized_frequencies,
        next_manifest.doc_frequencies,
        next_manifest.vocab_size
    );

    ASSERT_STATUS_OK(evoke_term_cow_tree_object_serialize(
        &tree,
        &tree.root,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(evoke_term_cow_object_deserialize(
        bytes,
        size,
        &restored
    ));
    ASSERT_TRUE(restored.ref.object_id == tree.root.object_id);
    ASSERT_TRUE(
        restored.ref.kind == EVOKE_TERM_COW_OBJECT_NODE
    );
    free(bytes);
    bytes = NULL;
    for (size_t object_index = tree.object_count;
         object_index > 0;
         object_index--)
    {
        if (tree.objects[object_index - 1].ref.kind ==
            EVOKE_TERM_COW_OBJECT_LEAF)
        {
            leaf_ref = &tree.objects[object_index - 1].ref;
            break;
        }
    }
    ASSERT_TRUE(leaf_ref != NULL);
    ASSERT_STATUS_OK(evoke_term_cow_tree_object_serialize(
        &tree,
        leaf_ref,
        &bytes,
        &size
    ));
    ASSERT_STATUS_OK(evoke_term_cow_object_deserialize(
        bytes,
        size,
        &restored
    ));
    ASSERT_TRUE(
        restored.ref.kind == EVOKE_TERM_COW_OBJECT_LEAF
    );
    {
        const size_t header_size = 64;
        const size_t record_size = 248;
        const size_t extent_size = 32;
        uint32_t record_count = test_read_u32_le(bytes + 12);
        size_t fixed_size = header_size +
            (size_t) record_count * record_size;
        size_t stored_extent_count =
            (size - fixed_size) /
            ((size_t) record_count * extent_size);
        size_t copied_extent_count = stored_extent_count <
            EVOKE_TERM_DIRECTORY_LEGACY_MAX_EXTENTS_PER_TERM
            ? stored_extent_count
            : EVOKE_TERM_DIRECTORY_LEGACY_MAX_EXTENTS_PER_TERM;
        size_t legacy_size = fixed_size +
            (size_t) record_count *
                EVOKE_TERM_DIRECTORY_LEGACY_MAX_EXTENTS_PER_TERM *
                extent_size;
        uint8_t *legacy = calloc(1, legacy_size);

        ASSERT_TRUE(legacy != NULL);
        memcpy(legacy, bytes, fixed_size);
        for (uint32_t record_index = 0;
             record_index < record_count;
             record_index++)
        {
            memcpy(
                legacy + fixed_size +
                    (size_t) record_index *
                        EVOKE_TERM_DIRECTORY_LEGACY_MAX_EXTENTS_PER_TERM *
                        extent_size,
                bytes + fixed_size +
                    (size_t) record_index *
                        stored_extent_count *
                        extent_size,
                copied_extent_count * extent_size
            );
        }
        test_write_u64_le(legacy + 32, legacy_size);
        test_write_u64_le(legacy + 56, 0);
        test_write_u64_le(
            legacy + 56,
            test_term_cow_checksum(legacy, legacy_size)
        );
        ASSERT_STATUS_OK(evoke_term_cow_object_deserialize(
            legacy,
            legacy_size,
            &restored
        ));
        ASSERT_TRUE(
            restored.ref.kind == EVOKE_TERM_COW_OBJECT_LEAF
        );
        free(legacy);
    }
    mutated = malloc(size);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, size);
    test_write_u16_le(mutated + 4, 7);
    test_write_u64_le(mutated + 56, 0);
    test_write_u64_le(
        mutated + 56,
        test_term_cow_checksum(mutated, size)
    );
    ASSERT_TRUE(evoke_term_cow_object_deserialize(
        mutated,
        size,
        &restored
    ) == EVOKE_ERR_FORMAT);
    memcpy(mutated, bytes, size);
    mutated[size - 1] ^= 0x1U;
    ASSERT_TRUE(evoke_term_cow_object_deserialize(
        mutated,
        size,
        &restored
    ) == EVOKE_ERR_FORMAT);

    free(bytes);
    bytes = NULL;
    free(mutated);
    mutated = NULL;

    for (uint64_t object_id = 1;
         object_id <= tree.object_count;
         object_id++)
    {
        evoke_segment_object_ref storage_ref;
        uint32_t page_count = 0;

        ASSERT_STATUS_OK(
            evoke_term_cow_tree_prepare_object_for_storage(
                &tree,
                object_id,
                &bytes,
                &size
            )
        );
        ASSERT_STATUS_OK(evoke_segment_page_count_required(
            size,
            8192,
            &page_count
        ));
        memset(&storage_ref, 0, sizeof(storage_ref));
        storage_ref.object_kind =
            EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY;
        storage_ref.start_block = next_block;
        storage_ref.page_count = page_count;
        storage_ref.object_id = object_id;
        storage_ref.owner_manifest_id =
            next_manifest.manifest_id;
        storage_ref.object_bytes = size;
        storage_ref.object_checksum =
            evoke_segment_blob_checksum(bytes, size);
        ASSERT_STATUS_OK(evoke_term_cow_tree_bind_object_storage(
            &tree,
            object_id,
            &storage_ref
        ));
        next_block += page_count;
        free(bytes);
        bytes = NULL;
    }
    ASSERT_STATUS_OK(evoke_term_cow_tree_validate(&tree));
    ASSERT_STATUS_OK(evoke_term_cow_tree_lookup(
        &tree,
        17,
        &next_record
    ));
    ASSERT_TRUE(next_record.raw_document_frequency == 2);
    {
        term_cow_fake_store store = {
            .tree = &tree
        };
        uint32_t reachable_object_count;

        ASSERT_STATUS_OK(evoke_term_cow_lookup_external(
            &tree.root,
            tree.vocab_size,
            17,
            load_term_cow_fake_object,
            &store,
            &next_record
        ));
        ASSERT_TRUE(
            store.load_count ==
            EVOKE_TERM_COW_RADIX_LEVELS + 1
        );
        ASSERT_TRUE(next_record.raw_document_frequency == 2);
        store.load_count = 0;
        ASSERT_STATUS_OK(evoke_term_cow_validate_external(
            &tree.root,
            tree.vocab_size,
            load_term_cow_fake_object,
            &store
        ));
        reachable_object_count = store.load_count;
        ASSERT_TRUE(reachable_object_count > 0);
        ASSERT_TRUE(reachable_object_count <= tree.object_count);
        store.load_count = 0;
        ASSERT_STATUS_OK(evoke_term_cow_materialize_external(
            &tree.root,
            &next_manifest,
            load_term_cow_fake_object,
            &store,
            &external_directory,
            &external_frequencies,
            NULL,
            NULL
        ));
        ASSERT_TRUE(store.load_count == reachable_object_count);
        ASSERT_TRUE(
            external_directory.extent_count ==
            expected_directory.extent_count
        );
        assert_uint64_array(
            external_directory.term_offsets,
            expected_directory.term_offsets,
            (size_t) expected_directory.vocab_size + 1
        );
        ASSERT_TRUE(memcmp(
            external_directory.extents,
            expected_directory.extents,
            (size_t) expected_directory.extent_count *
                sizeof(*expected_directory.extents)
        ) == 0);
        assert_uint32_array(
            external_frequencies,
            next_manifest.doc_frequencies,
            next_manifest.vocab_size
        );
        store.load_count = 0;
        store.corrupt_object_id = tree.root.object_id;
        ASSERT_TRUE(evoke_term_cow_lookup_external(
            &tree.root,
            tree.vocab_size,
            17,
            load_term_cow_fake_object,
            &store,
            &next_record
        ) == EVOKE_ERR_FORMAT);
        ASSERT_TRUE(store.load_count == 1);
    }
    {
        evoke_segment_object_ref storage_ref;

        ASSERT_STATUS_OK(evoke_term_cow_ref_as_segment_object_ref(
            &tree.root,
            &storage_ref
        ));
        ASSERT_STATUS_OK(evoke_term_cow_tree_object_serialize(
            &tree,
            &tree.root,
            &bytes,
            &size
        ));
        ASSERT_TRUE(
            storage_ref.object_checksum ==
            evoke_segment_blob_checksum(bytes, size)
        );
        ASSERT_STATUS_OK(evoke_term_cow_object_deserialize(
            bytes,
            size,
            &restored
        ));
        ASSERT_STATUS_OK(evoke_term_cow_object_bind_storage(
            &restored,
            &storage_ref
        ));
        ASSERT_TRUE(
            restored.ref.start_block ==
            tree.root.start_block
        );
        free(bytes);
        bytes = NULL;
    }

    free(materialized_frequencies);
    free(external_frequencies);
    evoke_term_cow_tree_free(&tree);
    evoke_term_directory_free(&external_directory);
    evoke_term_directory_free(&materialized_directory);
    evoke_term_directory_free(&expected_directory);
    evoke_term_directory_free(&old_directory);
    evoke_segment_manifest_free(&next_manifest);
    evoke_segment_manifest_free(&old_manifest);
}

typedef struct document_cow_fake_store
{
    const evoke_document_cow_tree *tree;
    uint64_t corrupt_object_id;
    uint32_t load_count;
} document_cow_fake_store;

typedef struct document_cow_composite_store
{
    document_cow_fake_store ancestor;
    document_cow_fake_store patch;
} document_cow_composite_store;

static evoke_status
count_document_cow_object(
    void *context,
    const evoke_document_cow_object *object
)
{
    cow_visit_counts *counts = context;

    if (counts == NULL || object == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    counts->object_count++;
    if (object->ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        counts->node_count++;
    }
    else if (object->ref.kind == EVOKE_DOCUMENT_COW_OBJECT_LEAF)
    {
        counts->leaf_count++;
    }
    else
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

static evoke_status
load_document_cow_fake_object(
    void *context,
    const evoke_document_cow_ref *ref,
    evoke_document_cow_object *object_out
)
{
    document_cow_fake_store *store = context;
    evoke_segment_object_ref storage_ref;
    uint8_t *bytes = NULL;
    size_t size = 0;
    evoke_status status;

    if (store == NULL || store->tree == NULL ||
        ref == NULL || object_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    store->load_count++;
    status = evoke_document_cow_tree_object_serialize(
        store->tree,
        ref,
        &bytes,
        &size
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (store->corrupt_object_id == ref->object_id)
    {
        bytes[size - 1] ^= UINT8_C(1);
    }
    status = evoke_document_cow_object_deserialize(
        bytes,
        size,
        object_out
    );
    free(bytes);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_document_cow_ref_as_segment_object_ref(
        ref,
        &storage_ref
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_document_cow_object_bind_storage(
        object_out,
        &storage_ref
    );
}

static evoke_status
load_document_cow_composite_object(
    void *context,
    const evoke_document_cow_ref *ref,
    evoke_document_cow_object *object_out
)
{
    document_cow_composite_store *store = context;

    if (store == NULL || ref == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (ref->owner_manifest_id ==
        store->patch.tree->root.owner_manifest_id)
    {
        return load_document_cow_fake_object(
            &store->patch,
            ref,
            object_out
        );
    }
    return load_document_cow_fake_object(
        &store->ancestor,
        ref,
        object_out
    );
}

static void
bind_document_cow_test_tree(
    evoke_document_cow_tree *tree,
    uint64_t owner_manifest_id,
    uint32_t *next_block
)
{
    for (uint64_t object_id = 1;
         object_id < tree->next_object_id;
         object_id++)
    {
        evoke_segment_object_ref storage_ref;
        uint8_t *bytes = NULL;
        size_t size = 0;
        uint32_t page_count = 0;

        ASSERT_STATUS_OK(
            evoke_document_cow_tree_prepare_object_for_storage(
                tree,
                object_id,
                &bytes,
                &size
            )
        );
        ASSERT_STATUS_OK(evoke_segment_page_count_required(
            size,
            8192,
            &page_count
        ));
        memset(&storage_ref, 0, sizeof(storage_ref));
        storage_ref.object_kind =
            EVOKE_SEGMENT_OBJECT_DOCUMENT_DIRECTORY;
        storage_ref.start_block = *next_block;
        storage_ref.page_count = page_count;
        storage_ref.object_id = object_id;
        storage_ref.owner_manifest_id = owner_manifest_id;
        storage_ref.object_bytes = size;
        storage_ref.object_checksum =
            evoke_segment_blob_checksum(bytes, size);
        ASSERT_STATUS_OK(
            evoke_document_cow_tree_bind_object_storage(
                tree,
                object_id,
                &storage_ref
            )
        );
        *next_block += page_count;
        free(bytes);
    }
}

static void
initialize_document_cow_record(
    evoke_document_cow_record *record,
    uint64_t document_slot,
    bool semantic_pending
)
{
    memset(record, 0, sizeof(*record));
    record->version.document_slot = document_slot;
    record->version.born_sequence = document_slot + 1;
    record->version.heap_block = (uint32_t) document_slot + 10;
    record->version.heap_offset = 1;
    record->version.document_length = (uint32_t) document_slot + 3;
    record->version.flags =
        EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;
    if (semantic_pending)
    {
        record->version.flags |=
            EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING;
        memset(
            record->version.semantic_input_fingerprint,
            (int) document_slot + 1,
            EVOKE_DOCUMENT_FINGERPRINT_BYTES
        );
    }
}

static bool
document_cow_even_slot_predicate(
    void *context,
    const evoke_document_cow_record *record
)
{
    (void) context;
    return (record->version.document_slot & UINT64_C(1)) == 0;
}

static void
initialize_complete_document_cow_record(
    evoke_document_cow_record *record,
    uint64_t document_slot
)
{
    initialize_document_cow_record(record, document_slot, false);
    record->version.flags |=
        EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE;
    memset(
        record->version.semantic_input_fingerprint,
        (int) document_slot + 1,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );
}

static void
complete_document_cow_record(
    evoke_document_cow_record *record,
    uint64_t transition_sequence
)
{
    memset(
        &record->semantic_state,
        0,
        sizeof(record->semantic_state)
    );
    record->semantic_state.document_slot =
        record->version.document_slot;
    record->semantic_state.transition_sequence =
        transition_sequence;
    record->semantic_state.flags =
        EVOKE_SEMANTIC_STATE_FLAG_COMPLETE |
        EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID;
    memcpy(
        record->semantic_state.semantic_input_fingerprint,
        record->version.semantic_input_fingerprint,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );
}

static void
quarantine_document_cow_record(
    evoke_document_cow_record *record,
    uint64_t transition_sequence,
    int64_t retry_after
)
{
    memset(
        &record->semantic_state,
        0,
        sizeof(record->semantic_state)
    );
    record->semantic_state.document_slot =
        record->version.document_slot;
    record->semantic_state.transition_sequence =
        transition_sequence;
    record->semantic_state.error_code = 7;
    record->semantic_state.retry_after = retry_after;
    record->semantic_state.pending_since = 1;
    record->semantic_state.error_hash = UINT64_C(0x123456789abcdef0);
    record->semantic_state.failure_count = 1;
    record->semantic_state.flags =
        EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED |
        EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID;
    memcpy(
        record->semantic_state.semantic_input_fingerprint,
        record->version.semantic_input_fingerprint,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );
}

static void
test_document_cow_record_identity_equality(void)
{
    evoke_document_cow_record source;
    evoke_document_cow_record candidate;

    initialize_document_cow_record(&source, 7, true);
    source.lexical_residency = 3;
    source.semantic_residency = 2;
    source.event_residency = 1;
    candidate = source;
    ASSERT_TRUE(evoke_document_cow_records_equal(&source, &candidate));

    candidate.version.born_sequence++;
    ASSERT_TRUE(!evoke_document_cow_records_equal(&source, &candidate));
    candidate = source;
    candidate.retirement.retirement_sequence = 99;
    ASSERT_TRUE(!evoke_document_cow_records_equal(&source, &candidate));
    candidate = source;
    candidate.semantic_state.transition_sequence = 100;
    ASSERT_TRUE(!evoke_document_cow_records_equal(&source, &candidate));
    candidate = source;
    candidate.event_residency++;
    ASSERT_TRUE(!evoke_document_cow_records_equal(&source, &candidate));
    ASSERT_TRUE(!evoke_document_cow_records_equal(NULL, &candidate));
}

static void
test_document_cow_tree_duplicate_object_guard(void)
{
    evoke_document_cow_record records[40];
    evoke_document_cow_record found;
    evoke_document_cow_tree tree;
    evoke_document_cow_object saved_object;
    uint64_t saved_object_id;

    evoke_document_cow_tree_init(&tree);
    for (uint64_t slot = 0; slot < 40; slot++)
    {
        initialize_document_cow_record(&records[slot], slot, false);
    }
    ASSERT_STATUS_OK(evoke_document_cow_tree_build(
        records,
        40,
        1,
        &tree
    ));
    ASSERT_TRUE(tree.object_count > 3);

    saved_object = tree.objects[0];
    tree.objects[0] = tree.objects[tree.object_count - 1];
    tree.objects[tree.object_count - 1] = saved_object;
    ASSERT_STATUS_OK(evoke_document_cow_tree_validate(&tree));
    ASSERT_STATUS_OK(evoke_document_cow_tree_lookup(&tree, 0, &found));
    ASSERT_TRUE(found.version.document_slot == 0);

    saved_object_id = tree.objects[2].ref.object_id;
    tree.objects[2].ref.object_id = tree.objects[0].ref.object_id;
    ASSERT_TRUE(
        evoke_document_cow_tree_validate(&tree) == EVOKE_ERR_FORMAT
    );
    tree.objects[2].ref.object_id = saved_object_id;
    ASSERT_STATUS_OK(evoke_document_cow_tree_validate(&tree));

    evoke_document_cow_tree_free(&tree);
}

static void
test_document_cow_pending_frontier_and_patch(void)
{
    evoke_document_cow_record records[40];
    evoke_document_cow_record updates[6];
    evoke_document_cow_record found;
    evoke_document_cow_record old_record;
    evoke_document_cow_record next_record;
    evoke_document_cow_record range_records[32];
    evoke_document_cow_record born_prefix[5];
    evoke_document_cow_length_extrema length_extrema;
    evoke_document_cow_born_prefix_stats born_stats;
    evoke_document_cow_length_extrema *block_extrema = NULL;
    size_t block_extrema_count = 0;
    size_t born_prefix_count = 0;
    evoke_document_cow_ref corrupt_root;
    evoke_document_cow_tree ancestor;
    evoke_document_cow_tree patch;
    evoke_document_cow_update_stats stats;
    document_cow_fake_store ancestor_store;
    document_cow_composite_store composite;
    uint8_t *bytes = NULL;
    size_t size = 0;
    uint32_t next_block = 10;
    bool actionable = false;
    bool reusable = false;
    cow_visit_counts visit_counts = {0};

    evoke_document_cow_tree_init(&ancestor);
    evoke_document_cow_tree_init(&patch);
    memset(&ancestor_store, 0, sizeof(ancestor_store));
    memset(&composite, 0, sizeof(composite));
    for (uint64_t slot = 0; slot < 40; slot++)
    {
        initialize_document_cow_record(
            &records[slot],
            slot,
            slot == 18 || slot == 33 || slot == 34
        );
    }
    records[7].version.record_xid = 0;
    records[7].version.heap_block = 0;
    records[7].version.heap_offset = 0;
    records[7].version.document_length = 0;
    records[7].version.flags =
        EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |
        EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE;
    records[7].event_residency = 1;
    quarantine_document_cow_record(&records[18], 80, 100);
    quarantine_document_cow_record(&records[33], 81, 20);
    complete_document_cow_record(&records[34], 82);
    records[5].retirement.document_slot = 5;
    records[5].retirement.retirement_sequence = 90;
    records[5].retirement.document_length =
        records[5].version.document_length;
    records[5].retirement.flags =
        EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID;
    records[5].lexical_residency = 1;

    ASSERT_STATUS_OK(evoke_document_cow_tree_build(
        records,
        40,
        1,
        &ancestor
    ));
    ASSERT_TRUE(ancestor.root.live_document_count == 38);
    ASSERT_TRUE(ancestor.root.semantic_pending_count == 2);
    ASSERT_TRUE(ancestor.root.earliest_retry_after == 20);
    ASSERT_TRUE(ancestor.root.bounded_document_count == 39);
    ASSERT_TRUE(ancestor.root.min_document_length == 3);
    ASSERT_TRUE(ancestor.root.max_document_length == 42);
    ASSERT_TRUE(ancestor.root.reusable_document_count == 0);
    ASSERT_TRUE(
        ancestor.root.first_reusable_document_slot == UINT64_MAX
    );
    ASSERT_TRUE(ancestor.root.min_live_born_sequence == 1);
    ASSERT_STATUS_OK(evoke_document_cow_tree_length_extrema(
        &ancestor,
        4,
        12,
        &length_extrema
    ));
    ASSERT_TRUE(length_extrema.document_count == 11);
    ASSERT_TRUE(length_extrema.min_document_length == 7);
    ASSERT_TRUE(length_extrema.max_document_length == 18);
    bind_document_cow_test_tree(&ancestor, 1, &next_block);
    ancestor_store.tree = &ancestor;
    ancestor_store.load_count = 0;
    ASSERT_STATUS_OK(
        evoke_document_cow_collect_live_born_prefix_external(
            &ancestor.root,
            ancestor.document_slot_count,
            5,
            load_document_cow_fake_object,
            &ancestor_store,
            born_prefix,
            5,
            &born_prefix_count,
            &born_stats
        )
    );
    ASSERT_TRUE(born_prefix_count == 5);
    ASSERT_TRUE(born_stats.objects_loaded == ancestor_store.load_count);
    ASSERT_TRUE(born_stats.objects_loaded <=
        EVOKE_DOCUMENT_COW_RADIX_LEVELS + 2);
    ASSERT_TRUE(born_stats.records_examined <=
        EVOKE_DOCUMENT_COW_LEAF_RECORDS);
    ASSERT_TRUE(born_stats.heap_peak <=
        EVOKE_DOCUMENT_COW_RADIX_FANOUT *
            EVOKE_DOCUMENT_COW_RADIX_LEVELS +
        EVOKE_DOCUMENT_COW_LEAF_RECORDS);
    for (uint32_t index = 0; index < 5; index++)
    {
        ASSERT_TRUE(born_prefix[index].version.document_slot == index);
        ASSERT_TRUE(
            born_prefix[index].version.born_sequence == index + 1
        );
    }
    ancestor_store.load_count = 0;
    born_prefix_count = 0;
    ASSERT_STATUS_OK(
        evoke_document_cow_collect_live_born_prefix_matching_external(
            &ancestor.root,
            ancestor.document_slot_count,
            5,
            load_document_cow_fake_object,
            &ancestor_store,
            document_cow_even_slot_predicate,
            NULL,
            born_prefix,
            5,
            &born_prefix_count,
            &born_stats
        )
    );
    ASSERT_TRUE(born_prefix_count == 5);
    for (uint32_t index = 0; index < 5; index++)
    {
        ASSERT_TRUE(
            born_prefix[index].version.document_slot == index * 2U
        );
    }
    ASSERT_TRUE(born_stats.records_examined <=
        EVOKE_DOCUMENT_COW_LEAF_RECORDS);
    ASSERT_TRUE(born_stats.heap_peak <=
        EVOKE_DOCUMENT_COW_RADIX_FANOUT *
            EVOKE_DOCUMENT_COW_RADIX_LEVELS +
        EVOKE_DOCUMENT_COW_LEAF_RECORDS);
    ASSERT_STATUS_OK(evoke_document_cow_validate_external(
        &ancestor.root,
        ancestor.document_slot_count,
        load_document_cow_fake_object,
        &ancestor_store
    ));
    ancestor_store.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_read_range_external(
        &ancestor.root,
        ancestor.document_slot_count,
        4,
        32,
        load_document_cow_fake_object,
        &ancestor_store,
        range_records
    ));
    ASSERT_TRUE(
        ancestor_store.load_count <=
        EVOKE_DOCUMENT_COW_RADIX_LEVELS + 3
    );
    for (uint32_t index = 0; index < 32; index++)
    {
        ASSERT_TRUE(evoke_document_cow_records_equal(
            &range_records[index],
            &records[index + 4]
        ));
    }
    ASSERT_TRUE(evoke_document_cow_read_range_external(
        &ancestor.root,
        ancestor.document_slot_count,
        39,
        2,
        load_document_cow_fake_object,
        &ancestor_store,
        range_records
    ) == EVOKE_ERR_INVALID);
    ancestor_store.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_find_reusable_external(
        &ancestor.root,
        ancestor.document_slot_count,
        load_document_cow_fake_object,
        &ancestor_store,
        &found,
        &reusable
    ));
    ASSERT_TRUE(!reusable);
    ASSERT_TRUE(ancestor_store.load_count == 0);
    ancestor_store.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_length_extrema_external(
        &ancestor.root,
        ancestor.document_slot_count,
        0,
        ancestor.document_slot_count,
        load_document_cow_fake_object,
        &ancestor_store,
        &length_extrema
    ));
    ASSERT_TRUE(ancestor_store.load_count == 1);
    ASSERT_TRUE(length_extrema.document_count == 39);
    ASSERT_TRUE(length_extrema.min_document_length == 3);
    ASSERT_TRUE(length_extrema.max_document_length == 42);
    ancestor_store.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_length_extrema_external(
        &ancestor.root,
        ancestor.document_slot_count,
        4,
        12,
        load_document_cow_fake_object,
        &ancestor_store,
        &length_extrema
    ));
    ASSERT_TRUE(ancestor_store.load_count <=
        2 * (EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1));
    ASSERT_TRUE(length_extrema.document_count == 11);
    ASSERT_TRUE(length_extrema.min_document_length == 7);
    ASSERT_TRUE(length_extrema.max_document_length == 18);
    ancestor_store.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_block_extrema_external(
        &ancestor.root,
        ancestor.document_slot_count,
        4,
        load_document_cow_fake_object,
        &ancestor_store,
        &block_extrema,
        &block_extrema_count
    ));
    ASSERT_TRUE(block_extrema_count == 3);
    ASSERT_TRUE(block_extrema[0].document_count == 15);
    ASSERT_TRUE(block_extrema[0].min_document_length == 3);
    ASSERT_TRUE(block_extrema[0].max_document_length == 18);
    ASSERT_TRUE(block_extrema[1].document_count == 16);
    ASSERT_TRUE(block_extrema[1].min_document_length == 19);
    ASSERT_TRUE(block_extrema[1].max_document_length == 34);
    ASSERT_TRUE(block_extrema[2].document_count == 8);
    ASSERT_TRUE(block_extrema[2].min_document_length == 35);
    ASSERT_TRUE(block_extrema[2].max_document_length == 42);
    ASSERT_TRUE(ancestor_store.load_count <=
        2 * (EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1));
    evoke_document_cow_block_extrema_free(block_extrema);
    block_extrema = NULL;
    block_extrema_count = 0;
    corrupt_root = ancestor.root;
    corrupt_root.min_document_length++;
    ASSERT_TRUE(evoke_document_cow_length_extrema_external(
        &corrupt_root,
        ancestor.document_slot_count,
        0,
        ancestor.document_slot_count,
        load_document_cow_fake_object,
        &ancestor_store,
        &length_extrema
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_document_cow_block_extrema_external(
        &corrupt_root,
        ancestor.document_slot_count,
        4,
        load_document_cow_fake_object,
        &ancestor_store,
        &block_extrema,
        &block_extrema_count
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(block_extrema == NULL);
    ASSERT_TRUE(block_extrema_count == 0);
    ancestor_store.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_find_actionable_external(
        &ancestor.root,
        ancestor.document_slot_count,
        0,
        load_document_cow_fake_object,
        &ancestor_store,
        &found,
        &actionable
    ));
    ASSERT_TRUE(!actionable);
    ASSERT_TRUE(ancestor_store.load_count == 0);
    ASSERT_STATUS_OK(evoke_document_cow_find_actionable_external(
        &ancestor.root,
        ancestor.document_slot_count,
        20,
        load_document_cow_fake_object,
        &ancestor_store,
        &found,
        &actionable
    ));
    ASSERT_TRUE(actionable);
    ASSERT_TRUE(found.version.document_slot == 33);
    ASSERT_TRUE(ancestor_store.load_count <=
        EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1);
    ancestor_store.load_count = 0;
    actionable = false;
    ASSERT_STATUS_OK(
        evoke_document_cow_find_actionable_external_from(
            &ancestor.root,
            ancestor.document_slot_count,
            19,
            1000,
            load_document_cow_fake_object,
            &ancestor_store,
            &found,
            &actionable
        )
    );
    ASSERT_TRUE(actionable);
    ASSERT_TRUE(found.version.document_slot == 33);
    ASSERT_TRUE(ancestor_store.load_count <=
        2 * (EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1));
    ancestor_store.load_count = 0;
    actionable = true;
    ASSERT_STATUS_OK(
        evoke_document_cow_find_actionable_external_from(
            &ancestor.root,
            ancestor.document_slot_count,
            34,
            1000,
            load_document_cow_fake_object,
            &ancestor_store,
            &found,
            &actionable
        )
    );
    ASSERT_TRUE(!actionable);
    ASSERT_TRUE(ancestor_store.load_count <=
        2 * (EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1));

    updates[0] = records[18];
    quarantine_document_cow_record(&updates[0], 101, INT64_MAX);
    updates[1] = records[33];
    complete_document_cow_record(&updates[1], 102);
    for (uint64_t slot = 40; slot < 43; slot++)
    {
        initialize_document_cow_record(
            &updates[2 + slot - 40],
            slot,
            slot == 41
        );
    }
    updates[5] = records[10];
    updates[5].retirement.document_slot = 10;
    updates[5].retirement.retirement_sequence = 103;
    updates[5].retirement.document_length =
        updates[5].version.document_length;
    updates[5].retirement.flags =
        EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID;

    ASSERT_STATUS_OK(evoke_document_cow_build_external_patch(
        &ancestor.root,
        40,
        43,
        updates,
        6,
        2,
        load_document_cow_fake_object,
        &ancestor_store,
        &patch,
        &stats
    ));
    ASSERT_TRUE(stats.changed_records == 6);
    ASSERT_TRUE(stats.written_leaves == 3);
    ASSERT_TRUE(stats.written_nodes ==
        EVOKE_DOCUMENT_COW_RADIX_LEVELS);
    ASSERT_TRUE(patch.object_count ==
        stats.written_leaves + stats.written_nodes);
    assert_retired_ranges(
        patch.retired_ranges,
        patch.retired_range_count,
        next_block
    );
    bind_document_cow_test_tree(&patch, 2, &next_block);
    composite.ancestor.tree = &ancestor;
    composite.patch.tree = &patch;
    ASSERT_STATUS_OK(evoke_document_cow_validate_external(
        &patch.root,
        patch.document_slot_count,
        load_document_cow_composite_object,
        &composite
    ));
    composite.ancestor.load_count = 0;
    composite.patch.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_visit_external(
        &patch.root,
        patch.document_slot_count,
        load_document_cow_composite_object,
        &composite,
        count_document_cow_object,
        &visit_counts
    ));
    ASSERT_TRUE(
        visit_counts.object_count ==
        composite.ancestor.load_count + composite.patch.load_count
    );
    ASSERT_TRUE(visit_counts.node_count > 0);
    ASSERT_TRUE(visit_counts.leaf_count > 0);
    ASSERT_TRUE(composite.patch.load_count > 0);

    composite.ancestor.load_count = 0;
    composite.patch.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_lookup_external(
        &patch.root,
        43,
        33,
        load_document_cow_composite_object,
        &composite,
        &next_record
    ));
    ASSERT_TRUE(
        (next_record.semantic_state.flags &
         EVOKE_SEMANTIC_STATE_FLAG_COMPLETE) != 0
    );
    ASSERT_TRUE(
        composite.ancestor.load_count + composite.patch.load_count <=
        EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1
    );
    ancestor_store.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_lookup_external(
        &ancestor.root,
        40,
        33,
        load_document_cow_fake_object,
        &ancestor_store,
        &old_record
    ));
    ASSERT_TRUE(
        (old_record.semantic_state.flags &
         EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED) != 0
    );
    ASSERT_TRUE(
        ancestor_store.load_count <=
        EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1
    );
    ASSERT_TRUE(patch.root.document_slot_count == 43);
    ASSERT_TRUE(patch.root.live_document_count == 40);
    ASSERT_TRUE(patch.root.semantic_pending_count == 1);
    ASSERT_TRUE(patch.root.bounded_document_count == 42);
    ASSERT_TRUE(patch.root.min_document_length == 3);
    ASSERT_TRUE(patch.root.max_document_length == 45);
    ASSERT_TRUE(patch.root.reusable_document_count == 1);
    ASSERT_TRUE(patch.root.first_reusable_document_slot == 10);
    composite.ancestor.load_count = 0;
    composite.patch.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_find_reusable_external(
        &patch.root,
        patch.document_slot_count,
        load_document_cow_composite_object,
        &composite,
        &found,
        &reusable
    ));
    ASSERT_TRUE(reusable);
    ASSERT_TRUE(found.version.document_slot == 10);
    ASSERT_TRUE(
        composite.ancestor.load_count + composite.patch.load_count <=
        EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1
    );
    reusable = true;
    ASSERT_STATUS_OK(evoke_document_cow_find_reusable_external_from(
        &patch.root,
        patch.document_slot_count,
        11,
        load_document_cow_composite_object,
        &composite,
        &found,
        &reusable
    ));
    ASSERT_TRUE(!reusable);
    ASSERT_STATUS_OK(evoke_document_cow_length_extrema_external(
        &patch.root,
        patch.document_slot_count,
        39,
        4,
        load_document_cow_composite_object,
        &composite,
        &length_extrema
    ));
    ASSERT_TRUE(length_extrema.document_count == 4);
    ASSERT_TRUE(length_extrema.min_document_length == 42);
    ASSERT_TRUE(length_extrema.max_document_length == 45);
    ASSERT_STATUS_OK(evoke_document_cow_block_extrema_external(
        &patch.root,
        patch.document_slot_count,
        4,
        load_document_cow_composite_object,
        &composite,
        &block_extrema,
        &block_extrema_count
    ));
    ASSERT_TRUE(block_extrema_count == 3);
    ASSERT_TRUE(block_extrema[0].document_count == 15);
    ASSERT_TRUE(block_extrema[0].min_document_length == 3);
    ASSERT_TRUE(block_extrema[0].max_document_length == 18);
    ASSERT_TRUE(block_extrema[1].document_count == 16);
    ASSERT_TRUE(block_extrema[1].min_document_length == 19);
    ASSERT_TRUE(block_extrema[1].max_document_length == 34);
    ASSERT_TRUE(block_extrema[2].document_count == 11);
    ASSERT_TRUE(block_extrema[2].min_document_length == 35);
    ASSERT_TRUE(block_extrema[2].max_document_length == 45);
    evoke_document_cow_block_extrema_free(block_extrema);
    block_extrema = NULL;
    block_extrema_count = 0;

    composite.ancestor.load_count = 0;
    composite.patch.load_count = 0;
    actionable = false;
    ASSERT_STATUS_OK(evoke_document_cow_find_actionable_external(
        &patch.root,
        43,
        1000,
        load_document_cow_composite_object,
        &composite,
        &found,
        &actionable
    ));
    ASSERT_TRUE(actionable);
    ASSERT_TRUE(found.version.document_slot == 41);
    ASSERT_TRUE(
        composite.ancestor.load_count +
            composite.patch.load_count <=
        EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1
    );

    ASSERT_STATUS_OK(evoke_document_cow_tree_object_serialize(
        &patch,
        &patch.root,
        &bytes,
        &size
    ));
    bytes[size - 1] ^= UINT8_C(1);
    ASSERT_TRUE(evoke_document_cow_object_deserialize(
        bytes,
        size,
        &(evoke_document_cow_object) {0}
    ) == EVOKE_ERR_FORMAT);
    free(bytes);
    evoke_document_cow_tree_free(&patch);
    evoke_document_cow_tree_free(&ancestor);
}

static void
test_document_cow_partial_leaf_append_and_transition_guards(void)
{
    evoke_document_cow_record records[15];
    evoke_document_cow_record appends[3];
    evoke_document_cow_record invalid_update;
    evoke_document_cow_record found;
    evoke_document_cow_tree ancestor;
    evoke_document_cow_tree patch;
    evoke_document_cow_tree rejected;
    evoke_document_cow_update_stats stats;
    document_cow_fake_store ancestor_store;
    document_cow_composite_store composite;
    uint32_t next_block = 100;

    evoke_document_cow_tree_init(&ancestor);
    evoke_document_cow_tree_init(&patch);
    evoke_document_cow_tree_init(&rejected);
    memset(&ancestor_store, 0, sizeof(ancestor_store));
    memset(&composite, 0, sizeof(composite));
    for (uint64_t slot = 0; slot < 15; slot++)
    {
        initialize_document_cow_record(
            &records[slot],
            slot,
            slot == 0
        );
    }
    ASSERT_STATUS_OK(evoke_document_cow_tree_build(
        records,
        15,
        10,
        &ancestor
    ));
    bind_document_cow_test_tree(&ancestor, 10, &next_block);
    ancestor_store.tree = &ancestor;
    ASSERT_STATUS_OK(evoke_document_cow_validate_external(
        &ancestor.root,
        ancestor.document_slot_count,
        load_document_cow_fake_object,
        &ancestor_store
    ));

    invalid_update = records[0];
    invalid_update.version.semantic_input_fingerprint[0] ^= UINT8_C(1);
    ASSERT_TRUE(evoke_document_cow_build_external_patch(
        &ancestor.root,
        15,
        15,
        &invalid_update,
        1,
        11,
        load_document_cow_fake_object,
        &ancestor_store,
        &rejected,
        &stats
    ) == EVOKE_ERR_FORMAT);
    for (uint64_t slot = 15; slot < 18; slot++)
    {
        initialize_document_cow_record(
            &appends[slot - 15],
            slot,
            slot == 17
        );
    }
    ASSERT_TRUE(evoke_document_cow_build_external_patch(
        &ancestor.root,
        15,
        18,
        appends,
        3,
        10,
        load_document_cow_fake_object,
        &ancestor_store,
        &rejected,
        &stats
    ) == EVOKE_ERR_INVALID);
    ASSERT_STATUS_OK(evoke_document_cow_build_external_patch(
        &ancestor.root,
        15,
        18,
        appends,
        3,
        11,
        load_document_cow_fake_object,
        &ancestor_store,
        &patch,
        &stats
    ));
    ASSERT_TRUE(stats.changed_records == 3);
    ASSERT_TRUE(stats.written_leaves == 2);
    ASSERT_TRUE(stats.written_nodes ==
        EVOKE_DOCUMENT_COW_RADIX_LEVELS);
    ASSERT_TRUE(patch.object_count ==
        stats.written_leaves + stats.written_nodes);
    bind_document_cow_test_tree(&patch, 11, &next_block);
    composite.ancestor.tree = &ancestor;
    composite.patch.tree = &patch;
    ASSERT_STATUS_OK(evoke_document_cow_validate_external(
        &patch.root,
        patch.document_slot_count,
        load_document_cow_composite_object,
        &composite
    ));
    ASSERT_STATUS_OK(evoke_document_cow_lookup_external(
        &patch.root,
        patch.document_slot_count,
        17,
        load_document_cow_composite_object,
        &composite,
        &found
    ));
    ASSERT_TRUE(found.version.document_slot == 17);

    composite.patch.corrupt_object_id = patch.root.object_id;
    ASSERT_TRUE(evoke_document_cow_validate_external(
        &patch.root,
        patch.document_slot_count,
        load_document_cow_composite_object,
        &composite
    ) == EVOKE_ERR_FORMAT);

    evoke_document_cow_tree_free(&rejected);
    evoke_document_cow_tree_free(&patch);
    evoke_document_cow_tree_free(&ancestor);
}

static void
test_document_cow_initial_semantic_completion(void)
{
    evoke_document_cow_record records[3];
    evoke_document_cow_record found;
    evoke_document_cow_tree tree;

    evoke_document_cow_tree_init(&tree);
    for (uint64_t slot = 0; slot < 3; slot++)
    {
        initialize_complete_document_cow_record(
            &records[slot],
            slot
        );
    }
    records[0].version.heap_block = 0;

    ASSERT_STATUS_OK(evoke_document_cow_tree_build(
        records,
        3,
        1,
        &tree
    ));
    ASSERT_TRUE(tree.root.live_document_count == 3);
    ASSERT_TRUE(tree.root.semantic_pending_count == 0);
    ASSERT_STATUS_OK(evoke_document_cow_tree_lookup(
        &tree,
        0,
        &found
    ));
    ASSERT_TRUE(
        (found.version.flags &
         EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE) != 0
    );
    ASSERT_TRUE(found.semantic_state.transition_sequence == 0);

    evoke_document_cow_tree_free(&tree);
}

static void
test_document_cow_l0_owned_transitions(void)
{
    evoke_document_cow_record original;
    evoke_document_cow_record l0_retired;
    evoke_document_cow_record sealed;
    evoke_document_cow_record placeholder;
    evoke_document_cow_record placeholder_retired;
    evoke_document_cow_record live;
    evoke_document_cow_record found;
    evoke_document_cow_tree ancestor;
    evoke_document_cow_tree l0_patch;
    evoke_document_cow_tree sealed_patch;
    evoke_document_cow_tree placeholder_tree;
    evoke_document_cow_tree placeholder_retired_patch;
    evoke_document_cow_tree live_patch;
    evoke_document_cow_update_stats stats;
    document_cow_fake_store ancestor_store;
    document_cow_fake_store placeholder_store;
    document_cow_composite_store composite;
    uint32_t next_block = 150;

    evoke_document_cow_tree_init(&ancestor);
    evoke_document_cow_tree_init(&l0_patch);
    evoke_document_cow_tree_init(&sealed_patch);
    evoke_document_cow_tree_init(&placeholder_tree);
    evoke_document_cow_tree_init(&placeholder_retired_patch);
    evoke_document_cow_tree_init(&live_patch);
    memset(&ancestor_store, 0, sizeof(ancestor_store));
    memset(&placeholder_store, 0, sizeof(placeholder_store));
    memset(&composite, 0, sizeof(composite));

    initialize_document_cow_record(&original, 0, false);
    original.retirement.document_slot = 0;
    original.retirement.retirement_sequence = 10;
    original.retirement.document_length =
        original.version.document_length;
    original.retirement.flags =
        EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID;
    ASSERT_STATUS_OK(evoke_document_cow_tree_build(
        &original,
        1,
        20,
        &ancestor
    ));
    bind_document_cow_test_tree(&ancestor, 20, &next_block);
    ancestor_store.tree = &ancestor;

    initialize_document_cow_record(&l0_retired, 0, false);
    l0_retired.version.born_sequence = 11;
    l0_retired.version.heap_block = 99;
    l0_retired.version.flags |=
        EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED;
    l0_retired.event_residency = 0;
    l0_retired.retirement.document_slot = 0;
    l0_retired.retirement.retirement_sequence = 12;
    l0_retired.retirement.document_length =
        l0_retired.version.document_length;
    l0_retired.retirement.flags =
        EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID;
    ASSERT_STATUS_OK(evoke_document_cow_build_external_patch(
        &ancestor.root,
        1,
        1,
        &l0_retired,
        1,
        21,
        load_document_cow_fake_object,
        &ancestor_store,
        &l0_patch,
        &stats
    ));
    ASSERT_TRUE(l0_patch.root.live_document_count == 0);
    ASSERT_TRUE(l0_patch.root.reusable_document_count == 0);
    bind_document_cow_test_tree(&l0_patch, 21, &next_block);
    composite.ancestor.tree = &ancestor;
    composite.patch.tree = &l0_patch;

    sealed = l0_retired;
    sealed.version.flags &= ~EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED;
    sealed.event_residency = 1;
    ASSERT_STATUS_OK(evoke_document_cow_build_external_patch(
        &l0_patch.root,
        1,
        1,
        &sealed,
        1,
        22,
        load_document_cow_composite_object,
        &composite,
        &sealed_patch,
        &stats
    ));
    ASSERT_STATUS_OK(evoke_document_cow_tree_lookup(
        &sealed_patch,
        0,
        &found
    ));
    ASSERT_TRUE(!evoke_document_cow_record_is_l0_owned(&found));
    ASSERT_TRUE(found.retirement.retirement_sequence == 12);
    ASSERT_TRUE(found.event_residency == 1);

    memset(&placeholder, 0, sizeof(placeholder));
    placeholder.version.document_slot = 0;
    placeholder.version.born_sequence = 30;
    placeholder.version.flags =
        EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |
        EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE |
        EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED;
    ASSERT_STATUS_OK(evoke_document_cow_tree_build(
        &placeholder,
        1,
        30,
        &placeholder_tree
    ));
    ASSERT_TRUE(placeholder_tree.root.reusable_document_count == 0);
    bind_document_cow_test_tree(&placeholder_tree, 30, &next_block);
    placeholder_store.tree = &placeholder_tree;

    initialize_document_cow_record(&placeholder_retired, 0, false);
    placeholder_retired.version.born_sequence = 30;
    placeholder_retired.version.heap_block = 102;
    placeholder_retired.version.flags |=
        EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED;
    placeholder_retired.event_residency = 0;
    placeholder_retired.retirement.document_slot = 0;
    placeholder_retired.retirement.retirement_sequence = 31;
    placeholder_retired.retirement.document_length =
        placeholder_retired.version.document_length;
    placeholder_retired.retirement.flags =
        EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID;
    ASSERT_STATUS_OK(evoke_document_cow_build_external_patch(
        &placeholder_tree.root,
        1,
        1,
        &placeholder_retired,
        1,
        31,
        load_document_cow_fake_object,
        &placeholder_store,
        &placeholder_retired_patch,
        &stats
    ));
    ASSERT_TRUE(placeholder_retired_patch.root.live_document_count == 0);
    ASSERT_TRUE(
        placeholder_retired_patch.root.reusable_document_count == 0
    );

    initialize_document_cow_record(&live, 0, false);
    live.version.born_sequence = 30;
    live.version.heap_block = 101;
    live.event_residency = 1;
    ASSERT_STATUS_OK(evoke_document_cow_build_external_patch(
        &placeholder_tree.root,
        1,
        1,
        &live,
        1,
        31,
        load_document_cow_fake_object,
        &placeholder_store,
        &live_patch,
        &stats
    ));
    ASSERT_TRUE(live_patch.root.live_document_count == 1);
    ASSERT_TRUE(live_patch.root.reusable_document_count == 0);

    evoke_document_cow_tree_free(&live_patch);
    evoke_document_cow_tree_free(&placeholder_retired_patch);
    evoke_document_cow_tree_free(&placeholder_tree);
    evoke_document_cow_tree_free(&sealed_patch);
    evoke_document_cow_tree_free(&l0_patch);
    evoke_document_cow_tree_free(&ancestor);
}

static void
test_document_cow_l0_semantic_transition_during_seal(void)
{
    evoke_document_cow_record l0_pending;
    evoke_document_cow_record sealed_complete;
    evoke_document_cow_record sealed_quarantined;
    evoke_document_cow_record found;
    evoke_document_cow_tree ancestor;
    evoke_document_cow_tree complete_patch;
    evoke_document_cow_tree quarantine_patch;
    evoke_document_cow_update_stats stats;
    document_cow_fake_store ancestor_store;
    uint32_t next_block = 175;

    evoke_document_cow_tree_init(&ancestor);
    evoke_document_cow_tree_init(&complete_patch);
    evoke_document_cow_tree_init(&quarantine_patch);
    memset(&ancestor_store, 0, sizeof(ancestor_store));

    initialize_document_cow_record(&l0_pending, 0, true);
    l0_pending.version.flags |=
        EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED;
    ASSERT_STATUS_OK(evoke_document_cow_tree_build(
        &l0_pending,
        1,
        20,
        &ancestor
    ));
    bind_document_cow_test_tree(&ancestor, 20, &next_block);
    ancestor_store.tree = &ancestor;

    sealed_complete = l0_pending;
    sealed_complete.version.flags &=
        ~EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED;
    sealed_complete.semantic_residency = 1;
    sealed_complete.event_residency = 1;
    complete_document_cow_record(&sealed_complete, 2);
    ASSERT_STATUS_OK(evoke_document_cow_build_external_patch(
        &ancestor.root,
        1,
        1,
        &sealed_complete,
        1,
        21,
        load_document_cow_fake_object,
        &ancestor_store,
        &complete_patch,
        &stats
    ));
    ASSERT_STATUS_OK(evoke_document_cow_tree_lookup(
        &complete_patch,
        0,
        &found
    ));
    ASSERT_TRUE(!evoke_document_cow_record_is_l0_owned(&found));
    ASSERT_TRUE(
        (found.semantic_state.flags &
         EVOKE_SEMANTIC_STATE_FLAG_COMPLETE) != 0
    );

    sealed_quarantined = l0_pending;
    sealed_quarantined.version.flags &=
        ~EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED;
    sealed_quarantined.event_residency = 1;
    quarantine_document_cow_record(&sealed_quarantined, 2, 10);
    ASSERT_STATUS_OK(evoke_document_cow_build_external_patch(
        &ancestor.root,
        1,
        1,
        &sealed_quarantined,
        1,
        22,
        load_document_cow_fake_object,
        &ancestor_store,
        &quarantine_patch,
        &stats
    ));
    ASSERT_STATUS_OK(evoke_document_cow_tree_lookup(
        &quarantine_patch,
        0,
        &found
    ));
    ASSERT_TRUE(!evoke_document_cow_record_is_l0_owned(&found));
    ASSERT_TRUE(
        (found.semantic_state.flags &
         EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED) != 0
    );

    evoke_document_cow_tree_free(&quarantine_patch);
    evoke_document_cow_tree_free(&complete_patch);
    evoke_document_cow_tree_free(&ancestor);
}

static void
test_document_cow_root_relative_reuse_guards(void)
{
    evoke_document_cow_record original;
    evoke_document_cow_record drained;
    evoke_document_cow_record replacement;
    evoke_document_cow_record invalid;
    evoke_document_cow_record found;
    evoke_document_cow_tree ancestor;
    evoke_document_cow_tree drained_tree;
    evoke_document_cow_tree reincarnated;
    evoke_document_cow_tree rejected;
    evoke_document_cow_update_stats stats;
    evoke_document_cow_born_prefix_stats born_stats;
    document_cow_fake_store ancestor_store;
    document_cow_fake_store drained_store;
    document_cow_fake_store reincarnated_store;
    uint32_t next_block = 200;
    size_t born_prefix_count = 0;
    bool reusable = true;

    evoke_document_cow_tree_init(&ancestor);
    evoke_document_cow_tree_init(&drained_tree);
    evoke_document_cow_tree_init(&reincarnated);
    evoke_document_cow_tree_init(&rejected);
    memset(&ancestor_store, 0, sizeof(ancestor_store));
    memset(&drained_store, 0, sizeof(drained_store));
    memset(&reincarnated_store, 0, sizeof(reincarnated_store));

    initialize_document_cow_record(&original, 0, false);
    original.retirement.document_slot = 0;
    original.retirement.retirement_sequence = 10;
    original.retirement.document_length =
        original.version.document_length;
    original.retirement.flags =
        EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID;
    original.event_residency = 1;
    ASSERT_STATUS_OK(evoke_document_cow_tree_build(
        &original,
        1,
        20,
        &ancestor
    ));
    bind_document_cow_test_tree(&ancestor, 20, &next_block);
    ancestor_store.tree = &ancestor;
    ASSERT_TRUE(ancestor.root.reusable_document_count == 0);
    ASSERT_STATUS_OK(evoke_document_cow_find_reusable_external(
        &ancestor.root,
        ancestor.document_slot_count,
        load_document_cow_fake_object,
        &ancestor_store,
        &found,
        &reusable
    ));
    ASSERT_TRUE(!reusable);
    ASSERT_TRUE(ancestor_store.load_count == 0);

    initialize_document_cow_record(&replacement, 0, false);
    replacement.version.born_sequence = 11;
    replacement.version.heap_block = 99;
    replacement.event_residency = 1;
    ASSERT_TRUE(evoke_document_cow_build_external_patch(
        &ancestor.root,
        1,
        1,
        &replacement,
        1,
        21,
        load_document_cow_fake_object,
        &ancestor_store,
        &rejected,
        &stats
    ) == EVOKE_ERR_FORMAT);

    drained = original;
    drained.event_residency = 0;
    ASSERT_STATUS_OK(evoke_document_cow_build_external_patch(
        &ancestor.root,
        1,
        1,
        &drained,
        1,
        21,
        load_document_cow_fake_object,
        &ancestor_store,
        &drained_tree,
        &stats
    ));
    bind_document_cow_test_tree(&drained_tree, 21, &next_block);
    drained_store.tree = &drained_tree;
    ASSERT_TRUE(drained_tree.root.reusable_document_count == 1);
    ASSERT_TRUE(drained_tree.root.first_reusable_document_slot == 0);
    reusable = false;
    ASSERT_STATUS_OK(evoke_document_cow_find_reusable_external(
        &drained_tree.root,
        drained_tree.document_slot_count,
        load_document_cow_fake_object,
        &drained_store,
        &found,
        &reusable
    ));
    ASSERT_TRUE(reusable);
    ASSERT_TRUE(found.version.born_sequence == 1);
    drained_store.load_count = 0;
    reusable = true;
    ASSERT_STATUS_OK(evoke_document_cow_find_reusable_external_from(
        &drained_tree.root,
        drained_tree.document_slot_count,
        drained_tree.document_slot_count,
        load_document_cow_fake_object,
        &drained_store,
        &found,
        &reusable
    ));
    ASSERT_TRUE(!reusable);
    ASSERT_TRUE(drained_store.load_count == 0);

    invalid = replacement;
    invalid.version.born_sequence = 10;
    ASSERT_TRUE(evoke_document_cow_build_external_patch(
        &drained_tree.root,
        1,
        1,
        &invalid,
        1,
        22,
        load_document_cow_fake_object,
        &drained_store,
        &rejected,
        &stats
    ) == EVOKE_ERR_FORMAT);
    invalid = replacement;
    invalid.event_residency = 0;
    ASSERT_TRUE(evoke_document_cow_build_external_patch(
        &drained_tree.root,
        1,
        1,
        &invalid,
        1,
        22,
        load_document_cow_fake_object,
        &drained_store,
        &rejected,
        &stats
    ) == EVOKE_ERR_FORMAT);

    ASSERT_STATUS_OK(evoke_document_cow_build_external_patch(
        &drained_tree.root,
        1,
        1,
        &replacement,
        1,
        22,
        load_document_cow_fake_object,
        &drained_store,
        &reincarnated,
        &stats
    ));
    bind_document_cow_test_tree(&reincarnated, 22, &next_block);
    reincarnated_store.tree = &reincarnated;
    ASSERT_STATUS_OK(evoke_document_cow_validate_external(
        &reincarnated.root,
        reincarnated.document_slot_count,
        load_document_cow_fake_object,
        &reincarnated_store
    ));
    ASSERT_TRUE(reincarnated.root.live_document_count == 1);
    ASSERT_TRUE(reincarnated.root.reusable_document_count == 0);
    ASSERT_TRUE(reincarnated.root.min_live_born_sequence == 11);
    reincarnated_store.load_count = 0;
    born_prefix_count = 0;
    ASSERT_STATUS_OK(
        evoke_document_cow_collect_live_born_prefix_external(
            &reincarnated.root,
            reincarnated.document_slot_count,
            1,
            load_document_cow_fake_object,
            &reincarnated_store,
            &found,
            1,
            &born_prefix_count,
            &born_stats
        )
    );
    ASSERT_TRUE(born_prefix_count == 1);
    ASSERT_TRUE(found.version.document_slot == 0);
    ASSERT_TRUE(found.version.born_sequence == 11);
    reusable = true;
    reincarnated_store.load_count = 0;
    ASSERT_STATUS_OK(evoke_document_cow_find_reusable_external(
        &reincarnated.root,
        reincarnated.document_slot_count,
        load_document_cow_fake_object,
        &reincarnated_store,
        &found,
        &reusable
    ));
    ASSERT_TRUE(!reusable);
    ASSERT_TRUE(reincarnated_store.load_count == 0);
    ASSERT_STATUS_OK(evoke_document_cow_tree_lookup(
        &reincarnated,
        0,
        &found
    ));
    ASSERT_TRUE(found.version.born_sequence == 11);
    ASSERT_TRUE(found.version.heap_block == 99);
    ASSERT_STATUS_OK(evoke_document_cow_tree_lookup(
        &drained_tree,
        0,
        &found
    ));
    ASSERT_TRUE(found.retirement.retirement_sequence == 10);

    evoke_document_cow_tree_free(&rejected);
    evoke_document_cow_tree_free(&reincarnated);
    evoke_document_cow_tree_free(&drained_tree);
    evoke_document_cow_tree_free(&ancestor);
}

static void
test_mixed_extent_scoring_combines_lexical_and_semantic(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    uint32_t query[] = {0};
    uint32_t weighted_query[] = {0, 0};
    float query_weights[] = {0.5f, 1.5f};
    float invalid_weights[] = {NAN, 1.0f};
    uint32_t semantic_doc_ids[] = {1, 3};
    float semantic_impacts[] = {0.25f, -0.10f};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index index;
    evoke_corpus_stats stats;
    evoke_posting_extent extents[2];
    evoke_term_extent_list terms[4];
    float *expected_scores = NULL;
    float *mixed_scores = NULL;
    float *weighted_scores = NULL;
    uint64_t start;
    uint64_t end;
    uint32_t doc_id;

    evoke_index_init(&index);
    memset(&stats, 0, sizeof(stats));
    memset(extents, 0, sizeof(extents));
    memset(terms, 0, sizeof(terms));
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        4,
        &params,
        false,
        &index
    ));

    stats.document_count = index.num_docs;
    stats.doc_frequencies = index.doc_frequencies;
    stats.vocab_size = index.vocab_size;
    for (doc_id = 0; doc_id < index.num_docs; doc_id++)
    {
        stats.total_document_length += index.doc_lengths[doc_id];
    }

    start = index.indptr[0];
    end = index.indptr[1];
    extents[0].indices = &index.indices[start];
    extents[0].term_frequencies = &index.term_frequencies[start];
    extents[0].len = end - start;
    extents[0].local_document_count = index.num_docs;
    extents[0].kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
    extents[1].data = semantic_impacts;
    extents[1].indices = semantic_doc_ids;
    extents[1].len = 2;
    extents[1].local_document_count = index.num_docs;
    extents[1].kind = EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
    terms[0].extents = extents;
    terms[0].len = 2;

    ASSERT_STATUS_OK(evoke_scores_from_ids_neutral(
        &index,
        &stats,
        NULL,
        0,
        query,
        1,
        NULL,
        &expected_scores
    ));
    expected_scores[1] += semantic_impacts[0];
    expected_scores[3] += semantic_impacts[1];
    ASSERT_STATUS_OK(evoke_scores_from_ids_mixed(
        &index,
        &stats,
        terms,
        index.vocab_size,
        query,
        1,
        NULL,
        &mixed_scores
    ));
    assert_float_array(mixed_scores, expected_scores, index.num_docs);
    ASSERT_STATUS_OK(evoke_scores_from_weighted_ids_mixed_retired(
        &index,
        &stats,
        terms,
        index.vocab_size,
        NULL,
        0,
        weighted_query,
        query_weights,
        2,
        NULL,
        &weighted_scores
    ));
    for (doc_id = 0; doc_id < index.num_docs; doc_id++)
    {
        assert_float_close(
            weighted_scores[doc_id],
            2.0f * mixed_scores[doc_id]
        );
    }
    free(weighted_scores);
    weighted_scores = NULL;
    ASSERT_TRUE(evoke_scores_from_weighted_ids_mixed_retired(
        &index,
        &stats,
        terms,
        index.vocab_size,
        NULL,
        0,
        weighted_query,
        invalid_weights,
        2,
        NULL,
        &weighted_scores
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(weighted_scores == NULL);

    free(mixed_scores);
    mixed_scores = NULL;
    extents[1].kind = 0;
    ASSERT_TRUE(evoke_scores_from_ids_mixed(
        &index,
        &stats,
        terms,
        index.vocab_size,
        query,
        1,
        NULL,
        &mixed_scores
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(mixed_scores == NULL);

    free(expected_scores);
    evoke_index_free(&index);
}

static void
test_empty_mixed_extent_scoring(void)
{
    evoke_index index;
    evoke_corpus_stats stats;
    uint32_t query[] = {0};
    float *scores = NULL;

    evoke_index_init(&index);
    memset(&stats, 0, sizeof(stats));
    ASSERT_STATUS_OK(evoke_scores_from_ids_mixed(
        &index,
        &stats,
        NULL,
        0,
        query,
        1,
        NULL,
        &scores
    ));
    ASSERT_TRUE(scores == NULL);
    evoke_index_free(&index);
}

static void
test_retired_mixed_scoring_matches_live_rebuild(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2, 4};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    uint32_t query[] = {0, 1, 2, 3, 4};
    uint32_t retired_ids[] = {1};
    uint32_t all_retired_ids[] = {0, 1, 2, 3};
    uint32_t duplicate_retired_ids[] = {1, 1};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 3),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_doc_ids live_docs[] = {
        make_doc(doc0, 3),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index index;
    evoke_index live;
    evoke_corpus_stats stats;
    evoke_posting_extent extents[5];
    evoke_term_extent_list terms[5];
    float *scores = NULL;
    float *expected = NULL;
    uint32_t original_term_zero_df;
    uint32_t term_id;

    evoke_index_init(&index);
    evoke_index_init(&live);
    memset(&stats, 0, sizeof(stats));
    memset(extents, 0, sizeof(extents));
    memset(terms, 0, sizeof(terms));
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        4,
        &params,
        false,
        &index
    ));
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        live_docs,
        3,
        &params,
        false,
        &live
    ));
    ASSERT_TRUE(index.vocab_size == 5);

    stats.document_count = 3;
    stats.total_document_length =
        index.doc_lengths[0] +
        index.doc_lengths[2] +
        index.doc_lengths[3];
    stats.doc_frequencies = index.doc_frequencies;
    stats.vocab_size = index.vocab_size;
    for (term_id = 0; term_id < index.vocab_size; term_id++)
    {
        uint64_t start = index.indptr[term_id];

        extents[term_id].indices = &index.indices[start];
        extents[term_id].term_frequencies =
            &index.term_frequencies[start];
        extents[term_id].len =
            index.indptr[term_id + 1] - start;
        extents[term_id].local_document_count = index.num_docs;
        extents[term_id].kind =
            EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
        terms[term_id].extents = &extents[term_id];
        terms[term_id].len = 1;
    }

    ASSERT_STATUS_OK(evoke_scores_from_ids_mixed_retired(
        &index,
        &stats,
        terms,
        index.vocab_size,
        retired_ids,
        1,
        query,
        5,
        NULL,
        &scores
    ));
    ASSERT_STATUS_OK(evoke_scores_from_ids_exact_stats(
        &live,
        query,
        5,
        NULL,
        &expected
    ));
    assert_float_close(scores[0], expected[0]);
    ASSERT_TRUE(scores[1] == 0.0f);
    assert_float_close(scores[2], expected[1]);
    assert_float_close(scores[3], expected[2]);

    free(scores);
    scores = NULL;
    original_term_zero_df = index.doc_frequencies[0];
    index.doc_frequencies[0] = 4;
    ASSERT_TRUE(evoke_scores_from_ids_mixed_retired(
        &index,
        &stats,
        terms,
        index.vocab_size,
        retired_ids,
        1,
        query,
        5,
        NULL,
        &scores
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(scores == NULL);
    index.doc_frequencies[0] = original_term_zero_df;
    ASSERT_TRUE(evoke_scores_from_ids_mixed_retired(
        &index,
        &stats,
        terms,
        index.vocab_size,
        duplicate_retired_ids,
        2,
        query,
        5,
        NULL,
        &scores
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(scores == NULL);

    stats.document_count = 0;
    stats.total_document_length = 0;
    ASSERT_STATUS_OK(evoke_scores_from_ids_mixed_retired(
        &index,
        &stats,
        terms,
        index.vocab_size,
        all_retired_ids,
        4,
        query,
        5,
        NULL,
        &scores
    ));
    for (term_id = 0; term_id < index.num_docs; term_id++)
    {
        ASSERT_TRUE(scores[term_id] == 0.0f);
    }

    free(scores);
    free(expected);
    evoke_index_free(&live);
    evoke_index_free(&index);
}

static void
test_lucene_index_layout(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index index;
    float expected_data[] = {
        0.35775340f,
        0.24109468f,
        0.24109468f,
        0.29185143f,
        0.29185143f,
        0.35775340f,
        0.64211881f
    };
    uint32_t expected_indices[] = {0, 2, 0, 1, 1, 2, 3};
    uint64_t expected_indptr[] = {0, 2, 4, 6, 7, 7};
    uint32_t expected_doc_lengths[] = {3, 2, 3, 1};
    uint32_t expected_doc_frequencies[] = {2, 2, 2, 1, 0};
    uint32_t expected_term_frequencies[] = {2, 1, 1, 1, 1, 2, 1};

    evoke_index_init(&index);
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        4,
        &params,
        true,
        &index
    ));

    ASSERT_TRUE(index.num_docs == 4);
    ASSERT_TRUE(index.vocab_size == 5);
    ASSERT_TRUE(index.has_empty_token);
    ASSERT_TRUE(index.empty_token_id == 4);
    assert_float_array(index.data, expected_data, 7);
    assert_uint32_array(index.indices, expected_indices, 7);
    assert_uint64_array(index.indptr, expected_indptr, 6);
    assert_uint32_array_present(index.doc_lengths, expected_doc_lengths, 4);
    assert_uint32_array_present(
        index.doc_frequencies,
        expected_doc_frequencies,
        5
    );
    assert_uint32_array_present(
        index.term_frequencies,
        expected_term_frequencies,
        7
    );
    ASSERT_TRUE(index.nonoccurrence == NULL);

    evoke_index_free(&index);
}

static void
test_compact_id_builder_matches_standard(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_BM25PLUS
    };
    evoke_index standard;
    evoke_index compact;

    evoke_index_init(&standard);
    evoke_index_init(&compact);

    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        4,
        &params,
        true,
        &standard
    ));
    ASSERT_STATUS_OK(evoke_build_index_from_ids_compact(
        docs,
        4,
        &params,
        true,
        &compact
    ));
    assert_index_layout_equal(&compact, &standard);

    evoke_index_free(&standard);
    evoke_index_free(&compact);
}

static void
test_compact_token_builder_matches_standard(void)
{
    const char *doc0[] = {"alpha", "alpha", "beta"};
    const char *doc1[] = {"beta", "gamma"};
    const char *doc2[] = {"alpha", "gamma", "gamma"};
    evoke_doc_tokens docs[] = {
        {doc0, 3},
        {doc1, 2},
        {doc2, 3}
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index standard;
    evoke_index compact;

    evoke_index_init(&standard);
    evoke_index_init(&compact);

    ASSERT_STATUS_OK(evoke_build_index_from_tokens(
        docs,
        3,
        &params,
        &standard
    ));
    ASSERT_STATUS_OK(evoke_build_index_from_tokens_compact(
        docs,
        3,
        &params,
        &compact
    ));
    assert_index_layout_equal(&compact, &standard);

    evoke_index_free(&standard);
    evoke_index_free(&compact);
}

typedef struct limited_term_entry_reader
{
    const evoke_term_entry *entries;
    uint64_t len;
    uint64_t pos;
} limited_term_entry_reader;

typedef struct changing_term_entry_reader
{
    evoke_term_entry entries[3][2];
    uint64_t pos;
    uint32_t pass;
} changing_term_entry_reader;

static evoke_status
limited_term_entry_read(void *ctx, evoke_term_entry *entry_out)
{
    limited_term_entry_reader *reader = ctx;

    if (reader == NULL || entry_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (reader->pos >= reader->len)
    {
        return EVOKE_ERR_RANGE;
    }
    *entry_out = reader->entries[reader->pos++];
    return EVOKE_OK;
}

static evoke_status
limited_term_entry_rewind(void *ctx)
{
    limited_term_entry_reader *reader = ctx;

    if (reader == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    reader->pos = 0;
    return EVOKE_OK;
}

static evoke_status
changing_term_entry_read(void *ctx, evoke_term_entry *entry_out)
{
    changing_term_entry_reader *reader = ctx;

    if (reader == NULL || entry_out == NULL || reader->pass >= 3 ||
        reader->pos >= 2)
    {
        return EVOKE_ERR_RANGE;
    }
    *entry_out = reader->entries[reader->pass][reader->pos++];
    return EVOKE_OK;
}

static evoke_status
changing_term_entry_rewind(void *ctx)
{
    changing_term_entry_reader *reader = ctx;

    if (reader == NULL || reader->pass >= 2)
    {
        return EVOKE_ERR_RANGE;
    }
    reader->pass++;
    reader->pos = 0;
    return EVOKE_OK;
}

static void
test_builders_reject_invalid_inputs(void)
{
    const char *empty_tokens[] = {NULL};
    const char *nonempty_tokens[] = {"term"};
    uint32_t zero_token_id[] = {0};
    uint32_t max_token_id[] = {UINT32_MAX};
    evoke_doc_tokens empty_docs[] = {
        {empty_tokens, 0}
    };
    evoke_doc_tokens missing_token_array[] = {
        {NULL, 1}
    };
    evoke_doc_tokens valid_token_docs[] = {
        {nonempty_tokens, 1}
    };
    evoke_doc_ids missing_id_array[] = {
        {NULL, 1}
    };
    evoke_doc_ids valid_id_docs[] = {
        {zero_token_id, 1}
    };
    evoke_doc_ids unrepresentable_vocab[] = {
        {max_token_id, 1}
    };
    uint32_t doc_lengths[] = {1};
    evoke_term_entry entry = {
        .token_id = 0,
        .doc_id = 0,
        .tf = 1
    };
    limited_term_entry_reader short_reader = {
        .entries = &entry,
        .len = 0,
        .pos = 0
    };
    changing_term_entry_reader changing_reader = {
        .entries = {
            {
                {.token_id = 0, .doc_id = 0, .tf = 1},
                {.token_id = 0, .doc_id = 1, .tf = 1}
            },
            {
                {.token_id = 0, .doc_id = 0, .tf = 1},
                {.token_id = 1, .doc_id = 1, .tf = 1}
            },
            {
                {.token_id = 0, .doc_id = 0, .tf = 1},
                {.token_id = 0, .doc_id = 1, .tf = 1}
            }
        },
        .pos = 0,
        .pass = 0
    };
    changing_term_entry_reader late_changing_reader = {
        .entries = {
            {
                {.token_id = 0, .doc_id = 0, .tf = 1},
                {.token_id = 1, .doc_id = 1, .tf = 1}
            },
            {
                {.token_id = 0, .doc_id = 0, .tf = 1},
                {.token_id = 1, .doc_id = 1, .tf = 1}
            },
            {
                {.token_id = 0, .doc_id = 0, .tf = 1},
                {.token_id = 0, .doc_id = 1, .tf = 1}
            }
        },
        .pos = 0,
        .pass = 0
    };
    uint32_t changing_doc_lengths[] = {1, 1};
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index index;

    evoke_index_init(&index);
    ASSERT_TRUE(evoke_build_index_from_tokens_compact(
        NULL,
        1,
        &params,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_tokens_compact(
        empty_docs,
        1,
        &params,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_tokens(
        empty_docs,
        1,
        &params,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_tokens(
        missing_token_array,
        1,
        &params,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_tokens_compact(
        missing_token_array,
        1,
        &params,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_ids(
        missing_id_array,
        1,
        &params,
        false,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_ids_compact(
        missing_id_array,
        1,
        &params,
        false,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_ids(
        unrepresentable_vocab,
        1,
        &params,
        false,
        &index
    ) == EVOKE_ERR_RANGE);
    ASSERT_TRUE(evoke_build_index_from_ids_compact(
        unrepresentable_vocab,
        1,
        &params,
        false,
        &index
    ) == EVOKE_ERR_RANGE);
    ASSERT_STATUS_OK(evoke_build_index_from_tokens(
        valid_token_docs,
        1,
        &params,
        &index
    ));
    evoke_index_free(&index);
    ASSERT_TRUE(evoke_build_index_from_term_entries(
        NULL,
        1,
        doc_lengths,
        1,
        1,
        &params,
        false,
        false,
        0,
        NULL,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_term_entries(
        NULL,
        0,
        NULL,
        0,
        UINT32_MAX,
        &params,
        false,
        false,
        0,
        NULL,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_term_entry_reader(
        1,
        limited_term_entry_read,
        limited_term_entry_rewind,
        &short_reader,
        doc_lengths,
        1,
        1,
        &params,
        false,
        false,
        0,
        NULL,
        &index
    ) == EVOKE_ERR_RANGE);
    ASSERT_TRUE(evoke_build_index_from_term_entries(
        &entry,
        1,
        NULL,
        0,
        1,
        &params,
        false,
        false,
        0,
        NULL,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_term_entry_reader(
        1,
        limited_term_entry_read,
        limited_term_entry_rewind,
        &short_reader,
        NULL,
        0,
        1,
        &params,
        false,
        false,
        0,
        NULL,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_term_entry_reader(
        2,
        changing_term_entry_read,
        changing_term_entry_rewind,
        &changing_reader,
        changing_doc_lengths,
        2,
        2,
        &params,
        false,
        false,
        0,
        NULL,
        &index
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_build_index_from_term_entry_reader(
        2,
        changing_term_entry_read,
        changing_term_entry_rewind,
        &late_changing_reader,
        changing_doc_lengths,
        2,
        2,
        &params,
        false,
        false,
        0,
        NULL,
        &index
    ) == EVOKE_ERR_INVALID);

    params.k1 = NAN;
    ASSERT_TRUE(evoke_build_index_from_term_entries(
        &entry,
        1,
        doc_lengths,
        1,
        1,
        &params,
        false,
        false,
        0,
        NULL,
        &index
    ) == EVOKE_ERR_INVALID);
    params.k1 = 1.5f;
    params.b = 1.1f;
    ASSERT_TRUE(evoke_build_index_from_ids(
        valid_id_docs,
        1,
        &params,
        false,
        &index
    ) == EVOKE_ERR_INVALID);
}

static void
test_empty_index_builders_and_roundtrip(void)
{
    evoke_doc_ids empty_docs[] = {
        {NULL, 0},
        {NULL, 0}
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index index;
    evoke_index restored;
    evoke_topk_result topk;
    uint8_t *serialized = NULL;
    size_t serialized_len = 0;
    float *scores = (float *) 1;

    evoke_index_init(&index);
    evoke_index_init(&restored);
    memset(&topk, 0, sizeof(topk));

    ASSERT_TRUE(evoke_build_index_from_tokens(
        NULL,
        0,
        &params,
        &index
    ) == EVOKE_OK);
    ASSERT_TRUE(index.num_docs == 0);
    ASSERT_TRUE(index.vocab_size == 0);
    ASSERT_TRUE(index.data_len == 0);
    ASSERT_TRUE(index.indptr != NULL);
    ASSERT_TRUE(index.indptr[0] == 0);
    ASSERT_TRUE(evoke_scores_from_ids(
        &index,
        NULL,
        0,
        NULL,
        &scores
    ) == EVOKE_OK);
    ASSERT_TRUE(scores == NULL);
    ASSERT_TRUE(evoke_scores_from_ids_exact_stats(
        &index,
        NULL,
        0,
        NULL,
        &scores
    ) == EVOKE_OK);
    ASSERT_TRUE(scores == NULL);
    ASSERT_TRUE(evoke_topk(NULL, 0, 0, true, &topk) == EVOKE_OK);
    ASSERT_TRUE(topk.len == 0);

    ASSERT_TRUE(evoke_serialize_index(
        &index,
        &serialized,
        &serialized_len
    ) == EVOKE_OK);
    ASSERT_TRUE(serialized != NULL);
    ASSERT_TRUE(evoke_deserialize_index(
        serialized,
        serialized_len,
        &restored
    ) == EVOKE_OK);
    ASSERT_TRUE(restored.num_docs == 0);
    ASSERT_TRUE(restored.vocab_size == 0);
    ASSERT_TRUE(restored.data_len == 0);
    ASSERT_TRUE(restored.indptr != NULL);
    ASSERT_TRUE(restored.indptr[0] == 0);

    evoke_index_free(&restored);
    evoke_index_free(&index);
    free(serialized);
    serialized = NULL;

    evoke_index_init(&index);
    evoke_index_init(&restored);
    ASSERT_TRUE(evoke_build_index_from_ids(
        empty_docs,
        2,
        &params,
        true,
        &index
    ) == EVOKE_OK);
    ASSERT_TRUE(index.num_docs == 2);
    ASSERT_TRUE(index.vocab_size == 1);
    ASSERT_TRUE(index.data_len == 0);
    ASSERT_TRUE(index.has_empty_token);
    ASSERT_TRUE(index.term_frequencies == NULL);
    ASSERT_TRUE(index.doc_lengths != NULL);
    ASSERT_TRUE(index.doc_frequencies != NULL);
    ASSERT_TRUE(evoke_serialize_index(
        &index,
        &serialized,
        &serialized_len
    ) == EVOKE_OK);
    ASSERT_TRUE(evoke_deserialize_index(
        serialized,
        serialized_len,
        &restored
    ) == EVOKE_OK);
    ASSERT_TRUE(restored.num_docs == 2);
    ASSERT_TRUE(restored.vocab_size == 1);
    ASSERT_TRUE(restored.data_len == 0);
    ASSERT_TRUE(restored.has_empty_token);
    ASSERT_TRUE(restored.doc_lengths != NULL);
    ASSERT_TRUE(restored.doc_frequencies != NULL);
    ASSERT_TRUE(restored.doc_lengths[0] == 0);
    ASSERT_TRUE(restored.doc_lengths[1] == 0);
    ASSERT_TRUE(restored.doc_frequencies[0] == 0);

    evoke_index_free(&restored);
    evoke_index_free(&index);
    free(serialized);
    serialized = NULL;

    evoke_index_init(&index);
    ASSERT_TRUE(evoke_build_index_from_tokens_compact(
        NULL,
        0,
        &params,
        &index
    ) == EVOKE_OK);
    ASSERT_TRUE(index.indptr != NULL);
    evoke_index_free(&index);

    evoke_index_init(&index);
    ASSERT_TRUE(evoke_build_index_from_ids(
        NULL,
        0,
        &params,
        false,
        &index
    ) == EVOKE_OK);
    ASSERT_TRUE(index.indptr != NULL);
    evoke_index_free(&index);

    evoke_index_init(&index);
    ASSERT_TRUE(evoke_build_index_from_ids_compact(
        NULL,
        0,
        &params,
        false,
        &index
    ) == EVOKE_OK);
    ASSERT_TRUE(index.indptr != NULL);
    evoke_index_free(&index);

    evoke_index_init(&index);
    ASSERT_TRUE(evoke_build_index_from_term_entries(
        NULL,
        0,
        NULL,
        0,
        0,
        &params,
        false,
        false,
        0,
        NULL,
        &index
    ) == EVOKE_OK);
    ASSERT_TRUE(index.indptr != NULL);
    evoke_index_free(&index);
}

static void
test_bm25plus_scores_and_weight_mask(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    uint32_t query[] = {0, 2};
    float weight_mask[] = {1.0f, 0.0f, 1.0f, 0.0f};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_BM25PLUS
    };
    evoke_index index;
    float *scores = NULL;
    evoke_topk_result topk;
    float expected_nonocc[] = {
        0.45814538f,
        0.45814538f,
        0.45814538f,
        0.80471897f,
        0.0f
    };
    uint32_t expected_doc_ids[] = {2, 0};
    float expected_scores[] = {2.89537597f, 2.09860134f};

    evoke_index_init(&index);
    memset(&topk, 0, sizeof(topk));
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        4,
        &params,
        true,
        &index
    ));
    assert_float_array(index.nonoccurrence, expected_nonocc, 5);

    ASSERT_STATUS_OK(evoke_scores_from_ids(
        &index,
        query,
        2,
        weight_mask,
        &scores
    ));
    ASSERT_STATUS_OK(evoke_topk(scores, index.num_docs, 2, true, &topk));
    assert_uint32_array(topk.doc_ids, expected_doc_ids, 2);
    assert_float_array(topk.scores, expected_scores, 2);

    free(scores);
    evoke_topk_result_free(&topk);
    evoke_index_free(&index);
}

static void
test_token_index_and_query_mapping(void)
{
    const char *doc0[] = {"cat", "cat", "feline"};
    const char *doc1[] = {"dog", "friend"};
    const char *doc2[] = {"cat", "bird", "bird"};
    evoke_doc_tokens docs[] = {
        {.tokens = doc0, .len = 3},
        {.tokens = doc1, .len = 2},
        {.tokens = doc2, .len = 3}
    };
    const char *query_tokens[] = {"bird", "cat", "missing"};
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index index;
    uint32_t *query_ids = NULL;
    size_t query_len = 0;
    float *scores = NULL;
    evoke_topk_result topk;

    evoke_index_init(&index);
    memset(&topk, 0, sizeof(topk));
    ASSERT_STATUS_OK(evoke_build_index_from_tokens(
        docs,
        3,
        &params,
        &index
    ));
    ASSERT_TRUE(index.vocab != NULL);
    ASSERT_TRUE(index.has_empty_token == false);

    ASSERT_STATUS_OK(evoke_query_token_ids(
        &index,
        query_tokens,
        3,
        &query_ids,
        &query_len
    ));
    ASSERT_TRUE(query_len == 2);
    ASSERT_STATUS_OK(evoke_scores_from_ids(
        &index,
        query_ids,
        query_len,
        NULL,
        &scores
    ));
    ASSERT_STATUS_OK(evoke_topk(scores, index.num_docs, 2, true, &topk));
    ASSERT_TRUE(topk.doc_ids[0] == 2);
    ASSERT_TRUE(topk.doc_ids[1] == 0);

    free(query_ids);
    free(scores);
    evoke_topk_result_free(&topk);
    evoke_index_free(&index);
}

static void
test_exact_stats_scoring_matches_dense_ids(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    uint32_t query[] = {0, 2};
    float weight_mask[] = {1.0f, 0.5f, 1.0f, 1.0f};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25L,
        .idf_method = EVOKE_METHOD_BM25L
    };
    evoke_index index;
    float *dense_scores = NULL;
    float *exact_scores = NULL;

    evoke_index_init(&index);
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        4,
        &params,
        true,
        &index
    ));
    ASSERT_STATUS_OK(evoke_scores_from_ids(
        &index,
        query,
        2,
        weight_mask,
        &dense_scores
    ));
    ASSERT_STATUS_OK(evoke_scores_from_ids_exact_stats(
        &index,
        query,
        2,
        weight_mask,
        &exact_scores
    ));
    assert_float_array(exact_scores, dense_scores, index.num_docs);

    free(dense_scores);
    free(exact_scores);
    evoke_index_free(&index);
}

static void
test_exact_stats_scoring_matches_dense_tokens(void)
{
    const char *doc0[] = {"cat", "cat", "feline"};
    const char *doc1[] = {"dog", "friend"};
    const char *doc2[] = {"cat", "bird", "bird"};
    const char *query_tokens[] = {"bird", "cat", "missing"};
    evoke_doc_tokens docs[] = {
        {.tokens = doc0, .len = 3},
        {.tokens = doc1, .len = 2},
        {.tokens = doc2, .len = 3}
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index index;
    uint32_t *query_ids = NULL;
    size_t query_len = 0;
    float *dense_scores = NULL;
    float *exact_scores = NULL;

    evoke_index_init(&index);
    ASSERT_STATUS_OK(evoke_build_index_from_tokens(
        docs,
        3,
        &params,
        &index
    ));
    ASSERT_STATUS_OK(evoke_query_token_ids(
        &index,
        query_tokens,
        3,
        &query_ids,
        &query_len
    ));
    ASSERT_STATUS_OK(evoke_scores_from_ids(
        &index,
        query_ids,
        query_len,
        NULL,
        &dense_scores
    ));
    ASSERT_STATUS_OK(evoke_scores_from_ids_exact_stats(
        &index,
        query_ids,
        query_len,
        NULL,
        &exact_scores
    ));
    assert_float_array(exact_scores, dense_scores, index.num_docs);

    free(query_ids);
    free(dense_scores);
    free(exact_scores);
    evoke_index_free(&index);
}

static void
test_fragmented_extent_scoring_matches_contiguous(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    uint32_t query[] = {0, 2, 0};
    float weight_mask[] = {1.0f, 0.5f, 1.0f, 0.75f};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25PLUS,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index index;
    evoke_posting_extent extents[7];
    evoke_term_extent_list terms[5];
    float *contiguous_scores = NULL;
    float *fragmented_scores = NULL;
    evoke_topk_result contiguous_topk;
    evoke_topk_result fragmented_topk;
    uint32_t mapped_ids[] = {3, 1};
    uint32_t mapped_postings[] = {0, 1};
    evoke_posting_extent mapped_extent;
    size_t i;

    evoke_index_init(&index);
    memset(extents, 0, sizeof(extents));
    memset(terms, 0, sizeof(terms));
    memset(&contiguous_topk, 0, sizeof(contiguous_topk));
    memset(&fragmented_topk, 0, sizeof(fragmented_topk));

    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        4,
        &params,
        true,
        &index
    ));
    ASSERT_TRUE(index.data_len == 7);

    for (i = 0; i < 7; i++)
    {
        extents[i].data = &index.data[i];
        extents[i].indices = &index.indices[i];
        extents[i].term_frequencies = &index.term_frequencies[i];
        extents[i].len = 1;
        extents[i].local_document_count = index.num_docs;
        ASSERT_STATUS_OK(evoke_posting_extent_validate_layout(
            &index,
            &extents[i]
        ));
    }
    terms[0].extents = &extents[0];
    terms[0].len = 2;
    terms[1].extents = &extents[2];
    terms[1].len = 2;
    terms[2].extents = &extents[4];
    terms[2].len = 2;
    terms[3].extents = &extents[6];
    terms[3].len = 1;

    ASSERT_STATUS_OK(evoke_scores_from_ids(
        &index,
        query,
        3,
        weight_mask,
        &contiguous_scores
    ));
    ASSERT_STATUS_OK(evoke_scores_from_ids_extents(
        &index,
        terms,
        5,
        query,
        3,
        weight_mask,
        &fragmented_scores
    ));
    ASSERT_TRUE(memcmp(
        contiguous_scores,
        fragmented_scores,
        index.num_docs * sizeof(*contiguous_scores)
    ) == 0);

    ASSERT_STATUS_OK(evoke_topk(
        contiguous_scores,
        index.num_docs,
        index.num_docs,
        true,
        &contiguous_topk
    ));
    ASSERT_STATUS_OK(evoke_topk(
        fragmented_scores,
        index.num_docs,
        index.num_docs,
        true,
        &fragmented_topk
    ));
    ASSERT_TRUE(contiguous_topk.len == fragmented_topk.len);
    assert_uint32_array(
        fragmented_topk.doc_ids,
        contiguous_topk.doc_ids,
        contiguous_topk.len
    );
    ASSERT_TRUE(memcmp(
        contiguous_topk.scores,
        fragmented_topk.scores,
        contiguous_topk.len * sizeof(*contiguous_topk.scores)
    ) == 0);

    free(fragmented_scores);
    fragmented_scores = NULL;
    terms[0].extents = NULL;
    ASSERT_TRUE(evoke_scores_from_ids_extents(
        &index,
        terms,
        5,
        query,
        3,
        weight_mask,
        &fragmented_scores
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(fragmented_scores == NULL);
    ASSERT_TRUE(evoke_scores_from_ids_extents(
        &index,
        terms,
        4,
        query,
        3,
        weight_mask,
        &fragmented_scores
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(fragmented_scores == NULL);

    memset(&mapped_extent, 0, sizeof(mapped_extent));
    mapped_extent.indices = mapped_postings;
    mapped_extent.document_id_map = mapped_ids;
    mapped_extent.len = 2;
    mapped_extent.local_document_count = 2;
    ASSERT_STATUS_OK(evoke_posting_extent_validate_layout(
        &index,
        &mapped_extent
    ));
    mapped_ids[1] = index.num_docs;
    ASSERT_TRUE(evoke_posting_extent_validate_layout(
        &index,
        &mapped_extent
    ) == EVOKE_ERR_RANGE);

    free(contiguous_scores);
    evoke_topk_result_free(&contiguous_topk);
    evoke_topk_result_free(&fragmented_topk);
    evoke_index_free(&index);
}

static void
assert_posting_bound_contains(float exact, float bound)
{
    ASSERT_TRUE(isfinite(exact));
    ASSERT_TRUE(!isnan(bound));
    ASSERT_TRUE(exact <= bound);
}

static void
test_document_block_bounds_are_conservative(void)
{
    uint32_t document_lengths[] = {
        4, 10, 7, 3, 8, 12, 6, 9, 5, 11, 2, 13
    };
    uint32_t document_ids[] = {0, 1, 3, 4, 6, 7, 8, 11};
    uint32_t term_frequencies[] = {1, 3, 2, 7, 1, 4, 5, 2};
    float impacts[] = {
        -2.0f, 0.5f, 3.0f, 1.25f, -0.25f, 4.0f, 2.0f, -1.0f
    };
    float query_weights[] = {2.5f, -1.25f};
    evoke_method methods[] = {
        EVOKE_METHOD_ROBERTSON,
        EVOKE_METHOD_LUCENE,
        EVOKE_METHOD_ATIRE,
        EVOKE_METHOD_BM25L,
        EVOKE_METHOD_BM25PLUS
    };
    evoke_index index;
    evoke_corpus_stats stats;
    evoke_posting_extent extent;
    evoke_posting_block_record *records = NULL;
    evoke_posting_block_bound *bounds = NULL;
    float lexical_impact_bound = 0.0f;
    size_t bound_count = 0;
    size_t record_count = 0;
    size_t method_index;
    size_t weight_index;
    size_t bound_index;

    evoke_index_init(&index);
    index.num_docs = 12;
    index.doc_lengths = document_lengths;
    index.params.k1 = 1.5f;
    index.params.b = 0.75f;
    index.params.delta = 0.5f;
    stats.document_count = 12;
    stats.total_document_length = 90;
    stats.doc_frequencies = NULL;
    stats.vocab_size = 0;

    memset(&extent, 0, sizeof(extent));
    extent.indices = document_ids;
    extent.term_frequencies = term_frequencies;
    extent.len = 8;
    extent.local_document_count = 12;
    extent.kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
    ASSERT_STATUS_OK(evoke_posting_extent_build_block_records(
        &extent,
        2,
        &records,
        &record_count
    ));
    ASSERT_TRUE(record_count == 3);
    ASSERT_TRUE(records[0].posting_offset == 0);
    ASSERT_TRUE(records[0].posting_count == 3);
    ASSERT_TRUE(records[0].min_term_frequency == 1);
    ASSERT_TRUE(records[0].max_term_frequency == 3);
    ASSERT_STATUS_OK(evoke_posting_extent_validate_block_records(
        &extent,
        2,
        records,
        record_count
    ));
    records[1].block_id++;
    ASSERT_TRUE(evoke_posting_extent_validate_block_records(
        &extent,
        2,
        records,
        record_count
    ) == EVOKE_ERR_FORMAT);
    records[1].block_id--;
    evoke_posting_block_records_free(records);
    records = NULL;
    ASSERT_STATUS_OK(evoke_posting_extent_build_block_bounds(
        &index,
        &extent,
        2,
        &bounds,
        &bound_count
    ));
    ASSERT_TRUE(bound_count == 3);
    ASSERT_TRUE(bounds[0].block_id == 0);
    ASSERT_TRUE(bounds[0].posting_offset == 0);
    ASSERT_TRUE(bounds[0].posting_count == 3);
    ASSERT_TRUE(bounds[1].block_id == 1);
    ASSERT_TRUE(bounds[1].posting_offset == 3);
    ASSERT_TRUE(bounds[1].posting_count == 3);
    ASSERT_TRUE(bounds[2].block_id == 2);
    ASSERT_TRUE(bounds[2].posting_offset == 6);
    ASSERT_TRUE(bounds[2].posting_count == 2);

    for (method_index = 0;
         method_index < sizeof(methods) / sizeof(methods[0]);
         method_index++)
    {
        index.params.method = methods[method_index];
        index.params.idf_method = methods[method_index];
        for (weight_index = 0;
             weight_index <
                sizeof(query_weights) / sizeof(query_weights[0]);
             weight_index++)
        {
            float query_weight = query_weights[weight_index];
            double average_document_length =
                (double) stats.total_document_length /
                (double) stats.document_count;
            double idf = evoke_score_idf(
                index.params.idf_method,
                8.0,
                (double) stats.document_count
            );
            double nonoccurrence = 0.0;

            if (evoke_method_requires_nonoccurrence(index.params.method))
            {
                nonoccurrence = idf * evoke_score_tfc(
                    index.params.method,
                    0.0,
                    0.0,
                    average_document_length,
                    index.params.k1,
                    index.params.b,
                    index.params.delta
                );
            }
            for (bound_index = 0;
                 bound_index < bound_count;
                 bound_index++)
            {
                float upper_bound = 0.0f;
                uint64_t first = bounds[bound_index].posting_offset;
                uint64_t end =
                    first + bounds[bound_index].posting_count;
                uint64_t posting_index;

                ASSERT_STATUS_OK(evoke_posting_block_score_upper_bound(
                    &index,
                    &stats,
                    8,
                    &bounds[bound_index],
                    query_weight,
                    &upper_bound
                ));
                for (posting_index = first;
                     posting_index < end;
                     posting_index++)
                {
                    uint32_t document_id = document_ids[posting_index];
                    double tfc = evoke_score_tfc(
                        index.params.method,
                        term_frequencies[posting_index],
                        document_lengths[document_id],
                        average_document_length,
                        index.params.k1,
                        index.params.b,
                        index.params.delta
                    );
                    float exact = (float) (
                        (double) query_weight *
                        (idf * tfc - nonoccurrence)
                    );

                    assert_posting_bound_contains(exact, upper_bound);
                }
            }
        }
    }
    evoke_posting_block_bounds_free(bounds);
    bounds = NULL;

    extent.term_frequencies = NULL;
    extent.data = impacts;
    extent.kind = EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
    ASSERT_STATUS_OK(evoke_posting_extent_build_block_bounds(
        &index,
        &extent,
        2,
        &bounds,
        &bound_count
    ));
    ASSERT_TRUE(bound_count == 3);
    for (weight_index = 0;
         weight_index < sizeof(query_weights) / sizeof(query_weights[0]);
         weight_index++)
    {
        float query_weight = query_weights[weight_index];

        for (bound_index = 0;
             bound_index < bound_count;
             bound_index++)
        {
            float upper_bound = 0.0f;
            uint64_t first = bounds[bound_index].posting_offset;
            uint64_t end = first + bounds[bound_index].posting_count;
            uint64_t posting_index;

            ASSERT_STATUS_OK(evoke_posting_block_score_upper_bound(
                &index,
                &stats,
                0,
                &bounds[bound_index],
                query_weight,
                &upper_bound
            ));
            for (posting_index = first;
                 posting_index < end;
                 posting_index++)
            {
                float exact = query_weight * impacts[posting_index];

                assert_posting_bound_contains(exact, upper_bound);
            }
        }
    }
    evoke_posting_block_bounds_free(bounds);
    bounds = NULL;

    extent.kind = EVOKE_POSTING_EXTENT_LEXICAL_IMPACT;
    ASSERT_STATUS_OK(evoke_posting_extent_build_block_bounds(
        &index,
        &extent,
        2,
        &bounds,
        &bound_count
    ));
    ASSERT_STATUS_OK(evoke_posting_block_score_upper_bound(
        &index,
        &stats,
        8,
        &bounds[1],
        1.0f,
        &lexical_impact_bound
    ));
    ASSERT_TRUE(lexical_impact_bound >= 4.0f);
    evoke_posting_block_bounds_free(bounds);
    bounds = NULL;

    {
        uint32_t unsorted_map[] = {3, 2};
        uint32_t local_ids[] = {0, 1};
        float local_impacts[] = {1.0f, 2.0f};

        memset(&extent, 0, sizeof(extent));
        extent.indices = local_ids;
        extent.data = local_impacts;
        extent.document_id_map = unsorted_map;
        extent.len = 2;
        extent.local_document_count = 2;
        extent.kind = EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
        ASSERT_TRUE(evoke_posting_extent_build_block_bounds(
            &index,
            &extent,
            2,
            &bounds,
            &bound_count
        ) == EVOKE_ERR_FORMAT);
        ASSERT_TRUE(bounds == NULL);
        ASSERT_TRUE(evoke_posting_extent_build_block_bounds(
            &index,
            &extent,
            32,
            &bounds,
            &bound_count
        ) == EVOKE_ERR_INVALID);
    }
}

static void
test_global_blockmax_matches_exact_mixed_scoring(void)
{
    uint32_t document_lengths[32];
    uint32_t document_frequencies[] = {11, 0, 5};
    uint32_t lexical_a_ids[] = {0, 2, 8, 10, 16, 24};
    uint32_t lexical_a_tfs[] = {3, 2, 4, 1, 5, 2};
    uint32_t lexical_b_ids[] = {3, 9, 17, 25, 31};
    uint32_t lexical_b_tfs[] = {1, 3, 2, 4, 1};
    uint32_t semantic_ids[] = {
        0, 1, 4, 8, 12, 16, 20, 24, 28, 31
    };
    float semantic_impacts[] = {
        8.0f, 7.0f, 6.0f, 1.0f, 0.5f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f
    };
    uint32_t lexical_impact_ids[] = {1, 4, 12, 20, 28};
    float lexical_impacts[] = {4.0f, 3.0f, 1.0f, 0.5f, 0.25f};
    uint32_t retired_ids[] = {9, 25};
    uint32_t query_ids[] = {0, 1, 2};
    float query_weights[] = {1.0f, -0.75f, 1.25f};
    evoke_posting_extent extents[4];
    evoke_term_extent_list terms[3];
    evoke_posting_block_record *block_records[4] = {0};
    size_t block_record_counts[4] = {0};
    evoke_document_block_extrema document_blocks[4];
    evoke_index index;
    evoke_corpus_stats stats;
    evoke_topk_result expected = {0};
    evoke_topk_result actual = {0};
    evoke_blockmax_stats blockmax_stats;
    float *scores = NULL;
    uint32_t live_document_frequency = 0;
    uint64_t total_document_length = 0;

    evoke_index_init(&index);
    index.num_docs = 32;
    index.vocab_size = 3;
    index.doc_lengths = document_lengths;
    index.doc_frequencies = document_frequencies;
    index.params.k1 = 1.5f;
    index.params.b = 0.75f;
    index.params.delta = 0.5f;
    index.params.method = EVOKE_METHOD_BM25PLUS;
    index.params.idf_method = EVOKE_METHOD_LUCENE;
    memset(extents, 0, sizeof(extents));
    memset(terms, 0, sizeof(terms));
    memset(document_blocks, 0, sizeof(document_blocks));

    for (uint32_t document_id = 0; document_id < 32; document_id++)
    {
        uint32_t block_id = document_id >> 3;
        uint32_t document_length = 3 + document_id % 11;
        evoke_document_block_extrema *block =
            &document_blocks[block_id];

        document_lengths[document_id] = document_length;
        if (document_id != retired_ids[0] &&
            document_id != retired_ids[1])
        {
            total_document_length += document_length;
        }
        if (block->document_count == 0)
        {
            block->min_document_length = document_length;
            block->max_document_length = document_length;
        }
        else
        {
            block->min_document_length = (
                document_length < block->min_document_length
                    ? document_length
                    : block->min_document_length
            );
            block->max_document_length = (
                document_length > block->max_document_length
                    ? document_length
                    : block->max_document_length
            );
        }
        block->document_count++;
    }

    extents[0].indices = lexical_a_ids;
    extents[0].term_frequencies = lexical_a_tfs;
    extents[0].len = sizeof(lexical_a_ids) /
        sizeof(lexical_a_ids[0]);
    extents[0].local_document_count = 32;
    extents[0].kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
    extents[1].indices = lexical_b_ids;
    extents[1].term_frequencies = lexical_b_tfs;
    extents[1].len = sizeof(lexical_b_ids) /
        sizeof(lexical_b_ids[0]);
    extents[1].local_document_count = 32;
    extents[1].kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
    extents[2].indices = semantic_ids;
    extents[2].data = semantic_impacts;
    extents[2].len = sizeof(semantic_ids) / sizeof(semantic_ids[0]);
    extents[2].local_document_count = 32;
    extents[2].kind = EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
    extents[3].indices = lexical_impact_ids;
    extents[3].data = lexical_impacts;
    extents[3].len = sizeof(lexical_impact_ids) /
        sizeof(lexical_impact_ids[0]);
    extents[3].local_document_count = 32;
    extents[3].kind = EVOKE_POSTING_EXTENT_LEXICAL_IMPACT;

    for (size_t extent_index = 0; extent_index < 4; extent_index++)
    {
        ASSERT_STATUS_OK(evoke_posting_extent_build_block_records(
            &extents[extent_index],
            3,
            &block_records[extent_index],
            &block_record_counts[extent_index]
        ));
        ASSERT_TRUE(block_record_counts[extent_index] <= UINT32_MAX);
        extents[extent_index].blocks = block_records[extent_index];
        extents[extent_index].block_count =
            (uint32_t) block_record_counts[extent_index];
        extents[extent_index].block_shift = 3;
    }
    terms[0].extents = &extents[0];
    terms[0].len = 2;
    terms[1].extents = &extents[2];
    terms[1].len = 1;
    terms[2].extents = &extents[3];
    terms[2].len = 1;

    stats.document_count = 30;
    stats.total_document_length = total_document_length;
    stats.doc_frequencies = document_frequencies;
    stats.vocab_size = 3;
    ASSERT_STATUS_OK(evoke_term_live_document_frequency(
        &index,
        &stats,
        terms,
        0,
        retired_ids,
        2,
        &live_document_frequency
    ));
    ASSERT_TRUE(live_document_frequency == 9);
    ASSERT_STATUS_OK(evoke_term_live_document_frequency(
        &index,
        &stats,
        terms,
        1,
        retired_ids,
        2,
        &live_document_frequency
    ));
    ASSERT_TRUE(live_document_frequency == 0);
    ASSERT_STATUS_OK(evoke_term_live_document_frequency(
        &index,
        &stats,
        terms,
        2,
        retired_ids,
        2,
        &live_document_frequency
    ));
    ASSERT_TRUE(live_document_frequency == 5);
    ASSERT_STATUS_OK(evoke_scores_from_weighted_ids_mixed_retired(
        &index,
        &stats,
        terms,
        3,
        retired_ids,
        2,
        query_ids,
        query_weights,
        3,
        NULL,
        &scores
    ));
    ASSERT_STATUS_OK(evoke_topk(scores, 32, 7, true, &expected));
    ASSERT_STATUS_OK(
        evoke_topk_from_weighted_ids_mixed_retired_blockmax(
            &index,
            &stats,
            terms,
            3,
            document_blocks,
            4,
            3,
            retired_ids,
            2,
            query_ids,
            query_weights,
            3,
            7,
            &actual,
            &blockmax_stats
        )
    );
    ASSERT_TRUE(actual.len == expected.len);
    for (size_t result_index = 0;
         result_index < expected.len;
         result_index++)
    {
        ASSERT_TRUE(actual.doc_ids[result_index] ==
            expected.doc_ids[result_index]);
        assert_float_close(
            actual.scores[result_index],
            expected.scores[result_index]
        );
    }
    ASSERT_TRUE(blockmax_stats.blocks_considered == 4);
    ASSERT_TRUE(blockmax_stats.blocks_scored +
        blockmax_stats.blocks_skipped == 4);
    ASSERT_TRUE(blockmax_stats.postings_scored <=
        extents[0].len + extents[1].len +
        extents[2].len + extents[3].len);

    {
        uint8_t allowed_document_bitmap[4] = {0};
        float filtered_scores[32];
        evoke_topk_result filtered_expected = {0};
        evoke_topk_result filtered_actual = {0};
        evoke_blockmax_stats filtered_stats = {0};

        allowed_document_bitmap[0] =
            (uint8_t) ((UINT8_C(1) << 0) |
                       (UINT8_C(1) << 1) |
                       (UINT8_C(1) << 4));
        for (uint32_t document_id = 0; document_id < 32; document_id++)
        {
            filtered_scores[document_id] =
                (allowed_document_bitmap[document_id >> 3] &
                 (uint8_t) (UINT8_C(1) << (document_id & 7))) != 0
                    ? scores[document_id]
                    : -INFINITY;
        }
        ASSERT_STATUS_OK(evoke_topk(
            filtered_scores,
            32,
            3,
            true,
            &filtered_expected
        ));
        ASSERT_STATUS_OK(
            evoke_topk_from_weighted_ids_mixed_retired_blockmax_filtered_with_tie_breaks(
                &index,
                &stats,
                terms,
                3,
                document_blocks,
                4,
                3,
                retired_ids,
                2,
                NULL,
                NULL,
                0,
                allowed_document_bitmap,
                3,
                query_ids,
                query_weights,
                3,
                3,
                &filtered_actual,
                &filtered_stats
            )
        );
        ASSERT_TRUE(filtered_actual.len == filtered_expected.len);
        for (size_t result_index = 0;
             result_index < filtered_expected.len;
             result_index++)
        {
            ASSERT_TRUE(filtered_actual.doc_ids[result_index] ==
                filtered_expected.doc_ids[result_index]);
            assert_float_close(
                filtered_actual.scores[result_index],
                filtered_expected.scores[result_index]
            );
        }
        ASSERT_TRUE(filtered_stats.blocks_considered == 1);
        ASSERT_TRUE(filtered_stats.blocks_scored == 1);
        ASSERT_TRUE(filtered_stats.blocks_skipped == 0);
        evoke_topk_result_free(&filtered_actual);
        evoke_topk_result_free(&filtered_expected);
    }

    {
        uint32_t semantic_query_id = 1;
        float semantic_query_weight = 1.0f;
        float *pruned_scores = NULL;
        evoke_topk_result pruned_expected = {0};
        evoke_topk_result pruned_actual = {0};
        evoke_blockmax_stats pruned_stats = {0};

        ASSERT_STATUS_OK(
            evoke_scores_from_weighted_ids_mixed_retired(
                &index,
                &stats,
                terms,
                3,
                retired_ids,
                2,
                &semantic_query_id,
                &semantic_query_weight,
                1,
                NULL,
                &pruned_scores
            )
        );
        ASSERT_STATUS_OK(evoke_topk(
            pruned_scores,
            32,
            1,
            true,
            &pruned_expected
        ));
        ASSERT_STATUS_OK(
            evoke_topk_from_weighted_ids_mixed_retired_blockmax(
                &index,
                &stats,
                terms,
                3,
                document_blocks,
                4,
                3,
                retired_ids,
                2,
                &semantic_query_id,
                &semantic_query_weight,
                1,
                1,
                &pruned_actual,
                &pruned_stats
            )
        );
        ASSERT_TRUE(pruned_actual.len == pruned_expected.len);
        ASSERT_TRUE(pruned_actual.doc_ids[0] ==
            pruned_expected.doc_ids[0]);
        assert_float_close(
            pruned_actual.scores[0],
            pruned_expected.scores[0]
        );
        ASSERT_TRUE(pruned_stats.blocks_scored > 0);
        ASSERT_TRUE(pruned_stats.blocks_skipped > 0);
        ASSERT_TRUE(pruned_stats.blocks_scored +
            pruned_stats.blocks_skipped == 4);

        free(pruned_scores);
        evoke_topk_result_free(&pruned_actual);
        evoke_topk_result_free(&pruned_expected);
    }

    {
        const evoke_posting_block_record *saved_blocks =
            extents[0].blocks;
        uint32_t saved_block_count = extents[0].block_count;
        uint32_t saved_block_shift = extents[0].block_shift;

        extents[0].blocks = NULL;
        extents[0].block_count = 0;
        extents[0].block_shift = 0;
        evoke_topk_result_free(&actual);
        ASSERT_TRUE(
            evoke_topk_from_weighted_ids_mixed_retired_blockmax(
                &index,
                &stats,
                terms,
                3,
                document_blocks,
                4,
                3,
                retired_ids,
                2,
                query_ids,
                query_weights,
                3,
                7,
                &actual,
                NULL
            ) == EVOKE_ERR_INVALID
        );
        ASSERT_TRUE(actual.len == 0);
        extents[0].blocks = saved_blocks;
        extents[0].block_count = saved_block_count;
        extents[0].block_shift = saved_block_shift;
    }

    document_blocks[0].min_document_length =
        document_blocks[0].max_document_length + 1;
    ASSERT_TRUE(evoke_topk_from_weighted_ids_mixed_retired_blockmax(
        &index,
        &stats,
        terms,
        3,
        document_blocks,
        4,
        3,
        retired_ids,
        2,
        query_ids,
        query_weights,
        3,
        7,
        &actual,
        NULL
    ) == EVOKE_ERR_FORMAT);
    document_blocks[0].min_document_length = 3;

    free(scores);
    evoke_topk_result_free(&actual);
    evoke_topk_result_free(&expected);
    for (size_t extent_index = 0; extent_index < 4; extent_index++)
    {
        evoke_posting_block_records_free(block_records[extent_index]);
    }
}

static void
test_neutral_segment_scoring_survives_statistics_change(void)
{
    uint32_t base_doc0[] = {0, 0, 1};
    uint32_t base_doc1[] = {1, 2};
    uint32_t base_doc2[] = {0, 2, 2};
    uint32_t base_doc3[] = {3};
    uint32_t added_doc[] = {0, 3, 3};
    uint32_t query[] = {0, 2, 3};
    evoke_doc_ids base_docs[] = {
        make_doc(base_doc0, 3),
        make_doc(base_doc1, 2),
        make_doc(base_doc2, 3),
        make_doc(base_doc3, 1)
    };
    evoke_doc_ids delta_docs[] = {
        make_doc(added_doc, 3)
    };
    evoke_doc_ids expanded_docs[] = {
        make_doc(base_doc0, 3),
        make_doc(base_doc1, 2),
        make_doc(base_doc2, 3),
        make_doc(base_doc3, 1),
        make_doc(added_doc, 3)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index base;
    evoke_index delta;
    evoke_index expanded;
    evoke_posting_extent extents[8];
    evoke_term_extent_list terms[4];
    evoke_corpus_stats stats;
    float *expected_scores = NULL;
    float *segmented_scores = NULL;
    size_t extent_cursor = 0;
    uint32_t term_id;
    uint64_t total_document_length = 0;

    evoke_index_init(&base);
    evoke_index_init(&delta);
    evoke_index_init(&expanded);
    memset(extents, 0, sizeof(extents));
    memset(terms, 0, sizeof(terms));

    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        base_docs,
        4,
        &params,
        false,
        &base
    ));
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        delta_docs,
        1,
        &params,
        false,
        &delta
    ));
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        expanded_docs,
        5,
        &params,
        false,
        &expanded
    ));
    ASSERT_TRUE(base.vocab_size == 4);
    ASSERT_TRUE(delta.vocab_size == 4);
    ASSERT_TRUE(expanded.vocab_size == 4);
    ASSERT_TRUE(delta.data_len == 2);

    for (term_id = 0; term_id < expanded.vocab_size; term_id++)
    {
        uint64_t base_start = base.indptr[term_id];
        uint64_t base_end = base.indptr[term_id + 1];
        uint64_t delta_start = delta.indptr[term_id];
        uint64_t delta_end = delta.indptr[term_id + 1];

        terms[term_id].extents = &extents[extent_cursor];
        if (base_end > base_start)
        {
            evoke_posting_extent *extent = &extents[extent_cursor++];

            extent->indices = &base.indices[base_start];
            extent->term_frequencies = &base.term_frequencies[base_start];
            extent->len = base_end - base_start;
            extent->local_document_count = base.num_docs;
            terms[term_id].len++;
        }
        if (delta_end > delta_start)
        {
            evoke_posting_extent *extent = &extents[extent_cursor++];

            extent->indices = &delta.indices[delta_start];
            extent->term_frequencies =
                &delta.term_frequencies[delta_start];
            extent->len = delta_end - delta_start;
            extent->document_id_base = base.num_docs;
            extent->local_document_count = delta.num_docs;
            terms[term_id].len++;
        }
    }

    for (term_id = 0; term_id < expanded.num_docs; term_id++)
    {
        total_document_length += expanded.doc_lengths[term_id];
    }
    stats.document_count = expanded.num_docs;
    stats.total_document_length = total_document_length;
    stats.doc_frequencies = expanded.doc_frequencies;
    stats.vocab_size = expanded.vocab_size;

    ASSERT_STATUS_OK(evoke_scores_from_ids_exact_stats(
        &expanded,
        query,
        3,
        NULL,
        &expected_scores
    ));
    ASSERT_STATUS_OK(evoke_scores_from_ids_neutral(
        &expanded,
        &stats,
        terms,
        expanded.vocab_size,
        query,
        3,
        NULL,
        &segmented_scores
    ));
    ASSERT_TRUE(memcmp(
        expected_scores,
        segmented_scores,
        expanded.num_docs * sizeof(*expected_scores)
    ) == 0);

    free(expected_scores);
    free(segmented_scores);
    evoke_index_free(&base);
    evoke_index_free(&delta);
    evoke_index_free(&expanded);
}

static void
test_neutral_scoring_matches_all_methods(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    uint32_t query[] = {0, 2, 3};
    float weight_mask[] = {1.0f, 0.75f, 1.0f, 0.5f};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_method methods[] = {
        EVOKE_METHOD_ROBERTSON,
        EVOKE_METHOD_LUCENE,
        EVOKE_METHOD_ATIRE,
        EVOKE_METHOD_BM25L,
        EVOKE_METHOD_BM25PLUS
    };
    size_t method_index;

    for (method_index = 0;
         method_index < sizeof(methods) / sizeof(methods[0]);
         method_index++)
    {
        evoke_params params = {
            .k1 = 1.5f,
            .b = 0.75f,
            .delta = 0.5f,
            .method = methods[method_index],
            .idf_method = methods[method_index]
        };
        evoke_corpus_stats stats;
        evoke_index index;
        float *impact_scores = NULL;
        float *neutral_scores = NULL;
        uint32_t doc_id;

        evoke_index_init(&index);
        ASSERT_STATUS_OK(evoke_build_index_from_ids(
            docs,
            4,
            &params,
            false,
            &index
        ));

        memset(&stats, 0, sizeof(stats));
        stats.document_count = index.num_docs;
        stats.doc_frequencies = index.doc_frequencies;
        stats.vocab_size = index.vocab_size;
        for (doc_id = 0; doc_id < index.num_docs; doc_id++)
        {
            stats.total_document_length += index.doc_lengths[doc_id];
        }

        ASSERT_STATUS_OK(evoke_scores_from_ids(
            &index,
            query,
            3,
            weight_mask,
            &impact_scores
        ));
        ASSERT_STATUS_OK(evoke_scores_from_ids_neutral(
            &index,
            &stats,
            NULL,
            0,
            query,
            3,
            weight_mask,
            &neutral_scores
        ));
        assert_float_array(
            neutral_scores,
            impact_scores,
            index.num_docs
        );

        free(impact_scores);
        free(neutral_scores);
        evoke_index_free(&index);
    }
}

static void
test_serialization_roundtrip(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    uint32_t doc2[] = {0, 2, 2};
    uint32_t doc3[] = {3};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2),
        make_doc(doc2, 3),
        make_doc(doc3, 1)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_BM25L,
        .idf_method = EVOKE_METHOD_BM25L
    };
    evoke_index original;
    evoke_index restored;
    uint8_t *bytes = NULL;
    size_t len = 0;
    uint32_t query[] = {0, 2};
    float *scores_a = NULL;
    float *scores_b = NULL;

    evoke_index_init(&original);
    evoke_index_init(&restored);
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        4,
        &params,
        true,
        &original
    ));
    ASSERT_STATUS_OK(evoke_serialize_index(&original, &bytes, &len));
    ASSERT_STATUS_OK(evoke_deserialize_index(bytes, len, &restored));

    ASSERT_TRUE(restored.num_docs == original.num_docs);
    ASSERT_TRUE(restored.vocab_size == original.vocab_size);
    ASSERT_TRUE(restored.data_len == original.data_len);
    assert_float_array(restored.data, original.data, original.data_len);
    assert_uint32_array(restored.indices, original.indices, original.data_len);
    assert_uint64_array(restored.indptr, original.indptr, original.vocab_size + 1);
    assert_uint32_array_present(
        restored.term_frequencies,
        original.term_frequencies,
        original.data_len
    );
    assert_uint32_array_present(
        restored.doc_lengths,
        original.doc_lengths,
        original.num_docs
    );
    assert_uint32_array_present(
        restored.doc_frequencies,
        original.doc_frequencies,
        original.vocab_size
    );
    assert_float_array(
        restored.nonoccurrence,
        original.nonoccurrence,
        original.vocab_size
    );

    ASSERT_STATUS_OK(evoke_scores_from_ids(
        &original,
        query,
        2,
        NULL,
        &scores_a
    ));
    ASSERT_STATUS_OK(evoke_scores_from_ids(
        &restored,
        query,
        2,
        NULL,
        &scores_b
    ));
    assert_float_array(scores_a, scores_b, original.num_docs);

    free(bytes);
    free(scores_a);
    free(scores_b);
    evoke_index_free(&original);
    evoke_index_free(&restored);
}

static void
test_storage_version_boundary(void)
{
    uint32_t doc0[] = {0, 0, 1};
    uint32_t doc1[] = {1, 2};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 3),
        make_doc(doc1, 2)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index original;
    evoke_index restored;
    uint8_t *current = NULL;
    size_t current_len = 0;

    evoke_index_init(&original);
    evoke_index_init(&restored);
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        2,
        &params,
        false,
        &original
    ));
    ASSERT_STATUS_OK(evoke_serialize_index(
        &original,
        &current,
        &current_len
    ));
    ASSERT_TRUE(current[4] == EVOKE_STORAGE_CURRENT_VERSION);
    ASSERT_TRUE(current[5] == 0);
    ASSERT_TRUE((current[6] & 0x08U) != 0);

    current[4] = 1;
    current[5] = 0;
    ASSERT_TRUE(evoke_deserialize_index(
        current,
        current_len,
        &restored
    ) == EVOKE_ERR_FORMAT);

    current[4] = 3;
    ASSERT_TRUE(evoke_deserialize_index(
        current,
        current_len,
        &restored
    ) == EVOKE_ERR_FORMAT);

    free(current);
    evoke_index_free(&original);
}

typedef struct test_stream_buffer
{
    uint8_t *bytes;
    size_t len;
    size_t capacity;
} test_stream_buffer;

static evoke_status
test_stream_write(void *ctx, const uint8_t *bytes, size_t len)
{
    test_stream_buffer *buffer = ctx;

    if (buffer->len + len > buffer->capacity)
    {
        size_t new_capacity = buffer->capacity == 0
            ? 128
            : buffer->capacity * 2;
        uint8_t *new_bytes;

        while (buffer->len + len > new_capacity)
        {
            new_capacity *= 2;
        }
        new_bytes = realloc(buffer->bytes, new_capacity);
        if (new_bytes == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        buffer->bytes = new_bytes;
        buffer->capacity = new_capacity;
    }
    memcpy(buffer->bytes + buffer->len, bytes, len);
    buffer->len += len;
    return EVOKE_OK;
}

static void
test_stream_serialization_matches_buffered(void)
{
    const char *doc0[] = {"alpha", "alpha", "beta"};
    const char *doc1[] = {"beta", "gamma"};
    const char *doc2[] = {"alpha", "delta", "gamma"};
    evoke_doc_tokens docs[] = {
        {doc0, 3},
        {doc1, 2},
        {doc2, 3}
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index index;
    test_stream_buffer streamed = {0};
    uint8_t *buffered = NULL;
    size_t buffered_len = 0;
    size_t streamed_len = 0;

    evoke_index_init(&index);
    ASSERT_STATUS_OK(evoke_build_index_from_tokens(
        docs,
        3,
        &params,
        &index
    ));
    ASSERT_STATUS_OK(evoke_serialize_index(
        &index,
        &buffered,
        &buffered_len
    ));
    ASSERT_STATUS_OK(evoke_serialize_index_stream(
        &index,
        test_stream_write,
        &streamed,
        &streamed_len
    ));
    ASSERT_TRUE(streamed_len == buffered_len);
    ASSERT_TRUE(streamed.len == buffered_len);
    ASSERT_TRUE(memcmp(streamed.bytes, buffered, buffered_len) == 0);

    free(streamed.bytes);
    free(buffered);
    evoke_index_free(&index);
}

static void
test_deserialize_rejects_corrupt_postings(void)
{
    uint32_t doc0[] = {0, 1};
    uint32_t doc1[] = {1, 2};
    evoke_doc_ids docs[] = {
        make_doc(doc0, 2),
        make_doc(doc1, 2)
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index original;
    evoke_index restored;
    uint8_t *bytes = NULL;
    size_t len = 0;
    size_t indices_offset;
    size_t indptr_offset;
    size_t final_indptr_offset;
    const size_t header_size = EVOKE_HEADER_SIZE;

    evoke_index_init(&original);
    evoke_index_init(&restored);
    ASSERT_STATUS_OK(evoke_build_index_from_ids(
        docs,
        2,
        &params,
        false,
        &original
    ));
    ASSERT_STATUS_OK(evoke_serialize_index(&original, &bytes, &len));

    indices_offset = header_size + (size_t) original.data_len * sizeof(float);
    indptr_offset = indices_offset + (size_t) original.data_len * sizeof(uint32_t);
    final_indptr_offset = indptr_offset + (size_t) original.vocab_size * sizeof(uint64_t);

    ASSERT_TRUE(final_indptr_offset + sizeof(uint64_t) <= len);
    ASSERT_TRUE(indices_offset + sizeof(uint32_t) <= len);

    write_u64_le(bytes + final_indptr_offset, original.data_len + 1);
    ASSERT_TRUE(evoke_deserialize_index(bytes, len, &restored) ==
                EVOKE_ERR_FORMAT);

    free(bytes);
    bytes = NULL;
    ASSERT_STATUS_OK(evoke_serialize_index(&original, &bytes, &len));
    write_u32_le(bytes + indices_offset, original.num_docs);
    ASSERT_TRUE(evoke_deserialize_index(bytes, len, &restored) ==
                EVOKE_ERR_FORMAT);

    free(bytes);
    evoke_index_free(&original);
    evoke_index_free(&restored);
}

static void
test_serialization_rejects_inconsistent_index(void)
{
    const char *doc0[] = {"alpha", "beta"};
    const char *doc1[] = {"beta", "gamma"};
    evoke_doc_tokens docs[] = {
        {doc0, 2},
        {doc1, 2}
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    evoke_index index;
    evoke_index malformed;
    float saved_score;
    uint32_t saved_frequency;
    char *saved_vocab;
    size_t len = 0;

    evoke_index_init(&index);
    ASSERT_STATUS_OK(evoke_build_index_from_tokens(
        docs,
        2,
        &params,
        &index
    ));

    malformed = index;
    malformed.data = NULL;
    ASSERT_TRUE(evoke_serialized_index_size(
        &malformed,
        &len
    ) == EVOKE_ERR_INVALID);

    malformed = index;
    malformed.indices = NULL;
    ASSERT_TRUE(evoke_serialized_index_size(
        &malformed,
        &len
    ) == EVOKE_ERR_INVALID);

    malformed = index;
    malformed.indptr = NULL;
    ASSERT_TRUE(evoke_serialized_index_size(
        &malformed,
        &len
    ) == EVOKE_ERR_INVALID);

    malformed = index;
    malformed.term_frequencies = NULL;
    ASSERT_TRUE(evoke_serialized_index_size(
        &malformed,
        &len
    ) == EVOKE_ERR_INVALID);

    saved_score = index.data[0];
    index.data[0] = NAN;
    ASSERT_TRUE(evoke_serialized_index_size(
        &index,
        &len
    ) == EVOKE_ERR_FORMAT);
    index.data[0] = saved_score;

    saved_frequency = index.doc_frequencies[0];
    index.doc_frequencies[0]++;
    ASSERT_TRUE(evoke_serialized_index_size(
        &index,
        &len
    ) == EVOKE_ERR_FORMAT);
    index.doc_frequencies[0] = saved_frequency;

    saved_vocab = index.vocab[0];
    index.vocab[0] = NULL;
    ASSERT_TRUE(evoke_serialized_index_size(
        &index,
        &len
    ) == EVOKE_ERR_INVALID);
    index.vocab[0] = saved_vocab;

    malformed = index;
    malformed.data_len = UINT32_MAX;
    ASSERT_TRUE(evoke_serialized_index_size(
        &malformed,
        &len
    ) == EVOKE_ERR_FORMAT);

    evoke_index_free(&index);
}

static void
test_deserialize_mutation_safety(void)
{
    const char *doc0[] = {"alpha", "beta"};
    const char *doc1[] = {"beta", "gamma"};
    evoke_doc_tokens docs[] = {
        {doc0, 2},
        {doc1, 2}
    };
    evoke_params params = {
        .k1 = 1.5f,
        .b = 0.75f,
        .delta = 0.5f,
        .method = EVOKE_METHOD_LUCENE,
        .idf_method = EVOKE_METHOD_LUCENE
    };
    const uint8_t masks[] = {0x01U, 0x80U, 0xFFU};
    evoke_index original;
    evoke_index restored;
    uint8_t *bytes = NULL;
    uint8_t *mutated = NULL;
    size_t len = 0;
    size_t i;
    size_t j;

    evoke_index_init(&original);
    ASSERT_STATUS_OK(evoke_build_index_from_tokens(
        docs,
        2,
        &params,
        &original
    ));
    ASSERT_STATUS_OK(evoke_serialize_index(&original, &bytes, &len));

    for (i = 0; i < len; i++)
    {
        evoke_index_init(&restored);
        ASSERT_TRUE(evoke_deserialize_index(
            bytes,
            i,
            &restored
        ) == EVOKE_ERR_FORMAT);
        evoke_index_free(&restored);
    }

    mutated = malloc(len + 1);
    ASSERT_TRUE(mutated != NULL);
    memcpy(mutated, bytes, len);
    mutated[len] = 0;
    evoke_index_init(&restored);
    ASSERT_TRUE(evoke_deserialize_index(
        mutated,
        len + 1,
        &restored
    ) == EVOKE_ERR_FORMAT);
    evoke_index_free(&restored);

    for (i = 0; i < len; i++)
    {
        for (j = 0; j < sizeof(masks); j++)
        {
            evoke_status status;

            memcpy(mutated, bytes, len);
            mutated[i] ^= masks[j];
            evoke_index_init(&restored);
            status = evoke_deserialize_index(mutated, len, &restored);
            ASSERT_TRUE(status == EVOKE_OK || status == EVOKE_ERR_FORMAT);
            evoke_index_free(&restored);
        }
    }

    memcpy(mutated, bytes, len);
    write_u32_le(mutated + 16, 0x7FC00000U);
    evoke_index_init(&restored);
    ASSERT_TRUE(evoke_deserialize_index(
        mutated,
        len,
        &restored
    ) == EVOKE_ERR_FORMAT);
    evoke_index_free(&restored);

    memcpy(mutated, bytes, len);
    mutated[len - 1] = '\0';
    evoke_index_init(&restored);
    ASSERT_TRUE(evoke_deserialize_index(
        mutated,
        len,
        &restored
    ) == EVOKE_ERR_FORMAT);
    evoke_index_free(&restored);

    free(mutated);
    free(bytes);
    evoke_index_free(&original);
}

static void
test_topk_numpy_compatible_shape(void)
{
    float scores[] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    evoke_topk_result sorted_result;
    evoke_topk_result unsorted_result;
    uint32_t expected_ids[] = {1, 4, 2};
    float expected_scores[] = {5.0f, 4.0f, 3.0f};
    size_t i;

    memset(&sorted_result, 0, sizeof(sorted_result));
    memset(&unsorted_result, 0, sizeof(unsorted_result));

    ASSERT_STATUS_OK(evoke_topk(scores, 5, 3, true, &sorted_result));
    assert_uint32_array(sorted_result.doc_ids, expected_ids, 3);
    assert_float_array(sorted_result.scores, expected_scores, 3);

    ASSERT_STATUS_OK(evoke_topk(scores, 5, 3, false, &unsorted_result));
    ASSERT_TRUE(unsorted_result.len == 3);
    for (i = 0; i < 3; i++)
    {
        bool seen = false;
        size_t j;

        for (j = 0; j < 3; j++)
        {
            if (unsorted_result.doc_ids[i] == expected_ids[j])
            {
                seen = true;
            }
        }
        ASSERT_TRUE(seen);
    }

    evoke_topk_result_free(&sorted_result);
    evoke_topk_result_free(&unsorted_result);
}

static void
test_topk_subset_buffered_keeps_stable_threshold(void)
{
    float scores[] = {100.0f, 50.0f, 40.0f, 60.0f, 55.0f, 10.0f};
    uint32_t candidate_doc_ids[] = {0, 1, 2, 3, 4, 5};
    evoke_topk_result result;
    uint32_t expected_ids[] = {0, 3, 4};
    float expected_scores[] = {100.0f, 60.0f, 55.0f};

    memset(&result, 0, sizeof(result));

    ASSERT_STATUS_OK(evoke_topk_subset(
        scores,
        candidate_doc_ids,
        6,
        3,
        true,
        false,
        &result
    ));

    ASSERT_TRUE(result.len == 3);
    assert_uint32_array(result.doc_ids, expected_ids, 3);
    assert_float_array(result.scores, expected_scores, 3);

    evoke_topk_result_free(&result);
}

static void
test_topk_subset_compact_matches_identity_tie_breaks(void)
{
    float scores[] = {1.0f, 3.0f, 3.0f, -1.0f, 2.0f, 3.0f, 0.0f};
    uint32_t candidate_doc_ids[] = {0, 1, 2, 3, 4, 5, 6};
    uint64_t identity_tie_break_keys[] = {0, 1, 2, 3, 4, 5, 6};
    uint32_t expected_ids[] = {1, 2, 5, 4};
    evoke_topk_result compact_result = {0};
    evoke_topk_result keyed_result = {0};

    ASSERT_STATUS_OK(evoke_topk_subset_with_tie_breaks(
        scores,
        candidate_doc_ids,
        7,
        4,
        true,
        true,
        NULL,
        &compact_result
    ));
    ASSERT_STATUS_OK(evoke_topk_subset_with_tie_breaks(
        scores,
        candidate_doc_ids,
        7,
        4,
        true,
        true,
        identity_tie_break_keys,
        &keyed_result
    ));

    ASSERT_TRUE(compact_result.len == 4);
    ASSERT_TRUE(keyed_result.len == compact_result.len);
    assert_uint32_array(compact_result.doc_ids, expected_ids, 4);
    assert_uint32_array(
        keyed_result.doc_ids,
        compact_result.doc_ids,
        compact_result.len
    );
    assert_float_array(
        keyed_result.scores,
        compact_result.scores,
        compact_result.len
    );

    evoke_topk_result_free(&compact_result);
    evoke_topk_result_free(&keyed_result);
}

static void
test_topk_custom_tie_breaks(void)
{
    float scores[] = {1.0f, 1.0f, 1.0f, 0.5f, 1.0f};
    uint64_t tie_break_keys[] = {30, 10, 20, 50, 40};
    uint32_t candidate_doc_ids[] = {4, 2, 0, 1};
    uint32_t expected_ids[] = {1, 2, 0};
    uint32_t expected_subset_ids[] = {1, 2};
    evoke_topk_result result = {0};

    ASSERT_STATUS_OK(evoke_topk_with_tie_breaks(
        scores,
        5,
        3,
        true,
        tie_break_keys,
        &result
    ));
    ASSERT_TRUE(result.len == 3);
    assert_uint32_array(result.doc_ids, expected_ids, 3);
    evoke_topk_result_free(&result);

    ASSERT_STATUS_OK(evoke_topk_subset_with_tie_breaks(
        scores,
        candidate_doc_ids,
        4,
        2,
        true,
        false,
        tie_break_keys,
        &result
    ));
    ASSERT_TRUE(result.len == 2);
    assert_uint32_array(result.doc_ids, expected_subset_ids, 2);
    evoke_topk_result_free(&result);
}

static void
test_streaming_topk_matches_array_tie_breaks(void)
{
    float scores[] = {1.0f, 1.0f, 1.0f, 0.5f, 1.0f};
    uint64_t tie_break_keys[] = {30, 10, 20, 50, 40};
    uint32_t expected_ids[] = {1, 2, 0};
    evoke_topk_accumulator accumulator;
    evoke_topk_result result = {0};

    ASSERT_STATUS_OK(evoke_topk_accumulator_init(&accumulator, 3));
    for (uint32_t doc_id = 0; doc_id < 5; doc_id++)
    {
        ASSERT_STATUS_OK(evoke_topk_accumulator_offer(
            &accumulator,
            scores[doc_id],
            doc_id,
            tie_break_keys[doc_id]
        ));
    }
    ASSERT_STATUS_OK(evoke_topk_accumulator_finish(
        &accumulator,
        true,
        &result
    ));
    ASSERT_TRUE(result.len == 3);
    assert_uint32_array(result.doc_ids, expected_ids, 3);
    ASSERT_TRUE(evoke_topk_accumulator_offer(
        &accumulator,
        2.0f,
        5,
        0
    ) == EVOKE_ERR_INVALID);
    evoke_topk_result_free(&result);
    evoke_topk_accumulator_free(&accumulator);
}

static void
test_blockmax_zero_score_uses_ordered_prefix(void)
{
    uint32_t document_lengths[] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t document_frequencies[] = {0};
    uint64_t tie_break_keys[] = {80, 10, 70, 20, 60, 30, 50, 40};
    uint32_t tie_break_order[] = {1, 3, 5, 7, 6, 4, 2, 0};
    uint32_t expected_ids[] = {1, 3, 5};
    uint32_t unknown_query_id = 1;
    evoke_term_extent_list terms[1] = {{0}};
    evoke_document_block_extrema document_blocks[] = {
        {
            .document_count = 8,
            .min_document_length = 1,
            .max_document_length = 1,
        },
    };
    evoke_index index = {
        .num_docs = 8,
        .vocab_size = 1,
        .doc_lengths = document_lengths,
    };
    evoke_corpus_stats stats = {
        .document_count = 8,
        .total_document_length = 8,
        .doc_frequencies = document_frequencies,
        .vocab_size = 1,
    };
    evoke_topk_result result = {0};
    evoke_blockmax_stats blockmax_stats = {0};

    ASSERT_STATUS_OK(
        evoke_topk_from_weighted_ids_mixed_retired_blockmax_with_tie_breaks(
            &index,
            &stats,
            terms,
            1,
            document_blocks,
            1,
            3,
            NULL,
            0,
            tie_break_keys,
            tie_break_order,
            8,
            &unknown_query_id,
            NULL,
            1,
            3,
            &result,
            &blockmax_stats
        )
    );
    ASSERT_TRUE(result.len == 3);
    assert_uint32_array(result.doc_ids, expected_ids, 3);
    ASSERT_TRUE(blockmax_stats.zero_score_documents_considered == 3);
    evoke_topk_result_free(&result);

    ASSERT_STATUS_OK(
        evoke_topk_from_weighted_ids_mixed_retired_blockmax_with_tie_breaks(
            &index,
            &stats,
            terms,
            1,
            document_blocks,
            1,
            3,
            NULL,
            0,
            tie_break_keys,
            NULL,
            0,
            &unknown_query_id,
            NULL,
            1,
            3,
            &result,
            &blockmax_stats
        )
    );
    ASSERT_TRUE(result.len == 3);
    assert_uint32_array(result.doc_ids, expected_ids, 3);
    ASSERT_TRUE(blockmax_stats.zero_score_documents_considered == 8);
    evoke_topk_result_free(&result);

    {
        uint8_t allowed_document_bitmap[] = {
            (uint8_t) ((UINT8_C(1) << 0) |
                       (UINT8_C(1) << 2) |
                       (UINT8_C(1) << 4) |
                       (UINT8_C(1) << 6))
        };
        uint32_t filtered_expected_ids[] = {6, 4, 2};

        ASSERT_STATUS_OK(
            evoke_topk_from_weighted_ids_mixed_retired_blockmax_filtered_with_tie_breaks(
                &index,
                &stats,
                terms,
                1,
                document_blocks,
                1,
                3,
                NULL,
                0,
                tie_break_keys,
                tie_break_order,
                8,
                allowed_document_bitmap,
                4,
                &unknown_query_id,
                NULL,
                1,
                3,
                &result,
                &blockmax_stats
            )
        );
        ASSERT_TRUE(result.len == 3);
        assert_uint32_array(result.doc_ids, filtered_expected_ids, 3);
        ASSERT_TRUE(blockmax_stats.zero_score_documents_considered == 3);
        evoke_topk_result_free(&result);
    }
}

static void
test_query_parser_basic_terms(void)
{
    evoke_query query;

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string("cat bird", &query));

    ASSERT_TRUE(query.len == 2);
    ASSERT_TRUE(query.terms[0].kind == EVOKE_QUERY_TERM);
    ASSERT_TRUE(query.terms[0].occur == EVOKE_QUERY_SHOULD);
    ASSERT_TRUE(query.terms[0].len == 1);
    ASSERT_TRUE(strcmp(query.terms[0].tokens[0], "cat") == 0);
    ASSERT_TRUE(query.terms[1].kind == EVOKE_QUERY_TERM);
    ASSERT_TRUE(query.terms[1].occur == EVOKE_QUERY_SHOULD);
    ASSERT_TRUE(strcmp(query.terms[1].tokens[0], "bird") == 0);

    evoke_query_free(&query);
}

static void
test_query_parser_occurs_prefix_and_phrase(void)
{
    evoke_query query;

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string(
        "+must -omit pref* \"exact phrase here\"",
        &query
    ));

    ASSERT_TRUE(query.len == 4);

    ASSERT_TRUE(query.terms[0].occur == EVOKE_QUERY_MUST);
    ASSERT_TRUE(query.terms[0].kind == EVOKE_QUERY_TERM);
    ASSERT_TRUE(strcmp(query.terms[0].tokens[0], "must") == 0);

    ASSERT_TRUE(query.terms[1].occur == EVOKE_QUERY_MUST_NOT);
    ASSERT_TRUE(query.terms[1].kind == EVOKE_QUERY_TERM);
    ASSERT_TRUE(strcmp(query.terms[1].tokens[0], "omit") == 0);

    ASSERT_TRUE(query.terms[2].occur == EVOKE_QUERY_SHOULD);
    ASSERT_TRUE(query.terms[2].kind == EVOKE_QUERY_PREFIX);
    ASSERT_TRUE(strcmp(query.terms[2].tokens[0], "pref") == 0);

    ASSERT_TRUE(query.terms[3].occur == EVOKE_QUERY_SHOULD);
    ASSERT_TRUE(query.terms[3].kind == EVOKE_QUERY_PHRASE);
    ASSERT_TRUE(query.terms[3].len == 3);
    ASSERT_TRUE(strcmp(query.terms[3].tokens[0], "exact") == 0);
    ASSERT_TRUE(strcmp(query.terms[3].tokens[1], "phrase") == 0);
    ASSERT_TRUE(strcmp(query.terms[3].tokens[2], "here") == 0);

    evoke_query_free(&query);
}

static void
test_query_parser_textual_boolean_aliases(void)
{
    evoke_query query;

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string(
        "cat OR bird AND fish",
        &query
    ));

    ASSERT_TRUE(evoke_query_uses_boolean_ast(&query));
    ASSERT_TRUE(query.len == 3);
    ASSERT_TRUE(query.root != NULL);
    ASSERT_TRUE(query.root->kind == EVOKE_QUERY_NODE_OR);
    ASSERT_TRUE(query.root->left != NULL);
    ASSERT_TRUE(query.root->left->kind == EVOKE_QUERY_NODE_TERM);
    ASSERT_TRUE(query.root->right != NULL);
    ASSERT_TRUE(query.root->right->kind == EVOKE_QUERY_NODE_AND);
    ASSERT_TRUE(strcmp(query.terms[0].tokens[0], "cat") == 0);
    ASSERT_TRUE(strcmp(query.terms[1].tokens[0], "bird") == 0);
    ASSERT_TRUE(strcmp(query.terms[2].tokens[0], "fish") == 0);
    evoke_query_free(&query);

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string(
        "cat AND (bird OR NOT omit)",
        &query
    ));
    ASSERT_TRUE(evoke_query_uses_boolean_ast(&query));
    ASSERT_TRUE(query.len == 3);
    ASSERT_TRUE(query.root != NULL);
    ASSERT_TRUE(query.root->kind == EVOKE_QUERY_NODE_AND);
    ASSERT_TRUE(strcmp(query.terms[0].tokens[0], "cat") == 0);
    ASSERT_TRUE(strcmp(query.terms[1].tokens[0], "bird") == 0);
    ASSERT_TRUE(strcmp(query.terms[2].tokens[0], "omit") == 0);
    evoke_query_free(&query);

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string("cat and bird", &query));
    ASSERT_TRUE(!evoke_query_uses_boolean_ast(&query));
    ASSERT_TRUE(query.len == 3);
    ASSERT_TRUE(strcmp(query.terms[1].tokens[0], "and") == 0);
    evoke_query_free(&query);
}

static void
test_query_parser_invalid_inputs(void)
{
    evoke_query query;

    evoke_query_init(&query);
    ASSERT_TRUE(evoke_parse_query_string("+", &query) ==
                EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_parse_query_string("\"unterminated", &query) ==
                EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_parse_query_string("*", &query) ==
                EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_parse_query_string("\"   \"", &query) ==
                EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_parse_query_string("cat AND", &query) ==
                EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_parse_query_string("OR cat", &query) ==
                EVOKE_ERR_INVALID);
    ASSERT_TRUE(evoke_parse_query_string("(cat OR bird", &query) ==
                EVOKE_ERR_INVALID);
    evoke_query_free(&query);
}

static void
test_query_parser_complexity_limits(void)
{
    const size_t max_nesting = (size_t) EVOKE_QUERY_MAX_NESTING;
    const size_t max_token_count = (size_t) EVOKE_QUERY_MAX_TOKENS;
    const size_t nesting = (size_t) EVOKE_QUERY_MAX_NESTING + 1;
    const size_t token_count = (size_t) EVOKE_QUERY_MAX_TOKENS + 1;
    evoke_query query;
    char *max_deep_query;
    char *max_wide_query;
    char *max_wide_phrase;
    char *deep_query;
    char *wide_query;
    char *wide_phrase;
    size_t i;

    max_deep_query = malloc(max_nesting * 2 + 2);
    max_wide_query = malloc(max_token_count * 2 + 1);
    max_wide_phrase = malloc(max_token_count * 2 + 3);
    deep_query = malloc(nesting * 2 + 2);
    wide_query = malloc(token_count * 2 + 1);
    wide_phrase = malloc(token_count * 2 + 3);
    ASSERT_TRUE(
        max_deep_query != NULL &&
        max_wide_query != NULL &&
        max_wide_phrase != NULL &&
        deep_query != NULL &&
        wide_query != NULL &&
        wide_phrase != NULL
    );

    for (i = 0; i < max_nesting; i++)
    {
        max_deep_query[i] = '(';
        max_deep_query[max_nesting + 1 + i] = ')';
    }
    max_deep_query[max_nesting] = 'x';
    max_deep_query[max_nesting * 2 + 1] = '\0';

    for (i = 0; i < max_token_count; i++)
    {
        max_wide_query[i * 2] = 'x';
        max_wide_query[i * 2 + 1] = ' ';
    }
    max_wide_query[max_token_count * 2] = '\0';

    max_wide_phrase[0] = '"';
    memcpy(max_wide_phrase + 1, max_wide_query, max_token_count * 2);
    max_wide_phrase[max_token_count * 2 + 1] = '"';
    max_wide_phrase[max_token_count * 2 + 2] = '\0';

    for (i = 0; i < nesting; i++)
    {
        deep_query[i] = '(';
        deep_query[nesting + 1 + i] = ')';
    }
    deep_query[nesting] = 'x';
    deep_query[nesting * 2 + 1] = '\0';

    for (i = 0; i < token_count; i++)
    {
        wide_query[i * 2] = 'x';
        wide_query[i * 2 + 1] = ' ';
    }
    wide_query[token_count * 2] = '\0';

    wide_phrase[0] = '"';
    memcpy(wide_phrase + 1, wide_query, token_count * 2);
    wide_phrase[token_count * 2 + 1] = '"';
    wide_phrase[token_count * 2 + 2] = '\0';

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string(max_deep_query, &query));
    evoke_query_free(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string(max_wide_query, &query));
    evoke_query_free(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string(max_wide_phrase, &query));
    evoke_query_free(&query);
    ASSERT_TRUE(evoke_parse_query_string(
        deep_query,
        &query
    ) == EVOKE_ERR_RANGE);
    ASSERT_TRUE(evoke_parse_query_string(
        wide_query,
        &query
    ) == EVOKE_ERR_RANGE);
    ASSERT_TRUE(evoke_parse_query_string(
        wide_phrase,
        &query
    ) == EVOKE_ERR_RANGE);
    evoke_query_free(&query);

    free(max_deep_query);
    free(max_wide_query);
    free(max_wide_phrase);
    free(deep_query);
    free(wide_query);
    free(wide_phrase);
}

static void
test_query_parser_simple_term_detection(void)
{
    evoke_query query;

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string("cat bird", &query));
    ASSERT_TRUE(evoke_query_is_simple_term_query(&query));
    evoke_query_free(&query);

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string("+cat", &query));
    ASSERT_TRUE(!evoke_query_is_simple_term_query(&query));
    evoke_query_free(&query);

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string("cat*", &query));
    ASSERT_TRUE(!evoke_query_is_simple_term_query(&query));
    evoke_query_free(&query);

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string("\"cat bird\"", &query));
    ASSERT_TRUE(!evoke_query_is_simple_term_query(&query));
    evoke_query_free(&query);

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string("cat OR bird", &query));
    ASSERT_TRUE(evoke_query_is_simple_term_query(&query));
    evoke_query_free(&query);
}

static void
test_text_normalize_token(void)
{
    const char *stopwords[] = {"the", "and"};
    evoke_text_options options;
    char *token = NULL;
    bool keep = false;

    evoke_text_options_init(&options);
    options.stopwords = stopwords;
    options.num_stopwords = 2;

    ASSERT_STATUS_OK(evoke_normalize_token(
        "Cat",
        &options,
        &token,
        &keep
    ));
    ASSERT_TRUE(keep);
    ASSERT_TRUE(strcmp(token, "cat") == 0);
    free(token);

    ASSERT_STATUS_OK(evoke_normalize_token(
        "The",
        &options,
        &token,
        &keep
    ));
    ASSERT_TRUE(!keep);
    ASSERT_TRUE(token == NULL);
}

static void
test_text_normalize_query(void)
{
    const char *stopwords[] = {"the"};
    evoke_text_options options;
    evoke_query query;

    evoke_text_options_init(&options);
    options.stopwords = stopwords;
    options.num_stopwords = 1;

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string(
        "+Cat \"The Bird\" -And",
        &query
    ));
    ASSERT_STATUS_OK(evoke_normalize_query(&query, &options));

    ASSERT_TRUE(query.len == 3);
    ASSERT_TRUE(strcmp(query.terms[0].tokens[0], "cat") == 0);
    ASSERT_TRUE(query.terms[1].kind == EVOKE_QUERY_PHRASE);
    ASSERT_TRUE(query.terms[1].len == 1);
    ASSERT_TRUE(strcmp(query.terms[1].tokens[0], "bird") == 0);
    ASSERT_TRUE(strcmp(query.terms[2].tokens[0], "and") == 0);

    evoke_query_free(&query);
}

static void
test_text_normalize_boolean_query(void)
{
    const char *stopwords[] = {"the"};
    evoke_text_options options;
    evoke_query query;

    evoke_text_options_init(&options);
    options.stopwords = stopwords;
    options.num_stopwords = 1;

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string(
        "Cat AND (The OR Bird)",
        &query
    ));
    ASSERT_TRUE(evoke_query_uses_boolean_ast(&query));
    ASSERT_STATUS_OK(evoke_normalize_query(&query, &options));

    ASSERT_TRUE(query.len == 2);
    ASSERT_TRUE(strcmp(query.terms[0].tokens[0], "cat") == 0);
    ASSERT_TRUE(strcmp(query.terms[1].tokens[0], "bird") == 0);
    ASSERT_TRUE(query.root != NULL);
    ASSERT_TRUE(query.root->kind == EVOKE_QUERY_NODE_AND);

    evoke_query_free(&query);
}

static void
test_text_tokenize_text(void)
{
    const char *stopwords[] = {"and"};
    evoke_text_options options;
    char **tokens = NULL;
    size_t len = 0;

    evoke_text_options_init(&options);
    options.stopwords = stopwords;
    options.num_stopwords = 1;

    ASSERT_STATUS_OK(evoke_tokenize_text(
        "Cat, and dog! \316\262eta",
        &options,
        &tokens,
        &len
    ));

    ASSERT_TRUE(len == 3);
    ASSERT_TRUE(strcmp(tokens[0], "cat") == 0);
    ASSERT_TRUE(strcmp(tokens[1], "dog") == 0);
    ASSERT_TRUE(strcmp(tokens[2], "\316\262eta") == 0);

    evoke_text_tokens_free(tokens, len);
}

static void
test_text_normalize_token_with_stemming(void)
{
    evoke_text_options options;
    char *token = NULL;
    bool keep = false;

    evoke_text_options_init(&options);
    options.lowercase = false;
    options.stem_english = true;

    ASSERT_STATUS_OK(evoke_normalize_token(
        "Running",
        &options,
        &token,
        &keep
    ));
    ASSERT_TRUE(keep);
    ASSERT_TRUE(strcmp(token, "run") == 0);
    free(token);
}

static void
test_text_normalize_query_with_stemming(void)
{
    evoke_text_options options;
    evoke_query query;

    evoke_text_options_init(&options);
    options.lowercase = false;
    options.stem_english = true;

    evoke_query_init(&query);
    ASSERT_STATUS_OK(evoke_parse_query_string(
        "Running \"Cats Running\"",
        &query
    ));
    ASSERT_STATUS_OK(evoke_normalize_query(&query, &options));

    ASSERT_TRUE(query.len == 2);
    ASSERT_TRUE(strcmp(query.terms[0].tokens[0], "run") == 0);
    ASSERT_TRUE(query.terms[1].kind == EVOKE_QUERY_PHRASE);
    ASSERT_TRUE(query.terms[1].len == 2);
    ASSERT_TRUE(strcmp(query.terms[1].tokens[0], "cat") == 0);
    ASSERT_TRUE(strcmp(query.terms[1].tokens[1], "run") == 0);

    evoke_query_free(&query);
}

static void
test_text_tokenize_text_with_stemming(void)
{
    evoke_text_options options;
    char **tokens = NULL;
    size_t len = 0;

    evoke_text_options_init(&options);
    options.stem_english = true;

    ASSERT_STATUS_OK(evoke_tokenize_text(
        "Running runs quickly",
        &options,
        &tokens,
        &len
    ));

    ASSERT_TRUE(len == 3);
    ASSERT_TRUE(strcmp(tokens[0], "run") == 0);
    ASSERT_TRUE(strcmp(tokens[1], "run") == 0);
    ASSERT_TRUE(strcmp(tokens[2], "quickli") == 0);

    evoke_text_tokens_free(tokens, len);
}

static void
test_text_normalize_token_with_diacritic_folding(void)
{
    evoke_text_options options;
    char *token = NULL;
    bool keep = false;

    evoke_text_options_init(&options);
    options.fold_diacritics = true;

    ASSERT_STATUS_OK(evoke_normalize_token(
        "Stra\303\237e",
        &options,
        &token,
        &keep
    ));
    ASSERT_TRUE(keep);
    ASSERT_TRUE(strcmp(token, "strasse") == 0);
    free(token);
}

static void
test_posting_heat_bounded_decay_and_root_stability(void)
{
    evoke_posting_heat_entry entries[4] = {0};
    evoke_posting_heat_observation observation = {
        .database_id = 11,
        .index_id = 22,
        .term_id = 7,
        .root_id = 100,
        .query_count = 64,
        .root_query_count = 64,
        .extent_transition_work = 128,
        .posting_work = 6400,
        .block_work = 320,
        .blocks_considered = 256,
        .blocks_scored = 192,
        .blocks_skipped = 64,
        .postings_scored = 4096,
        .last_posting_count = 100,
        .last_extent_count = 3,
        .last_block_count = 5,
        .observed_at_ms = 1000
    };
    evoke_posting_heat_merge_result merge_result;
    evoke_posting_heat_entry candidate;
    evoke_posting_heat_summary summary;

    ASSERT_TRUE(evoke_posting_heat_table_valid(4));
    ASSERT_TRUE(!evoke_posting_heat_table_valid(6));
    merge_result = evoke_posting_heat_merge(
        entries,
        4,
        &observation,
        1000,
        64,
        32
    );
    ASSERT_TRUE(merge_result.valid);
    ASSERT_TRUE(!merge_result.evicted);
    ASSERT_TRUE(merge_result.became_candidate);
    ASSERT_TRUE(evoke_posting_heat_select(
        entries,
        4,
        11,
        22,
        100,
        1000,
        1000,
        64,
        32,
        500,
        &candidate
    ));
    ASSERT_TRUE(candidate.term_id == 7);
    ASSERT_TRUE(candidate.query_heat == 64);
    ASSERT_TRUE(candidate.root_heat == 64);

    {
        evoke_posting_heat_entry specialization_entries[4] = {0};
        evoke_posting_heat_observation specialization_observation =
            observation;

        specialization_observation.root_id = 200;
        specialization_observation.query_count = 32;
        specialization_observation.root_query_count = 32;
        specialization_observation.last_extent_count = 1;
        specialization_observation.observed_at_ms = 1050;
        merge_result = evoke_posting_heat_merge(
            specialization_entries,
            4,
            &specialization_observation,
            1000,
            64,
            32
        );
        ASSERT_TRUE(merge_result.valid);
        ASSERT_TRUE(!merge_result.became_candidate);
        ASSERT_TRUE(!evoke_posting_heat_select(
            specialization_entries,
            4,
            11,
            22,
            200,
            1050,
            1000,
            64,
            32,
            500,
            &candidate
        ));

        specialization_observation.query_count = 32;
        specialization_observation.root_query_count = 32;
        specialization_observation.impact_specialization_ready = true;
        specialization_observation.observed_at_ms = 1100;
        merge_result = evoke_posting_heat_merge(
            specialization_entries,
            4,
            &specialization_observation,
            1000,
            64,
            32
        );
        ASSERT_TRUE(merge_result.valid);
        ASSERT_TRUE(merge_result.became_candidate);
        ASSERT_TRUE(evoke_posting_heat_select(
            specialization_entries,
            4,
            11,
            22,
            200,
            1100,
            1000,
            64,
            32,
            500,
            &candidate
        ));
        ASSERT_TRUE(candidate.impact_specialization_ready);

        specialization_observation.query_count = 1;
        specialization_observation.root_query_count = 1;
        specialization_observation.impact_specialization_ready = false;
        specialization_observation.hot_cache_republish_ready = true;
        specialization_observation.observed_at_ms = 1150;
        merge_result = evoke_posting_heat_merge(
            specialization_entries,
            4,
            &specialization_observation,
            1000,
            64,
            32
        );
        ASSERT_TRUE(merge_result.valid);
        ASSERT_TRUE(evoke_posting_heat_select(
            specialization_entries,
            4,
            11,
            22,
            200,
            1150,
            1000,
            64,
            32,
            500,
            &candidate
        ));
        ASSERT_TRUE(candidate.hot_cache_republish_ready);
        ASSERT_TRUE(evoke_posting_heat_note_attempt(
            specialization_entries,
            4,
            11,
            22,
            7,
            200,
            1150
        ));
        ASSERT_TRUE(evoke_posting_heat_note_hot_cache_resident(
            specialization_entries,
            4,
            11,
            22,
            7,
            200
        ));
        ASSERT_TRUE(!evoke_posting_heat_select(
            specialization_entries,
            4,
            11,
            22,
            200,
            1150,
            1000,
            64,
            32,
            500,
            &candidate
        ));

        specialization_observation.observed_at_ms = 1160;
        merge_result = evoke_posting_heat_merge(
            specialization_entries,
            4,
            &specialization_observation,
            1000,
            64,
            32
        );
        ASSERT_TRUE(merge_result.valid);
        ASSERT_TRUE(evoke_posting_heat_select(
            specialization_entries,
            4,
            11,
            22,
            200,
            1160,
            1000,
            64,
            32,
            500,
            &candidate
        ));
    }

    ASSERT_TRUE(evoke_posting_heat_note_attempt(
        entries,
        4,
        11,
        22,
        7,
        100,
        1100
    ));
    ASSERT_TRUE(!evoke_posting_heat_select(
        entries,
        4,
        11,
        22,
        100,
        1200,
        1000,
        64,
        32,
        500,
        &candidate
    ));
    ASSERT_TRUE(evoke_posting_heat_select(
        entries,
        4,
        11,
        22,
        100,
        1600,
        1000,
        64,
        32,
        500,
        &candidate
    ));

    observation.root_id = 101;
    observation.query_count = 32;
    observation.root_query_count = 32;
    observation.observed_at_ms = 1700;
    merge_result = evoke_posting_heat_merge(
        entries,
        4,
        &observation,
        1000,
        64,
        32
    );
    ASSERT_TRUE(merge_result.valid);
    ASSERT_TRUE(merge_result.root_heat == 32);
    ASSERT_TRUE(evoke_posting_heat_select(
        entries,
        4,
        11,
        22,
        101,
        1700,
        1000,
        64,
        32,
        500,
        &candidate
    ));
    ASSERT_TRUE(candidate.extent_transition_work == 128);
    ASSERT_TRUE(candidate.posting_work == 6400);
    ASSERT_TRUE(candidate.block_work == 320);
    ASSERT_TRUE(candidate.blocks_considered == 256);
    ASSERT_TRUE(candidate.blocks_scored == 192);
    ASSERT_TRUE(candidate.blocks_skipped == 64);
    ASSERT_TRUE(candidate.postings_scored == 4096);

    observation.root_id = 100;
    observation.query_count = 1;
    observation.root_query_count = 1;
    observation.extent_transition_work = 999;
    observation.posting_work = 999;
    observation.block_work = 999;
    observation.blocks_considered = 999;
    observation.blocks_scored = 999;
    observation.blocks_skipped = 0;
    observation.postings_scored = 999;
    observation.observed_at_ms = 1750;
    merge_result = evoke_posting_heat_merge(
        entries,
        4,
        &observation,
        1000,
        64,
        32
    );
    ASSERT_TRUE(merge_result.valid);
    ASSERT_TRUE(evoke_posting_heat_select(
        entries,
        4,
        11,
        22,
        101,
        1750,
        1000,
        64,
        32,
        500,
        &candidate
    ));
    ASSERT_TRUE(candidate.extent_transition_work == 128);
    ASSERT_TRUE(candidate.posting_work == 6400);
    ASSERT_TRUE(candidate.block_work == 320);
    ASSERT_TRUE(candidate.blocks_considered == 256);
    ASSERT_TRUE(candidate.blocks_scored == 192);
    ASSERT_TRUE(candidate.blocks_skipped == 64);
    ASSERT_TRUE(candidate.postings_scored == 4096);

    evoke_posting_heat_summarize(
        entries,
        4,
        11,
        22,
        101,
        2700,
        1000,
        32,
        16,
        500,
        &summary
    );
    ASSERT_TRUE(summary.entry_count == 1);
    ASSERT_TRUE(summary.total_query_heat == 48);
    ASSERT_TRUE(summary.candidate_found);

    for (uint32_t term_id = 8; term_id <= 11; term_id++)
    {
        observation.term_id = term_id;
        observation.root_id = 101;
        observation.query_count = term_id;
        observation.root_query_count = term_id;
        observation.observed_at_ms = 2800 + term_id;
        merge_result = evoke_posting_heat_merge(
            entries,
            4,
            &observation,
            1000,
            64,
            32
        );
    }
    ASSERT_TRUE(merge_result.evicted);

    evoke_posting_heat_summarize(
        entries,
        4,
        11,
        22,
        101,
        100000,
        1000,
        1,
        1,
        500,
        &summary
    );
    ASSERT_TRUE(summary.total_query_heat == 0);
    ASSERT_TRUE(!summary.candidate_found);
}

static bool
lexicon_cow_test_ref_equal(
    const evoke_lexicon_cow_ref *left,
    const evoke_lexicon_cow_ref *right
)
{
    return left->kind == right->kind &&
        left->start_block == right->start_block &&
        left->page_count == right->page_count &&
        left->reserved == right->reserved &&
        left->object_id == right->object_id &&
        left->owner_manifest_id == right->owner_manifest_id &&
        left->object_bytes == right->object_bytes &&
        left->checksum == right->checksum &&
        left->blob_checksum == right->blob_checksum;
}

typedef struct lexicon_cow_test_object
{
    evoke_lexicon_cow_ref ref;
    uint8_t *bytes;
    size_t size;
} lexicon_cow_test_object;

typedef struct lexicon_cow_test_store
{
    lexicon_cow_test_object *objects;
    size_t object_count;
    size_t object_capacity;
    uint64_t corrupt_owner_manifest_id;
    uint64_t corrupt_object_id;
    uint32_t load_count;
} lexicon_cow_test_store;

typedef struct lexicon_cow_visit_counts
{
    uint32_t objects;
    uint32_t nodes;
    uint32_t buckets;
    uint32_t entries;
    uint64_t term_id_sum;
} lexicon_cow_visit_counts;

static void
lexicon_cow_test_store_free(lexicon_cow_test_store *store)
{
    if (store == NULL)
    {
        return;
    }
    for (size_t index = 0; index < store->object_count; index++)
    {
        free(store->objects[index].bytes);
    }
    free(store->objects);
    memset(store, 0, sizeof(*store));
}

static evoke_status
load_lexicon_cow_test_object(
    void *context,
    const evoke_lexicon_cow_ref *ref,
    evoke_lexicon_cow_object *object_out
)
{
    lexicon_cow_test_store *store = context;

    if (store == NULL || ref == NULL || object_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    store->load_count++;
    for (size_t index = 0; index < store->object_count; index++)
    {
        lexicon_cow_test_object *stored = &store->objects[index];
        evoke_segment_object_ref storage_ref;
        const uint8_t *bytes = stored->bytes;
        uint8_t *corrupt = NULL;
        evoke_status status;

        if (stored->ref.owner_manifest_id != ref->owner_manifest_id ||
            stored->ref.object_id != ref->object_id)
        {
            continue;
        }
        if (!lexicon_cow_test_ref_equal(&stored->ref, ref))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (store->corrupt_owner_manifest_id ==
                ref->owner_manifest_id &&
            store->corrupt_object_id == ref->object_id)
        {
            corrupt = malloc(stored->size);
            if (corrupt == NULL)
            {
                return EVOKE_ERR_NOMEM;
            }
            memcpy(corrupt, stored->bytes, stored->size);
            corrupt[stored->size - 1] ^= UINT8_C(1);
            bytes = corrupt;
        }
        status = evoke_lexicon_cow_object_deserialize(
            bytes,
            stored->size,
            object_out
        );
        free(corrupt);
        if (status != EVOKE_OK)
        {
            return status;
        }
        status = evoke_lexicon_cow_ref_as_segment_object_ref(
            ref,
            &storage_ref
        );
        if (status == EVOKE_OK)
        {
            status = evoke_lexicon_cow_object_bind_storage(
                object_out,
                &storage_ref
            );
        }
        if (status != EVOKE_OK)
        {
            evoke_lexicon_cow_object_free(object_out);
        }
        return status;
    }
    return EVOKE_ERR_FORMAT;
}

static evoke_status
load_serialized_lexicon_cow_test_object(
    void *context,
    const evoke_lexicon_cow_ref *ref,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    lexicon_cow_test_store *store = context;

    if (store == NULL || ref == NULL || bytes_out == NULL ||
        size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    store->load_count++;
    for (size_t index = 0; index < store->object_count; index++)
    {
        const lexicon_cow_test_object *stored = &store->objects[index];
        uint8_t *bytes;

        if (stored->ref.owner_manifest_id != ref->owner_manifest_id ||
            stored->ref.object_id != ref->object_id)
        {
            continue;
        }
        if (!lexicon_cow_test_ref_equal(&stored->ref, ref))
        {
            return EVOKE_ERR_FORMAT;
        }
        bytes = malloc(stored->size);
        if (bytes == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        memcpy(bytes, stored->bytes, stored->size);
        if (store->corrupt_owner_manifest_id == ref->owner_manifest_id &&
            store->corrupt_object_id == ref->object_id)
        {
            bytes[stored->size - 1] ^= UINT8_C(1);
        }
        *bytes_out = bytes;
        *size_out = stored->size;
        return EVOKE_OK;
    }
    return EVOKE_ERR_FORMAT;
}

static void
release_serialized_lexicon_cow_test_object(
    void *context,
    uint8_t *bytes
)
{
    (void) context;
    free(bytes);
}

static void
store_lexicon_cow_test_tree(
    lexicon_cow_test_store *store,
    evoke_lexicon_cow_tree *tree,
    uint32_t *next_block
)
{
    for (uint64_t object_id = 1;
         object_id <= tree->object_count;
         object_id++)
    {
        lexicon_cow_test_object *stored;
        evoke_segment_object_ref storage_ref;
        lexicon_cow_test_object *resized;
        uint8_t *bytes = NULL;
        size_t size = 0;
        uint32_t page_count = 0;

        ASSERT_STATUS_OK(
            evoke_lexicon_cow_tree_prepare_object_for_storage(
                tree,
                object_id,
                &bytes,
                &size
            )
        );
        ASSERT_STATUS_OK(evoke_segment_page_count_required(
            size,
            8192,
            &page_count
        ));
        memset(&storage_ref, 0, sizeof(storage_ref));
        storage_ref.object_kind = EVOKE_SEGMENT_OBJECT_LEXICON_LOOKUP;
        storage_ref.start_block = *next_block;
        storage_ref.page_count = page_count;
        storage_ref.object_id = object_id;
        storage_ref.owner_manifest_id =
            tree->objects[object_id - 1].ref.owner_manifest_id;
        storage_ref.object_bytes = size;
        storage_ref.object_checksum =
            evoke_segment_blob_checksum(bytes, size);
        ASSERT_STATUS_OK(evoke_lexicon_cow_tree_bind_object_storage(
            tree,
            object_id,
            &storage_ref
        ));
        if (store->object_count == store->object_capacity)
        {
            size_t next_capacity = store->object_capacity == 0
                ? 16
                : store->object_capacity * 2;

            resized = realloc(
                store->objects,
                next_capacity * sizeof(*store->objects)
            );
            ASSERT_TRUE(resized != NULL);
            store->objects = resized;
            store->object_capacity = next_capacity;
        }
        stored = &store->objects[store->object_count++];
        memset(stored, 0, sizeof(*stored));
        stored->ref = tree->objects[object_id - 1].ref;
        stored->bytes = bytes;
        stored->size = size;
        *next_block += page_count;
    }
}

static evoke_status
count_lexicon_cow_test_object(
    void *context,
    const evoke_lexicon_cow_object *object
)
{
    lexicon_cow_visit_counts *counts = context;

    if (counts == NULL || object == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    counts->objects++;
    if (object->ref.kind == EVOKE_LEXICON_COW_OBJECT_NODE)
    {
        counts->nodes++;
    }
    else if (object->ref.kind == EVOKE_LEXICON_COW_OBJECT_BUCKET)
    {
        counts->buckets++;
    }
    else
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

static evoke_status
count_lexicon_cow_test_entry(
    void *context,
    const evoke_lexicon_cow_entry *entry
)
{
    lexicon_cow_visit_counts *counts = context;

    if (counts == NULL || entry == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    counts->entries++;
    counts->term_id_sum += entry->term_id;
    return EVOKE_OK;
}

static void
test_lexicon_cow_restart_and_external_patch(void)
{
    enum
    {
        base_count = 768,
        added_count = 3,
        term_width = 64
    };
    char base_terms[base_count][term_width];
    char added_terms[added_count][term_width];
    evoke_lexicon_cow_key base_keys[base_count];
    evoke_lexicon_cow_key added_keys[added_count];
    evoke_lexicon_cow_tree tree;
    evoke_lexicon_cow_tree patch;
    evoke_lexicon_cow_object serialized_root = {0};
    evoke_lexicon_cow_ref old_root;
    evoke_lexicon_cow_ref next_root;
    evoke_lexicon_cow_ref bad_root;
    evoke_lexicon_cow_update_stats stats;
    lexicon_cow_test_store store = {0};
    lexicon_cow_visit_counts counts = {0};
    uint32_t next_block = 20;

    evoke_lexicon_cow_tree_init(&tree);
    evoke_lexicon_cow_tree_init(&patch);
    for (uint32_t index = 0; index < base_count; index++)
    {
        int length = snprintf(
            base_terms[index],
            term_width,
            "restart-safe-lexical-term-%06u",
            index
        );

        ASSERT_TRUE(length > 0 && length < term_width);
        base_keys[index].bytes =
            (const uint8_t *) base_terms[index];
        base_keys[index].bytes_len = (uint32_t) length;
        base_keys[index].term_id = index;
    }
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_build(
        base_keys,
        base_count,
        UINT64_C(0x9182),
        71,
        &tree
    ));
    store_lexicon_cow_test_tree(&store, &tree, &next_block);
    old_root = tree.root;
    evoke_lexicon_cow_tree_free(&tree);

    ASSERT_STATUS_OK(evoke_lexicon_cow_validate_external(
        &old_root,
        UINT64_C(0x9182),
        base_count,
        load_lexicon_cow_test_object,
        &store
    ));
    ASSERT_STATUS_OK(load_lexicon_cow_test_object(
        &store,
        &old_root,
        &serialized_root
    ));
    for (uint32_t index = 0; index < base_count; index += 97)
    {
        uint32_t term_id = UINT32_MAX;
        bool found = false;

        ASSERT_STATUS_OK(evoke_lexicon_cow_lookup_external(
            &old_root,
            UINT64_C(0x9182),
            base_keys[index].bytes,
            base_keys[index].bytes_len,
            load_lexicon_cow_test_object,
            &store,
            &term_id,
            &found
        ));
        ASSERT_TRUE(found && term_id == index);
        term_id = UINT32_MAX;
        found = false;
        ASSERT_STATUS_OK(evoke_lexicon_cow_lookup_serialized_external(
            &serialized_root,
            UINT64_C(0x9182),
            base_keys[index].bytes,
            base_keys[index].bytes_len,
            load_serialized_lexicon_cow_test_object,
            release_serialized_lexicon_cow_test_object,
            &store,
            &term_id,
            &found
        ));
        ASSERT_TRUE(found && term_id == index);
    }
    evoke_lexicon_cow_object_free(&serialized_root);
    for (uint32_t index = 0; index < added_count; index++)
    {
        int length = snprintf(
            added_terms[index],
            term_width,
            "restart-added-lexical-term-%06u",
            index
        );

        ASSERT_TRUE(length > 0 && length < term_width);
        added_keys[index].bytes =
            (const uint8_t *) added_terms[index];
        added_keys[index].bytes_len = (uint32_t) length;
        added_keys[index].term_id = base_count + index;
    }
    store.load_count = 0;
    ASSERT_STATUS_OK(evoke_lexicon_cow_build_external_append_patch(
        &old_root,
        UINT64_C(0x9182),
        base_count,
        added_keys,
        added_count,
        72,
        load_lexicon_cow_test_object,
        &store,
        &patch,
        &stats
    ));
    ASSERT_TRUE(stats.changed_terms == added_count);
    ASSERT_TRUE(stats.read_buckets <= added_count);
    ASSERT_TRUE(
        stats.read_nodes <=
            added_count * EVOKE_LEXICON_COW_RADIX_LEVELS
    );
    ASSERT_TRUE(
        store.load_count <=
            added_count * 2 *
                (EVOKE_LEXICON_COW_RADIX_LEVELS + 1)
    );
    assert_retired_ranges(
        patch.retired_ranges,
        patch.retired_range_count,
        next_block
    );
    store_lexicon_cow_test_tree(&store, &patch, &next_block);
    next_root = patch.root;
    evoke_lexicon_cow_tree_free(&patch);

    ASSERT_STATUS_OK(load_lexicon_cow_test_object(
        &store,
        &next_root,
        &serialized_root
    ));

    ASSERT_STATUS_OK(evoke_lexicon_cow_validate_external(
        &old_root,
        UINT64_C(0x9182),
        base_count,
        load_lexicon_cow_test_object,
        &store
    ));
    ASSERT_STATUS_OK(evoke_lexicon_cow_visit_external(
        &next_root,
        UINT64_C(0x9182),
        base_count + added_count,
        load_lexicon_cow_test_object,
        &store,
        count_lexicon_cow_test_object,
        &counts
    ));
    ASSERT_TRUE(counts.objects == counts.nodes + counts.buckets);
    ASSERT_TRUE(counts.nodes > 0 && counts.buckets > 0);
    ASSERT_STATUS_OK(evoke_lexicon_cow_scan_external(
        &next_root,
        UINT64_C(0x9182),
        base_count + added_count,
        load_lexicon_cow_test_object,
        &store,
        count_lexicon_cow_test_entry,
        &counts
    ));
    ASSERT_TRUE(counts.entries == base_count + added_count);
    ASSERT_TRUE(
        counts.term_id_sum ==
            ((uint64_t) (base_count + added_count - 1) *
             (base_count + added_count)) / 2
    );
    for (uint32_t index = 0; index < added_count; index++)
    {
        uint32_t term_id = UINT32_MAX;
        bool found = false;

        ASSERT_STATUS_OK(evoke_lexicon_cow_lookup_external(
            &next_root,
            UINT64_C(0x9182),
            added_keys[index].bytes,
            added_keys[index].bytes_len,
            load_lexicon_cow_test_object,
            &store,
            &term_id,
            &found
        ));
        ASSERT_TRUE(found && term_id == base_count + index);
        term_id = UINT32_MAX;
        found = false;
        ASSERT_STATUS_OK(evoke_lexicon_cow_lookup_serialized_external(
            &serialized_root,
            UINT64_C(0x9182),
            added_keys[index].bytes,
            added_keys[index].bytes_len,
            load_serialized_lexicon_cow_test_object,
            release_serialized_lexicon_cow_test_object,
            &store,
            &term_id,
            &found
        ));
        ASSERT_TRUE(found && term_id == base_count + index);
        ASSERT_STATUS_OK(evoke_lexicon_cow_lookup_external(
            &old_root,
            UINT64_C(0x9182),
            added_keys[index].bytes,
            added_keys[index].bytes_len,
            load_lexicon_cow_test_object,
            &store,
            &term_id,
            &found
        ));
        ASSERT_TRUE(!found);
    }
    evoke_lexicon_cow_object_free(&serialized_root);
    store.corrupt_owner_manifest_id = next_root.owner_manifest_id;
    store.corrupt_object_id = next_root.object_id;
    ASSERT_TRUE(evoke_lexicon_cow_scan_external(
        &next_root,
        UINT64_C(0x9182),
        base_count + added_count,
        load_lexicon_cow_test_object,
        &store,
        count_lexicon_cow_test_entry,
        &counts
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_lexicon_cow_validate_external(
        &next_root,
        UINT64_C(0x9182),
        base_count + added_count,
        load_lexicon_cow_test_object,
        &store
    ) == EVOKE_ERR_FORMAT);
    store.corrupt_owner_manifest_id = 0;
    store.corrupt_object_id = 0;
    bad_root = next_root;
    bad_root.object_bytes++;
    ASSERT_TRUE(evoke_lexicon_cow_validate_external(
        &bad_root,
        UINT64_C(0x9182),
        base_count + added_count,
        load_lexicon_cow_test_object,
        &store
    ) == EVOKE_ERR_FORMAT);

    lexicon_cow_test_store_free(&store);
}

static void
test_lexicon_cow_incremental_suffix(void)
{
    enum
    {
        base_count = 384,
        added_count = 3,
        term_capacity = 72
    };
    char base_terms[base_count][term_capacity];
    char added_terms[added_count][term_capacity];
    evoke_lexicon_cow_key base_keys[base_count];
    evoke_lexicon_cow_key added_keys[added_count];
    evoke_lexicon_cow_tree tree;
    evoke_lexicon_cow_ref old_root;
    evoke_lexicon_cow_update_stats stats;
    evoke_lexicon_cow_object restored = {0};
    uint8_t *serialized = NULL;
    size_t serialized_len = 0;
    size_t old_object_count;
    uint32_t collision_left = UINT32_MAX;
    uint32_t collision_right = UINT32_MAX;
    uint32_t first_by_slot[EVOKE_LEXICON_COW_RADIX_FANOUT];

    memset(first_by_slot, 0xff, sizeof(first_by_slot));
    evoke_lexicon_cow_tree_init(&tree);
    for (uint32_t index = 0; index < base_count; index++)
    {
        int length = snprintf(
            base_terms[index],
            term_capacity,
            "lexical-term-%04u-with-a-stable-payload",
            index
        );
        uint64_t hash;
        uint32_t slot;

        ASSERT_TRUE(length > 0 && length < term_capacity);
        base_keys[index].bytes = (const uint8_t *) base_terms[index];
        base_keys[index].bytes_len = (uint32_t) length;
        base_keys[index].term_id = index;
        hash = evoke_lexicon_cow_hash(
            base_keys[index].bytes,
            base_keys[index].bytes_len,
            UINT64_C(0x5a17)
        );
        slot = (uint32_t) hash &
            (EVOKE_LEXICON_COW_RADIX_FANOUT - 1);
        if (first_by_slot[slot] == UINT32_MAX)
        {
            first_by_slot[slot] = index;
        }
        else if (collision_left == UINT32_MAX)
        {
            collision_left = first_by_slot[slot];
            collision_right = index;
        }
    }
    ASSERT_TRUE(collision_left != UINT32_MAX);
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_build(
        base_keys,
        base_count,
        UINT64_C(0x5a17),
        41,
        &tree
    ));
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_validate_at(
        &tree,
        &tree.root,
        base_count
    ));
    ASSERT_TRUE(tree.root.kind == EVOKE_LEXICON_COW_OBJECT_NODE);

    for (uint32_t index = 0; index < base_count; index++)
    {
        uint32_t term_id = UINT32_MAX;
        bool found = false;

        ASSERT_STATUS_OK(evoke_lexicon_cow_tree_lookup(
            &tree,
            base_keys[index].bytes,
            base_keys[index].bytes_len,
            &term_id,
            &found
        ));
        ASSERT_TRUE(found && term_id == index);
    }
    ASSERT_TRUE(collision_left != collision_right);
    old_root = tree.root;
    old_object_count = tree.object_count;

    for (uint32_t index = 0; index < added_count; index++)
    {
        int length = snprintf(
            added_terms[index],
            term_capacity,
            "new-lexical-term-%04u-with-a-stable-payload",
            index
        );

        ASSERT_TRUE(length > 0 && length < term_capacity);
        added_keys[index].bytes = (const uint8_t *) added_terms[index];
        added_keys[index].bytes_len = (uint32_t) length;
        added_keys[index].term_id = base_count + index;
    }
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_append(
        &tree,
        &old_root,
        base_count,
        added_keys,
        added_count,
        42,
        &stats
    ));
    ASSERT_TRUE(stats.changed_terms == added_count);
    ASSERT_TRUE(stats.read_buckets <= added_count);
    ASSERT_TRUE(
        stats.read_nodes <=
            added_count * EVOKE_LEXICON_COW_RADIX_LEVELS
    );
    ASSERT_TRUE(
        stats.written_nodes + stats.written_buckets ==
            tree.object_count - old_object_count
    );
    ASSERT_TRUE(
        stats.written_nodes + stats.written_buckets <
            EVOKE_LEXICON_COW_RADIX_FANOUT +
                added_count * EVOKE_LEXICON_COW_RADIX_LEVELS
    );
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_validate_at(
        &tree,
        &old_root,
        base_count
    ));
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_validate_at(
        &tree,
        &tree.root,
        base_count + added_count
    ));
    for (uint32_t index = 0; index < added_count; index++)
    {
        uint32_t term_id = UINT32_MAX;
        bool found = false;

        ASSERT_STATUS_OK(evoke_lexicon_cow_tree_lookup_at(
            &tree,
            &old_root,
            added_keys[index].bytes,
            added_keys[index].bytes_len,
            &term_id,
            &found
        ));
        ASSERT_TRUE(!found);
        ASSERT_STATUS_OK(evoke_lexicon_cow_tree_lookup(
            &tree,
            added_keys[index].bytes,
            added_keys[index].bytes_len,
            &term_id,
            &found
        ));
        ASSERT_TRUE(found && term_id == base_count + index);
    }

    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_object_serialize(
        &tree,
        &tree.root,
        &serialized,
        &serialized_len
    ));
    ASSERT_STATUS_OK(evoke_lexicon_cow_object_deserialize(
        serialized,
        serialized_len,
        &restored
    ));
    ASSERT_TRUE(restored.ref.object_id == tree.root.object_id);
    ASSERT_TRUE(restored.ref.owner_manifest_id == 42);
    evoke_lexicon_cow_object_free(&restored);
    serialized[serialized_len - 1] ^= UINT8_C(0x80);
    ASSERT_TRUE(evoke_lexicon_cow_object_deserialize(
        serialized,
        serialized_len,
        &restored
    ) == EVOKE_ERR_FORMAT);

    free(serialized);
    evoke_lexicon_cow_tree_free(&tree);
}

static void
test_lexicon_cow_large_vocabulary_update_is_bounded(void)
{
    const uint32_t base_count = 100000;
    const size_t term_width = 48;
    evoke_lexicon_cow_key *keys = NULL;
    char *terms = NULL;
    evoke_lexicon_cow_key added[2] = {0};
    const char *added_terms[] = {
        "bounded-new-term-alpha",
        "bounded-new-term-omega"
    };
    evoke_lexicon_cow_tree tree;
    evoke_lexicon_cow_ref old_root;
    evoke_lexicon_cow_update_stats stats;
    size_t old_object_count;

    keys = calloc(base_count, sizeof(*keys));
    terms = calloc(base_count, term_width);
    ASSERT_TRUE(keys != NULL && terms != NULL);
    for (uint32_t index = 0; index < base_count; index++)
    {
        char *term = terms + (size_t) index * term_width;
        int length = snprintf(
            term,
            term_width,
            "large-vocabulary-term-%08u",
            index
        );

        ASSERT_TRUE(length > 0 && (size_t) length < term_width);
        keys[index].bytes = (const uint8_t *) term;
        keys[index].bytes_len = (uint32_t) length;
        keys[index].term_id = index;
    }
    evoke_lexicon_cow_tree_init(&tree);
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_build(
        keys,
        base_count,
        UINT64_C(0x7139),
        51,
        &tree
    ));
    old_root = tree.root;
    old_object_count = tree.object_count;
    for (uint32_t index = 0; index < 2; index++)
    {
        added[index].bytes = (const uint8_t *) added_terms[index];
        added[index].bytes_len = (uint32_t) strlen(added_terms[index]);
        added[index].term_id = base_count + index;
    }
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_append(
        &tree,
        &old_root,
        base_count,
        added,
        2,
        52,
        &stats
    ));
    ASSERT_TRUE(stats.changed_terms == 2);
    ASSERT_TRUE(stats.read_buckets <= 2);
    ASSERT_TRUE(stats.read_nodes <= 2 * EVOKE_LEXICON_COW_RADIX_LEVELS);
    ASSERT_TRUE(
        stats.written_nodes + stats.written_buckets <
            EVOKE_LEXICON_COW_RADIX_FANOUT +
                2 * EVOKE_LEXICON_COW_RADIX_LEVELS
    );
    ASSERT_TRUE(
        tree.object_count - old_object_count ==
            stats.written_nodes + stats.written_buckets
    );
    ASSERT_TRUE(stats.written_bytes < UINT64_C(512) * 1024);
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_validate_at(
        &tree,
        &old_root,
        base_count
    ));
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_validate_at(
        &tree,
        &tree.root,
        base_count + 2
    ));

    evoke_lexicon_cow_tree_free(&tree);
    free(terms);
    free(keys);
}

static void
test_lexicon_cow_compact_sealed_geometry(void)
{
    const uint32_t term_count = 26500;
    const size_t term_width = 24;
    evoke_lexicon_cow_key *keys = NULL;
    char *terms = NULL;
    evoke_lexicon_cow_tree tree;
    uint64_t total_pages = 0;

    keys = calloc(term_count, sizeof(*keys));
    terms = calloc(term_count, term_width);
    ASSERT_TRUE(keys != NULL && terms != NULL);
    for (uint32_t index = 0; index < term_count; index++)
    {
        char *term = terms + (size_t) index * term_width;
        int length = snprintf(term, term_width, "term-%08u", index);

        ASSERT_TRUE(length > 0 && (size_t) length < term_width);
        keys[index].bytes = (const uint8_t *) term;
        keys[index].bytes_len = (uint32_t) length;
        keys[index].term_id = index;
    }
    evoke_lexicon_cow_tree_init(&tree);
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_build(
        keys,
        term_count,
        UINT64_C(0x51a1),
        61,
        &tree
    ));
    ASSERT_TRUE(tree.root.kind == EVOKE_LEXICON_COW_OBJECT_NODE);
    ASSERT_TRUE(
        tree.object_count <= EVOKE_LEXICON_COW_RADIX_FANOUT + 1
    );
    for (size_t index = 0; index < tree.object_count; index++)
    {
        const evoke_lexicon_cow_object *object = &tree.objects[index];
        uint32_t page_count = 0;

        ASSERT_STATUS_OK(evoke_segment_page_count_required(
            object->ref.object_bytes,
            8192,
            &page_count
        ));
        ASSERT_TRUE(page_count <= 2);
        total_pages += page_count;
    }
    ASSERT_TRUE(
        total_pages <= 1 + EVOKE_LEXICON_COW_RADIX_FANOUT * 2
    );
    ASSERT_STATUS_OK(evoke_lexicon_cow_tree_validate_at(
        &tree,
        &tree.root,
        term_count
    ));

    evoke_lexicon_cow_tree_free(&tree);
    free(terms);
    free(keys);
}

static bool
prefix_cow_test_ref_equal(
    const evoke_prefix_cow_ref *left,
    const evoke_prefix_cow_ref *right
)
{
    return left->kind == right->kind &&
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

typedef struct prefix_cow_test_object
{
    evoke_prefix_cow_ref ref;
    uint8_t *bytes;
    size_t size;
} prefix_cow_test_object;

typedef struct prefix_cow_test_store
{
    prefix_cow_test_object *objects;
    size_t object_count;
    size_t object_capacity;
    uint32_t load_count;
} prefix_cow_test_store;

typedef struct prefix_cow_test_scan
{
    const uint8_t *prefix;
    size_t prefix_len;
    uint32_t count;
    uint64_t term_id_sum;
} prefix_cow_test_scan;

static void
prefix_cow_test_store_free(prefix_cow_test_store *store)
{
    if (store == NULL)
    {
        return;
    }
    for (size_t index = 0; index < store->object_count; index++)
    {
        free(store->objects[index].bytes);
    }
    free(store->objects);
    memset(store, 0, sizeof(*store));
}

static evoke_status
load_prefix_cow_test_object(
    void *context,
    const evoke_prefix_cow_ref *ref,
    evoke_prefix_cow_object *object_out
)
{
    prefix_cow_test_store *store = context;

    if (store == NULL || ref == NULL || object_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    store->load_count++;
    for (size_t index = 0; index < store->object_count; index++)
    {
        prefix_cow_test_object *stored = &store->objects[index];
        evoke_segment_object_ref storage_ref;
        evoke_status status;

        if (stored->ref.owner_manifest_id != ref->owner_manifest_id ||
            stored->ref.object_id != ref->object_id)
        {
            continue;
        }
        if (!prefix_cow_test_ref_equal(&stored->ref, ref))
        {
            return EVOKE_ERR_FORMAT;
        }
        status = evoke_prefix_cow_object_deserialize(
            stored->bytes,
            stored->size,
            object_out
        );
        if (status == EVOKE_OK)
        {
            status = evoke_prefix_cow_ref_as_segment_object_ref(
                ref,
                &storage_ref
            );
        }
        if (status == EVOKE_OK)
        {
            status = evoke_prefix_cow_object_bind_storage(
                object_out,
                &storage_ref
            );
        }
        if (status != EVOKE_OK)
        {
            evoke_prefix_cow_object_free(object_out);
        }
        return status;
    }
    return EVOKE_ERR_FORMAT;
}

static void
store_prefix_cow_test_tree(
    prefix_cow_test_store *store,
    evoke_prefix_cow_tree *tree,
    uint32_t *next_block
)
{
    for (uint64_t object_id = 1;
         object_id <= tree->object_count;
         object_id++)
    {
        prefix_cow_test_object *stored;
        prefix_cow_test_object *resized;
        evoke_segment_object_ref storage_ref;
        uint8_t *bytes = NULL;
        size_t size = 0;
        uint32_t page_count = 0;

        ASSERT_STATUS_OK(
            evoke_prefix_cow_tree_prepare_object_for_storage(
                tree,
                object_id,
                &bytes,
                &size
            )
        );
        ASSERT_STATUS_OK(evoke_segment_page_count_required(
            size,
            8192,
            &page_count
        ));
        memset(&storage_ref, 0, sizeof(storage_ref));
        storage_ref.object_kind = EVOKE_SEGMENT_OBJECT_PREFIX_LOOKUP;
        storage_ref.start_block = *next_block;
        storage_ref.page_count = page_count;
        storage_ref.object_id = object_id;
        storage_ref.owner_manifest_id =
            tree->objects[object_id - 1].ref.owner_manifest_id;
        storage_ref.object_bytes = size;
        storage_ref.object_checksum =
            evoke_segment_blob_checksum(bytes, size);
        ASSERT_STATUS_OK(evoke_prefix_cow_tree_bind_object_storage(
            tree,
            object_id,
            &storage_ref
        ));
        if (store->object_count == store->object_capacity)
        {
            size_t capacity = store->object_capacity == 0
                ? 16
                : store->object_capacity * 2;

            resized = realloc(
                store->objects,
                capacity * sizeof(*store->objects)
            );
            ASSERT_TRUE(resized != NULL);
            store->objects = resized;
            store->object_capacity = capacity;
        }
        stored = &store->objects[store->object_count++];
        memset(stored, 0, sizeof(*stored));
        stored->ref = tree->objects[object_id - 1].ref;
        stored->bytes = bytes;
        stored->size = size;
        *next_block += page_count;
    }
}

static evoke_status
scan_prefix_cow_test_entry(
    void *context,
    const evoke_prefix_cow_entry *entry
)
{
    prefix_cow_test_scan *scan = context;

    if (scan == NULL || entry == NULL ||
        entry->bytes_len < scan->prefix_len ||
        memcmp(entry->bytes, scan->prefix, scan->prefix_len) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    scan->count++;
    scan->term_id_sum += entry->term_id;
    return EVOKE_OK;
}

static evoke_status
count_prefix_cow_test_object(
    void *context,
    const evoke_prefix_cow_object *object
)
{
    uint32_t *count = context;

    if (count == NULL || object == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    (*count)++;
    return EVOKE_OK;
}

static void
test_prefix_cow_range_seek_and_external_patch(void)
{
    enum
    {
        base_count = 20000,
        added_count = 16,
        term_width = 40
    };
    evoke_prefix_cow_key *base_keys;
    evoke_prefix_cow_key added_keys[added_count];
    char *base_terms;
    char added_terms[added_count][term_width];
    evoke_prefix_cow_tree tree;
    evoke_prefix_cow_tree patch;
    evoke_prefix_cow_ref old_root;
    evoke_prefix_cow_update_stats stats;
    prefix_cow_test_store store = {0};
    prefix_cow_test_scan scan = {0};
    uint32_t next_block = 1;
    uint32_t object_count = 0;
    const uint8_t broad_prefix[] = "term_";
    const uint8_t before_prefix[] = "aaa-new-prefix";
    const uint8_t after_prefix[] = "zzz-new-prefix";
    const uint8_t missing_prefix[] = "zzzz-no-prefix-match";

    base_keys = calloc(base_count, sizeof(*base_keys));
    base_terms = calloc(base_count, term_width);
    ASSERT_TRUE(base_keys != NULL && base_terms != NULL);
    for (uint32_t index = 0; index < base_count; index++)
    {
        char *term = base_terms + (size_t) index * term_width;
        int length = snprintf(
            term,
            term_width,
            "term_%08u",
            index * 2
        );

        ASSERT_TRUE(length > 0 && (size_t) length < term_width);
        base_keys[index].bytes = (const uint8_t *) term;
        base_keys[index].bytes_len = (uint32_t) length;
        base_keys[index].term_id = index;
    }
    for (uint32_t index = 0; index < added_count; index++)
    {
        uint32_t number = 1 + index * 2500;
        int length;

        if (index == 0)
        {
            length = snprintf(
                added_terms[index],
                term_width,
                "%s",
                (const char *) before_prefix
            );
        }
        else if (index + 1 == added_count)
        {
            length = snprintf(
                added_terms[index],
                term_width,
                "%s",
                (const char *) after_prefix
            );
        }
        else
        {
            length = snprintf(
                added_terms[index],
                term_width,
                "term_%08u",
                number
            );
        }

        ASSERT_TRUE(length > 0 && (size_t) length < term_width);
        added_keys[index].bytes =
            (const uint8_t *) added_terms[index];
        added_keys[index].bytes_len = (uint32_t) length;
        added_keys[index].term_id = base_count + index;
    }
    evoke_prefix_cow_tree_init(&tree);
    evoke_prefix_cow_tree_init(&patch);
    ASSERT_STATUS_OK(evoke_prefix_cow_tree_build(
        base_keys,
        base_count,
        101,
        &tree
    ));
    store_prefix_cow_test_tree(&store, &tree, &next_block);
    old_root = tree.root;
    evoke_prefix_cow_tree_free(&tree);

    ASSERT_STATUS_OK(evoke_prefix_cow_validate_external(
        &old_root,
        base_count,
        load_prefix_cow_test_object,
        &store
    ));
    store.load_count = 0;
    scan.prefix = missing_prefix;
    scan.prefix_len = sizeof(missing_prefix) - 1;
    ASSERT_STATUS_OK(evoke_prefix_cow_scan_prefix_external(
        &old_root,
        base_count,
        scan.prefix,
        scan.prefix_len,
        load_prefix_cow_test_object,
        &store,
        scan_prefix_cow_test_entry,
        &scan
    ));
    ASSERT_TRUE(scan.count == 0);
    ASSERT_TRUE(store.load_count <= EVOKE_PREFIX_COW_MAX_DEPTH);

    memset(&stats, 0, sizeof(stats));
    ASSERT_STATUS_OK(evoke_prefix_cow_build_external_append_patch(
        &old_root,
        base_count,
        added_keys,
        added_count,
        102,
        load_prefix_cow_test_object,
        &store,
        &patch,
        &stats
    ));
    ASSERT_TRUE(stats.changed_terms == added_count);
    ASSERT_TRUE(stats.read_leaves <= added_count);
    ASSERT_TRUE(
        stats.read_nodes <= added_count * EVOKE_PREFIX_COW_MAX_DEPTH
    );
    ASSERT_TRUE(stats.written_bytes < UINT64_C(1024) * 1024);
    store_prefix_cow_test_tree(&store, &patch, &next_block);

    ASSERT_STATUS_OK(evoke_prefix_cow_validate_external(
        &old_root,
        base_count,
        load_prefix_cow_test_object,
        &store
    ));
    ASSERT_STATUS_OK(evoke_prefix_cow_validate_external(
        &patch.root,
        base_count + added_count,
        load_prefix_cow_test_object,
        &store
    ));
    ASSERT_STATUS_OK(evoke_prefix_cow_visit_external(
        &patch.root,
        base_count + added_count,
        load_prefix_cow_test_object,
        &store,
        count_prefix_cow_test_object,
        &object_count
    ));
    ASSERT_TRUE(object_count > 1);

    memset(&scan, 0, sizeof(scan));
    scan.prefix = broad_prefix;
    scan.prefix_len = sizeof(broad_prefix) - 1;
    ASSERT_STATUS_OK(evoke_prefix_cow_scan_prefix_external(
        &patch.root,
        base_count + added_count,
        scan.prefix,
        scan.prefix_len,
        load_prefix_cow_test_object,
        &store,
        scan_prefix_cow_test_entry,
        &scan
    ));
    ASSERT_TRUE(scan.count == base_count + added_count - 2);

    memset(&scan, 0, sizeof(scan));
    scan.prefix = before_prefix;
    scan.prefix_len = sizeof(before_prefix) - 1;
    ASSERT_STATUS_OK(evoke_prefix_cow_scan_prefix_external(
        &patch.root,
        base_count + added_count,
        scan.prefix,
        scan.prefix_len,
        load_prefix_cow_test_object,
        &store,
        scan_prefix_cow_test_entry,
        &scan
    ));
    ASSERT_TRUE(scan.count == 1);

    memset(&scan, 0, sizeof(scan));
    scan.prefix = after_prefix;
    scan.prefix_len = sizeof(after_prefix) - 1;
    ASSERT_STATUS_OK(evoke_prefix_cow_scan_prefix_external(
        &patch.root,
        base_count + added_count,
        scan.prefix,
        scan.prefix_len,
        load_prefix_cow_test_object,
        &store,
        scan_prefix_cow_test_entry,
        &scan
    ));
    ASSERT_TRUE(scan.count == 1);

    evoke_prefix_cow_tree_free(&patch);
    prefix_cow_test_store_free(&store);
    free(base_terms);
    free(base_keys);
}

static void
test_prefix_cow_parent_split_rebalances(void)
{
    enum
    {
        base_count = EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES *
            (EVOKE_PREFIX_COW_NODE_MAX_CHILDREN - 2),
        append_count = 3,
        term_width = 40
    };
    evoke_prefix_cow_key *base_keys;
    char *base_terms;
    evoke_prefix_cow_tree tree;
    evoke_prefix_cow_tree patch;
    evoke_prefix_cow_ref root;
    prefix_cow_test_store store = {0};
    uint32_t next_block = 1;

    base_keys = calloc(base_count, sizeof(*base_keys));
    base_terms = calloc(base_count, term_width);
    ASSERT_TRUE(base_keys != NULL && base_terms != NULL);
    for (uint32_t index = 0; index < base_count; index++)
    {
        char *term = base_terms + (size_t) index * term_width;
        int length = snprintf(term, term_width, "v%08u", index);

        ASSERT_TRUE(length > 0 && (size_t) length < term_width);
        base_keys[index].bytes = (const uint8_t *) term;
        base_keys[index].bytes_len = (uint32_t) length;
        base_keys[index].term_id = index;
    }
    evoke_prefix_cow_tree_init(&tree);
    evoke_prefix_cow_tree_init(&patch);
    ASSERT_STATUS_OK(evoke_prefix_cow_tree_build(
        base_keys,
        base_count,
        201,
        &tree
    ));
    store_prefix_cow_test_tree(&store, &tree, &next_block);
    root = tree.root;
    evoke_prefix_cow_tree_free(&tree);

    for (uint32_t index = 0; index < append_count; index++)
    {
        char term[term_width];
        evoke_prefix_cow_key key;
        evoke_prefix_cow_update_stats stats = {0};
        int length = snprintf(term, term_width, "aaa-prefix-%08u", index);

        ASSERT_TRUE(length > 0 && (size_t) length < term_width);
        key.bytes = (const uint8_t *) term;
        key.bytes_len = (uint32_t) length;
        key.term_id = base_count + index;
        ASSERT_STATUS_OK(evoke_prefix_cow_build_external_append_patch(
            &root,
            base_count + index,
            &key,
            1,
            202 + index,
            load_prefix_cow_test_object,
            &store,
            &patch,
            &stats
        ));
        ASSERT_TRUE(stats.changed_terms == 1);
        store_prefix_cow_test_tree(&store, &patch, &next_block);
        root = patch.root;
        ASSERT_STATUS_OK(evoke_prefix_cow_validate_external(
            &root,
            base_count + index + 1,
            load_prefix_cow_test_object,
            &store
        ));
        evoke_prefix_cow_tree_free(&patch);
        evoke_prefix_cow_tree_init(&patch);
    }

    evoke_prefix_cow_tree_free(&patch);
    prefix_cow_test_store_free(&store);
    free(base_terms);
    free(base_keys);
}

static void
test_text_tokenize_text_with_diacritic_folding(void)
{
    evoke_text_options options;
    char **tokens = NULL;
    size_t len = 0;

    evoke_text_options_init(&options);
    options.fold_diacritics = true;

    ASSERT_STATUS_OK(evoke_tokenize_text(
        "\303\234ber caf\303\251 fa\303\247ade",
        &options,
        &tokens,
        &len
    ));

    ASSERT_TRUE(len == 3);
    ASSERT_TRUE(strcmp(tokens[0], "uber") == 0);
    ASSERT_TRUE(strcmp(tokens[1], "cafe") == 0);
    ASSERT_TRUE(strcmp(tokens[2], "facade") == 0);

    evoke_text_tokens_free(tokens, len);
}

static void
test_u32_saturating_add(void)
{
    ASSERT_TRUE(evoke_u32_saturating_add(0, 0) == 0);
    ASSERT_TRUE(evoke_u32_saturating_add(40, 2) == 42);
    ASSERT_TRUE(
        evoke_u32_saturating_add(UINT32_MAX - 1, 1) == UINT32_MAX
    );
    ASSERT_TRUE(
        evoke_u32_saturating_add(UINT32_MAX - 1, 2) == UINT32_MAX
    );
}

static void
test_u64_saturating_arithmetic(void)
{
    ASSERT_TRUE(evoke_u64_saturating_add(40, 2) == 42);
    ASSERT_TRUE(
        evoke_u64_saturating_add(UINT64_MAX - 1, 2) == UINT64_MAX
    );
    ASSERT_TRUE(evoke_u64_saturating_mul(0, UINT64_MAX) == 0);
    ASSERT_TRUE(evoke_u64_saturating_mul(6, 7) == 42);
    ASSERT_TRUE(
        evoke_u64_saturating_mul(UINT64_MAX, 2) == UINT64_MAX
    );
}

static void
test_semantic_bmp_exact_signed_topk(void)
{
    const evoke_semantic_bmp_posting postings[] = {
        {1, 0, 0.5f},
        {1, 17, 2.0f},
        {1, 18, -1.0f},
        {1, 35, 1.0f},
        {2, 1, 3.0f},
        {2, 17, -0.5f},
        {2, 34, 2.0f},
        {3, 0, 1.0f},
        {3, 16, 2.0f},
        {3, 32, 4.0f},
        {5, 64, 0.01f},
    };
    const uint32_t query_ids[] = {5, 3, 1, 2};
    const float query_weights[] = {1.0f, -0.25f, 1.0f, 0.5f};
    const uint32_t expected_ids[] = {17, 1, 34};
    const float expected_scores[] = {1.75f, 1.5f, 1.0f};
    const uint32_t run1_ids[] = {0, 17, 18, 35};
    const uint32_t run2_ids[] = {1, 17, 34};
    const uint32_t run3_ids[] = {0, 16, 32};
    const uint32_t run5_ids[] = {64};
    const evoke_posting_value run1_values[] = {
        {.impact = 0.5f}, {.impact = 2.0f},
        {.impact = -1.0f}, {.impact = 1.0f}
    };
    const evoke_posting_value run2_values[] = {
        {.impact = 3.0f}, {.impact = -0.5f}, {.impact = 2.0f}
    };
    const evoke_posting_value run3_values[] = {
        {.impact = 1.0f}, {.impact = 2.0f}, {.impact = 4.0f}
    };
    const evoke_posting_value run5_values[] = {{.impact = 0.01f}};
    const evoke_semantic_bmp_run runs[] = {
        {1, 4, run1_ids, run1_values, NULL, 0, 80},
        {2, 3, run2_ids, run2_values, NULL, 0, 80},
        {3, 3, run3_ids, run3_values, NULL, 0, 80},
        {5, 1, run5_ids, run5_values, NULL, 0, 80}
    };
    evoke_semantic_bmp_index index;
    evoke_semantic_bmp_index run_index;
    evoke_semantic_bmp_index restored;
    evoke_semantic_bmp_packed_index packed_index;
    evoke_semantic_bmp_packed_index direct_packed_index;
    evoke_semantic_bmp_packed_index packed_restored;
    evoke_semantic_bmp_packed_disk_header packed_header;
    evoke_semantic_bmp_packed_term packed_term;
    evoke_semantic_bmp_packed_super_ref packed_super_ref;
    evoke_semantic_bmp_packed_ref packed_ref;
    evoke_semantic_bmp_stats stats;
    evoke_semantic_bmp_stats restored_stats;
    evoke_semantic_bmp_stats packed_stats;
    evoke_semantic_bmp_stats packed_taat_stats;
    evoke_topk_result result;
    evoke_topk_result run_result;
    evoke_topk_result restored_result;
    evoke_topk_result packed_result;
    evoke_topk_result direct_packed_result;
    evoke_topk_result packed_restored_result;
    evoke_topk_result packed_taat_result;
    uint8_t *serialized = NULL;
    uint8_t *packed_serialized = NULL;
    uint8_t *direct_packed_serialized = NULL;
    uint8_t *direct_packed_buffer = NULL;
    size_t expected_size = 0;
    size_t serialized_size = 0;
    size_t packed_size = 0;
    size_t packed_serialized_size = 0;
    size_t direct_packed_serialized_size = 0;
    uint32_t materialized_ids[4] = {0};
    evoke_posting_value materialized_values[4] = {0};

    evoke_semantic_bmp_index_init(&index);
    evoke_semantic_bmp_index_init(&run_index);
    evoke_semantic_bmp_index_init(&restored);
    evoke_semantic_bmp_packed_index_init(&packed_index);
    evoke_semantic_bmp_packed_index_init(&direct_packed_index);
    evoke_semantic_bmp_packed_index_init(&packed_restored);
    memset(&stats, 0, sizeof(stats));
    memset(&restored_stats, 0, sizeof(restored_stats));
    memset(&packed_stats, 0, sizeof(packed_stats));
    memset(&packed_taat_stats, 0, sizeof(packed_taat_stats));
    memset(&result, 0, sizeof(result));
    memset(&run_result, 0, sizeof(run_result));
    memset(&restored_result, 0, sizeof(restored_result));
    memset(&packed_result, 0, sizeof(packed_result));
    memset(&direct_packed_result, 0, sizeof(direct_packed_result));
    memset(&packed_restored_result, 0, sizeof(packed_restored_result));
    memset(&packed_taat_result, 0, sizeof(packed_taat_result));
    ASSERT_STATUS_OK(evoke_semantic_bmp_index_build(
        80,
        postings,
        sizeof(postings) / sizeof(postings[0]),
        &index
    ));
    ASSERT_STATUS_OK(evoke_semantic_bmp_topk(
        &index,
        query_ids,
        query_weights,
        sizeof(query_ids) / sizeof(query_ids[0]),
        3,
        &result,
        &stats
    ));
    ASSERT_TRUE(result.len == 3);
    for (size_t rank = 0; rank < result.len; rank++)
    {
        ASSERT_TRUE(result.doc_ids[rank] == expected_ids[rank]);
        ASSERT_TRUE(fabsf(result.scores[rank] - expected_scores[rank]) <
            1e-6f);
    }
    ASSERT_TRUE(stats.bound_entries_visited ==
        sizeof(postings) / sizeof(postings[0]) - 1);
    ASSERT_TRUE(stats.super_bound_entries_visited > 0);
    ASSERT_TRUE(stats.superblocks_scored > 0);
    ASSERT_TRUE(stats.blocks_scored > 0);
    ASSERT_TRUE(stats.blocks_skipped > 0);
    ASSERT_TRUE(stats.postings_examined <
        sizeof(postings) / sizeof(postings[0]));
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_index_build(
        &index,
        &packed_index
    ));
    ASSERT_TRUE(packed_index.block_count == 2);
    ASSERT_TRUE(packed_index.superblock_count == 1);
    ASSERT_TRUE(packed_index.block_membership_bytes > 0);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_serialized_size(
        &packed_index,
        &packed_size
    ));
    ASSERT_TRUE(packed_size > 0);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_topk(
        &packed_index,
        query_ids,
        query_weights,
        sizeof(query_ids) / sizeof(query_ids[0]),
        3,
        &packed_result,
        &packed_stats
    ));
    ASSERT_TRUE(packed_result.len == result.len);
    for (size_t rank = 0; rank < result.len; rank++)
    {
        ASSERT_TRUE(packed_result.doc_ids[rank] == result.doc_ids[rank]);
        ASSERT_TRUE(fabsf(
            packed_result.scores[rank] - result.scores[rank]
        ) < 1e-6f);
    }
    ASSERT_TRUE(packed_stats.blocks_scored > 0);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_taat_topk(
        &packed_index,
        query_ids,
        query_weights,
        sizeof(query_ids) / sizeof(query_ids[0]),
        3,
        &packed_taat_result,
        &packed_taat_stats
    ));
    ASSERT_TRUE(packed_taat_result.len == result.len);
    for (size_t rank = 0; rank < result.len; rank++)
    {
        ASSERT_TRUE(
            packed_taat_result.doc_ids[rank] == result.doc_ids[rank]
        );
        ASSERT_TRUE(fabsf(
            packed_taat_result.scores[rank] - result.scores[rank]
        ) < 1e-6f);
    }
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_serialize(
        &packed_index,
        &packed_serialized,
        &packed_serialized_size
    ));
    ASSERT_TRUE(packed_serialized_size == packed_size);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_index_build_runs(
        80,
        runs,
        sizeof(runs) / sizeof(runs[0]),
        &direct_packed_index
    ));
    ASSERT_TRUE(
        direct_packed_index.block_membership_bytes ==
            packed_index.block_membership_bytes
    );
    ASSERT_TRUE(memcmp(
        direct_packed_index.block_membership,
        packed_index.block_membership,
        packed_index.block_membership_bytes
    ) == 0);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_serialize(
        &direct_packed_index,
        &direct_packed_serialized,
        &direct_packed_serialized_size
    ));
    ASSERT_TRUE(direct_packed_serialized_size == packed_serialized_size);
    ASSERT_TRUE(memcmp(
        direct_packed_serialized,
        packed_serialized,
        packed_serialized_size
    ) == 0);
    direct_packed_buffer = malloc(direct_packed_serialized_size);
    ASSERT_TRUE(direct_packed_buffer != NULL);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_serialize_into(
        &direct_packed_index,
        direct_packed_buffer,
        direct_packed_serialized_size
    ));
    ASSERT_TRUE(memcmp(
        direct_packed_buffer,
        direct_packed_serialized,
        direct_packed_serialized_size
    ) == 0);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_disk_header_decode(
        direct_packed_serialized,
        EVOKE_SEMANTIC_BMP_HEADER_SIZE,
        &packed_header
    ));
    ASSERT_TRUE(packed_header.total_size == direct_packed_serialized_size);
    ASSERT_TRUE(
        packed_header.block_membership_bytes ==
            direct_packed_index.block_membership_bytes
    );
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_term_decode(
        direct_packed_serialized + packed_header.terms_offset,
        EVOKE_SEMANTIC_BMP_PACKED_TERM_SIZE,
        &packed_term
    ));
    ASSERT_TRUE(packed_term.term_id == runs[0].term_id);
    ASSERT_TRUE(packed_term.block_membership_bytes > 0);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_super_ref_decode(
        direct_packed_serialized + packed_header.super_refs_offset +
            (size_t) packed_term.first_super_ref *
                EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE,
        EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE,
        packed_term.min_impact,
        packed_term.max_impact,
        &packed_super_ref
    ));
    ASSERT_TRUE(packed_super_ref.first_ref == packed_term.first_ref);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_ref_decode(
        direct_packed_serialized + packed_header.refs_offset +
            (size_t) packed_term.first_ref *
                EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
        EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
        packed_term.min_impact,
        packed_term.max_impact,
        &packed_ref
    ));
    ASSERT_TRUE(
        packed_ref.document_mask ==
            (UINT64_C(1) |
             (UINT64_C(1) << 17) |
             (UINT64_C(1) << 18) |
             (UINT64_C(1) << 35))
    );
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_term_materialize(
        &direct_packed_index,
        0,
        materialized_ids,
        materialized_values,
        sizeof(materialized_ids) / sizeof(materialized_ids[0])
    ));
    for (size_t posting_index = 0;
         posting_index < runs[0].posting_count;
         posting_index++)
    {
        ASSERT_TRUE(
            materialized_ids[posting_index] == run1_ids[posting_index]
        );
        ASSERT_TRUE(materialized_values[posting_index].impact ==
            run1_values[posting_index].impact);
    }
    ASSERT_TRUE(evoke_semantic_bmp_packed_serialize_into(
        &direct_packed_index,
        direct_packed_buffer,
        direct_packed_serialized_size - 1U
    ) == EVOKE_ERR_RANGE);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_topk(
        &direct_packed_index,
        query_ids,
        query_weights,
        sizeof(query_ids) / sizeof(query_ids[0]),
        3,
        &direct_packed_result,
        &packed_stats
    ));
    ASSERT_TRUE(direct_packed_result.len == result.len);
    for (size_t rank = 0; rank < result.len; rank++)
    {
        ASSERT_TRUE(
            direct_packed_result.doc_ids[rank] == result.doc_ids[rank]
        );
        ASSERT_TRUE(fabsf(
            direct_packed_result.scores[rank] - result.scores[rank]
        ) < 1e-6f);
    }
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_deserialize(
        packed_serialized,
        packed_serialized_size,
        &packed_restored
    ));
    ASSERT_TRUE(
        packed_restored.block_membership_bytes ==
            packed_index.block_membership_bytes
    );
    ASSERT_TRUE(memcmp(
        packed_restored.block_membership,
        packed_index.block_membership,
        packed_index.block_membership_bytes
    ) == 0);
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_topk(
        &packed_restored,
        query_ids,
        query_weights,
        sizeof(query_ids) / sizeof(query_ids[0]),
        3,
        &packed_restored_result,
        &packed_stats
    ));
    ASSERT_TRUE(packed_restored_result.len == result.len);
    for (size_t rank = 0; rank < result.len; rank++)
    {
        ASSERT_TRUE(
            packed_restored_result.doc_ids[rank] == result.doc_ids[rank]
        );
        ASSERT_TRUE(fabsf(
            packed_restored_result.scores[rank] - result.scores[rank]
        ) < 1e-6f);
    }
    packed_serialized[packed_serialized_size - 1U] ^= UINT8_C(0x01);
    {
        uint8_t membership = packed_restored.block_membership[0];

        packed_restored.block_membership[0] ^= UINT8_C(0x01);
        ASSERT_TRUE(evoke_semantic_bmp_packed_index_validate(
            &packed_restored
        ) == EVOKE_ERR_FORMAT);
        packed_restored.block_membership[0] = membership;
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_index_validate(
            &packed_restored
        ));
    }
    ASSERT_TRUE(evoke_semantic_bmp_packed_deserialize(
        packed_serialized,
        packed_serialized_size,
        &packed_restored
    ) == EVOKE_ERR_FORMAT);
    packed_serialized[packed_serialized_size - 1U] ^= UINT8_C(0x01);

    ASSERT_STATUS_OK(evoke_semantic_bmp_index_build_runs(
        80,
        runs,
        sizeof(runs) / sizeof(runs[0]),
        &run_index
    ));
    ASSERT_STATUS_OK(evoke_semantic_bmp_topk(
        &run_index,
        query_ids,
        query_weights,
        sizeof(query_ids) / sizeof(query_ids[0]),
        3,
        &run_result,
        &stats
    ));
    ASSERT_TRUE(run_result.len == result.len);
    for (size_t rank = 0; rank < result.len; rank++)
    {
        ASSERT_TRUE(run_result.doc_ids[rank] == result.doc_ids[rank]);
        ASSERT_TRUE(fabsf(run_result.scores[rank] -
            result.scores[rank]) < 1e-6f);
    }

    ASSERT_STATUS_OK(evoke_semantic_bmp_serialized_size(
        &index,
        &expected_size
    ));
    ASSERT_STATUS_OK(evoke_semantic_bmp_serialize(
        &index,
        &serialized,
        &serialized_size
    ));
    ASSERT_TRUE(serialized_size == expected_size);
    ASSERT_STATUS_OK(evoke_semantic_bmp_deserialize(
        serialized,
        serialized_size,
        &restored
    ));
    ASSERT_STATUS_OK(evoke_semantic_bmp_topk(
        &restored,
        query_ids,
        query_weights,
        sizeof(query_ids) / sizeof(query_ids[0]),
        3,
        &restored_result,
        &restored_stats
    ));
    ASSERT_TRUE(restored_result.len == result.len);
    for (size_t rank = 0; rank < result.len; rank++)
    {
        ASSERT_TRUE(restored_result.doc_ids[rank] == result.doc_ids[rank]);
        ASSERT_TRUE(fabsf(
            restored_result.scores[rank] - result.scores[rank]
        ) < 1e-6f);
    }
    serialized[serialized_size - 1] ^= UINT8_C(0x01);
    ASSERT_TRUE(evoke_semantic_bmp_deserialize(
        serialized,
        serialized_size,
        &restored
    ) == EVOKE_ERR_FORMAT);
    serialized[serialized_size - 1] ^= UINT8_C(0x01);
    {
        uint32_t record_index = restored.refs[0].record_index;

        restored.refs[0].record_index = restored.record_count;
        ASSERT_TRUE(evoke_semantic_bmp_index_validate(&restored) ==
            EVOKE_ERR_FORMAT);
        restored.refs[0].record_index = record_index;
    }
    restored.super_refs[0].ref_count = 0;
    ASSERT_TRUE(evoke_semantic_bmp_index_validate(&restored) ==
        EVOKE_ERR_FORMAT);

    free(serialized);
    free(packed_serialized);
    free(direct_packed_serialized);
    free(direct_packed_buffer);
    evoke_topk_result_free(&direct_packed_result);
    evoke_topk_result_free(&packed_restored_result);
    evoke_topk_result_free(&packed_taat_result);
    evoke_topk_result_free(&packed_result);
    evoke_topk_result_free(&run_result);
    evoke_topk_result_free(&restored_result);
    evoke_topk_result_free(&result);
    evoke_semantic_bmp_index_free(&run_index);
    evoke_semantic_bmp_index_free(&restored);
    evoke_semantic_bmp_index_free(&index);
    evoke_semantic_bmp_packed_index_free(&packed_index);
    evoke_semantic_bmp_packed_index_free(&direct_packed_index);
    evoke_semantic_bmp_packed_index_free(&packed_restored);

    {
        const uint32_t sparse_ids[] = {0};
        const evoke_posting_value sparse_values[] = {
            {.impact = 1.0f}
        };
        const evoke_semantic_bmp_run sparse_runs[] = {
            {7, 1, sparse_ids, sparse_values, NULL, 0, 257}
        };
        evoke_semantic_bmp_packed_index sparse_index;

        evoke_semantic_bmp_packed_index_init(&sparse_index);
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_index_build_runs(
            257,
            sparse_runs,
            sizeof(sparse_runs) / sizeof(sparse_runs[0]),
            &sparse_index
        ));
        ASSERT_TRUE(sparse_index.block_count == 5);
        ASSERT_TRUE(sparse_index.terms[0].ref_count == 1);
        ASSERT_TRUE(sparse_index.block_membership_bytes == 1);
        ASSERT_TRUE(sparse_index.terms[0].block_membership_bytes == 1);
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_index_validate(
            &sparse_index
        ));
        evoke_semantic_bmp_packed_index_free(&sparse_index);
    }
}

static uint32_t
test_semantic_bmp_random_u32(uint32_t *state)
{
    uint32_t value = *state;

    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static void
test_semantic_bmp_impact_precision_roundtrip(void)
{
    const uint32_t positive_ids[] = {0, 17, 35};
    const uint32_t secondary_ids[] = {1, 18, 36};
    const evoke_posting_value positive_values[] = {
        {.impact = 0.125f},
        {.impact = 1.234f},
        {.impact = 5.0f}
    };
    const evoke_posting_value secondary_values[] = {
        {.impact = 0.0625f},
        {.impact = 0.75f},
        {.impact = 3.5f}
    };
    const evoke_semantic_bmp_run runs[] = {
        {11, 3, positive_ids, positive_values, NULL, 0, 64},
        {19, 3, secondary_ids, secondary_values, NULL, 0, 64}
    };
    const evoke_semantic_impact_precision precisions[] = {
        EVOKE_SEMANTIC_IMPACT_PRECISION_F32,
        EVOKE_SEMANTIC_IMPACT_PRECISION_FP16,
        EVOKE_SEMANTIC_IMPACT_PRECISION_U8
    };
    size_t sizes[3] = {0};

    for (size_t precision_index = 0;
         precision_index < sizeof(precisions) / sizeof(precisions[0]);
         precision_index++)
    {
        evoke_semantic_bmp_packed_index index;
        evoke_semantic_bmp_packed_index restored;
        evoke_semantic_bmp_packed_disk_header header;
        uint8_t *serialized = NULL;
        uint8_t *reserialized = NULL;
        size_t serialized_size = 0;
        size_t reserialized_size = 0;

        evoke_semantic_bmp_packed_index_init(&index);
        evoke_semantic_bmp_packed_index_init(&restored);
        ASSERT_STATUS_OK(
            evoke_semantic_bmp_packed_index_build_runs_with_precision(
                64,
                runs,
                sizeof(runs) / sizeof(runs[0]),
                precisions[precision_index],
                &index
            )
        );
        ASSERT_TRUE(index.impact_precision == precisions[precision_index]);
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_serialize(
            &index,
            &serialized,
            &serialized_size
        ));
        sizes[precision_index] = serialized_size;
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_disk_header_decode(
            serialized,
            serialized_size,
            &header
        ));
        ASSERT_TRUE(header.impact_precision == precisions[precision_index]);
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_deserialize(
            serialized,
            serialized_size,
            &restored
        ));
        ASSERT_TRUE(restored.impact_precision == precisions[precision_index]);
        ASSERT_TRUE(restored.posting_count == index.posting_count);
        for (uint64_t posting_index = 0;
             posting_index < index.posting_count;
             posting_index++)
        {
            ASSERT_TRUE(index.impacts[posting_index] != 0.0f);
            ASSERT_TRUE(fabsf(
                restored.impacts[posting_index] - index.impacts[posting_index]
            ) < 1e-6f);
        }
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_serialize(
            &restored,
            &reserialized,
            &reserialized_size
        ));
        ASSERT_TRUE(reserialized_size == serialized_size);
        ASSERT_TRUE(memcmp(
            reserialized,
            serialized,
            serialized_size
        ) == 0);
        free(reserialized);
        free(serialized);
        evoke_semantic_bmp_packed_index_free(&restored);
        evoke_semantic_bmp_packed_index_free(&index);
    }
    ASSERT_TRUE(sizes[0] > sizes[1]);
    ASSERT_TRUE(sizes[1] > sizes[2]);
    ASSERT_TRUE(sizes[0] - sizes[1] == 6 * sizeof(uint16_t));
    ASSERT_TRUE(sizes[1] - sizes[2] == 6 * sizeof(uint8_t));

    for (size_t precision_index = 0;
         precision_index < sizeof(precisions) / sizeof(precisions[0]);
         precision_index++)
    {
        uint8_t encoded[
            (EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS + 1U) *
            sizeof(uint32_t)
        ] = {0};
        float decoded[EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS + 1U];
        size_t impact_width = evoke_semantic_bmp_impact_width(
            precisions[precision_index]
        );

        for (uint32_t index = 0;
             index < EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS + 1U;
             index++)
        {
            float quantized;

            ASSERT_STATUS_OK(evoke_semantic_impact_encode(
                encoded + (size_t) index * impact_width,
                sizeof(encoded) - (size_t) index * impact_width,
                precisions[precision_index],
                0.25f + (float) index / 128.0f,
                &quantized
            ));
        }
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_impacts_decode(
            encoded,
            sizeof(encoded),
            EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS,
            precisions[precision_index],
            0.25f,
            0.75f,
            decoded
        ));
        ASSERT_TRUE(evoke_semantic_bmp_packed_impacts_decode(
            encoded,
            sizeof(encoded),
            EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS + 1U,
            precisions[precision_index],
            0.25f,
            0.75f,
            decoded
        ) == EVOKE_ERR_INVALID);
    }

    {
        const uint32_t signed_ids[] = {0, 1};
        const evoke_posting_value signed_values[] = {
            {.impact = -0.5f},
            {.impact = 1.0f}
        };
        const evoke_semantic_bmp_run signed_run = {
            23,
            2,
            signed_ids,
            signed_values,
            NULL,
            0,
            64
        };
        evoke_semantic_bmp_packed_index index;

        evoke_semantic_bmp_packed_index_init(&index);
        ASSERT_TRUE(
            evoke_semantic_bmp_packed_index_build_runs_with_precision(
                64,
                &signed_run,
                1,
                EVOKE_SEMANTIC_IMPACT_PRECISION_U8,
                &index
            ) == EVOKE_ERR_RANGE
        );
        evoke_semantic_bmp_packed_index_free(&index);
    }
}

static void
test_semantic_bmp_randomized_exactness(void)
{
    enum
    {
        DOCUMENT_COUNT = 257,
        TERM_COUNT = 19,
        MAX_POSTINGS = DOCUMENT_COUNT * TERM_COUNT,
        TOP_K = 17
    };
    evoke_semantic_bmp_posting *postings = malloc(
        MAX_POSTINGS * sizeof(*postings)
    );

    ASSERT_TRUE(postings != NULL);
    for (uint32_t trial = 1; trial <= 24; trial++)
    {
        uint32_t query_ids[TERM_COUNT];
        float query_weights[TERM_COUNT];
        float dense_scores[DOCUMENT_COUNT] = {0};
        uint32_t state = UINT32_C(0x9e3779b9) ^ trial;
        size_t posting_count = 0;
        evoke_semantic_bmp_index index;
        evoke_semantic_bmp_packed_index packed_index;
        evoke_semantic_bmp_stats stats;
        evoke_semantic_bmp_stats packed_stats;
        evoke_semantic_bmp_stats packed_taat_stats;
        evoke_topk_result expected;
        evoke_topk_result actual;
        evoke_topk_result packed_actual;
        evoke_topk_result packed_taat_actual;

        evoke_semantic_bmp_index_init(&index);
        evoke_semantic_bmp_packed_index_init(&packed_index);
        memset(&stats, 0, sizeof(stats));
        memset(&packed_stats, 0, sizeof(packed_stats));
        memset(&packed_taat_stats, 0, sizeof(packed_taat_stats));
        memset(&expected, 0, sizeof(expected));
        memset(&actual, 0, sizeof(actual));
        memset(&packed_actual, 0, sizeof(packed_actual));
        memset(&packed_taat_actual, 0, sizeof(packed_taat_actual));
        for (uint32_t term = 0; term < TERM_COUNT; term++)
        {
            int32_t raw_weight = (int32_t)
                (test_semantic_bmp_random_u32(&state) % 2001) - 1000;

            query_ids[term] = 100 + term;
            query_weights[term] = raw_weight == 0
                ? 0.125f
                : (float) raw_weight / 113.0f;
            for (uint32_t document = 0;
                 document < DOCUMENT_COUNT;
                 document++)
            {
                int32_t raw_impact;
                float impact;

                if (test_semantic_bmp_random_u32(&state) % 7 != 0)
                {
                    continue;
                }
                raw_impact = (int32_t)
                    (test_semantic_bmp_random_u32(&state) % 2001) - 1000;
                impact = raw_impact == 0
                    ? -0.125f
                    : (float) raw_impact / 127.0f;
                postings[posting_count].term_id = query_ids[term];
                postings[posting_count].document_id = document;
                postings[posting_count].impact = impact;
                posting_count++;
                dense_scores[document] += query_weights[term] * impact;
            }
        }
        ASSERT_STATUS_OK(evoke_semantic_bmp_index_build(
            DOCUMENT_COUNT,
            postings,
            posting_count,
            &index
        ));
        ASSERT_STATUS_OK(evoke_topk(
            dense_scores,
            DOCUMENT_COUNT,
            TOP_K,
            true,
            &expected
        ));
        ASSERT_STATUS_OK(evoke_semantic_bmp_topk(
            &index,
            query_ids,
            query_weights,
            TERM_COUNT,
            TOP_K,
            &actual,
            &stats
        ));
        ASSERT_TRUE(actual.len == expected.len);
        for (size_t rank = 0; rank < expected.len; rank++)
        {
            ASSERT_TRUE(actual.doc_ids[rank] == expected.doc_ids[rank]);
            ASSERT_TRUE(fabsf(actual.scores[rank] -
                expected.scores[rank]) < 1e-4f);
        }
        ASSERT_TRUE(stats.postings_examined <= posting_count);
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_index_build(
            &index,
            &packed_index
        ));
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_topk(
            &packed_index,
            query_ids,
            query_weights,
            TERM_COUNT,
            TOP_K,
            &packed_actual,
            &packed_stats
        ));
        ASSERT_TRUE(packed_actual.len == expected.len);
        for (size_t rank = 0; rank < expected.len; rank++)
        {
            ASSERT_TRUE(
                packed_actual.doc_ids[rank] == expected.doc_ids[rank]
            );
            ASSERT_TRUE(fabsf(
                packed_actual.scores[rank] - expected.scores[rank]
            ) < 1e-4f);
        }
        ASSERT_TRUE(packed_stats.postings_examined <= posting_count);
        ASSERT_STATUS_OK(evoke_semantic_bmp_packed_taat_topk(
            &packed_index,
            query_ids,
            query_weights,
            TERM_COUNT,
            TOP_K,
            &packed_taat_actual,
            &packed_taat_stats
        ));
        ASSERT_TRUE(packed_taat_actual.len == expected.len);
        for (size_t rank = 0; rank < expected.len; rank++)
        {
            ASSERT_TRUE(
                packed_taat_actual.doc_ids[rank] == expected.doc_ids[rank]
            );
            ASSERT_TRUE(fabsf(
                packed_taat_actual.scores[rank] - expected.scores[rank]
            ) < 1e-4f);
        }
        ASSERT_TRUE(
            packed_taat_stats.postings_examined == posting_count
        );
        evoke_topk_result_free(&packed_taat_actual);
        evoke_topk_result_free(&packed_actual);
        evoke_topk_result_free(&actual);
        evoke_topk_result_free(&expected);
        evoke_semantic_bmp_index_free(&index);
        evoke_semantic_bmp_packed_index_free(&packed_index);
    }
    free(postings);
}

static void
test_semantic_bmp_packed_adaptive_fallback(void)
{
    enum
    {
        DOCUMENT_COUNT = 65 * 1024,
        TOP_K = 100
    };
    evoke_semantic_bmp_posting *postings = malloc(
        DOCUMENT_COUNT * sizeof(*postings)
    );
    uint32_t query_id = 42;
    float query_weight = 1.0f;
    evoke_semantic_bmp_index index;
    evoke_semantic_bmp_packed_index packed_index;
    evoke_semantic_bmp_stats adaptive_stats;
    evoke_semantic_bmp_stats taat_stats;
    evoke_topk_result adaptive;
    evoke_topk_result taat;

    ASSERT_TRUE(postings != NULL);
    evoke_semantic_bmp_index_init(&index);
    evoke_semantic_bmp_packed_index_init(&packed_index);
    memset(&adaptive_stats, 0, sizeof(adaptive_stats));
    memset(&taat_stats, 0, sizeof(taat_stats));
    memset(&adaptive, 0, sizeof(adaptive));
    memset(&taat, 0, sizeof(taat));
    for (uint32_t document_id = 0;
         document_id < DOCUMENT_COUNT;
         document_id++)
    {
        postings[document_id].term_id = query_id;
        postings[document_id].document_id = document_id;
        postings[document_id].impact = 1.0f;
    }
    ASSERT_STATUS_OK(evoke_semantic_bmp_index_build(
        DOCUMENT_COUNT,
        postings,
        DOCUMENT_COUNT,
        &index
    ));
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_index_build(
        &index,
        &packed_index
    ));
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_topk(
        &packed_index,
        &query_id,
        &query_weight,
        1,
        TOP_K,
        &adaptive,
        &adaptive_stats
    ));
    ASSERT_STATUS_OK(evoke_semantic_bmp_packed_taat_topk(
        &packed_index,
        &query_id,
        &query_weight,
        1,
        TOP_K,
        &taat,
        &taat_stats
    ));
    ASSERT_TRUE(adaptive_stats.adaptive_fallbacks == 1);
    ASSERT_TRUE(adaptive.len == taat.len);
    for (size_t rank = 0; rank < adaptive.len; rank++)
    {
        ASSERT_TRUE(adaptive.doc_ids[rank] == taat.doc_ids[rank]);
        ASSERT_TRUE(memcmp(
            &adaptive.scores[rank],
            &taat.scores[rank],
            sizeof(adaptive.scores[rank])
        ) == 0);
    }
    evoke_topk_result_free(&taat);
    evoke_topk_result_free(&adaptive);
    evoke_semantic_bmp_packed_index_free(&packed_index);
    evoke_semantic_bmp_index_free(&index);
    free(postings);
}

typedef struct test_semantic_accelerator_scores
{
    float scores[6][3];
    uint32_t fail_document_id;
    bool return_nan;
} test_semantic_accelerator_scores;

static evoke_status
test_semantic_accelerator_score(
    void *context,
    uint32_t document_id,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out
)
{
    const test_semantic_accelerator_scores *scores = context;
    float score = 0.0f;

    if (scores == NULL || score_out == NULL || document_id >= 6)
    {
        return EVOKE_ERR_INVALID;
    }
    if (document_id == scores->fail_document_id)
    {
        return EVOKE_ERR_INVALID;
    }
    if (scores->return_nan)
    {
        *score_out = NAN;
        return EVOKE_OK;
    }
    for (size_t query_index = 0;
         query_index < query_count;
         query_index++)
    {
        if (query_ids[query_index] >= 3)
        {
            return EVOKE_ERR_INVALID;
        }
        score += query_weights[query_index] *
            scores->scores[document_id][query_ids[query_index]];
    }
    *score_out = score;
    return EVOKE_OK;
}

typedef struct test_semantic_accelerator_documents
{
    evoke_semantic_accelerator_document_view documents[6];
} test_semantic_accelerator_documents;

static evoke_status
test_semantic_accelerator_read_document(
    void *context,
    uint32_t document_id,
    evoke_semantic_accelerator_document_view *view_out
)
{
    const test_semantic_accelerator_documents *documents = context;

    if (documents == NULL || view_out == NULL || document_id >= 6)
    {
        return EVOKE_ERR_INVALID;
    }
    *view_out = documents->documents[document_id];
    return EVOKE_OK;
}

static void
test_semantic_accelerator_geometric_builder(void)
{
    const uint32_t document_frequencies[] = {10, 1, 8, 0, 1};
    const uint32_t ids_0[] = {0, 1, 4};
    const uint32_t ids_1[] = {2, 3, 5};
    const uint32_t ids_2[] = {0, 1, 6};
    const uint32_t ids_3[] = {2, 3, 4};
    const uint32_t ids_4[] = {0, 1, 5};
    const uint32_t ids_5[] = {2, 3, 6};
    const float impacts_0[] = {1.0f, 0.9f, 0.1f};
    const float impacts_1[] = {0.9f, 1.0f, 0.2f};
    const float impacts_2[] = {0.8f, 0.7f, 0.3f};
    const float impacts_3[] = {1.0f, 0.9f, 0.1f};
    const float impacts_4[] = {0.9f, 1.0f, 0.2f};
    const float impacts_5[] = {0.8f, 0.7f, 0.3f};
    const uint32_t retained_documents[] = {5, 2, 0, 4, 1, 3};
    const uint32_t duplicate_documents[] = {0, 0};
    test_semantic_accelerator_documents documents = {
        .documents = {
            {ids_0, impacts_0, 3},
            {ids_1, impacts_1, 3},
            {ids_2, impacts_2, 3},
            {ids_3, impacts_3, 3},
            {ids_4, impacts_4, 3},
            {ids_5, impacts_5, 3}
        }
    };
    evoke_semantic_accelerator_builder_options options = {
        .centroid_fraction = 0.5f,
        .minimum_cluster_size = 1,
        .document_cut = 2,
        .summary_energy = 0.7f,
        .random_seed = 1142
    };
    evoke_semantic_accelerator_owned_term first;
    evoke_semantic_accelerator_owned_term second;
    evoke_semantic_accelerator_index index;
    evoke_semantic_accelerator_index term_object;
    uint8_t *term_bytes = NULL;
    size_t term_size = 0;
    uint32_t *selected_terms = NULL;
    uint32_t selected_term_count = 0;
    float actual_posting_mass = 0.0f;
    uint32_t seen = 0;

    evoke_semantic_accelerator_owned_term_init(&first);
    evoke_semantic_accelerator_owned_term_init(&second);
    evoke_semantic_accelerator_index_init(&index);
    evoke_semantic_accelerator_index_init(&term_object);
    ASSERT_STATUS_OK(evoke_semantic_accelerator_select_terms(
        document_frequencies,
        5,
        0.75f,
        &selected_terms,
        &selected_term_count,
        &actual_posting_mass
    ));
    ASSERT_TRUE(selected_term_count == 2);
    ASSERT_TRUE(selected_terms[0] == 0);
    ASSERT_TRUE(selected_terms[1] == 2);
    ASSERT_TRUE(fabsf(actual_posting_mass - 0.9f) < 1e-6f);
    free(selected_terms);
    ASSERT_STATUS_OK(evoke_semantic_accelerator_build_term(
        17,
        retained_documents,
        6,
        &options,
        test_semantic_accelerator_read_document,
        &documents,
        &first
    ));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_build_term(
        17,
        retained_documents,
        6,
        &options,
        test_semantic_accelerator_read_document,
        &documents,
        &second
    ));
    ASSERT_TRUE(first.input.term_id == 17);
    ASSERT_TRUE(first.input.cluster_count > 0);
    ASSERT_TRUE(first.input.cluster_count <= 3);
    ASSERT_TRUE(first.input.cluster_count == second.input.cluster_count);
    for (uint32_t cluster = 0;
         cluster < first.input.cluster_count;
         cluster++)
    {
        const evoke_semantic_accelerator_cluster_input *a =
            &first.clusters[cluster];
        const evoke_semantic_accelerator_cluster_input *b =
            &second.clusters[cluster];
        bool even_document_cluster;

        ASSERT_TRUE(a->document_count == b->document_count);
        ASSERT_TRUE(a->summary_count == b->summary_count);
        ASSERT_TRUE(a->document_count > 0);
        ASSERT_TRUE(a->summary_count > 0);
        even_document_cluster = (a->document_ids[0] & 1U) == 0;
        for (uint32_t position = 0;
             position < a->document_count;
             position++)
        {
            ASSERT_TRUE(a->document_ids[position] ==
                b->document_ids[position]);
            ASSERT_TRUE(
                ((a->document_ids[position] & 1U) == 0) ==
                even_document_cluster
            );
            if (position > 0)
            {
                ASSERT_TRUE(a->document_ids[position - 1U] <
                    a->document_ids[position]);
            }
            seen |= UINT32_C(1) << a->document_ids[position];
        }
        for (uint32_t position = 0;
             position < a->summary_count;
             position++)
        {
            ASSERT_TRUE(a->summary[position].term_id ==
                b->summary[position].term_id);
            ASSERT_TRUE(fabsf(
                a->summary[position].max_impact -
                b->summary[position].max_impact
            ) < 1e-7f);
            if (position > 0)
            {
                ASSERT_TRUE(a->summary[position - 1U].term_id <
                    a->summary[position].term_id);
            }
        }
    }
    ASSERT_TRUE(seen == UINT32_C(0x3f));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_index_build(
        UINT64_C(0x89abcdef01234567),
        6,
        &first.input,
        1,
        &index
    ));
    ASSERT_TRUE(index.source_root_checksum ==
        UINT64_C(0x89abcdef01234567));
    ASSERT_TRUE(evoke_semantic_accelerator_root_matches(
        &index,
        UINT64_C(0x89abcdef01234567)
    ));
    ASSERT_TRUE(!evoke_semantic_accelerator_root_matches(
        &index,
        UINT64_C(0x89abcdef01234568)
    ));
    ASSERT_TRUE(index.document_ref_count == 6);
    ASSERT_STATUS_OK(evoke_semantic_accelerator_term_serialize(
        UINT64_C(0x89abcdef01234567),
        6,
        &first.input,
        &term_bytes,
        &term_size
    ));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_term_deserialize(
        term_bytes,
        term_size,
        UINT64_C(0x89abcdef01234567),
        17,
        &term_object
    ));
    ASSERT_TRUE(term_object.term_count == 1);
    ASSERT_TRUE(term_object.document_ref_count == 6);
    evoke_semantic_accelerator_index_free(&term_object);
    ASSERT_TRUE(evoke_semantic_accelerator_term_deserialize(
        term_bytes,
        term_size,
        UINT64_C(0x89abcdef01234568),
        17,
        &term_object
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(evoke_semantic_accelerator_term_deserialize(
        term_bytes,
        term_size,
        UINT64_C(0x89abcdef01234567),
        18,
        &term_object
    ) == EVOKE_ERR_FORMAT);
    free(term_bytes);
    ASSERT_TRUE(evoke_semantic_accelerator_build_term(
        17,
        duplicate_documents,
        2,
        &options,
        test_semantic_accelerator_read_document,
        &documents,
        &second
    ) == EVOKE_ERR_INVALID);
    evoke_semantic_accelerator_index_free(&index);
    evoke_semantic_accelerator_index_free(&term_object);
    evoke_semantic_accelerator_owned_term_free(&second);
    evoke_semantic_accelerator_owned_term_free(&first);
}

static void
test_semantic_accelerator_roundtrip_and_search(void)
{
    const uint32_t term0_cluster0_documents[] = {1, 0};
    const uint32_t term0_cluster1_documents[] = {2, 4};
    const uint32_t term1_cluster0_documents[] = {3, 1};
    const uint32_t term1_cluster1_documents[] = {4, 5};
    const evoke_semantic_accelerator_summary_input summary_a[] = {
        {0, 1.0f}, {1, 0.8f}
    };
    const evoke_semantic_accelerator_summary_input summary_b[] = {
        {0, 0.8f}, {1, 1.0f}
    };
    const evoke_semantic_accelerator_summary_input summary_c[] = {
        {0, 0.9f}, {1, 2.0f}
    };
    const evoke_semantic_accelerator_summary_input summary_d[] = {
        {0, 0.2f}, {1, 1.0f}
    };
    const evoke_semantic_accelerator_cluster_input term0_clusters[] = {
        {
            term0_cluster0_documents,
            2,
            summary_a,
            2
        },
        {
            term0_cluster1_documents,
            2,
            summary_b,
            2
        }
    };
    const evoke_semantic_accelerator_cluster_input term1_clusters[] = {
        {
            term1_cluster0_documents,
            2,
            summary_c,
            2
        },
        {
            term1_cluster1_documents,
            2,
            summary_d,
            2
        }
    };
    const evoke_semantic_accelerator_term_input terms[] = {
        {0, term0_clusters, 2},
        {1, term1_clusters, 2}
    };
    const uint32_t query_ids[] = {0, 1};
    const float query_weights[] = {1.0f, 0.5f};
    const uint32_t expected_ids[] = {1, 0, 3};
    const float expected_scores[] = {1.3f, 1.0f, 1.0f};
    test_semantic_accelerator_scores scores = {
        .scores = {
            {1.0f, 0.0f, 0.0f},
            {0.9f, 0.8f, 0.0f},
            {0.8f, 0.0f, 0.0f},
            {0.0f, 2.0f, 0.0f},
            {0.2f, 1.0f, 0.0f},
            {0.0f, 0.1f, 0.0f}
        },
        .fail_document_id = UINT32_MAX
    };
    evoke_semantic_accelerator_options options = {
        .query_cut = 2,
        .candidate_multiplier = 1,
        .heap_factor = 1.0f
    };
    evoke_semantic_accelerator_index index;
    evoke_semantic_accelerator_index restored;
    evoke_semantic_accelerator_index term_indexes[2];
    const evoke_semantic_accelerator_index *term_index_refs[2];
    evoke_semantic_accelerator_stats stats;
    evoke_semantic_accelerator_stats restored_stats;
    evoke_semantic_accelerator_stats split_stats;
    evoke_semantic_accelerator_stats block_stats;
    evoke_semantic_accelerator_query_scratch *query_scratch;
    evoke_topk_result result;
    evoke_topk_result restored_result;
    evoke_topk_result split_result;
    evoke_topk_result block_result;
    uint8_t *serialized = NULL;
    size_t serialized_size = 0;

    evoke_semantic_accelerator_index_init(&index);
    evoke_semantic_accelerator_index_init(&restored);
    evoke_semantic_accelerator_index_init(&term_indexes[0]);
    evoke_semantic_accelerator_index_init(&term_indexes[1]);
    term_index_refs[0] = &term_indexes[0];
    term_index_refs[1] = &term_indexes[1];
    memset(&stats, 0, sizeof(stats));
    memset(&restored_stats, 0, sizeof(restored_stats));
    memset(&split_stats, 0, sizeof(split_stats));
    memset(&block_stats, 0, sizeof(block_stats));
    memset(&result, 0, sizeof(result));
    memset(&restored_result, 0, sizeof(restored_result));
    memset(&split_result, 0, sizeof(split_result));
    memset(&block_result, 0, sizeof(block_result));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_index_build(
        UINT64_C(0x1020304050607080),
        6,
        terms,
        2,
        &index
    ));
    ASSERT_TRUE(index.term_count == 2);
    ASSERT_TRUE(index.source_root_checksum ==
        UINT64_C(0x1020304050607080));
    ASSERT_TRUE(index.cluster_count == 4);
    ASSERT_TRUE(index.document_ref_count == 8);
    ASSERT_TRUE(index.summary_count == 8);
    for (uint32_t cluster_index = 0;
         cluster_index < index.cluster_count;
         cluster_index++)
    {
        const evoke_semantic_accelerator_cluster *cluster =
            &index.clusters[cluster_index];

        for (uint32_t offset = 0;
             offset < cluster->summary_count;
             offset++)
        {
            const evoke_semantic_accelerator_summary *summary =
                &index.summaries[cluster->first_summary + offset];
            float decoded = summary->quantized_impact * cluster->quantum;
            const evoke_semantic_accelerator_summary_input *source =
                terms[cluster_index / 2].clusters[cluster_index % 2].summary;

            ASSERT_TRUE(decoded + 1e-7f >= source[offset].max_impact);
        }
    }
    ASSERT_STATUS_OK(evoke_semantic_accelerator_topk(
        &index,
        query_ids,
        query_weights,
        2,
        3,
        &options,
        test_semantic_accelerator_score,
        &scores,
        &result,
        &stats
    ));
    ASSERT_TRUE(result.len == 3);
    for (size_t rank = 0; rank < result.len; rank++)
    {
        ASSERT_TRUE(result.doc_ids[rank] == expected_ids[rank]);
        ASSERT_TRUE(fabsf(result.scores[rank] - expected_scores[rank]) <
            1e-6f);
    }
    ASSERT_TRUE(stats.clusters_opened == 3);
    ASSERT_TRUE(stats.clusters_skipped == 1);
    ASSERT_TRUE(stats.documents_scored == 5);
    ASSERT_TRUE(stats.duplicate_documents_skipped == 1);
    ASSERT_TRUE(stats.query_scratch_peak_bytes > 0);
    ASSERT_STATUS_OK(evoke_semantic_accelerator_index_build(
        index.source_root_checksum,
        index.document_count,
        &terms[0],
        1,
        &term_indexes[0]
    ));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_index_build(
        index.source_root_checksum,
        index.document_count,
        &terms[1],
        1,
        &term_indexes[1]
    ));
    query_scratch = evoke_semantic_accelerator_query_scratch_create();
    ASSERT_TRUE(query_scratch != NULL);
    ASSERT_STATUS_OK(evoke_semantic_accelerator_topk_many_owned(
        term_index_refs,
        2,
        query_ids,
        query_weights,
        2,
        3,
        &options,
        test_semantic_accelerator_score,
        &scores,
        &split_result,
        &split_stats,
        query_scratch
    ));
    ASSERT_TRUE(split_result.len == result.len);
    for (size_t rank = 0; rank < result.len; rank++)
    {
        ASSERT_TRUE(split_result.doc_ids[rank] == result.doc_ids[rank]);
        ASSERT_TRUE(fabsf(
            split_result.scores[rank] - result.scores[rank]
        ) < 1e-6f);
    }
    ASSERT_TRUE(split_stats.documents_scored == stats.documents_scored);
    evoke_topk_result_free(&split_result);
    scores.fail_document_id = 1;
    ASSERT_TRUE(evoke_semantic_accelerator_topk_many_owned(
        term_index_refs,
        2,
        query_ids,
        query_weights,
        2,
        3,
        &options,
        test_semantic_accelerator_score,
        &scores,
        &split_result,
        &split_stats,
        query_scratch
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(split_result.len == 0);
    scores.fail_document_id = UINT32_MAX;
    ASSERT_STATUS_OK(evoke_semantic_accelerator_topk_many_owned(
        term_index_refs,
        2,
        query_ids,
        query_weights,
        2,
        3,
        &options,
        test_semantic_accelerator_score,
        &scores,
        &split_result,
        &split_stats,
        query_scratch
    ));
    ASSERT_TRUE(split_result.len == result.len);
    evoke_semantic_accelerator_query_scratch_destroy(query_scratch);
    ASSERT_STATUS_OK(evoke_semantic_accelerator_serialize(
        &index,
        &serialized,
        &serialized_size
    ));
    ASSERT_TRUE(serialized_size > EVOKE_SEMANTIC_ACCELERATOR_HEADER_SIZE);
    ASSERT_STATUS_OK(evoke_semantic_accelerator_deserialize(
        serialized,
        serialized_size,
        &restored
    ));
    ASSERT_TRUE(restored.source_root_checksum ==
        index.source_root_checksum);
    ASSERT_TRUE(evoke_semantic_accelerator_root_matches(
        &restored,
        index.source_root_checksum
    ));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_topk(
        &restored,
        query_ids,
        query_weights,
        2,
        3,
        &options,
        test_semantic_accelerator_score,
        &scores,
        &restored_result,
        &restored_stats
    ));
    ASSERT_TRUE(restored_result.len == result.len);
    for (size_t rank = 0; rank < result.len; rank++)
    {
        ASSERT_TRUE(restored_result.doc_ids[rank] == result.doc_ids[rank]);
        ASSERT_TRUE(fabsf(
            restored_result.scores[rank] - result.scores[rank]
        ) < 1e-6f);
    }
    evoke_topk_result_free(&restored_result);
    {
        evoke_semantic_accelerator_options block_options = options;

        block_options.document_shift = 3;
        block_options.heap_factor = 0.0f;
        ASSERT_STATUS_OK(evoke_semantic_accelerator_topk_many(
            term_index_refs,
            2,
            query_ids,
            query_weights,
            2,
            3,
            &block_options,
            test_semantic_accelerator_score,
            &scores,
            &block_result,
            &block_stats
        ));
        ASSERT_TRUE(block_result.len == result.len);
        for (size_t rank = 0; rank < result.len; rank++)
        {
            ASSERT_TRUE(block_result.doc_ids[rank] == result.doc_ids[rank]);
            ASSERT_TRUE(fabsf(
                block_result.scores[rank] - result.scores[rank]
            ) < 1e-6f);
        }
        ASSERT_TRUE(block_stats.clusters_opened == 4);
        ASSERT_TRUE(block_stats.clusters_skipped == 0);
        ASSERT_TRUE(block_stats.documents_scored == 6);
        ASSERT_TRUE(block_stats.duplicate_documents_skipped == 2);
        ASSERT_TRUE(block_stats.block_major_query);
        ASSERT_TRUE(block_stats.query_scratch_peak_bytes > 0);
    }
    serialized[serialized_size - 1U] ^= UINT8_C(1);
    ASSERT_TRUE(evoke_semantic_accelerator_deserialize(
        serialized,
        serialized_size,
        &restored
    ) == EVOKE_ERR_FORMAT);
    serialized[serialized_size - 1U] ^= UINT8_C(1);
    {
        float negative_weight = -1.0f;

        ASSERT_TRUE(evoke_semantic_accelerator_topk(
            &index,
            query_ids,
            &negative_weight,
            1,
            3,
            &options,
            test_semantic_accelerator_score,
            &scores,
            &restored_result,
            &restored_stats
        ) == EVOKE_ERR_INVALID);
    }
    {
        uint32_t original = index.terms[1].first_cluster;

        index.terms[1].first_cluster = original - 1U;
        ASSERT_TRUE(evoke_semantic_accelerator_index_validate(&index) ==
            EVOKE_ERR_FORMAT);
        index.terms[1].first_cluster = original;
    }
    {
        float original = index.clusters[0].quantum;

        index.clusters[0].quantum = 0.0f;
        ASSERT_TRUE(evoke_semantic_accelerator_index_validate(&index) ==
            EVOKE_ERR_FORMAT);
        index.clusters[0].quantum = original;
    }
    {
        uint64_t original = index.source_root_checksum;

        index.source_root_checksum = 0;
        ASSERT_TRUE(evoke_semantic_accelerator_index_validate(&index) ==
            EVOKE_ERR_INVALID);
        index.source_root_checksum = original;
    }
    scores.fail_document_id = 1;
    ASSERT_TRUE(evoke_semantic_accelerator_topk(
        &index,
        query_ids,
        query_weights,
        2,
        3,
        &options,
        test_semantic_accelerator_score,
        &scores,
        &restored_result,
        &restored_stats
    ) == EVOKE_ERR_INVALID);
    ASSERT_TRUE(restored_result.len == 0);
    scores.fail_document_id = UINT32_MAX;
    scores.return_nan = true;
    ASSERT_TRUE(evoke_semantic_accelerator_topk(
        &index,
        query_ids,
        query_weights,
        2,
        3,
        &options,
        test_semantic_accelerator_score,
        &scores,
        &restored_result,
        &restored_stats
    ) == EVOKE_ERR_FORMAT);
    ASSERT_TRUE(restored_result.len == 0);
    free(serialized);
    evoke_topk_result_free(&block_result);
    evoke_topk_result_free(&split_result);
    evoke_topk_result_free(&restored_result);
    evoke_topk_result_free(&result);
    evoke_semantic_accelerator_index_free(&term_indexes[1]);
    evoke_semantic_accelerator_index_free(&term_indexes[0]);
    evoke_semantic_accelerator_index_free(&restored);
    evoke_semantic_accelerator_index_free(&index);
}

static void
test_semantic_accelerator_candidate_oversampling(void)
{
    const uint32_t first_documents[] = {0};
    const uint32_t second_documents[] = {1};
    const evoke_semantic_accelerator_summary_input first_summary[] = {
        {0, 0.9f}
    };
    const evoke_semantic_accelerator_summary_input second_summary[] = {
        {0, 0.1f}
    };
    const evoke_semantic_accelerator_cluster_input clusters[] = {
        {first_documents, 1, first_summary, 1},
        {second_documents, 1, second_summary, 1}
    };
    const evoke_semantic_accelerator_term_input terms[] = {
        {0, clusters, 2}
    };
    const uint32_t query_ids[] = {0, 1};
    const float query_weights[] = {1.0f, 0.5f};
    test_semantic_accelerator_scores scores = {
        .scores = {
            {0.9f, 0.0f, 0.0f},
            {0.1f, 2.0f, 0.0f}
        },
        .fail_document_id = UINT32_MAX
    };
    evoke_semantic_accelerator_options narrow = {
        .query_cut = 1,
        .candidate_multiplier = 1,
        .heap_factor = 1.0f
    };
    evoke_semantic_accelerator_options oversampled = {
        .query_cut = 1,
        .candidate_multiplier = 2,
        .heap_factor = 1.0f
    };
    evoke_semantic_accelerator_index index;
    evoke_semantic_accelerator_stats stats;
    evoke_topk_result result;

    evoke_semantic_accelerator_index_init(&index);
    memset(&stats, 0, sizeof(stats));
    memset(&result, 0, sizeof(result));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_index_build(
        UINT64_C(0x1020304050607080),
        2,
        terms,
        1,
        &index
    ));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_topk(
        &index,
        query_ids,
        query_weights,
        2,
        1,
        &narrow,
        test_semantic_accelerator_score,
        &scores,
        &result,
        &stats
    ));
    ASSERT_TRUE(result.len == 1);
    ASSERT_TRUE(result.doc_ids[0] == 0);
    ASSERT_TRUE(stats.clusters_opened == 1);
    evoke_topk_result_free(&result);
    ASSERT_STATUS_OK(evoke_semantic_accelerator_topk(
        &index,
        query_ids,
        query_weights,
        2,
        1,
        &oversampled,
        test_semantic_accelerator_score,
        &scores,
        &result,
        &stats
    ));
    ASSERT_TRUE(result.len == 1);
    ASSERT_TRUE(result.doc_ids[0] == 1);
    ASSERT_TRUE(stats.clusters_opened == 2);
    evoke_topk_result_free(&result);
    oversampled.candidate_multiplier = 0;
    ASSERT_TRUE(evoke_semantic_accelerator_topk(
        &index,
        query_ids,
        query_weights,
        2,
        1,
        &oversampled,
        test_semantic_accelerator_score,
        &scores,
        &result,
        &stats
    ) == EVOKE_ERR_INVALID);
    evoke_semantic_accelerator_index_free(&index);
}

static void
test_semantic_accelerator_empty_and_duplicate_query(void)
{
    const uint32_t documents[] = {0};
    const evoke_semantic_accelerator_summary_input summary[] = {
        {0, 1.0f}
    };
    const evoke_semantic_accelerator_cluster_input clusters[] = {
        {documents, 1, summary, 1}
    };
    const evoke_semantic_accelerator_term_input terms[] = {
        {0, clusters, 1}
    };
    const uint32_t query_ids[] = {0, 0};
    const float query_weights[] = {0.25f, 0.75f};
    test_semantic_accelerator_scores scores = {
        .scores = {{1.0f, 0.0f, 0.0f}},
        .fail_document_id = UINT32_MAX
    };
    evoke_semantic_accelerator_options options = {
        .query_cut = 2,
        .candidate_multiplier = 1,
        .heap_factor = 0.0f
    };
    evoke_semantic_accelerator_index empty;
    evoke_semantic_accelerator_index index;
    evoke_semantic_accelerator_index restored;
    evoke_semantic_accelerator_stats stats;
    evoke_topk_result result;
    uint8_t *bytes = NULL;
    size_t size = 0;

    evoke_semantic_accelerator_index_init(&empty);
    evoke_semantic_accelerator_index_init(&index);
    evoke_semantic_accelerator_index_init(&restored);
    memset(&stats, 0, sizeof(stats));
    memset(&result, 0, sizeof(result));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_index_build(
        UINT64_C(0x1020304050607080),
        0,
        NULL,
        0,
        &empty
    ));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_serialize(
        &empty,
        &bytes,
        &size
    ));
    ASSERT_TRUE(size == EVOKE_SEMANTIC_ACCELERATOR_HEADER_SIZE);
    ASSERT_STATUS_OK(evoke_semantic_accelerator_deserialize(
        bytes,
        size,
        &restored
    ));
    free(bytes);
    bytes = NULL;
    ASSERT_STATUS_OK(evoke_semantic_accelerator_topk(
        &restored,
        NULL,
        NULL,
        0,
        0,
        &options,
        test_semantic_accelerator_score,
        &scores,
        &result,
        &stats
    ));
    ASSERT_TRUE(result.len == 0);
    ASSERT_STATUS_OK(evoke_semantic_accelerator_index_build(
        UINT64_C(0x1020304050607080),
        1,
        terms,
        1,
        &index
    ));
    ASSERT_STATUS_OK(evoke_semantic_accelerator_topk(
        &index,
        query_ids,
        query_weights,
        2,
        1,
        &options,
        test_semantic_accelerator_score,
        &scores,
        &result,
        &stats
    ));
    ASSERT_TRUE(result.len == 1);
    ASSERT_TRUE(result.doc_ids[0] == 0);
    ASSERT_TRUE(fabsf(result.scores[0] - 1.0f) < 1e-6f);
    evoke_topk_result_free(&result);
    evoke_semantic_accelerator_index_free(&restored);
    evoke_semantic_accelerator_index_free(&index);
    evoke_semantic_accelerator_index_free(&empty);
}

static void
test_semantic_accelerator_workspace_estimate(void)
{
    uint64_t fixed = UINT64_C(64) * 1024 * 1024;
    uint64_t estimate = evoke_semantic_accelerator_workspace_estimate(
        100,
        80,
        10,
        1000
    );

    ASSERT_TRUE(
        estimate == fixed + UINT64_C(100) * 72 +
            UINT64_C(80) * 640 + UINT64_C(10) * 32 +
            UINT64_C(1000)
    );
    ASSERT_TRUE(
        evoke_semantic_accelerator_workspace_estimate(
            UINT64_MAX,
            UINT64_MAX,
            UINT64_MAX,
            UINT64_MAX
        ) == UINT64_MAX
    );
}

static void
test_weighted_space_saving_bounds(void)
{
    evoke_weighted_space_saving summary;
    double exact[256] = {0};
    uint32_t state = UINT32_C(0x12345678);

    evoke_weighted_space_saving_init_empty(&summary);
    ASSERT_STATUS_OK(evoke_weighted_space_saving_init(&summary, 16));
    for (size_t update = 0; update < 10000; update++)
    {
        uint32_t item;
        double weight;

        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        item = (state >> 16) & UINT32_C(255);
        weight = (double) ((state & UINT32_C(7)) + 1U) / 8.0;
        exact[item] += weight;
        ASSERT_STATUS_OK(evoke_weighted_space_saving_offer(
            &summary,
            item,
            weight
        ));
    }
    ASSERT_TRUE(summary.len == 16);
    ASSERT_TRUE(summary.replacements > 0);
    ASSERT_TRUE(summary.maximum_error <= summary.total_weight / 16.0);
    for (size_t index = 0; index < summary.len; index++)
    {
        const evoke_weighted_space_saving_counter *counter =
            &summary.counters[index];

        ASSERT_TRUE(counter->estimate + 1e-9 >= exact[counter->item]);
        ASSERT_TRUE(
            counter->estimate - counter->error <=
                exact[counter->item] + 1e-9
        );
        ASSERT_TRUE(summary.heap[counter->heap_position] == index);
    }
    ASSERT_TRUE(
        evoke_weighted_space_saving_offer(&summary, 1, 0.0) ==
            EVOKE_ERR_INVALID
    );
    evoke_weighted_space_saving_free(&summary);
}

int
main(void)
{
    test_u32_saturating_add();
    test_u64_saturating_arithmetic();
    test_semantic_bmp_exact_signed_topk();
    test_semantic_bmp_impact_precision_roundtrip();
    test_semantic_bmp_randomized_exactness();
    test_semantic_bmp_packed_adaptive_fallback();
    test_semantic_accelerator_geometric_builder();
    test_semantic_accelerator_roundtrip_and_search();
    test_semantic_accelerator_candidate_oversampling();
    test_semantic_accelerator_empty_and_duplicate_query();
    test_semantic_accelerator_workspace_estimate();
    test_weighted_space_saving_bounds();
    test_block_range_inventory_classifies_physical_debt();
    test_segment_page_header_roundtrip_and_validation();
    test_active_l0_header_and_frontier_validation();
    test_l0_record_and_frame_roundtrip();
    test_segment_read_root_validation();
    test_segment_manifest_roundtrip_and_validation();
    test_cow_segment_manifest_roundtrip_and_validation();
    test_semantic_accelerator_manifest_lifecycle();
    test_document_tid_lookup_roundtrip();
    test_semantic_accelerator_directory_roundtrip();
    test_scope_roundtrip_and_validation();
    test_scope_ascii_ilike_contains();
    test_semantic_forward_chunk_roundtrip();
    test_semantic_forward_bound_roundtrip();
    test_semantic_impact_frontier_roundtrip();
    test_semantic_impact_frontier_precision_profiles();
    test_semantic_forward_adaptive_dense_lane();
    test_segment_manifest_published_closure();
    test_segment_query_contract_roundtrip();
    test_lexical_catalog_roundtrip_and_validation();
    test_term_directory_roundtrip_and_validation();
    test_term_fold_bundle_roundtrip_and_validation();
    test_document_version_records_validation();
    test_aborted_slot_hole_metadata();
    test_frozen_retirement_ids_build();
    test_segment_payload_roundtrip_and_validation();
    test_contiguous_rebuild_payload_partition();
    test_sparse_contiguous_rebuild_payload_partition();
    test_semantic_state_segment_and_merge();
    test_attach_sorted_semantic_stream_matches_array();
    test_initial_fold_stream_matches_combined_payload();
    test_attach_semantic_postings_to_lexical_payload();
    test_sparse_lexical_entries_build_canonical_segment();
    test_lexical_index_builds_canonical_segment();
    test_segment_payload_merge_and_directory_replacement();
    test_segment_read_view_matches_expanded_index();
    test_empty_term_cow_external_closure();
    test_initial_folded_term_cow_tree();
    test_term_cow_neutral_fold_patch();
    test_term_cow_external_append_patch();
    test_term_cow_external_replace_patch();
    test_term_cow_tree_incremental_append();
    test_lexicon_cow_restart_and_external_patch();
    test_lexicon_cow_incremental_suffix();
    test_lexicon_cow_large_vocabulary_update_is_bounded();
    test_lexicon_cow_compact_sealed_geometry();
    test_prefix_cow_range_seek_and_external_patch();
    test_prefix_cow_parent_split_rebalances();
    test_document_cow_record_identity_equality();
    test_document_cow_tree_duplicate_object_guard();
    test_document_cow_pending_frontier_and_patch();
    test_document_cow_partial_leaf_append_and_transition_guards();
    test_document_cow_initial_semantic_completion();
    test_document_cow_l0_owned_transitions();
    test_document_cow_l0_semantic_transition_during_seal();
    test_document_cow_root_relative_reuse_guards();
    test_mixed_extent_scoring_combines_lexical_and_semantic();
    test_empty_mixed_extent_scoring();
    test_retired_mixed_scoring_matches_live_rebuild();
    test_lucene_index_layout();
    test_compact_id_builder_matches_standard();
    test_compact_token_builder_matches_standard();
    test_builders_reject_invalid_inputs();
    test_empty_index_builders_and_roundtrip();
    test_bm25plus_scores_and_weight_mask();
    test_token_index_and_query_mapping();
    test_exact_stats_scoring_matches_dense_ids();
    test_exact_stats_scoring_matches_dense_tokens();
    test_fragmented_extent_scoring_matches_contiguous();
    test_document_block_bounds_are_conservative();
    test_global_blockmax_matches_exact_mixed_scoring();
    test_posting_heat_bounded_decay_and_root_stability();
    test_neutral_segment_scoring_survives_statistics_change();
    test_neutral_scoring_matches_all_methods();
    test_serialization_roundtrip();
    test_storage_version_boundary();
    test_stream_serialization_matches_buffered();
    test_deserialize_rejects_corrupt_postings();
    test_serialization_rejects_inconsistent_index();
    test_deserialize_mutation_safety();
    test_topk_numpy_compatible_shape();
    test_topk_subset_buffered_keeps_stable_threshold();
    test_topk_subset_compact_matches_identity_tie_breaks();
    test_topk_custom_tie_breaks();
    test_streaming_topk_matches_array_tie_breaks();
    test_blockmax_zero_score_uses_ordered_prefix();
    test_query_parser_basic_terms();
    test_query_parser_occurs_prefix_and_phrase();
    test_query_parser_textual_boolean_aliases();
    test_query_parser_invalid_inputs();
    test_query_parser_complexity_limits();
    test_query_parser_simple_term_detection();
    test_text_normalize_token();
    test_text_normalize_query();
    test_text_normalize_boolean_query();
    test_text_tokenize_text();
    test_text_normalize_token_with_stemming();
    test_text_normalize_query_with_stemming();
    test_text_tokenize_text_with_stemming();
    test_text_normalize_token_with_diacritic_folding();
    test_text_tokenize_text_with_diacritic_folding();
    puts("all unit tests passed");
    return 0;
}
