#include "evoke_segments.h"
#include "evoke_semantic_bmp.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_SEGMENT_MANIFEST_MAGIC UINT32_C(0x4D534932)
#define EVOKE_SEGMENT_MANIFEST_HEADER_SIZE 592U
#define EVOKE_SEGMENT_DESCRIPTOR_SIZE 120U
#define EVOKE_SEGMENT_RETIRED_RANGE_SIZE 8U
#define EVOKE_SEGMENT_OBJECT_REF_SIZE 48U
#define EVOKE_SEGMENT_CHECKSUM_OFFSET 584U
#define EVOKE_QUERY_CONTRACT_MAGIC UINT32_C(0x43514932)
#define EVOKE_QUERY_CONTRACT_VERSION UINT16_C(3)
#define EVOKE_QUERY_CONTRACT_HEADER_SIZE 96U
#define EVOKE_QUERY_CONTRACT_CHECKSUM_OFFSET 88U
#define EVOKE_LEXICAL_CATALOG_MAGIC UINT32_C(0x434C4932)
#define EVOKE_LEXICAL_CATALOG_VERSION UINT16_C(1)
#define EVOKE_LEXICAL_CATALOG_HEADER_SIZE 64U
#define EVOKE_LEXICAL_CATALOG_CHECKSUM_OFFSET 56U
#define EVOKE_TERM_DIRECTORY_MAGIC UINT32_C(0x44544932)
#define EVOKE_TERM_DIRECTORY_VERSION UINT16_C(2)
#define EVOKE_TERM_DIRECTORY_HEADER_SIZE 64U
#define EVOKE_TERM_EXTENT_DESCRIPTOR_SIZE 24U
#define EVOKE_TERM_DIRECTORY_CHECKSUM_OFFSET 48U
#define EVOKE_TERM_FOLD_MAGIC UINT32_C(0x46464932)
#define EVOKE_TERM_FOLD_VERSION UINT16_C(4)
#define EVOKE_TERM_FOLD_CHECKSUM_OFFSET 96U
#define EVOKE_SEGMENT_PAYLOAD_MAGIC UINT32_C(0x50534932)
#define EVOKE_SEGMENT_PAYLOAD_VERSION UINT16_C(7)
#define EVOKE_DOCUMENT_VERSION_RECORD_SIZE 48U
#define EVOKE_DOCUMENT_RETIREMENT_RECORD_SIZE 32U
#define EVOKE_SEMANTIC_STATE_RECORD_SIZE 72U
#define EVOKE_SEGMENT_PAYLOAD_CHECKSUM_OFFSET 168U
#define EVOKE_SEGMENT_PAGE_MAGIC UINT32_C(0x47504932)
#define EVOKE_SEGMENT_PAGE_VERSION UINT16_C(2)
#define EVOKE_SEGMENT_PAGE_CHECKSUM_OFFSET 56U
#define EVOKE_ACTIVE_L0_PAGE_MAGIC UINT32_C(0x304C4932)
#define EVOKE_ACTIVE_L0_PAGE_VERSION UINT16_C(1)
#define EVOKE_ACTIVE_L0_PAGE_CHECKSUM_OFFSET 56U
#define EVOKE_L0_RECORD_MAGIC UINT32_C(0x524C4932)
#define EVOKE_L0_RECORD_VERSION UINT16_C(4)
#define EVOKE_L0_RECORD_CHECKSUM_OFFSET 104U
#define EVOKE_L0_FRAME_MAGIC UINT32_C(0x464C4932)
#define EVOKE_L0_FRAME_VERSION UINT16_C(1)
#define EVOKE_SEGMENT_READ_ROOT_MAGIC UINT32_C(0x52524932)
#define EVOKE_SEGMENT_READ_ROOT_VERSION UINT16_C(3)
#define EVOKE_SEGMENT_READ_ROOT_CHECKSUM_OFFSET 192U

#define EVOKE_SEGMENT_MANIFEST_KNOWN_FLAGS                              \
    (EVOKE_SEGMENT_MANIFEST_FLAG_SAE |                                 \
     EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY |                      \
     EVOKE_SEGMENT_MANIFEST_FLAG_NEUTRAL_FOLD |                        \
     EVOKE_SEGMENT_MANIFEST_FLAG_IMPACT_FOLD |                         \
     EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY |                  \
     EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY |                  \
     EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP |                       \
     EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP |                        \
     EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR)

#define EVOKE_SEGMENT_KNOWN_FLAGS                                       \
    (EVOKE_SEGMENT_FLAG_SEALED |                                       \
     EVOKE_SEGMENT_FLAG_LEXICAL |                                      \
     EVOKE_SEGMENT_FLAG_SEMANTIC |                                     \
     EVOKE_SEGMENT_FLAG_PENDING |                                      \
     EVOKE_SEGMENT_FLAG_RETIREMENTS |                                  \
     EVOKE_SEGMENT_FLAG_QUARANTINE |                                   \
     EVOKE_SEGMENT_FLAG_HISTORY_BARRIER)

#define EVOKE_DOCUMENT_VERSION_KNOWN_FLAGS                             \
    (EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING |                    \
     EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE |                   \
     EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_QUARANTINED |                \
     EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |                          \
     EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE)

#define EVOKE_DOCUMENT_RETIREMENT_KNOWN_FLAGS                          \
    EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID

#define EVOKE_SEMANTIC_STATE_KNOWN_FLAGS                              \
    (EVOKE_SEMANTIC_STATE_FLAG_COMPLETE |                             \
     EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED |                          \
     EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID)

#define EVOKE_SEGMENT_PAYLOAD_KNOWN_FLAGS                              \
    EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP

#define EVOKE_QUERY_CONTRACT_KNOWN_FLAGS                               \
    (EVOKE_QUERY_CONTRACT_FLAG_VOCABULARY |                            \
     EVOKE_QUERY_CONTRACT_FLAG_EMPTY_TOKEN)

_Static_assert(
    sizeof(evoke_posting_value) == sizeof(uint32_t),
    "segment posting values must remain exactly four bytes"
);

static void evoke_serialize_object_ref(
    uint8_t *bytes,
    const evoke_segment_object_ref *ref
);

static void evoke_serialize_descriptor(
    uint8_t *bytes,
    const evoke_segment_descriptor *segment
);

bool
evoke_segment_object_ref_equal(
    const evoke_segment_object_ref *left,
    const evoke_segment_object_ref *right
)
{
    return left != NULL &&
        right != NULL &&
        left->object_kind == right->object_kind &&
        left->start_block == right->start_block &&
        left->page_count == right->page_count &&
        left->object_id == right->object_id &&
        left->owner_manifest_id == right->owner_manifest_id &&
        left->object_bytes == right->object_bytes &&
        left->object_checksum == right->object_checksum;
}

bool
evoke_active_l0_frontier_equal(
    const evoke_active_l0_frontier *left,
    const evoke_active_l0_frontier *right
)
{
    return left != NULL &&
        right != NULL &&
        left->segment_id == right->segment_id &&
        left->min_sequence == right->min_sequence &&
        left->max_sequence == right->max_sequence &&
        left->payload_bytes == right->payload_bytes &&
        left->head_block == right->head_block &&
        left->tail_block == right->tail_block &&
        left->page_count == right->page_count &&
        left->record_count == right->record_count;
}

static void
evoke_write_u16_le(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t) (value & UINT16_C(0x00FF));
    dst[1] = (uint8_t) ((value >> 8) & UINT16_C(0x00FF));
}

static void
evoke_write_u32_le(uint8_t *dst, uint32_t value)
{
    dst[0] = (uint8_t) (value & UINT32_C(0x000000FF));
    dst[1] = (uint8_t) ((value >> 8) & UINT32_C(0x000000FF));
    dst[2] = (uint8_t) ((value >> 16) & UINT32_C(0x000000FF));
    dst[3] = (uint8_t) ((value >> 24) & UINT32_C(0x000000FF));
}

static void
evoke_write_u64_le(uint8_t *dst, uint64_t value)
{
    size_t i;

    for (i = 0; i < sizeof(value); i++)
    {
        dst[i] = (uint8_t) ((value >> (i * 8)) & UINT64_C(0xFF));
    }
}

static uint16_t
evoke_read_u16_le(const uint8_t *src)
{
    return (uint16_t) src[0] |
           (uint16_t) ((uint16_t) src[1] << 8);
}

static uint32_t
evoke_read_u32_le(const uint8_t *src)
{
    return (uint32_t) src[0] |
           ((uint32_t) src[1] << 8) |
           ((uint32_t) src[2] << 16) |
           ((uint32_t) src[3] << 24);
}

static uint64_t
evoke_read_u64_le(const uint8_t *src)
{
    uint64_t value = 0;
    size_t i;

    for (i = 0; i < sizeof(value); i++)
    {
        value |= (uint64_t) src[i] << (i * 8);
    }
    return value;
}

static bool
evoke_checked_add_size(size_t left, size_t right, size_t *result_out)
{
    if (result_out == NULL || left > SIZE_MAX - right)
    {
        return false;
    }
    *result_out = left + right;
    return true;
}

static bool
evoke_checked_mul_size(size_t left, size_t right, size_t *result_out)
{
    if (result_out == NULL)
    {
        return false;
    }
    if (left == 0 || right == 0)
    {
        *result_out = 0;
        return true;
    }
    if (left > SIZE_MAX / right)
    {
        return false;
    }
    *result_out = left * right;
    return true;
}

static uint64_t
evoke_checksum_update(uint64_t checksum, const uint8_t *bytes, size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        checksum ^= bytes[i];
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

uint64_t
evoke_segment_blob_checksum(const uint8_t *bytes, size_t size)
{
    if (bytes == NULL && size > 0)
    {
        return 0;
    }
    return evoke_checksum_update(
        UINT64_C(14695981039346656037),
        bytes,
        size
    );
}

uint32_t
evoke_segment_page_payload_checksum(const uint8_t *bytes, size_t size)
{
    uint64_t checksum = evoke_segment_blob_checksum(bytes, size);
    uint32_t folded = (uint32_t) checksum ^ (uint32_t) (checksum >> 32);

    return folded == 0 ? UINT32_MAX : folded;
}

static uint64_t
evoke_manifest_checksum(const uint8_t *bytes, size_t size)
{
    static const uint8_t zeros[sizeof(uint64_t)] = {0};
    uint64_t checksum = UINT64_C(14695981039346656037);

    checksum = evoke_checksum_update(
        checksum,
        bytes,
        EVOKE_SEGMENT_CHECKSUM_OFFSET
    );
    checksum = evoke_checksum_update(checksum, zeros, sizeof(zeros));
    checksum = evoke_checksum_update(
        checksum,
        bytes + EVOKE_SEGMENT_CHECKSUM_OFFSET + sizeof(uint64_t),
        size - EVOKE_SEGMENT_CHECKSUM_OFFSET - sizeof(uint64_t)
    );
    return checksum;
}

static uint64_t
evoke_checksum_with_zero_range(
    const uint8_t *bytes,
    size_t size,
    size_t zero_offset,
    size_t zero_size
)
{
    static const uint8_t zeros[sizeof(uint64_t)] = {0};
    uint64_t checksum = UINT64_C(14695981039346656037);

    if (zero_size != sizeof(zeros))
    {
        return 0;
    }
    checksum = evoke_checksum_update(checksum, bytes, zero_offset);
    checksum = evoke_checksum_update(checksum, zeros, sizeof(zeros));
    checksum = evoke_checksum_update(
        checksum,
        bytes + zero_offset + zero_size,
        size - zero_offset - zero_size
    );
    return checksum;
}

static bool
evoke_block_range_is_valid(uint32_t start_block, uint32_t block_count)
{
    if (block_count == 0)
    {
        return start_block == 0;
    }
    if (start_block == 0)
    {
        return false;
    }
    return start_block <= UINT32_MAX - block_count;
}

static bool
evoke_block_ranges_overlap(
    uint32_t left_start,
    uint32_t left_count,
    uint32_t right_start,
    uint32_t right_count
)
{
    uint64_t left_end;
    uint64_t right_end;

    if (left_count == 0 || right_count == 0)
    {
        return false;
    }
    left_end = (uint64_t) left_start + left_count;
    right_end = (uint64_t) right_start + right_count;
    return (uint64_t) left_start < right_end &&
           (uint64_t) right_start < left_end;
}

static bool
evoke_segment_object_kind_valid(evoke_segment_object_kind kind)
{
    return kind >= EVOKE_SEGMENT_OBJECT_MANIFEST &&
           kind <= EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD_BOUND;
}

static bool
evoke_segment_object_ref_is_absent(const evoke_segment_object_ref *ref)
{
    return ref != NULL &&
        ref->object_kind == EVOKE_SEGMENT_OBJECT_INVALID &&
        ref->start_block == 0 &&
        ref->page_count == 0 &&
        ref->object_id == 0 &&
        ref->owner_manifest_id == 0 &&
        ref->object_bytes == 0 &&
        ref->object_checksum == 0;
}

static bool
evoke_flag_matches_object_ref(
    uint32_t flags,
    uint32_t flag,
    const evoke_segment_object_ref *ref,
    evoke_segment_object_kind expected_kind,
    uint64_t manifest_id
)
{
    bool has_flag = (flags & flag) != 0;

    if (!has_flag)
    {
        return evoke_segment_object_ref_is_absent(ref);
    }
    return ref != NULL &&
        ref->object_kind == expected_kind &&
        ref->object_id == manifest_id &&
        ref->owner_manifest_id == manifest_id &&
        evoke_segment_object_ref_validate(ref, UINT32_MAX) == EVOKE_OK;
}

static bool
evoke_flag_matches_inherited_object_ref(
    uint32_t flags,
    uint32_t flag,
    const evoke_segment_object_ref *ref,
    evoke_segment_object_kind expected_kind,
    uint64_t manifest_id
)
{
    bool has_flag = (flags & flag) != 0;

    if (!has_flag)
    {
        return evoke_segment_object_ref_is_absent(ref);
    }
    return ref != NULL &&
        ref->object_kind == expected_kind &&
        ref->object_id == ref->owner_manifest_id &&
        ref->owner_manifest_id <= manifest_id &&
        evoke_segment_object_ref_validate(ref, UINT32_MAX) == EVOKE_OK;
}

static bool
evoke_segment_object_refs_overlap(
    const evoke_segment_object_ref *left,
    const evoke_segment_object_ref *right
)
{
    if (evoke_segment_object_ref_is_absent(left) ||
        evoke_segment_object_ref_is_absent(right))
    {
        return false;
    }
    return evoke_block_ranges_overlap(
        left->start_block,
        left->page_count,
        right->start_block,
        right->page_count
    );
}

evoke_status
evoke_segment_page_count_required(
    size_t object_bytes,
    size_t page_content_bytes,
    uint32_t *page_count_out
)
{
    size_t payload_capacity;
    size_t page_count;

    if (page_count_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *page_count_out = 0;
    if (object_bytes == 0 ||
        page_content_bytes <= EVOKE_SEGMENT_PAGE_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    payload_capacity =
        page_content_bytes - EVOKE_SEGMENT_PAGE_HEADER_SIZE;
    page_count = object_bytes / payload_capacity;
    if (object_bytes % payload_capacity != 0)
    {
        page_count++;
    }
    if (page_count == 0 || page_count > UINT32_MAX)
    {
        return EVOKE_ERR_FORMAT;
    }
    *page_count_out = (uint32_t) page_count;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_page_header_validate(
    const evoke_segment_page_header *header,
    size_t page_content_bytes
)
{
    uint32_t expected_page_count;
    size_t payload_capacity;
    uint64_t page_offset;
    uint64_t remaining;
    uint32_t expected_used_bytes;
    evoke_status status;

    if (header == NULL ||
        !evoke_segment_object_kind_valid(header->object_kind) ||
        header->object_id == 0 ||
        header->owner_manifest_id == 0 ||
        header->object_bytes == 0 ||
        header->object_checksum == 0 ||
        header->payload_checksum == 0 ||
        header->object_bytes > SIZE_MAX)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_segment_page_count_required(
        (size_t) header->object_bytes,
        page_content_bytes,
        &expected_page_count
    );
    if (status != EVOKE_OK ||
        header->page_count != expected_page_count ||
        header->ordinal >= header->page_count)
    {
        return EVOKE_ERR_FORMAT;
    }

    payload_capacity =
        page_content_bytes - EVOKE_SEGMENT_PAGE_HEADER_SIZE;
    page_offset = (uint64_t) header->ordinal * payload_capacity;
    if (page_offset >= header->object_bytes)
    {
        return EVOKE_ERR_FORMAT;
    }
    remaining = header->object_bytes - page_offset;
    expected_used_bytes = remaining > payload_capacity
        ? (uint32_t) payload_capacity
        : (uint32_t) remaining;
    if (header->used_bytes != expected_used_bytes)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

evoke_status
evoke_segment_page_header_serialize(
    const evoke_segment_page_header *header,
    size_t page_content_bytes,
    uint8_t *bytes_out,
    size_t size_out
)
{
    uint64_t checksum;

    if (bytes_out == NULL ||
        size_out < EVOKE_SEGMENT_PAGE_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_segment_page_header_validate(
            header,
            page_content_bytes) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }

    memset(bytes_out, 0, EVOKE_SEGMENT_PAGE_HEADER_SIZE);
    evoke_write_u32_le(bytes_out, EVOKE_SEGMENT_PAGE_MAGIC);
    evoke_write_u16_le(bytes_out + 4, EVOKE_SEGMENT_PAGE_VERSION);
    evoke_write_u16_le(bytes_out + 6, (uint16_t) header->object_kind);
    evoke_write_u64_le(bytes_out + 8, header->object_id);
    evoke_write_u64_le(bytes_out + 16, header->owner_manifest_id);
    evoke_write_u64_le(bytes_out + 24, header->object_bytes);
    evoke_write_u64_le(bytes_out + 32, header->object_checksum);
    evoke_write_u32_le(bytes_out + 40, header->ordinal);
    evoke_write_u32_le(bytes_out + 44, header->page_count);
    evoke_write_u32_le(bytes_out + 48, header->used_bytes);
    evoke_write_u32_le(bytes_out + 52, header->payload_checksum);
    checksum = evoke_checksum_with_zero_range(
        bytes_out,
        EVOKE_SEGMENT_PAGE_HEADER_SIZE,
        EVOKE_SEGMENT_PAGE_CHECKSUM_OFFSET,
        sizeof(uint64_t)
    );
    evoke_write_u64_le(
        bytes_out + EVOKE_SEGMENT_PAGE_CHECKSUM_OFFSET,
        checksum
    );
    return EVOKE_OK;
}

evoke_status
evoke_segment_page_header_deserialize(
    const uint8_t *bytes,
    size_t size,
    size_t page_content_bytes,
    evoke_segment_page_header *header_out
)
{
    evoke_segment_page_header header;
    uint64_t checksum;

    if (bytes == NULL || header_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (size < EVOKE_SEGMENT_PAGE_HEADER_SIZE ||
        evoke_read_u32_le(bytes) != EVOKE_SEGMENT_PAGE_MAGIC ||
        evoke_read_u16_le(bytes + 4) != EVOKE_SEGMENT_PAGE_VERSION)
    {
        return EVOKE_ERR_FORMAT;
    }
    checksum = evoke_read_u64_le(
        bytes + EVOKE_SEGMENT_PAGE_CHECKSUM_OFFSET
    );
    if (checksum == 0 ||
        checksum != evoke_checksum_with_zero_range(
            bytes,
            EVOKE_SEGMENT_PAGE_HEADER_SIZE,
            EVOKE_SEGMENT_PAGE_CHECKSUM_OFFSET,
            sizeof(uint64_t)))
    {
        return EVOKE_ERR_FORMAT;
    }

    memset(&header, 0, sizeof(header));
    header.object_kind =
        (evoke_segment_object_kind) evoke_read_u16_le(bytes + 6);
    header.object_id = evoke_read_u64_le(bytes + 8);
    header.owner_manifest_id = evoke_read_u64_le(bytes + 16);
    header.object_bytes = evoke_read_u64_le(bytes + 24);
    header.object_checksum = evoke_read_u64_le(bytes + 32);
    header.ordinal = evoke_read_u32_le(bytes + 40);
    header.page_count = evoke_read_u32_le(bytes + 44);
    header.used_bytes = evoke_read_u32_le(bytes + 48);
    header.payload_checksum = evoke_read_u32_le(bytes + 52);
    if (evoke_segment_page_header_validate(
            &header,
            page_content_bytes) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    *header_out = header;
    return EVOKE_OK;
}

evoke_status
evoke_segment_page_payload_validate(
    const evoke_segment_page_header *header,
    const uint8_t *payload,
    size_t payload_size
)
{
    if (header == NULL || payload == NULL ||
        payload_size != header->used_bytes)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_segment_page_payload_checksum(payload, payload_size) !=
        header->payload_checksum)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_active_l0_page_header_validate(
    const evoke_active_l0_page_header *header,
    size_t page_content_bytes
)
{
    size_t payload_capacity;

    if (header == NULL ||
        page_content_bytes <= EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE ||
        header->segment_id == 0 ||
        header->min_sequence == 0 ||
        header->min_sequence > header->max_sequence ||
        header->payload_checksum == 0 ||
        header->next_block == 0 ||
        header->frame_count == 0 ||
        header->used_bytes == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    payload_capacity =
        page_content_bytes - EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE;
    if (header->used_bytes > payload_capacity)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

evoke_status
evoke_active_l0_page_header_serialize(
    const evoke_active_l0_page_header *header,
    size_t page_content_bytes,
    uint8_t *bytes_out,
    size_t size_out
)
{
    uint64_t checksum;

    if (bytes_out == NULL ||
        size_out < EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_active_l0_page_header_validate(
            header,
            page_content_bytes) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }

    memset(bytes_out, 0, EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE);
    evoke_write_u32_le(bytes_out, EVOKE_ACTIVE_L0_PAGE_MAGIC);
    evoke_write_u16_le(bytes_out + 4, EVOKE_ACTIVE_L0_PAGE_VERSION);
    evoke_write_u64_le(bytes_out + 8, header->segment_id);
    evoke_write_u64_le(bytes_out + 16, header->min_sequence);
    evoke_write_u64_le(bytes_out + 24, header->max_sequence);
    evoke_write_u64_le(bytes_out + 32, header->payload_checksum);
    evoke_write_u32_le(bytes_out + 40, header->ordinal);
    evoke_write_u32_le(bytes_out + 44, header->next_block);
    evoke_write_u32_le(bytes_out + 48, header->frame_count);
    evoke_write_u32_le(bytes_out + 52, header->used_bytes);
    checksum = evoke_checksum_with_zero_range(
        bytes_out,
        EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE,
        EVOKE_ACTIVE_L0_PAGE_CHECKSUM_OFFSET,
        sizeof(uint64_t)
    );
    evoke_write_u64_le(
        bytes_out + EVOKE_ACTIVE_L0_PAGE_CHECKSUM_OFFSET,
        checksum
    );
    return EVOKE_OK;
}

evoke_status
evoke_active_l0_page_header_deserialize(
    const uint8_t *bytes,
    size_t size,
    size_t page_content_bytes,
    evoke_active_l0_page_header *header_out
)
{
    evoke_active_l0_page_header header;
    uint64_t checksum;

    if (bytes == NULL || header_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (size < EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE ||
        evoke_read_u32_le(bytes) != EVOKE_ACTIVE_L0_PAGE_MAGIC ||
        evoke_read_u16_le(bytes + 4) != EVOKE_ACTIVE_L0_PAGE_VERSION ||
        evoke_read_u16_le(bytes + 6) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    checksum = evoke_read_u64_le(
        bytes + EVOKE_ACTIVE_L0_PAGE_CHECKSUM_OFFSET
    );
    if (checksum == 0 ||
        checksum != evoke_checksum_with_zero_range(
            bytes,
            EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE,
            EVOKE_ACTIVE_L0_PAGE_CHECKSUM_OFFSET,
            sizeof(uint64_t)))
    {
        return EVOKE_ERR_FORMAT;
    }

    memset(&header, 0, sizeof(header));
    header.segment_id = evoke_read_u64_le(bytes + 8);
    header.min_sequence = evoke_read_u64_le(bytes + 16);
    header.max_sequence = evoke_read_u64_le(bytes + 24);
    header.payload_checksum = evoke_read_u64_le(bytes + 32);
    header.ordinal = evoke_read_u32_le(bytes + 40);
    header.next_block = evoke_read_u32_le(bytes + 44);
    header.frame_count = evoke_read_u32_le(bytes + 48);
    header.used_bytes = evoke_read_u32_le(bytes + 52);
    if (evoke_active_l0_page_header_validate(
            &header,
            page_content_bytes) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    *header_out = header;
    return EVOKE_OK;
}

evoke_status
evoke_active_l0_page_payload_validate(
    const evoke_active_l0_page_header *header,
    const uint8_t *payload,
    size_t payload_size
)
{
    if (header == NULL || payload == NULL ||
        payload_size != header->used_bytes)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_segment_blob_checksum(payload, payload_size) !=
        header->payload_checksum)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

evoke_status
evoke_active_l0_frontier_validate(
    const evoke_active_l0_frontier *frontier,
    bool allow_absent
)
{
    bool absent;
    bool empty;

    if (frontier == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    absent = frontier->segment_id == 0;
    if (absent)
    {
        return allow_absent &&
               frontier->min_sequence == 0 &&
               frontier->max_sequence == 0 &&
               frontier->payload_bytes == 0 &&
               frontier->head_block == 0 &&
               frontier->tail_block == 0 &&
               frontier->page_count == 0 &&
               frontier->record_count == 0
            ? EVOKE_OK
            : EVOKE_ERR_FORMAT;
    }

    empty = frontier->record_count == 0;
    if (empty)
    {
        return frontier->min_sequence == 0 &&
               frontier->max_sequence == 0 &&
               frontier->payload_bytes == 0 &&
               frontier->head_block == 0 &&
               frontier->tail_block == 0 &&
               frontier->page_count == 0
            ? EVOKE_OK
            : EVOKE_ERR_FORMAT;
    }
    if (frontier->min_sequence == 0 ||
        frontier->min_sequence > frontier->max_sequence ||
        frontier->record_count > EVOKE_ACTIVE_L0_MAX_RECORDS ||
        frontier->max_sequence - frontier->min_sequence !=
            (uint64_t) frontier->record_count - 1 ||
        frontier->payload_bytes == 0 ||
        frontier->head_block == 0 ||
        frontier->head_block == EVOKE_ACTIVE_L0_NO_NEXT_BLOCK ||
        frontier->tail_block == 0 ||
        frontier->tail_block == EVOKE_ACTIVE_L0_NO_NEXT_BLOCK ||
        frontier->page_count == 0 ||
        frontier->page_count > EVOKE_ACTIVE_L0_MAX_PAGES)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

static bool
evoke_l0_aligned_size(size_t size, size_t *aligned_out)
{
    size_t padded;

    if (!evoke_checked_add_size(size, 3, &padded) ||
        aligned_out == NULL)
    {
        return false;
    }
    *aligned_out = padded & ~(size_t) 3;
    return true;
}

static int
evoke_l0_term_compare(
    const uint8_t *left,
    size_t left_size,
    const uint8_t *right,
    size_t right_size
)
{
    size_t common_size = left_size < right_size ? left_size : right_size;
    int result;

    result = memcmp(left, right, common_size);
    if (result != 0)
    {
        return result;
    }
    if (left_size < right_size)
    {
        return -1;
    }
    if (left_size > right_size)
    {
        return 1;
    }
    return 0;
}

static evoke_status
evoke_l0_record_input_validate(
    const evoke_l0_record *record,
    size_t *payload_size_out
)
{
    size_t payload_size = 0;
    uint32_t atom_index;

    if (record == NULL || payload_size_out == NULL ||
        record->flags != 0 || record->sequence == 0 ||
        record->record_xid == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    if (record->kind == EVOKE_L0_RECORD_RETIRE)
    {
        if (record->term_encoding != EVOKE_L0_TERM_ENCODING_NONE ||
            record->heap_block != 0 || record->heap_offset != 0 ||
            record->atom_count != 0 || record->atoms != NULL ||
            record->semantic_atoms != NULL ||
            record->semantic_failure_count != 0 ||
            record->semantic_error_code != 0 ||
            record->semantic_retry_after != 0 ||
            record->semantic_pending_since != 0 ||
            record->semantic_error_hash != 0 ||
            record->semantic_input_fingerprint != NULL)
        {
            return EVOKE_ERR_FORMAT;
        }
        *payload_size_out = 0;
        return EVOKE_OK;
    }
    if (record->kind == EVOKE_L0_RECORD_SEMANTIC_QUARANTINE)
    {
        if (record->term_encoding != EVOKE_L0_TERM_ENCODING_NONE ||
            record->heap_block != 0 || record->heap_offset != 0 ||
            record->document_length != 0 ||
            record->atom_count != 0 || record->atoms != NULL ||
            record->semantic_atoms != NULL ||
            record->semantic_failure_count == 0 ||
            record->semantic_error_code == 0 ||
            record->semantic_pending_since <= 0 ||
            record->semantic_retry_after < record->semantic_pending_since ||
            evoke_document_fingerprint_is_zero(
                record->semantic_input_fingerprint))
        {
            return EVOKE_ERR_FORMAT;
        }
        *payload_size_out = 0;
        return EVOKE_OK;
    }
    if (record->kind == EVOKE_L0_RECORD_SEMANTIC_COMPLETE)
    {
        if (record->term_encoding != EVOKE_L0_TERM_ENCODING_NUMERIC ||
            record->heap_block != 0 || record->heap_offset != 0 ||
            record->document_length != 0 || record->atoms != NULL ||
            record->semantic_failure_count != 0 ||
            record->semantic_error_code != 0 ||
            record->semantic_retry_after != 0 ||
            record->semantic_pending_since != 0 ||
            record->semantic_error_hash != 0 ||
            evoke_document_fingerprint_is_zero(
                record->semantic_input_fingerprint) ||
            (record->atom_count > 0 &&
             record->semantic_atoms == NULL))
        {
            return EVOKE_ERR_FORMAT;
        }
        for (atom_index = 0;
             atom_index < record->atom_count;
             atom_index++)
        {
            const evoke_l0_semantic_atom *atom =
                &record->semantic_atoms[atom_index];

            if (!isfinite(atom->impact) || atom->impact <= 0.0f ||
                (atom_index > 0 &&
                 record->semantic_atoms[atom_index - 1].term_id >=
                    atom->term_id))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        if (record->atom_count >
            UINT32_MAX / (sizeof(uint32_t) * 2))
        {
            return EVOKE_ERR_RANGE;
        }
        *payload_size_out =
            (size_t) record->atom_count * sizeof(uint32_t) * 2;
        return EVOKE_OK;
    }
    if (record->kind != EVOKE_L0_RECORD_UPSERT ||
        (record->term_encoding != EVOKE_L0_TERM_ENCODING_NUMERIC &&
         record->term_encoding != EVOKE_L0_TERM_ENCODING_UTF8) ||
        record->heap_offset == 0 ||
        record->semantic_atoms != NULL ||
        record->semantic_failure_count != 0 ||
        record->semantic_error_code != 0 ||
        record->semantic_retry_after != 0 ||
        record->semantic_pending_since != 0 ||
        record->semantic_error_hash != 0 ||
        (record->atom_count > 0 && record->atoms == NULL))
    {
        return EVOKE_ERR_FORMAT;
    }

    for (atom_index = 0; atom_index < record->atom_count; atom_index++)
    {
        const evoke_l0_lexical_atom *atom = &record->atoms[atom_index];
        size_t atom_size;

        if (atom->term_frequency == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (record->term_encoding == EVOKE_L0_TERM_ENCODING_NUMERIC)
        {
            if (atom->term_bytes != NULL || atom->term_bytes_len != 0 ||
                (atom_index > 0 &&
                 record->atoms[atom_index - 1].term_id >= atom->term_id))
            {
                return EVOKE_ERR_FORMAT;
            }
            atom_size = sizeof(uint32_t) * 2;
        }
        else
        {
            size_t aligned_term_size;

            if (atom->term_id != 0 || atom->term_bytes == NULL ||
                atom->term_bytes_len == 0 ||
                memchr(
                    atom->term_bytes,
                    0,
                    atom->term_bytes_len
                ) != NULL ||
                (atom_index > 0 &&
                 evoke_l0_term_compare(
                     record->atoms[atom_index - 1].term_bytes,
                     record->atoms[atom_index - 1].term_bytes_len,
                     atom->term_bytes,
                     atom->term_bytes_len
                 ) >= 0) ||
                !evoke_l0_aligned_size(
                    atom->term_bytes_len,
                    &aligned_term_size
                ) ||
                !evoke_checked_add_size(
                    sizeof(uint32_t) * 2,
                    aligned_term_size,
                    &atom_size
                ))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        if (!evoke_checked_add_size(
                payload_size,
                atom_size,
                &payload_size))
        {
            return EVOKE_ERR_RANGE;
        }
    }
    if (payload_size > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    *payload_size_out = payload_size;
    return EVOKE_OK;
}

evoke_status
evoke_l0_record_serialized_size(
    const evoke_l0_record *record,
    size_t *size_out
)
{
    size_t payload_size;
    size_t total_size;
    evoke_status status;

    if (size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_l0_record_input_validate(record, &payload_size);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (!evoke_checked_add_size(
            EVOKE_L0_RECORD_HEADER_SIZE,
            payload_size,
            &total_size) ||
        total_size > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    *size_out = total_size;
    return EVOKE_OK;
}

evoke_status
evoke_l0_record_serialize(
    const evoke_l0_record *record,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes;
    size_t total_size;
    size_t payload_size;
    size_t offset = EVOKE_L0_RECORD_HEADER_SIZE;
    uint32_t atom_index;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_l0_record_serialized_size(record, &total_size);
    if (status != EVOKE_OK)
    {
        return status;
    }
    payload_size = total_size - EVOKE_L0_RECORD_HEADER_SIZE;
    bytes = calloc(1, total_size);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    evoke_write_u32_le(bytes + 0, EVOKE_L0_RECORD_MAGIC);
    evoke_write_u16_le(bytes + 4, EVOKE_L0_RECORD_VERSION);
    evoke_write_u16_le(bytes + 6, EVOKE_L0_RECORD_HEADER_SIZE);
    evoke_write_u16_le(bytes + 8, (uint16_t) record->kind);
    evoke_write_u16_le(bytes + 10, (uint16_t) record->term_encoding);
    evoke_write_u16_le(bytes + 12, record->flags);
    evoke_write_u64_le(bytes + 16, record->sequence);
    evoke_write_u64_le(bytes + 24, record->document_slot);
    evoke_write_u32_le(bytes + 32, record->record_xid);
    evoke_write_u32_le(bytes + 36, record->heap_block);
    evoke_write_u16_le(bytes + 40, record->heap_offset);
    evoke_write_u16_le(bytes + 42, record->semantic_failure_count);
    evoke_write_u32_le(bytes + 44, record->document_length);
    evoke_write_u32_le(bytes + 48, record->atom_count);
    evoke_write_u32_le(bytes + 52, (uint32_t) payload_size);
    evoke_write_u32_le(bytes + 56, record->semantic_error_code);
    evoke_write_u64_le(
        bytes + 64,
        (uint64_t) record->semantic_retry_after
    );
    if (record->semantic_input_fingerprint != NULL)
    {
        memcpy(
            bytes + 72,
            record->semantic_input_fingerprint,
            EVOKE_DOCUMENT_FINGERPRINT_BYTES
        );
    }
    evoke_write_u64_le(
        bytes + 88,
        (uint64_t) record->semantic_pending_since
    );
    evoke_write_u64_le(bytes + 96, record->semantic_error_hash);

    for (atom_index = 0; atom_index < record->atom_count; atom_index++)
    {
        if (record->kind == EVOKE_L0_RECORD_SEMANTIC_COMPLETE)
        {
            const evoke_l0_semantic_atom *atom =
                &record->semantic_atoms[atom_index];
            uint32_t impact_bits;

            memcpy(&impact_bits, &atom->impact, sizeof(impact_bits));
            evoke_write_u32_le(bytes + offset, atom->term_id);
            evoke_write_u32_le(
                bytes + offset + sizeof(uint32_t),
                impact_bits
            );
            offset += sizeof(uint32_t) * 2;
        }
        else
        {
            const evoke_l0_lexical_atom *atom =
                &record->atoms[atom_index];

            if (record->term_encoding == EVOKE_L0_TERM_ENCODING_NUMERIC)
            {
                evoke_write_u32_le(bytes + offset, atom->term_id);
                evoke_write_u32_le(
                    bytes + offset + sizeof(uint32_t),
                    atom->term_frequency
                );
                offset += sizeof(uint32_t) * 2;
            }
            else
            {
                size_t aligned_term_size;

                (void) evoke_l0_aligned_size(
                    atom->term_bytes_len,
                    &aligned_term_size
                );
                evoke_write_u32_le(bytes + offset, atom->term_bytes_len);
                evoke_write_u32_le(
                    bytes + offset + sizeof(uint32_t),
                    atom->term_frequency
                );
                memcpy(
                    bytes + offset + sizeof(uint32_t) * 2,
                    atom->term_bytes,
                    atom->term_bytes_len
                );
                offset += sizeof(uint32_t) * 2 + aligned_term_size;
            }
        }
    }
    if (offset != total_size)
    {
        free(bytes);
        return EVOKE_ERR_FORMAT;
    }
    evoke_write_u64_le(
        bytes + EVOKE_L0_RECORD_CHECKSUM_OFFSET,
        evoke_checksum_with_zero_range(
            bytes,
            total_size,
            EVOKE_L0_RECORD_CHECKSUM_OFFSET,
            sizeof(uint64_t)
        )
    );
    *bytes_out = bytes;
    *size_out = total_size;
    return EVOKE_OK;
}

static evoke_status
evoke_l0_record_view_validate_payload(
    const evoke_l0_record_view *view
)
{
    const uint8_t *previous_term = NULL;
    uint32_t previous_term_size = 0;
    uint32_t previous_term_id = 0;
    size_t offset = 0;
    uint32_t atom_index;

    if (view->kind == EVOKE_L0_RECORD_RETIRE)
    {
        if (view->term_encoding != EVOKE_L0_TERM_ENCODING_NONE ||
            view->heap_block != 0 || view->heap_offset != 0 ||
            view->atom_count != 0 || view->payload_size != 0 ||
            view->semantic_failure_count != 0 ||
            view->semantic_error_code != 0 ||
            view->semantic_retry_after != 0 ||
            view->semantic_pending_since != 0 ||
            view->semantic_error_hash != 0 ||
            !evoke_document_fingerprint_is_zero(
                view->semantic_input_fingerprint))
        {
            return EVOKE_ERR_FORMAT;
        }
        return EVOKE_OK;
    }
    if (view->kind == EVOKE_L0_RECORD_SEMANTIC_QUARANTINE)
    {
        if (view->term_encoding != EVOKE_L0_TERM_ENCODING_NONE ||
            view->heap_block != 0 || view->heap_offset != 0 ||
            view->document_length != 0 ||
            view->atom_count != 0 || view->payload_size != 0 ||
            view->semantic_failure_count == 0 ||
            view->semantic_error_code == 0 ||
            view->semantic_pending_since <= 0 ||
            view->semantic_retry_after < view->semantic_pending_since ||
            evoke_document_fingerprint_is_zero(
                view->semantic_input_fingerprint))
        {
            return EVOKE_ERR_FORMAT;
        }
        return EVOKE_OK;
    }
    if (view->kind == EVOKE_L0_RECORD_SEMANTIC_COMPLETE)
    {
        if (view->term_encoding != EVOKE_L0_TERM_ENCODING_NUMERIC ||
            view->heap_block != 0 || view->heap_offset != 0 ||
            view->document_length != 0 ||
            view->semantic_failure_count != 0 ||
            view->semantic_error_code != 0 ||
            view->semantic_retry_after != 0 ||
            view->semantic_pending_since != 0 ||
            view->semantic_error_hash != 0 ||
            evoke_document_fingerprint_is_zero(
                view->semantic_input_fingerprint) ||
            view->atom_count >
                view->payload_size / (sizeof(uint32_t) * 2) ||
            view->payload_size !=
                (size_t) view->atom_count * sizeof(uint32_t) * 2)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (atom_index = 0; atom_index < view->atom_count; atom_index++)
        {
            uint32_t term_id;
            uint32_t impact_bits;
            float impact;

            term_id = evoke_read_u32_le(view->payload + offset);
            impact_bits = evoke_read_u32_le(
                view->payload + offset + sizeof(uint32_t)
            );
            memcpy(&impact, &impact_bits, sizeof(impact));
            if (!isfinite(impact) || impact <= 0.0f ||
                (atom_index > 0 && previous_term_id >= term_id))
            {
                return EVOKE_ERR_FORMAT;
            }
            previous_term_id = term_id;
            offset += sizeof(uint32_t) * 2;
        }
        return offset == view->payload_size ? EVOKE_OK : EVOKE_ERR_FORMAT;
    }
    if (view->kind != EVOKE_L0_RECORD_UPSERT ||
        (view->term_encoding != EVOKE_L0_TERM_ENCODING_NUMERIC &&
         view->term_encoding != EVOKE_L0_TERM_ENCODING_UTF8) ||
        view->heap_offset == 0 ||
        view->semantic_failure_count != 0 ||
        view->semantic_error_code != 0 ||
        view->semantic_retry_after != 0 ||
        view->semantic_pending_since != 0 ||
        view->semantic_error_hash != 0)
    {
        return EVOKE_ERR_FORMAT;
    }

    for (atom_index = 0; atom_index < view->atom_count; atom_index++)
    {
        uint32_t term_size_or_id;
        uint32_t term_frequency;

        if (offset > view->payload_size ||
            view->payload_size - offset < sizeof(uint32_t) * 2)
        {
            return EVOKE_ERR_FORMAT;
        }
        term_size_or_id = evoke_read_u32_le(view->payload + offset);
        term_frequency = evoke_read_u32_le(
            view->payload + offset + sizeof(uint32_t)
        );
        if (term_frequency == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        offset += sizeof(uint32_t) * 2;

        if (view->term_encoding == EVOKE_L0_TERM_ENCODING_NUMERIC)
        {
            if (atom_index > 0 && previous_term_id >= term_size_or_id)
            {
                return EVOKE_ERR_FORMAT;
            }
            previous_term_id = term_size_or_id;
        }
        else
        {
            size_t aligned_term_size;

            if (term_size_or_id == 0 ||
                !evoke_l0_aligned_size(
                    term_size_or_id,
                    &aligned_term_size
                ) ||
                aligned_term_size > view->payload_size - offset ||
                memchr(
                    view->payload + offset,
                    0,
                    term_size_or_id
                ) != NULL ||
                (atom_index > 0 &&
                 evoke_l0_term_compare(
                     previous_term,
                     previous_term_size,
                     view->payload + offset,
                     term_size_or_id
                 ) >= 0))
            {
                return EVOKE_ERR_FORMAT;
            }
            previous_term = view->payload + offset;
            previous_term_size = term_size_or_id;
            offset += aligned_term_size;
        }
    }
    return offset == view->payload_size ? EVOKE_OK : EVOKE_ERR_FORMAT;
}

evoke_status
evoke_l0_record_view_parse(
    const uint8_t *bytes,
    size_t size,
    evoke_l0_record_view *view_out
)
{
    evoke_l0_record_view view;
    uint32_t payload_size;
    uint64_t checksum;

    if (bytes == NULL || view_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (size < EVOKE_L0_RECORD_HEADER_SIZE ||
        evoke_read_u32_le(bytes + 0) != EVOKE_L0_RECORD_MAGIC ||
        evoke_read_u16_le(bytes + 4) != EVOKE_L0_RECORD_VERSION ||
        evoke_read_u16_le(bytes + 6) != EVOKE_L0_RECORD_HEADER_SIZE ||
        evoke_read_u16_le(bytes + 14) != 0 ||
        evoke_read_u32_le(bytes + 60) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    payload_size = evoke_read_u32_le(bytes + 52);
    if ((size_t) payload_size != size - EVOKE_L0_RECORD_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    checksum = evoke_read_u64_le(bytes + EVOKE_L0_RECORD_CHECKSUM_OFFSET);
    if (checksum == 0 ||
        checksum != evoke_checksum_with_zero_range(
            bytes,
            size,
            EVOKE_L0_RECORD_CHECKSUM_OFFSET,
            sizeof(uint64_t)))
    {
        return EVOKE_ERR_FORMAT;
    }

    memset(&view, 0, sizeof(view));
    view.kind = (evoke_l0_record_kind) evoke_read_u16_le(bytes + 8);
    view.term_encoding =
        (evoke_l0_term_encoding) evoke_read_u16_le(bytes + 10);
    view.flags = evoke_read_u16_le(bytes + 12);
    view.sequence = evoke_read_u64_le(bytes + 16);
    view.document_slot = evoke_read_u64_le(bytes + 24);
    view.record_xid = evoke_read_u32_le(bytes + 32);
    view.heap_block = evoke_read_u32_le(bytes + 36);
    view.heap_offset = evoke_read_u16_le(bytes + 40);
    view.semantic_failure_count = evoke_read_u16_le(bytes + 42);
    view.document_length = evoke_read_u32_le(bytes + 44);
    view.atom_count = evoke_read_u32_le(bytes + 48);
    view.semantic_error_code = evoke_read_u32_le(bytes + 56);
    view.semantic_retry_after = (int64_t) evoke_read_u64_le(bytes + 64);
    memcpy(
        view.semantic_input_fingerprint,
        bytes + 72,
        sizeof(view.semantic_input_fingerprint)
    );
    view.semantic_pending_since = (int64_t) evoke_read_u64_le(bytes + 88);
    view.semantic_error_hash = evoke_read_u64_le(bytes + 96);
    view.payload = bytes + EVOKE_L0_RECORD_HEADER_SIZE;
    view.payload_size = payload_size;
    if (view.flags != 0 || view.sequence == 0 || view.record_xid == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (evoke_l0_record_view_validate_payload(&view) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    *view_out = view;
    return EVOKE_OK;
}

evoke_status
evoke_l0_record_view_atom(
    const evoke_l0_record_view *view,
    uint32_t atom_index,
    evoke_l0_lexical_atom *atom_out
)
{
    size_t offset = 0;
    uint32_t current;

    if (view == NULL || atom_out == NULL ||
        view->kind != EVOKE_L0_RECORD_UPSERT ||
        atom_index >= view->atom_count ||
        (view->term_encoding != EVOKE_L0_TERM_ENCODING_NUMERIC &&
         view->term_encoding != EVOKE_L0_TERM_ENCODING_UTF8))
    {
        return EVOKE_ERR_INVALID;
    }
    memset(atom_out, 0, sizeof(*atom_out));
    for (current = 0; current <= atom_index; current++)
    {
        uint32_t term_size_or_id;
        uint32_t term_frequency;

        if (offset > view->payload_size ||
            view->payload_size - offset < sizeof(uint32_t) * 2)
        {
            return EVOKE_ERR_FORMAT;
        }
        term_size_or_id = evoke_read_u32_le(view->payload + offset);
        term_frequency = evoke_read_u32_le(
            view->payload + offset + sizeof(uint32_t)
        );
        offset += sizeof(uint32_t) * 2;
        if (view->term_encoding == EVOKE_L0_TERM_ENCODING_NUMERIC)
        {
            if (current == atom_index)
            {
                atom_out->term_id = term_size_or_id;
                atom_out->term_frequency = term_frequency;
                return EVOKE_OK;
            }
        }
        else
        {
            size_t aligned_term_size;

            if (!evoke_l0_aligned_size(
                    term_size_or_id,
                    &aligned_term_size) ||
                aligned_term_size > view->payload_size - offset)
            {
                return EVOKE_ERR_FORMAT;
            }
            if (current == atom_index)
            {
                atom_out->term_frequency = term_frequency;
                atom_out->term_bytes = view->payload + offset;
                atom_out->term_bytes_len = term_size_or_id;
                return EVOKE_OK;
            }
            offset += aligned_term_size;
        }
    }
    return EVOKE_ERR_FORMAT;
}

evoke_status
evoke_l0_record_view_decode_atoms(
    const evoke_l0_record_view *view,
    evoke_l0_lexical_atom *atoms_out,
    size_t atom_capacity
)
{
    size_t offset = 0;
    uint32_t atom_index;

    if (view == NULL ||
        view->kind != EVOKE_L0_RECORD_UPSERT ||
        (view->atom_count > 0 && atoms_out == NULL) ||
        atom_capacity < view->atom_count ||
        (view->term_encoding != EVOKE_L0_TERM_ENCODING_NUMERIC &&
         view->term_encoding != EVOKE_L0_TERM_ENCODING_UTF8))
    {
        return EVOKE_ERR_INVALID;
    }
    for (atom_index = 0; atom_index < view->atom_count; atom_index++)
    {
        uint32_t term_size_or_id;
        uint32_t term_frequency;

        if (offset > view->payload_size ||
            view->payload_size - offset < sizeof(uint32_t) * 2)
        {
            return EVOKE_ERR_FORMAT;
        }
        term_size_or_id = evoke_read_u32_le(view->payload + offset);
        term_frequency = evoke_read_u32_le(
            view->payload + offset + sizeof(uint32_t)
        );
        offset += sizeof(uint32_t) * 2;
        memset(&atoms_out[atom_index], 0, sizeof(atoms_out[atom_index]));
        atoms_out[atom_index].term_frequency = term_frequency;

        if (view->term_encoding == EVOKE_L0_TERM_ENCODING_NUMERIC)
        {
            atoms_out[atom_index].term_id = term_size_or_id;
        }
        else
        {
            size_t aligned_term_size;

            if (!evoke_l0_aligned_size(
                    term_size_or_id,
                    &aligned_term_size) ||
                aligned_term_size > view->payload_size - offset)
            {
                return EVOKE_ERR_FORMAT;
            }
            atoms_out[atom_index].term_bytes = view->payload + offset;
            atoms_out[atom_index].term_bytes_len = term_size_or_id;
            offset += aligned_term_size;
        }
    }
    return offset == view->payload_size ? EVOKE_OK : EVOKE_ERR_FORMAT;
}

evoke_status
evoke_l0_record_view_semantic_atom(
    const evoke_l0_record_view *view,
    uint32_t atom_index,
    evoke_l0_semantic_atom *atom_out
)
{
    size_t offset;
    uint32_t impact_bits;

    if (view == NULL || atom_out == NULL ||
        view->kind != EVOKE_L0_RECORD_SEMANTIC_COMPLETE ||
        view->term_encoding != EVOKE_L0_TERM_ENCODING_NUMERIC ||
        atom_index >= view->atom_count)
    {
        return EVOKE_ERR_INVALID;
    }
    offset = (size_t) atom_index * sizeof(uint32_t) * 2;
    if (offset > view->payload_size ||
        view->payload_size - offset < sizeof(uint32_t) * 2)
    {
        return EVOKE_ERR_FORMAT;
    }
    atom_out->term_id = evoke_read_u32_le(view->payload + offset);
    impact_bits = evoke_read_u32_le(
        view->payload + offset + sizeof(uint32_t)
    );
    memcpy(&atom_out->impact, &impact_bits, sizeof(atom_out->impact));
    return EVOKE_OK;
}

evoke_status
evoke_l0_record_view_decode_semantic_atoms(
    const evoke_l0_record_view *view,
    evoke_l0_semantic_atom *atoms_out,
    size_t atom_capacity
)
{
    uint32_t atom_index;

    if (view == NULL ||
        view->kind != EVOKE_L0_RECORD_SEMANTIC_COMPLETE ||
        view->term_encoding != EVOKE_L0_TERM_ENCODING_NUMERIC ||
        (view->atom_count > 0 && atoms_out == NULL) ||
        atom_capacity < view->atom_count)
    {
        return EVOKE_ERR_INVALID;
    }
    for (atom_index = 0; atom_index < view->atom_count; atom_index++)
    {
        evoke_status status = evoke_l0_record_view_semantic_atom(
            view,
            atom_index,
            &atoms_out[atom_index]
        );

        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_l0_frame_header_validate(const evoke_l0_frame_header *header)
{
    uint64_t fragment_end;
    bool is_start;
    bool is_end;

    if (header == NULL ||
        (header->flags &
         ~(EVOKE_L0_FRAME_FLAG_START | EVOKE_L0_FRAME_FLAG_END)) != 0 ||
        header->sequence == 0 || header->record_checksum == 0 ||
        header->record_bytes < EVOKE_L0_RECORD_HEADER_SIZE ||
        header->fragment_bytes == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    fragment_end =
        (uint64_t) header->fragment_offset + header->fragment_bytes;
    if (fragment_end > header->record_bytes)
    {
        return EVOKE_ERR_FORMAT;
    }
    is_start = (header->flags & EVOKE_L0_FRAME_FLAG_START) != 0;
    is_end = (header->flags & EVOKE_L0_FRAME_FLAG_END) != 0;
    if (is_start != (header->fragment_offset == 0) ||
        is_end != (fragment_end == header->record_bytes))
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

evoke_status
evoke_l0_frame_header_serialize(
    const evoke_l0_frame_header *header,
    uint8_t *bytes_out,
    size_t size_out
)
{
    if (bytes_out == NULL || size_out < EVOKE_L0_FRAME_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_l0_frame_header_validate(header) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(bytes_out, 0, EVOKE_L0_FRAME_HEADER_SIZE);
    evoke_write_u32_le(bytes_out + 0, EVOKE_L0_FRAME_MAGIC);
    evoke_write_u16_le(bytes_out + 4, EVOKE_L0_FRAME_VERSION);
    evoke_write_u16_le(bytes_out + 6, header->flags);
    evoke_write_u64_le(bytes_out + 8, header->sequence);
    evoke_write_u64_le(bytes_out + 16, header->record_checksum);
    evoke_write_u32_le(bytes_out + 24, header->record_bytes);
    evoke_write_u32_le(bytes_out + 28, header->fragment_offset);
    evoke_write_u32_le(bytes_out + 32, header->fragment_bytes);
    return EVOKE_OK;
}

evoke_status
evoke_l0_frame_header_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_l0_frame_header *header_out
)
{
    evoke_l0_frame_header header;

    if (bytes == NULL || header_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (size < EVOKE_L0_FRAME_HEADER_SIZE ||
        evoke_read_u32_le(bytes + 0) != EVOKE_L0_FRAME_MAGIC ||
        evoke_read_u16_le(bytes + 4) != EVOKE_L0_FRAME_VERSION ||
        evoke_read_u32_le(bytes + 36) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&header, 0, sizeof(header));
    header.flags = evoke_read_u16_le(bytes + 6);
    header.sequence = evoke_read_u64_le(bytes + 8);
    header.record_checksum = evoke_read_u64_le(bytes + 16);
    header.record_bytes = evoke_read_u32_le(bytes + 24);
    header.fragment_offset = evoke_read_u32_le(bytes + 28);
    header.fragment_bytes = evoke_read_u32_le(bytes + 32);
    if (evoke_l0_frame_header_validate(&header) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    *header_out = header;
    return EVOKE_OK;
}

evoke_status
evoke_segment_object_ref_validate(
    const evoke_segment_object_ref *ref,
    uint32_t published_block_high_watermark
)
{
    uint64_t end_block;

    if (ref == NULL ||
        !evoke_segment_object_kind_valid(ref->object_kind) ||
        ref->start_block == 0 ||
        ref->start_block == UINT32_MAX ||
        ref->page_count == 0 ||
        ref->object_id == 0 ||
        ref->owner_manifest_id == 0 ||
        ref->object_bytes == 0 ||
        ref->object_bytes > SIZE_MAX ||
        ref->object_checksum == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    end_block = (uint64_t) ref->start_block + ref->page_count;
    if (published_block_high_watermark <= 1 ||
        end_block > published_block_high_watermark ||
        end_block > UINT32_MAX)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (ref->object_kind == EVOKE_SEGMENT_OBJECT_MANIFEST &&
        ref->object_id != ref->owner_manifest_id)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

evoke_status
evoke_segment_read_root_validate(const evoke_segment_read_root *root)
{
    uint64_t maximum_sequence = 0;
    bool pending_present;
    bool active_nonempty;
    evoke_status status;

    if (root == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (root->root_id == 0 ||
        root->next_sequence == 0 ||
        root->next_document_slot > UINT32_MAX ||
        root->reusable_document_slot_cursor >
            root->next_document_slot ||
        root->next_segment_id == 0 ||
        root->next_segment_id <= root->root_id ||
        root->manifest.object_kind != EVOKE_SEGMENT_OBJECT_MANIFEST ||
        root->manifest.object_id != root->root_id)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_segment_object_ref_validate(
        &root->manifest,
        root->published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_active_l0_frontier_validate(
        &root->active_l0,
        false
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_active_l0_frontier_validate(
        &root->pending_l0,
        true
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    pending_present = root->pending_l0.segment_id != 0;
    active_nonempty = root->active_l0.record_count != 0;
    if ((pending_present && root->pending_l0.record_count == 0) ||
        (pending_present &&
         root->pending_l0.segment_id == root->active_l0.segment_id) ||
        root->active_l0.segment_id >= root->next_segment_id ||
        (pending_present &&
         root->pending_l0.segment_id >= root->next_segment_id))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (pending_present)
    {
        maximum_sequence = root->pending_l0.max_sequence;
        if (root->pending_l0.head_block >=
                root->published_block_high_watermark ||
            root->pending_l0.tail_block >=
                root->published_block_high_watermark)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    if (active_nonempty)
    {
        if (root->active_l0.head_block >=
                root->published_block_high_watermark ||
            root->active_l0.tail_block >=
                root->published_block_high_watermark ||
            (pending_present &&
             root->pending_l0.max_sequence >=
                root->active_l0.min_sequence))
        {
            return EVOKE_ERR_FORMAT;
        }
        maximum_sequence = root->active_l0.max_sequence;
    }
    if (root->next_sequence <= maximum_sequence)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

evoke_status
evoke_segment_read_root_rotate_l0(evoke_segment_read_root *root)
{
    evoke_segment_read_root next;
    evoke_status status;

    if (root == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_read_root_validate(root);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (root->active_l0.record_count == 0 ||
        root->pending_l0.segment_id != 0 ||
        root->next_segment_id == UINT64_MAX)
    {
        return EVOKE_ERR_FORMAT;
    }

    next = *root;
    next.pending_l0 = next.active_l0;
    memset(&next.active_l0, 0, sizeof(next.active_l0));
    next.active_l0.segment_id = next.next_segment_id;
    next.next_segment_id++;
    status = evoke_segment_read_root_validate(&next);
    if (status != EVOKE_OK)
    {
        return status;
    }
    *root = next;
    return EVOKE_OK;
}

evoke_status
evoke_segment_read_root_seal_pending(
    evoke_segment_read_root *root,
    const evoke_segment_object_ref *manifest_ref,
    uint32_t published_block_high_watermark
)
{
    evoke_segment_read_root next;
    evoke_status status;

    if (root == NULL || manifest_ref == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_read_root_validate(root);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (root->pending_l0.segment_id == 0 ||
        root->next_segment_id == UINT64_MAX ||
        manifest_ref->object_kind != EVOKE_SEGMENT_OBJECT_MANIFEST ||
        manifest_ref->object_id != root->next_segment_id ||
        manifest_ref->owner_manifest_id != root->next_segment_id ||
        published_block_high_watermark <
            root->published_block_high_watermark)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_segment_object_ref_validate(
        manifest_ref,
        published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    next = *root;
    next.root_id = manifest_ref->object_id;
    next.manifest = *manifest_ref;
    memset(&next.pending_l0, 0, sizeof(next.pending_l0));
    if (next.active_l0.record_count == 0)
    {
        next.reusable_document_slot_cursor = 0;
    }
    next.next_segment_id++;
    next.published_block_high_watermark =
        published_block_high_watermark;
    status = evoke_segment_read_root_validate(&next);
    if (status != EVOKE_OK)
    {
        return status;
    }
    *root = next;
    return EVOKE_OK;
}

evoke_status
evoke_segment_read_root_replace_manifest(
    evoke_segment_read_root *root,
    const evoke_segment_object_ref *manifest_ref,
    uint32_t published_block_high_watermark
)
{
    evoke_segment_read_root next;
    evoke_status status;

    if (root == NULL || manifest_ref == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_read_root_validate(root);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (root->next_segment_id == UINT64_MAX ||
        manifest_ref->object_kind != EVOKE_SEGMENT_OBJECT_MANIFEST ||
        manifest_ref->object_id != root->next_segment_id ||
        manifest_ref->owner_manifest_id != root->next_segment_id ||
        published_block_high_watermark <
            root->published_block_high_watermark)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_segment_object_ref_validate(
        manifest_ref,
        published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    next = *root;
    next.root_id = manifest_ref->object_id;
    next.manifest = *manifest_ref;
    if (next.active_l0.record_count == 0 &&
        next.pending_l0.record_count == 0)
    {
        next.reusable_document_slot_cursor = 0;
    }
    next.next_segment_id++;
    next.published_block_high_watermark =
        published_block_high_watermark;
    status = evoke_segment_read_root_validate(&next);
    if (status != EVOKE_OK)
    {
        return status;
    }
    *root = next;
    return EVOKE_OK;
}

void
evoke_segment_manifest_init(evoke_segment_manifest *manifest)
{
    if (manifest != NULL)
    {
        memset(manifest, 0, sizeof(*manifest));
    }
}

void
evoke_segment_manifest_free(evoke_segment_manifest *manifest)
{
    if (manifest == NULL)
    {
        return;
    }
    free(manifest->segments);
    free(manifest->doc_frequencies);
    free(manifest->retired_ranges);
    memset(manifest, 0, sizeof(*manifest));
}

evoke_status
evoke_segment_manifest_build_identity(
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *old_manifest,
    evoke_segment_manifest *next_manifest
)
{
    if (build_root == NULL || old_manifest == NULL ||
        next_manifest == NULL ||
        build_root->root_id != old_manifest->manifest_id ||
        build_root->next_segment_id <= old_manifest->manifest_id ||
        (old_manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0 ||
        old_manifest->doc_frequencies != NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    next_manifest->flags = old_manifest->flags;
    next_manifest->manifest_id = build_root->next_segment_id;
    next_manifest->parent_manifest_id = old_manifest->manifest_id;
    next_manifest->max_sequence = old_manifest->max_sequence;
    next_manifest->statistics_epoch = old_manifest->statistics_epoch;
    next_manifest->visible_document_count =
        old_manifest->visible_document_count;
    next_manifest->document_slot_count =
        old_manifest->document_slot_count;
    next_manifest->total_document_length =
        old_manifest->total_document_length;
    next_manifest->reclaim_before_sequence =
        old_manifest->reclaim_before_sequence;
    next_manifest->neutral_fold_coverage =
        old_manifest->neutral_fold_coverage;
    next_manifest->impact_fold_coverage =
        old_manifest->impact_fold_coverage;
    next_manifest->impact_statistics_epoch =
        old_manifest->impact_statistics_epoch;
    next_manifest->vocab_size = old_manifest->vocab_size;
    next_manifest->query_contract = old_manifest->query_contract;
    next_manifest->term_directory = old_manifest->term_directory;
    next_manifest->neutral_fold = old_manifest->neutral_fold;
    next_manifest->impact_fold = old_manifest->impact_fold;
    next_manifest->document_directory =
        old_manifest->document_directory;
    next_manifest->lexicon_lookup = old_manifest->lexicon_lookup;
    next_manifest->prefix_lookup = old_manifest->prefix_lookup;
    next_manifest->semantic_accelerator_directory =
        old_manifest->semantic_accelerator_directory;
    if ((old_manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR) != 0)
    {
        next_manifest->semantic_accelerator_max_sequence =
            evoke_segment_manifest_semantic_accelerator_baseline_sequence(
                old_manifest
            );
    }
    next_manifest->lexicon_hash_seed =
        old_manifest->lexicon_hash_seed;
    memcpy(
        next_manifest->contract_hash,
        old_manifest->contract_hash,
        sizeof(next_manifest->contract_hash)
    );
    next_manifest->segment_count = old_manifest->segment_count;
    if (next_manifest->segment_count > 0)
    {
        next_manifest->segments = calloc(
            next_manifest->segment_count,
            sizeof(*next_manifest->segments)
        );
        if (next_manifest->segments == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        memcpy(
            next_manifest->segments,
            old_manifest->segments,
            (size_t) next_manifest->segment_count *
                sizeof(*next_manifest->segments)
        );
    }
    return EVOKE_OK;
}

static uint64_t
evoke_manifest_authority_u32(uint64_t checksum, uint32_t value)
{
    uint8_t bytes[sizeof(value)];

    evoke_write_u32_le(bytes, value);
    return evoke_checksum_update(checksum, bytes, sizeof(bytes));
}

static uint64_t
evoke_manifest_authority_u64(uint64_t checksum, uint64_t value)
{
    uint8_t bytes[sizeof(value)];

    evoke_write_u64_le(bytes, value);
    return evoke_checksum_update(checksum, bytes, sizeof(bytes));
}

static uint64_t
evoke_manifest_authority_object_ref(
    uint64_t checksum,
    const evoke_segment_object_ref *ref
)
{
    uint8_t bytes[EVOKE_SEGMENT_OBJECT_REF_SIZE] = {0};

    evoke_serialize_object_ref(bytes, ref);
    return evoke_checksum_update(checksum, bytes, sizeof(bytes));
}

evoke_status
evoke_segment_manifest_authority_checksum(
    const evoke_segment_manifest *manifest,
    uint64_t *checksum_out
)
{
    uint64_t checksum = UINT64_C(14695981039346656037);
    uint32_t index;
    evoke_status status;

    if (checksum_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *checksum_out = 0;
    status = evoke_segment_manifest_validate(manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }

    checksum = evoke_manifest_authority_u32(
        checksum,
        manifest->flags &
            ~EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR
    );
    checksum = evoke_manifest_authority_u64(checksum, manifest->max_sequence);
    checksum = evoke_manifest_authority_u64(
        checksum,
        manifest->statistics_epoch
    );
    checksum = evoke_manifest_authority_u64(
        checksum,
        manifest->visible_document_count
    );
    checksum = evoke_manifest_authority_u64(
        checksum,
        manifest->document_slot_count
    );
    checksum = evoke_manifest_authority_u64(
        checksum,
        manifest->total_document_length
    );
    checksum = evoke_manifest_authority_u64(
        checksum,
        manifest->reclaim_before_sequence
    );
    checksum = evoke_manifest_authority_u64(
        checksum,
        manifest->neutral_fold_coverage
    );
    checksum = evoke_manifest_authority_u64(
        checksum,
        manifest->impact_fold_coverage
    );
    checksum = evoke_manifest_authority_u64(
        checksum,
        manifest->impact_statistics_epoch
    );
    checksum = evoke_manifest_authority_u32(checksum, manifest->vocab_size);
    checksum = evoke_manifest_authority_object_ref(
        checksum,
        &manifest->query_contract
    );
    checksum = evoke_manifest_authority_object_ref(
        checksum,
        &manifest->term_directory
    );
    checksum = evoke_manifest_authority_object_ref(
        checksum,
        &manifest->neutral_fold
    );
    checksum = evoke_manifest_authority_object_ref(
        checksum,
        &manifest->impact_fold
    );
    checksum = evoke_manifest_authority_object_ref(
        checksum,
        &manifest->document_directory
    );
    checksum = evoke_manifest_authority_object_ref(
        checksum,
        &manifest->lexicon_lookup
    );
    checksum = evoke_manifest_authority_object_ref(
        checksum,
        &manifest->prefix_lookup
    );
    checksum = evoke_manifest_authority_u64(
        checksum,
        manifest->lexicon_hash_seed
    );
    checksum = evoke_checksum_update(
        checksum,
        manifest->contract_hash,
        sizeof(manifest->contract_hash)
    );
    checksum = evoke_manifest_authority_u32(checksum, manifest->segment_count);
    for (index = 0; index < manifest->segment_count; index++)
    {
        uint8_t bytes[EVOKE_SEGMENT_DESCRIPTOR_SIZE] = {0};

        evoke_serialize_descriptor(bytes, &manifest->segments[index]);
        checksum = evoke_checksum_update(checksum, bytes, sizeof(bytes));
    }
    if ((manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) == 0)
    {
        for (index = 0; index < manifest->vocab_size; index++)
        {
            checksum = evoke_manifest_authority_u32(
                checksum,
                manifest->doc_frequencies[index]
            );
        }
    }
    *checksum_out = checksum == 0 ? UINT64_MAX : checksum;
    return EVOKE_OK;
}

bool
evoke_segment_manifest_semantic_accelerator_published(
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest
)
{
    return root != NULL && manifest != NULL &&
        root->root_id == manifest->manifest_id &&
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR) != 0;
}

bool
evoke_segment_manifest_semantic_accelerator_eligible(
    const evoke_segment_read_root *root,
    const evoke_segment_manifest *manifest
)
{
    return evoke_segment_manifest_semantic_accelerator_published(
        root,
        manifest
    );
}

uint64_t
evoke_segment_manifest_semantic_accelerator_baseline_sequence(
    const evoke_segment_manifest *manifest
)
{
    if (manifest == NULL ||
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR) == 0)
    {
        return 0;
    }
    return manifest->semantic_accelerator_max_sequence != 0
        ? manifest->semantic_accelerator_max_sequence
        : manifest->max_sequence;
}

evoke_status
evoke_segment_manifest_validate(const evoke_segment_manifest *manifest)
{
    bool has_cow_term_directory;
    bool has_flat_term_directory;
    bool has_lexicon_lookup;
    bool has_prefix_lookup;
    bool has_semantic_accelerator;
    uint32_t i;

    if (manifest == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if ((manifest->flags & ~EVOKE_SEGMENT_MANIFEST_KNOWN_FLAGS) != 0 ||
        manifest->manifest_id == 0 ||
        (manifest->parent_manifest_id != 0 &&
         manifest->parent_manifest_id >= manifest->manifest_id) ||
        manifest->statistics_epoch == 0 ||
        manifest->segment_count > EVOKE_SEGMENT_MANIFEST_MAX_SEGMENTS ||
        (manifest->segment_count > 0 && manifest->segments == NULL) ||
        manifest->retired_range_count >
            EVOKE_SEGMENT_MANIFEST_MAX_RETIRED_RANGES ||
        (manifest->retired_range_count > 0 &&
         manifest->retired_ranges == NULL) ||
        manifest->visible_document_count >
            manifest->document_slot_count ||
        manifest->document_slot_count > UINT32_MAX ||
        manifest->reclaim_before_sequence > manifest->max_sequence ||
        manifest->neutral_fold_coverage > manifest->max_sequence ||
        manifest->impact_fold_coverage > manifest->max_sequence)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (i = 0; i < manifest->retired_range_count; i++)
    {
        const evoke_block_range *range =
            &manifest->retired_ranges[i];

        if (!evoke_block_range_is_valid(
                range->start_block,
                range->block_count) ||
            range->block_count == 0 ||
            (i > 0 &&
             (uint64_t) manifest->retired_ranges[i - 1].start_block +
                 manifest->retired_ranges[i - 1].block_count >=
                 range->start_block))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    has_flat_term_directory =
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY) != 0;
    has_cow_term_directory =
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) != 0;
    has_lexicon_lookup =
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_LEXICON_LOOKUP) != 0;
    has_prefix_lookup =
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_PREFIX_LOOKUP) != 0;
    has_semantic_accelerator =
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR) != 0;
    if ((has_flat_term_directory && has_cow_term_directory) ||
        (has_cow_term_directory &&
         manifest->doc_frequencies != NULL) ||
        (!has_cow_term_directory &&
         manifest->vocab_size > 0 &&
         manifest->doc_frequencies == NULL))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (manifest->query_contract.object_kind !=
            EVOKE_SEGMENT_OBJECT_QUERY_CONTRACT ||
        manifest->query_contract.object_id !=
            manifest->query_contract.owner_manifest_id ||
        manifest->query_contract.owner_manifest_id >
            manifest->manifest_id ||
        evoke_segment_object_ref_validate(
            &manifest->query_contract,
            UINT32_MAX) != EVOKE_OK ||
        !evoke_flag_matches_inherited_object_ref(
            manifest->flags,
            EVOKE_SEGMENT_MANIFEST_FLAG_NEUTRAL_FOLD,
            &manifest->neutral_fold,
            EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
            manifest->manifest_id) ||
        !evoke_flag_matches_inherited_object_ref(
            manifest->flags,
            EVOKE_SEGMENT_MANIFEST_FLAG_IMPACT_FOLD,
            &manifest->impact_fold,
            EVOKE_SEGMENT_OBJECT_IMPACT_FOLD,
            manifest->manifest_id) ||
        (manifest->document_slot_count == 0 &&
         ((manifest->flags &
           EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY) != 0 ||
          !evoke_segment_object_ref_is_absent(
              &manifest->document_directory))) ||
        (manifest->document_slot_count > 0 &&
         ((manifest->flags &
           EVOKE_SEGMENT_MANIFEST_FLAG_DOCUMENT_DIRECTORY) == 0 ||
          manifest->document_directory.object_kind !=
              EVOKE_SEGMENT_OBJECT_DOCUMENT_DIRECTORY ||
          manifest->document_directory.owner_manifest_id >
              manifest->manifest_id ||
          evoke_segment_object_ref_validate(
              &manifest->document_directory,
              UINT32_MAX) != EVOKE_OK)))
    {
        return EVOKE_ERR_FORMAT;
    }
    if ((has_lexicon_lookup &&
         (manifest->vocab_size == 0 ||
          manifest->lexicon_hash_seed == 0 ||
          manifest->lexicon_lookup.object_kind !=
              EVOKE_SEGMENT_OBJECT_LEXICON_LOOKUP ||
          manifest->lexicon_lookup.owner_manifest_id >
              manifest->manifest_id ||
          evoke_segment_object_ref_validate(
              &manifest->lexicon_lookup,
              UINT32_MAX) != EVOKE_OK)) ||
        (!has_lexicon_lookup &&
         (manifest->lexicon_hash_seed != 0 ||
          !evoke_segment_object_ref_is_absent(
              &manifest->lexicon_lookup))))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (has_prefix_lookup != has_lexicon_lookup ||
        (has_prefix_lookup &&
         (manifest->prefix_lookup.object_kind !=
              EVOKE_SEGMENT_OBJECT_PREFIX_LOOKUP ||
          manifest->prefix_lookup.owner_manifest_id >
              manifest->manifest_id ||
          evoke_segment_object_ref_validate(
              &manifest->prefix_lookup,
              UINT32_MAX) != EVOKE_OK)) ||
        (!has_prefix_lookup &&
         !evoke_segment_object_ref_is_absent(
             &manifest->prefix_lookup)))
    {
        return EVOKE_ERR_FORMAT;
    }
    if ((has_semantic_accelerator &&
         ((manifest->flags & EVOKE_SEGMENT_MANIFEST_FLAG_SAE) == 0 ||
          manifest->parent_manifest_id == 0 ||
          !evoke_flag_matches_inherited_object_ref(
              manifest->flags,
              EVOKE_SEGMENT_MANIFEST_FLAG_SEMANTIC_ACCELERATOR,
              &manifest->semantic_accelerator_directory,
              EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_DIRECTORY,
              manifest->manifest_id))) ||
        (!has_semantic_accelerator &&
         (!evoke_segment_object_ref_is_absent(
              &manifest->semantic_accelerator_directory) ||
          manifest->semantic_accelerator_max_sequence != 0)) ||
        (has_semantic_accelerator &&
         manifest->semantic_accelerator_max_sequence != 0 &&
         manifest->semantic_accelerator_max_sequence >
             manifest->max_sequence))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (has_flat_term_directory)
    {
        if (!evoke_flag_matches_object_ref(
                manifest->flags,
                EVOKE_SEGMENT_MANIFEST_FLAG_TERM_DIRECTORY,
                &manifest->term_directory,
                EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY,
                manifest->manifest_id))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    else if (has_cow_term_directory)
    {
        if (manifest->term_directory.object_kind !=
                EVOKE_SEGMENT_OBJECT_TERM_DIRECTORY ||
            manifest->term_directory.owner_manifest_id >
                manifest->manifest_id ||
            evoke_segment_object_ref_validate(
                &manifest->term_directory,
                UINT32_MAX) != EVOKE_OK)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    else if (!evoke_segment_object_ref_is_absent(
                 &manifest->term_directory))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (manifest->segment_count > 0 &&
        manifest->vocab_size > 0 &&
        !has_flat_term_directory &&
        !has_cow_term_directory)
    {
        return EVOKE_ERR_FORMAT;
    }
    if ((manifest->flags & EVOKE_SEGMENT_MANIFEST_FLAG_IMPACT_FOLD) != 0 &&
        ((manifest->flags &
          EVOKE_SEGMENT_MANIFEST_FLAG_NEUTRAL_FOLD) == 0 ||
         manifest->impact_statistics_epoch != manifest->statistics_epoch))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (evoke_segment_object_refs_overlap(
            &manifest->query_contract,
            &manifest->term_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->query_contract,
            &manifest->neutral_fold) ||
        evoke_segment_object_refs_overlap(
            &manifest->query_contract,
            &manifest->impact_fold) ||
        evoke_segment_object_refs_overlap(
            &manifest->query_contract,
            &manifest->document_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->term_directory,
            &manifest->neutral_fold) ||
        evoke_segment_object_refs_overlap(
            &manifest->term_directory,
            &manifest->impact_fold) ||
        evoke_segment_object_refs_overlap(
            &manifest->term_directory,
            &manifest->document_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->neutral_fold,
            &manifest->impact_fold) ||
        evoke_segment_object_refs_overlap(
            &manifest->neutral_fold,
            &manifest->document_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->impact_fold,
            &manifest->document_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->query_contract,
            &manifest->lexicon_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->term_directory,
            &manifest->lexicon_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->neutral_fold,
            &manifest->lexicon_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->impact_fold,
            &manifest->lexicon_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->document_directory,
            &manifest->lexicon_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->query_contract,
            &manifest->prefix_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->term_directory,
            &manifest->prefix_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->neutral_fold,
            &manifest->prefix_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->impact_fold,
            &manifest->prefix_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->document_directory,
            &manifest->prefix_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->lexicon_lookup,
            &manifest->prefix_lookup) ||
        evoke_segment_object_refs_overlap(
            &manifest->query_contract,
            &manifest->semantic_accelerator_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->term_directory,
            &manifest->semantic_accelerator_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->neutral_fold,
            &manifest->semantic_accelerator_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->impact_fold,
            &manifest->semantic_accelerator_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->document_directory,
            &manifest->semantic_accelerator_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->lexicon_lookup,
            &manifest->semantic_accelerator_directory) ||
        evoke_segment_object_refs_overlap(
            &manifest->prefix_lookup,
            &manifest->semantic_accelerator_directory))
    {
        return EVOKE_ERR_FORMAT;
    }

    for (i = 0; !has_cow_term_directory &&
         i < manifest->vocab_size; i++)
    {
        if ((uint64_t) manifest->doc_frequencies[i] >
            manifest->document_slot_count)
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    for (i = 0; i < manifest->segment_count; i++)
    {
        const evoke_segment_descriptor *segment = &manifest->segments[i];
        bool history_barrier =
            (segment->flags &
             EVOKE_SEGMENT_FLAG_HISTORY_BARRIER) != 0;
        uint32_t j;

        if (segment->segment_id == 0 ||
            segment->min_sequence == 0 ||
            segment->min_sequence > segment->max_sequence ||
            segment->max_sequence > manifest->max_sequence ||
            segment->payload_checksum == 0 ||
            segment->payload_bytes == 0 ||
            segment->payload_owner_manifest_id == 0 ||
            segment->payload_owner_manifest_id > manifest->manifest_id ||
            segment->semantic_state_count >
                manifest->document_slot_count ||
            (segment->semantic_state_count > 0 &&
             (segment->flags & EVOKE_SEGMENT_FLAG_SEMANTIC) == 0) ||
            (segment->flags & ~EVOKE_SEGMENT_KNOWN_FLAGS) != 0 ||
            ((!history_barrier &&
              (segment->flags &
               (EVOKE_SEGMENT_FLAG_LEXICAL |
                EVOKE_SEGMENT_FLAG_SEMANTIC |
                EVOKE_SEGMENT_FLAG_RETIREMENTS)) == 0) ||
             (history_barrier &&
              (segment->flags !=
                   (EVOKE_SEGMENT_FLAG_SEALED |
                    EVOKE_SEGMENT_FLAG_HISTORY_BARRIER) ||
               segment->posting_count != 0 ||
               segment->retirement_count != 0 ||
               segment->document_count != 0 ||
               segment->total_document_length != 0 ||
               segment->document_slot_count != 0 ||
               segment->semantic_state_count != 0))) ||
            (segment->flags & EVOKE_SEGMENT_FLAG_SEALED) == 0 ||
            !evoke_block_range_is_valid(
                segment->start_block,
                segment->block_count))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (i > 0 &&
            manifest->segments[i - 1].max_sequence >=
            segment->min_sequence)
        {
            return EVOKE_ERR_FORMAT;
        }

        if (evoke_block_ranges_overlap(
                segment->start_block,
                segment->block_count,
                manifest->query_contract.start_block,
                manifest->query_contract.page_count) ||
            evoke_block_ranges_overlap(
                segment->start_block,
                segment->block_count,
                manifest->term_directory.start_block,
                manifest->term_directory.page_count) ||
            evoke_block_ranges_overlap(
                segment->start_block,
                segment->block_count,
                manifest->neutral_fold.start_block,
                manifest->neutral_fold.page_count) ||
            evoke_block_ranges_overlap(
                segment->start_block,
                segment->block_count,
                manifest->impact_fold.start_block,
                manifest->impact_fold.page_count) ||
            evoke_block_ranges_overlap(
                segment->start_block,
                segment->block_count,
                manifest->document_directory.start_block,
                manifest->document_directory.page_count) ||
            evoke_block_ranges_overlap(
                segment->start_block,
                segment->block_count,
                manifest->lexicon_lookup.start_block,
                manifest->lexicon_lookup.page_count) ||
            evoke_block_ranges_overlap(
                segment->start_block,
                segment->block_count,
                manifest->prefix_lookup.start_block,
                manifest->prefix_lookup.page_count) ||
            evoke_block_ranges_overlap(
                segment->start_block,
                segment->block_count,
                manifest->semantic_accelerator_directory.start_block,
                manifest->semantic_accelerator_directory.page_count))
        {
            return EVOKE_ERR_FORMAT;
        }

        for (j = 0; j < i; j++)
        {
            const evoke_segment_descriptor *other = &manifest->segments[j];

            if (other->segment_id == segment->segment_id ||
                evoke_block_ranges_overlap(
                    other->start_block,
                    other->block_count,
                    segment->start_block,
                    segment->block_count))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    for (i = 0; i < manifest->retired_range_count; i++)
    {
        const evoke_block_range *range =
            &manifest->retired_ranges[i];
        const evoke_segment_object_ref *owned_refs[8] = {
            &manifest->query_contract,
            &manifest->term_directory,
            &manifest->neutral_fold,
            &manifest->impact_fold,
            &manifest->document_directory,
            &manifest->lexicon_lookup,
            &manifest->prefix_lookup,
            &manifest->semantic_accelerator_directory
        };

        for (uint32_t owned_index = 0;
             owned_index < 8;
             owned_index++)
        {
            if (evoke_block_ranges_overlap(
                    range->start_block,
                    range->block_count,
                    owned_refs[owned_index]->start_block,
                    owned_refs[owned_index]->page_count))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        for (uint32_t segment_index = 0;
             segment_index < manifest->segment_count;
             segment_index++)
        {
            if (evoke_block_ranges_overlap(
                    range->start_block,
                    range->block_count,
                    manifest->segments[segment_index].start_block,
                    manifest->segments[segment_index].block_count))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    if (manifest->segment_count > 0 &&
        manifest->segments[manifest->segment_count - 1].max_sequence !=
        manifest->max_sequence)
    {
        return EVOKE_ERR_FORMAT;
    }

    return EVOKE_OK;
}

evoke_status
evoke_segment_descriptor_payload_ref(
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *segment,
    evoke_segment_object_ref *ref_out
)
{
    evoke_segment_object_ref ref;
    evoke_status status;

    if (manifest == NULL || segment == NULL || ref_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&ref, 0, sizeof(ref));
    ref.object_kind = EVOKE_SEGMENT_OBJECT_PAYLOAD;
    ref.start_block = segment->start_block;
    ref.page_count = segment->block_count;
    ref.object_id = segment->segment_id;
    ref.owner_manifest_id = segment->payload_owner_manifest_id;
    ref.object_bytes = segment->payload_bytes;
    ref.object_checksum = segment->payload_checksum;
    status = evoke_segment_object_ref_validate(&ref, UINT32_MAX);
    if (status != EVOKE_OK)
    {
        return status;
    }
    *ref_out = ref;
    return EVOKE_OK;
}

evoke_status
evoke_segment_manifest_validate_published(
    const evoke_segment_manifest *manifest,
    const evoke_segment_object_ref *manifest_ref,
    uint32_t published_block_high_watermark
)
{
    const evoke_segment_object_ref *owned_refs[8];
    evoke_status status;
    uint32_t i;

    status = evoke_segment_manifest_validate(manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (manifest_ref == NULL ||
        manifest_ref->object_kind != EVOKE_SEGMENT_OBJECT_MANIFEST ||
        manifest_ref->object_id != manifest->manifest_id ||
        manifest_ref->owner_manifest_id != manifest->manifest_id)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_segment_object_ref_validate(
        manifest_ref,
        published_block_high_watermark
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    owned_refs[0] = &manifest->query_contract;
    owned_refs[1] = &manifest->term_directory;
    owned_refs[2] = &manifest->neutral_fold;
    owned_refs[3] = &manifest->impact_fold;
    owned_refs[4] = &manifest->document_directory;
    owned_refs[5] = &manifest->lexicon_lookup;
    owned_refs[6] = &manifest->prefix_lookup;
    owned_refs[7] = &manifest->semantic_accelerator_directory;
    for (i = 0; i < 8; i++)
    {
        if (evoke_segment_object_ref_is_absent(owned_refs[i]))
        {
            continue;
        }
        status = evoke_segment_object_ref_validate(
            owned_refs[i],
            published_block_high_watermark
        );
        if (status != EVOKE_OK ||
            evoke_block_ranges_overlap(
                manifest_ref->start_block,
                manifest_ref->page_count,
                owned_refs[i]->start_block,
                owned_refs[i]->page_count))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    for (i = 0; i < manifest->segment_count; i++)
    {
        evoke_segment_object_ref payload_ref;
        uint32_t owned_index;

        status = evoke_segment_descriptor_payload_ref(
            manifest,
            &manifest->segments[i],
            &payload_ref
        );
        if (status != EVOKE_OK ||
            evoke_segment_object_ref_validate(
                &payload_ref,
                published_block_high_watermark
            ) != EVOKE_OK ||
            evoke_block_ranges_overlap(
                manifest_ref->start_block,
                manifest_ref->page_count,
                payload_ref.start_block,
                payload_ref.page_count))
        {
            return EVOKE_ERR_FORMAT;
        }
        for (owned_index = 0; owned_index < 8; owned_index++)
        {
            if (!evoke_segment_object_ref_is_absent(
                    owned_refs[owned_index]) &&
                evoke_block_ranges_overlap(
                    payload_ref.start_block,
                    payload_ref.page_count,
                    owned_refs[owned_index]->start_block,
                    owned_refs[owned_index]->page_count))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    for (i = 0; i < manifest->retired_range_count; i++)
    {
        const evoke_block_range *range =
            &manifest->retired_ranges[i];

        if ((uint64_t) range->start_block + range->block_count >
                published_block_high_watermark ||
            evoke_block_ranges_overlap(
                manifest_ref->start_block,
                manifest_ref->page_count,
                range->start_block,
                range->block_count))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    return EVOKE_OK;
}

evoke_status
evoke_segment_manifest_serialized_size(
    const evoke_segment_manifest *manifest,
    size_t *size_out
)
{
    size_t descriptor_bytes;
    size_t retired_range_bytes;
    size_t frequency_bytes;
    size_t total_size;
    evoke_status status;

    if (size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *size_out = 0;
    status = evoke_segment_manifest_validate(manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (!evoke_checked_mul_size(
            manifest->segment_count,
            EVOKE_SEGMENT_DESCRIPTOR_SIZE,
            &descriptor_bytes) ||
        !evoke_checked_mul_size(
            manifest->retired_range_count,
            EVOKE_SEGMENT_RETIRED_RANGE_SIZE,
            &retired_range_bytes) ||
        !evoke_checked_mul_size(
            (manifest->flags &
             EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) != 0
                ? 0
                : manifest->vocab_size,
            sizeof(uint32_t),
            &frequency_bytes) ||
        !evoke_checked_add_size(
            EVOKE_SEGMENT_MANIFEST_HEADER_SIZE,
            descriptor_bytes,
            &total_size) ||
        !evoke_checked_add_size(
            total_size,
            retired_range_bytes,
            &total_size) ||
        !evoke_checked_add_size(
            total_size,
            frequency_bytes,
            &total_size))
    {
        return EVOKE_ERR_RANGE;
    }

    *size_out = total_size;
    return EVOKE_OK;
}

static void
evoke_serialize_object_ref(
    uint8_t *bytes,
    const evoke_segment_object_ref *ref
)
{
    evoke_write_u32_le(bytes + 0, (uint32_t) ref->object_kind);
    evoke_write_u32_le(bytes + 8, ref->start_block);
    evoke_write_u32_le(bytes + 12, ref->page_count);
    evoke_write_u64_le(bytes + 16, ref->object_id);
    evoke_write_u64_le(bytes + 24, ref->owner_manifest_id);
    evoke_write_u64_le(bytes + 32, ref->object_bytes);
    evoke_write_u64_le(bytes + 40, ref->object_checksum);
}

static void
evoke_serialize_descriptor(
    uint8_t *bytes,
    const evoke_segment_descriptor *segment
)
{
    evoke_write_u64_le(bytes + 0, segment->segment_id);
    evoke_write_u64_le(bytes + 8, segment->min_sequence);
    evoke_write_u64_le(bytes + 16, segment->max_sequence);
    evoke_write_u64_le(bytes + 24, segment->posting_count);
    evoke_write_u64_le(bytes + 32, segment->retirement_count);
    evoke_write_u64_le(bytes + 40, segment->document_count);
    evoke_write_u64_le(bytes + 48, segment->total_document_length);
    evoke_write_u64_le(bytes + 56, segment->first_document_slot);
    evoke_write_u64_le(bytes + 64, segment->document_slot_count);
    evoke_write_u64_le(bytes + 72, segment->payload_checksum);
    evoke_write_u32_le(bytes + 80, segment->start_block);
    evoke_write_u32_le(bytes + 84, segment->block_count);
    evoke_write_u32_le(bytes + 88, segment->size_class);
    evoke_write_u32_le(bytes + 92, segment->flags);
    evoke_write_u64_le(bytes + 96, segment->payload_bytes);
    evoke_write_u64_le(
        bytes + 104,
        segment->payload_owner_manifest_id
    );
    evoke_write_u64_le(bytes + 112, segment->semantic_state_count);
}

evoke_status
evoke_segment_manifest_serialize(
    const evoke_segment_manifest *manifest,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes;
    size_t total_size;
    size_t segment_offset = EVOKE_SEGMENT_MANIFEST_HEADER_SIZE;
    size_t retired_range_offset;
    size_t frequency_offset = 0;
    bool has_cow_term_directory;
    uint32_t i;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_segment_manifest_serialized_size(manifest, &total_size);
    if (status != EVOKE_OK)
    {
        return status;
    }

    bytes = calloc(total_size, 1);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    has_cow_term_directory =
        (manifest->flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) != 0;
    retired_range_offset =
        segment_offset +
        (size_t) manifest->segment_count *
            EVOKE_SEGMENT_DESCRIPTOR_SIZE;
    if (!has_cow_term_directory)
    {
        frequency_offset =
            retired_range_offset +
            (size_t) manifest->retired_range_count *
                EVOKE_SEGMENT_RETIRED_RANGE_SIZE;
    }

    evoke_write_u32_le(bytes + 0, EVOKE_SEGMENT_MANIFEST_MAGIC);
    evoke_write_u16_le(bytes + 4, EVOKE_SEGMENT_MANIFEST_VERSION);
    evoke_write_u16_le(bytes + 6, EVOKE_SEGMENT_MANIFEST_HEADER_SIZE);
    evoke_write_u16_le(bytes + 8, EVOKE_SEGMENT_DESCRIPTOR_SIZE);
    evoke_write_u32_le(bytes + 12, manifest->flags);
    evoke_write_u64_le(bytes + 16, manifest->manifest_id);
    evoke_write_u64_le(bytes + 24, manifest->parent_manifest_id);
    evoke_write_u64_le(bytes + 32, manifest->max_sequence);
    evoke_write_u64_le(bytes + 40, manifest->statistics_epoch);
    evoke_write_u64_le(bytes + 48, manifest->visible_document_count);
    evoke_write_u64_le(bytes + 56, manifest->total_document_length);
    evoke_write_u64_le(bytes + 64, manifest->reclaim_before_sequence);
    evoke_write_u64_le(bytes + 72, manifest->neutral_fold_coverage);
    evoke_write_u64_le(bytes + 80, manifest->impact_fold_coverage);
    evoke_write_u64_le(bytes + 88, manifest->impact_statistics_epoch);
    evoke_write_u32_le(bytes + 96, manifest->vocab_size);
    evoke_write_u32_le(bytes + 100, manifest->segment_count);
    evoke_serialize_object_ref(bytes + 104, &manifest->query_contract);
    evoke_serialize_object_ref(bytes + 152, &manifest->term_directory);
    evoke_serialize_object_ref(bytes + 200, &manifest->neutral_fold);
    evoke_serialize_object_ref(bytes + 248, &manifest->impact_fold);
    evoke_write_u64_le(bytes + 296, frequency_offset);
    evoke_write_u64_le(bytes + 304, segment_offset);
    evoke_write_u64_le(bytes + 312, total_size);
    evoke_write_u64_le(bytes + 320, manifest->document_slot_count);
    memcpy(
        bytes + 328,
        manifest->contract_hash,
        sizeof(manifest->contract_hash)
    );
    evoke_serialize_object_ref(
        bytes + 360,
        &manifest->document_directory
    );
    evoke_serialize_object_ref(bytes + 408, &manifest->lexicon_lookup);
    evoke_write_u64_le(bytes + 456, manifest->lexicon_hash_seed);
    evoke_write_u32_le(bytes + 464, manifest->retired_range_count);
    evoke_write_u64_le(bytes + 472, retired_range_offset);
    evoke_serialize_object_ref(bytes + 480, &manifest->prefix_lookup);
    evoke_serialize_object_ref(
        bytes + 528,
        &manifest->semantic_accelerator_directory
    );
    evoke_write_u64_le(
        bytes + 576,
        manifest->semantic_accelerator_max_sequence
    );

    for (i = 0; i < manifest->segment_count; i++)
    {
        evoke_serialize_descriptor(
            bytes + segment_offset +
                (size_t) i * EVOKE_SEGMENT_DESCRIPTOR_SIZE,
            &manifest->segments[i]
        );
    }
    for (i = 0; i < manifest->retired_range_count; i++)
    {
        evoke_write_u32_le(
            bytes + retired_range_offset +
                (size_t) i * EVOKE_SEGMENT_RETIRED_RANGE_SIZE,
            manifest->retired_ranges[i].start_block
        );
        evoke_write_u32_le(
            bytes + retired_range_offset +
                (size_t) i * EVOKE_SEGMENT_RETIRED_RANGE_SIZE + 4,
            manifest->retired_ranges[i].block_count
        );
    }
    for (i = 0; !has_cow_term_directory &&
         i < manifest->vocab_size; i++)
    {
        evoke_write_u32_le(
            bytes + frequency_offset + (size_t) i * sizeof(uint32_t),
            manifest->doc_frequencies[i]
        );
    }
    evoke_write_u64_le(
        bytes + EVOKE_SEGMENT_CHECKSUM_OFFSET,
        evoke_manifest_checksum(bytes, total_size)
    );

    *bytes_out = bytes;
    *size_out = total_size;
    return EVOKE_OK;
}

static void
evoke_deserialize_object_ref(
    const uint8_t *bytes,
    evoke_segment_object_ref *ref
)
{
    memset(ref, 0, sizeof(*ref));
    ref->object_kind =
        (evoke_segment_object_kind) evoke_read_u32_le(bytes + 0);
    ref->start_block = evoke_read_u32_le(bytes + 8);
    ref->page_count = evoke_read_u32_le(bytes + 12);
    ref->object_id = evoke_read_u64_le(bytes + 16);
    ref->owner_manifest_id = evoke_read_u64_le(bytes + 24);
    ref->object_bytes = evoke_read_u64_le(bytes + 32);
    ref->object_checksum = evoke_read_u64_le(bytes + 40);
}

static void
evoke_serialize_l0_frontier(
    uint8_t *bytes,
    const evoke_active_l0_frontier *frontier
)
{
    evoke_write_u64_le(bytes + 0, frontier->segment_id);
    evoke_write_u64_le(bytes + 8, frontier->min_sequence);
    evoke_write_u64_le(bytes + 16, frontier->max_sequence);
    evoke_write_u64_le(bytes + 24, frontier->payload_bytes);
    evoke_write_u32_le(bytes + 32, frontier->head_block);
    evoke_write_u32_le(bytes + 36, frontier->tail_block);
    evoke_write_u32_le(bytes + 40, frontier->page_count);
    evoke_write_u32_le(bytes + 44, frontier->record_count);
}

static void
evoke_deserialize_l0_frontier(
    const uint8_t *bytes,
    evoke_active_l0_frontier *frontier
)
{
    memset(frontier, 0, sizeof(*frontier));
    frontier->segment_id = evoke_read_u64_le(bytes + 0);
    frontier->min_sequence = evoke_read_u64_le(bytes + 8);
    frontier->max_sequence = evoke_read_u64_le(bytes + 16);
    frontier->payload_bytes = evoke_read_u64_le(bytes + 24);
    frontier->head_block = evoke_read_u32_le(bytes + 32);
    frontier->tail_block = evoke_read_u32_le(bytes + 36);
    frontier->page_count = evoke_read_u32_le(bytes + 40);
    frontier->record_count = evoke_read_u32_le(bytes + 44);
}

evoke_status
evoke_segment_read_root_serialize(
    const evoke_segment_read_root *root,
    uint8_t *bytes_out,
    size_t size_out
)
{
    evoke_status status = evoke_segment_read_root_validate(root);

    if (bytes_out == NULL ||
        size_out < EVOKE_SEGMENT_READ_ROOT_SERIALIZED_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (status != EVOKE_OK)
    {
        return status;
    }
    memset(bytes_out, 0, EVOKE_SEGMENT_READ_ROOT_SERIALIZED_SIZE);
    evoke_write_u32_le(bytes_out + 0, EVOKE_SEGMENT_READ_ROOT_MAGIC);
    evoke_write_u16_le(bytes_out + 4, EVOKE_SEGMENT_READ_ROOT_VERSION);
    evoke_write_u16_le(
        bytes_out + 6,
        EVOKE_SEGMENT_READ_ROOT_SERIALIZED_SIZE
    );
    evoke_write_u64_le(bytes_out + 8, root->root_id);
    evoke_write_u64_le(bytes_out + 16, root->next_sequence);
    evoke_write_u32_le(
        bytes_out + 24,
        root->published_block_high_watermark
    );
    evoke_write_u32_le(
        bytes_out + 28,
        root->reusable_document_slot_cursor
    );
    evoke_serialize_object_ref(bytes_out + 32, &root->manifest);
    evoke_serialize_l0_frontier(bytes_out + 80, &root->active_l0);
    evoke_serialize_l0_frontier(bytes_out + 128, &root->pending_l0);
    evoke_write_u64_le(bytes_out + 176, root->next_document_slot);
    evoke_write_u64_le(bytes_out + 184, root->next_segment_id);
    evoke_write_u64_le(
        bytes_out + EVOKE_SEGMENT_READ_ROOT_CHECKSUM_OFFSET,
        evoke_checksum_with_zero_range(
            bytes_out,
            EVOKE_SEGMENT_READ_ROOT_SERIALIZED_SIZE,
            EVOKE_SEGMENT_READ_ROOT_CHECKSUM_OFFSET,
            sizeof(uint64_t)
        )
    );
    return EVOKE_OK;
}

evoke_status
evoke_segment_read_root_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_segment_read_root *root_out
)
{
    evoke_segment_read_root root;
    uint64_t expected_checksum;
    evoke_status status;

    if (bytes == NULL || root_out == NULL ||
        size != EVOKE_SEGMENT_READ_ROOT_SERIALIZED_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_read_u32_le(bytes + 0) !=
            EVOKE_SEGMENT_READ_ROOT_MAGIC ||
        evoke_read_u16_le(bytes + 4) !=
            EVOKE_SEGMENT_READ_ROOT_VERSION ||
        evoke_read_u16_le(bytes + 6) !=
            EVOKE_SEGMENT_READ_ROOT_SERIALIZED_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    expected_checksum = evoke_checksum_with_zero_range(
        bytes,
        size,
        EVOKE_SEGMENT_READ_ROOT_CHECKSUM_OFFSET,
        sizeof(uint64_t)
    );
    if (evoke_read_u64_le(
            bytes + EVOKE_SEGMENT_READ_ROOT_CHECKSUM_OFFSET) !=
        expected_checksum)
    {
        return EVOKE_ERR_FORMAT;
    }

    memset(&root, 0, sizeof(root));
    root.root_id = evoke_read_u64_le(bytes + 8);
    root.next_sequence = evoke_read_u64_le(bytes + 16);
    root.published_block_high_watermark =
        evoke_read_u32_le(bytes + 24);
    root.reusable_document_slot_cursor =
        evoke_read_u32_le(bytes + 28);
    evoke_deserialize_object_ref(bytes + 32, &root.manifest);
    evoke_deserialize_l0_frontier(bytes + 80, &root.active_l0);
    evoke_deserialize_l0_frontier(bytes + 128, &root.pending_l0);
    root.next_document_slot = evoke_read_u64_le(bytes + 176);
    root.next_segment_id = evoke_read_u64_le(bytes + 184);
    status = evoke_segment_read_root_validate(&root);
    if (status != EVOKE_OK)
    {
        return status;
    }
    *root_out = root;
    return EVOKE_OK;
}

static void
evoke_deserialize_descriptor(
    const uint8_t *bytes,
    evoke_segment_descriptor *segment
)
{
    segment->segment_id = evoke_read_u64_le(bytes + 0);
    segment->min_sequence = evoke_read_u64_le(bytes + 8);
    segment->max_sequence = evoke_read_u64_le(bytes + 16);
    segment->posting_count = evoke_read_u64_le(bytes + 24);
    segment->retirement_count = evoke_read_u64_le(bytes + 32);
    segment->document_count = evoke_read_u64_le(bytes + 40);
    segment->total_document_length = evoke_read_u64_le(bytes + 48);
    segment->first_document_slot = evoke_read_u64_le(bytes + 56);
    segment->document_slot_count = evoke_read_u64_le(bytes + 64);
    segment->payload_checksum = evoke_read_u64_le(bytes + 72);
    segment->start_block = evoke_read_u32_le(bytes + 80);
    segment->block_count = evoke_read_u32_le(bytes + 84);
    segment->size_class = evoke_read_u32_le(bytes + 88);
    segment->flags = evoke_read_u32_le(bytes + 92);
    segment->payload_bytes = evoke_read_u64_le(bytes + 96);
    segment->payload_owner_manifest_id = evoke_read_u64_le(bytes + 104);
    segment->semantic_state_count = evoke_read_u64_le(bytes + 112);
}

evoke_status
evoke_segment_manifest_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_segment_manifest *manifest_out
)
{
    evoke_segment_manifest manifest;
    uint64_t expected_checksum;
    uint64_t total_size;
    uint64_t frequency_offset;
    uint64_t retired_range_offset;
    uint64_t segment_offset;
    size_t expected_frequency_offset;
    size_t expected_retired_range_offset;
    bool has_cow_term_directory;
    uint32_t i;
    evoke_status status;

    if (bytes == NULL || manifest_out == NULL ||
        size < EVOKE_SEGMENT_MANIFEST_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_read_u32_le(bytes + 0) != EVOKE_SEGMENT_MANIFEST_MAGIC ||
        (evoke_read_u16_le(bytes + 4) != EVOKE_SEGMENT_MANIFEST_VERSION &&
         evoke_read_u16_le(bytes + 4) !=
             EVOKE_SEGMENT_MANIFEST_LEGACY_VERSION) ||
        evoke_read_u16_le(bytes + 6) !=
            EVOKE_SEGMENT_MANIFEST_HEADER_SIZE ||
        evoke_read_u16_le(bytes + 8) != EVOKE_SEGMENT_DESCRIPTOR_SIZE ||
        evoke_read_u16_le(bytes + 10) != 0 ||
        evoke_read_u32_le(bytes + 108) != 0 ||
        evoke_read_u32_le(bytes + 156) != 0 ||
        evoke_read_u32_le(bytes + 204) != 0 ||
        evoke_read_u32_le(bytes + 252) != 0 ||
        evoke_read_u32_le(bytes + 364) != 0 ||
        evoke_read_u32_le(bytes + 412) != 0 ||
        evoke_read_u32_le(bytes + 468) != 0 ||
        evoke_read_u32_le(bytes + 484) != 0 ||
        (evoke_read_u16_le(bytes + 4) ==
             EVOKE_SEGMENT_MANIFEST_LEGACY_VERSION &&
         evoke_read_u64_le(bytes + 576) != 0))
    {
        return EVOKE_ERR_FORMAT;
    }

    total_size = evoke_read_u64_le(bytes + 312);
    segment_offset = evoke_read_u64_le(bytes + 304);
    frequency_offset = evoke_read_u64_le(bytes + 296);
    retired_range_offset = evoke_read_u64_le(bytes + 472);
    expected_checksum =
        evoke_read_u64_le(bytes + EVOKE_SEGMENT_CHECKSUM_OFFSET);
    if (total_size != size ||
        segment_offset != EVOKE_SEGMENT_MANIFEST_HEADER_SIZE ||
        expected_checksum != evoke_manifest_checksum(bytes, size))
    {
        return EVOKE_ERR_FORMAT;
    }

    evoke_segment_manifest_init(&manifest);
    manifest.flags = evoke_read_u32_le(bytes + 12);
    manifest.manifest_id = evoke_read_u64_le(bytes + 16);
    manifest.parent_manifest_id = evoke_read_u64_le(bytes + 24);
    manifest.max_sequence = evoke_read_u64_le(bytes + 32);
    manifest.statistics_epoch = evoke_read_u64_le(bytes + 40);
    manifest.visible_document_count = evoke_read_u64_le(bytes + 48);
    manifest.document_slot_count = evoke_read_u64_le(bytes + 320);
    manifest.total_document_length = evoke_read_u64_le(bytes + 56);
    manifest.reclaim_before_sequence = evoke_read_u64_le(bytes + 64);
    manifest.neutral_fold_coverage = evoke_read_u64_le(bytes + 72);
    manifest.impact_fold_coverage = evoke_read_u64_le(bytes + 80);
    manifest.impact_statistics_epoch = evoke_read_u64_le(bytes + 88);
    manifest.vocab_size = evoke_read_u32_le(bytes + 96);
    manifest.segment_count = evoke_read_u32_le(bytes + 100);
    has_cow_term_directory =
        (manifest.flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) != 0;
    evoke_deserialize_object_ref(
        bytes + 104,
        &manifest.query_contract
    );
    evoke_deserialize_object_ref(
        bytes + 152,
        &manifest.term_directory
    );
    evoke_deserialize_object_ref(
        bytes + 200,
        &manifest.neutral_fold
    );
    evoke_deserialize_object_ref(
        bytes + 248,
        &manifest.impact_fold
    );
    evoke_deserialize_object_ref(
        bytes + 360,
        &manifest.document_directory
    );
    evoke_deserialize_object_ref(
        bytes + 408,
        &manifest.lexicon_lookup
    );
    evoke_deserialize_object_ref(
        bytes + 480,
        &manifest.prefix_lookup
    );
    evoke_deserialize_object_ref(
        bytes + 528,
        &manifest.semantic_accelerator_directory
    );
    manifest.semantic_accelerator_max_sequence =
        evoke_read_u16_le(bytes + 4) == EVOKE_SEGMENT_MANIFEST_VERSION
            ? evoke_read_u64_le(bytes + 576)
            : 0;
    manifest.lexicon_hash_seed = evoke_read_u64_le(bytes + 456);
    manifest.retired_range_count = evoke_read_u32_le(bytes + 464);
    memcpy(
        manifest.contract_hash,
        bytes + 328,
        sizeof(manifest.contract_hash)
    );

    if (manifest.segment_count > EVOKE_SEGMENT_MANIFEST_MAX_SEGMENTS ||
        manifest.retired_range_count >
            EVOKE_SEGMENT_MANIFEST_MAX_RETIRED_RANGES ||
        (size_t) manifest.segment_count >
            (SIZE_MAX - EVOKE_SEGMENT_MANIFEST_HEADER_SIZE) /
            EVOKE_SEGMENT_DESCRIPTOR_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    expected_retired_range_offset =
        EVOKE_SEGMENT_MANIFEST_HEADER_SIZE +
        (size_t) manifest.segment_count * EVOKE_SEGMENT_DESCRIPTOR_SIZE;
    if ((size_t) manifest.retired_range_count >
        (SIZE_MAX - expected_retired_range_offset) /
            EVOKE_SEGMENT_RETIRED_RANGE_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    expected_frequency_offset =
        expected_retired_range_offset +
        (size_t) manifest.retired_range_count *
            EVOKE_SEGMENT_RETIRED_RANGE_SIZE;
    if (retired_range_offset != expected_retired_range_offset ||
        (has_cow_term_directory &&
         (frequency_offset != 0 ||
          expected_frequency_offset != size)) ||
        (!has_cow_term_directory &&
         (frequency_offset != expected_frequency_offset ||
          manifest.vocab_size >
              (size - expected_frequency_offset) / sizeof(uint32_t) ||
          expected_frequency_offset +
              (size_t) manifest.vocab_size * sizeof(uint32_t) != size)))
    {
        return EVOKE_ERR_FORMAT;
    }

    if (manifest.retired_range_count > 0)
    {
        manifest.retired_ranges = calloc(
            manifest.retired_range_count,
            sizeof(*manifest.retired_ranges)
        );
        if (manifest.retired_ranges == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
    }
    if (manifest.segment_count > 0)
    {
        manifest.segments = calloc(
            manifest.segment_count,
            sizeof(*manifest.segments)
        );
        if (manifest.segments == NULL)
        {
            evoke_segment_manifest_free(&manifest);
            return EVOKE_ERR_NOMEM;
        }
    }
    if (!has_cow_term_directory && manifest.vocab_size > 0)
    {
        manifest.doc_frequencies = calloc(
            manifest.vocab_size,
            sizeof(*manifest.doc_frequencies)
        );
        if (manifest.doc_frequencies == NULL)
        {
            evoke_segment_manifest_free(&manifest);
            return EVOKE_ERR_NOMEM;
        }
    }

    for (i = 0; i < manifest.segment_count; i++)
    {
        evoke_deserialize_descriptor(
            bytes + EVOKE_SEGMENT_MANIFEST_HEADER_SIZE +
                (size_t) i * EVOKE_SEGMENT_DESCRIPTOR_SIZE,
            &manifest.segments[i]
        );
    }
    for (i = 0; i < manifest.retired_range_count; i++)
    {
        manifest.retired_ranges[i].start_block = evoke_read_u32_le(
            bytes + expected_retired_range_offset +
                (size_t) i * EVOKE_SEGMENT_RETIRED_RANGE_SIZE
        );
        manifest.retired_ranges[i].block_count = evoke_read_u32_le(
            bytes + expected_retired_range_offset +
                (size_t) i * EVOKE_SEGMENT_RETIRED_RANGE_SIZE + 4
        );
    }
    for (i = 0; !has_cow_term_directory &&
         i < manifest.vocab_size; i++)
    {
        manifest.doc_frequencies[i] = evoke_read_u32_le(
            bytes + expected_frequency_offset +
                (size_t) i * sizeof(uint32_t)
        );
    }

    status = evoke_segment_manifest_validate(&manifest);
    if (status != EVOKE_OK)
    {
        evoke_segment_manifest_free(&manifest);
        return status;
    }

    evoke_segment_manifest_free(manifest_out);
    *manifest_out = manifest;
    return EVOKE_OK;
}

void
evoke_segment_query_contract_init(evoke_segment_query_contract *contract)
{
    if (contract != NULL)
    {
        memset(contract, 0, sizeof(*contract));
    }
}

void
evoke_segment_query_contract_free(evoke_segment_query_contract *contract)
{
    uint32_t i;

    if (contract == NULL)
    {
        return;
    }
    if (contract->vocab != NULL)
    {
        for (i = 0; i < contract->vocab_size; i++)
        {
            free(contract->vocab[i]);
        }
    }
    free(contract->vocab);
    memset(contract, 0, sizeof(*contract));
}

evoke_status
evoke_segment_query_contract_validate(
    const evoke_segment_query_contract *contract,
    const evoke_segment_manifest *manifest
)
{
    bool has_vocabulary;
    bool has_empty_token;
    uint32_t i;

    if (contract == NULL || manifest == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    has_vocabulary =
        (contract->flags & EVOKE_QUERY_CONTRACT_FLAG_VOCABULARY) != 0;
    has_empty_token =
        (contract->flags & EVOKE_QUERY_CONTRACT_FLAG_EMPTY_TOKEN) != 0;
    if ((contract->flags & ~EVOKE_QUERY_CONTRACT_KNOWN_FLAGS) != 0 ||
        contract->vocab_size != manifest->vocab_size ||
        !evoke_params_are_valid(&contract->params) ||
        contract->block_shift == 0 ||
        contract->block_shift >= 32 ||
        (has_vocabulary && contract->vocab_size == 0) ||
        (!has_vocabulary && contract->vocab != NULL) ||
        (has_empty_token &&
         contract->empty_token_id >= contract->vocab_size) ||
        (!has_empty_token && contract->empty_token_id != 0))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (contract->vocab != NULL)
    {
        for (i = 0; i < contract->vocab_size; i++)
        {
            if (contract->vocab[i] == NULL)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    return EVOKE_OK;
}

evoke_status
evoke_segment_query_contract_build(
    const evoke_index *index,
    const evoke_segment_manifest *manifest,
    evoke_segment_query_contract *contract_out
)
{
    evoke_segment_query_contract contract;
    uint32_t i;
    evoke_status status;

    if (index == NULL || manifest == NULL || contract_out == NULL ||
        index->vocab_size != manifest->vocab_size ||
        !evoke_params_are_valid(&index->params) ||
        (index->has_empty_token &&
         index->empty_token_id >= index->vocab_size))
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_segment_query_contract_init(&contract);
    contract.params = index->params;
    contract.vocab_size = index->vocab_size;
    contract.block_shift = EVOKE_DEFAULT_POSTING_BLOCK_SHIFT;
    if (index->has_empty_token)
    {
        contract.flags |= EVOKE_QUERY_CONTRACT_FLAG_EMPTY_TOKEN;
        contract.empty_token_id = index->empty_token_id;
    }
    if (index->vocab != NULL)
    {
        contract.flags |= EVOKE_QUERY_CONTRACT_FLAG_VOCABULARY;
        contract.vocab = calloc(
            contract.vocab_size,
            sizeof(*contract.vocab)
        );
        if (contract.vocab == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        for (i = 0; i < contract.vocab_size; i++)
        {
            size_t length;

            if (index->vocab[i] == NULL)
            {
                evoke_segment_query_contract_free(&contract);
                return EVOKE_ERR_FORMAT;
            }
            length = strlen(index->vocab[i]);
            contract.vocab[i] = malloc(length + 1);
            if (contract.vocab[i] == NULL)
            {
                evoke_segment_query_contract_free(&contract);
                return EVOKE_ERR_NOMEM;
            }
            memcpy(
                contract.vocab[i],
                index->vocab[i],
                length + 1
            );
        }
    }
    status = evoke_segment_query_contract_validate(
        &contract,
        manifest
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_query_contract_free(&contract);
        return status;
    }
    evoke_segment_query_contract_free(contract_out);
    *contract_out = contract;
    return EVOKE_OK;
}

static uint32_t
evoke_float_bits(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float
evoke_float_from_bits(uint32_t bits)
{
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

evoke_status
evoke_segment_query_contract_serialize(
    const evoke_segment_query_contract *contract,
    const evoke_segment_manifest *manifest,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_segment_query_contract_validate(contract, manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    bytes = calloc(EVOKE_QUERY_CONTRACT_HEADER_SIZE, 1);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    evoke_write_u32_le(bytes + 0, EVOKE_QUERY_CONTRACT_MAGIC);
    evoke_write_u16_le(bytes + 4, EVOKE_QUERY_CONTRACT_VERSION);
    evoke_write_u16_le(bytes + 6, EVOKE_QUERY_CONTRACT_HEADER_SIZE);
    evoke_write_u32_le(bytes + 8, contract->flags);
    /*
     * Vocabulary cardinality and bytes belong to the descendant manifest's
     * immutable lexical catalogs. Keeping this field zero makes the scorer
     * contract exactly reusable as the vocabulary grows.
     */
    evoke_write_u32_le(bytes + 12, 0);
    evoke_write_u32_le(bytes + 16, (uint32_t) contract->params.method);
    evoke_write_u32_le(
        bytes + 20,
        (uint32_t) contract->params.idf_method
    );
    evoke_write_u32_le(bytes + 24, evoke_float_bits(contract->params.k1));
    evoke_write_u32_le(bytes + 28, evoke_float_bits(contract->params.b));
    evoke_write_u32_le(
        bytes + 32,
        evoke_float_bits(contract->params.delta)
    );
    evoke_write_u32_le(bytes + 36, contract->empty_token_id);
    evoke_write_u32_le(bytes + 40, contract->block_shift);
    evoke_write_u32_le(bytes + 44, 0);
    evoke_write_u64_le(bytes + 48, 0);
    evoke_write_u64_le(bytes + 56, 0);
    evoke_write_u64_le(bytes + 64, EVOKE_QUERY_CONTRACT_HEADER_SIZE);
    evoke_write_u64_le(bytes + 72, manifest->manifest_id);
    evoke_write_u64_le(
        bytes + EVOKE_QUERY_CONTRACT_CHECKSUM_OFFSET,
        evoke_checksum_with_zero_range(
            bytes,
            EVOKE_QUERY_CONTRACT_HEADER_SIZE,
            EVOKE_QUERY_CONTRACT_CHECKSUM_OFFSET,
            sizeof(uint64_t)
        )
    );
    *bytes_out = bytes;
    *size_out = EVOKE_QUERY_CONTRACT_HEADER_SIZE;
    return EVOKE_OK;
}

evoke_status
evoke_segment_query_contract_deserialize(
    const uint8_t *bytes,
    size_t size,
    const evoke_segment_manifest *manifest,
    evoke_segment_query_contract *contract_out
)
{
    evoke_segment_query_contract contract;
    uint64_t expected_owner_manifest_id;
    evoke_status status;

    if (bytes == NULL || manifest == NULL || contract_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (size != EVOKE_QUERY_CONTRACT_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    expected_owner_manifest_id =
        manifest->query_contract.owner_manifest_id != 0
            ? manifest->query_contract.owner_manifest_id
            : manifest->manifest_id;
    if (evoke_read_u32_le(bytes + 0) != EVOKE_QUERY_CONTRACT_MAGIC ||
        evoke_read_u16_le(bytes + 4) != EVOKE_QUERY_CONTRACT_VERSION ||
        evoke_read_u16_le(bytes + 6) !=
            EVOKE_QUERY_CONTRACT_HEADER_SIZE ||
        evoke_read_u32_le(bytes + 12) != 0 ||
        evoke_read_u32_le(bytes + 44) != 0 ||
        evoke_read_u64_le(bytes + 48) != 0 ||
        evoke_read_u64_le(bytes + 56) != 0 ||
        evoke_read_u64_le(bytes + 64) !=
            EVOKE_QUERY_CONTRACT_HEADER_SIZE ||
        evoke_read_u64_le(bytes + 72) != expected_owner_manifest_id ||
        evoke_read_u64_le(bytes + 80) != 0 ||
        evoke_read_u64_le(bytes + EVOKE_QUERY_CONTRACT_CHECKSUM_OFFSET) !=
            evoke_checksum_with_zero_range(
                bytes,
                size,
                EVOKE_QUERY_CONTRACT_CHECKSUM_OFFSET,
                sizeof(uint64_t)))
    {
        return EVOKE_ERR_FORMAT;
    }

    evoke_segment_query_contract_init(&contract);
    contract.flags = evoke_read_u32_le(bytes + 8);
    contract.vocab_size = manifest->vocab_size;
    contract.params.method =
        (evoke_method) evoke_read_u32_le(bytes + 16);
    contract.params.idf_method =
        (evoke_method) evoke_read_u32_le(bytes + 20);
    contract.params.k1 = evoke_float_from_bits(
        evoke_read_u32_le(bytes + 24)
    );
    contract.params.b = evoke_float_from_bits(
        evoke_read_u32_le(bytes + 28)
    );
    contract.params.delta = evoke_float_from_bits(
        evoke_read_u32_le(bytes + 32)
    );
    contract.empty_token_id = evoke_read_u32_le(bytes + 36);
    contract.block_shift = evoke_read_u32_le(bytes + 40);
    status = evoke_segment_query_contract_validate(
        &contract,
        manifest
    );
    if (status != EVOKE_OK)
    {
        evoke_segment_query_contract_free(&contract);
        return status;
    }
    evoke_segment_query_contract_free(contract_out);
    *contract_out = contract;
    return EVOKE_OK;
}

void
evoke_lexical_catalog_init(evoke_lexical_catalog *catalog)
{
    if (catalog != NULL)
    {
        memset(catalog, 0, sizeof(*catalog));
    }
}

void
evoke_lexical_catalog_free(evoke_lexical_catalog *catalog)
{
    uint32_t term_index;

    if (catalog == NULL)
    {
        return;
    }
    if (catalog->terms != NULL)
    {
        for (term_index = 0;
             term_index < catalog->term_count;
             term_index++)
        {
            free(catalog->terms[term_index]);
        }
    }
    free(catalog->terms);
    memset(catalog, 0, sizeof(*catalog));
}

evoke_status
evoke_lexical_catalog_validate(const evoke_lexical_catalog *catalog)
{
    uint64_t end_term_id;
    uint32_t term_index;

    if (catalog == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    end_term_id =
        (uint64_t) catalog->first_term_id + catalog->term_count;
    if (catalog->owner_manifest_id == 0 ||
        catalog->term_count == 0 ||
        end_term_id > UINT32_MAX ||
        catalog->terms == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (term_index = 0;
         term_index < catalog->term_count;
         term_index++)
    {
        if (catalog->terms[term_index] == NULL)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    return EVOKE_OK;
}

evoke_status
evoke_lexical_catalog_build(
    uint64_t owner_manifest_id,
    uint32_t first_term_id,
    uint32_t term_count,
    const char *const *terms,
    evoke_lexical_catalog *catalog_out
)
{
    evoke_lexical_catalog catalog;
    uint32_t term_index;
    evoke_status status;

    if (catalog_out == NULL ||
        (term_count > 0 && terms == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_lexical_catalog_init(&catalog);
    catalog.owner_manifest_id = owner_manifest_id;
    catalog.first_term_id = first_term_id;
    catalog.term_count = term_count;
    if (term_count > 0)
    {
        catalog.terms = calloc(term_count, sizeof(*catalog.terms));
        if (catalog.terms == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
    }
    for (term_index = 0; term_index < term_count; term_index++)
    {
        size_t length;

        if (terms[term_index] == NULL)
        {
            evoke_lexical_catalog_free(&catalog);
            return EVOKE_ERR_FORMAT;
        }
        length = strlen(terms[term_index]);
        catalog.terms[term_index] = malloc(length + 1);
        if (catalog.terms[term_index] == NULL)
        {
            evoke_lexical_catalog_free(&catalog);
            return EVOKE_ERR_NOMEM;
        }
        memcpy(
            catalog.terms[term_index],
            terms[term_index],
            length + 1
        );
    }
    status = evoke_lexical_catalog_validate(&catalog);
    if (status != EVOKE_OK)
    {
        evoke_lexical_catalog_free(&catalog);
        return status;
    }
    evoke_lexical_catalog_free(catalog_out);
    *catalog_out = catalog;
    return EVOKE_OK;
}

evoke_status
evoke_lexical_catalog_serialize(
    const evoke_lexical_catalog *catalog,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes;
    size_t offsets_bytes;
    size_t vocabulary_offset;
    size_t vocabulary_bytes = 0;
    size_t total_size;
    size_t cursor = 0;
    uint32_t term_index;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_lexical_catalog_validate(catalog);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (!evoke_checked_mul_size(
            (size_t) catalog->term_count + 1,
            sizeof(uint64_t),
            &offsets_bytes) ||
        !evoke_checked_add_size(
            EVOKE_LEXICAL_CATALOG_HEADER_SIZE,
            offsets_bytes,
            &vocabulary_offset))
    {
        return EVOKE_ERR_RANGE;
    }
    for (term_index = 0;
         term_index < catalog->term_count;
         term_index++)
    {
        if (!evoke_checked_add_size(
                vocabulary_bytes,
                strlen(catalog->terms[term_index]),
                &vocabulary_bytes))
        {
            return EVOKE_ERR_RANGE;
        }
    }
    if (!evoke_checked_add_size(
            vocabulary_offset,
            vocabulary_bytes,
            &total_size))
    {
        return EVOKE_ERR_RANGE;
    }
    bytes = calloc(total_size, 1);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    evoke_write_u32_le(bytes + 0, EVOKE_LEXICAL_CATALOG_MAGIC);
    evoke_write_u16_le(bytes + 4, EVOKE_LEXICAL_CATALOG_VERSION);
    evoke_write_u16_le(bytes + 6, EVOKE_LEXICAL_CATALOG_HEADER_SIZE);
    evoke_write_u32_le(bytes + 8, catalog->first_term_id);
    evoke_write_u32_le(bytes + 12, catalog->term_count);
    evoke_write_u64_le(bytes + 16, catalog->owner_manifest_id);
    evoke_write_u64_le(
        bytes + 24,
        EVOKE_LEXICAL_CATALOG_HEADER_SIZE
    );
    evoke_write_u64_le(bytes + 32, vocabulary_offset);
    evoke_write_u64_le(bytes + 40, vocabulary_bytes);
    evoke_write_u64_le(bytes + 48, total_size);
    for (term_index = 0;
         term_index < catalog->term_count;
         term_index++)
    {
        size_t length = strlen(catalog->terms[term_index]);

        evoke_write_u64_le(
            bytes + EVOKE_LEXICAL_CATALOG_HEADER_SIZE +
                (size_t) term_index * sizeof(uint64_t),
            cursor
        );
        memcpy(
            bytes + vocabulary_offset + cursor,
            catalog->terms[term_index],
            length
        );
        cursor += length;
    }
    evoke_write_u64_le(
        bytes + EVOKE_LEXICAL_CATALOG_HEADER_SIZE +
            (size_t) catalog->term_count * sizeof(uint64_t),
        cursor
    );
    evoke_write_u64_le(
        bytes + EVOKE_LEXICAL_CATALOG_CHECKSUM_OFFSET,
        evoke_checksum_with_zero_range(
            bytes,
            total_size,
            EVOKE_LEXICAL_CATALOG_CHECKSUM_OFFSET,
            sizeof(uint64_t)
        )
    );
    *bytes_out = bytes;
    *size_out = total_size;
    return EVOKE_OK;
}

evoke_status
evoke_lexical_catalog_deserialize(
    const uint8_t *bytes,
    size_t size,
    uint64_t expected_owner_manifest_id,
    evoke_lexical_catalog *catalog_out
)
{
    evoke_lexical_catalog catalog;
    uint64_t offsets_offset;
    uint64_t vocabulary_offset;
    uint64_t vocabulary_bytes;
    uint64_t total_size;
    uint64_t expected_vocabulary_offset;
    uint64_t previous_offset = 0;
    uint32_t term_index;
    evoke_status status;

    if (bytes == NULL || catalog_out == NULL ||
        expected_owner_manifest_id == 0 ||
        size < EVOKE_LEXICAL_CATALOG_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_read_u32_le(bytes + 0) !=
            EVOKE_LEXICAL_CATALOG_MAGIC ||
        evoke_read_u16_le(bytes + 4) !=
            EVOKE_LEXICAL_CATALOG_VERSION ||
        evoke_read_u16_le(bytes + 6) !=
            EVOKE_LEXICAL_CATALOG_HEADER_SIZE ||
        evoke_read_u64_le(bytes + 16) !=
            expected_owner_manifest_id ||
        evoke_read_u64_le(
            bytes + EVOKE_LEXICAL_CATALOG_CHECKSUM_OFFSET) !=
            evoke_checksum_with_zero_range(
                bytes,
                size,
                EVOKE_LEXICAL_CATALOG_CHECKSUM_OFFSET,
                sizeof(uint64_t)))
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_lexical_catalog_init(&catalog);
    catalog.first_term_id = evoke_read_u32_le(bytes + 8);
    catalog.term_count = evoke_read_u32_le(bytes + 12);
    catalog.owner_manifest_id = evoke_read_u64_le(bytes + 16);
    offsets_offset = evoke_read_u64_le(bytes + 24);
    vocabulary_offset = evoke_read_u64_le(bytes + 32);
    vocabulary_bytes = evoke_read_u64_le(bytes + 40);
    total_size = evoke_read_u64_le(bytes + 48);
    if (catalog.term_count == 0 ||
        (uint64_t) catalog.first_term_id + catalog.term_count >
            UINT32_MAX ||
        (uint64_t) catalog.term_count + 1 >
            (UINT64_MAX - EVOKE_LEXICAL_CATALOG_HEADER_SIZE) /
                sizeof(uint64_t))
    {
        return EVOKE_ERR_FORMAT;
    }
    expected_vocabulary_offset =
        EVOKE_LEXICAL_CATALOG_HEADER_SIZE +
        ((uint64_t) catalog.term_count + 1) * sizeof(uint64_t);
    if (offsets_offset != EVOKE_LEXICAL_CATALOG_HEADER_SIZE ||
        vocabulary_offset != expected_vocabulary_offset ||
        vocabulary_offset > size ||
        vocabulary_bytes != size - (size_t) vocabulary_offset ||
        total_size != size)
    {
        return EVOKE_ERR_FORMAT;
    }
    catalog.terms = calloc(
        catalog.term_count,
        sizeof(*catalog.terms)
    );
    if (catalog.terms == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (term_index = 0;
         term_index <= catalog.term_count;
         term_index++)
    {
        uint64_t offset = evoke_read_u64_le(
            bytes + (size_t) offsets_offset +
                (size_t) term_index * sizeof(uint64_t)
        );

        if ((term_index == 0 && offset != 0) ||
            offset < previous_offset ||
            offset > vocabulary_bytes)
        {
            evoke_lexical_catalog_free(&catalog);
            return EVOKE_ERR_FORMAT;
        }
        if (term_index > 0)
        {
            size_t length = (size_t) (offset - previous_offset);
            const uint8_t *source =
                bytes + (size_t) vocabulary_offset +
                (size_t) previous_offset;

            if (memchr(source, '\0', length) != NULL)
            {
                evoke_lexical_catalog_free(&catalog);
                return EVOKE_ERR_FORMAT;
            }
            catalog.terms[term_index - 1] = malloc(length + 1);
            if (catalog.terms[term_index - 1] == NULL)
            {
                evoke_lexical_catalog_free(&catalog);
                return EVOKE_ERR_NOMEM;
            }
            memcpy(
                catalog.terms[term_index - 1],
                source,
                length
            );
            catalog.terms[term_index - 1][length] = '\0';
        }
        previous_offset = offset;
    }
    if (previous_offset != vocabulary_bytes)
    {
        evoke_lexical_catalog_free(&catalog);
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_lexical_catalog_validate(&catalog);
    if (status != EVOKE_OK)
    {
        evoke_lexical_catalog_free(&catalog);
        return status;
    }
    evoke_lexical_catalog_free(catalog_out);
    *catalog_out = catalog;
    return EVOKE_OK;
}

void
evoke_term_directory_init(evoke_term_directory *directory)
{
    if (directory != NULL)
    {
        memset(directory, 0, sizeof(*directory));
    }
}

void
evoke_term_directory_free(evoke_term_directory *directory)
{
    if (directory == NULL)
    {
        return;
    }
    free(directory->term_offsets);
    free(directory->extents);
    memset(directory, 0, sizeof(*directory));
}

evoke_status
evoke_term_directory_validate_logical(
    const evoke_term_directory *directory,
    const evoke_segment_manifest *manifest
)
{
    uint32_t term_id;
    uint32_t extent_index;

    if (directory == NULL || manifest == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (manifest->segment_count > EVOKE_SEGMENT_MANIFEST_MAX_SEGMENTS ||
        (manifest->segment_count > 0 && manifest->segments == NULL) ||
        directory->vocab_size != manifest->vocab_size ||
        directory->term_offsets == NULL ||
        (directory->extent_count > 0 && directory->extents == NULL) ||
        directory->term_offsets[0] != 0 ||
        directory->term_offsets[directory->vocab_size] !=
            directory->extent_count)
    {
        return EVOKE_ERR_FORMAT;
    }

    for (term_id = 0; term_id < directory->vocab_size; term_id++)
    {
        uint64_t start = directory->term_offsets[term_id];
        uint64_t end = directory->term_offsets[term_id + 1];
        uint32_t previous_segment_index = 0;
        uint64_t previous_posting_offset = 0;
        bool has_previous = false;

        if (end < start || end > directory->extent_count ||
            end - start > EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (extent_index = (uint32_t) start;
             extent_index < (uint32_t) end;
             extent_index++)
        {
            const evoke_term_extent_descriptor *extent =
                &directory->extents[extent_index];
            const evoke_segment_descriptor *segment;

            if (extent->segment_index >= manifest->segment_count ||
                (extent->kind != EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                 extent->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT &&
                 extent->kind != EVOKE_POSTING_EXTENT_LEXICAL_IMPACT) ||
                extent->posting_count == 0)
            {
                return EVOKE_ERR_FORMAT;
            }
            segment = &manifest->segments[extent->segment_index];
            if (extent->posting_offset > segment->posting_count ||
                extent->posting_count >
                    segment->posting_count - extent->posting_offset ||
                (extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                 (segment->flags & EVOKE_SEGMENT_FLAG_LEXICAL) == 0) ||
                (extent->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT &&
                 (segment->flags & EVOKE_SEGMENT_FLAG_SEMANTIC) == 0) ||
                (extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT &&
                 (segment->flags & EVOKE_SEGMENT_FLAG_LEXICAL) == 0) ||
                (has_previous &&
                 (extent->segment_index < previous_segment_index ||
                  (extent->segment_index == previous_segment_index &&
                   extent->posting_offset <= previous_posting_offset))))
            {
                return EVOKE_ERR_FORMAT;
            }
            previous_segment_index = extent->segment_index;
            previous_posting_offset = extent->posting_offset;
            has_previous = true;
        }
    }
    return EVOKE_OK;
}

evoke_status
evoke_term_directory_validate(
    const evoke_term_directory *directory,
    const evoke_segment_manifest *manifest
)
{
    evoke_status status = evoke_segment_manifest_validate(manifest);

    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_term_directory_validate_logical(directory, manifest);
}

evoke_status
evoke_term_directory_serialized_size(
    const evoke_term_directory *directory,
    const evoke_segment_manifest *manifest,
    size_t *size_out
)
{
    size_t offset_bytes;
    size_t extent_bytes;
    size_t total_size;
    evoke_status status;

    if (size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *size_out = 0;
    status = evoke_term_directory_validate_logical(
        directory,
        manifest
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (!evoke_checked_mul_size(
            (size_t) directory->vocab_size + 1,
            sizeof(uint64_t),
            &offset_bytes) ||
        !evoke_checked_mul_size(
            directory->extent_count,
            EVOKE_TERM_EXTENT_DESCRIPTOR_SIZE,
            &extent_bytes) ||
        !evoke_checked_add_size(
            EVOKE_TERM_DIRECTORY_HEADER_SIZE,
            offset_bytes,
            &total_size) ||
        !evoke_checked_add_size(total_size, extent_bytes, &total_size))
    {
        return EVOKE_ERR_RANGE;
    }
    *size_out = total_size;
    return EVOKE_OK;
}

evoke_status
evoke_term_directory_serialize(
    const evoke_term_directory *directory,
    const evoke_segment_manifest *manifest,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes;
    size_t total_size;
    size_t offsets_offset = EVOKE_TERM_DIRECTORY_HEADER_SIZE;
    size_t extents_offset;
    uint32_t i;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_term_directory_serialized_size(
        directory,
        manifest,
        &total_size
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    extents_offset =
        offsets_offset +
        ((size_t) directory->vocab_size + 1) * sizeof(uint64_t);

    bytes = calloc(total_size, 1);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    evoke_write_u32_le(bytes + 0, EVOKE_TERM_DIRECTORY_MAGIC);
    evoke_write_u16_le(bytes + 4, EVOKE_TERM_DIRECTORY_VERSION);
    evoke_write_u16_le(bytes + 6, EVOKE_TERM_DIRECTORY_HEADER_SIZE);
    evoke_write_u16_le(bytes + 8, EVOKE_TERM_EXTENT_DESCRIPTOR_SIZE);
    evoke_write_u32_le(bytes + 12, directory->vocab_size);
    evoke_write_u32_le(bytes + 16, directory->extent_count);
    evoke_write_u32_le(
        bytes + 20,
        EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM
    );
    evoke_write_u64_le(bytes + 24, offsets_offset);
    evoke_write_u64_le(bytes + 32, extents_offset);
    evoke_write_u64_le(bytes + 40, total_size);

    for (i = 0; i <= directory->vocab_size; i++)
    {
        evoke_write_u64_le(
            bytes + offsets_offset + (size_t) i * sizeof(uint64_t),
            directory->term_offsets[i]
        );
    }
    for (i = 0; i < directory->extent_count; i++)
    {
        uint8_t *extent_bytes =
            bytes + extents_offset +
            (size_t) i * EVOKE_TERM_EXTENT_DESCRIPTOR_SIZE;

        evoke_write_u32_le(
            extent_bytes + 0,
            directory->extents[i].segment_index
        );
        evoke_write_u32_le(
            extent_bytes + 4,
            (uint32_t) directory->extents[i].kind
        );
        evoke_write_u64_le(
            extent_bytes + 8,
            directory->extents[i].posting_offset
        );
        evoke_write_u64_le(
            extent_bytes + 16,
            directory->extents[i].posting_count
        );
    }
    evoke_write_u64_le(
        bytes + EVOKE_TERM_DIRECTORY_CHECKSUM_OFFSET,
        evoke_checksum_with_zero_range(
            bytes,
            total_size,
            EVOKE_TERM_DIRECTORY_CHECKSUM_OFFSET,
            sizeof(uint64_t)
        )
    );

    *bytes_out = bytes;
    *size_out = total_size;
    return EVOKE_OK;
}

evoke_status
evoke_term_directory_deserialize(
    const uint8_t *bytes,
    size_t size,
    const evoke_segment_manifest *manifest,
    evoke_term_directory *directory_out
)
{
    evoke_term_directory directory;
    uint64_t offsets_offset;
    uint64_t extents_offset;
    uint64_t total_size;
    uint64_t expected_extents_offset;
    uint64_t checksum;
    uint32_t max_extents_per_term;
    uint32_t i;
    evoke_status status;

    if (bytes == NULL || manifest == NULL || directory_out == NULL ||
        size < EVOKE_TERM_DIRECTORY_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_read_u32_le(bytes + 0) != EVOKE_TERM_DIRECTORY_MAGIC ||
        evoke_read_u16_le(bytes + 4) != EVOKE_TERM_DIRECTORY_VERSION ||
        evoke_read_u16_le(bytes + 6) != EVOKE_TERM_DIRECTORY_HEADER_SIZE ||
        evoke_read_u16_le(bytes + 8) != EVOKE_TERM_EXTENT_DESCRIPTOR_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }

    max_extents_per_term = evoke_read_u32_le(bytes + 20);
    if (max_extents_per_term == 0 ||
        max_extents_per_term >
            EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM)
    {
        return EVOKE_ERR_FORMAT;
    }

    evoke_term_directory_init(&directory);
    directory.vocab_size = evoke_read_u32_le(bytes + 12);
    directory.extent_count = evoke_read_u32_le(bytes + 16);
    offsets_offset = evoke_read_u64_le(bytes + 24);
    extents_offset = evoke_read_u64_le(bytes + 32);
    total_size = evoke_read_u64_le(bytes + 40);
    checksum = evoke_read_u64_le(
        bytes + EVOKE_TERM_DIRECTORY_CHECKSUM_OFFSET
    );
    if (directory.vocab_size != manifest->vocab_size ||
        total_size != size ||
        offsets_offset != EVOKE_TERM_DIRECTORY_HEADER_SIZE ||
        checksum != evoke_checksum_with_zero_range(
            bytes,
            size,
            EVOKE_TERM_DIRECTORY_CHECKSUM_OFFSET,
            sizeof(uint64_t)))
    {
        return EVOKE_ERR_FORMAT;
    }
    if ((uint64_t) directory.vocab_size + 1 >
        (UINT64_MAX - offsets_offset) / sizeof(uint64_t))
    {
        return EVOKE_ERR_FORMAT;
    }
    expected_extents_offset =
        offsets_offset +
        ((uint64_t) directory.vocab_size + 1) * sizeof(uint64_t);
    if (extents_offset != expected_extents_offset ||
        extents_offset > size ||
        directory.extent_count >
            (size - (size_t) extents_offset) /
            EVOKE_TERM_EXTENT_DESCRIPTOR_SIZE ||
        extents_offset +
            (uint64_t) directory.extent_count *
            EVOKE_TERM_EXTENT_DESCRIPTOR_SIZE != size)
    {
        return EVOKE_ERR_FORMAT;
    }

    directory.term_offsets = calloc(
        (size_t) directory.vocab_size + 1,
        sizeof(*directory.term_offsets)
    );
    if (directory.term_offsets == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    if (directory.extent_count > 0)
    {
        directory.extents = calloc(
            directory.extent_count,
            sizeof(*directory.extents)
        );
        if (directory.extents == NULL)
        {
            evoke_term_directory_free(&directory);
            return EVOKE_ERR_NOMEM;
        }
    }

    for (i = 0; i <= directory.vocab_size; i++)
    {
        directory.term_offsets[i] = evoke_read_u64_le(
            bytes + (size_t) offsets_offset +
                (size_t) i * sizeof(uint64_t)
        );
    }
    for (i = 0; i < directory.vocab_size; i++)
    {
        if (directory.term_offsets[i + 1] <
                directory.term_offsets[i] ||
            directory.term_offsets[i + 1] -
                directory.term_offsets[i] > max_extents_per_term)
        {
            evoke_term_directory_free(&directory);
            return EVOKE_ERR_FORMAT;
        }
    }
    for (i = 0; i < directory.extent_count; i++)
    {
        const uint8_t *extent_bytes =
            bytes + (size_t) extents_offset +
            (size_t) i * EVOKE_TERM_EXTENT_DESCRIPTOR_SIZE;

        directory.extents[i].segment_index =
            evoke_read_u32_le(extent_bytes + 0);
        directory.extents[i].kind =
            (evoke_posting_extent_kind) evoke_read_u32_le(
                extent_bytes + 4
            );
        directory.extents[i].posting_offset =
            evoke_read_u64_le(extent_bytes + 8);
        directory.extents[i].posting_count =
            evoke_read_u64_le(extent_bytes + 16);
    }

    status = evoke_term_directory_validate(&directory, manifest);
    if (status != EVOKE_OK)
    {
        evoke_term_directory_free(&directory);
        return status;
    }

    evoke_term_directory_free(directory_out);
    *directory_out = directory;
    return EVOKE_OK;
}

static bool
evoke_one_or_zero_bits(uint16_t value)
{
    return value == 0 || (value & (uint16_t) (value - 1)) == 0;
}

static bool
evoke_bytes_are_zero(const uint8_t *bytes, size_t size)
{
    size_t i;

    for (i = 0; i < size; i++)
    {
        if (bytes[i] != 0)
        {
            return false;
        }
    }
    return true;
}

evoke_status
evoke_document_version_records_validate(
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *segment,
    const evoke_document_version_record *versions,
    size_t version_count,
    const evoke_document_retirement_record *retirements,
    size_t retirement_count
)
{
    size_t i;

    if (manifest == NULL || segment == NULL ||
        (version_count > 0 && versions == NULL) ||
        (retirement_count > 0 && retirements == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    if (version_count != segment->document_count ||
        retirement_count != segment->retirement_count ||
        version_count > manifest->document_slot_count ||
        retirement_count > manifest->document_slot_count)
    {
        return EVOKE_ERR_FORMAT;
    }

    for (i = 0; i < version_count; i++)
    {
        const evoke_document_version_record *record = &versions[i];
        bool is_aborted_hole =
            (record->flags &
             EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) != 0;
        uint16_t semantic_flags =
            record->flags &
            (EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING |
             EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE |
             EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_QUARANTINED);

        if (record->document_slot >= manifest->document_slot_count ||
            record->born_sequence < segment->min_sequence ||
            record->born_sequence > segment->max_sequence ||
            (record->flags & ~EVOKE_DOCUMENT_VERSION_KNOWN_FLAGS) != 0 ||
            !evoke_one_or_zero_bits(semantic_flags) ||
            (((record->flags &
               EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID) != 0) !=
             (record->record_xid == 0)) ||
            (i > 0 &&
             versions[i - 1].document_slot >= record->document_slot))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (is_aborted_hole)
        {
            if (record->flags !=
                    (EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE |
                     EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID) ||
                record->record_xid != 0 ||
                record->heap_block != 0 ||
                record->document_length != 0 ||
                record->heap_offset != 0 ||
                !evoke_bytes_are_zero(
                    record->semantic_input_fingerprint,
                    sizeof(record->semantic_input_fingerprint)))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        else if (record->heap_offset == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    for (i = 0; i < retirement_count; i++)
    {
        const evoke_document_retirement_record *record = &retirements[i];

        if (record->document_slot >= manifest->document_slot_count ||
            record->retirement_sequence < segment->min_sequence ||
            record->retirement_sequence > segment->max_sequence ||
            record->reserved != 0 ||
            record->reserved2 != 0 ||
            (record->flags & ~EVOKE_DOCUMENT_RETIREMENT_KNOWN_FLAGS) != 0 ||
            (((record->flags &
               EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID) != 0) !=
             (record->record_xid == 0)) ||
            (i > 0 &&
             retirements[i - 1].document_slot >=
                record->document_slot))
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    return EVOKE_OK;
}

evoke_status
evoke_semantic_state_records_validate(
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *segment,
    const evoke_semantic_state_record *states,
    size_t state_count
)
{
    size_t state_index;

    if (manifest == NULL || segment == NULL ||
        (state_count > 0 && states == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    if (state_count != segment->semantic_state_count ||
        state_count > manifest->document_slot_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (state_index = 0; state_index < state_count; state_index++)
    {
        const evoke_semantic_state_record *state = &states[state_index];
        uint16_t outcome_flags =
            state->flags &
            (EVOKE_SEMANTIC_STATE_FLAG_COMPLETE |
             EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED);
        bool complete =
            outcome_flags == EVOKE_SEMANTIC_STATE_FLAG_COMPLETE;
        bool quarantined =
            outcome_flags == EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED;

        if (state->document_slot >= manifest->document_slot_count ||
            state->transition_sequence < segment->min_sequence ||
            state->transition_sequence > segment->max_sequence ||
            (state->flags & ~EVOKE_SEMANTIC_STATE_KNOWN_FLAGS) != 0 ||
            (!complete && !quarantined) ||
            (((state->flags &
               EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID) != 0) !=
             (state->record_xid == 0)) ||
            state->reserved != 0 ||
            evoke_bytes_are_zero(
                state->semantic_input_fingerprint,
                sizeof(state->semantic_input_fingerprint)) ||
            (state_index > 0 &&
             states[state_index - 1].document_slot >=
                state->document_slot))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (complete &&
            (state->failure_count != 0 ||
             state->error_code != 0 ||
             state->retry_after != 0 ||
             state->pending_since != 0 ||
             state->error_hash != 0))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (quarantined &&
            (state->failure_count == 0 ||
             state->error_code == 0 ||
             state->pending_since <= 0 ||
             state->retry_after < state->pending_since))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    return EVOKE_OK;
}

static int
evoke_compare_u32_ascending(const void *left, const void *right)
{
    uint32_t left_value = *(const uint32_t *) left;
    uint32_t right_value = *(const uint32_t *) right;

    return (left_value > right_value) - (left_value < right_value);
}

evoke_status
evoke_segment_frozen_retirement_ids_build(
    const evoke_segment_manifest *manifest,
    const evoke_segment_payload *payloads,
    size_t payload_count,
    uint32_t **document_ids_out,
    size_t *document_id_count_out
)
{
    uint32_t *document_ids = NULL;
    size_t total_count = 0;
    size_t cursor = 0;
    size_t payload_index;
    evoke_status status;

    if (manifest == NULL || document_ids_out == NULL ||
        document_id_count_out == NULL ||
        payload_count != manifest->segment_count ||
        (payload_count > 0 && payloads == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    *document_ids_out = NULL;
    *document_id_count_out = 0;

    status = evoke_segment_manifest_validate(manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    for (payload_index = 0; payload_index < payload_count; payload_index++)
    {
        const evoke_segment_descriptor *segment =
            &manifest->segments[payload_index];
        const evoke_segment_payload *payload = &payloads[payload_index];

        if (payload->segment_id != segment->segment_id ||
            payload->retirement_count != segment->retirement_count ||
            (payload->retirement_count > 0 &&
             payload->retirements == NULL) ||
            SIZE_MAX - total_count < payload->retirement_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        total_count += payload->retirement_count;
    }
    if (total_count > manifest->document_slot_count ||
        total_count > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    if (total_count > 0)
    {
        document_ids = malloc(total_count * sizeof(*document_ids));
        if (document_ids == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
    }

    for (payload_index = 0; payload_index < payload_count; payload_index++)
    {
        const evoke_segment_payload *payload = &payloads[payload_index];
        uint32_t retirement_index;

        for (retirement_index = 0;
             retirement_index < payload->retirement_count;
             retirement_index++)
        {
            const evoke_document_retirement_record *retirement =
                &payload->retirements[retirement_index];

            if (retirement->document_slot >=
                    manifest->document_slot_count ||
                retirement->document_slot > UINT32_MAX ||
                retirement->record_xid != 0 ||
                retirement->flags !=
                    EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID ||
                retirement->reserved != 0 ||
                retirement->reserved2 != 0 ||
                (retirement_index > 0 &&
                 payload->retirements[retirement_index - 1].
                    document_slot >= retirement->document_slot))
            {
                free(document_ids);
                return EVOKE_ERR_FORMAT;
            }
            document_ids[cursor++] = (uint32_t) retirement->document_slot;
        }
    }
    if (cursor != total_count)
    {
        free(document_ids);
        return EVOKE_ERR_FORMAT;
    }
    if (total_count > 1)
    {
        qsort(
            document_ids,
            total_count,
            sizeof(*document_ids),
            evoke_compare_u32_ascending
        );
        for (cursor = 1; cursor < total_count; cursor++)
        {
            if (document_ids[cursor - 1] == document_ids[cursor])
            {
                free(document_ids);
                return EVOKE_ERR_FORMAT;
            }
        }
    }

    *document_ids_out = document_ids;
    *document_id_count_out = total_count;
    return EVOKE_OK;
}

evoke_status
evoke_segment_posting_payload_validate(
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *segment,
    const evoke_segment_payload_view *payload
)
{
    uint64_t posting_cursor = 0;
    uint32_t run_index;
    uint32_t i;

    if (manifest == NULL || segment == NULL || payload == NULL ||
        payload->segment_id != segment->segment_id ||
        payload->posting_count != segment->posting_count ||
        segment->document_slot_count > UINT32_MAX ||
        payload->local_document_count !=
            (uint32_t) segment->document_slot_count ||
        (payload->run_count > 0 && payload->runs == NULL) ||
        (payload->posting_count > 0 &&
         (payload->indices == NULL ||
          (payload->values == NULL &&
           payload->data == NULL &&
           payload->term_frequencies == NULL))))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (payload->document_id_map == NULL)
    {
        if (segment->first_document_slot > UINT32_MAX ||
            payload->document_id_base !=
                (uint32_t) segment->first_document_slot ||
            (uint64_t) payload->document_id_base +
                payload->local_document_count >
                manifest->document_slot_count)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    else
    {
        for (i = 0; i < payload->local_document_count; i++)
        {
            if (payload->document_id_map[i] >=
                    manifest->document_slot_count ||
                (i > 0 &&
                 payload->document_id_map[i - 1] >=
                    payload->document_id_map[i]))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }

    for (run_index = 0; run_index < payload->run_count; run_index++)
    {
        const evoke_segment_term_run *run = &payload->runs[run_index];
        uint64_t end;
        uint64_t posting_index;

        if (run->term_id >= manifest->vocab_size ||
            run->posting_count == 0 ||
            run->posting_offset != posting_cursor ||
            run->posting_offset > payload->posting_count ||
            run->posting_count >
                payload->posting_count - run->posting_offset ||
            (run->kind != EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
             run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT &&
             run->kind != EVOKE_POSTING_EXTENT_LEXICAL_IMPACT) ||
            (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
             (segment->flags & EVOKE_SEGMENT_FLAG_LEXICAL) == 0) ||
            (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT &&
             (segment->flags & EVOKE_SEGMENT_FLAG_SEMANTIC) == 0) ||
            (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT &&
             (segment->flags & EVOKE_SEGMENT_FLAG_LEXICAL) == 0) ||
            (run_index > 0 &&
             (payload->runs[run_index - 1].term_id > run->term_id ||
              (payload->runs[run_index - 1].term_id == run->term_id &&
               payload->runs[run_index - 1].kind >= run->kind))))
        {
            return EVOKE_ERR_FORMAT;
        }
        end = run->posting_offset + run->posting_count;
        for (posting_index = run->posting_offset;
             posting_index < end;
             posting_index++)
        {
            if (payload->indices[posting_index] >=
                    payload->local_document_count ||
                (posting_index > run->posting_offset &&
                 payload->indices[posting_index - 1] >=
                    payload->indices[posting_index]) ||
                (run->kind ==
                     EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                 ((payload->values != NULL &&
                   payload->values[posting_index].term_frequency == 0) ||
                  (payload->values == NULL &&
                   (payload->term_frequencies == NULL ||
                    payload->term_frequencies[posting_index] == 0)))) ||
                (run->kind !=
                     EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                 ((payload->values != NULL &&
                   !isfinite(payload->values[posting_index].impact)) ||
                  (payload->values == NULL &&
                   (payload->data == NULL ||
                    !isfinite(payload->data[posting_index]))))))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        posting_cursor = end;
    }
    if (posting_cursor != payload->posting_count)
    {
        return EVOKE_ERR_FORMAT;
    }

    return EVOKE_OK;
}

evoke_status
evoke_segment_payload_view_validate(
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *segment,
    const evoke_segment_payload_view *payload,
    const evoke_document_version_record *versions,
    size_t version_count,
    const evoke_document_retirement_record *retirements,
    size_t retirement_count
)
{
    evoke_status status = evoke_segment_posting_payload_validate(
        manifest,
        segment,
        payload
    );

    if (status != EVOKE_OK)
    {
        return status;
    }
    return evoke_document_version_records_validate(
        manifest,
        segment,
        versions,
        version_count,
        retirements,
        retirement_count
    );
}

evoke_status
evoke_term_directory_build_from_payloads(
    const evoke_segment_manifest *manifest,
    const evoke_segment_payload_view *payloads,
    size_t payload_count,
    evoke_term_directory *directory_out
)
{
    evoke_term_directory directory;
    uint64_t *write_offsets = NULL;
    uint64_t extent_count = 0;
    uint32_t segment_index;
    uint32_t term_id;
    evoke_status status;

    if (manifest == NULL || directory_out == NULL ||
        payload_count != manifest->segment_count ||
        (payload_count > 0 && payloads == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_term_directory_init(&directory);
    directory.vocab_size = manifest->vocab_size;
    directory.term_offsets = calloc(
        (size_t) directory.vocab_size + 1,
        sizeof(*directory.term_offsets)
    );
    write_offsets = calloc(
        (size_t) directory.vocab_size + 1,
        sizeof(*write_offsets)
    );
    if (directory.term_offsets == NULL || write_offsets == NULL)
    {
        free(write_offsets);
        evoke_term_directory_free(&directory);
        return EVOKE_ERR_NOMEM;
    }

    for (segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        const evoke_segment_payload_view *payload =
            &payloads[segment_index];
        uint32_t run_index;

        status = evoke_segment_posting_payload_validate(
            manifest,
            &manifest->segments[segment_index],
            payload
        );
        if (status != EVOKE_OK)
        {
            free(write_offsets);
            evoke_term_directory_free(&directory);
            return status;
        }
        for (run_index = 0; run_index < payload->run_count; run_index++)
        {
            uint32_t run_term = payload->runs[run_index].term_id;

            if (directory.term_offsets[run_term + 1] ==
                EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM)
            {
                free(write_offsets);
                evoke_term_directory_free(&directory);
                return EVOKE_ERR_FORMAT;
            }
            directory.term_offsets[run_term + 1]++;
            extent_count++;
        }
    }
    if (extent_count > UINT32_MAX)
    {
        free(write_offsets);
        evoke_term_directory_free(&directory);
        return EVOKE_ERR_RANGE;
    }
    for (term_id = 0; term_id < directory.vocab_size; term_id++)
    {
        directory.term_offsets[term_id + 1] +=
            directory.term_offsets[term_id];
        write_offsets[term_id] = directory.term_offsets[term_id];
    }
    write_offsets[directory.vocab_size] =
        directory.term_offsets[directory.vocab_size];
    directory.extent_count = (uint32_t) extent_count;
    if (directory.extent_count > 0)
    {
        directory.extents = calloc(
            directory.extent_count,
            sizeof(*directory.extents)
        );
        if (directory.extents == NULL)
        {
            free(write_offsets);
            evoke_term_directory_free(&directory);
            return EVOKE_ERR_NOMEM;
        }
    }

    for (segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        const evoke_segment_payload_view *payload =
            &payloads[segment_index];
        uint32_t run_index;

        for (run_index = 0; run_index < payload->run_count; run_index++)
        {
            const evoke_segment_term_run *run = &payload->runs[run_index];
            uint64_t destination = write_offsets[run->term_id]++;

            directory.extents[destination].segment_index = segment_index;
            directory.extents[destination].kind = run->kind;
            directory.extents[destination].posting_offset =
                run->posting_offset;
            directory.extents[destination].posting_count =
                run->posting_count;
        }
    }

    status = evoke_term_directory_validate_logical(
        &directory,
        manifest
    );
    free(write_offsets);
    if (status != EVOKE_OK)
    {
        evoke_term_directory_free(&directory);
        return status;
    }
    evoke_term_directory_free(directory_out);
    *directory_out = directory;
    return EVOKE_OK;
}

bool
evoke_segment_descriptor_equal(
    const evoke_segment_descriptor *left,
    const evoke_segment_descriptor *right
)
{
    return left != NULL &&
        right != NULL &&
        left->segment_id == right->segment_id &&
        left->min_sequence == right->min_sequence &&
        left->max_sequence == right->max_sequence &&
        left->posting_count == right->posting_count &&
        left->retirement_count == right->retirement_count &&
        left->document_count == right->document_count &&
        left->total_document_length == right->total_document_length &&
        left->first_document_slot == right->first_document_slot &&
        left->document_slot_count == right->document_slot_count &&
        left->payload_checksum == right->payload_checksum &&
        left->start_block == right->start_block &&
        left->block_count == right->block_count &&
        left->size_class == right->size_class &&
        left->flags == right->flags &&
        left->payload_bytes == right->payload_bytes &&
        left->payload_owner_manifest_id ==
            right->payload_owner_manifest_id &&
        left->semantic_state_count == right->semantic_state_count;
}

evoke_status
evoke_term_directory_append_payload(
    const evoke_term_directory *old_directory,
    const evoke_segment_manifest *old_manifest,
    const evoke_segment_manifest *next_manifest,
    const evoke_segment_payload_view *new_payload,
    evoke_term_directory *directory_out
)
{
    evoke_term_directory directory;
    uint32_t *new_run_counts = NULL;
    uint64_t extent_count;
    uint64_t extent_cursor = 0;
    uint32_t new_segment_index;
    uint32_t term_id;
    uint32_t run_index;
    evoke_status status;

    if (old_directory == NULL || old_manifest == NULL ||
        next_manifest == NULL || new_payload == NULL ||
        directory_out == NULL ||
        old_manifest->segment_count >=
            EVOKE_SEGMENT_MANIFEST_MAX_SEGMENTS ||
        next_manifest->segment_count != old_manifest->segment_count + 1 ||
        next_manifest->segments == NULL ||
        next_manifest->vocab_size < old_manifest->vocab_size ||
        next_manifest->manifest_id <= old_manifest->manifest_id ||
        next_manifest->parent_manifest_id != old_manifest->manifest_id ||
        memcmp(
            next_manifest->contract_hash,
            old_manifest->contract_hash,
            EVOKE_SEGMENT_CONTRACT_HASH_BYTES
        ) != 0)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_term_directory_validate(old_directory, old_manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    for (new_segment_index = 0;
         new_segment_index < old_manifest->segment_count;
         new_segment_index++)
    {
        if (!evoke_segment_descriptor_equal(
                &old_manifest->segments[new_segment_index],
                &next_manifest->segments[new_segment_index]))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    new_segment_index = old_manifest->segment_count;
    status = evoke_segment_posting_payload_validate(
        next_manifest,
        &next_manifest->segments[new_segment_index],
        new_payload
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (new_payload->run_count >
        UINT32_MAX - old_directory->extent_count)
    {
        return EVOKE_ERR_RANGE;
    }

    evoke_term_directory_init(&directory);
    directory.vocab_size = next_manifest->vocab_size;
    extent_count =
        (uint64_t) old_directory->extent_count + new_payload->run_count;
    directory.extent_count = (uint32_t) extent_count;
    directory.term_offsets = calloc(
        (size_t) directory.vocab_size + 1,
        sizeof(*directory.term_offsets)
    );
    new_run_counts = calloc(
        (size_t) directory.vocab_size,
        sizeof(*new_run_counts)
    );
    if (directory.term_offsets == NULL ||
        (directory.vocab_size > 0 && new_run_counts == NULL))
    {
        free(new_run_counts);
        evoke_term_directory_free(&directory);
        return EVOKE_ERR_NOMEM;
    }
    if (directory.extent_count > 0)
    {
        directory.extents = calloc(
            directory.extent_count,
            sizeof(*directory.extents)
        );
        if (directory.extents == NULL)
        {
            free(new_run_counts);
            evoke_term_directory_free(&directory);
            return EVOKE_ERR_NOMEM;
        }
    }

    for (run_index = 0; run_index < new_payload->run_count; run_index++)
    {
        new_run_counts[new_payload->runs[run_index].term_id]++;
    }
    run_index = 0;
    for (term_id = 0; term_id < directory.vocab_size; term_id++)
    {
        uint64_t old_start = 0;
        uint64_t old_end = 0;
        uint64_t term_extent_count;

        if (term_id < old_directory->vocab_size)
        {
            old_start = old_directory->term_offsets[term_id];
            old_end = old_directory->term_offsets[term_id + 1];
        }
        term_extent_count =
            old_end - old_start + new_run_counts[term_id];
        if (term_extent_count >
            EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM)
        {
            free(new_run_counts);
            evoke_term_directory_free(&directory);
            return EVOKE_ERR_FORMAT;
        }
        directory.term_offsets[term_id] = extent_cursor;
        if (old_end > old_start)
        {
            memcpy(
                &directory.extents[extent_cursor],
                &old_directory->extents[old_start],
                (size_t) (old_end - old_start) *
                    sizeof(*directory.extents)
            );
            extent_cursor += old_end - old_start;
        }
        while (run_index < new_payload->run_count &&
               new_payload->runs[run_index].term_id == term_id)
        {
            const evoke_segment_term_run *run =
                &new_payload->runs[run_index++];
            evoke_term_extent_descriptor *extent =
                &directory.extents[extent_cursor++];

            extent->segment_index = new_segment_index;
            extent->kind = run->kind;
            extent->posting_offset = run->posting_offset;
            extent->posting_count = run->posting_count;
        }
    }
    directory.term_offsets[directory.vocab_size] = extent_cursor;
    free(new_run_counts);
    if (run_index != new_payload->run_count ||
        extent_cursor != directory.extent_count)
    {
        evoke_term_directory_free(&directory);
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_term_directory_validate_logical(
        &directory,
        next_manifest
    );
    if (status != EVOKE_OK)
    {
        evoke_term_directory_free(&directory);
        return status;
    }
    evoke_term_directory_free(directory_out);
    *directory_out = directory;
    return EVOKE_OK;
}

evoke_status
evoke_term_directory_replace_payloads(
    const evoke_term_directory *old_directory,
    const evoke_segment_manifest *old_manifest,
    const evoke_segment_manifest *next_manifest,
    uint32_t first_segment_index,
    uint32_t replaced_segment_count,
    const evoke_segment_payload_view *replacement_payload,
    evoke_term_directory *directory_out
)
{
    evoke_term_directory directory;
    uint32_t *replacement_run_counts = NULL;
    uint32_t replaced_end;
    uint32_t replacement_run_index = 0;
    uint64_t extent_count = 0;
    uint64_t extent_cursor = 0;
    uint32_t term_id;
    uint32_t segment_index;
    evoke_status status;

    if (old_directory == NULL || old_manifest == NULL ||
        next_manifest == NULL || replacement_payload == NULL ||
        directory_out == NULL || replaced_segment_count < 2 ||
        first_segment_index >= old_manifest->segment_count ||
        replaced_segment_count >
            old_manifest->segment_count - first_segment_index ||
        next_manifest->segment_count !=
            old_manifest->segment_count - replaced_segment_count + 1 ||
        next_manifest->segments == NULL ||
        next_manifest->vocab_size != old_manifest->vocab_size ||
        next_manifest->manifest_id <= old_manifest->manifest_id ||
        next_manifest->parent_manifest_id != old_manifest->manifest_id ||
        memcmp(
            next_manifest->contract_hash,
            old_manifest->contract_hash,
            EVOKE_SEGMENT_CONTRACT_HASH_BYTES
        ) != 0)
    {
        return EVOKE_ERR_INVALID;
    }
    replaced_end = first_segment_index + replaced_segment_count;
    status = evoke_term_directory_validate(old_directory, old_manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    for (segment_index = 0;
         segment_index < first_segment_index;
         segment_index++)
    {
        if (!evoke_segment_descriptor_equal(
                &old_manifest->segments[segment_index],
                &next_manifest->segments[segment_index]))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    for (segment_index = replaced_end;
         segment_index < old_manifest->segment_count;
         segment_index++)
    {
        uint32_t next_segment_index =
            segment_index - replaced_segment_count + 1;

        if (!evoke_segment_descriptor_equal(
                &old_manifest->segments[segment_index],
                &next_manifest->segments[next_segment_index]))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    status = evoke_segment_posting_payload_validate(
        next_manifest,
        &next_manifest->segments[first_segment_index],
        replacement_payload
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    evoke_term_directory_init(&directory);
    directory.vocab_size = next_manifest->vocab_size;
    replacement_run_counts = calloc(
        directory.vocab_size,
        sizeof(*replacement_run_counts)
    );
    if (directory.vocab_size > 0 && replacement_run_counts == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t run_index = 0;
         run_index < replacement_payload->run_count;
         run_index++)
    {
        replacement_run_counts[
            replacement_payload->runs[run_index].term_id
        ]++;
    }
    for (term_id = 0; term_id < directory.vocab_size; term_id++)
    {
        uint64_t start = old_directory->term_offsets[term_id];
        uint64_t end = old_directory->term_offsets[term_id + 1];
        uint64_t retained_count = 0;

        for (uint64_t old_extent_index = start;
             old_extent_index < end;
             old_extent_index++)
        {
            uint32_t old_segment_index =
                old_directory->extents[old_extent_index].segment_index;

            if (old_segment_index < first_segment_index ||
                old_segment_index >= replaced_end)
            {
                retained_count++;
            }
        }
        if (retained_count + replacement_run_counts[term_id] >
                EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM ||
            extent_count > UINT32_MAX -
                retained_count - replacement_run_counts[term_id])
        {
            free(replacement_run_counts);
            return EVOKE_ERR_FORMAT;
        }
        extent_count +=
            retained_count + replacement_run_counts[term_id];
    }

    directory.extent_count = (uint32_t) extent_count;
    directory.term_offsets = calloc(
        (size_t) directory.vocab_size + 1,
        sizeof(*directory.term_offsets)
    );
    if (directory.extent_count > 0)
    {
        directory.extents = calloc(
            directory.extent_count,
            sizeof(*directory.extents)
        );
    }
    if (directory.term_offsets == NULL ||
        (directory.extent_count > 0 && directory.extents == NULL))
    {
        free(replacement_run_counts);
        evoke_term_directory_free(&directory);
        return EVOKE_ERR_NOMEM;
    }

    for (term_id = 0; term_id < directory.vocab_size; term_id++)
    {
        uint64_t start = old_directory->term_offsets[term_id];
        uint64_t end = old_directory->term_offsets[term_id + 1];

        directory.term_offsets[term_id] = extent_cursor;
        for (uint64_t old_extent_index = start;
             old_extent_index < end;
             old_extent_index++)
        {
            const evoke_term_extent_descriptor *old_extent =
                &old_directory->extents[old_extent_index];

            if (old_extent->segment_index >= first_segment_index)
            {
                break;
            }
            directory.extents[extent_cursor++] = *old_extent;
        }
        while (replacement_run_index <
                   replacement_payload->run_count &&
               replacement_payload->runs[replacement_run_index].term_id ==
                   term_id)
        {
            const evoke_segment_term_run *run =
                &replacement_payload->runs[replacement_run_index++];
            evoke_term_extent_descriptor *extent =
                &directory.extents[extent_cursor++];

            extent->segment_index = first_segment_index;
            extent->kind = run->kind;
            extent->posting_offset = run->posting_offset;
            extent->posting_count = run->posting_count;
        }
        for (uint64_t old_extent_index = start;
             old_extent_index < end;
             old_extent_index++)
        {
            const evoke_term_extent_descriptor *old_extent =
                &old_directory->extents[old_extent_index];
            evoke_term_extent_descriptor *extent;

            if (old_extent->segment_index < replaced_end)
            {
                continue;
            }
            extent = &directory.extents[extent_cursor++];
            *extent = *old_extent;
            extent->segment_index =
                old_extent->segment_index - replaced_segment_count + 1;
        }
    }
    directory.term_offsets[directory.vocab_size] = extent_cursor;
    free(replacement_run_counts);
    if (replacement_run_index != replacement_payload->run_count ||
        extent_cursor != directory.extent_count)
    {
        evoke_term_directory_free(&directory);
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_term_directory_validate_logical(
        &directory,
        next_manifest
    );
    if (status != EVOKE_OK)
    {
        evoke_term_directory_free(&directory);
        return status;
    }
    evoke_term_directory_free(directory_out);
    *directory_out = directory;
    return EVOKE_OK;
}

typedef struct evoke_persistent_block_directory
{
    uint32_t block_shift;
    uint32_t block_count;
    evoke_posting_block_record *blocks;
    uint64_t *run_block_offsets;
    uint32_t *run_block_counts;
} evoke_persistent_block_directory;

static void
evoke_persistent_block_directory_init(
    evoke_persistent_block_directory *directory
)
{
    memset(directory, 0, sizeof(*directory));
}

static void
evoke_persistent_block_directory_free(
    evoke_persistent_block_directory *directory
)
{
    if (directory == NULL)
    {
        return;
    }
    free(directory->blocks);
    free(directory->run_block_offsets);
    free(directory->run_block_counts);
    memset(directory, 0, sizeof(*directory));
}

static evoke_status
evoke_persistent_block_directory_append(
    evoke_persistent_block_directory *directory,
    uint32_t run_index,
    evoke_posting_block_record *records,
    size_t record_count
)
{
    evoke_posting_block_record *next_blocks;
    size_t next_count;

    if (record_count > UINT32_MAX - directory->block_count)
    {
        return EVOKE_ERR_RANGE;
    }
    directory->run_block_offsets[run_index] =
        directory->block_count;
    directory->run_block_counts[run_index] =
        (uint32_t) record_count;
    if (record_count == 0)
    {
        return EVOKE_OK;
    }
    next_count = (size_t) directory->block_count + record_count;
    if (next_count > SIZE_MAX / sizeof(*next_blocks))
    {
        return EVOKE_ERR_RANGE;
    }
    next_blocks = realloc(
        directory->blocks,
        next_count * sizeof(*next_blocks)
    );
    if (next_blocks == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    directory->blocks = next_blocks;
    memcpy(
        &directory->blocks[directory->block_count],
        records,
        record_count * sizeof(*records)
    );
    directory->block_count = (uint32_t) next_count;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_payload_block_directory_build(
    const evoke_segment_payload *payload,
    bool include_semantic,
    evoke_persistent_block_directory *directory_out
)
{
    evoke_persistent_block_directory directory;
    uint32_t block_shift;
    evoke_status status = EVOKE_OK;

    if (payload == NULL || directory_out == NULL ||
        (payload->run_count > 0 && payload->runs == NULL) ||
        (payload->posting_count > 0 &&
         (payload->indices == NULL || payload->values == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_persistent_block_directory_init(&directory);
    block_shift = payload->block_shift == 0
        ? EVOKE_DEFAULT_POSTING_BLOCK_SHIFT
        : payload->block_shift;
    if (block_shift == 0 || block_shift >= 32)
    {
        return EVOKE_ERR_FORMAT;
    }
    directory.block_shift = block_shift;
    if (payload->run_count > 0)
    {
        directory.run_block_offsets = calloc(
            payload->run_count,
            sizeof(*directory.run_block_offsets)
        );
        directory.run_block_counts = calloc(
            payload->run_count,
            sizeof(*directory.run_block_counts)
        );
        if (directory.run_block_offsets == NULL ||
            directory.run_block_counts == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
    }

    for (uint32_t run_index = 0;
         run_index < payload->run_count;
         run_index++)
    {
        const evoke_segment_term_run *run =
            &payload->runs[run_index];
        evoke_posting_extent extent;
        evoke_posting_block_record *records = NULL;
        size_t record_count = 0;

        if (!include_semantic &&
            run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            status = evoke_persistent_block_directory_append(
                &directory,
                run_index,
                NULL,
                0
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            continue;
        }

        if (run->posting_offset > payload->posting_count ||
            run->posting_count >
                payload->posting_count - run->posting_offset)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        memset(&extent, 0, sizeof(extent));
        extent.indices = &payload->indices[run->posting_offset];
        extent.values = &payload->values[run->posting_offset];
        extent.document_id_map = payload->document_id_map;
        extent.len = run->posting_count;
        extent.document_id_base = payload->document_id_base;
        extent.local_document_count = payload->local_document_count;
        extent.kind = run->kind;
        status = evoke_posting_extent_build_block_records(
            &extent,
            block_shift,
            &records,
            &record_count
        );
        if (status == EVOKE_OK)
        {
            status = evoke_persistent_block_directory_append(
                &directory,
                run_index,
                records,
                record_count
            );
        }
        free(records);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }

    evoke_persistent_block_directory_free(directory_out);
    *directory_out = directory;
    return EVOKE_OK;

cleanup:
    evoke_persistent_block_directory_free(&directory);
    return status;
}

static evoke_status
evoke_term_fold_block_directory_build(
    const evoke_term_fold_bundle *bundle,
    bool include_semantic,
    evoke_persistent_block_directory *directory_out
)
{
    evoke_persistent_block_directory directory;
    uint32_t block_shift;
    evoke_status status = EVOKE_OK;

    if (bundle == NULL || directory_out == NULL ||
        (bundle->run_count > 0 && bundle->runs == NULL) ||
        (bundle->posting_count > 0 &&
         (bundle->document_slots == NULL || bundle->values == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_persistent_block_directory_init(&directory);
    block_shift = bundle->block_shift == 0
        ? EVOKE_DEFAULT_POSTING_BLOCK_SHIFT
        : bundle->block_shift;
    if (block_shift == 0 || block_shift >= 32)
    {
        return EVOKE_ERR_FORMAT;
    }
    directory.block_shift = block_shift;
    if (bundle->run_count > 0)
    {
        directory.run_block_offsets = calloc(
            bundle->run_count,
            sizeof(*directory.run_block_offsets)
        );
        directory.run_block_counts = calloc(
            bundle->run_count,
            sizeof(*directory.run_block_counts)
        );
        if (directory.run_block_offsets == NULL ||
            directory.run_block_counts == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
    }

    for (uint32_t run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle->runs[run_index];
        evoke_posting_extent extent;
        evoke_posting_block_record *records = NULL;
        size_t record_count = 0;
        uint32_t last_document_id;

        if (!include_semantic &&
            run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            status = evoke_persistent_block_directory_append(
                &directory,
                run_index,
                NULL,
                0
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            continue;
        }

        if (run->posting_offset > bundle->posting_count ||
            run->posting_count >
                bundle->posting_count - run->posting_offset ||
            run->posting_count == 0)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        last_document_id = bundle->document_slots[
            run->posting_offset + run->posting_count - 1
        ];
        if (last_document_id == UINT32_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        memset(&extent, 0, sizeof(extent));
        extent.indices =
            &bundle->document_slots[run->posting_offset];
        extent.values = &bundle->values[run->posting_offset];
        extent.len = run->posting_count;
        extent.local_document_count = last_document_id + 1;
        extent.kind = run->kind;
        status = evoke_posting_extent_build_block_records(
            &extent,
            block_shift,
            &records,
            &record_count
        );
        if (status == EVOKE_OK)
        {
            status = evoke_persistent_block_directory_append(
                &directory,
                run_index,
                records,
                record_count
            );
        }
        free(records);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }

    evoke_persistent_block_directory_free(directory_out);
    *directory_out = directory;
    return EVOKE_OK;

cleanup:
    evoke_persistent_block_directory_free(&directory);
    return status;
}

typedef struct evoke_term_fold_layout
{
    size_t runs_offset;
    size_t blocks_offset;
    size_t document_slots_offset;
    size_t values_offset;
    size_t semantic_bmp_offset;
    size_t semantic_bmp_size;
    size_t total_size;
} evoke_term_fold_layout;

static evoke_status
evoke_term_fold_layout_build(
    uint32_t run_count,
    uint32_t block_count,
    uint64_t generic_posting_count,
    size_t semantic_bmp_size,
    evoke_term_fold_layout *layout_out
)
{
    evoke_term_fold_layout layout;
    size_t run_bytes;
    size_t block_bytes;
    size_t posting_bytes;

    if (layout_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (generic_posting_count > SIZE_MAX ||
        !evoke_checked_mul_size(
            run_count,
            EVOKE_TERM_FOLD_RUN_SIZE,
            &run_bytes
        ) ||
        !evoke_checked_mul_size(
            block_count,
            EVOKE_POSTING_BLOCK_RECORD_SIZE,
            &block_bytes
        ) ||
        !evoke_checked_mul_size(
            (size_t) generic_posting_count,
            sizeof(uint32_t),
            &posting_bytes
        ))
    {
        return EVOKE_ERR_RANGE;
    }
    memset(&layout, 0, sizeof(layout));
    layout.runs_offset = EVOKE_TERM_FOLD_HEADER_SIZE;
    if (!evoke_checked_add_size(
            layout.runs_offset,
            run_bytes,
            &layout.blocks_offset
        ) ||
        !evoke_checked_add_size(
            layout.blocks_offset,
            block_bytes,
            &layout.document_slots_offset
        ) ||
        !evoke_checked_add_size(
            layout.document_slots_offset,
            posting_bytes,
            &layout.values_offset
        ) ||
        !evoke_checked_add_size(
            layout.values_offset,
            posting_bytes,
            &layout.semantic_bmp_offset
        ) ||
        !evoke_checked_add_size(
            layout.semantic_bmp_offset,
            semantic_bmp_size,
            &layout.total_size
        ))
    {
        return EVOKE_ERR_RANGE;
    }
    layout.semantic_bmp_size = semantic_bmp_size;
    *layout_out = layout;
    return EVOKE_OK;
}

static evoke_status
evoke_term_fold_generic_posting_count(
    const evoke_term_fold_bundle *bundle,
    uint64_t *count_out
)
{
    uint64_t count = 0;

    if (bundle == NULL || count_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint32_t run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle->runs[run_index];

        if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            continue;
        }
        if (count > UINT64_MAX - run->posting_count)
        {
            return EVOKE_ERR_RANGE;
        }
        count += run->posting_count;
    }
    *count_out = count;
    return EVOKE_OK;
}

static evoke_status
evoke_term_fold_semantic_bmp_build(
    const evoke_term_fold_bundle *bundle,
    evoke_semantic_bmp_packed_index *index_out,
    size_t *size_out
)
{
    evoke_semantic_bmp_run *runs = NULL;
    uint32_t document_count = 0;
    size_t semantic_run_count = 0;
    evoke_status status = EVOKE_OK;

    if (bundle == NULL || index_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *size_out = 0;
    for (uint32_t run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle->runs[run_index];

        if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            continue;
        }
        semantic_run_count++;
        if (run->posting_count > 0)
        {
            uint32_t last_document = bundle->document_slots[
                run->posting_offset + run->posting_count - 1
            ];

            if (last_document == UINT32_MAX)
            {
                return EVOKE_ERR_RANGE;
            }
            if (document_count < last_document + UINT32_C(1))
            {
                document_count = last_document + UINT32_C(1);
            }
        }
    }
    if (semantic_run_count == 0)
    {
        return EVOKE_OK;
    }
    if (document_count == 0 ||
        semantic_run_count > SIZE_MAX / sizeof(*runs))
    {
        return EVOKE_ERR_RANGE;
    }
    runs = calloc(semantic_run_count, sizeof(*runs));
    if (runs == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    semantic_run_count = 0;
    for (uint32_t run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle->runs[run_index];
        evoke_semantic_bmp_run *target;

        if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            continue;
        }
        target = &runs[semantic_run_count++];
        target->term_id = run->term_id;
        target->posting_count = run->posting_count;
        target->local_document_ids =
            &bundle->document_slots[run->posting_offset];
        target->values = &bundle->values[run->posting_offset];
        target->local_document_count = document_count;
    }
    status = evoke_semantic_bmp_packed_index_build_runs_with_precision(
        document_count,
        runs,
        semantic_run_count,
        bundle->semantic_impact_precision,
        index_out
    );
    if (status == EVOKE_OK)
    {
        status = evoke_semantic_bmp_packed_serialized_size(
            index_out,
            size_out
        );
    }
    free(runs);
    return status;
}

void
evoke_term_fold_bundle_init(evoke_term_fold_bundle *bundle)
{
    if (bundle != NULL)
    {
        memset(bundle, 0, sizeof(*bundle));
        bundle->semantic_impact_precision =
            EVOKE_SEMANTIC_IMPACT_PRECISION_F32;
    }
}

void
evoke_term_fold_bundle_free(evoke_term_fold_bundle *bundle)
{
    if (bundle == NULL)
    {
        return;
    }
    free(bundle->runs);
    free(bundle->blocks);
    free(bundle->document_slots);
    free(bundle->values);
    evoke_term_fold_bundle_init(bundle);
}

evoke_status
evoke_term_fold_bundle_validate(const evoke_term_fold_bundle *bundle)
{
    bool has_blocks;
    uint64_t block_cursor = 0;
    uint64_t posting_cursor = 0;

    if (bundle == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if ((bundle->object_kind != EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD &&
         bundle->object_kind != EVOKE_SEGMENT_OBJECT_IMPACT_FOLD) ||
        bundle->owner_manifest_id == 0 ||
        bundle->run_count == 0 ||
        bundle->posting_count == 0 ||
        evoke_semantic_bmp_impact_width(
            bundle->semantic_impact_precision
        ) == 0 ||
        bundle->runs == NULL ||
        bundle->document_slots == NULL ||
        bundle->values == NULL ||
        (bundle->object_kind == EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD &&
         bundle->statistics_epoch != 0) ||
        (bundle->object_kind == EVOKE_SEGMENT_OBJECT_IMPACT_FOLD &&
         bundle->statistics_epoch == 0))
    {
        return EVOKE_ERR_FORMAT;
    }

    for (uint32_t run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle->runs[run_index];
        uint64_t posting_end;

        if (run->coverage_sequence == 0 ||
            run->posting_count == 0 ||
            run->posting_offset != posting_cursor ||
            run->posting_offset > bundle->posting_count ||
            run->posting_count >
                bundle->posting_count - run->posting_offset ||
            (bundle->object_kind ==
                 EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD &&
             run->kind != EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
             run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT) ||
            (bundle->object_kind ==
                 EVOKE_SEGMENT_OBJECT_IMPACT_FOLD &&
             run->kind != EVOKE_POSTING_EXTENT_LEXICAL_IMPACT))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (run_index > 0)
        {
            const evoke_term_fold_run *previous =
                &bundle->runs[run_index - 1];

            if (previous->term_id > run->term_id ||
                (previous->term_id == run->term_id &&
                 previous->kind >= run->kind) ||
                (previous->term_id == run->term_id &&
                 previous->coverage_sequence !=
                    run->coverage_sequence))
            {
                return EVOKE_ERR_FORMAT;
            }
        }

        posting_end = run->posting_offset + run->posting_count;
        for (uint64_t posting_index = run->posting_offset;
             posting_index < posting_end;
             posting_index++)
        {
            if ((posting_index > run->posting_offset &&
                 bundle->document_slots[posting_index - 1] >=
                    bundle->document_slots[posting_index]) ||
                (run->kind ==
                     EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                 bundle->values[posting_index].term_frequency == 0) ||
                (run->kind !=
                     EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                 !isfinite(bundle->values[posting_index].impact)))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        posting_cursor = posting_end;
    }
    if (posting_cursor != bundle->posting_count)
    {
        return EVOKE_ERR_FORMAT;
    }

    has_blocks = bundle->block_shift != 0 ||
        bundle->block_count != 0 || bundle->blocks != NULL;
    if (!has_blocks)
    {
        for (uint32_t run_index = 0;
             run_index < bundle->run_count;
             run_index++)
        {
            if (bundle->runs[run_index].block_offset != 0 ||
                bundle->runs[run_index].block_count != 0)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        return EVOKE_OK;
    }
    if (bundle->block_shift == 0 ||
        bundle->block_shift >= 32 ||
        bundle->block_count == 0 ||
        bundle->blocks == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle->runs[run_index];
        evoke_posting_extent extent;
        uint32_t last_document_id;

        if (run->block_offset != block_cursor ||
            run->block_count == 0 ||
            run->block_offset > bundle->block_count ||
            run->block_count >
                bundle->block_count - run->block_offset)
        {
            return EVOKE_ERR_FORMAT;
        }
        last_document_id = bundle->document_slots[
            run->posting_offset + run->posting_count - 1
        ];
        if (last_document_id == UINT32_MAX)
        {
            return EVOKE_ERR_FORMAT;
        }
        memset(&extent, 0, sizeof(extent));
        extent.indices =
            &bundle->document_slots[run->posting_offset];
        extent.values = &bundle->values[run->posting_offset];
        extent.len = run->posting_count;
        extent.local_document_count = last_document_id + 1;
        extent.kind = run->kind;
        if (evoke_posting_extent_validate_block_records(
                &extent,
                bundle->block_shift,
                &bundle->blocks[run->block_offset],
                run->block_count) != EVOKE_OK)
        {
            return EVOKE_ERR_FORMAT;
        }
        block_cursor += run->block_count;
    }
    return block_cursor == bundle->block_count
        ? EVOKE_OK
        : EVOKE_ERR_FORMAT;
}

static void
evoke_serialize_posting_block_record(
    uint8_t *bytes,
    const evoke_posting_block_record *record
)
{
    uint32_t min_impact_bits;
    uint32_t max_impact_bits;

    memcpy(
        &min_impact_bits,
        &record->min_impact,
        sizeof(min_impact_bits)
    );
    memcpy(
        &max_impact_bits,
        &record->max_impact,
        sizeof(max_impact_bits)
    );
    evoke_write_u64_le(bytes + 0, record->posting_offset);
    evoke_write_u32_le(bytes + 8, record->posting_count);
    evoke_write_u32_le(bytes + 12, record->block_id);
    evoke_write_u32_le(bytes + 16, record->first_document_id);
    evoke_write_u32_le(bytes + 20, record->last_document_id);
    evoke_write_u32_le(bytes + 24, record->min_term_frequency);
    evoke_write_u32_le(bytes + 28, record->max_term_frequency);
    evoke_write_u32_le(bytes + 32, min_impact_bits);
    evoke_write_u32_le(bytes + 36, max_impact_bits);
    evoke_write_u32_le(bytes + 40, (uint32_t) record->kind);
    evoke_write_u32_le(bytes + 44, 0);
}

evoke_status
evoke_posting_block_record_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_posting_block_record *record
)
{
    uint32_t min_impact_bits;
    uint32_t max_impact_bits;

    if (bytes == NULL || record == NULL ||
        size < EVOKE_POSTING_BLOCK_RECORD_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_read_u32_le(bytes + 44) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(record, 0, sizeof(*record));
    record->posting_offset = evoke_read_u64_le(bytes + 0);
    record->posting_count = evoke_read_u32_le(bytes + 8);
    record->block_id = evoke_read_u32_le(bytes + 12);
    record->first_document_id = evoke_read_u32_le(bytes + 16);
    record->last_document_id = evoke_read_u32_le(bytes + 20);
    record->min_term_frequency = evoke_read_u32_le(bytes + 24);
    record->max_term_frequency = evoke_read_u32_le(bytes + 28);
    min_impact_bits = evoke_read_u32_le(bytes + 32);
    max_impact_bits = evoke_read_u32_le(bytes + 36);
    memcpy(
        &record->min_impact,
        &min_impact_bits,
        sizeof(record->min_impact)
    );
    memcpy(
        &record->max_impact,
        &max_impact_bits,
        sizeof(record->max_impact)
    );
    record->kind =
        (evoke_posting_extent_kind) evoke_read_u32_le(bytes + 40);
    return EVOKE_OK;
}

evoke_status
evoke_term_fold_bundle_serialize(
    const evoke_term_fold_bundle *bundle,
    uint8_t **bytes_out,
    size_t *size_out,
    uint64_t *checksum_out
)
{
    evoke_persistent_block_directory directory;
    evoke_semantic_bmp_packed_index semantic_bmp;
    evoke_term_fold_layout layout;
    size_t semantic_bmp_size = 0;
    uint64_t generic_posting_count = 0;
    uint64_t generic_cursor = 0;
    uint64_t semantic_cursor = 0;
    uint8_t *bytes;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL || checksum_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    *checksum_out = 0;
    evoke_persistent_block_directory_init(&directory);
    evoke_semantic_bmp_packed_index_init(&semantic_bmp);
    status = evoke_term_fold_bundle_validate(bundle);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_term_fold_generic_posting_count(
        bundle,
        &generic_posting_count
    );
    if (status == EVOKE_OK)
    {
        status = evoke_term_fold_semantic_bmp_build(
        bundle,
        &semantic_bmp,
        &semantic_bmp_size
        );
    }
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_term_fold_block_directory_build(
        bundle,
        false,
        &directory
    );
    if (status != EVOKE_OK)
    {
        evoke_semantic_bmp_packed_index_free(&semantic_bmp);
        return status;
    }
    status = evoke_term_fold_layout_build(
        bundle->run_count,
        directory.block_count,
        generic_posting_count,
        semantic_bmp_size,
        &layout
    );
    if (status != EVOKE_OK)
    {
        evoke_persistent_block_directory_free(&directory);
        evoke_semantic_bmp_packed_index_free(&semantic_bmp);
        return status;
    }
    bytes = calloc(layout.total_size, 1);
    if (bytes == NULL)
    {
        evoke_persistent_block_directory_free(&directory);
        evoke_semantic_bmp_packed_index_free(&semantic_bmp);
        return EVOKE_ERR_NOMEM;
    }

    evoke_write_u32_le(bytes + 0, EVOKE_TERM_FOLD_MAGIC);
    evoke_write_u16_le(bytes + 4, EVOKE_TERM_FOLD_VERSION);
    evoke_write_u16_le(bytes + 6, EVOKE_TERM_FOLD_HEADER_SIZE);
    evoke_write_u16_le(bytes + 8, EVOKE_TERM_FOLD_RUN_SIZE);
    evoke_write_u16_le(bytes + 10, EVOKE_POSTING_BLOCK_RECORD_SIZE);
    evoke_write_u32_le(bytes + 12, (uint32_t) bundle->object_kind);
    evoke_write_u32_le(bytes + 16, bundle->run_count);
    evoke_write_u32_le(bytes + 20, directory.block_count);
    evoke_write_u64_le(bytes + 24, bundle->owner_manifest_id);
    evoke_write_u64_le(bytes + 32, bundle->statistics_epoch);
    evoke_write_u64_le(bytes + 40, bundle->posting_count);
    evoke_write_u64_le(bytes + 48, layout.runs_offset);
    evoke_write_u64_le(bytes + 56, layout.blocks_offset);
    evoke_write_u64_le(bytes + 64, layout.document_slots_offset);
    evoke_write_u64_le(bytes + 72, layout.values_offset);
    evoke_write_u64_le(bytes + 80, layout.total_size);
    evoke_write_u32_le(bytes + 88, directory.block_shift);
    evoke_write_u32_le(
        bytes + 92,
        semantic_bmp_size == 0
            ? 0
            : EVOKE_SEMANTIC_BMP_PACKED_FORMAT_VERSION
    );
    evoke_write_u64_le(bytes + 104, layout.semantic_bmp_offset);
    evoke_write_u64_le(bytes + 112, semantic_bmp_size);
    evoke_write_u64_le(bytes + 120, generic_posting_count);

    for (uint32_t run_index = 0;
         run_index < bundle->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle->runs[run_index];
        uint8_t *run_bytes =
            bytes + layout.runs_offset +
            (size_t) run_index * EVOKE_TERM_FOLD_RUN_SIZE;
        uint64_t physical_offset =
            run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
                ? semantic_cursor
                : generic_cursor;

        evoke_write_u32_le(run_bytes + 0, run->term_id);
        evoke_write_u32_le(run_bytes + 4, (uint32_t) run->kind);
        evoke_write_u64_le(run_bytes + 8, run->coverage_sequence);
        evoke_write_u64_le(run_bytes + 16, physical_offset);
        evoke_write_u64_le(run_bytes + 24, run->posting_count);
        evoke_write_u64_le(
            run_bytes + 32,
            directory.run_block_offsets[run_index]
        );
        evoke_write_u32_le(
            run_bytes + 40,
            directory.run_block_counts[run_index]
        );
        evoke_write_u32_le(run_bytes + 44, 0);
        if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            semantic_cursor += run->posting_count;
            continue;
        }
        for (uint64_t posting_index = 0;
             posting_index < run->posting_count;
             posting_index++)
        {
            uint32_t value_bits;
            uint64_t source_index = run->posting_offset + posting_index;
            uint64_t target_index = physical_offset + posting_index;

            if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                value_bits =
                    bundle->values[source_index].term_frequency;
            }
            else
            {
                memcpy(
                    &value_bits,
                    &bundle->values[source_index].impact,
                    sizeof(value_bits)
                );
            }
            evoke_write_u32_le(
                bytes + layout.document_slots_offset +
                    (size_t) target_index * sizeof(uint32_t),
                bundle->document_slots[source_index]
            );
            evoke_write_u32_le(
                bytes + layout.values_offset +
                    (size_t) target_index * sizeof(uint32_t),
                value_bits
            );
        }
        generic_cursor += run->posting_count;
    }
    if (generic_cursor != generic_posting_count ||
        semantic_cursor != bundle->posting_count - generic_posting_count)
    {
        free(bytes);
        evoke_persistent_block_directory_free(&directory);
        evoke_semantic_bmp_packed_index_free(&semantic_bmp);
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t block_index = 0;
         block_index < directory.block_count;
         block_index++)
    {
        evoke_serialize_posting_block_record(
            bytes + layout.blocks_offset +
                (size_t) block_index *
                    EVOKE_POSTING_BLOCK_RECORD_SIZE,
            &directory.blocks[block_index]
        );
    }
    if (semantic_bmp_size > 0)
    {
        status = evoke_semantic_bmp_packed_serialize_into(
            &semantic_bmp,
            bytes + layout.semantic_bmp_offset,
            semantic_bmp_size
        );
        if (status != EVOKE_OK)
        {
            free(bytes);
            evoke_persistent_block_directory_free(&directory);
            evoke_semantic_bmp_packed_index_free(&semantic_bmp);
            return status;
        }
    }

    evoke_write_u64_le(
        bytes + EVOKE_TERM_FOLD_CHECKSUM_OFFSET,
        evoke_checksum_with_zero_range(
            bytes,
            layout.total_size,
            EVOKE_TERM_FOLD_CHECKSUM_OFFSET,
            sizeof(uint64_t)
        )
    );
    *bytes_out = bytes;
    *size_out = layout.total_size;
    *checksum_out = evoke_segment_blob_checksum(bytes, layout.total_size);
    evoke_persistent_block_directory_free(&directory);
    evoke_semantic_bmp_packed_index_free(&semantic_bmp);
    return EVOKE_OK;
}

typedef struct evoke_term_fold_build_posting
{
    uint32_t document_slot;
    evoke_posting_value value;
} evoke_term_fold_build_posting;

static int
evoke_compare_term_fold_build_posting(
    const void *left_pointer,
    const void *right_pointer
)
{
    const evoke_term_fold_build_posting *left = left_pointer;
    const evoke_term_fold_build_posting *right = right_pointer;

    return (left->document_slot > right->document_slot) -
        (left->document_slot < right->document_slot);
}

static bool
evoke_segment_payload_view_has_run(
    const evoke_segment_payload_view *payload,
    uint32_t term_id,
    const evoke_term_extent_descriptor *extent
)
{
    for (uint32_t run_index = 0;
         run_index < payload->run_count;
         run_index++)
    {
        const evoke_segment_term_run *run = &payload->runs[run_index];

        if (run->term_id == term_id &&
            run->kind == extent->kind &&
            run->posting_offset == extent->posting_offset &&
            run->posting_count == extent->posting_count)
        {
            return true;
        }
    }
    return false;
}

static const evoke_segment_payload_view *
evoke_term_fold_find_payload(
    const evoke_segment_payload_view *payloads,
    const uint32_t *payload_segment_indices,
    size_t payload_count,
    uint32_t segment_index
)
{
    for (size_t payload_index = 0;
         payload_index < payload_count;
         payload_index++)
    {
        if (payload_segment_indices[payload_index] == segment_index)
        {
            return &payloads[payload_index];
        }
    }
    return NULL;
}

evoke_status
evoke_term_fold_bundle_advance_neutral_extents(
    const evoke_segment_manifest *manifest,
    uint32_t term_id,
    const evoke_posting_extent *tail_extents,
    size_t tail_extent_count,
    const evoke_term_fold_bundle *prior_bundle,
    uint64_t prior_coverage_sequence,
    bool prior_coverage_is_fold_boundary,
    uint64_t owner_manifest_id,
    uint64_t coverage_sequence,
    evoke_term_fold_bundle *bundle_out
)
{
    evoke_term_fold_bundle bundle;
    evoke_term_fold_build_posting *postings = NULL;
    uint64_t posting_counts[EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT + 1] =
        {0};
    uint64_t posting_offsets[
        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT + 1] = {0};
    uint64_t kind_cursors[
        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT + 1] = {0};
    uint64_t posting_cursor = 0;
    uint32_t output_run_index = 0;
    uint32_t run_count = 0;
    bool is_manifest_boundary = false;
    bool is_prior_boundary = prior_coverage_sequence == 0 ||
        prior_bundle != NULL || prior_coverage_is_fold_boundary;
    bool prior_precedes_manifest = prior_coverage_sequence != 0;
    evoke_status status = EVOKE_OK;

    if (manifest == NULL || bundle_out == NULL ||
        (tail_extent_count > 0 && tail_extents == NULL) ||
        tail_extent_count == 0 ||
        term_id >= manifest->vocab_size ||
        owner_manifest_id < manifest->manifest_id ||
        coverage_sequence == 0 ||
        coverage_sequence > manifest->max_sequence ||
        coverage_sequence <= prior_coverage_sequence ||
        (prior_bundle != NULL && prior_coverage_sequence == 0) ||
        (prior_coverage_is_fold_boundary &&
         (prior_bundle != NULL || prior_coverage_sequence == 0)))
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_manifest_validate(manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    for (uint32_t segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        if (manifest->segments[segment_index].max_sequence ==
            coverage_sequence)
        {
            is_manifest_boundary = true;
        }
        if (manifest->segments[segment_index].max_sequence ==
            prior_coverage_sequence)
        {
            is_prior_boundary = true;
        }
        if (manifest->segments[segment_index].min_sequence <=
            prior_coverage_sequence)
        {
            prior_precedes_manifest = false;
        }
    }
    if (!is_prior_boundary && prior_precedes_manifest)
    {
        /* The prior coverage belongs to an inherited fold-only prefix. */
        is_prior_boundary = true;
    }
    if (!is_manifest_boundary || !is_prior_boundary)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (prior_bundle != NULL)
    {
        bool found_prior_run = false;

        status = evoke_term_fold_bundle_validate(prior_bundle);
        if (status != EVOKE_OK ||
            prior_bundle->object_kind !=
                EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD ||
            prior_bundle->owner_manifest_id > manifest->manifest_id)
        {
            return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        }
        for (uint32_t run_index = 0;
             run_index < prior_bundle->run_count;
             run_index++)
        {
            const evoke_term_fold_run *run =
                &prior_bundle->runs[run_index];

            if (run->term_id != term_id)
            {
                continue;
            }
            if (run->coverage_sequence !=
                    prior_coverage_sequence ||
                posting_counts[run->kind] >
                    UINT64_MAX - run->posting_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            posting_counts[run->kind] += run->posting_count;
            found_prior_run = true;
        }
        if (!found_prior_run)
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    for (size_t extent_index = 0;
         extent_index < tail_extent_count;
         extent_index++)
    {
        const evoke_posting_extent *extent =
            &tail_extents[extent_index];

        if (extent->kind < EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL ||
            extent->kind > EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
            extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT ||
            extent->len == 0 || extent->indices == NULL ||
            extent->values == NULL || extent->local_document_count == 0 ||
            (extent->document_id_map == NULL &&
             ((uint64_t) extent->document_id_base +
              extent->local_document_count >
              manifest->document_slot_count)) ||
            posting_counts[extent->kind] > UINT64_MAX - extent->len)
        {
            return EVOKE_ERR_FORMAT;
        }
        posting_counts[extent->kind] += extent->len;
    }

    for (evoke_posting_extent_kind kind =
             EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
         kind <= EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
         kind++)
    {
        if (posting_counts[kind] == 0)
        {
            continue;
        }
        posting_offsets[kind] = posting_cursor;
        if (posting_cursor > UINT64_MAX - posting_counts[kind])
        {
            return EVOKE_ERR_RANGE;
        }
        posting_cursor += posting_counts[kind];
        run_count++;
    }
    if (posting_cursor == 0 ||
        posting_cursor > SIZE_MAX / sizeof(*postings) ||
        posting_cursor > SIZE_MAX / sizeof(*bundle.document_slots) ||
        posting_cursor > SIZE_MAX / sizeof(*bundle.values))
    {
        return posting_cursor == 0
            ? EVOKE_ERR_FORMAT
            : EVOKE_ERR_RANGE;
    }

    evoke_term_fold_bundle_init(&bundle);
    bundle.object_kind = EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    bundle.owner_manifest_id = owner_manifest_id;
    bundle.run_count = run_count;
    bundle.posting_count = posting_cursor;
    bundle.runs = calloc(run_count, sizeof(*bundle.runs));
    bundle.document_slots = calloc(
        (size_t) posting_cursor,
        sizeof(*bundle.document_slots)
    );
    bundle.values = calloc(
        (size_t) posting_cursor,
        sizeof(*bundle.values)
    );
    postings = calloc((size_t) posting_cursor, sizeof(*postings));
    if (bundle.runs == NULL || bundle.document_slots == NULL ||
        bundle.values == NULL || postings == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto fail;
    }
    memcpy(kind_cursors, posting_offsets, sizeof(kind_cursors));

    if (prior_bundle != NULL)
    {
        for (uint32_t run_index = 0;
             run_index < prior_bundle->run_count;
             run_index++)
        {
            const evoke_term_fold_run *run =
                &prior_bundle->runs[run_index];
            uint64_t *kind_cursor;

            if (run->term_id != term_id)
            {
                continue;
            }
            kind_cursor = &kind_cursors[run->kind];
            for (uint64_t source_index = run->posting_offset;
                 source_index <
                    run->posting_offset + run->posting_count;
                 source_index++)
            {
                if (*kind_cursor >=
                        posting_offsets[run->kind] +
                            posting_counts[run->kind] ||
                    prior_bundle->document_slots[source_index] >=
                        manifest->document_slot_count)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto fail;
                }
                postings[*kind_cursor].document_slot =
                    prior_bundle->document_slots[source_index];
                postings[*kind_cursor].value =
                    prior_bundle->values[source_index];
                (*kind_cursor)++;
            }
        }
    }

    for (size_t extent_index = 0;
         extent_index < tail_extent_count;
         extent_index++)
    {
        const evoke_posting_extent *extent =
            &tail_extents[extent_index];
        for (uint64_t source_index = 0;
             source_index < extent->len;
             source_index++)
        {
            uint32_t local_document_id = extent->indices[source_index];
            uint32_t document_slot;
            uint64_t *kind_cursor = &kind_cursors[extent->kind];

            if (local_document_id >= extent->local_document_count)
            {
                status = EVOKE_ERR_RANGE;
                goto fail;
            }
            document_slot = extent->document_id_map == NULL
                ? extent->document_id_base + local_document_id
                : extent->document_id_map[local_document_id];
            if (*kind_cursor >=
                    posting_offsets[extent->kind] +
                        posting_counts[extent->kind] ||
                document_slot >= manifest->document_slot_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto fail;
            }
            postings[*kind_cursor].document_slot = document_slot;
            postings[*kind_cursor].value =
                extent->values[source_index];
            (*kind_cursor)++;
        }
    }

    posting_cursor = 0;
    for (evoke_posting_extent_kind kind =
             EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
         kind <= EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
         kind++)
    {
        uint64_t run_offset;

        if (posting_counts[kind] == 0)
        {
            continue;
        }
        if (kind_cursors[kind] !=
                posting_offsets[kind] + posting_counts[kind])
        {
            status = EVOKE_ERR_FORMAT;
            goto fail;
        }
        run_offset = posting_cursor;
        qsort(
            &postings[run_offset],
            (size_t) posting_counts[kind],
            sizeof(*postings),
            evoke_compare_term_fold_build_posting
        );
        for (uint64_t kind_index = 0;
             kind_index < posting_counts[kind];
             kind_index++)
        {
            uint64_t source_index = run_offset + kind_index;

            if (kind_index > 0 &&
                postings[source_index - 1].document_slot >=
                    postings[source_index].document_slot)
            {
                status = EVOKE_ERR_FORMAT;
                goto fail;
            }
            bundle.document_slots[source_index] =
                postings[source_index].document_slot;
            bundle.values[source_index] = postings[source_index].value;
        }
        bundle.runs[output_run_index].term_id = term_id;
        bundle.runs[output_run_index].kind = kind;
        bundle.runs[output_run_index].coverage_sequence =
            coverage_sequence;
        bundle.runs[output_run_index].posting_offset = run_offset;
        bundle.runs[output_run_index].posting_count =
            posting_counts[kind];
        output_run_index++;
        posting_cursor += posting_counts[kind];
    }
    if (output_run_index != bundle.run_count ||
        posting_cursor != bundle.posting_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto fail;
    }
    free(postings);
    status = evoke_term_fold_bundle_validate(&bundle);
    if (status != EVOKE_OK)
    {
        evoke_term_fold_bundle_free(&bundle);
        return status;
    }
    evoke_term_fold_bundle_free(bundle_out);
    *bundle_out = bundle;
    return EVOKE_OK;

fail:
    free(postings);
    evoke_term_fold_bundle_free(&bundle);
    return status;
}

evoke_status
evoke_term_fold_bundle_advance_neutral(
    const evoke_segment_manifest *manifest,
    uint32_t term_id,
    const evoke_term_extent_descriptor *tail_extents,
    size_t tail_extent_count,
    const evoke_segment_payload_view *payloads,
    const uint32_t *payload_segment_indices,
    size_t payload_count,
    const evoke_term_fold_bundle *prior_bundle,
    uint64_t prior_coverage_sequence,
    uint64_t owner_manifest_id,
    uint64_t coverage_sequence,
    evoke_term_fold_bundle *bundle_out
)
{
    evoke_posting_extent *term_extents = NULL;
    size_t term_extent_count = 0;
    evoke_status status;

    if (manifest == NULL || bundle_out == NULL ||
        (tail_extent_count > 0 && tail_extents == NULL) ||
        (payload_count > 0 &&
         (payloads == NULL || payload_segment_indices == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_manifest_validate(manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    for (size_t payload_index = 0;
         payload_index < payload_count;
         payload_index++)
    {
        uint32_t segment_index = payload_segment_indices[payload_index];

        if (segment_index >= manifest->segment_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (size_t prior_index = 0;
             prior_index < payload_index;
             prior_index++)
        {
            if (payload_segment_indices[prior_index] == segment_index)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        status = evoke_segment_posting_payload_validate(
            manifest,
            &manifest->segments[segment_index],
            &payloads[payload_index]
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    if (tail_extent_count > 0)
    {
        term_extents = calloc(tail_extent_count, sizeof(*term_extents));
        if (term_extents == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
    }
    for (size_t extent_index = 0;
         extent_index < tail_extent_count;
         extent_index++)
    {
        const evoke_term_extent_descriptor *extent =
            &tail_extents[extent_index];
        const evoke_segment_descriptor *segment;
        const evoke_segment_payload_view *payload;
        evoke_posting_extent *term_extent;

        if (extent->segment_index >= manifest->segment_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        segment = &manifest->segments[extent->segment_index];
        if (segment->max_sequence <= prior_coverage_sequence ||
            (segment->max_sequence > coverage_sequence &&
             segment->min_sequence <= coverage_sequence))
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        if (segment->max_sequence > coverage_sequence)
        {
            continue;
        }
        if (extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        payload = evoke_term_fold_find_payload(
            payloads,
            payload_segment_indices,
            payload_count,
            extent->segment_index
        );
        if (payload == NULL ||
            !evoke_segment_payload_view_has_run(payload, term_id, extent))
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        term_extent = &term_extents[term_extent_count++];
        term_extent->indices =
            &payload->indices[extent->posting_offset];
        term_extent->values =
            &payload->values[extent->posting_offset];
        term_extent->document_id_map = payload->document_id_map;
        term_extent->len = extent->posting_count;
        term_extent->document_id_base = payload->document_id_base;
        term_extent->local_document_count =
            payload->local_document_count;
        term_extent->kind = extent->kind;
    }
    status = evoke_term_fold_bundle_advance_neutral_extents(
        manifest,
        term_id,
        term_extents,
        term_extent_count,
        prior_bundle,
        prior_coverage_sequence,
        false,
        owner_manifest_id,
        coverage_sequence,
        bundle_out
    );

done:
    free(term_extents);
    return status;
}

evoke_status
evoke_term_fold_bundle_merge_neutral(
    const evoke_segment_manifest *manifest,
    uint32_t term_id,
    const evoke_term_fold_bundle *major_bundle,
    const evoke_term_fold_bundle *minor_bundle,
    uint64_t owner_manifest_id,
    evoke_term_fold_bundle *bundle_out
)
{
    const evoke_term_fold_bundle *sources[2] = {
        major_bundle,
        minor_bundle
    };
    evoke_term_fold_bundle bundle;
    evoke_term_fold_build_posting *postings = NULL;
    uint64_t posting_counts[EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT + 1] =
        {0};
    uint64_t posting_offsets[
        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT + 1] = {0};
    uint64_t kind_cursors[
        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT + 1] = {0};
    uint64_t coverages[2] = {0};
    uint64_t posting_cursor = 0;
    uint32_t output_run_index = 0;
    uint32_t run_count = 0;
    evoke_status status;

    if (manifest == NULL || major_bundle == NULL ||
        minor_bundle == NULL || bundle_out == NULL ||
        term_id >= manifest->vocab_size ||
        owner_manifest_id < manifest->manifest_id)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_manifest_validate(manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    for (size_t source_index = 0; source_index < 2; source_index++)
    {
        const evoke_term_fold_bundle *source = sources[source_index];
        bool found_term = false;

        status = evoke_term_fold_bundle_validate(source);
        if (status != EVOKE_OK ||
            source->object_kind != EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD ||
            source->owner_manifest_id > manifest->manifest_id)
        {
            return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        }
        for (uint32_t run_index = 0;
             run_index < source->run_count;
             run_index++)
        {
            const evoke_term_fold_run *run = &source->runs[run_index];

            if (run->term_id != term_id)
            {
                continue;
            }
            if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT ||
                (found_term &&
                 coverages[source_index] != run->coverage_sequence) ||
                posting_counts[run->kind] >
                    UINT64_MAX - run->posting_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            coverages[source_index] = run->coverage_sequence;
            posting_counts[run->kind] += run->posting_count;
            found_term = true;
        }
        if (!found_term)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    if (coverages[0] == 0 || coverages[1] <= coverages[0] ||
        coverages[1] > manifest->max_sequence)
    {
        return EVOKE_ERR_FORMAT;
    }

    for (evoke_posting_extent_kind kind =
             EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
         kind <= EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
         kind++)
    {
        if (posting_counts[kind] == 0)
        {
            continue;
        }
        posting_offsets[kind] = posting_cursor;
        if (posting_cursor > UINT64_MAX - posting_counts[kind])
        {
            return EVOKE_ERR_RANGE;
        }
        posting_cursor += posting_counts[kind];
        run_count++;
    }
    if (posting_cursor == 0 ||
        posting_cursor > SIZE_MAX / sizeof(*postings) ||
        posting_cursor > SIZE_MAX / sizeof(*bundle.document_slots) ||
        posting_cursor > SIZE_MAX / sizeof(*bundle.values))
    {
        return posting_cursor == 0
            ? EVOKE_ERR_FORMAT
            : EVOKE_ERR_RANGE;
    }

    evoke_term_fold_bundle_init(&bundle);
    bundle.object_kind = EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    bundle.owner_manifest_id = owner_manifest_id;
    bundle.run_count = run_count;
    bundle.posting_count = posting_cursor;
    bundle.runs = calloc(run_count, sizeof(*bundle.runs));
    bundle.document_slots = calloc(
        (size_t) posting_cursor,
        sizeof(*bundle.document_slots)
    );
    bundle.values = calloc(
        (size_t) posting_cursor,
        sizeof(*bundle.values)
    );
    postings = calloc((size_t) posting_cursor, sizeof(*postings));
    if (bundle.runs == NULL || bundle.document_slots == NULL ||
        bundle.values == NULL || postings == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto fail;
    }
    memcpy(kind_cursors, posting_offsets, sizeof(kind_cursors));

    for (size_t source_index = 0; source_index < 2; source_index++)
    {
        const evoke_term_fold_bundle *source = sources[source_index];

        for (uint32_t run_index = 0;
             run_index < source->run_count;
             run_index++)
        {
            const evoke_term_fold_run *run = &source->runs[run_index];
            uint64_t *kind_cursor;

            if (run->term_id != term_id)
            {
                continue;
            }
            kind_cursor = &kind_cursors[run->kind];
            for (uint64_t posting_index = run->posting_offset;
                 posting_index <
                    run->posting_offset + run->posting_count;
                 posting_index++)
            {
                if (*kind_cursor >=
                        posting_offsets[run->kind] +
                            posting_counts[run->kind] ||
                    source->document_slots[posting_index] >=
                        manifest->document_slot_count)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto fail;
                }
                postings[*kind_cursor].document_slot =
                    source->document_slots[posting_index];
                postings[*kind_cursor].value =
                    source->values[posting_index];
                (*kind_cursor)++;
            }
        }
    }

    posting_cursor = 0;
    for (evoke_posting_extent_kind kind =
             EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
         kind <= EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
         kind++)
    {
        uint64_t run_offset;

        if (posting_counts[kind] == 0)
        {
            continue;
        }
        if (kind_cursors[kind] !=
                posting_offsets[kind] + posting_counts[kind])
        {
            status = EVOKE_ERR_FORMAT;
            goto fail;
        }
        run_offset = posting_cursor;
        qsort(
            &postings[run_offset],
            (size_t) posting_counts[kind],
            sizeof(*postings),
            evoke_compare_term_fold_build_posting
        );
        for (uint64_t kind_index = 0;
             kind_index < posting_counts[kind];
             kind_index++)
        {
            uint64_t source_index = run_offset + kind_index;

            if (kind_index > 0 &&
                postings[source_index - 1].document_slot >=
                    postings[source_index].document_slot)
            {
                status = EVOKE_ERR_FORMAT;
                goto fail;
            }
            bundle.document_slots[source_index] =
                postings[source_index].document_slot;
            bundle.values[source_index] = postings[source_index].value;
        }
        bundle.runs[output_run_index].term_id = term_id;
        bundle.runs[output_run_index].kind = kind;
        bundle.runs[output_run_index].coverage_sequence = coverages[1];
        bundle.runs[output_run_index].posting_offset = run_offset;
        bundle.runs[output_run_index].posting_count =
            posting_counts[kind];
        output_run_index++;
        posting_cursor += posting_counts[kind];
    }
    if (output_run_index != bundle.run_count ||
        posting_cursor != bundle.posting_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto fail;
    }
    free(postings);
    status = evoke_term_fold_bundle_validate(&bundle);
    if (status != EVOKE_OK)
    {
        evoke_term_fold_bundle_free(&bundle);
        return status;
    }
    evoke_term_fold_bundle_free(bundle_out);
    *bundle_out = bundle;
    return EVOKE_OK;

fail:
    free(postings);
    evoke_term_fold_bundle_free(&bundle);
    return status;
}

evoke_status
evoke_term_fold_bundle_build_impact(
    const evoke_index *global_index,
    const evoke_corpus_stats *stats,
    uint32_t term_id,
    uint32_t live_document_frequency,
    const evoke_term_fold_bundle *major_bundle,
    const evoke_term_fold_bundle *minor_bundle,
    uint64_t owner_manifest_id,
    uint64_t statistics_epoch,
    evoke_term_fold_bundle *bundle_out
)
{
    const evoke_term_fold_bundle *sources[2] = {
        major_bundle,
        minor_bundle
    };
    evoke_term_fold_bundle bundle;
    evoke_term_fold_build_posting *postings = NULL;
    uint64_t coverages[2] = {0};
    uint64_t posting_count = 0;
    size_t source_count = minor_bundle == NULL ? 1 : 2;
    double average_document_length;
    double idf;
    double nonoccurrence = 0.0;
    evoke_status status;

    if (global_index == NULL || stats == NULL ||
        major_bundle == NULL || bundle_out == NULL ||
        global_index->doc_lengths == NULL ||
        term_id >= global_index->vocab_size ||
        stats->document_count == 0 ||
        stats->total_document_length == 0 ||
        live_document_frequency == 0 ||
        (uint64_t) live_document_frequency > stats->document_count ||
        owner_manifest_id == 0 || statistics_epoch == 0)
    {
        return EVOKE_ERR_INVALID;
    }

    for (size_t source_index = 0;
         source_index < source_count;
         source_index++)
    {
        const evoke_term_fold_bundle *source = sources[source_index];
        bool found_term = false;

        status = evoke_term_fold_bundle_validate(source);
        if (status != EVOKE_OK ||
            source->object_kind != EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD ||
            source->owner_manifest_id >= owner_manifest_id)
        {
            return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        }
        for (uint32_t run_index = 0;
             run_index < source->run_count;
             run_index++)
        {
            const evoke_term_fold_run *run = &source->runs[run_index];

            if (run->term_id != term_id)
            {
                continue;
            }
            if ((found_term &&
                 coverages[source_index] != run->coverage_sequence) ||
                (run->kind ==
                     EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                 posting_count >
                    UINT64_MAX - run->posting_count))
            {
                return EVOKE_ERR_FORMAT;
            }
            coverages[source_index] = run->coverage_sequence;
            if (run->kind ==
                EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                posting_count += run->posting_count;
            }
            found_term = true;
        }
        if (!found_term || coverages[source_index] == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    if ((source_count == 2 && coverages[1] <= coverages[0]) ||
        posting_count == 0 ||
        posting_count > SIZE_MAX / sizeof(*postings) ||
        posting_count > SIZE_MAX / sizeof(*bundle.document_slots) ||
        posting_count > SIZE_MAX / sizeof(*bundle.values))
    {
        return posting_count == 0
            ? EVOKE_ERR_FORMAT
            : EVOKE_ERR_RANGE;
    }

    evoke_term_fold_bundle_init(&bundle);
    bundle.object_kind = EVOKE_SEGMENT_OBJECT_IMPACT_FOLD;
    bundle.owner_manifest_id = owner_manifest_id;
    bundle.statistics_epoch = statistics_epoch;
    bundle.run_count = 1;
    bundle.posting_count = posting_count;
    bundle.runs = calloc(1, sizeof(*bundle.runs));
    bundle.document_slots = calloc(
        (size_t) posting_count,
        sizeof(*bundle.document_slots)
    );
    bundle.values = calloc(
        (size_t) posting_count,
        sizeof(*bundle.values)
    );
    postings = calloc((size_t) posting_count, sizeof(*postings));
    if (bundle.runs == NULL || bundle.document_slots == NULL ||
        bundle.values == NULL || postings == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto fail;
    }

    {
        uint64_t posting_cursor = 0;

        for (size_t source_index = 0;
             source_index < source_count;
             source_index++)
        {
            const evoke_term_fold_bundle *source =
                sources[source_index];

            for (uint32_t run_index = 0;
                 run_index < source->run_count;
                 run_index++)
            {
                const evoke_term_fold_run *run =
                    &source->runs[run_index];

                if (run->term_id != term_id ||
                    run->kind !=
                        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    continue;
                }
                for (uint64_t posting_index = run->posting_offset;
                     posting_index <
                        run->posting_offset + run->posting_count;
                     posting_index++)
                {
                    if (posting_cursor >= posting_count ||
                        source->document_slots[posting_index] >=
                            global_index->num_docs)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto fail;
                    }
                    postings[posting_cursor].document_slot =
                        source->document_slots[posting_index];
                    postings[posting_cursor].value =
                        source->values[posting_index];
                    posting_cursor++;
                }
            }
        }
        if (posting_cursor != posting_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto fail;
        }
    }

    qsort(
        postings,
        (size_t) posting_count,
        sizeof(*postings),
        evoke_compare_term_fold_build_posting
    );
    average_document_length =
        (double) stats->total_document_length /
        (double) stats->document_count;
    idf = evoke_score_idf(
        global_index->params.idf_method,
        (double) live_document_frequency,
        (double) stats->document_count
    );
    if (evoke_method_requires_nonoccurrence(
            global_index->params.method))
    {
        nonoccurrence = idf * evoke_score_tfc(
            global_index->params.method,
            0.0,
            0.0,
            average_document_length,
            global_index->params.k1,
            global_index->params.b,
            global_index->params.delta
        );
    }
    for (uint64_t posting_index = 0;
         posting_index < posting_count;
         posting_index++)
    {
        uint32_t document_slot =
            postings[posting_index].document_slot;
        uint32_t term_frequency =
            postings[posting_index].value.term_frequency;
        double impact;

        if ((posting_index > 0 &&
             postings[posting_index - 1].document_slot >=
                document_slot) ||
            term_frequency == 0)
        {
            status = EVOKE_ERR_FORMAT;
            goto fail;
        }
        impact = idf * evoke_score_tfc(
            global_index->params.method,
            (double) term_frequency,
            (double) global_index->doc_lengths[document_slot],
            average_document_length,
            global_index->params.k1,
            global_index->params.b,
            global_index->params.delta
        ) - nonoccurrence;
        if (!isfinite(impact))
        {
            status = EVOKE_ERR_FORMAT;
            goto fail;
        }
        bundle.document_slots[posting_index] = document_slot;
        bundle.values[posting_index].impact = (float) impact;
    }
    bundle.runs[0].term_id = term_id;
    bundle.runs[0].kind = EVOKE_POSTING_EXTENT_LEXICAL_IMPACT;
    bundle.runs[0].coverage_sequence =
        source_count == 2 ? coverages[1] : coverages[0];
    bundle.runs[0].posting_count = posting_count;
    free(postings);
    status = evoke_term_fold_bundle_validate(&bundle);
    if (status != EVOKE_OK)
    {
        evoke_term_fold_bundle_free(&bundle);
        return status;
    }
    evoke_term_fold_bundle_free(bundle_out);
    *bundle_out = bundle;
    return EVOKE_OK;

fail:
    free(postings);
    evoke_term_fold_bundle_free(&bundle);
    return status;
}

evoke_status
evoke_term_fold_bundle_build_neutral_prefix(
    const evoke_segment_manifest *manifest,
    const evoke_term_directory *directory,
    const evoke_segment_payload_view *payloads,
    size_t payload_count,
    uint64_t owner_manifest_id,
    uint32_t term_id,
    uint64_t coverage_sequence,
    evoke_term_fold_bundle *bundle_out
)
{
    uint32_t *payload_segment_indices = NULL;
    const evoke_term_extent_descriptor *term_extents = NULL;
    uint64_t term_start;
    uint64_t term_end;
    evoke_status status;

    if (manifest == NULL || directory == NULL || bundle_out == NULL ||
        payload_count != manifest->segment_count ||
        (payload_count > 0 && payloads == NULL) ||
        term_id >= manifest->vocab_size ||
        owner_manifest_id < manifest->manifest_id ||
        coverage_sequence == 0 ||
        coverage_sequence > manifest->max_sequence)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_term_directory_validate(directory, manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (payload_count > SIZE_MAX / sizeof(*payload_segment_indices))
    {
        return EVOKE_ERR_RANGE;
    }
    payload_segment_indices = calloc(
        payload_count,
        sizeof(*payload_segment_indices)
    );
    if (payload_count > 0 && payload_segment_indices == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        payload_segment_indices[segment_index] = segment_index;
    }

    term_start = directory->term_offsets[term_id];
    term_end = directory->term_offsets[term_id + 1];
    if (term_end > term_start)
    {
        term_extents = &directory->extents[term_start];
    }
    status = evoke_term_fold_bundle_advance_neutral(
        manifest,
        term_id,
        term_extents,
        (size_t) (term_end - term_start),
        payloads,
        payload_segment_indices,
        payload_count,
        NULL,
        0,
        owner_manifest_id,
        coverage_sequence,
        bundle_out
    );
    free(payload_segment_indices);
    return status;
}

evoke_status
evoke_term_fold_disk_header_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_term_fold_disk_header *header_out
)
{
    evoke_term_fold_layout layout;
    evoke_term_fold_disk_header header;
    uint16_t format_version;
    evoke_status status;

    if (bytes == NULL || header_out == NULL ||
        size < EVOKE_TERM_FOLD_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    format_version = evoke_read_u16_le(bytes + 4);
    if (evoke_read_u32_le(bytes + 0) != EVOKE_TERM_FOLD_MAGIC ||
        format_version != EVOKE_TERM_FOLD_VERSION ||
        evoke_read_u16_le(bytes + 6) != EVOKE_TERM_FOLD_HEADER_SIZE ||
        evoke_read_u16_le(bytes + 8) != EVOKE_TERM_FOLD_RUN_SIZE ||
        evoke_read_u16_le(bytes + 10) !=
            EVOKE_POSTING_BLOCK_RECORD_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }

    memset(&header, 0, sizeof(header));
    header.format_version = format_version;
    header.object_kind =
        (evoke_segment_object_kind) evoke_read_u32_le(bytes + 12);
    header.run_count = evoke_read_u32_le(bytes + 16);
    header.block_count = evoke_read_u32_le(bytes + 20);
    header.owner_manifest_id = evoke_read_u64_le(bytes + 24);
    header.statistics_epoch = evoke_read_u64_le(bytes + 32);
    header.posting_count = evoke_read_u64_le(bytes + 40);
    header.generic_posting_count = evoke_read_u64_le(bytes + 120);
    header.runs_offset = evoke_read_u64_le(bytes + 48);
    header.blocks_offset = evoke_read_u64_le(bytes + 56);
    header.document_slots_offset = evoke_read_u64_le(bytes + 64);
    header.values_offset = evoke_read_u64_le(bytes + 72);
    header.total_size = evoke_read_u64_le(bytes + 80);
    header.block_shift = evoke_read_u32_le(bytes + 88);
    header.semantic_bmp_version = evoke_read_u32_le(bytes + 92);
    header.semantic_bmp_offset = evoke_read_u64_le(bytes + 104);
    header.semantic_bmp_size = evoke_read_u64_le(bytes + 112);
    header.internal_checksum = evoke_read_u64_le(
        bytes + EVOKE_TERM_FOLD_CHECKSUM_OFFSET
    );
    if ((header.object_kind != EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD &&
         header.object_kind != EVOKE_SEGMENT_OBJECT_IMPACT_FOLD) ||
        header.owner_manifest_id == 0 ||
        header.run_count == 0 ||
        header.block_shift == 0 || header.block_shift >= 32 ||
        header.posting_count == 0 || header.internal_checksum == 0 ||
        header.generic_posting_count > header.posting_count ||
        ((header.generic_posting_count == 0) !=
             (header.block_count == 0)) ||
        (header.object_kind == EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD &&
         header.statistics_epoch != 0) ||
        (header.object_kind == EVOKE_SEGMENT_OBJECT_IMPACT_FOLD &&
         header.statistics_epoch == 0) ||
        ((header.semantic_bmp_size == 0) !=
              (header.semantic_bmp_version == 0) ||
          (header.semantic_bmp_size > 0 &&
           header.semantic_bmp_version !=
               EVOKE_SEMANTIC_BMP_PACKED_FORMAT_VERSION) ||
          (header.semantic_bmp_size > 0 &&
           header.generic_posting_count == header.posting_count) ||
          (header.object_kind == EVOKE_SEGMENT_OBJECT_IMPACT_FOLD &&
           header.semantic_bmp_size != 0) ||
          header.semantic_bmp_size > SIZE_MAX ||
          (header.semantic_bmp_size == 0 &&
           header.generic_posting_count != header.posting_count)))
    {
        return EVOKE_ERR_FORMAT;
    }

    status = evoke_term_fold_layout_build(
        header.run_count,
        header.block_count,
        header.generic_posting_count,
        (size_t) header.semantic_bmp_size,
        &layout
    );
    if (status != EVOKE_OK ||
        header.runs_offset != layout.runs_offset ||
        header.blocks_offset != layout.blocks_offset ||
        header.document_slots_offset !=
            layout.document_slots_offset ||
        header.values_offset != layout.values_offset ||
        header.total_size != layout.total_size ||
        header.semantic_bmp_offset != layout.semantic_bmp_offset)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    *header_out = header;
    return EVOKE_OK;
}

evoke_status
evoke_term_fold_run_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_term_fold_run *run_out
)
{
    evoke_term_fold_run run;

    if (bytes == NULL || run_out == NULL ||
        size < EVOKE_TERM_FOLD_RUN_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_read_u32_le(bytes + 44) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&run, 0, sizeof(run));
    run.term_id = evoke_read_u32_le(bytes + 0);
    run.kind =
        (evoke_posting_extent_kind) evoke_read_u32_le(bytes + 4);
    run.coverage_sequence = evoke_read_u64_le(bytes + 8);
    run.posting_offset = evoke_read_u64_le(bytes + 16);
    run.posting_count = evoke_read_u64_le(bytes + 24);
    run.block_offset = evoke_read_u64_le(bytes + 32);
    run.block_count = evoke_read_u32_le(bytes + 40);
    if (run.kind < EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL ||
        run.kind > EVOKE_POSTING_EXTENT_LEXICAL_IMPACT ||
        run.coverage_sequence == 0 || run.posting_count == 0 ||
        (run.kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
            ? run.block_count != 0
            : run.block_count == 0) ||
        run.posting_offset > UINT64_MAX - run.posting_count ||
        run.block_offset > UINT64_MAX - run.block_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    *run_out = run;
    return EVOKE_OK;
}

evoke_status
evoke_term_fold_bundle_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_term_fold_bundle *bundle_out
)
{
    evoke_term_fold_bundle bundle;
    evoke_term_fold_disk_header header;
    evoke_term_fold_layout layout;
    evoke_persistent_block_directory directory;
    evoke_semantic_bmp_packed_index semantic_bmp;
    evoke_term_fold_run *disk_runs = NULL;
    evoke_posting_block_record *disk_blocks = NULL;
    uint64_t logical_cursor = 0;
    uint64_t generic_cursor = 0;
    uint64_t semantic_cursor = 0;
    uint32_t disk_block_cursor = 0;
    uint32_t semantic_run_index = 0;
    uint64_t checksum;
    evoke_status status;

    if (bytes == NULL || bundle_out == NULL ||
        size < EVOKE_TERM_FOLD_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_term_fold_disk_header_decode(bytes, size, &header);
    if (status != EVOKE_OK)
    {
        return status;
    }
    checksum = evoke_read_u64_le(bytes + EVOKE_TERM_FOLD_CHECKSUM_OFFSET);
    if (checksum == 0 ||
        checksum != evoke_checksum_with_zero_range(
            bytes,
            size,
            EVOKE_TERM_FOLD_CHECKSUM_OFFSET,
            sizeof(uint64_t)
        ))
    {
        return EVOKE_ERR_FORMAT;
    }

    evoke_term_fold_bundle_init(&bundle);
    evoke_persistent_block_directory_init(&directory);
    evoke_semantic_bmp_packed_index_init(&semantic_bmp);
    bundle.object_kind = header.object_kind;
    bundle.run_count = header.run_count;
    bundle.owner_manifest_id = header.owner_manifest_id;
    bundle.statistics_epoch = header.statistics_epoch;
    bundle.posting_count = header.posting_count;
    bundle.block_shift = header.block_shift;
    status = evoke_term_fold_layout_build(
        bundle.run_count,
        header.block_count,
        header.generic_posting_count,
        (size_t) header.semantic_bmp_size,
        &layout
    );
    if (status != EVOKE_OK ||
        header.runs_offset != layout.runs_offset ||
        header.blocks_offset != layout.blocks_offset ||
        header.document_slots_offset != layout.document_slots_offset ||
        header.values_offset != layout.values_offset ||
        header.total_size != layout.total_size ||
        layout.total_size != size)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    bundle.runs = calloc(bundle.run_count, sizeof(*bundle.runs));
    disk_runs = calloc(bundle.run_count, sizeof(*disk_runs));
    disk_blocks = calloc(header.block_count, sizeof(*disk_blocks));
    if (bundle.posting_count > SIZE_MAX / sizeof(*bundle.document_slots))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    bundle.document_slots = calloc(
        (size_t) bundle.posting_count,
        sizeof(*bundle.document_slots)
    );
    bundle.values = calloc(
        (size_t) bundle.posting_count,
        sizeof(*bundle.values)
    );
    if (bundle.runs == NULL || disk_runs == NULL ||
        (header.block_count > 0 && disk_blocks == NULL) ||
        bundle.document_slots == NULL ||
        bundle.values == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    if (header.semantic_bmp_size > 0)
    {
        status = evoke_semantic_bmp_packed_deserialize(
            bytes + header.semantic_bmp_offset,
            (size_t) header.semantic_bmp_size,
            &semantic_bmp
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }

    for (uint32_t run_index = 0;
         run_index < bundle.run_count;
         run_index++)
    {
        evoke_term_fold_run *run = &bundle.runs[run_index];
        evoke_term_fold_run *disk_run = &disk_runs[run_index];
        const uint8_t *run_bytes =
            bytes + layout.runs_offset +
            (size_t) run_index * EVOKE_TERM_FOLD_RUN_SIZE;

        status = evoke_term_fold_run_decode(
            run_bytes,
            EVOKE_TERM_FOLD_RUN_SIZE,
            disk_run
        );
        if (status != EVOKE_OK ||
            disk_run->block_offset != disk_block_cursor ||
            disk_run->block_count >
                header.block_count - disk_block_cursor)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        run->term_id = disk_run->term_id;
        run->kind = disk_run->kind;
        run->coverage_sequence = disk_run->coverage_sequence;
        run->posting_offset = logical_cursor;
        run->posting_count = disk_run->posting_count;
        if (run->posting_count > bundle.posting_count - logical_cursor)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            const evoke_semantic_bmp_packed_term *term;

            if (disk_run->posting_offset != semantic_cursor ||
                semantic_run_index >= semantic_bmp.term_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            term = &semantic_bmp.terms[semantic_run_index];
            if (term->term_id != run->term_id ||
                term->posting_count != run->posting_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            status = evoke_semantic_bmp_packed_term_materialize(
                &semantic_bmp,
                semantic_run_index,
                &bundle.document_slots[logical_cursor],
                &bundle.values[logical_cursor],
                (size_t) run->posting_count
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            semantic_cursor += run->posting_count;
            semantic_run_index++;
        }
        else
        {
            if (disk_run->posting_offset != generic_cursor ||
                run->posting_count >
                    header.generic_posting_count - generic_cursor)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            for (uint64_t posting_index = 0;
                 posting_index < run->posting_count;
                 posting_index++)
            {
                uint64_t source_index = generic_cursor + posting_index;
                uint64_t target_index = logical_cursor + posting_index;
                uint32_t value_bits;

                bundle.document_slots[target_index] = evoke_read_u32_le(
                    bytes + layout.document_slots_offset +
                        (size_t) source_index * sizeof(uint32_t)
                );
                value_bits = evoke_read_u32_le(
                    bytes + layout.values_offset +
                        (size_t) source_index * sizeof(uint32_t)
                );
                if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    bundle.values[target_index].term_frequency =
                        value_bits;
                }
                else
                {
                    memcpy(
                        &bundle.values[target_index].impact,
                        &value_bits,
                        sizeof(value_bits)
                    );
                }
            }
            generic_cursor += run->posting_count;
            disk_block_cursor += disk_run->block_count;
        }
        logical_cursor += run->posting_count;
    }
    if (logical_cursor != bundle.posting_count ||
        generic_cursor != header.generic_posting_count ||
        semantic_cursor != semantic_bmp.posting_count ||
        semantic_run_index != semantic_bmp.term_count ||
        disk_block_cursor != header.block_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    for (uint32_t block_index = 0;
         block_index < header.block_count;
         block_index++)
    {
        status = evoke_posting_block_record_decode(
            bytes + layout.blocks_offset +
                (size_t) block_index *
                    EVOKE_POSTING_BLOCK_RECORD_SIZE,
                    EVOKE_POSTING_BLOCK_RECORD_SIZE,
            &disk_blocks[block_index]
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    for (uint32_t run_index = 0;
         run_index < bundle.run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &bundle.runs[run_index];
        const evoke_term_fold_run *disk_run = &disk_runs[run_index];
        evoke_posting_extent extent;
        uint32_t last_document_id;

        if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            continue;
        }
        last_document_id = bundle.document_slots[
            run->posting_offset + run->posting_count - 1
        ];
        if (last_document_id == UINT32_MAX)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        memset(&extent, 0, sizeof(extent));
        extent.indices = &bundle.document_slots[run->posting_offset];
        extent.values = &bundle.values[run->posting_offset];
        extent.len = run->posting_count;
        extent.local_document_count = last_document_id + 1;
        extent.kind = run->kind;
        status = evoke_posting_extent_validate_block_records(
            &extent,
            bundle.block_shift,
            &disk_blocks[disk_run->block_offset],
            disk_run->block_count
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    status = evoke_term_fold_block_directory_build(
        &bundle,
        true,
        &directory
    );
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    bundle.blocks = directory.blocks;
    bundle.block_count = directory.block_count;
    bundle.block_shift = directory.block_shift;
    directory.blocks = NULL;
    directory.block_count = 0;
    for (uint32_t run_index = 0;
         run_index < bundle.run_count;
         run_index++)
    {
        bundle.runs[run_index].block_offset =
            directory.run_block_offsets[run_index];
        bundle.runs[run_index].block_count =
            directory.run_block_counts[run_index];
    }
    status = evoke_term_fold_bundle_validate(&bundle);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    evoke_term_fold_bundle_free(bundle_out);
    *bundle_out = bundle;
    evoke_term_fold_bundle_init(&bundle);

cleanup:
    free(disk_runs);
    free(disk_blocks);
    evoke_persistent_block_directory_free(&directory);
    evoke_semantic_bmp_packed_index_free(&semantic_bmp);
    evoke_term_fold_bundle_free(&bundle);
    if (status != EVOKE_OK)
    {
        return status;
    }
    return EVOKE_OK;
}

typedef struct evoke_segment_payload_layout
{
    size_t runs_offset;
    size_t blocks_offset;
    size_t indices_offset;
    size_t values_offset;
    size_t document_map_offset;
    size_t versions_offset;
    size_t retirements_offset;
    size_t semantic_states_offset;
    size_t semantic_bmp_offset;
    size_t semantic_bmp_size;
    size_t total_size;
} evoke_segment_payload_layout;

void
evoke_segment_payload_init(evoke_segment_payload *payload)
{
    if (payload != NULL)
    {
        memset(payload, 0, sizeof(*payload));
        payload->semantic_impact_precision =
            EVOKE_SEMANTIC_IMPACT_PRECISION_F32;
    }
}

void
evoke_segment_payload_free(evoke_segment_payload *payload)
{
    if (payload == NULL)
    {
        return;
    }
    free(payload->runs);
    free(payload->blocks);
    free(payload->indices);
    free(payload->values);
    free(payload->document_id_map);
    free(payload->versions);
    free(payload->retirements);
    free(payload->semantic_states);
    evoke_segment_payload_init(payload);
}

static uint64_t
evoke_segment_payload_global_document_base(
    const evoke_segment_payload *payload
)
{
    if (payload->document_id_map != NULL)
    {
        return payload->document_id_map[0];
    }
    return payload->document_id_base;
}

static uint64_t
evoke_segment_run_lower_bound(
    const evoke_segment_payload *payload,
    const evoke_segment_term_run *run,
    uint32_t document_id
)
{
    uint64_t low = run->posting_offset;
    uint64_t high = run->posting_offset + run->posting_count;

    while (low < high)
    {
        uint64_t middle = low + (high - low) / 2;

        if (payload->indices[middle] < document_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low;
}

static evoke_status
evoke_segment_payload_contiguous_validate(
    const evoke_segment_payload *payload,
    uint64_t *global_document_base_out
)
{
    uint64_t global_document_base;

    if (payload == NULL || global_document_base_out == NULL ||
        payload->segment_id == 0 || payload->local_document_count == 0 ||
        payload->version_count != payload->local_document_count ||
        payload->versions == NULL || payload->retirement_count != 0 ||
        payload->retirements != NULL || payload->semantic_state_count != 0 ||
        payload->semantic_states != NULL || payload->block_count != 0 ||
        payload->blocks != NULL ||
        (payload->run_count > 0 && payload->runs == NULL) ||
        (payload->posting_count > 0 &&
         (payload->indices == NULL || payload->values == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    if ((payload->flags & EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP) != 0)
    {
        if (payload->document_id_map == NULL)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    else if (payload->document_id_map != NULL)
    {
        return EVOKE_ERR_FORMAT;
    }

    global_document_base = evoke_segment_payload_global_document_base(payload);
    if (global_document_base > UINT32_MAX ||
        global_document_base + payload->local_document_count >
            (uint64_t) UINT32_MAX + 1)
    {
        return EVOKE_ERR_RANGE;
    }
    for (uint32_t document_index = 0;
         document_index < payload->local_document_count;
         document_index++)
    {
        uint64_t expected_slot = global_document_base + document_index;

        if (payload->versions[document_index].document_slot != expected_slot ||
            (payload->document_id_map != NULL &&
             payload->document_id_map[document_index] != expected_slot))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    for (uint32_t run_index = 0;
         run_index < payload->run_count;
         run_index++)
    {
        const evoke_segment_term_run *run = &payload->runs[run_index];

        if (run->posting_count == 0 ||
            run->posting_offset > payload->posting_count ||
            run->posting_count >
                payload->posting_count - run->posting_offset)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint64_t posting_index = run->posting_offset;
             posting_index < run->posting_offset + run->posting_count;
             posting_index++)
        {
            if (payload->indices[posting_index] >=
                    payload->local_document_count ||
                (posting_index > run->posting_offset &&
                 payload->indices[posting_index - 1] >=
                    payload->indices[posting_index]))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    *global_document_base_out = global_document_base;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_payload_block_transition_prefix_build(
    const evoke_segment_payload *payload,
    uint32_t block_shift,
    uint32_t **prefix_out
)
{
    uint32_t *prefix;
    uint64_t global_document_base;

    if (payload == NULL || prefix_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *prefix_out = NULL;
    if (payload->posting_count > SIZE_MAX / sizeof(*prefix))
    {
        return EVOKE_ERR_RANGE;
    }
    if (block_shift == 0 || block_shift >= 32)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (payload->posting_count == 0)
    {
        return EVOKE_OK;
    }
    prefix = malloc(
        (size_t) payload->posting_count * sizeof(*prefix)
    );
    if (prefix == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    global_document_base =
        evoke_segment_payload_global_document_base(payload);
    for (uint32_t run_index = 0;
         run_index < payload->run_count;
         run_index++)
    {
        const evoke_segment_term_run *run = &payload->runs[run_index];
        uint64_t end = run->posting_offset + run->posting_count;
        uint32_t previous_block = 0;

        for (uint64_t posting_index = run->posting_offset;
             posting_index < end;
             posting_index++)
        {
            uint32_t block_id = (uint32_t) (
                (global_document_base + payload->indices[posting_index])
                >> block_shift
            );

            if (posting_index == run->posting_offset)
            {
                prefix[posting_index] = 0;
            }
            else
            {
                prefix[posting_index] = prefix[posting_index - 1] +
                    (block_id != previous_block ? 1U : 0U);
            }
            previous_block = block_id;
        }
    }
    *prefix_out = prefix;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_payload_contiguous_range_size(
    const evoke_segment_payload *payload,
    const uint32_t *block_transition_prefix,
    const uint32_t *semantic_block_transition_prefix,
    uint32_t first_document,
    uint32_t document_count,
    size_t *size_out
)
{
    uint64_t posting_count = 0;
    uint64_t block_count = 0;
    uint64_t semantic_posting_count = 0;
    uint64_t semantic_record_count = 0;
    uint32_t run_count = 0;
    uint32_t semantic_run_count = 0;
    uint32_t end_document;
    size_t size = EVOKE_SEGMENT_PAYLOAD_HEADER_SIZE;
    size_t bytes;

    if (payload == NULL || size_out == NULL ||
        (payload->posting_count > 0 && block_transition_prefix == NULL) ||
        (payload->posting_count > 0 &&
         semantic_block_transition_prefix == NULL) ||
        document_count == 0 ||
        first_document >= payload->local_document_count ||
        document_count > payload->local_document_count - first_document)
    {
        return EVOKE_ERR_INVALID;
    }
    end_document = first_document + document_count;
    for (uint32_t run_index = 0;
         run_index < payload->run_count;
         run_index++)
    {
        const evoke_segment_term_run *run = &payload->runs[run_index];
        uint64_t first_posting = evoke_segment_run_lower_bound(
            payload,
            run,
            first_document
        );
        uint64_t end_posting = evoke_segment_run_lower_bound(
            payload,
            run,
            end_document
        );
        uint64_t count = end_posting - first_posting;

        if (count == 0)
        {
            continue;
        }
        if (posting_count > UINT64_MAX - count)
        {
            return EVOKE_ERR_RANGE;
        }
        posting_count += count;
        if (block_count > UINT64_MAX -
                (UINT64_C(1) +
                 block_transition_prefix[end_posting - 1] -
                 block_transition_prefix[first_posting]))
        {
            return EVOKE_ERR_RANGE;
        }
        block_count += UINT64_C(1) +
            block_transition_prefix[end_posting - 1] -
            block_transition_prefix[first_posting];
        run_count++;
        if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            uint64_t records = UINT64_C(1) +
                semantic_block_transition_prefix[end_posting - 1] -
                semantic_block_transition_prefix[first_posting];

            if (semantic_posting_count > UINT64_MAX - count ||
                semantic_record_count > UINT64_MAX - records)
            {
                return EVOKE_ERR_RANGE;
            }
            semantic_posting_count += count;
            semantic_record_count += records;
            semantic_run_count++;
        }
    }
    if (!evoke_checked_mul_size(
            run_count,
            EVOKE_SEGMENT_TERM_RUN_SIZE,
            &bytes) ||
        !evoke_checked_add_size(size, bytes, &size) ||
        block_count > SIZE_MAX / EVOKE_POSTING_BLOCK_RECORD_SIZE ||
        !evoke_checked_add_size(
            size,
            (size_t) block_count * EVOKE_POSTING_BLOCK_RECORD_SIZE,
            &size) ||
        posting_count > SIZE_MAX / (sizeof(uint32_t) * 2) ||
        !evoke_checked_add_size(
            size,
            (size_t) posting_count * sizeof(uint32_t) * 2,
            &size) ||
        !evoke_checked_mul_size(
            document_count,
            EVOKE_DOCUMENT_VERSION_RECORD_SIZE,
            &bytes) ||
        !evoke_checked_add_size(size, bytes, &size))
    {
        return EVOKE_ERR_RANGE;
    }
    if (semantic_posting_count > 0)
    {
        if (!evoke_checked_add_size(
                size,
                EVOKE_SEMANTIC_BMP_HEADER_SIZE,
                &size) ||
            !evoke_checked_mul_size(
                semantic_run_count,
                sizeof(evoke_semantic_bmp_term),
                &bytes) ||
            !evoke_checked_add_size(size, bytes, &size) ||
            semantic_record_count > SIZE_MAX /
                (sizeof(evoke_semantic_bmp_ref) +
                 sizeof(evoke_semantic_bmp_block) +
                 sizeof(evoke_semantic_bmp_record)) ||
            !evoke_checked_add_size(
                size,
                (size_t) semantic_record_count *
                    (sizeof(evoke_semantic_bmp_ref) +
                     sizeof(evoke_semantic_bmp_block) +
                     sizeof(evoke_semantic_bmp_record)),
                &size) ||
            semantic_posting_count > SIZE_MAX / sizeof(uint32_t) ||
            !evoke_checked_add_size(
                size,
                (size_t) semantic_posting_count * sizeof(uint32_t),
                &size))
        {
            return EVOKE_ERR_RANGE;
        }
    }
    *size_out = size;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_payload_contiguous_range_build(
    const evoke_segment_payload *source,
    uint64_t segment_id,
    uint64_t global_document_base,
    uint32_t first_document,
    uint32_t document_count,
    evoke_segment_payload *payload_out
)
{
    evoke_segment_payload payload;
    uint32_t end_document = first_document + document_count;
    uint32_t run_count = 0;
    uint32_t run_cursor = 0;
    uint64_t posting_count = 0;
    uint64_t posting_cursor = 0;

    evoke_segment_payload_init(&payload);
    for (uint32_t run_index = 0;
         run_index < source->run_count;
         run_index++)
    {
        const evoke_segment_term_run *run = &source->runs[run_index];
        uint64_t first_posting = evoke_segment_run_lower_bound(
            source,
            run,
            first_document
        );
        uint64_t end_posting = evoke_segment_run_lower_bound(
            source,
            run,
            end_document
        );

        if (end_posting > first_posting)
        {
            run_count++;
            posting_count += end_posting - first_posting;
        }
    }
    if (posting_count > SIZE_MAX / sizeof(*payload.indices) ||
        posting_count > SIZE_MAX / sizeof(*payload.values))
    {
        return EVOKE_ERR_RANGE;
    }

    payload.segment_id = segment_id;
    payload.vocab_size = source->vocab_size;
    payload.document_id_base = (uint32_t) (
        global_document_base + first_document
    );
    payload.local_document_count = document_count;
    payload.run_count = run_count;
    payload.posting_count = posting_count;
    payload.version_count = document_count;
    if (run_count > 0)
    {
        payload.runs = calloc(run_count, sizeof(*payload.runs));
    }
    if (posting_count > 0)
    {
        payload.indices = malloc(
            (size_t) posting_count * sizeof(*payload.indices)
        );
        payload.values = malloc(
            (size_t) posting_count * sizeof(*payload.values)
        );
    }
    payload.versions = malloc(
        (size_t) document_count * sizeof(*payload.versions)
    );
    if ((run_count > 0 && payload.runs == NULL) ||
        (posting_count > 0 &&
         (payload.indices == NULL || payload.values == NULL)) ||
        payload.versions == NULL)
    {
        evoke_segment_payload_free(&payload);
        return EVOKE_ERR_NOMEM;
    }

    for (uint32_t run_index = 0;
         run_index < source->run_count;
         run_index++)
    {
        const evoke_segment_term_run *source_run = &source->runs[run_index];
        uint64_t first_posting = evoke_segment_run_lower_bound(
            source,
            source_run,
            first_document
        );
        uint64_t end_posting = evoke_segment_run_lower_bound(
            source,
            source_run,
            end_document
        );
        uint64_t count = end_posting - first_posting;
        evoke_segment_term_run *run;

        if (count == 0)
        {
            continue;
        }
        run = &payload.runs[run_cursor++];
        run->term_id = source_run->term_id;
        run->kind = source_run->kind;
        run->posting_offset = posting_cursor;
        run->posting_count = count;
        for (uint64_t source_posting = first_posting;
             source_posting < end_posting;
             source_posting++)
        {
            payload.indices[posting_cursor] =
                source->indices[source_posting] - first_document;
            payload.values[posting_cursor] =
                source->values[source_posting];
            posting_cursor++;
        }
    }
    memcpy(
        payload.versions,
        &source->versions[first_document],
        (size_t) document_count * sizeof(*payload.versions)
    );
    if (run_cursor != run_count || posting_cursor != posting_count)
    {
        evoke_segment_payload_free(&payload);
        return EVOKE_ERR_FORMAT;
    }
    evoke_segment_payload_free(payload_out);
    *payload_out = payload;
    return EVOKE_OK;
}

evoke_status
evoke_segment_payload_partition_contiguous(
    evoke_segment_payload *payload,
    uint64_t first_segment_id,
    size_t max_bytes,
    evoke_segment_payload **payloads_out,
    uint32_t *payload_count_out
)
{
    evoke_segment_payload *payloads = NULL;
    uint32_t *boundaries = NULL;
    uint32_t *block_transition_prefix = NULL;
    uint32_t *semantic_block_transition_prefix = NULL;
    uint64_t global_document_base = 0;
    uint32_t payload_count = 0;
    uint32_t first_document = 0;
    evoke_status status;

    if (payloads_out == NULL || payload_count_out == NULL ||
        first_segment_id == 0 || max_bytes == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    *payloads_out = NULL;
    *payload_count_out = 0;
    status = evoke_segment_payload_contiguous_validate(
        payload,
        &global_document_base
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_segment_payload_block_transition_prefix_build(
        payload,
        payload->block_shift == 0
            ? EVOKE_DEFAULT_POSTING_BLOCK_SHIFT
            : payload->block_shift,
        &block_transition_prefix
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_segment_payload_block_transition_prefix_build(
        payload,
        EVOKE_SEMANTIC_BMP_BLOCK_SHIFT,
        &semantic_block_transition_prefix
    );
    if (status != EVOKE_OK)
    {
        free(block_transition_prefix);
        return status;
    }
    boundaries = calloc(
        (size_t) EVOKE_SEGMENT_MANIFEST_MAX_SEGMENTS + 1,
        sizeof(*boundaries)
    );
    if (boundaries == NULL)
    {
        free(block_transition_prefix);
        free(semantic_block_transition_prefix);
        return EVOKE_ERR_NOMEM;
    }
    boundaries[0] = 0;
    while (first_document < payload->local_document_count)
    {
        uint32_t low = first_document + 1;
        uint32_t high = payload->local_document_count;
        uint32_t best = first_document;

        while (low <= high)
        {
            uint32_t middle = low + (high - low) / 2;
            size_t size = 0;

            status = evoke_segment_payload_contiguous_range_size(
                payload,
                block_transition_prefix,
                semantic_block_transition_prefix,
                first_document,
                middle - first_document,
                &size
            );
            if (status != EVOKE_OK)
            {
                free(block_transition_prefix);
                free(semantic_block_transition_prefix);
                free(boundaries);
                return status;
            }
            if (size <= max_bytes)
            {
                best = middle;
                low = middle + 1;
            }
            else
            {
                high = middle - 1;
            }
        }
        if (best == first_document ||
            payload_count == EVOKE_SEGMENT_MANIFEST_MAX_SEGMENTS ||
            first_segment_id > UINT64_MAX - payload_count)
        {
            free(block_transition_prefix);
            free(semantic_block_transition_prefix);
            free(boundaries);
            return EVOKE_ERR_RANGE;
        }
        payload_count++;
        boundaries[payload_count] = best;
        first_document = best;
    }

    payloads = calloc(payload_count, sizeof(*payloads));
    if (payloads == NULL)
    {
        free(block_transition_prefix);
        free(semantic_block_transition_prefix);
        free(boundaries);
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t payload_index = 0;
         payload_index < payload_count;
         payload_index++)
    {
        evoke_segment_payload_init(&payloads[payload_index]);
        status = evoke_segment_payload_contiguous_range_build(
            payload,
            first_segment_id + payload_index,
            global_document_base,
            boundaries[payload_index],
            boundaries[payload_index + 1] - boundaries[payload_index],
            &payloads[payload_index]
        );
        if (status != EVOKE_OK)
        {
            for (uint32_t free_index = 0;
                 free_index <= payload_index;
                 free_index++)
            {
                evoke_segment_payload_free(&payloads[free_index]);
            }
            free(payloads);
            free(block_transition_prefix);
            free(semantic_block_transition_prefix);
            free(boundaries);
            return status;
        }
    }
    free(block_transition_prefix);
    free(semantic_block_transition_prefix);
    free(boundaries);
    evoke_segment_payload_free(payload);
    *payloads_out = payloads;
    *payload_count_out = payload_count;
    return EVOKE_OK;
}

uint32_t
evoke_segment_size_class(uint64_t payload_bytes)
{
    uint32_t size_class = 0;

    while (payload_bytes > EVOKE_SEGMENT_SIZE_CLASS_BASE_BYTES)
    {
        payload_bytes =
            payload_bytes / 2 + payload_bytes % 2;
        size_class++;
    }
    return size_class;
}

static evoke_status
evoke_segment_lexical_source_validate(
    const evoke_index *index,
    uint32_t document_id_base,
    const evoke_document_version_record *versions,
    size_t version_count,
    uint32_t *run_count_out
)
{
    uint64_t document_limit;
    uint32_t run_count = 0;
    uint32_t term_id;
    uint32_t document_id;

    if (index == NULL || run_count_out == NULL ||
        version_count != index->num_docs ||
        (version_count > 0 && versions == NULL) ||
        (index->vocab_size > 0 &&
         (index->indptr == NULL || index->doc_frequencies == NULL)) ||
        (index->num_docs > 0 && index->doc_lengths == NULL) ||
        (index->data_len > 0 &&
         (index->indices == NULL || index->term_frequencies == NULL)) ||
        index->data_len > SIZE_MAX / sizeof(uint32_t) ||
        index->data_len > SIZE_MAX / sizeof(evoke_posting_value))
    {
        return EVOKE_ERR_INVALID;
    }
    document_limit = (uint64_t) document_id_base + index->num_docs;
    if (document_limit > (uint64_t) UINT32_MAX + 1)
    {
        return EVOKE_ERR_RANGE;
    }
    if (index->vocab_size > 0 &&
        (index->indptr[0] != 0 ||
         index->indptr[index->vocab_size] != index->data_len))
    {
        return EVOKE_ERR_FORMAT;
    }

    for (document_id = 0; document_id < index->num_docs; document_id++)
    {
        if (versions[document_id].document_slot !=
                (uint64_t) document_id_base + document_id ||
            versions[document_id].document_length !=
                index->doc_lengths[document_id])
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    for (term_id = 0; term_id < index->vocab_size; term_id++)
    {
        uint64_t start = index->indptr[term_id];
        uint64_t end = index->indptr[term_id + 1];
        uint64_t posting_index;

        if (start > end || end > index->data_len ||
            end - start != index->doc_frequencies[term_id])
        {
            return EVOKE_ERR_FORMAT;
        }
        if (end > start)
        {
            run_count++;
        }
        for (posting_index = start; posting_index < end; posting_index++)
        {
            if (index->indices[posting_index] >= index->num_docs ||
                index->term_frequencies[posting_index] == 0 ||
                (posting_index > start &&
                 index->indices[posting_index - 1] >=
                    index->indices[posting_index]))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    *run_count_out = run_count;
    return EVOKE_OK;
}

evoke_status
evoke_segment_payload_build_lexical(
    const evoke_index *index,
    uint64_t segment_id,
    uint32_t document_id_base,
    const evoke_document_version_record *versions,
    size_t version_count,
    evoke_segment_payload *payload_out
)
{
    evoke_segment_payload payload;
    uint32_t run_count = 0;
    uint32_t run_index = 0;
    uint32_t term_id;
    uint64_t posting_index;
    evoke_status status;

    if (segment_id == 0 || payload_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_lexical_source_validate(
        index,
        document_id_base,
        versions,
        version_count,
        &run_count
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    evoke_segment_payload_init(&payload);
    payload.segment_id = segment_id;
    payload.vocab_size = index->vocab_size;
    payload.document_id_base = document_id_base;
    payload.local_document_count = index->num_docs;
    payload.run_count = run_count;
    payload.posting_count = index->data_len;
    payload.version_count = index->num_docs;
    if (run_count > 0)
    {
        payload.runs = calloc(run_count, sizeof(*payload.runs));
    }
    if (payload.posting_count > 0)
    {
        payload.indices = malloc(
            (size_t) payload.posting_count * sizeof(*payload.indices)
        );
        payload.values = calloc(
            (size_t) payload.posting_count,
            sizeof(*payload.values)
        );
    }
    if (payload.version_count > 0)
    {
        payload.versions = malloc(
            (size_t) payload.version_count * sizeof(*payload.versions)
        );
    }
    if ((run_count > 0 && payload.runs == NULL) ||
        (payload.posting_count > 0 &&
         (payload.indices == NULL || payload.values == NULL)) ||
        (payload.version_count > 0 && payload.versions == NULL))
    {
        evoke_segment_payload_free(&payload);
        return EVOKE_ERR_NOMEM;
    }

    for (term_id = 0; term_id < index->vocab_size; term_id++)
    {
        uint64_t start = index->indptr[term_id];
        uint64_t count = index->indptr[term_id + 1] - start;

        if (count == 0)
        {
            continue;
        }
        payload.runs[run_index].term_id = term_id;
        payload.runs[run_index].kind =
            EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
        payload.runs[run_index].posting_offset = start;
        payload.runs[run_index].posting_count = count;
        run_index++;
    }
    if (payload.posting_count > 0)
    {
        memcpy(
            payload.indices,
            index->indices,
            (size_t) payload.posting_count * sizeof(*payload.indices)
        );
        for (posting_index = 0;
             posting_index < payload.posting_count;
             posting_index++)
        {
            payload.values[posting_index].term_frequency =
                index->term_frequencies[posting_index];
        }
    }
    if (payload.version_count > 0)
    {
        memcpy(
            payload.versions,
            versions,
            (size_t) payload.version_count * sizeof(*payload.versions)
        );
    }

    evoke_segment_payload_free(payload_out);
    *payload_out = payload;
    return EVOKE_OK;
}

static int
evoke_compare_sparse_lexical_entry(const void *left, const void *right)
{
    const evoke_term_entry *left_entry = left;
    const evoke_term_entry *right_entry = right;

    if (left_entry->token_id != right_entry->token_id)
    {
        return (left_entry->token_id > right_entry->token_id) -
            (left_entry->token_id < right_entry->token_id);
    }
    return (left_entry->doc_id > right_entry->doc_id) -
        (left_entry->doc_id < right_entry->doc_id);
}

evoke_status
evoke_segment_payload_build_lexical_entries(
    const evoke_term_entry *entries,
    size_t entry_count,
    uint32_t vocab_size,
    uint64_t segment_id,
    uint32_t document_id_base,
    const evoke_document_version_record *versions,
    size_t version_count,
    evoke_segment_payload *payload_out
)
{
    evoke_segment_payload payload;
    evoke_term_entry *sorted_entries = NULL;
    uint32_t run_count = 0;
    uint32_t run_index = 0;
    size_t bytes;

    if (segment_id == 0 || payload_out == NULL ||
        entry_count > UINT32_MAX || version_count > UINT32_MAX ||
        (entry_count > 0 && (entries == NULL || vocab_size == 0)) ||
        (version_count > 0 && versions == NULL) ||
        (uint64_t) document_id_base + version_count >
            (uint64_t) UINT32_MAX + 1)
    {
        return EVOKE_ERR_INVALID;
    }
    for (size_t version_index = 0;
         version_index < version_count;
         version_index++)
    {
        if (versions[version_index].document_slot !=
            (uint64_t) document_id_base + version_index)
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    evoke_segment_payload_init(&payload);
    payload.segment_id = segment_id;
    payload.vocab_size = vocab_size;
    payload.document_id_base = document_id_base;
    payload.local_document_count = (uint32_t) version_count;
    payload.posting_count = entry_count;
    payload.version_count = (uint32_t) version_count;

    if (entry_count > 0)
    {
        if (!evoke_checked_mul_size(
                entry_count,
                sizeof(*sorted_entries),
                &bytes))
        {
            return EVOKE_ERR_RANGE;
        }
        sorted_entries = malloc(bytes);
        if (sorted_entries == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        memcpy(sorted_entries, entries, bytes);
        qsort(
            sorted_entries,
            entry_count,
            sizeof(*sorted_entries),
            evoke_compare_sparse_lexical_entry
        );

        for (size_t entry_index = 0;
             entry_index < entry_count;
             entry_index++)
        {
            const evoke_term_entry *entry = &sorted_entries[entry_index];

            if (entry->token_id >= vocab_size ||
                entry->doc_id >= version_count ||
                entry->tf == 0 ||
                entry->tf > versions[entry->doc_id].document_length)
            {
                free(sorted_entries);
                return EVOKE_ERR_RANGE;
            }
            if (entry_index > 0 &&
                sorted_entries[entry_index - 1].token_id ==
                    entry->token_id &&
                sorted_entries[entry_index - 1].doc_id == entry->doc_id)
            {
                free(sorted_entries);
                return EVOKE_ERR_FORMAT;
            }
            if (entry_index == 0 ||
                sorted_entries[entry_index - 1].token_id !=
                    entry->token_id)
            {
                run_count++;
            }
        }
    }

    payload.run_count = run_count;
    if (run_count > 0)
    {
        payload.runs = calloc(run_count, sizeof(*payload.runs));
    }
    if (entry_count > 0)
    {
        payload.indices = malloc(
            entry_count * sizeof(*payload.indices)
        );
        payload.values = calloc(
            entry_count,
            sizeof(*payload.values)
        );
    }
    if (version_count > 0)
    {
        payload.versions = malloc(
            version_count * sizeof(*payload.versions)
        );
    }
    if ((run_count > 0 && payload.runs == NULL) ||
        (entry_count > 0 &&
         (payload.indices == NULL || payload.values == NULL)) ||
        (version_count > 0 && payload.versions == NULL))
    {
        free(sorted_entries);
        evoke_segment_payload_free(&payload);
        return EVOKE_ERR_NOMEM;
    }

    for (size_t entry_index = 0;
         entry_index < entry_count;
         entry_index++)
    {
        const evoke_term_entry *entry = &sorted_entries[entry_index];
        evoke_segment_term_run *run;

        if (entry_index == 0 ||
            sorted_entries[entry_index - 1].token_id != entry->token_id)
        {
            run = &payload.runs[run_index++];
            run->term_id = entry->token_id;
            run->kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
            run->posting_offset = entry_index;
        }
        else
        {
            run = &payload.runs[run_index - 1];
        }
        run->posting_count++;
        payload.indices[entry_index] = entry->doc_id;
        payload.values[entry_index].term_frequency = entry->tf;
    }
    if (version_count > 0)
    {
        memcpy(
            payload.versions,
            versions,
            version_count * sizeof(*payload.versions)
        );
    }

    free(sorted_entries);
    evoke_segment_payload_free(payload_out);
    *payload_out = payload;
    return EVOKE_OK;
}

evoke_status
evoke_segment_payload_build_lexical_mapped_entries(
    const evoke_term_entry *entries,
    size_t entry_count,
    uint32_t vocab_size,
    uint64_t segment_id,
    const uint32_t *document_id_map,
    const evoke_document_version_record *versions,
    size_t version_count,
    evoke_segment_payload *payload_out
)
{
    evoke_document_version_record *local_versions = NULL;
    evoke_status status;

    if (version_count == 0 || document_id_map == NULL ||
        versions == NULL || payload_out == NULL ||
        version_count > UINT32_MAX ||
        version_count > SIZE_MAX / sizeof(*local_versions))
    {
        return EVOKE_ERR_INVALID;
    }
    local_versions = malloc(version_count * sizeof(*local_versions));
    if (local_versions == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (size_t version_index = 0;
         version_index < version_count;
         version_index++)
    {
        if (versions[version_index].document_slot !=
                document_id_map[version_index] ||
            (version_index > 0 &&
             document_id_map[version_index - 1] >=
                document_id_map[version_index]))
        {
            free(local_versions);
            return EVOKE_ERR_FORMAT;
        }
        local_versions[version_index] = versions[version_index];
        local_versions[version_index].document_slot = version_index;
    }
    status = evoke_segment_payload_build_lexical_entries(
        entries,
        entry_count,
        vocab_size,
        segment_id,
        0,
        local_versions,
        version_count,
        payload_out
    );
    free(local_versions);
    if (status != EVOKE_OK)
    {
        return status;
    }
    payload_out->document_id_map = malloc(
        version_count * sizeof(*payload_out->document_id_map)
    );
    if (payload_out->document_id_map == NULL)
    {
        evoke_segment_payload_free(payload_out);
        return EVOKE_ERR_NOMEM;
    }
    memcpy(
        payload_out->document_id_map,
        document_id_map,
        version_count * sizeof(*payload_out->document_id_map)
    );
    memcpy(
        payload_out->versions,
        versions,
        version_count * sizeof(*payload_out->versions)
    );
    payload_out->flags |= EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP;
    return EVOKE_OK;
}

static int evoke_compare_semantic_state(
    const void *left,
    const void *right
);

static bool evoke_segment_merge_find_document(
    const uint32_t *document_ids,
    uint32_t document_count,
    uint32_t document_id,
    uint32_t *local_document_id_out
);

static int
evoke_compare_semantic_posting(
    const void *left,
    const void *right
)
{
    const evoke_segment_semantic_posting *left_posting = left;
    const evoke_segment_semantic_posting *right_posting = right;

    if (left_posting->term_id != right_posting->term_id)
    {
        return (left_posting->term_id > right_posting->term_id) -
            (left_posting->term_id < right_posting->term_id);
    }
    return (left_posting->document_slot >
            right_posting->document_slot) -
        (left_posting->document_slot <
         right_posting->document_slot);
}

typedef struct evoke_segment_semantic_array_reader
{
    const evoke_segment_semantic_posting *postings;
    size_t posting_count;
    size_t cursor;
} evoke_segment_semantic_array_reader;

static evoke_status
evoke_segment_semantic_array_read(
    void *context,
    evoke_segment_semantic_posting *posting_out
)
{
    evoke_segment_semantic_array_reader *reader = context;

    if (reader == NULL || posting_out == NULL ||
        reader->cursor >= reader->posting_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    *posting_out = reader->postings[reader->cursor++];
    return EVOKE_OK;
}

static evoke_status
evoke_segment_semantic_array_rewind(void *context)
{
    evoke_segment_semantic_array_reader *reader = context;

    if (reader == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    reader->cursor = 0;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_payload_attach_semantic_sorted_source(
    evoke_segment_payload *payload,
    size_t posting_count,
    evoke_segment_semantic_posting_reader reader,
    evoke_segment_semantic_posting_rewind rewind,
    void *reader_context,
    const evoke_semantic_state_record *states,
    size_t state_count
)
{
    evoke_semantic_state_record *sorted_states = NULL;
    evoke_segment_term_run *next_runs = NULL;
    uint32_t *next_indices = NULL;
    evoke_posting_value *next_values = NULL;
    uint32_t *document_ids = NULL;
    size_t source_document_count;
    size_t semantic_run_count = 0;
    size_t next_run_count;
    size_t semantic_cursor = 0;
    uint64_t next_posting_count;
    uint64_t posting_cursor = 0;
    uint32_t old_run_cursor = 0;
    uint32_t next_run_cursor = 0;
    uint32_t document_count = 0;
    evoke_segment_semantic_posting semantic_posting = {0};
    bool have_semantic_posting = false;
    evoke_status status = EVOKE_OK;

    if (payload == NULL ||
        (posting_count > 0 &&
         (reader == NULL || rewind == NULL || reader_context == NULL)) ||
        (state_count > 0 && states == NULL) ||
        payload->semantic_state_count != 0 ||
        payload->semantic_states != NULL ||
        (payload->run_count > 0 && payload->runs == NULL) ||
        (payload->posting_count > 0 &&
         (payload->indices == NULL || payload->values == NULL)) ||
        posting_count > UINT32_MAX ||
        state_count > UINT32_MAX ||
        payload->local_document_count >
            SIZE_MAX - state_count)
    {
        return EVOKE_ERR_INVALID;
    }
    if (posting_count == 0 && state_count == 0)
    {
        return EVOKE_OK;
    }
    if (state_count > SIZE_MAX / sizeof(*sorted_states))
    {
        return EVOKE_ERR_RANGE;
    }
    if (state_count > 0)
    {
        sorted_states = malloc(state_count * sizeof(*sorted_states));
        if (sorted_states == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        memcpy(
            sorted_states,
            states,
            state_count * sizeof(*sorted_states)
        );
        qsort(
            sorted_states,
            state_count,
            sizeof(*sorted_states),
            evoke_compare_semantic_state
        );
    }
    for (size_t state_index = 0;
         state_index < state_count;
         state_index++)
    {
        if (sorted_states[state_index].document_slot > UINT32_MAX ||
            (state_index > 0 &&
             sorted_states[state_index - 1].document_slot >=
                sorted_states[state_index].document_slot))
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
    }

    if (posting_count > 0)
    {
        status = rewind(reader_context);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    {
        evoke_segment_semantic_posting previous = {0};
        bool have_previous = false;

        for (size_t posting_index = 0;
             posting_index < posting_count;
             posting_index++)
        {
            evoke_segment_semantic_posting posting;

            status = reader(reader_context, &posting);
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }

            if (posting.term_id >= payload->vocab_size ||
                !isfinite(posting.impact) ||
                posting.impact < 0.0f ||
                (have_previous && previous.term_id == posting.term_id &&
                 previous.document_slot >= posting.document_slot) ||
                (have_previous && previous.term_id > posting.term_id))
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            if (!have_previous || previous.term_id != posting.term_id)
            {
                semantic_run_count++;
            }
            previous = posting;
            have_previous = true;
        }
    }
    if (semantic_run_count > UINT32_MAX - payload->run_count ||
        payload->posting_count > UINT64_MAX - posting_count)
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    next_posting_count = payload->posting_count + posting_count;
    next_run_count =
        (size_t) payload->run_count + semantic_run_count;
    if (next_posting_count >
            SIZE_MAX / sizeof(*next_indices) ||
        next_posting_count >
            SIZE_MAX / sizeof(*next_values))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }

    source_document_count =
        (size_t) payload->local_document_count + state_count;
    document_ids = malloc(
        source_document_count * sizeof(*document_ids)
    );
    if (document_ids == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (uint32_t local_document_id = 0;
         local_document_id < payload->local_document_count;
         local_document_id++)
    {
        uint64_t document_slot = payload->document_id_map != NULL
            ? payload->document_id_map[local_document_id]
            : (uint64_t) payload->document_id_base +
                local_document_id;

        if (document_slot > UINT32_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        document_ids[document_count++] = (uint32_t) document_slot;
    }
    for (size_t state_index = 0;
         state_index < state_count;
         state_index++)
    {
        document_ids[document_count++] =
            (uint32_t) sorted_states[state_index].document_slot;
    }
    qsort(
        document_ids,
        document_count,
        sizeof(*document_ids),
        evoke_compare_u32_ascending
    );
    {
        uint32_t unique_count = 0;

        for (uint32_t source_index = 0;
             source_index < document_count;
             source_index++)
        {
            if (unique_count == 0 ||
                document_ids[unique_count - 1] !=
                    document_ids[source_index])
            {
                document_ids[unique_count++] =
                    document_ids[source_index];
            }
        }
        document_count = unique_count;
    }
    if (posting_count > 0)
    {
        status = rewind(reader_context);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    for (size_t posting_index = 0;
         posting_index < posting_count;
         posting_index++)
    {
        evoke_segment_semantic_posting posting;
        uint32_t state_low = 0;
        uint32_t state_high = (uint32_t) state_count;
        uint32_t version_low = 0;
        uint32_t version_high = payload->version_count;
        bool complete = false;

        status = reader(reader_context, &posting);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }

        while (state_low < state_high)
        {
            uint32_t middle =
                state_low + (state_high - state_low) / 2;
            uint64_t candidate =
                sorted_states[middle].document_slot;

            if (candidate < posting.document_slot)
            {
                state_low = middle + 1;
            }
            else
            {
                state_high = middle;
            }
        }
        if (state_low < state_count &&
            sorted_states[state_low].document_slot ==
                posting.document_slot &&
            (sorted_states[state_low].flags &
             EVOKE_SEMANTIC_STATE_FLAG_COMPLETE) != 0)
        {
            complete = true;
        }
        while (!complete && version_low < version_high)
        {
            uint32_t middle =
                version_low + (version_high - version_low) / 2;
            uint64_t candidate =
                payload->versions[middle].document_slot;

            if (candidate < posting.document_slot)
            {
                version_low = middle + 1;
            }
            else
            {
                version_high = middle;
            }
        }
        if (!complete &&
            version_low < payload->version_count &&
            payload->versions[version_low].document_slot ==
                posting.document_slot &&
            (payload->versions[version_low].flags &
             EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE) != 0)
        {
            complete = true;
        }
        if (!complete)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
    }

    if (next_run_count > 0)
    {
        next_runs = calloc(next_run_count, sizeof(*next_runs));
    }
    if (next_posting_count > 0)
    {
        next_indices = malloc(
            (size_t) next_posting_count * sizeof(*next_indices)
        );
        next_values = malloc(
            (size_t) next_posting_count * sizeof(*next_values)
        );
    }
    if ((next_run_count > 0 && next_runs == NULL) ||
        (next_posting_count > 0 &&
         (next_indices == NULL || next_values == NULL)))
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    if (posting_count > 0)
    {
        status = rewind(reader_context);
        if (status == EVOKE_OK)
        {
            status = reader(reader_context, &semantic_posting);
        }
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        have_semantic_posting = true;
    }
    while (old_run_cursor < payload->run_count ||
           have_semantic_posting)
    {
        uint32_t old_term_id = old_run_cursor < payload->run_count
            ? payload->runs[old_run_cursor].term_id
            : UINT32_MAX;
        uint32_t semantic_term_id = have_semantic_posting
            ? semantic_posting.term_id
            : UINT32_MAX;

        if (old_term_id <= semantic_term_id)
        {
            const evoke_segment_term_run *source_run =
                &payload->runs[old_run_cursor++];
            evoke_segment_term_run *next_run =
                &next_runs[next_run_cursor++];
            uint64_t source_end;

            if (source_run->kind !=
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL ||
                source_run->posting_offset >
                    payload->posting_count ||
                source_run->posting_count >
                    payload->posting_count -
                        source_run->posting_offset)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            *next_run = *source_run;
            next_run->posting_offset = posting_cursor;
            next_run->block_offset = 0;
            next_run->block_count = 0;
            source_end = source_run->posting_offset +
                source_run->posting_count;
            for (uint64_t source_index =
                     source_run->posting_offset;
                 source_index < source_end;
                 source_index++)
            {
                uint32_t source_local_id =
                    payload->indices[source_index];
                uint64_t global_document_id;
                uint32_t next_local_id;

                if (source_local_id >=
                    payload->local_document_count)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                global_document_id =
                    payload->document_id_map != NULL
                    ? payload->document_id_map[source_local_id]
                    : (uint64_t) payload->document_id_base +
                        source_local_id;
                if (global_document_id > UINT32_MAX ||
                    !evoke_segment_merge_find_document(
                        document_ids,
                        document_count,
                        (uint32_t) global_document_id,
                        &next_local_id))
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                next_indices[posting_cursor] = next_local_id;
                next_values[posting_cursor] =
                    payload->values[source_index];
                posting_cursor++;
            }
        }
        else
        {
            evoke_segment_term_run *next_run =
                &next_runs[next_run_cursor++];
            size_t run_start = semantic_cursor;

            while (have_semantic_posting &&
                   semantic_posting.term_id == semantic_term_id)
            {
                uint32_t local_document_id;

                if (!evoke_segment_merge_find_document(
                        document_ids,
                        document_count,
                        semantic_posting.document_slot,
                        &local_document_id))
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                next_indices[posting_cursor] = local_document_id;
                next_values[posting_cursor].impact =
                    semantic_posting.impact;
                posting_cursor++;
                semantic_cursor++;
                if (semantic_cursor < posting_count)
                {
                    status = reader(reader_context, &semantic_posting);
                    if (status != EVOKE_OK)
                    {
                        goto cleanup;
                    }
                }
                else
                {
                    have_semantic_posting = false;
                }
            }
            next_run->term_id = semantic_term_id;
            next_run->kind =
                EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
            next_run->posting_offset =
                posting_cursor - (semantic_cursor - run_start);
            next_run->posting_count =
                semantic_cursor - run_start;
        }
    }
    if (posting_cursor != next_posting_count ||
        next_run_cursor != next_run_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }

    free(payload->semantic_states);
    free(payload->document_id_map);
    free(payload->blocks);
    free(payload->values);
    free(payload->indices);
    free(payload->runs);
    payload->semantic_states = sorted_states;
    payload->semantic_state_count = (uint32_t) state_count;
    payload->document_id_map = document_ids;
    payload->blocks = NULL;
    payload->block_count = 0;
    payload->block_shift = 0;
    payload->document_id_base = 0;
    payload->local_document_count = document_count;
    payload->values = next_values;
    payload->indices = next_indices;
    payload->posting_count = next_posting_count;
    payload->runs = next_runs;
    payload->run_count = (uint32_t) next_run_count;
    payload->flags |= EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP;
    sorted_states = NULL;
    document_ids = NULL;
    next_values = NULL;
    next_indices = NULL;
    next_runs = NULL;

cleanup:
    free(next_runs);
    free(next_values);
    free(next_indices);
    free(document_ids);
    free(sorted_states);
    return status;
}

evoke_status
evoke_segment_payload_attach_semantic_sorted_reader(
    evoke_segment_payload *payload,
    size_t posting_count,
    evoke_segment_semantic_posting_reader reader,
    evoke_segment_semantic_posting_rewind rewind,
    void *reader_context,
    const evoke_semantic_state_record *states,
    size_t state_count
)
{
    return evoke_segment_payload_attach_semantic_sorted_source(
        payload,
        posting_count,
        reader,
        rewind,
        reader_context,
        states,
        state_count
    );
}

evoke_status
evoke_segment_payload_attach_semantic(
    evoke_segment_payload *payload,
    const evoke_segment_semantic_posting *postings,
    size_t posting_count,
    const evoke_semantic_state_record *states,
    size_t state_count
)
{
    evoke_segment_semantic_array_reader reader = {0};
    evoke_segment_semantic_posting *sorted_postings = NULL;
    evoke_status status;

    if (posting_count > 0 && postings == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (posting_count > SIZE_MAX / sizeof(*sorted_postings))
    {
        return EVOKE_ERR_RANGE;
    }
    if (posting_count > 0)
    {
        sorted_postings = malloc(
            posting_count * sizeof(*sorted_postings)
        );
        if (sorted_postings == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        memcpy(
            sorted_postings,
            postings,
            posting_count * sizeof(*sorted_postings)
        );
        qsort(
            sorted_postings,
            posting_count,
            sizeof(*sorted_postings),
            evoke_compare_semantic_posting
        );
    }
    reader.postings = sorted_postings;
    reader.posting_count = posting_count;
    status = evoke_segment_payload_attach_semantic_sorted_source(
        payload,
        posting_count,
        evoke_segment_semantic_array_read,
        evoke_segment_semantic_array_rewind,
        &reader,
        states,
        state_count
    );
    free(sorted_postings);
    return status;
}

void
evoke_segment_payload_as_view(
    const evoke_segment_payload *payload,
    evoke_segment_payload_view *view_out
)
{
    if (view_out == NULL)
    {
        return;
    }
    memset(view_out, 0, sizeof(*view_out));
    if (payload == NULL)
    {
        return;
    }
    view_out->segment_id = payload->segment_id;
    view_out->posting_count = payload->posting_count;
    view_out->runs = payload->runs;
    view_out->run_count = payload->run_count;
    view_out->block_shift = payload->block_shift;
    view_out->block_count = payload->block_count;
    view_out->blocks = payload->blocks;
    view_out->values = payload->values;
    view_out->indices = payload->indices;
    view_out->document_id_map = payload->document_id_map;
    view_out->document_id_base = payload->document_id_base;
    view_out->local_document_count = payload->local_document_count;
}

evoke_status
evoke_segment_payload_validate(
    const evoke_segment_payload *payload,
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *segment
)
{
    evoke_segment_payload_view view;
    bool has_aborted_hole = false;
    bool has_blocks;
    bool has_map;
    uint64_t block_cursor = 0;
    uint32_t i;
    evoke_status status;

    if (payload == NULL || manifest == NULL || segment == NULL ||
        evoke_semantic_bmp_impact_width(
            payload->semantic_impact_precision
        ) == 0 ||
        (payload->flags & ~EVOKE_SEGMENT_PAYLOAD_KNOWN_FLAGS) != 0 ||
        payload->vocab_size > manifest->vocab_size ||
        payload->version_count != segment->document_count ||
        payload->retirement_count != segment->retirement_count ||
        payload->semantic_state_count !=
            segment->semantic_state_count ||
        (payload->run_count > 0 && payload->runs == NULL) ||
        (payload->posting_count > 0 &&
         (payload->indices == NULL || payload->values == NULL)) ||
        (payload->version_count > 0 && payload->versions == NULL) ||
        (payload->retirement_count > 0 &&
         payload->retirements == NULL) ||
        (payload->semantic_state_count > 0 &&
         payload->semantic_states == NULL))
    {
        return EVOKE_ERR_FORMAT;
    }
    has_map = (payload->flags &
        EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP) != 0;
    if (has_map != (payload->document_id_map != NULL) ||
        (has_map && (payload->local_document_count == 0 ||
                     payload->document_id_base != 0)))
    {
        return EVOKE_ERR_FORMAT;
    }
    for (i = 0; i < payload->run_count; i++)
    {
        if (payload->runs[i].term_id >= payload->vocab_size)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    evoke_segment_payload_as_view(payload, &view);
    status = evoke_segment_payload_view_validate(
        manifest,
        segment,
        &view,
        payload->versions,
        payload->version_count,
        payload->retirements,
        payload->retirement_count
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    has_blocks = payload->block_shift != 0 ||
        payload->block_count != 0 || payload->blocks != NULL;
    if (!has_blocks)
    {
        for (i = 0; i < payload->run_count; i++)
        {
            if (payload->runs[i].block_offset != 0 ||
                payload->runs[i].block_count != 0)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    else
    {
        if (payload->block_shift == 0 ||
            payload->block_shift >= 32 ||
            (payload->block_count == 0) !=
                (payload->blocks == NULL) ||
            (payload->block_count == 0 &&
             (payload->run_count != 0 ||
              payload->posting_count != 0)))
        {
            return EVOKE_ERR_FORMAT;
        }
        for (i = 0; i < payload->run_count; i++)
        {
            const evoke_segment_term_run *run = &payload->runs[i];
            evoke_posting_extent extent;

            if (run->block_offset != block_cursor ||
                run->block_count == 0 ||
                run->block_offset > payload->block_count ||
                run->block_count >
                    payload->block_count - run->block_offset)
            {
                return EVOKE_ERR_FORMAT;
            }
            memset(&extent, 0, sizeof(extent));
            extent.indices =
                &payload->indices[run->posting_offset];
            extent.values =
                &payload->values[run->posting_offset];
            extent.document_id_map = payload->document_id_map;
            extent.len = run->posting_count;
            extent.document_id_base = payload->document_id_base;
            extent.local_document_count =
                payload->local_document_count;
            extent.kind = run->kind;
            if (evoke_posting_extent_validate_block_records(
                    &extent,
                    payload->block_shift,
                    &payload->blocks[run->block_offset],
                    run->block_count) != EVOKE_OK)
            {
                return EVOKE_ERR_FORMAT;
            }
            block_cursor += run->block_count;
        }
        if (block_cursor != payload->block_count)
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    status = evoke_semantic_state_records_validate(
        manifest,
        segment,
        payload->semantic_states,
        payload->semantic_state_count
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    for (i = 0; i < payload->version_count; i++)
    {
        uint64_t slot = payload->versions[i].document_slot;
        bool found = false;

        if ((payload->versions[i].flags &
             EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) != 0)
        {
            has_aborted_hole = true;
        }
        if (!has_map)
        {
            found = slot >= payload->document_id_base &&
                slot < (uint64_t) payload->document_id_base +
                    payload->local_document_count;
        }
        else
        {
            uint32_t low = 0;
            uint32_t high = payload->local_document_count;

            while (low < high)
            {
                uint32_t middle = low + (high - low) / 2;
                uint32_t candidate = payload->document_id_map[middle];

                if ((uint64_t) candidate < slot)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle;
                }
            }
            found = low < payload->local_document_count &&
                payload->document_id_map[low] == slot;
        }
        if (!found)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    for (i = 0; i < payload->semantic_state_count; i++)
    {
        uint64_t slot = payload->semantic_states[i].document_slot;
        bool found = false;

        if (!has_map)
        {
            found = slot >= payload->document_id_base &&
                slot < (uint64_t) payload->document_id_base +
                    payload->local_document_count;
        }
        else
        {
            uint32_t low = 0;
            uint32_t high = payload->local_document_count;

            while (low < high)
            {
                uint32_t middle = low + (high - low) / 2;
                uint32_t candidate = payload->document_id_map[middle];

                if ((uint64_t) candidate < slot)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle;
                }
            }
            found = low < payload->local_document_count &&
                payload->document_id_map[low] == slot;
        }
        if (!found)
        {
            return EVOKE_ERR_FORMAT;
        }
        {
            uint32_t low = 0;
            uint32_t high = payload->version_count;

            while (low < high)
            {
                uint32_t middle = low + (high - low) / 2;
                uint64_t candidate =
                    payload->versions[middle].document_slot;

                if (candidate < slot)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle;
                }
            }
            if (low < payload->version_count &&
                payload->versions[low].document_slot == slot &&
                memcmp(
                    payload->versions[low].
                        semantic_input_fingerprint,
                    payload->semantic_states[i].
                        semantic_input_fingerprint,
                    EVOKE_DOCUMENT_FINGERPRINT_BYTES
                ) != 0)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    for (i = 0; i < payload->run_count; i++)
    {
        const evoke_segment_term_run *run = &payload->runs[i];

        if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            continue;
        }
        for (uint64_t posting_index = run->posting_offset;
             posting_index <
                run->posting_offset + run->posting_count;
             posting_index++)
        {
            uint32_t local_document_id =
                payload->indices[posting_index];
            uint64_t document_slot = payload->document_id_map != NULL
                ? payload->document_id_map[local_document_id]
                : payload->document_id_base + local_document_id;
            uint32_t state_low = 0;
            uint32_t state_high = payload->semantic_state_count;
            uint32_t version_low = 0;
            uint32_t version_high = payload->version_count;
            bool complete = false;

            while (state_low < state_high)
            {
                uint32_t middle =
                    state_low + (state_high - state_low) / 2;
                uint64_t candidate =
                    payload->semantic_states[middle].document_slot;

                if (candidate < document_slot)
                {
                    state_low = middle + 1;
                }
                else
                {
                    state_high = middle;
                }
            }
            if (state_low < payload->semantic_state_count &&
                payload->semantic_states[state_low].document_slot ==
                    document_slot &&
                (payload->semantic_states[state_low].flags &
                 EVOKE_SEMANTIC_STATE_FLAG_COMPLETE) != 0)
            {
                complete = true;
            }
            while (!complete && version_low < version_high)
            {
                uint32_t middle =
                    version_low + (version_high - version_low) / 2;
                uint64_t candidate =
                    payload->versions[middle].document_slot;

                if (candidate < document_slot)
                {
                    version_low = middle + 1;
                }
                else
                {
                    version_high = middle;
                }
            }
            if (!complete &&
                version_low < payload->version_count &&
                payload->versions[version_low].document_slot ==
                    document_slot &&
                (payload->versions[version_low].flags &
                 EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE) != 0)
            {
                complete = true;
            }
            if (!complete)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    if (has_aborted_hole)
    {
        uint64_t posting_index;

        for (posting_index = 0;
             posting_index < payload->posting_count;
             posting_index++)
        {
            uint32_t local_document_id =
                payload->indices[posting_index];
            uint32_t document_id = payload->document_id_map != NULL
                ? payload->document_id_map[local_document_id]
                : payload->document_id_base + local_document_id;
            uint32_t low = 0;
            uint32_t high = payload->version_count;

            while (low < high)
            {
                uint32_t middle = low + (high - low) / 2;
                uint64_t candidate =
                    payload->versions[middle].document_slot;

                if (candidate < document_id)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle;
                }
            }
            if (low < payload->version_count &&
                payload->versions[low].document_slot == document_id &&
                (payload->versions[low].flags &
                 EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) != 0)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    return EVOKE_OK;
}

typedef struct evoke_segment_merge_posting
{
    uint32_t document_id;
    evoke_posting_value value;
} evoke_segment_merge_posting;

static int
evoke_compare_document_version(
    const void *left,
    const void *right
)
{
    const evoke_document_version_record *left_record = left;
    const evoke_document_version_record *right_record = right;

    return (left_record->document_slot >
            right_record->document_slot) -
        (left_record->document_slot < right_record->document_slot);
}

static int
evoke_compare_document_retirement(
    const void *left,
    const void *right
)
{
    const evoke_document_retirement_record *left_record = left;
    const evoke_document_retirement_record *right_record = right;

    return (left_record->document_slot >
            right_record->document_slot) -
        (left_record->document_slot < right_record->document_slot);
}

static int
evoke_compare_semantic_state(
    const void *left,
    const void *right
)
{
    const evoke_semantic_state_record *left_record = left;
    const evoke_semantic_state_record *right_record = right;

    if (left_record->document_slot != right_record->document_slot)
    {
        return (left_record->document_slot >
                right_record->document_slot) -
            (left_record->document_slot <
             right_record->document_slot);
    }
    return (left_record->transition_sequence >
            right_record->transition_sequence) -
        (left_record->transition_sequence <
         right_record->transition_sequence);
}

static int
evoke_compare_merge_posting(
    const void *left,
    const void *right
)
{
    const evoke_segment_merge_posting *left_posting = left;
    const evoke_segment_merge_posting *right_posting = right;

    return (left_posting->document_id >
            right_posting->document_id) -
        (left_posting->document_id <
         right_posting->document_id);
}

static bool
evoke_segment_merge_find_document(
    const uint32_t *document_ids,
    uint32_t document_count,
    uint32_t document_id,
    uint32_t *local_document_id_out
)
{
    uint32_t low = 0;
    uint32_t high = document_count;

    while (low < high)
    {
        uint32_t middle = low + (high - low) / 2;

        if (document_ids[middle] < document_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= document_count || document_ids[low] != document_id)
    {
        return false;
    }
    *local_document_id_out = low;
    return true;
}

static bool
evoke_segment_merge_document_is_excluded(
    const uint32_t *excluded_document_ids,
    uint32_t excluded_document_count,
    uint32_t document_id
)
{
    uint32_t low = 0;
    uint32_t high = excluded_document_count;

    while (low < high)
    {
        uint32_t middle = low + (high - low) / 2;

        if (excluded_document_ids[middle] < document_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low < excluded_document_count &&
        excluded_document_ids[low] == document_id;
}

evoke_status
evoke_segment_payload_merge_excluding(
    const evoke_segment_manifest *manifest,
    uint32_t first_segment_index,
    const evoke_segment_payload *payloads,
    uint32_t payload_count,
    uint64_t merged_segment_id,
    const uint32_t *excluded_document_ids,
    uint32_t excluded_document_count,
    evoke_segment_payload *payload_out
)
{
    evoke_segment_payload merged;
    evoke_segment_merge_posting *posting_buffer = NULL;
    uint32_t *run_cursors = NULL;
    uint64_t total_posting_count = 0;
    uint64_t total_local_document_count = 0;
    uint64_t total_version_count = 0;
    uint64_t total_retirement_count = 0;
    uint64_t total_semantic_state_count = 0;
    uint64_t total_run_count = 0;
    uint64_t posting_cursor = 0;
    uint32_t document_cursor = 0;
    uint32_t version_cursor = 0;
    uint32_t retirement_cursor = 0;
    uint32_t semantic_state_cursor = 0;
    uint32_t output_run_count = 0;
    evoke_status status;

    if (manifest == NULL || payloads == NULL || payload_out == NULL ||
        payload_count < 2 || merged_segment_id == 0 ||
        (excluded_document_count > 0 &&
         excluded_document_ids == NULL) ||
        first_segment_index >= manifest->segment_count ||
        payload_count >
            manifest->segment_count - first_segment_index)
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint32_t excluded_index = 1;
         excluded_index < excluded_document_count;
         excluded_index++)
    {
        if (excluded_document_ids[excluded_index - 1] >=
            excluded_document_ids[excluded_index])
        {
            return EVOKE_ERR_INVALID;
        }
    }
    status = evoke_segment_manifest_validate(manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    for (uint32_t payload_index = 0;
         payload_index < payload_count;
         payload_index++)
    {
        const evoke_segment_descriptor *segment =
            &manifest->segments[first_segment_index + payload_index];
        const evoke_segment_payload *payload = &payloads[payload_index];

        status = evoke_segment_payload_validate(
            payload,
            manifest,
            segment
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        if (total_run_count > UINT64_MAX - payload->run_count)
        {
            return EVOKE_ERR_RANGE;
        }
        total_run_count += payload->run_count;
        for (uint32_t local_document_id = 0;
             local_document_id < payload->local_document_count;
             local_document_id++)
        {
            uint32_t document_id = payload->document_id_map != NULL
                ? payload->document_id_map[local_document_id]
                : payload->document_id_base + local_document_id;

            if (!evoke_segment_merge_document_is_excluded(
                    excluded_document_ids,
                    excluded_document_count,
                    document_id))
            {
                if (total_local_document_count == UINT64_MAX)
                {
                    return EVOKE_ERR_RANGE;
                }
                total_local_document_count++;
            }
        }
        for (uint32_t version_index = 0;
             version_index < payload->version_count;
             version_index++)
        {
            if (!evoke_segment_merge_document_is_excluded(
                    excluded_document_ids,
                    excluded_document_count,
                    (uint32_t) payload->versions[
                        version_index
                    ].document_slot))
            {
                if (total_version_count == UINT64_MAX)
                {
                    return EVOKE_ERR_RANGE;
                }
                total_version_count++;
            }
        }
        for (uint32_t retirement_index = 0;
             retirement_index < payload->retirement_count;
             retirement_index++)
        {
            if (!evoke_segment_merge_document_is_excluded(
                    excluded_document_ids,
                    excluded_document_count,
                    (uint32_t) payload->retirements[
                        retirement_index
                    ].document_slot))
            {
                if (total_retirement_count == UINT64_MAX)
                {
                    return EVOKE_ERR_RANGE;
                }
                total_retirement_count++;
            }
        }
        for (uint32_t state_index = 0;
             state_index < payload->semantic_state_count;
             state_index++)
        {
            if (!evoke_segment_merge_document_is_excluded(
                    excluded_document_ids,
                    excluded_document_count,
                    (uint32_t) payload->semantic_states[
                        state_index
                    ].document_slot))
            {
                if (total_semantic_state_count == UINT64_MAX)
                {
                    return EVOKE_ERR_RANGE;
                }
                total_semantic_state_count++;
            }
        }
        for (uint64_t posting_index = 0;
             posting_index < payload->posting_count;
             posting_index++)
        {
            uint32_t local_document_id =
                payload->indices[posting_index];
            uint32_t document_id = payload->document_id_map != NULL
                ? payload->document_id_map[local_document_id]
                : payload->document_id_base + local_document_id;

            if (!evoke_segment_merge_document_is_excluded(
                    excluded_document_ids,
                    excluded_document_count,
                    document_id))
            {
                if (total_posting_count == UINT64_MAX)
                {
                    return EVOKE_ERR_RANGE;
                }
                total_posting_count++;
            }
        }
    }
    if (total_posting_count >
            SIZE_MAX / sizeof(*posting_buffer) ||
        total_posting_count >
            SIZE_MAX / sizeof(*merged.indices) ||
        total_posting_count >
            SIZE_MAX / sizeof(*merged.values) ||
        total_local_document_count > UINT32_MAX ||
        total_version_count > UINT32_MAX ||
        total_retirement_count > UINT32_MAX ||
        total_semantic_state_count > UINT32_MAX ||
        total_run_count > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }

    evoke_segment_payload_init(&merged);
    merged.segment_id = merged_segment_id;
    merged.vocab_size = manifest->vocab_size;
    merged.version_count = (uint32_t) total_version_count;
    merged.retirement_count = (uint32_t) total_retirement_count;
    merged.semantic_state_count =
        (uint32_t) total_semantic_state_count;
    merged.posting_count = total_posting_count;
    if (total_local_document_count > 0)
    {
        merged.flags |= EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP;
        merged.document_id_map = malloc(
            (size_t) total_local_document_count *
                sizeof(*merged.document_id_map)
        );
    }
    if (merged.version_count > 0)
    {
        merged.versions = malloc(
            (size_t) merged.version_count *
                sizeof(*merged.versions)
        );
    }
    if (merged.retirement_count > 0)
    {
        merged.retirements = malloc(
            (size_t) merged.retirement_count *
                sizeof(*merged.retirements)
        );
    }
    if (merged.semantic_state_count > 0)
    {
        merged.semantic_states = malloc(
            (size_t) merged.semantic_state_count *
                sizeof(*merged.semantic_states)
        );
    }
    if (merged.posting_count > 0)
    {
        merged.indices = malloc(
            (size_t) merged.posting_count * sizeof(*merged.indices)
        );
        merged.values = malloc(
            (size_t) merged.posting_count * sizeof(*merged.values)
        );
        posting_buffer = malloc(
            (size_t) merged.posting_count * sizeof(*posting_buffer)
        );
    }
    if (total_run_count > 0)
    {
        merged.runs = calloc(
            (size_t) total_run_count,
            sizeof(*merged.runs)
        );
    }
    run_cursors = calloc(payload_count, sizeof(*run_cursors));
    if ((total_local_document_count > 0 &&
         merged.document_id_map == NULL) ||
        (merged.version_count > 0 && merged.versions == NULL) ||
        (merged.retirement_count > 0 && merged.retirements == NULL) ||
        (merged.semantic_state_count > 0 &&
         merged.semantic_states == NULL) ||
        (merged.posting_count > 0 &&
         (merged.indices == NULL || merged.values == NULL ||
          posting_buffer == NULL)) ||
        (total_run_count > 0 && merged.runs == NULL) ||
        run_cursors == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    for (uint32_t payload_index = 0;
         payload_index < payload_count;
         payload_index++)
    {
        const evoke_segment_payload *payload = &payloads[payload_index];

        for (uint32_t local_document_id = 0;
             local_document_id < payload->local_document_count;
             local_document_id++)
        {
            uint32_t document_id = payload->document_id_map != NULL
                ? payload->document_id_map[local_document_id]
                : payload->document_id_base + local_document_id;

            if (!evoke_segment_merge_document_is_excluded(
                    excluded_document_ids,
                    excluded_document_count,
                    document_id))
            {
                merged.document_id_map[document_cursor++] = document_id;
            }
        }
        for (uint32_t version_index = 0;
             version_index < payload->version_count;
             version_index++)
        {
            const evoke_document_version_record *version =
                &payload->versions[version_index];

            if (!evoke_segment_merge_document_is_excluded(
                    excluded_document_ids,
                    excluded_document_count,
                    (uint32_t) version->document_slot))
            {
                merged.versions[version_cursor++] = *version;
            }
        }
        for (uint32_t retirement_index = 0;
             retirement_index < payload->retirement_count;
             retirement_index++)
        {
            const evoke_document_retirement_record *retirement =
                &payload->retirements[retirement_index];

            if (!evoke_segment_merge_document_is_excluded(
                    excluded_document_ids,
                    excluded_document_count,
                    (uint32_t) retirement->document_slot))
            {
                merged.retirements[retirement_cursor++] = *retirement;
            }
        }
        for (uint32_t state_index = 0;
             state_index < payload->semantic_state_count;
             state_index++)
        {
            const evoke_semantic_state_record *state =
                &payload->semantic_states[state_index];

            if (!evoke_segment_merge_document_is_excluded(
                    excluded_document_ids,
                    excluded_document_count,
                    (uint32_t) state->document_slot))
            {
                merged.semantic_states[semantic_state_cursor++] = *state;
            }
        }
    }
    if (document_cursor != total_local_document_count ||
        version_cursor != merged.version_count ||
        retirement_cursor != merged.retirement_count ||
        semantic_state_cursor != merged.semantic_state_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    if (document_cursor > 1)
    {
        qsort(
            merged.document_id_map,
            document_cursor,
            sizeof(*merged.document_id_map),
            evoke_compare_u32_ascending
        );
    }
    merged.local_document_count = 0;
    for (uint32_t source_index = 0;
         source_index < document_cursor;
         source_index++)
    {
        if (merged.local_document_count == 0 ||
            merged.document_id_map[
                merged.local_document_count - 1
            ] != merged.document_id_map[source_index])
        {
            merged.document_id_map[
                merged.local_document_count++
            ] = merged.document_id_map[source_index];
        }
    }
    if (merged.version_count > 1)
    {
        qsort(
            merged.versions,
            merged.version_count,
            sizeof(*merged.versions),
            evoke_compare_document_version
        );
    }
    for (uint32_t version_index = 0;
         version_index < merged.version_count;
         version_index++)
    {
        if (merged.versions[version_index].document_slot > UINT32_MAX ||
            (version_index > 0 &&
             merged.versions[version_index - 1].document_slot >=
                merged.versions[version_index].document_slot))
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
    }
    if (merged.retirement_count > 1)
    {
        qsort(
            merged.retirements,
            merged.retirement_count,
            sizeof(*merged.retirements),
            evoke_compare_document_retirement
        );
    }
    for (uint32_t retirement_index = 1;
         retirement_index < merged.retirement_count;
         retirement_index++)
    {
        if (merged.retirements[retirement_index - 1].document_slot >=
            merged.retirements[retirement_index].document_slot)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
    }
    if (merged.semantic_state_count > 1)
    {
        qsort(
            merged.semantic_states,
            merged.semantic_state_count,
            sizeof(*merged.semantic_states),
            evoke_compare_semantic_state
        );
    }
    semantic_state_cursor = 0;
    for (uint32_t source_index = 0;
         source_index < merged.semantic_state_count;
         source_index++)
    {
        evoke_semantic_state_record *source =
            &merged.semantic_states[source_index];

        if (semantic_state_cursor > 0 &&
            merged.semantic_states[
                semantic_state_cursor - 1
            ].document_slot == source->document_slot)
        {
            evoke_semantic_state_record *prior =
                &merged.semantic_states[semantic_state_cursor - 1];

            if (prior->transition_sequence >=
                    source->transition_sequence ||
                memcmp(
                    prior->semantic_input_fingerprint,
                    source->semantic_input_fingerprint,
                    EVOKE_DOCUMENT_FINGERPRINT_BYTES
                ) != 0)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            *prior = *source;
        }
        else
        {
            merged.semantic_states[semantic_state_cursor++] = *source;
        }
    }
    merged.semantic_state_count = semantic_state_cursor;

    while (true)
    {
        uint32_t minimum_term_id = UINT32_MAX;
        evoke_posting_extent_kind minimum_kind =
            EVOKE_POSTING_EXTENT_LEXICAL_IMPACT;
        uint64_t run_start = posting_cursor;
        bool found_run = false;

        for (uint32_t payload_index = 0;
             payload_index < payload_count;
             payload_index++)
        {
            const evoke_segment_payload *payload =
                &payloads[payload_index];
            uint32_t run_index = run_cursors[payload_index];

            if (run_index >= payload->run_count)
            {
                continue;
            }
            if (!found_run ||
                payload->runs[run_index].term_id < minimum_term_id ||
                (payload->runs[run_index].term_id == minimum_term_id &&
                 payload->runs[run_index].kind < minimum_kind))
            {
                minimum_term_id = payload->runs[run_index].term_id;
                minimum_kind = payload->runs[run_index].kind;
                found_run = true;
            }
        }
        if (!found_run)
        {
            break;
        }
        for (uint32_t payload_index = 0;
             payload_index < payload_count;
             payload_index++)
        {
            const evoke_segment_payload *payload =
                &payloads[payload_index];
            uint32_t run_index = run_cursors[payload_index];
            const evoke_segment_term_run *run;

            if (run_index >= payload->run_count)
            {
                continue;
            }
            run = &payload->runs[run_index];
            if (run->term_id != minimum_term_id ||
                run->kind != minimum_kind)
            {
                continue;
            }
            for (uint64_t source_index = run->posting_offset;
                 source_index <
                    run->posting_offset + run->posting_count;
                 source_index++)
            {
                uint32_t source_local_id =
                    payload->indices[source_index];
                uint32_t global_document_id =
                    payload->document_id_map != NULL
                        ? payload->document_id_map[source_local_id]
                        : payload->document_id_base + source_local_id;
                uint32_t merged_local_id;

                if (evoke_segment_merge_document_is_excluded(
                        excluded_document_ids,
                        excluded_document_count,
                        global_document_id))
                {
                    continue;
                }
                if (!evoke_segment_merge_find_document(
                        merged.document_id_map,
                        merged.local_document_count,
                        global_document_id,
                        &merged_local_id))
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                posting_buffer[posting_cursor].document_id =
                    merged_local_id;
                posting_buffer[posting_cursor].value =
                    payload->values[source_index];
                posting_cursor++;
            }
            run_cursors[payload_index]++;
        }
        if (posting_cursor == run_start)
        {
            continue;
        }
        qsort(
            &posting_buffer[run_start],
            (size_t) (posting_cursor - run_start),
            sizeof(*posting_buffer),
            evoke_compare_merge_posting
        );
        for (uint64_t output_index = run_start;
             output_index < posting_cursor;
             output_index++)
        {
            if (output_index > run_start &&
                posting_buffer[output_index - 1].document_id >=
                    posting_buffer[output_index].document_id)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            merged.indices[output_index] =
                posting_buffer[output_index].document_id;
            merged.values[output_index] =
                posting_buffer[output_index].value;
        }
        merged.runs[output_run_count].term_id = minimum_term_id;
        merged.runs[output_run_count].kind = minimum_kind;
        merged.runs[output_run_count].posting_offset = run_start;
        merged.runs[output_run_count].posting_count =
            posting_cursor - run_start;
        output_run_count++;
    }
    if (posting_cursor != merged.posting_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    merged.run_count = output_run_count;

    free(run_cursors);
    free(posting_buffer);
    evoke_segment_payload_free(payload_out);
    *payload_out = merged;
    return EVOKE_OK;

cleanup:
    free(run_cursors);
    free(posting_buffer);
    evoke_segment_payload_free(&merged);
    return status;
}

evoke_status
evoke_segment_payload_merge(
    const evoke_segment_manifest *manifest,
    uint32_t first_segment_index,
    const evoke_segment_payload *payloads,
    uint32_t payload_count,
    uint64_t merged_segment_id,
    evoke_segment_payload *payload_out
)
{
    return evoke_segment_payload_merge_excluding(
        manifest,
        first_segment_index,
        payloads,
        payload_count,
        merged_segment_id,
        NULL,
        0,
        payload_out
    );
}

static evoke_status
evoke_segment_payload_layout_build(
    const evoke_segment_payload *payload,
    uint32_t block_count,
    uint64_t generic_posting_count,
    size_t semantic_bmp_size,
    evoke_segment_payload_layout *layout_out
)
{
    evoke_segment_payload_layout layout;
    size_t bytes;
    size_t block_bytes;

    if (payload == NULL || layout_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&layout, 0, sizeof(layout));
    layout.runs_offset = EVOKE_SEGMENT_PAYLOAD_HEADER_SIZE;
    if (!evoke_checked_mul_size(
            payload->run_count,
            EVOKE_SEGMENT_TERM_RUN_SIZE,
            &bytes) ||
        !evoke_checked_mul_size(
            block_count,
            EVOKE_POSTING_BLOCK_RECORD_SIZE,
            &block_bytes) ||
        !evoke_checked_add_size(
            layout.runs_offset,
            bytes,
            &layout.blocks_offset) ||
        !evoke_checked_add_size(
            layout.blocks_offset,
            block_bytes,
            &layout.indices_offset) ||
        generic_posting_count > SIZE_MAX ||
        !evoke_checked_mul_size(
            (size_t) generic_posting_count,
            sizeof(uint32_t),
            &bytes) ||
        !evoke_checked_add_size(
            layout.indices_offset,
            bytes,
            &layout.values_offset) ||
        !evoke_checked_add_size(
            layout.values_offset,
            bytes,
            &layout.versions_offset))
    {
        return EVOKE_ERR_RANGE;
    }
    if ((payload->flags &
         EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP) != 0)
    {
        layout.document_map_offset = layout.versions_offset;
        if (!evoke_checked_mul_size(
                payload->local_document_count,
                sizeof(uint32_t),
                &bytes) ||
            !evoke_checked_add_size(
                layout.versions_offset,
                bytes,
                &layout.versions_offset))
        {
            return EVOKE_ERR_RANGE;
        }
    }
    if (!evoke_checked_mul_size(
            payload->version_count,
            EVOKE_DOCUMENT_VERSION_RECORD_SIZE,
            &bytes) ||
        !evoke_checked_add_size(
            layout.versions_offset,
            bytes,
            &layout.retirements_offset) ||
        !evoke_checked_mul_size(
            payload->retirement_count,
            EVOKE_DOCUMENT_RETIREMENT_RECORD_SIZE,
            &bytes) ||
        !evoke_checked_add_size(
            layout.retirements_offset,
            bytes,
            &layout.semantic_states_offset) ||
        !evoke_checked_mul_size(
            payload->semantic_state_count,
            EVOKE_SEMANTIC_STATE_RECORD_SIZE,
            &bytes) ||
        !evoke_checked_add_size(
            layout.semantic_states_offset,
            bytes,
            &layout.total_size))
    {
        return EVOKE_ERR_RANGE;
    }
    if (semantic_bmp_size > 0)
    {
        layout.semantic_bmp_offset = layout.total_size;
        layout.semantic_bmp_size = semantic_bmp_size;
        if (!evoke_checked_add_size(
                layout.total_size,
                semantic_bmp_size,
                &layout.total_size))
        {
            return EVOKE_ERR_RANGE;
        }
    }
    *layout_out = layout;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_payload_generic_posting_count(
    const evoke_segment_payload *payload,
    uint64_t *count_out
)
{
    uint64_t count = 0;

    if (payload == NULL || count_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint32_t run_index = 0;
         run_index < payload->run_count;
         run_index++)
    {
        const evoke_segment_term_run *run = &payload->runs[run_index];

        if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            continue;
        }
        if (count > UINT64_MAX - run->posting_count)
        {
            return EVOKE_ERR_RANGE;
        }
        count += run->posting_count;
    }
    *count_out = count;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_payload_semantic_bmp_build(
    const evoke_segment_payload *payload,
    const evoke_segment_manifest *manifest,
    evoke_semantic_bmp_packed_index *bmp_out,
    size_t *size_out
)
{
    evoke_semantic_bmp_run *runs = NULL;
    uint32_t semantic_run_count = 0;
    uint32_t cursor = 0;
    evoke_status status;

    if (payload == NULL || manifest == NULL || bmp_out == NULL ||
        size_out == NULL || manifest->document_slot_count == 0 ||
        manifest->document_slot_count > UINT32_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    *size_out = 0;
    for (uint32_t run_index = 0;
         run_index < payload->run_count;
         run_index++)
    {
        if (payload->runs[run_index].kind ==
            EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            semantic_run_count++;
        }
    }
    if (semantic_run_count == 0)
    {
        return EVOKE_OK;
    }
    runs = calloc(semantic_run_count, sizeof(*runs));
    if (runs == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t run_index = 0;
         run_index < payload->run_count;
         run_index++)
    {
        const evoke_segment_term_run *source = &payload->runs[run_index];
        evoke_semantic_bmp_run *destination;

        if (source->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            continue;
        }
        destination = &runs[cursor++];
        destination->term_id = source->term_id;
        destination->posting_count = source->posting_count;
        destination->local_document_ids =
            &payload->indices[source->posting_offset];
        destination->values = &payload->values[source->posting_offset];
        destination->document_id_map = payload->document_id_map;
        destination->document_id_base = payload->document_id_base;
        destination->local_document_count = payload->local_document_count;
    }
    status = evoke_semantic_bmp_packed_index_build_runs_with_precision(
        (uint32_t) manifest->document_slot_count,
        runs,
        semantic_run_count,
        payload->semantic_impact_precision,
        bmp_out
    );
    free(runs);
    return status == EVOKE_OK
        ? evoke_semantic_bmp_packed_serialized_size(bmp_out, size_out)
        : status;
}

evoke_status
evoke_segment_payload_serialized_size(
    const evoke_segment_payload *payload,
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *segment,
    size_t *size_out
)
{
    evoke_persistent_block_directory directory;
    evoke_semantic_bmp_packed_index bmp;
    evoke_segment_payload_layout layout;
    size_t semantic_bmp_size = 0;
    uint64_t generic_posting_count = 0;
    evoke_status status;

    if (size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *size_out = 0;
    evoke_persistent_block_directory_init(&directory);
    evoke_semantic_bmp_packed_index_init(&bmp);
    status = evoke_segment_payload_validate(payload, manifest, segment);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_segment_payload_generic_posting_count(
        payload,
        &generic_posting_count
    );
    if (status == EVOKE_OK && generic_posting_count > UINT32_MAX)
    {
        status = EVOKE_ERR_RANGE;
    }
    if (status == EVOKE_OK)
    {
        status = evoke_segment_payload_block_directory_build(
            payload,
            false,
            &directory
        );
    }
    if (status == EVOKE_OK)
    {
        status = evoke_segment_payload_semantic_bmp_build(
            payload,
            manifest,
            &bmp,
            &semantic_bmp_size
        );
    }
    if (status == EVOKE_OK)
    {
        status = evoke_segment_payload_layout_build(
            payload,
            directory.block_count,
            generic_posting_count,
            semantic_bmp_size,
            &layout
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_persistent_block_directory_free(&directory);
        evoke_semantic_bmp_packed_index_free(&bmp);
        return status;
    }
    *size_out = layout.total_size;
    evoke_persistent_block_directory_free(&directory);
    evoke_semantic_bmp_packed_index_free(&bmp);
    return EVOKE_OK;
}

static void
evoke_serialize_document_version(
    uint8_t *bytes,
    const evoke_document_version_record *record
)
{
    evoke_write_u64_le(bytes + 0, record->document_slot);
    evoke_write_u64_le(bytes + 8, record->born_sequence);
    evoke_write_u32_le(bytes + 16, record->record_xid);
    evoke_write_u32_le(bytes + 20, record->heap_block);
    evoke_write_u32_le(bytes + 24, record->document_length);
    evoke_write_u16_le(bytes + 28, record->heap_offset);
    evoke_write_u16_le(bytes + 30, record->flags);
    memcpy(
        bytes + 32,
        record->semantic_input_fingerprint,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );
}

static void
evoke_serialize_document_retirement(
    uint8_t *bytes,
    const evoke_document_retirement_record *record
)
{
    evoke_write_u64_le(bytes + 0, record->document_slot);
    evoke_write_u64_le(bytes + 8, record->retirement_sequence);
    evoke_write_u32_le(bytes + 16, record->record_xid);
    evoke_write_u32_le(bytes + 20, record->document_length);
    evoke_write_u16_le(bytes + 24, record->flags);
    evoke_write_u16_le(bytes + 26, record->reserved);
    evoke_write_u32_le(bytes + 28, record->reserved2);
}

static void
evoke_serialize_semantic_state(
    uint8_t *bytes,
    const evoke_semantic_state_record *record
)
{
    evoke_write_u64_le(bytes + 0, record->document_slot);
    evoke_write_u64_le(bytes + 8, record->transition_sequence);
    evoke_write_u32_le(bytes + 16, record->record_xid);
    evoke_write_u32_le(bytes + 20, record->error_code);
    evoke_write_u64_le(bytes + 24, (uint64_t) record->retry_after);
    evoke_write_u16_le(bytes + 32, record->flags);
    evoke_write_u16_le(bytes + 34, record->failure_count);
    evoke_write_u32_le(bytes + 36, record->reserved);
    memcpy(
        bytes + 40,
        record->semantic_input_fingerprint,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );
    evoke_write_u64_le(bytes + 56, (uint64_t) record->pending_since);
    evoke_write_u64_le(bytes + 64, record->error_hash);
}

evoke_status
evoke_segment_payload_serialize(
    const evoke_segment_payload *payload,
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *segment,
    uint8_t **bytes_out,
    size_t *size_out,
    uint64_t *checksum_out
)
{
    evoke_persistent_block_directory directory;
    evoke_semantic_bmp_packed_index bmp;
    evoke_segment_payload_layout layout;
    uint8_t *bytes;
    size_t semantic_bmp_size = 0;
    uint64_t generic_posting_count = 0;
    uint64_t generic_cursor = 0;
    uint64_t semantic_cursor = 0;
    uint32_t run_index;
    uint32_t i;
    uint64_t posting_index;
    uint64_t checksum;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL || checksum_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    *checksum_out = 0;
    evoke_persistent_block_directory_init(&directory);
    evoke_semantic_bmp_packed_index_init(&bmp);
    status = evoke_segment_payload_validate(
        payload,
        manifest,
        segment
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_segment_payload_generic_posting_count(
        payload,
        &generic_posting_count
    );
    if (status == EVOKE_OK && generic_posting_count > UINT32_MAX)
    {
        status = EVOKE_ERR_RANGE;
    }
    if (status == EVOKE_OK)
    {
        status = evoke_segment_payload_block_directory_build(
            payload,
            false,
            &directory
        );
    }
    if (status == EVOKE_OK)
    {
        status = evoke_segment_payload_semantic_bmp_build(
            payload,
            manifest,
            &bmp,
            &semantic_bmp_size
        );
    }
    if (status == EVOKE_OK)
    {
        status = evoke_segment_payload_layout_build(
            payload,
            directory.block_count,
            generic_posting_count,
            semantic_bmp_size,
            &layout
        );
    }
    if (status != EVOKE_OK)
    {
        evoke_persistent_block_directory_free(&directory);
        evoke_semantic_bmp_packed_index_free(&bmp);
        return status;
    }
    *size_out = layout.total_size;
    bytes = calloc(layout.total_size, 1);
    if (bytes == NULL)
    {
        evoke_persistent_block_directory_free(&directory);
        evoke_semantic_bmp_packed_index_free(&bmp);
        return EVOKE_ERR_NOMEM;
    }

    evoke_write_u32_le(bytes + 0, EVOKE_SEGMENT_PAYLOAD_MAGIC);
    evoke_write_u16_le(bytes + 4, EVOKE_SEGMENT_PAYLOAD_VERSION);
    evoke_write_u16_le(bytes + 6, EVOKE_SEGMENT_PAYLOAD_HEADER_SIZE);
    evoke_write_u16_le(bytes + 8, EVOKE_SEGMENT_TERM_RUN_SIZE);
    evoke_write_u16_le(bytes + 10, EVOKE_DOCUMENT_VERSION_RECORD_SIZE);
    evoke_write_u16_le(
        bytes + 12,
        EVOKE_DOCUMENT_RETIREMENT_RECORD_SIZE
    );
    evoke_write_u16_le(bytes + 14, EVOKE_SEMANTIC_STATE_RECORD_SIZE);
    evoke_write_u32_le(bytes + 16, payload->flags);
    evoke_write_u32_le(bytes + 20, payload->vocab_size);
    evoke_write_u32_le(bytes + 24, payload->local_document_count);
    evoke_write_u32_le(bytes + 28, payload->run_count);
    evoke_write_u64_le(bytes + 32, payload->segment_id);
    evoke_write_u64_le(bytes + 40, payload->posting_count);
    evoke_write_u32_le(bytes + 48, payload->version_count);
    evoke_write_u32_le(bytes + 52, payload->retirement_count);
    evoke_write_u32_le(bytes + 56, payload->document_id_base);
    evoke_write_u32_le(bytes + 60, payload->semantic_state_count);
    evoke_write_u64_le(bytes + 64, layout.runs_offset);
    evoke_write_u64_le(bytes + 72, layout.blocks_offset);
    evoke_write_u64_le(bytes + 80, layout.indices_offset);
    evoke_write_u64_le(bytes + 88, layout.values_offset);
    evoke_write_u64_le(bytes + 96, layout.document_map_offset);
    evoke_write_u64_le(bytes + 104, layout.versions_offset);
    evoke_write_u64_le(bytes + 112, layout.retirements_offset);
    evoke_write_u64_le(bytes + 120, layout.semantic_states_offset);
    evoke_write_u64_le(bytes + 128, layout.total_size);
    evoke_write_u32_le(bytes + 136, directory.block_count);
    evoke_write_u32_le(bytes + 140, directory.block_shift);
    evoke_write_u16_le(bytes + 144, EVOKE_POSTING_BLOCK_RECORD_SIZE);
    evoke_write_u16_le(
        bytes + 146,
        semantic_bmp_size > 0
            ? EVOKE_SEMANTIC_BMP_PACKED_FORMAT_VERSION
            : 0
    );
    evoke_write_u32_le(bytes + 148, (uint32_t) generic_posting_count);
    evoke_write_u64_le(bytes + 152, layout.semantic_bmp_offset);
    evoke_write_u64_le(bytes + 160, layout.semantic_bmp_size);

    for (run_index = 0; run_index < payload->run_count; run_index++)
    {
        const evoke_segment_term_run *run = &payload->runs[run_index];
        uint8_t *destination =
            bytes + layout.runs_offset +
            (size_t) run_index * EVOKE_SEGMENT_TERM_RUN_SIZE;
        uint64_t physical_offset =
            run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
                ? semantic_cursor
                : generic_cursor;

        evoke_write_u32_le(destination + 0, run->term_id);
        evoke_write_u32_le(destination + 4, (uint32_t) run->kind);
        evoke_write_u64_le(destination + 8, physical_offset);
        evoke_write_u64_le(destination + 16, run->posting_count);
        evoke_write_u64_le(
            destination + 24,
            directory.run_block_offsets[run_index]
        );
        evoke_write_u32_le(
            destination + 32,
            directory.run_block_counts[run_index]
        );
        evoke_write_u32_le(destination + 36, 0);
        if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            semantic_cursor += run->posting_count;
            continue;
        }
        for (posting_index = 0;
             posting_index < run->posting_count;
             posting_index++)
        {
            uint64_t source_index = run->posting_offset + posting_index;
            uint64_t target_index = physical_offset + posting_index;
            uint32_t bits;

            if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                bits = payload->values[source_index].term_frequency;
            }
            else
            {
                memcpy(
                    &bits,
                    &payload->values[source_index].impact,
                    sizeof(bits)
                );
            }
            evoke_write_u32_le(
                bytes + layout.indices_offset +
                    (size_t) target_index * sizeof(uint32_t),
                payload->indices[source_index]
            );
            evoke_write_u32_le(
                bytes + layout.values_offset +
                    (size_t) target_index * sizeof(uint32_t),
                bits
            );
        }
        generic_cursor += run->posting_count;
    }
    if (generic_cursor != generic_posting_count ||
        semantic_cursor != payload->posting_count - generic_posting_count)
    {
        free(bytes);
        evoke_persistent_block_directory_free(&directory);
        evoke_semantic_bmp_packed_index_free(&bmp);
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t block_index = 0;
         block_index < directory.block_count;
         block_index++)
    {
        evoke_serialize_posting_block_record(
            bytes + layout.blocks_offset +
                (size_t) block_index *
                    EVOKE_POSTING_BLOCK_RECORD_SIZE,
            &directory.blocks[block_index]
        );
    }
    if (layout.document_map_offset != 0)
    {
        for (i = 0; i < payload->local_document_count; i++)
        {
            evoke_write_u32_le(
                bytes + layout.document_map_offset +
                    (size_t) i * sizeof(uint32_t),
                payload->document_id_map[i]
            );
        }
    }
    for (i = 0; i < payload->version_count; i++)
    {
        evoke_serialize_document_version(
            bytes + layout.versions_offset +
                (size_t) i * EVOKE_DOCUMENT_VERSION_RECORD_SIZE,
            &payload->versions[i]
        );
    }
    for (i = 0; i < payload->retirement_count; i++)
    {
        evoke_serialize_document_retirement(
            bytes + layout.retirements_offset +
                (size_t) i * EVOKE_DOCUMENT_RETIREMENT_RECORD_SIZE,
            &payload->retirements[i]
        );
    }
    for (i = 0; i < payload->semantic_state_count; i++)
    {
        evoke_serialize_semantic_state(
            bytes + layout.semantic_states_offset +
                (size_t) i * EVOKE_SEMANTIC_STATE_RECORD_SIZE,
            &payload->semantic_states[i]
        );
    }
    if (semantic_bmp_size > 0)
    {
        status = evoke_semantic_bmp_packed_serialize_into(
            &bmp,
            bytes + layout.semantic_bmp_offset,
            layout.semantic_bmp_size
        );
        if (status != EVOKE_OK)
        {
            free(bytes);
            evoke_persistent_block_directory_free(&directory);
            evoke_semantic_bmp_packed_index_free(&bmp);
            return status;
        }
    }
    checksum = evoke_checksum_with_zero_range(
        bytes,
        layout.total_size,
        EVOKE_SEGMENT_PAYLOAD_CHECKSUM_OFFSET,
        sizeof(uint64_t)
    );
    evoke_write_u64_le(
        bytes + EVOKE_SEGMENT_PAYLOAD_CHECKSUM_OFFSET,
        checksum
    );

    *bytes_out = bytes;
    *size_out = layout.total_size;
    *checksum_out = evoke_segment_blob_checksum(bytes, layout.total_size);
    evoke_persistent_block_directory_free(&directory);
    evoke_semantic_bmp_packed_index_free(&bmp);
    return EVOKE_OK;
}

static void
evoke_deserialize_document_version(
    const uint8_t *bytes,
    evoke_document_version_record *record
)
{
    record->document_slot = evoke_read_u64_le(bytes + 0);
    record->born_sequence = evoke_read_u64_le(bytes + 8);
    record->record_xid = evoke_read_u32_le(bytes + 16);
    record->heap_block = evoke_read_u32_le(bytes + 20);
    record->document_length = evoke_read_u32_le(bytes + 24);
    record->heap_offset = evoke_read_u16_le(bytes + 28);
    record->flags = evoke_read_u16_le(bytes + 30);
    memcpy(
        record->semantic_input_fingerprint,
        bytes + 32,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );
}

static void
evoke_deserialize_document_retirement(
    const uint8_t *bytes,
    evoke_document_retirement_record *record
)
{
    record->document_slot = evoke_read_u64_le(bytes + 0);
    record->retirement_sequence = evoke_read_u64_le(bytes + 8);
    record->record_xid = evoke_read_u32_le(bytes + 16);
    record->document_length = evoke_read_u32_le(bytes + 20);
    record->flags = evoke_read_u16_le(bytes + 24);
    record->reserved = evoke_read_u16_le(bytes + 26);
    record->reserved2 = evoke_read_u32_le(bytes + 28);
}

static void
evoke_deserialize_semantic_state(
    const uint8_t *bytes,
    evoke_semantic_state_record *record
)
{
    record->document_slot = evoke_read_u64_le(bytes + 0);
    record->transition_sequence = evoke_read_u64_le(bytes + 8);
    record->record_xid = evoke_read_u32_le(bytes + 16);
    record->error_code = evoke_read_u32_le(bytes + 20);
    record->retry_after = (int64_t) evoke_read_u64_le(bytes + 24);
    record->flags = evoke_read_u16_le(bytes + 32);
    record->failure_count = evoke_read_u16_le(bytes + 34);
    record->reserved = evoke_read_u32_le(bytes + 36);
    memcpy(
        record->semantic_input_fingerprint,
        bytes + 40,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );
    record->pending_since = (int64_t) evoke_read_u64_le(bytes + 56);
    record->error_hash = evoke_read_u64_le(bytes + 64);
}

evoke_status
evoke_segment_payload_disk_header_decode(
    const uint8_t *bytes,
    size_t size,
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *segment,
    evoke_segment_payload_disk_header *header_out
)
{
    evoke_segment_payload shape;
    evoke_segment_payload_layout layout;
    evoke_segment_payload_disk_header header;
    evoke_status status;

    if (bytes == NULL || manifest == NULL || segment == NULL ||
        header_out == NULL || size < EVOKE_SEGMENT_PAYLOAD_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_read_u32_le(bytes + 0) != EVOKE_SEGMENT_PAYLOAD_MAGIC ||
        evoke_read_u16_le(bytes + 4) != EVOKE_SEGMENT_PAYLOAD_VERSION ||
        evoke_read_u16_le(bytes + 6) !=
            EVOKE_SEGMENT_PAYLOAD_HEADER_SIZE ||
        evoke_read_u16_le(bytes + 8) != EVOKE_SEGMENT_TERM_RUN_SIZE ||
        evoke_read_u16_le(bytes + 10) !=
            EVOKE_DOCUMENT_VERSION_RECORD_SIZE ||
        evoke_read_u16_le(bytes + 12) !=
            EVOKE_DOCUMENT_RETIREMENT_RECORD_SIZE ||
        evoke_read_u16_le(bytes + 14) !=
            EVOKE_SEMANTIC_STATE_RECORD_SIZE ||
        evoke_read_u16_le(bytes + 144) !=
            EVOKE_POSTING_BLOCK_RECORD_SIZE ||
        evoke_segment_manifest_validate(manifest) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }

    memset(&header, 0, sizeof(header));
    header.payload_version = evoke_read_u16_le(bytes + 4);
    header.semantic_bmp_version = evoke_read_u16_le(bytes + 146);
    header.flags = evoke_read_u32_le(bytes + 16);
    header.vocab_size = evoke_read_u32_le(bytes + 20);
    header.local_document_count = evoke_read_u32_le(bytes + 24);
    header.run_count = evoke_read_u32_le(bytes + 28);
    header.segment_id = evoke_read_u64_le(bytes + 32);
    header.posting_count = evoke_read_u64_le(bytes + 40);
    header.version_count = evoke_read_u32_le(bytes + 48);
    header.retirement_count = evoke_read_u32_le(bytes + 52);
    header.document_id_base = evoke_read_u32_le(bytes + 56);
    header.semantic_state_count = evoke_read_u32_le(bytes + 60);
    header.runs_offset = evoke_read_u64_le(bytes + 64);
    header.blocks_offset = evoke_read_u64_le(bytes + 72);
    header.indices_offset = evoke_read_u64_le(bytes + 80);
    header.values_offset = evoke_read_u64_le(bytes + 88);
    header.document_map_offset = evoke_read_u64_le(bytes + 96);
    header.versions_offset = evoke_read_u64_le(bytes + 104);
    header.retirements_offset = evoke_read_u64_le(bytes + 112);
    header.semantic_states_offset = evoke_read_u64_le(bytes + 120);
    header.semantic_bmp_offset = evoke_read_u64_le(bytes + 152);
    header.semantic_bmp_size = evoke_read_u64_le(bytes + 160);
    header.total_size = evoke_read_u64_le(bytes + 128);
    header.block_count = evoke_read_u32_le(bytes + 136);
    header.block_shift = evoke_read_u32_le(bytes + 140);
    header.generic_posting_count = evoke_read_u32_le(bytes + 148);
    header.internal_checksum = evoke_read_u64_le(
        bytes + EVOKE_SEGMENT_PAYLOAD_CHECKSUM_OFFSET
    );
    if ((header.flags & ~EVOKE_SEGMENT_PAYLOAD_KNOWN_FLAGS) != 0 ||
        header.vocab_size > manifest->vocab_size ||
        header.segment_id != segment->segment_id ||
        header.posting_count != segment->posting_count ||
        header.version_count != segment->document_count ||
        header.retirement_count != segment->retirement_count ||
        header.semantic_state_count != segment->semantic_state_count ||
        header.local_document_count != segment->document_slot_count ||
        header.total_size != segment->payload_bytes ||
        header.internal_checksum == 0 ||
        (uint64_t) (size_t) header.semantic_bmp_size !=
            header.semantic_bmp_size ||
        header.generic_posting_count > header.posting_count ||
        ((header.generic_posting_count == 0) !=
             (header.block_count == 0)) ||
        header.block_shift == 0 || header.block_shift >= 32 ||
        ((header.semantic_bmp_size == 0 &&
          (header.semantic_bmp_version != 0 ||
           header.semantic_bmp_offset != 0 ||
           header.generic_posting_count != header.posting_count)) ||
         (header.semantic_bmp_size > 0 &&
          (header.semantic_bmp_version !=
               EVOKE_SEMANTIC_BMP_PACKED_FORMAT_VERSION ||
           header.semantic_bmp_offset == 0 ||
           header.generic_posting_count == header.posting_count))) ||
        ((header.flags & EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP) == 0 &&
         ((uint64_t) header.document_id_base +
              header.local_document_count >
              manifest->document_slot_count ||
          header.document_id_base != segment->first_document_slot)) ||
        ((header.flags & EVOKE_SEGMENT_PAYLOAD_FLAG_DOCUMENT_MAP) != 0 &&
         header.document_id_base != 0))
    {
        return EVOKE_ERR_FORMAT;
    }

    evoke_segment_payload_init(&shape);
    shape.flags = header.flags;
    shape.local_document_count = header.local_document_count;
    shape.run_count = header.run_count;
    shape.posting_count = header.posting_count;
    shape.version_count = header.version_count;
    shape.retirement_count = header.retirement_count;
    shape.semantic_state_count = header.semantic_state_count;
    status = evoke_segment_payload_layout_build(
        &shape,
        header.block_count,
        header.generic_posting_count,
        (size_t) header.semantic_bmp_size,
        &layout
    );
    if (status != EVOKE_OK ||
        header.runs_offset != layout.runs_offset ||
        header.blocks_offset != layout.blocks_offset ||
        header.indices_offset != layout.indices_offset ||
        header.values_offset != layout.values_offset ||
        header.document_map_offset != layout.document_map_offset ||
        header.versions_offset != layout.versions_offset ||
        header.retirements_offset != layout.retirements_offset ||
        header.semantic_states_offset !=
            layout.semantic_states_offset ||
        header.semantic_bmp_offset != layout.semantic_bmp_offset ||
        header.semantic_bmp_size != layout.semantic_bmp_size ||
        header.total_size != layout.total_size)
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    *header_out = header;
    return EVOKE_OK;
}

evoke_status
evoke_segment_term_run_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_segment_term_run *run_out
)
{
    evoke_segment_term_run run;

    if (bytes == NULL || run_out == NULL ||
        size < EVOKE_SEGMENT_TERM_RUN_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_read_u32_le(bytes + 36) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&run, 0, sizeof(run));
    run.term_id = evoke_read_u32_le(bytes + 0);
    run.kind =
        (evoke_posting_extent_kind) evoke_read_u32_le(bytes + 4);
    run.posting_offset = evoke_read_u64_le(bytes + 8);
    run.posting_count = evoke_read_u64_le(bytes + 16);
    run.block_offset = evoke_read_u64_le(bytes + 24);
    run.block_count = evoke_read_u32_le(bytes + 32);
    if (run.kind < EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL ||
        run.kind > EVOKE_POSTING_EXTENT_LEXICAL_IMPACT ||
        run.posting_count == 0 ||
        (run.kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT
            ? run.block_count != 0
            : run.block_count == 0) ||
        run.posting_offset > UINT64_MAX - run.posting_count ||
        run.block_offset > UINT64_MAX - run.block_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    *run_out = run;
    return EVOKE_OK;
}

evoke_status
evoke_segment_payload_deserialize(
    const uint8_t *bytes,
    size_t size,
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *segment,
    evoke_segment_payload *payload_out
)
{
    evoke_segment_payload payload;
    evoke_segment_payload_disk_header header;
    evoke_segment_payload_layout layout;
    evoke_persistent_block_directory directory;
    evoke_semantic_bmp_packed_index semantic_bmp;
    evoke_segment_term_run *disk_runs = NULL;
    evoke_posting_block_record *disk_blocks = NULL;
    uint64_t checksum;
    uint64_t logical_cursor = 0;
    uint64_t generic_cursor = 0;
    uint64_t semantic_cursor = 0;
    uint32_t disk_block_cursor = 0;
    uint32_t semantic_run_index = 0;
    evoke_status status;

    if (bytes == NULL || manifest == NULL || segment == NULL ||
        payload_out == NULL || size < EVOKE_SEGMENT_PAYLOAD_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_payload_disk_header_decode(
        bytes,
        size,
        manifest,
        segment,
        &header
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    checksum = evoke_read_u64_le(
        bytes + EVOKE_SEGMENT_PAYLOAD_CHECKSUM_OFFSET
    );
    if (checksum == 0 ||
        checksum != evoke_checksum_with_zero_range(
            bytes,
            size,
            EVOKE_SEGMENT_PAYLOAD_CHECKSUM_OFFSET,
            sizeof(uint64_t)) ||
        segment->payload_checksum !=
            evoke_segment_blob_checksum(bytes, size) ||
        segment->payload_bytes != size)
    {
        return EVOKE_ERR_FORMAT;
    }

    evoke_segment_payload_init(&payload);
    evoke_persistent_block_directory_init(&directory);
    evoke_semantic_bmp_packed_index_init(&semantic_bmp);
    payload.flags = header.flags;
    payload.vocab_size = header.vocab_size;
    payload.local_document_count = header.local_document_count;
    payload.run_count = header.run_count;
    payload.segment_id = header.segment_id;
    payload.posting_count = header.posting_count;
    payload.version_count = header.version_count;
    payload.retirement_count = header.retirement_count;
    payload.document_id_base = header.document_id_base;
    payload.semantic_state_count = header.semantic_state_count;
    payload.block_shift = header.block_shift;
    if (payload.block_shift == 0 || payload.block_shift >= 32)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_segment_payload_layout_build(
        &payload,
        header.block_count,
        header.generic_posting_count,
        (size_t) header.semantic_bmp_size,
        &layout
    );
    if (status != EVOKE_OK ||
        evoke_read_u64_le(bytes + 64) != layout.runs_offset ||
        evoke_read_u64_le(bytes + 72) != layout.blocks_offset ||
        evoke_read_u64_le(bytes + 80) != layout.indices_offset ||
        evoke_read_u64_le(bytes + 88) != layout.values_offset ||
        evoke_read_u64_le(bytes + 96) != layout.document_map_offset ||
        evoke_read_u64_le(bytes + 104) != layout.versions_offset ||
        evoke_read_u64_le(bytes + 112) != layout.retirements_offset ||
        evoke_read_u64_le(bytes + 120) !=
            layout.semantic_states_offset ||
        header.semantic_bmp_offset != layout.semantic_bmp_offset ||
        header.semantic_bmp_size != layout.semantic_bmp_size ||
        evoke_read_u64_le(bytes + 128) != layout.total_size ||
        layout.total_size != size)
    {
        return EVOKE_ERR_FORMAT;
    }

    if (payload.run_count > 0)
    {
        payload.runs = calloc(payload.run_count, sizeof(*payload.runs));
        disk_runs = calloc(payload.run_count, sizeof(*disk_runs));
    }
    if (header.block_count > 0)
    {
        disk_blocks = calloc(
            header.block_count,
            sizeof(*disk_blocks)
        );
    }
    if (payload.posting_count > 0)
    {
        if (payload.posting_count > SIZE_MAX / sizeof(*payload.indices))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        payload.indices = calloc(
            (size_t) payload.posting_count,
            sizeof(*payload.indices)
        );
        payload.values = calloc(
            (size_t) payload.posting_count,
            sizeof(*payload.values)
        );
    }
    if (layout.document_map_offset != 0)
    {
        payload.document_id_map = calloc(
            payload.local_document_count,
            sizeof(*payload.document_id_map)
        );
    }
    if (payload.version_count > 0)
    {
        payload.versions = calloc(
            payload.version_count,
            sizeof(*payload.versions)
        );
    }
    if (payload.retirement_count > 0)
    {
        payload.retirements = calloc(
            payload.retirement_count,
            sizeof(*payload.retirements)
        );
    }
    if (payload.semantic_state_count > 0)
    {
        payload.semantic_states = calloc(
            payload.semantic_state_count,
            sizeof(*payload.semantic_states)
        );
    }
    if ((payload.run_count > 0 &&
         (payload.runs == NULL || disk_runs == NULL)) ||
        (header.block_count > 0 && disk_blocks == NULL) ||
        (payload.posting_count > 0 &&
         (payload.indices == NULL || payload.values == NULL)) ||
        (layout.document_map_offset != 0 &&
         payload.document_id_map == NULL) ||
        (payload.version_count > 0 && payload.versions == NULL) ||
        (payload.retirement_count > 0 &&
         payload.retirements == NULL) ||
        (payload.semantic_state_count > 0 &&
         payload.semantic_states == NULL))
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }

    if (layout.document_map_offset != 0)
    {
        for (uint32_t i = 0; i < payload.local_document_count; i++)
        {
            payload.document_id_map[i] = evoke_read_u32_le(
                bytes + layout.document_map_offset +
                    (size_t) i * sizeof(uint32_t)
            );
        }
    }
    if (header.semantic_bmp_size > 0)
    {
        status = evoke_semantic_bmp_packed_deserialize(
            bytes + layout.semantic_bmp_offset,
            layout.semantic_bmp_size,
            &semantic_bmp
        );
        if (status != EVOKE_OK ||
            semantic_bmp.document_count >
                manifest->document_slot_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        payload.semantic_impact_precision =
            semantic_bmp.impact_precision;
    }

    for (uint32_t run_index = 0;
         run_index < payload.run_count;
         run_index++)
    {
        evoke_segment_term_run *run = &payload.runs[run_index];
        evoke_segment_term_run *disk_run = &disk_runs[run_index];
        const uint8_t *source = bytes + layout.runs_offset +
            (size_t) run_index * EVOKE_SEGMENT_TERM_RUN_SIZE;

        status = evoke_segment_term_run_decode(
            source,
            EVOKE_SEGMENT_TERM_RUN_SIZE,
            disk_run
        );
        if (status != EVOKE_OK ||
            disk_run->block_offset != disk_block_cursor ||
            disk_run->block_count >
                header.block_count - disk_block_cursor)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        run->term_id = disk_run->term_id;
        run->kind = disk_run->kind;
        run->posting_offset = logical_cursor;
        run->posting_count = disk_run->posting_count;
        if (run->posting_count > payload.posting_count - logical_cursor)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            const evoke_semantic_bmp_packed_term *term;

            if (disk_run->posting_offset != semantic_cursor ||
                semantic_run_index >= semantic_bmp.term_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            term = &semantic_bmp.terms[semantic_run_index];
            if (term->term_id != run->term_id ||
                term->posting_count != run->posting_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            status = evoke_semantic_bmp_packed_term_materialize(
                &semantic_bmp,
                semantic_run_index,
                &payload.indices[logical_cursor],
                &payload.values[logical_cursor],
                (size_t) run->posting_count
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            for (uint64_t posting_index = 0;
                 posting_index < run->posting_count;
                 posting_index++)
            {
                uint64_t target_index = logical_cursor + posting_index;
                uint32_t global_document_id =
                    payload.indices[target_index];
                uint32_t local_document_id;

                if (payload.document_id_map != NULL)
                {
                    if (!evoke_segment_merge_find_document(
                            payload.document_id_map,
                            payload.local_document_count,
                            global_document_id,
                            &local_document_id))
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto cleanup;
                    }
                }
                else
                {
                    if (global_document_id < payload.document_id_base ||
                        global_document_id - payload.document_id_base >=
                            payload.local_document_count)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto cleanup;
                    }
                    local_document_id =
                        global_document_id - payload.document_id_base;
                }
                payload.indices[target_index] = local_document_id;
            }
            semantic_cursor += run->posting_count;
            semantic_run_index++;
        }
        else
        {
            if (disk_run->posting_offset != generic_cursor ||
                run->posting_count >
                    header.generic_posting_count - generic_cursor)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            for (uint64_t posting_index = 0;
                 posting_index < run->posting_count;
                 posting_index++)
            {
                uint64_t source_index = generic_cursor + posting_index;
                uint64_t target_index = logical_cursor + posting_index;
                uint32_t bits;

                payload.indices[target_index] = evoke_read_u32_le(
                    bytes + layout.indices_offset +
                        (size_t) source_index * sizeof(uint32_t)
                );
                bits = evoke_read_u32_le(
                    bytes + layout.values_offset +
                        (size_t) source_index * sizeof(uint32_t)
                );
                if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    payload.values[target_index].term_frequency = bits;
                }
                else
                {
                    memcpy(
                        &payload.values[target_index].impact,
                        &bits,
                        sizeof(bits)
                    );
                }
            }
            generic_cursor += run->posting_count;
            disk_block_cursor += disk_run->block_count;
        }
        logical_cursor += run->posting_count;
    }
    if (logical_cursor != payload.posting_count ||
        generic_cursor != header.generic_posting_count ||
        semantic_cursor != semantic_bmp.posting_count ||
        semantic_run_index != semantic_bmp.term_count ||
        disk_block_cursor != header.block_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    for (uint32_t block_index = 0;
         block_index < header.block_count;
         block_index++)
    {
        status = evoke_posting_block_record_decode(
            bytes + layout.blocks_offset +
                (size_t) block_index *
                    EVOKE_POSTING_BLOCK_RECORD_SIZE,
            EVOKE_POSTING_BLOCK_RECORD_SIZE,
            &disk_blocks[block_index]
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    for (uint32_t run_index = 0;
         run_index < payload.run_count;
         run_index++)
    {
        const evoke_segment_term_run *run = &payload.runs[run_index];
        const evoke_segment_term_run *disk_run = &disk_runs[run_index];
        evoke_posting_extent extent;

        if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            continue;
        }
        memset(&extent, 0, sizeof(extent));
        extent.indices = &payload.indices[run->posting_offset];
        extent.values = &payload.values[run->posting_offset];
        extent.document_id_map = payload.document_id_map;
        extent.len = run->posting_count;
        extent.document_id_base = payload.document_id_base;
        extent.local_document_count = payload.local_document_count;
        extent.kind = run->kind;
        status = evoke_posting_extent_validate_block_records(
            &extent,
            payload.block_shift,
            &disk_blocks[disk_run->block_offset],
            disk_run->block_count
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    status = evoke_segment_payload_block_directory_build(
        &payload,
        true,
        &directory
    );
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    payload.blocks = directory.blocks;
    payload.block_count = directory.block_count;
    payload.block_shift = directory.block_shift;
    directory.blocks = NULL;
    directory.block_count = 0;
    for (uint32_t run_index = 0;
         run_index < payload.run_count;
         run_index++)
    {
        payload.runs[run_index].block_offset =
            directory.run_block_offsets[run_index];
        payload.runs[run_index].block_count =
            directory.run_block_counts[run_index];
    }

    for (uint32_t i = 0; i < payload.version_count; i++)
    {
        evoke_deserialize_document_version(
            bytes + layout.versions_offset +
                (size_t) i * EVOKE_DOCUMENT_VERSION_RECORD_SIZE,
            &payload.versions[i]
        );
    }
    for (uint32_t i = 0; i < payload.retirement_count; i++)
    {
        evoke_deserialize_document_retirement(
            bytes + layout.retirements_offset +
                (size_t) i * EVOKE_DOCUMENT_RETIREMENT_RECORD_SIZE,
            &payload.retirements[i]
        );
    }
    for (uint32_t i = 0; i < payload.semantic_state_count; i++)
    {
        evoke_deserialize_semantic_state(
            bytes + layout.semantic_states_offset +
                (size_t) i * EVOKE_SEMANTIC_STATE_RECORD_SIZE,
            &payload.semantic_states[i]
        );
    }
    status = evoke_segment_payload_validate(
        &payload,
        manifest,
        segment
    );
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }

    evoke_segment_payload_free(payload_out);
    *payload_out = payload;
    evoke_segment_payload_init(&payload);

cleanup:
    free(disk_runs);
    free(disk_blocks);
    evoke_persistent_block_directory_free(&directory);
    evoke_semantic_bmp_packed_index_free(&semantic_bmp);
    evoke_segment_payload_free(&payload);
    if (status != EVOKE_OK)
    {
        return status;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_segment_index_copy_vocabulary(
    const evoke_segment_query_contract *contract,
    evoke_index *index
)
{
    uint32_t term_id;

    if ((contract->flags &
         EVOKE_QUERY_CONTRACT_FLAG_VOCABULARY) == 0)
    {
        return EVOKE_OK;
    }
    if (contract->vocab == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    index->vocab = calloc(index->vocab_size, sizeof(*index->vocab));
    if (index->vocab == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (term_id = 0; term_id < index->vocab_size; term_id++)
    {
        size_t length = strlen(contract->vocab[term_id]);

        index->vocab[term_id] = malloc(length + 1);
        if (index->vocab[term_id] == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        memcpy(
            index->vocab[term_id],
            contract->vocab[term_id],
            length + 1
        );
    }
    return EVOKE_OK;
}

static uint32_t
evoke_segment_payload_global_document_id(
    const evoke_segment_payload *payload,
    uint32_t local_document_id
)
{
    if (payload->document_id_map != NULL)
    {
        return payload->document_id_map[local_document_id];
    }
    return payload->document_id_base + local_document_id;
}

evoke_status
evoke_segment_index_metadata_build_base(
    const evoke_segment_query_contract *contract,
    const evoke_segment_manifest *manifest,
    const uint32_t *doc_frequencies,
    evoke_index *index_out
)
{
    evoke_index index;
    size_t bytes;
    evoke_status status;

    if (contract == NULL || manifest == NULL || index_out == NULL ||
        (manifest->vocab_size > 0 && doc_frequencies == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_segment_manifest_validate(manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_segment_query_contract_validate(contract, manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }

    evoke_index_init(&index);
    index.params = contract->params;
    index.num_docs = (uint32_t) manifest->document_slot_count;
    index.vocab_size = manifest->vocab_size;
    index.has_empty_token =
        (contract->flags & EVOKE_QUERY_CONTRACT_FLAG_EMPTY_TOKEN) != 0;
    index.empty_token_id = contract->empty_token_id;

    if (!evoke_checked_mul_size(
            index.num_docs,
            sizeof(*index.doc_lengths),
            &bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    if (index.num_docs > 0)
    {
        index.doc_lengths = calloc(
            index.num_docs,
            sizeof(*index.doc_lengths)
        );
        if (index.doc_lengths == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto fail;
        }
    }
    if (!evoke_checked_mul_size(
            index.vocab_size,
            sizeof(*index.doc_frequencies),
            &bytes))
    {
        status = EVOKE_ERR_RANGE;
        goto fail;
    }
    if (index.vocab_size > 0)
    {
        index.doc_frequencies = malloc(bytes);
        if (index.doc_frequencies == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto fail;
        }
        memcpy(
            index.doc_frequencies,
            doc_frequencies,
            bytes
        );
    }
    status = evoke_segment_index_copy_vocabulary(contract, &index);
    if (status != EVOKE_OK)
    {
        goto fail;
    }

    evoke_index_free(index_out);
    *index_out = index;
    return EVOKE_OK;

fail:
    evoke_index_free(&index);
    return status;
}

evoke_status
evoke_segment_payload_document_references_validate(
    const evoke_segment_manifest *manifest,
    const evoke_segment_payload *payloads,
    size_t payload_count,
    const uint8_t *available_document_slots,
    size_t document_slot_count
)
{
    size_t payload_index;
    evoke_status status;

    if (manifest == NULL ||
        payload_count != manifest->segment_count ||
        (payload_count > 0 && payloads == NULL) ||
        document_slot_count != manifest->document_slot_count ||
        (document_slot_count > 0 &&
         available_document_slots == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    for (payload_index = 0;
         payload_index < payload_count;
         payload_index++)
    {
        const evoke_segment_payload *payload =
            &payloads[payload_index];

        status = evoke_segment_payload_validate(
            payload,
            manifest,
            &manifest->segments[payload_index]
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }

    for (payload_index = 0;
         payload_index < payload_count;
         payload_index++)
    {
        const evoke_segment_payload *payload =
            &payloads[payload_index];
        uint64_t posting_index;

        for (posting_index = 0;
             posting_index < payload->posting_count;
             posting_index++)
        {
            uint32_t document_id =
                evoke_segment_payload_global_document_id(
                    payload,
                    payload->indices[posting_index]
                );

            if (available_document_slots[document_id] != 1)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    return EVOKE_OK;
}

evoke_status
evoke_segment_index_metadata_build(
    const evoke_segment_query_contract *contract,
    const evoke_segment_manifest *manifest,
    const uint32_t *doc_frequencies,
    const evoke_segment_payload *payloads,
    size_t payload_count,
    evoke_index *index_out
)
{
    enum
    {
        EVOKE_SLOT_MISSING = 0,
        EVOKE_SLOT_LIVE = 1,
        EVOKE_SLOT_ABORTED_HOLE = 2
    };
    evoke_index index;
    uint8_t *slot_states = NULL;
    size_t payload_index;
    evoke_status status;

    if (contract == NULL || manifest == NULL || index_out == NULL ||
        payload_count != manifest->segment_count ||
        (payload_count > 0 && payloads == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_index_init(&index);
    status = evoke_segment_index_metadata_build_base(
        contract,
        manifest,
        doc_frequencies,
        &index
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (index.num_docs > 0)
    {
        slot_states = calloc(index.num_docs, sizeof(*slot_states));
        if (slot_states == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto fail;
        }
    }

    for (payload_index = 0;
         payload_index < payload_count;
         payload_index++)
    {
        const evoke_segment_payload *payload =
            &payloads[payload_index];

        status = evoke_segment_payload_validate(
            payload,
            manifest,
            &manifest->segments[payload_index]
        );
        if (status != EVOKE_OK)
        {
            goto fail;
        }
        for (size_t version_index = 0;
             version_index < payload->version_count;
             version_index++)
        {
            const evoke_document_version_record *version =
                &payload->versions[version_index];
            uint32_t document_id = (uint32_t) version->document_slot;

            if (slot_states[document_id] != EVOKE_SLOT_MISSING)
            {
                status = EVOKE_ERR_FORMAT;
                goto fail;
            }
            slot_states[document_id] =
                (version->flags &
                 EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) != 0
                ? EVOKE_SLOT_ABORTED_HOLE
                : EVOKE_SLOT_LIVE;
            index.doc_lengths[document_id] = version->document_length;
        }
    }
    for (payload_index = 0;
         payload_index < index.num_docs;
         payload_index++)
    {
        if (slot_states[payload_index] == EVOKE_SLOT_MISSING)
        {
            status = EVOKE_ERR_FORMAT;
            goto fail;
        }
    }
    status = evoke_segment_payload_document_references_validate(
        manifest,
        payloads,
        payload_count,
        slot_states,
        index.num_docs
    );
    if (status != EVOKE_OK)
    {
        goto fail;
    }

    free(slot_states);
    evoke_index_free(index_out);
    *index_out = index;
    return EVOKE_OK;

fail:
    free(slot_states);
    evoke_index_free(&index);
    return status;
}

void
evoke_segment_read_view_init(evoke_segment_read_view *view)
{
    if (view != NULL)
    {
        memset(view, 0, sizeof(*view));
    }
}

void
evoke_segment_read_view_free(evoke_segment_read_view *view)
{
    if (view == NULL)
    {
        return;
    }
    free(view->terms);
    free(view->extents);
    memset(view, 0, sizeof(*view));
}

static evoke_status
evoke_segment_payload_attach_validate(
    const evoke_index *global_index,
    const evoke_segment_manifest *manifest,
    const evoke_segment_descriptor *descriptor,
    const evoke_segment_payload_view *payload
)
{
    evoke_status status;

    if (global_index == NULL || manifest == NULL ||
        descriptor == NULL || payload == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_segment_posting_payload_validate(
        manifest,
        descriptor,
        payload
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (global_index->num_docs != manifest->document_slot_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

evoke_status
evoke_segment_read_view_build(
    const evoke_index *global_index,
    const evoke_segment_manifest *manifest,
    const evoke_term_directory *directory,
    const evoke_segment_payload_view *payloads,
    size_t payload_count,
    evoke_segment_read_view *view_out
)
{
    evoke_segment_read_view view;
    uint64_t *next_posting_offsets = NULL;
    uint32_t *next_run_indices = NULL;
    uint32_t term_id;
    uint32_t segment_index;
    evoke_status status;

    if (global_index == NULL || manifest == NULL || directory == NULL ||
        view_out == NULL ||
        (payload_count > 0 && payloads == NULL) ||
        payload_count != manifest->segment_count ||
        global_index->vocab_size != manifest->vocab_size ||
        global_index->num_docs != manifest->document_slot_count)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_term_directory_validate(directory, manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }

    evoke_segment_read_view_init(&view);
    view.vocab_size = directory->vocab_size;
    view.extent_count = directory->extent_count;
    if (view.vocab_size > 0)
    {
        view.terms = calloc(view.vocab_size, sizeof(*view.terms));
        if (view.terms == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
    }
    if (view.extent_count > 0)
    {
        view.extents = calloc(view.extent_count, sizeof(*view.extents));
        if (view.extents == NULL)
        {
            evoke_segment_read_view_free(&view);
            return EVOKE_ERR_NOMEM;
        }
    }
    if (manifest->segment_count > 0)
    {
        next_posting_offsets = calloc(
            manifest->segment_count,
            sizeof(*next_posting_offsets)
        );
        next_run_indices = calloc(
            manifest->segment_count,
            sizeof(*next_run_indices)
        );
        if (next_posting_offsets == NULL || next_run_indices == NULL)
        {
            free(next_posting_offsets);
            free(next_run_indices);
            evoke_segment_read_view_free(&view);
            return EVOKE_ERR_NOMEM;
        }
    }

    for (segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        status = evoke_segment_payload_attach_validate(
            global_index,
            manifest,
            &manifest->segments[segment_index],
            &payloads[segment_index]
        );
        if (status != EVOKE_OK)
        {
            free(next_posting_offsets);
            free(next_run_indices);
            evoke_segment_read_view_free(&view);
            return status;
        }
    }

    for (term_id = 0; term_id < directory->vocab_size; term_id++)
    {
        uint64_t start = directory->term_offsets[term_id];
        uint64_t end = directory->term_offsets[term_id + 1];
        uint64_t extent_index;

        if (end > start)
        {
            view.terms[term_id].extents = &view.extents[start];
            view.terms[term_id].len = (size_t) (end - start);
        }
        for (extent_index = start; extent_index < end; extent_index++)
        {
            const evoke_term_extent_descriptor *descriptor =
                &directory->extents[extent_index];
            const evoke_segment_payload_view *payload =
                &payloads[descriptor->segment_index];
            uint32_t run_index =
                next_run_indices[descriptor->segment_index];
            const evoke_segment_term_run *run;
            evoke_posting_extent *extent = &view.extents[extent_index];

            if (run_index >= payload->run_count)
            {
                free(next_posting_offsets);
                free(next_run_indices);
                evoke_segment_read_view_free(&view);
                return EVOKE_ERR_FORMAT;
            }
            run = &payload->runs[run_index];
            if (run->term_id != term_id ||
                run->kind != descriptor->kind ||
                run->posting_offset != descriptor->posting_offset ||
                run->posting_count != descriptor->posting_count ||
                descriptor->posting_offset !=
                    next_posting_offsets[descriptor->segment_index])
            {
                free(next_posting_offsets);
                free(next_run_indices);
                evoke_segment_read_view_free(&view);
                return EVOKE_ERR_FORMAT;
            }
            extent->data = payload->data == NULL ? NULL :
                &payload->data[descriptor->posting_offset];
            extent->indices =
                &payload->indices[descriptor->posting_offset];
            extent->term_frequencies =
                payload->term_frequencies == NULL ? NULL :
                &payload->term_frequencies[descriptor->posting_offset];
            extent->values = payload->values == NULL ? NULL :
                &payload->values[descriptor->posting_offset];
            extent->document_id_map = payload->document_id_map;
            if (payload->blocks != NULL && run->block_count > 0)
            {
                extent->blocks =
                    &payload->blocks[run->block_offset];
                extent->block_count = run->block_count;
                extent->block_shift = payload->block_shift;
            }
            extent->len = descriptor->posting_count;
            extent->document_id_base = payload->document_id_base;
            extent->local_document_count =
                payload->local_document_count;
            extent->kind = descriptor->kind;

            if ((extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                 extent->term_frequencies == NULL &&
                 extent->values == NULL) ||
                (extent->kind != EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                 extent->data == NULL &&
                 extent->values == NULL))
            {
                free(next_posting_offsets);
                free(next_run_indices);
                evoke_segment_read_view_free(&view);
                return EVOKE_ERR_FORMAT;
            }
            status = evoke_posting_extent_validate_layout(
                global_index,
                extent
            );
            if (status != EVOKE_OK)
            {
                free(next_posting_offsets);
                free(next_run_indices);
                evoke_segment_read_view_free(&view);
                return status;
            }
            next_posting_offsets[descriptor->segment_index] +=
                descriptor->posting_count;
            next_run_indices[descriptor->segment_index]++;
        }
    }

    for (segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        if (next_posting_offsets[segment_index] !=
                payloads[segment_index].posting_count ||
            next_run_indices[segment_index] !=
                payloads[segment_index].run_count)
        {
            free(next_posting_offsets);
            free(next_run_indices);
            evoke_segment_read_view_free(&view);
            return EVOKE_ERR_FORMAT;
        }
    }

    free(next_posting_offsets);
    free(next_run_indices);
    evoke_segment_read_view_free(view_out);
    *view_out = view;
    return EVOKE_OK;
}

static evoke_status
evoke_segment_attach_payload_extent(
    const evoke_index *global_index,
    const evoke_segment_payload_view *payload,
    const evoke_segment_term_run *run,
    evoke_posting_extent *extent
)
{
    memset(extent, 0, sizeof(*extent));
    extent->data = payload->data == NULL ? NULL :
        &payload->data[run->posting_offset];
    extent->indices = &payload->indices[run->posting_offset];
    extent->term_frequencies =
        payload->term_frequencies == NULL ? NULL :
        &payload->term_frequencies[run->posting_offset];
    extent->values = payload->values == NULL ? NULL :
        &payload->values[run->posting_offset];
    extent->document_id_map = payload->document_id_map;
    if (payload->blocks != NULL && run->block_count > 0)
    {
        extent->blocks = &payload->blocks[run->block_offset];
        extent->block_count = run->block_count;
        extent->block_shift = payload->block_shift;
    }
    extent->len = run->posting_count;
    extent->document_id_base = payload->document_id_base;
    extent->local_document_count = payload->local_document_count;
    extent->kind = run->kind;

    if ((extent->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
         extent->term_frequencies == NULL &&
         extent->values == NULL) ||
        (extent->kind != EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
         extent->data == NULL &&
         extent->values == NULL))
    {
        return EVOKE_ERR_FORMAT;
    }
    return evoke_posting_extent_validate_layout(global_index, extent);
}

static evoke_status
evoke_segment_attach_fold_extent(
    const evoke_index *global_index,
    const evoke_term_fold_bundle *bundle,
    const evoke_term_fold_run *run,
    evoke_posting_extent *extent
)
{
    memset(extent, 0, sizeof(*extent));
    extent->indices = &bundle->document_slots[run->posting_offset];
    extent->values = &bundle->values[run->posting_offset];
    if (bundle->blocks != NULL && run->block_count > 0)
    {
        extent->blocks = &bundle->blocks[run->block_offset];
        extent->block_count = run->block_count;
        extent->block_shift = bundle->block_shift;
    }
    extent->len = run->posting_count;
    extent->local_document_count = global_index->num_docs;
    extent->kind = run->kind;
    return evoke_posting_extent_validate_layout(global_index, extent);
}

static bool
evoke_segment_manifest_has_sequence_boundary(
    const evoke_segment_manifest *manifest,
    uint64_t coverage
)
{
    for (uint32_t segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        if (manifest->segments[segment_index].max_sequence == coverage)
        {
            return true;
        }
    }
    return false;
}

static bool
evoke_segment_tail_directory_has_run(
    const evoke_term_directory *tail_directory,
    uint64_t tail_start,
    uint64_t tail_end,
    uint32_t segment_index,
    const evoke_segment_term_run *run
)
{
    for (uint64_t tail_index = tail_start;
         tail_index < tail_end;
         tail_index++)
    {
        const evoke_term_extent_descriptor *descriptor =
            &tail_directory->extents[tail_index];

        if (descriptor->segment_index == segment_index &&
            descriptor->kind == run->kind &&
            descriptor->posting_offset == run->posting_offset &&
            descriptor->posting_count == run->posting_count)
        {
            return true;
        }
    }
    return false;
}

static evoke_status
evoke_segment_fold_plan_mark_runs(
    const evoke_segment_manifest *manifest,
    uint32_t term_id,
    uint64_t coverage,
    uint32_t bundle_index,
    uint32_t first_run,
    uint32_t run_count,
    const evoke_term_fold_bundle *fold_bundles,
    size_t fold_bundle_count,
    evoke_segment_object_kind expected_object_kind,
    const size_t *bundle_run_offsets,
    bool *bundle_runs_used
)
{
    const evoke_term_fold_bundle *bundle;
    uint64_t run_end;

    if (run_count == 0)
    {
        return coverage == 0 && bundle_index == 0 && first_run == 0
            ? EVOKE_OK
            : EVOKE_ERR_FORMAT;
    }
    if (coverage == 0 || coverage > manifest->max_sequence ||
        bundle_index >= fold_bundle_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    bundle = &fold_bundles[bundle_index];
    run_end = (uint64_t) first_run + run_count;
    if (bundle->object_kind != expected_object_kind ||
        run_end > bundle->run_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t plan_run_index = 0;
         plan_run_index < run_count;
         plan_run_index++)
    {
        uint32_t run_index = first_run + plan_run_index;
        const evoke_term_fold_run *run = &bundle->runs[run_index];
        size_t used_index =
            bundle_run_offsets[bundle_index] + run_index;

        if (run->term_id != term_id ||
            run->coverage_sequence != coverage ||
            bundle_runs_used[used_index])
        {
            return EVOKE_ERR_FORMAT;
        }
        bundle_runs_used[used_index] = true;
    }
    return EVOKE_OK;
}

evoke_status
evoke_segment_read_view_build_folded(
    const evoke_index *global_index,
    const evoke_segment_manifest *manifest,
    const evoke_term_directory *tail_directory,
    const evoke_segment_payload_view *payloads,
    size_t payload_count,
    const evoke_term_fold_bundle *fold_bundles,
    size_t fold_bundle_count,
    const evoke_term_fold_read_plan *fold_plans,
    size_t fold_plan_count,
    evoke_segment_read_view *view_out
)
{
    evoke_segment_read_view view;
    size_t *bundle_run_offsets = NULL;
    bool *bundle_runs_used = NULL;
    uint64_t *next_posting_offsets = NULL;
    uint32_t *next_run_indices = NULL;
    uint64_t total_bundle_runs = 0;
    uint64_t total_extents;
    uint32_t extent_cursor = 0;
    evoke_status status;

    if (global_index == NULL || manifest == NULL ||
        tail_directory == NULL || view_out == NULL ||
        (payload_count > 0 && payloads == NULL) ||
        (fold_bundle_count > 0 && fold_bundles == NULL) ||
        (fold_plan_count > 0 && fold_plans == NULL) ||
        payload_count != manifest->segment_count ||
        fold_plan_count != manifest->vocab_size ||
        global_index->vocab_size != manifest->vocab_size ||
        global_index->num_docs != manifest->document_slot_count)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_term_directory_validate(tail_directory, manifest);
    if (status != EVOKE_OK)
    {
        return status;
    }

    bundle_run_offsets = calloc(
        fold_bundle_count + 1,
        sizeof(*bundle_run_offsets)
    );
    if (bundle_run_offsets == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (size_t bundle_index = 0;
         bundle_index < fold_bundle_count;
         bundle_index++)
    {
        const evoke_term_fold_bundle *bundle =
            &fold_bundles[bundle_index];

        status = evoke_term_fold_bundle_validate(bundle);
        if (status != EVOKE_OK ||
            (bundle->object_kind !=
                EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD &&
             bundle->object_kind !=
                EVOKE_SEGMENT_OBJECT_IMPACT_FOLD) ||
            bundle->owner_manifest_id > manifest->manifest_id ||
            (bundle->object_kind ==
                 EVOKE_SEGMENT_OBJECT_IMPACT_FOLD &&
             bundle->statistics_epoch !=
                manifest->statistics_epoch) ||
            total_bundle_runs > SIZE_MAX - bundle->run_count)
        {
            status = status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
            goto fail;
        }
        total_bundle_runs += bundle->run_count;
        bundle_run_offsets[bundle_index + 1] =
            (size_t) total_bundle_runs;
    }
    if (total_bundle_runs > 0)
    {
        bundle_runs_used = calloc(
            (size_t) total_bundle_runs,
            sizeof(*bundle_runs_used)
        );
        if (bundle_runs_used == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto fail;
        }
    }

    total_extents = tail_directory->extent_count;
    for (uint32_t term_id = 0;
         term_id < manifest->vocab_size;
         term_id++)
    {
        const evoke_term_fold_read_plan *plan = &fold_plans[term_id];

        status = evoke_segment_fold_plan_mark_runs(
            manifest,
            term_id,
            plan->neutral_coverage,
            plan->neutral_bundle_index,
            plan->neutral_first_run,
            plan->neutral_run_count,
            fold_bundles,
            fold_bundle_count,
            EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
            bundle_run_offsets,
            bundle_runs_used
        );
        if (status != EVOKE_OK)
        {
            goto fail;
        }
        status = evoke_segment_fold_plan_mark_runs(
            manifest,
            term_id,
            plan->neutral_minor_coverage,
            plan->neutral_minor_bundle_index,
            plan->neutral_minor_first_run,
            plan->neutral_minor_run_count,
            fold_bundles,
            fold_bundle_count,
            EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD,
            bundle_run_offsets,
            bundle_runs_used
        );
        if (status != EVOKE_OK ||
            (plan->neutral_minor_run_count > 0 &&
             (plan->neutral_run_count == 0 ||
              plan->neutral_minor_coverage <=
                plan->neutral_coverage)))
        {
            status = EVOKE_ERR_FORMAT;
            goto fail;
        }
        status = evoke_segment_fold_plan_mark_runs(
            manifest,
            term_id,
            plan->impact_coverage,
            plan->impact_bundle_index,
            plan->impact_first_run,
            plan->impact_run_count,
            fold_bundles,
            fold_bundle_count,
            EVOKE_SEGMENT_OBJECT_IMPACT_FOLD,
            bundle_run_offsets,
            bundle_runs_used
        );
        if (status != EVOKE_OK)
        {
            goto fail;
        }
        {
            uint64_t effective_coverage =
                plan->neutral_minor_run_count > 0
                    ? plan->neutral_minor_coverage
                    : plan->neutral_coverage;
            bool use_impact = plan->impact_run_count > 0;
            bool lexical_neutral_seen = false;
            uint32_t neutral_bundle_indices[2] = {
                plan->neutral_bundle_index,
                plan->neutral_minor_bundle_index
            };
            uint32_t neutral_first_runs[2] = {
                plan->neutral_first_run,
                plan->neutral_minor_first_run
            };
            uint32_t neutral_run_counts[2] = {
                plan->neutral_run_count,
                plan->neutral_minor_run_count
            };
            uint64_t selected_count = plan->impact_run_count;

            if ((!use_impact &&
                 (plan->impact_statistics_epoch != 0 ||
                  plan->impact_coverage != 0)) ||
                (use_impact &&
                 (effective_coverage == 0 ||
                  plan->impact_coverage != effective_coverage ||
                  plan->impact_statistics_epoch !=
                    manifest->statistics_epoch)))
            {
                status = EVOKE_ERR_FORMAT;
                goto fail;
            }
            for (size_t level = 0; level < 2; level++)
            {
                const evoke_term_fold_bundle *bundle;

                if (neutral_run_counts[level] == 0)
                {
                    continue;
                }
                bundle = &fold_bundles[
                    neutral_bundle_indices[level]
                ];
                for (uint32_t run_offset = 0;
                     run_offset < neutral_run_counts[level];
                     run_offset++)
                {
                    const evoke_term_fold_run *run =
                        &bundle->runs[
                            neutral_first_runs[level] + run_offset
                        ];

                    if (run->kind ==
                        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                    {
                        lexical_neutral_seen = true;
                        if (use_impact)
                        {
                            continue;
                        }
                    }
                    selected_count++;
                }
            }
            if ((use_impact && !lexical_neutral_seen) ||
                total_extents > UINT32_MAX - selected_count)
            {
                status = use_impact && !lexical_neutral_seen
                    ? EVOKE_ERR_FORMAT
                    : EVOKE_ERR_RANGE;
                goto fail;
            }
            total_extents += selected_count;
        }
    }
    evoke_segment_read_view_init(&view);
    view.vocab_size = manifest->vocab_size;
    view.extent_count = (uint32_t) total_extents;
    if (view.vocab_size > 0)
    {
        view.terms = calloc(view.vocab_size, sizeof(*view.terms));
        if (view.terms == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto fail_view;
        }
    }
    if (view.extent_count > 0)
    {
        view.extents = calloc(view.extent_count, sizeof(*view.extents));
        if (view.extents == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto fail_view;
        }
    }
    if (manifest->segment_count > 0)
    {
        next_posting_offsets = calloc(
            manifest->segment_count,
            sizeof(*next_posting_offsets)
        );
        next_run_indices = calloc(
            manifest->segment_count,
            sizeof(*next_run_indices)
        );
        if (next_posting_offsets == NULL || next_run_indices == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto fail_view;
        }
    }
    for (uint32_t segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        status = evoke_segment_payload_attach_validate(
            global_index,
            manifest,
            &manifest->segments[segment_index],
            &payloads[segment_index]
        );
        if (status != EVOKE_OK)
        {
            goto fail_view;
        }
    }

    for (uint32_t term_id = 0;
         term_id < manifest->vocab_size;
         term_id++)
    {
        const evoke_term_fold_read_plan *plan = &fold_plans[term_id];
        uint64_t tail_start = tail_directory->term_offsets[term_id];
        uint64_t tail_end = tail_directory->term_offsets[term_id + 1];
        uint64_t effective_coverage =
            plan->neutral_minor_run_count > 0
                ? plan->neutral_minor_coverage
                : plan->neutral_coverage;
        bool stale_coverage_boundary =
            effective_coverage > 0 &&
            (manifest->flags &
             EVOKE_SEGMENT_MANIFEST_FLAG_COW_TERM_DIRECTORY) != 0 &&
            !evoke_segment_manifest_has_sequence_boundary(
                manifest,
                effective_coverage
            );
        bool use_impact = plan->impact_run_count > 0;
        bool coverage_source_segment_seen = false;
        uint32_t folded_kind_mask = 0;
        uint32_t covered_kind_mask = 0;
        uint32_t term_extent_count = (uint32_t) (
            tail_end - tail_start
        );

        for (size_t level = 0; level < 2; level++)
        {
            uint32_t bundle_index = level == 0
                ? plan->neutral_bundle_index
                : plan->neutral_minor_bundle_index;
            uint32_t first_run = level == 0
                ? plan->neutral_first_run
                : plan->neutral_minor_first_run;
            uint32_t run_count = level == 0
                ? plan->neutral_run_count
                : plan->neutral_minor_run_count;

            if (run_count == 0)
            {
                continue;
            }
            for (uint32_t run_offset = 0;
                 run_offset < run_count;
                 run_offset++)
            {
                const evoke_term_fold_run *run =
                    &fold_bundles[bundle_index].runs[
                        first_run + run_offset
                    ];

                if (!use_impact ||
                    run->kind !=
                        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    term_extent_count++;
                }
            }
        }
        term_extent_count += plan->impact_run_count;

        if (term_extent_count > 0)
        {
            view.terms[term_id].extents =
                &view.extents[extent_cursor];
            view.terms[term_id].len = term_extent_count;
        }
        if (plan->neutral_run_count > 0)
        {
            const evoke_term_fold_bundle *bundle =
                &fold_bundles[plan->neutral_bundle_index];

            for (uint32_t plan_run_index = 0;
                 plan_run_index < plan->neutral_run_count;
                 plan_run_index++)
            {
                const evoke_term_fold_run *run =
                    &bundle->runs[
                        plan->neutral_first_run + plan_run_index
                    ];

                if (use_impact &&
                    run->kind ==
                        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    continue;
                }
                folded_kind_mask |= UINT32_C(1) << run->kind;
                status = evoke_segment_attach_fold_extent(
                    global_index,
                    bundle,
                    run,
                    &view.extents[extent_cursor++]
                );
                if (status != EVOKE_OK)
                {
                    goto fail_view;
                }
            }
        }
        if (plan->neutral_minor_run_count > 0)
        {
            const evoke_term_fold_bundle *bundle =
                &fold_bundles[plan->neutral_minor_bundle_index];

            for (uint32_t plan_run_index = 0;
                 plan_run_index < plan->neutral_minor_run_count;
                 plan_run_index++)
            {
                const evoke_term_fold_run *run =
                    &bundle->runs[
                        plan->neutral_minor_first_run +
                        plan_run_index
                    ];

                if (use_impact &&
                    run->kind ==
                        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    continue;
                }
                folded_kind_mask |= UINT32_C(1) << run->kind;
                status = evoke_segment_attach_fold_extent(
                    global_index,
                    bundle,
                    run,
                    &view.extents[extent_cursor++]
                );
                if (status != EVOKE_OK)
                {
                    goto fail_view;
                }
            }
        }
        if (use_impact)
        {
            const evoke_term_fold_bundle *bundle =
                &fold_bundles[plan->impact_bundle_index];

            for (uint32_t plan_run_index = 0;
                 plan_run_index < plan->impact_run_count;
                 plan_run_index++)
            {
                const evoke_term_fold_run *run =
                    &bundle->runs[
                        plan->impact_first_run + plan_run_index
                    ];

                folded_kind_mask |=
                    UINT32_C(1) <<
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
                status = evoke_segment_attach_fold_extent(
                    global_index,
                    bundle,
                    run,
                    &view.extents[extent_cursor++]
                );
                if (status != EVOKE_OK)
                {
                    goto fail_view;
                }
            }
        }

        for (uint32_t segment_index = 0;
             segment_index < manifest->segment_count;
             segment_index++)
        {
            const evoke_segment_descriptor *segment =
                &manifest->segments[segment_index];
            const evoke_segment_payload_view *payload =
                &payloads[segment_index];

            if (effective_coverage > 0 &&
                segment->min_sequence <= effective_coverage)
            {
                coverage_source_segment_seen = true;
            }

            while (next_run_indices[segment_index] <
                   payload->run_count)
            {
                const evoke_segment_term_run *run =
                    &payload->runs[next_run_indices[segment_index]];
                bool segment_straddles_coverage;
                bool run_is_tail = false;

                if (run->term_id < term_id)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto fail_view;
                }
                if (run->term_id != term_id ||
                    effective_coverage == 0)
                {
                    break;
                }
                segment_straddles_coverage =
                    segment->min_sequence <= effective_coverage &&
                    segment->max_sequence > effective_coverage;
                if (segment_straddles_coverage &&
                    stale_coverage_boundary)
                {
                    /*
                     * Older compaction could erase a fold boundary. Its
                     * COW tail descriptor is the authority for which side
                     * owned the replacement run.
                     */
                    run_is_tail =
                        evoke_segment_tail_directory_has_run(
                            tail_directory,
                            tail_start,
                            tail_end,
                            segment_index,
                            run
                        );
                }
                if (segment->max_sequence > effective_coverage &&
                    (!segment_straddles_coverage || run_is_tail))
                {
                    break;
                }
                if (segment_straddles_coverage &&
                    !stale_coverage_boundary)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto fail_view;
                }
                if (run->posting_offset !=
                    next_posting_offsets[segment_index])
                {
                    status = EVOKE_ERR_FORMAT;
                    goto fail_view;
                }
                covered_kind_mask |= UINT32_C(1) << run->kind;
                next_posting_offsets[segment_index] +=
                    run->posting_count;
                next_run_indices[segment_index]++;
            }
            if (effective_coverage > 0 &&
                next_run_indices[segment_index] <
                    payload->run_count &&
                payload->runs[
                    next_run_indices[segment_index]
                ].term_id == term_id &&
                segment->min_sequence <=
                    effective_coverage &&
                segment->max_sequence >
                    effective_coverage &&
                !stale_coverage_boundary)
            {
                status = EVOKE_ERR_FORMAT;
                goto fail_view;
            }
        }

        for (uint64_t tail_index = tail_start;
             tail_index < tail_end;
             tail_index++)
        {
            const evoke_term_extent_descriptor *descriptor =
                &tail_directory->extents[tail_index];
            const evoke_segment_descriptor *segment =
                &manifest->segments[descriptor->segment_index];
            const evoke_segment_payload_view *payload =
                &payloads[descriptor->segment_index];
            uint32_t run_index =
                next_run_indices[descriptor->segment_index];
            const evoke_segment_term_run *run;

            if ((effective_coverage > 0 &&
                 segment->max_sequence <= effective_coverage) ||
                (effective_coverage > 0 &&
                 segment->min_sequence <= effective_coverage &&
                 !stale_coverage_boundary) ||
                run_index >= payload->run_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto fail_view;
            }
            run = &payload->runs[run_index];
            if (run->term_id != term_id ||
                run->kind != descriptor->kind ||
                run->posting_offset != descriptor->posting_offset ||
                run->posting_count != descriptor->posting_count ||
                run->posting_offset !=
                    next_posting_offsets[descriptor->segment_index])
            {
                status = EVOKE_ERR_FORMAT;
                goto fail_view;
            }
            status = evoke_segment_attach_payload_extent(
                global_index,
                payload,
                run,
                &view.extents[extent_cursor++]
            );
            if (status != EVOKE_OK)
            {
                goto fail_view;
            }
            next_posting_offsets[descriptor->segment_index] +=
                run->posting_count;
            next_run_indices[descriptor->segment_index]++;
        }

        for (uint32_t segment_index = 0;
             segment_index < manifest->segment_count;
             segment_index++)
        {
            const evoke_segment_payload_view *payload =
                &payloads[segment_index];

            if (next_run_indices[segment_index] <
                    payload->run_count &&
                payload->runs[
                    next_run_indices[segment_index]
                ].term_id <= term_id)
            {
                status = EVOKE_ERR_FORMAT;
                goto fail_view;
            }
        }
        if (effective_coverage > 0 && coverage_source_segment_seen &&
            (covered_kind_mask == 0 ||
             covered_kind_mask != folded_kind_mask))
        {
            status = EVOKE_ERR_FORMAT;
            goto fail_view;
        }
    }

    for (uint32_t segment_index = 0;
         segment_index < manifest->segment_count;
         segment_index++)
    {
        if (next_posting_offsets[segment_index] !=
                payloads[segment_index].posting_count ||
            next_run_indices[segment_index] !=
                payloads[segment_index].run_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto fail_view;
        }
    }
    if (extent_cursor != view.extent_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto fail_view;
    }

    free(next_posting_offsets);
    free(next_run_indices);
    free(bundle_runs_used);
    free(bundle_run_offsets);
    evoke_segment_read_view_free(view_out);
    *view_out = view;
    return EVOKE_OK;

fail_view:
    free(next_posting_offsets);
    free(next_run_indices);
    evoke_segment_read_view_free(&view);
fail:
    free(bundle_runs_used);
    free(bundle_run_offsets);
    return status;
}
