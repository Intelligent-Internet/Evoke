#include "evoke_semantic_forward_bound.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_SEMANTIC_FORWARD_BOUND_MAGIC UINT32_C(0x31424653)
#define EVOKE_SEMANTIC_FORWARD_BOUND_VERSION UINT16_C(1)
#define EVOKE_SEMANTIC_FORWARD_BOUND_CHECKSUM_OFFSET 56U

static void
evoke_semantic_forward_bound_write_u16(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t) value;
    bytes[1] = (uint8_t) (value >> 8);
}

static void
evoke_semantic_forward_bound_write_u32(uint8_t *bytes, uint32_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static void
evoke_semantic_forward_bound_write_u64(uint8_t *bytes, uint64_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static void
evoke_semantic_forward_bound_write_f32(uint8_t *bytes, float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    evoke_semantic_forward_bound_write_u32(bytes, bits);
}

static uint16_t
evoke_semantic_forward_bound_read_u16(const uint8_t *bytes)
{
    return (uint16_t) bytes[0] |
        (uint16_t) ((uint16_t) bytes[1] << 8);
}

static uint32_t
evoke_semantic_forward_bound_read_u32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0] |
        ((uint32_t) bytes[1] << 8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static uint64_t
evoke_semantic_forward_bound_read_u64(const uint8_t *bytes)
{
    uint64_t value = 0;

    for (size_t index = 0; index < sizeof(value); index++)
    {
        value |= (uint64_t) bytes[index] << (index * 8);
    }
    return value;
}

static float
evoke_semantic_forward_bound_read_f32(const uint8_t *bytes)
{
    uint32_t bits = evoke_semantic_forward_bound_read_u32(bytes);
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t
evoke_semantic_forward_bound_header_checksum(
    const uint8_t *bytes,
    size_t size
)
{
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= EVOKE_SEMANTIC_FORWARD_BOUND_CHECKSUM_OFFSET &&
            index < EVOKE_SEMANTIC_FORWARD_BOUND_CHECKSUM_OFFSET +
                sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static size_t
evoke_semantic_forward_bound_write_varint(
    uint8_t *bytes,
    uint32_t value
)
{
    size_t size = 0;

    do
    {
        uint8_t byte = (uint8_t) (value & UINT32_C(0x7f));

        value >>= 7;
        if (value != 0)
        {
            byte |= UINT8_C(0x80);
        }
        bytes[size++] = byte;
    }
    while (value != 0);
    return size;
}

static bool
evoke_semantic_forward_bound_read_varint(
    const uint8_t *bytes,
    size_t size,
    size_t *cursor,
    uint32_t *value_out
)
{
    uint32_t value = 0;

    for (uint32_t shift = 0; shift <= 28; shift += 7)
    {
        uint8_t byte;

        if (*cursor >= size)
        {
            return false;
        }
        byte = bytes[(*cursor)++];
        if (shift == 28 && (byte & UINT8_C(0xf0)) != 0)
        {
            return false;
        }
        value |= (uint32_t) (byte & UINT8_C(0x7f)) << shift;
        if ((byte & UINT8_C(0x80)) == 0)
        {
            *value_out = value;
            return true;
        }
    }
    return false;
}

size_t
evoke_semantic_forward_bound_entry_size(uint32_t block_delta)
{
    size_t size = 1U + sizeof(float);

    while (block_delta >= UINT32_C(0x80))
    {
        block_delta >>= 7;
        size++;
    }
    return size;
}

evoke_status
evoke_semantic_forward_bound_encode_entry(
    uint8_t *bytes,
    size_t capacity,
    uint32_t block_delta,
    float maximum,
    size_t *size_out
)
{
    size_t required;
    size_t cursor;

    if (bytes == NULL || size_out == NULL ||
        !isfinite(maximum) || maximum <= 0.0f)
    {
        return EVOKE_ERR_INVALID;
    }
    required = evoke_semantic_forward_bound_entry_size(block_delta);
    if (required > capacity)
    {
        return EVOKE_ERR_RANGE;
    }
    cursor = evoke_semantic_forward_bound_write_varint(bytes, block_delta);
    evoke_semantic_forward_bound_write_f32(bytes + cursor, maximum);
    *size_out = required;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_forward_bound_shard_serialize(
    uint64_t source_authority_checksum,
    uint32_t document_count,
    uint32_t vocab_size,
    uint32_t first_term,
    uint32_t term_count,
    const uint64_t *payload_offsets,
    const uint8_t *payload,
    size_t payload_size,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes;
    size_t offsets_size;
    size_t total_size;

    if (source_authority_checksum == 0 || document_count == 0 ||
        vocab_size == 0 || term_count == 0 ||
        term_count > EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD ||
        first_term >= vocab_size || term_count > vocab_size - first_term ||
        payload_offsets == NULL ||
        (payload_size > 0 && payload == NULL) ||
        bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    offsets_size = ((size_t) term_count + 1U) * sizeof(uint64_t);
    if (payload_offsets[0] != 0 ||
        payload_offsets[term_count] != payload_size ||
        offsets_size > SIZE_MAX - EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE ||
        payload_size > SIZE_MAX -
            EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE - offsets_size)
    {
        return EVOKE_ERR_RANGE;
    }
    for (uint32_t term = 1; term <= term_count; term++)
    {
        if (payload_offsets[term] < payload_offsets[term - 1U] ||
            payload_offsets[term] > payload_size)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    total_size = EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE +
        offsets_size + payload_size;
    bytes = calloc(total_size, 1);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    evoke_semantic_forward_bound_write_u32(
        bytes + 0,
        EVOKE_SEMANTIC_FORWARD_BOUND_MAGIC
    );
    evoke_semantic_forward_bound_write_u16(
        bytes + 4,
        EVOKE_SEMANTIC_FORWARD_BOUND_VERSION
    );
    evoke_semantic_forward_bound_write_u16(
        bytes + 6,
        EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE
    );
    evoke_semantic_forward_bound_write_u16(
        bytes + 8,
        EVOKE_SEMANTIC_FORWARD_BOUND_BLOCK_SHIFT
    );
    evoke_semantic_forward_bound_write_u32(bytes + 12, first_term);
    evoke_semantic_forward_bound_write_u32(bytes + 16, term_count);
    evoke_semantic_forward_bound_write_u32(bytes + 20, document_count);
    evoke_semantic_forward_bound_write_u32(bytes + 24, vocab_size);
    evoke_semantic_forward_bound_write_u64(
        bytes + 32,
        source_authority_checksum
    );
    evoke_semantic_forward_bound_write_u64(bytes + 40, total_size);
    evoke_semantic_forward_bound_write_u64(bytes + 48, offsets_size);
    for (uint32_t term = 0; term <= term_count; term++)
    {
        evoke_semantic_forward_bound_write_u64(
            bytes + EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE +
                (size_t) term * sizeof(uint64_t),
            payload_offsets[term]
        );
    }
    if (payload_size > 0)
    {
        memcpy(
            bytes + EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE + offsets_size,
            payload,
            payload_size
        );
    }
    evoke_semantic_forward_bound_write_u64(
        bytes + EVOKE_SEMANTIC_FORWARD_BOUND_CHECKSUM_OFFSET,
        evoke_semantic_forward_bound_header_checksum(
            bytes,
            EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE
        )
    );
    *bytes_out = bytes;
    *size_out = total_size;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_forward_bound_header_deserialize(
    const uint8_t *bytes,
    size_t size,
    size_t object_size,
    uint64_t expected_source_authority_checksum,
    evoke_semantic_forward_bound_summary *summary_out
)
{
    evoke_semantic_forward_bound_summary summary;
    uint64_t offsets_size;

    if (bytes == NULL || summary_out == NULL ||
        size < EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE ||
        object_size < EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_semantic_forward_bound_read_u32(bytes + 0) !=
            EVOKE_SEMANTIC_FORWARD_BOUND_MAGIC ||
        evoke_semantic_forward_bound_read_u16(bytes + 4) !=
            EVOKE_SEMANTIC_FORWARD_BOUND_VERSION ||
        evoke_semantic_forward_bound_read_u16(bytes + 6) !=
            EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE ||
        evoke_semantic_forward_bound_read_u16(bytes + 8) !=
            EVOKE_SEMANTIC_FORWARD_BOUND_BLOCK_SHIFT ||
        evoke_semantic_forward_bound_read_u16(bytes + 10) != 0 ||
        evoke_semantic_forward_bound_read_u32(bytes + 28) != 0 ||
        evoke_semantic_forward_bound_read_u64(bytes + 40) != object_size ||
        evoke_semantic_forward_bound_read_u64(bytes + 56) !=
            evoke_semantic_forward_bound_header_checksum(
                bytes,
                EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE
            ))
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&summary, 0, sizeof(summary));
    summary.block_shift =
        evoke_semantic_forward_bound_read_u16(bytes + 8);
    summary.first_term = evoke_semantic_forward_bound_read_u32(bytes + 12);
    summary.term_count = evoke_semantic_forward_bound_read_u32(bytes + 16);
    summary.document_count =
        evoke_semantic_forward_bound_read_u32(bytes + 20);
    summary.vocab_size = evoke_semantic_forward_bound_read_u32(bytes + 24);
    summary.source_authority_checksum =
        evoke_semantic_forward_bound_read_u64(bytes + 32);
    offsets_size = evoke_semantic_forward_bound_read_u64(bytes + 48);
    summary.object_size = object_size;
    summary.offsets_offset = EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE;
    if (summary.source_authority_checksum == 0 ||
        (expected_source_authority_checksum != 0 &&
         summary.source_authority_checksum !=
            expected_source_authority_checksum) ||
        summary.document_count == 0 || summary.vocab_size == 0 ||
        summary.term_count == 0 ||
        summary.term_count > EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD ||
        summary.first_term >= summary.vocab_size ||
        summary.term_count > summary.vocab_size - summary.first_term ||
        offsets_size !=
            ((uint64_t) summary.term_count + 1U) * sizeof(uint64_t) ||
        offsets_size > object_size - summary.offsets_offset)
    {
        return EVOKE_ERR_FORMAT;
    }
    summary.payload_offset = summary.offsets_offset + (size_t) offsets_size;
    *summary_out = summary;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_forward_bound_term_slice(
    const evoke_semantic_forward_bound_summary *summary,
    const uint8_t *offset_bytes,
    size_t offset_size,
    uint32_t term_id,
    size_t *offset_out,
    size_t *size_out
)
{
    uint32_t local_term;
    uint64_t start;
    uint64_t end;

    if (summary == NULL || offset_bytes == NULL ||
        offset_size < 2U * sizeof(uint64_t) ||
        term_id < summary->first_term ||
        term_id >= summary->first_term + summary->term_count ||
        offset_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    local_term = term_id - summary->first_term;
    start = evoke_semantic_forward_bound_read_u64(offset_bytes);
    end = evoke_semantic_forward_bound_read_u64(
        offset_bytes + sizeof(uint64_t)
    );
    if (start > end ||
        end > summary->object_size - summary->payload_offset ||
        start > SIZE_MAX || end - start > SIZE_MAX)
    {
        return EVOKE_ERR_FORMAT;
    }
    (void) local_term;
    *offset_out = summary->payload_offset + (size_t) start;
    *size_out = (size_t) (end - start);
    return EVOKE_OK;
}

evoke_status
evoke_semantic_forward_bound_visit(
    const uint8_t *bytes,
    size_t size,
    uint32_t block_count,
    evoke_semantic_forward_bound_visitor visitor,
    void *visitor_context,
    uint32_t *entry_count_out
)
{
    size_t cursor = 0;
    uint32_t block_id = 0;
    uint32_t entry_count = 0;

    if ((size > 0 && bytes == NULL) || block_count == 0 ||
        visitor == NULL || entry_count_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    while (cursor < size)
    {
        uint32_t delta;
        float maximum;
        evoke_status status;

        if (!evoke_semantic_forward_bound_read_varint(
                bytes,
                size,
                &cursor,
                &delta) ||
            cursor > size - sizeof(float) ||
            (entry_count > 0 && delta == 0) ||
            block_id > UINT32_MAX - delta)
        {
            return EVOKE_ERR_FORMAT;
        }
        block_id += delta;
        maximum = evoke_semantic_forward_bound_read_f32(bytes + cursor);
        cursor += sizeof(float);
        if (block_id >= block_count || !isfinite(maximum) || maximum <= 0.0f)
        {
            return EVOKE_ERR_FORMAT;
        }
        status = visitor(visitor_context, block_id, maximum);
        if (status != EVOKE_OK)
        {
            return status;
        }
        if (entry_count == UINT32_MAX)
        {
            return EVOKE_ERR_RANGE;
        }
        entry_count++;
    }
    *entry_count_out = entry_count;
    return EVOKE_OK;
}
