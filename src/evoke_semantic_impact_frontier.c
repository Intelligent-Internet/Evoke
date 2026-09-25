#include "evoke_semantic_impact_frontier.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_SEMANTIC_IMPACT_FRONTIER_MAGIC UINT32_C(0x31464953)
#define EVOKE_SEMANTIC_IMPACT_FRONTIER_VERSION UINT16_C(2)
#define EVOKE_SEMANTIC_IMPACT_FRONTIER_CHECKSUM_OFFSET 56U

static void
evoke_semantic_impact_frontier_write_u16(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t) value;
    bytes[1] = (uint8_t) (value >> 8);
}

static void
evoke_semantic_impact_frontier_write_u32(uint8_t *bytes, uint32_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static void
evoke_semantic_impact_frontier_write_u64(uint8_t *bytes, uint64_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static uint16_t
evoke_semantic_impact_frontier_read_u16(const uint8_t *bytes)
{
    return (uint16_t) bytes[0] |
        (uint16_t) ((uint16_t) bytes[1] << 8);
}

static uint32_t
evoke_semantic_impact_frontier_read_u32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0] |
        ((uint32_t) bytes[1] << 8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static uint64_t
evoke_semantic_impact_frontier_read_u64(const uint8_t *bytes)
{
    uint64_t value = 0;

    for (size_t index = 0; index < sizeof(value); index++)
    {
        value |= (uint64_t) bytes[index] << (index * 8);
    }
    return value;
}

static uint64_t
evoke_semantic_impact_frontier_header_checksum(
    const uint8_t *bytes,
    size_t size
)
{
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= EVOKE_SEMANTIC_IMPACT_FRONTIER_CHECKSUM_OFFSET &&
            index < EVOKE_SEMANTIC_IMPACT_FRONTIER_CHECKSUM_OFFSET +
                sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static size_t
evoke_semantic_impact_frontier_varint_size(uint32_t value)
{
    size_t size = 1;

    while (value >= UINT32_C(0x80))
    {
        value >>= 7;
        size++;
    }
    return size;
}

static size_t
evoke_semantic_impact_frontier_write_varint(
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
evoke_semantic_impact_frontier_read_varint(
    const uint8_t *bytes,
    size_t size,
    size_t *offset,
    uint32_t *value_out
)
{
    uint32_t value = 0;

    for (uint32_t shift = 0; shift <= 28; shift += 7)
    {
        uint8_t byte;

        if (*offset >= size)
        {
            return false;
        }
        byte = bytes[(*offset)++];
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

evoke_status
evoke_semantic_impact_frontier_round_up(
    double impact,
    float *impact_out
)
{
    float rounded;

    if (impact_out == NULL || !isfinite(impact) || impact <= 0.0 ||
        impact > (double) FLT_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    rounded = (float) impact;
    if ((double) rounded < impact)
    {
        rounded = nextafterf(rounded, INFINITY);
    }
    if (!isfinite(rounded) || rounded <= 0.0f)
    {
        return EVOKE_ERR_RANGE;
    }
    *impact_out = rounded;
    return EVOKE_OK;
}

size_t
evoke_semantic_impact_frontier_block_size(
    uint32_t block_delta,
    uint32_t posting_count,
    evoke_semantic_impact_precision impact_precision
)
{
    size_t impact_width = evoke_semantic_bmp_impact_width(impact_precision);
    size_t posting_bytes;
    size_t header_bytes;

    if (impact_width == 0 || posting_count == 0 ||
        posting_count > EVOKE_SEMANTIC_IMPACT_FRONTIER_BLOCK_SIZE)
    {
        return 0;
    }
    posting_bytes = (size_t) posting_count *
        (EVOKE_SEMANTIC_IMPACT_FRONTIER_LOCAL_DOCUMENT_SIZE + impact_width);
    header_bytes = evoke_semantic_impact_frontier_varint_size(block_delta) +
        evoke_semantic_impact_frontier_varint_size(posting_count);
    if (posting_bytes > SIZE_MAX - header_bytes)
    {
        return 0;
    }
    return header_bytes + posting_bytes;
}

evoke_status
evoke_semantic_impact_frontier_encode_block(
    uint8_t *bytes,
    size_t capacity,
    uint32_t block_delta,
    const uint8_t *local_documents,
    const double *impacts,
    uint32_t posting_count,
    evoke_semantic_impact_precision impact_precision,
    size_t *size_out
)
{
    bool seen[EVOKE_SEMANTIC_IMPACT_FRONTIER_BLOCK_SIZE] = {false};
    size_t required;
    size_t offset;
    size_t impact_width = evoke_semantic_bmp_impact_width(impact_precision);
    size_t posting_width =
        EVOKE_SEMANTIC_IMPACT_FRONTIER_LOCAL_DOCUMENT_SIZE + impact_width;
    float previous = INFINITY;

    if (bytes == NULL || local_documents == NULL || impacts == NULL ||
        size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    required = evoke_semantic_impact_frontier_block_size(
        block_delta,
        posting_count,
        impact_precision
    );
    if (required == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    if (required > capacity)
    {
        return EVOKE_ERR_RANGE;
    }
    offset = evoke_semantic_impact_frontier_write_varint(bytes, block_delta);
    offset += evoke_semantic_impact_frontier_write_varint(
        bytes + offset,
        posting_count
    );
    for (uint32_t index = 0; index < posting_count; index++)
    {
        uint8_t local_document = local_documents[index];
        float impact;
        evoke_status status;

        if (local_document >= EVOKE_SEMANTIC_IMPACT_FRONTIER_BLOCK_SIZE ||
            seen[local_document])
        {
            return EVOKE_ERR_INVALID;
        }
        status = evoke_semantic_impact_frontier_round_up(
            impacts[index],
            &impact
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        status = evoke_semantic_impact_encode(
            bytes + offset +
                EVOKE_SEMANTIC_IMPACT_FRONTIER_LOCAL_DOCUMENT_SIZE,
            impact_width,
            impact_precision,
            impact,
            &impact
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        if (impact <= 0.0f || impact > previous)
        {
            return EVOKE_ERR_INVALID;
        }
        seen[local_document] = true;
        bytes[offset] = local_document;
        previous = impact;
        offset += posting_width;
    }
    *size_out = offset;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_impact_frontier_shard_serialize(
    uint64_t source_authority_checksum,
    uint32_t document_count,
    uint32_t vocab_size,
    uint32_t first_term,
    uint32_t term_count,
    evoke_semantic_impact_precision impact_precision,
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
        term_count > EVOKE_SEMANTIC_IMPACT_FRONTIER_TERMS_PER_SHARD ||
        first_term >= vocab_size || term_count > vocab_size - first_term ||
        evoke_semantic_bmp_impact_width(impact_precision) == 0 ||
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
        offsets_size > SIZE_MAX -
            EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE ||
        payload_size > SIZE_MAX -
            EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE - offsets_size)
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
    total_size = EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE +
        offsets_size + payload_size;
    bytes = calloc(total_size, 1);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    evoke_semantic_impact_frontier_write_u32(
        bytes + 0,
        EVOKE_SEMANTIC_IMPACT_FRONTIER_MAGIC
    );
    evoke_semantic_impact_frontier_write_u16(
        bytes + 4,
        EVOKE_SEMANTIC_IMPACT_FRONTIER_VERSION
    );
    evoke_semantic_impact_frontier_write_u16(
        bytes + 6,
        EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE
    );
    evoke_semantic_impact_frontier_write_u16(
        bytes + 8,
        EVOKE_SEMANTIC_IMPACT_FRONTIER_BLOCK_SHIFT
    );
    evoke_semantic_impact_frontier_write_u16(
        bytes + 10,
        (uint16_t) impact_precision
    );
    evoke_semantic_impact_frontier_write_u32(bytes + 12, first_term);
    evoke_semantic_impact_frontier_write_u32(bytes + 16, term_count);
    evoke_semantic_impact_frontier_write_u32(bytes + 20, document_count);
    evoke_semantic_impact_frontier_write_u32(bytes + 24, vocab_size);
    evoke_semantic_impact_frontier_write_u64(
        bytes + 32,
        source_authority_checksum
    );
    evoke_semantic_impact_frontier_write_u64(bytes + 40, total_size);
    evoke_semantic_impact_frontier_write_u64(bytes + 48, offsets_size);
    for (uint32_t term = 0; term <= term_count; term++)
    {
        evoke_semantic_impact_frontier_write_u64(
            bytes + EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE +
                (size_t) term * sizeof(uint64_t),
            payload_offsets[term]
        );
    }
    if (payload_size > 0)
    {
        memcpy(
            bytes + EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE + offsets_size,
            payload,
            payload_size
        );
    }
    evoke_semantic_impact_frontier_write_u64(
        bytes + EVOKE_SEMANTIC_IMPACT_FRONTIER_CHECKSUM_OFFSET,
        evoke_semantic_impact_frontier_header_checksum(
            bytes,
            EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE
        )
    );
    *bytes_out = bytes;
    *size_out = total_size;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_impact_frontier_header_deserialize(
    const uint8_t *bytes,
    size_t size,
    size_t object_size,
    uint64_t expected_source_authority_checksum,
    evoke_semantic_impact_frontier_summary *summary_out
)
{
    evoke_semantic_impact_frontier_summary summary;
    uint64_t offsets_size;

    if (bytes == NULL || summary_out == NULL ||
        size < EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE ||
        object_size < EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_semantic_impact_frontier_read_u32(bytes + 0) !=
            EVOKE_SEMANTIC_IMPACT_FRONTIER_MAGIC ||
        evoke_semantic_impact_frontier_read_u16(bytes + 4) !=
            EVOKE_SEMANTIC_IMPACT_FRONTIER_VERSION ||
        evoke_semantic_impact_frontier_read_u16(bytes + 6) !=
            EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE ||
        evoke_semantic_impact_frontier_read_u16(bytes + 8) !=
            EVOKE_SEMANTIC_IMPACT_FRONTIER_BLOCK_SHIFT ||
        evoke_semantic_bmp_impact_width(
            (evoke_semantic_impact_precision)
                evoke_semantic_impact_frontier_read_u16(bytes + 10)
        ) == 0 ||
        evoke_semantic_impact_frontier_read_u32(bytes + 28) != 0 ||
        evoke_semantic_impact_frontier_read_u64(bytes + 40) != object_size ||
        evoke_semantic_impact_frontier_read_u64(bytes + 56) !=
            evoke_semantic_impact_frontier_header_checksum(
                bytes,
                EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE
            ))
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&summary, 0, sizeof(summary));
    summary.block_shift =
        evoke_semantic_impact_frontier_read_u16(bytes + 8);
    summary.impact_precision = (evoke_semantic_impact_precision)
        evoke_semantic_impact_frontier_read_u16(bytes + 10);
    summary.first_term =
        evoke_semantic_impact_frontier_read_u32(bytes + 12);
    summary.term_count =
        evoke_semantic_impact_frontier_read_u32(bytes + 16);
    summary.document_count =
        evoke_semantic_impact_frontier_read_u32(bytes + 20);
    summary.vocab_size =
        evoke_semantic_impact_frontier_read_u32(bytes + 24);
    summary.source_authority_checksum =
        evoke_semantic_impact_frontier_read_u64(bytes + 32);
    offsets_size = evoke_semantic_impact_frontier_read_u64(bytes + 48);
    summary.object_size = object_size;
    summary.offsets_offset = EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE;
    if (summary.source_authority_checksum == 0 ||
        (expected_source_authority_checksum != 0 &&
         summary.source_authority_checksum !=
            expected_source_authority_checksum) ||
        summary.document_count == 0 || summary.vocab_size == 0 ||
        summary.term_count == 0 ||
        summary.term_count >
            EVOKE_SEMANTIC_IMPACT_FRONTIER_TERMS_PER_SHARD ||
        summary.first_term >= summary.vocab_size ||
        summary.term_count > summary.vocab_size - summary.first_term ||
        offsets_size !=
            ((uint64_t) summary.term_count + 1U) * sizeof(uint64_t) ||
        offsets_size > object_size -
            EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    summary.payload_offset =
        EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE + (size_t) offsets_size;
    *summary_out = summary;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_impact_frontier_term_slice(
    const evoke_semantic_impact_frontier_summary *summary,
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
    size_t local_offset;
    size_t payload_size;

    if (summary == NULL || offset_bytes == NULL ||
        offset_out == NULL || size_out == NULL ||
        term_id < summary->first_term ||
        term_id >= summary->first_term + summary->term_count)
    {
        return EVOKE_ERR_INVALID;
    }
    local_term = term_id - summary->first_term;
    local_offset = (size_t) local_term * sizeof(uint64_t);
    if (local_offset > offset_size ||
        offset_size - local_offset < 2U * sizeof(uint64_t))
    {
        return EVOKE_ERR_RANGE;
    }
    start = evoke_semantic_impact_frontier_read_u64(
        offset_bytes + local_offset
    );
    end = evoke_semantic_impact_frontier_read_u64(
        offset_bytes + local_offset + sizeof(uint64_t)
    );
    payload_size = summary->object_size - summary->payload_offset;
    if (start > end || end > payload_size ||
        summary->payload_offset > SIZE_MAX - (size_t) start)
    {
        return EVOKE_ERR_FORMAT;
    }
    *offset_out = summary->payload_offset + (size_t) start;
    *size_out = (size_t) (end - start);
    return EVOKE_OK;
}

evoke_status
evoke_semantic_impact_frontier_cursor_init(
    evoke_semantic_impact_frontier_cursor *cursor,
    const uint8_t *bytes,
    size_t size,
    uint32_t block_count,
    evoke_semantic_impact_precision impact_precision
)
{
    size_t impact_width = evoke_semantic_bmp_impact_width(impact_precision);

    if (cursor == NULL || (size > 0 && bytes == NULL) || block_count == 0 ||
        impact_width == 0 || impact_width >= UINT8_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(cursor, 0, sizeof(*cursor));
    cursor->bytes = bytes;
    cursor->size = size;
    cursor->block_count = block_count;
    cursor->posting_width = (uint8_t)
        (EVOKE_SEMANTIC_IMPACT_FRONTIER_LOCAL_DOCUMENT_SIZE + impact_width);
    cursor->impact_precision = impact_precision;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_impact_frontier_cursor_next(
    evoke_semantic_impact_frontier_cursor *cursor,
    evoke_semantic_impact_frontier_block *block_out,
    bool *has_block_out
)
{
    bool seen[EVOKE_SEMANTIC_IMPACT_FRONTIER_BLOCK_SIZE] = {false};
    uint32_t block_delta;
    uint32_t block_id;
    uint32_t posting_count;
    size_t offset;
    size_t posting_bytes;
    float previous = INFINITY;

    if (cursor == NULL || block_out == NULL || has_block_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *has_block_out = false;
    memset(block_out, 0, sizeof(*block_out));
    if (cursor->offset == cursor->size)
    {
        return EVOKE_OK;
    }
    offset = cursor->offset;
    if (!evoke_semantic_impact_frontier_read_varint(
            cursor->bytes,
            cursor->size,
            &offset,
            &block_delta) ||
        !evoke_semantic_impact_frontier_read_varint(
            cursor->bytes,
            cursor->size,
            &offset,
            &posting_count))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (posting_count == 0 ||
        posting_count > EVOKE_SEMANTIC_IMPACT_FRONTIER_BLOCK_SIZE ||
        (cursor->has_previous && block_delta == 0))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (cursor->has_previous)
    {
        if (block_delta > UINT32_MAX - cursor->previous_block_id)
        {
            return EVOKE_ERR_FORMAT;
        }
        block_id = cursor->previous_block_id + block_delta;
    }
    else
    {
        block_id = block_delta;
    }
    if (block_id >= cursor->block_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    posting_bytes = (size_t) posting_count *
        cursor->posting_width;
    if (posting_bytes > cursor->size - offset)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t index = 0; index < posting_count; index++)
    {
        const uint8_t *posting = cursor->bytes + offset +
            (size_t) index * cursor->posting_width;
        uint8_t local_document = posting[0];
        float impact;
        evoke_status status = evoke_semantic_impact_decode(
            posting + EVOKE_SEMANTIC_IMPACT_FRONTIER_LOCAL_DOCUMENT_SIZE,
            cursor->posting_width -
                EVOKE_SEMANTIC_IMPACT_FRONTIER_LOCAL_DOCUMENT_SIZE,
            cursor->impact_precision,
            &impact
        );

        if (local_document >= EVOKE_SEMANTIC_IMPACT_FRONTIER_BLOCK_SIZE ||
            seen[local_document] || status != EVOKE_OK || impact <= 0.0f ||
            impact > previous)
        {
            return EVOKE_ERR_FORMAT;
        }
        seen[local_document] = true;
        previous = impact;
    }
    block_out->block_id = block_id;
    block_out->posting_count = posting_count;
    block_out->postings = cursor->bytes + offset;
    block_out->posting_width = cursor->posting_width;
    block_out->impact_precision = cursor->impact_precision;
    cursor->offset = offset + posting_bytes;
    cursor->previous_block_id = block_id;
    cursor->has_previous = true;
    *has_block_out = true;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_impact_frontier_block_posting(
    const evoke_semantic_impact_frontier_block *block,
    uint32_t posting_index,
    uint8_t *local_document_out,
    float *impact_out
)
{
    const uint8_t *posting;

    if (block == NULL || block->postings == NULL ||
        local_document_out == NULL || impact_out == NULL ||
        posting_index >= block->posting_count ||
        block->posting_width <=
            EVOKE_SEMANTIC_IMPACT_FRONTIER_LOCAL_DOCUMENT_SIZE)
    {
        return EVOKE_ERR_INVALID;
    }
    posting = block->postings +
        (size_t) posting_index * block->posting_width;
    *local_document_out = posting[0];
    return evoke_semantic_impact_decode(
        posting + EVOKE_SEMANTIC_IMPACT_FRONTIER_LOCAL_DOCUMENT_SIZE,
        block->posting_width -
            EVOKE_SEMANTIC_IMPACT_FRONTIER_LOCAL_DOCUMENT_SIZE,
        block->impact_precision,
        impact_out
    );
}
