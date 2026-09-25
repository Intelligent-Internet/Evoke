#include "evoke_semantic_forward.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_SEMANTIC_FORWARD_MAGIC UINT32_C(0x31574653)
#define EVOKE_SEMANTIC_FORWARD_VERSION \
    EVOKE_SEMANTIC_FORWARD_TRANSPOSE_VERSION
#define EVOKE_SEMANTIC_FORWARD_HEADER_SIZE \
    EVOKE_SEMANTIC_FORWARD_HEADER_BYTES
#define EVOKE_SEMANTIC_FORWARD_CHECKSUM_OFFSET 40U
#define EVOKE_SEMANTIC_FORWARD_QUANTIZATION_BITS UINT32_C(8)

static void
evoke_semantic_forward_write_u16(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t) value;
    bytes[1] = (uint8_t) (value >> 8);
}

static void
evoke_semantic_forward_write_u32(uint8_t *bytes, uint32_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static void
evoke_semantic_forward_write_f32(uint8_t *bytes, float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    evoke_semantic_forward_write_u32(bytes, bits);
}

static void
evoke_semantic_forward_write_u64(uint8_t *bytes, uint64_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static uint16_t
evoke_semantic_forward_read_u16(const uint8_t *bytes)
{
    return (uint16_t) bytes[0] | (uint16_t) ((uint16_t) bytes[1] << 8);
}

static uint32_t
evoke_semantic_forward_read_u32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0] |
        ((uint32_t) bytes[1] << 8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static float
evoke_semantic_forward_read_f32(const uint8_t *bytes)
{
    uint32_t bits = evoke_semantic_forward_read_u32(bytes);
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t
evoke_semantic_forward_varint_size(uint32_t value)
{
    uint32_t size = 1;

    while (value >= UINT32_C(128))
    {
        value >>= 7;
        size++;
    }
    return size;
}

static size_t
evoke_semantic_forward_write_varint(uint8_t *bytes, uint32_t value)
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
evoke_semantic_forward_read_varint(
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

static uint64_t
evoke_semantic_forward_read_u64(const uint8_t *bytes)
{
    uint64_t value = 0;

    for (size_t index = 0; index < sizeof(value); index++)
    {
        value |= (uint64_t) bytes[index] << (index * 8);
    }
    return value;
}

static uint64_t
evoke_semantic_forward_checksum(const uint8_t *bytes, size_t size)
{
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t value =
            index >= EVOKE_SEMANTIC_FORWARD_CHECKSUM_OFFSET &&
            index < EVOKE_SEMANTIC_FORWARD_CHECKSUM_OFFSET + sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static bool
evoke_semantic_forward_add_size(
    size_t *size,
    uint64_t count,
    size_t width
)
{
    if (count > SIZE_MAX / width || *size > SIZE_MAX - count * width)
    {
        return false;
    }
    *size += (size_t) count * width;
    return true;
}

size_t
evoke_semantic_forward_header_size(void)
{
    return EVOKE_SEMANTIC_FORWARD_HEADER_SIZE;
}

evoke_status
evoke_semantic_forward_header_deserialize(
    const uint8_t *bytes,
    size_t size,
    size_t object_size,
    uint64_t expected_source_authority_checksum,
    evoke_semantic_forward_header *header_out
)
{
    evoke_semantic_forward_header header;
    size_t expected_size;
    size_t transpose_size;
    uint16_t declared_header_size;
    uint16_t version;

    if (bytes == NULL || header_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (size < EVOKE_SEMANTIC_FORWARD_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&header, 0, sizeof(header));
    version = evoke_semantic_forward_read_u16(bytes + 4);
    declared_header_size = evoke_semantic_forward_read_u16(bytes + 6);
    expected_size = declared_header_size;
    header.format_version = version;
    header.first_document = evoke_semantic_forward_read_u32(bytes + 8);
    header.document_count = evoke_semantic_forward_read_u32(bytes + 12);
    header.posting_count = evoke_semantic_forward_read_u32(bytes + 16);
    header.source_authority_checksum =
        evoke_semantic_forward_read_u64(bytes + 24);
    header.object_size = object_size;
    if (evoke_semantic_forward_read_u32(bytes + 0) !=
            EVOKE_SEMANTIC_FORWARD_MAGIC ||
        version != EVOKE_SEMANTIC_FORWARD_VERSION ||
        declared_header_size != EVOKE_SEMANTIC_FORWARD_HEADER_SIZE ||
        size < declared_header_size ||
        header.document_count == 0 ||
        header.document_count > (uint32_t) UINT16_MAX + UINT32_C(1) ||
        header.first_document > UINT32_MAX - header.document_count ||
        evoke_semantic_forward_read_u32(bytes + 20) !=
            EVOKE_SEMANTIC_FORWARD_QUANTIZATION_BITS ||
        evoke_semantic_forward_read_u64(bytes + 32) != object_size ||
        evoke_semantic_forward_read_u64(
            bytes + EVOKE_SEMANTIC_FORWARD_CHECKSUM_OFFSET) == 0 ||
        header.source_authority_checksum == 0 ||
        (expected_source_authority_checksum != 0 &&
         header.source_authority_checksum !=
            expected_source_authority_checksum))
    {
        return EVOKE_ERR_FORMAT;
    }
    header.row_offsets_offset = expected_size;
    if (!evoke_semantic_forward_add_size(
            &expected_size,
            (uint64_t) header.document_count + 1U,
            sizeof(uint32_t)))
    {
        return EVOKE_ERR_RANGE;
    }
    header.row_data_offset = expected_size;
    header.row_data_end = object_size;
    header.quantization_bits = evoke_semantic_forward_read_u32(bytes + 20);
    if (expected_size > object_size)
    {
        return EVOKE_ERR_FORMAT;
    }
    {
        uint64_t transpose_offset =
            evoke_semantic_forward_read_u64(bytes + 48);

        header.vocab_size = evoke_semantic_forward_read_u32(bytes + 56);
        header.transpose_entry_size =
            evoke_semantic_forward_read_u32(bytes + 60);
        if (transpose_offset < expected_size ||
            transpose_offset > object_size ||
            header.vocab_size == 0 ||
            header.transpose_entry_size !=
                EVOKE_SEMANTIC_FORWARD_TRANSPOSE_ENTRY_SIZE)
        {
            return EVOKE_ERR_FORMAT;
        }
        header.row_data_end = (size_t) transpose_offset;
        header.transpose_scales_offset = header.row_data_end;
        transpose_size = header.transpose_scales_offset;
        if (!evoke_semantic_forward_add_size(
                &transpose_size,
                header.document_count,
                sizeof(float)))
        {
            return EVOKE_ERR_RANGE;
        }
        header.transpose_term_offsets_offset = transpose_size;
        if (!evoke_semantic_forward_add_size(
                &transpose_size,
                (uint64_t) header.vocab_size + 1U,
                sizeof(uint32_t)))
        {
            return EVOKE_ERR_RANGE;
        }
        header.transpose_postings_offset = transpose_size;
        header.transpose_sparse_posting_count =
            evoke_semantic_forward_read_u32(bytes + 64);
        header.transpose_dense_term_count =
            evoke_semantic_forward_read_u32(bytes + 68);
        if (header.transpose_sparse_posting_count > header.posting_count ||
            header.transpose_dense_term_count > header.vocab_size)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (!evoke_semantic_forward_add_size(
                &transpose_size,
                header.transpose_sparse_posting_count,
                EVOKE_SEMANTIC_FORWARD_TRANSPOSE_ENTRY_SIZE))
        {
            return EVOKE_ERR_RANGE;
        }
        header.transpose_dense_term_ids_offset = transpose_size;
        if (!evoke_semantic_forward_add_size(
                &transpose_size,
                header.transpose_dense_term_count,
                sizeof(uint32_t)))
        {
            return EVOKE_ERR_RANGE;
        }
        header.transpose_dense_codes_offset = transpose_size;
        if (!evoke_semantic_forward_add_size(
                &transpose_size,
                (uint64_t) header.transpose_dense_term_count *
                    header.document_count,
                sizeof(int8_t)) ||
            transpose_size != object_size)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    *header_out = header;
    return EVOKE_OK;
}

static evoke_status
evoke_semantic_forward_validate_transpose_equivalence(
    const uint8_t *bytes,
    const evoke_semantic_forward_header *header
)
{
    uint32_t *dense_term_indexes = NULL;
    uint32_t *term_cursors = NULL;
    uint8_t *dense_presence = NULL;
    size_t dense_presence_stride = 0;
    uint32_t posting_count = 0;
    uint32_t dense_posting_count = 0;
    uint32_t previous_offset = 0;
    evoke_status status = EVOKE_OK;

    if (header->format_version != EVOKE_SEMANTIC_FORWARD_VERSION)
    {
        return EVOKE_ERR_FORMAT;
    }
    dense_term_indexes = malloc(
        (size_t) header->vocab_size * sizeof(*dense_term_indexes)
    );
    term_cursors = malloc(
        (size_t) header->vocab_size * sizeof(*term_cursors)
    );
    if (dense_term_indexes == NULL || term_cursors == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (uint32_t term_id = 0;
         term_id < header->vocab_size;
         term_id++)
    {
        dense_term_indexes[term_id] = UINT32_MAX;
    }
    if (header->transpose_dense_term_count > 0)
    {
        uint32_t previous_term = 0;

        dense_presence_stride =
            ((size_t) header->document_count + 7U) / 8U;
        if (header->transpose_dense_term_count >
                SIZE_MAX / dense_presence_stride)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        dense_presence = calloc(
            header->transpose_dense_term_count,
            dense_presence_stride
        );
        if (dense_presence == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        for (uint32_t dense_index = 0;
             dense_index < header->transpose_dense_term_count;
             dense_index++)
        {
            uint32_t term_id = evoke_semantic_forward_read_u32(
                bytes + header->transpose_dense_term_ids_offset +
                    (size_t) dense_index * sizeof(uint32_t)
            );

            if (term_id >= header->vocab_size ||
                (dense_index > 0 && term_id <= previous_term))
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            dense_term_indexes[term_id] = dense_index;
            previous_term = term_id;
        }
    }
    for (uint32_t term_id = 0;
         term_id <= header->vocab_size;
         term_id++)
    {
        uint32_t offset = evoke_semantic_forward_read_u32(
            bytes + header->transpose_term_offsets_offset +
                (size_t) term_id * sizeof(uint32_t)
        );

        if (offset < previous_offset ||
            offset > header->transpose_sparse_posting_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        if (term_id < header->vocab_size)
        {
            term_cursors[term_id] = offset;
            if (dense_term_indexes[term_id] != UINT32_MAX)
            {
                uint32_t end = evoke_semantic_forward_read_u32(
                    bytes + header->transpose_term_offsets_offset +
                        ((size_t) term_id + 1U) * sizeof(uint32_t)
                );

                if (offset != end)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
            }
        }
        previous_offset = offset;
    }
    if (previous_offset != header->transpose_sparse_posting_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    for (uint32_t document = 0;
         document < header->document_count;
         document++)
    {
        uint32_t row_start = evoke_semantic_forward_read_u32(
            bytes + header->row_offsets_offset +
                (size_t) document * sizeof(uint32_t)
        );
        uint32_t row_end = evoke_semantic_forward_read_u32(
            bytes + header->row_offsets_offset +
                ((size_t) document + 1U) * sizeof(uint32_t)
        );
        size_t row_size;
        size_t row_cursor;
        uint32_t previous_term = 0;
        const uint8_t *transpose_scale =
            bytes + header->transpose_scales_offset +
                (size_t) document * sizeof(float);

        if (row_start > row_end ||
            row_end > header->row_data_end - header->row_data_offset)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        row_size = (size_t) row_end - row_start;
        if (row_size == 0)
        {
            float scale = evoke_semantic_forward_read_f32(transpose_scale);

            if (scale != 1.0f)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            continue;
        }
        if (row_size < sizeof(float) ||
            memcmp(
                bytes + header->row_data_offset + row_start,
                transpose_scale,
                sizeof(float)
            ) != 0)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        row_cursor = sizeof(float);
        while (row_cursor < row_size)
        {
            uint32_t delta;
            uint32_t term_id;
            uint32_t transpose_posting;
            uint32_t transpose_end;
            uint32_t dense_index;
            size_t entry_offset;
            int8_t code;

            if (!evoke_semantic_forward_read_varint(
                    bytes + header->row_data_offset + row_start,
                    row_size,
                    &row_cursor,
                    &delta
                ) ||
                row_cursor >= row_size ||
                previous_term > UINT32_MAX - delta)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            term_id = previous_term + delta;
            code = (int8_t) bytes[
                header->row_data_offset + row_start + row_cursor
            ];
            row_cursor++;
            if (term_id >= header->vocab_size || code == INT8_MIN ||
                posting_count == UINT32_MAX)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            dense_index = dense_term_indexes[term_id];
            if (dense_index != UINT32_MAX)
            {
                size_t presence_offset =
                    (size_t) dense_index * dense_presence_stride +
                    document / 8U;
                size_t code_offset = header->transpose_dense_codes_offset +
                    (size_t) dense_index * header->document_count + document;
                uint8_t presence_mask =
                    (uint8_t) (UINT8_C(1) << (document & 7U));

                if ((dense_presence[presence_offset] & presence_mask) != 0 ||
                    (int8_t) bytes[code_offset] != code)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                dense_presence[presence_offset] |= presence_mask;
                dense_posting_count++;
            }
            else
            {
                transpose_posting = term_cursors[term_id]++;
                transpose_end = evoke_semantic_forward_read_u32(
                    bytes + header->transpose_term_offsets_offset +
                        ((size_t) term_id + 1U) * sizeof(uint32_t)
                );
                if (transpose_posting >= transpose_end)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                entry_offset = header->transpose_postings_offset +
                    (size_t) transpose_posting *
                        header->transpose_entry_size;
                if (evoke_semantic_forward_read_u16(bytes + entry_offset) !=
                        document ||
                    (int8_t) bytes[entry_offset + sizeof(uint16_t)] != code)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
            }
            posting_count++;
            previous_term = term_id;
        }
    }
    if (posting_count != header->posting_count ||
        dense_posting_count !=
            header->posting_count -
                header->transpose_sparse_posting_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    for (uint32_t term_id = 0;
         term_id < header->vocab_size;
         term_id++)
    {
        uint32_t end = evoke_semantic_forward_read_u32(
            bytes + header->transpose_term_offsets_offset +
                ((size_t) term_id + 1U) * sizeof(uint32_t)
        );

        if (term_cursors[term_id] != end)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
    }
    for (uint32_t dense_index = 0;
         dense_index < header->transpose_dense_term_count;
         dense_index++)
    {
        for (uint32_t document = 0;
             document < header->document_count;
             document++)
        {
            size_t presence_offset =
                (size_t) dense_index * dense_presence_stride + document / 8U;
            size_t code_offset = header->transpose_dense_codes_offset +
                (size_t) dense_index * header->document_count + document;
            uint8_t presence_mask =
                (uint8_t) (UINT8_C(1) << (document & 7U));
            int8_t code = (int8_t) bytes[code_offset];

            if (code == INT8_MIN ||
                ((dense_presence[presence_offset] & presence_mask) == 0 &&
                 code != 0))
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
        }
    }

cleanup:
    free(dense_presence);
    free(term_cursors);
    free(dense_term_indexes);
    return status;
}

void
evoke_semantic_forward_chunk_init(evoke_semantic_forward_chunk *chunk)
{
    if (chunk != NULL)
    {
        memset(chunk, 0, sizeof(*chunk));
    }
}

void
evoke_semantic_forward_chunk_free(evoke_semantic_forward_chunk *chunk)
{
    if (chunk == NULL)
    {
        return;
    }
    free(chunk->row_offsets);
    free(chunk->term_ids);
    free(chunk->operations);
    free(chunk->contributions);
    evoke_semantic_forward_chunk_init(chunk);
}

evoke_status
evoke_semantic_forward_chunk_validate(
    const evoke_semantic_forward_chunk *chunk
)
{
    if (chunk == NULL || chunk->source_authority_checksum == 0 ||
        chunk->document_count == 0 || chunk->vocab_size == 0 ||
        chunk->row_offsets == NULL ||
        (chunk->posting_count > 0 &&
         (chunk->term_ids == NULL || chunk->operations == NULL ||
          chunk->contributions == NULL)) ||
        chunk->first_document > UINT32_MAX - chunk->document_count ||
        chunk->row_offsets[0] != 0 ||
        chunk->row_offsets[chunk->document_count] != chunk->posting_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t document = 0;
         document < chunk->document_count;
         document++)
    {
        uint32_t start = chunk->row_offsets[document];
        uint32_t end = chunk->row_offsets[document + 1U];

        if (start > end || end > chunk->posting_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t posting = start; posting < end; posting++)
        {
            if ((chunk->operations[posting] !=
                    EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT &&
                 chunk->operations[posting] !=
                    EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT) ||
                !isfinite(chunk->contributions[posting]) ||
                chunk->term_ids[posting] >= chunk->vocab_size ||
                (posting > start &&
                 chunk->term_ids[posting - 1U] > chunk->term_ids[posting]))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    return EVOKE_OK;
}

evoke_status
evoke_semantic_forward_chunk_serialize(
    const evoke_semantic_forward_chunk *chunk,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes = NULL;
    uint8_t *dense_terms = NULL;
    float *scales = NULL;
    int8_t *codes = NULL;
    uint32_t *term_counts = NULL;
    uint32_t *term_cursors = NULL;
    uint32_t *term_dense_indexes = NULL;
    size_t size = EVOKE_SEMANTIC_FORWARD_HEADER_SIZE;
    size_t row_data_size = 0;
    size_t row_data_offset;
    size_t transpose_offset;
    size_t transpose_scales_offset;
    size_t transpose_term_offsets_offset;
    size_t transpose_postings_offset;
    size_t transpose_dense_term_ids_offset;
    size_t transpose_dense_codes_offset;
    size_t row_cursor = 0;
    size_t cursor;
    uint32_t sparse_posting_count = 0;
    uint32_t dense_term_count = 0;
    const int32_t maximum_code = INT8_MAX;
    evoke_status status = EVOKE_OK;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_semantic_forward_chunk_validate(chunk);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (!evoke_semantic_forward_add_size(
            &size,
            (uint64_t) chunk->document_count + 1U,
            sizeof(uint32_t)))
    {
        return EVOKE_ERR_RANGE;
    }
    row_data_offset = size;
    dense_terms = calloc(chunk->vocab_size, sizeof(*dense_terms));
    scales = malloc((size_t) chunk->document_count * sizeof(*scales));
    term_counts = calloc(chunk->vocab_size, sizeof(*term_counts));
    term_cursors = malloc(
        (size_t) chunk->vocab_size * sizeof(*term_cursors)
    );
    term_dense_indexes = malloc(
        (size_t) chunk->vocab_size * sizeof(*term_dense_indexes)
    );
    if (chunk->posting_count > 0)
    {
        codes = malloc((size_t) chunk->posting_count * sizeof(*codes));
    }
    if (dense_terms == NULL || scales == NULL || term_counts == NULL ||
        term_cursors == NULL || term_dense_indexes == NULL ||
        (chunk->posting_count > 0 && codes == NULL))
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (uint32_t document = 0;
         document < chunk->document_count;
         document++)
    {
        uint32_t start = chunk->row_offsets[document];
        uint32_t end = chunk->row_offsets[document + 1U];
        double maximum_absolute = 0.0;
        uint32_t previous_term = 0;

        for (uint32_t posting = start; posting < end; posting++)
        {
            double absolute = fabs(chunk->contributions[posting]);

            if (absolute > maximum_absolute)
            {
                maximum_absolute = absolute;
            }
        }
        scales[document] = maximum_absolute == 0.0
            ? 1.0f
            : (float) (maximum_absolute / (double) maximum_code);
        if (!isfinite(scales[document]) || scales[document] <= 0.0f)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        if (start == end)
        {
            continue;
        }
        if (!evoke_semantic_forward_add_size(
                &row_data_size,
                1,
                sizeof(float)))
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        for (uint32_t posting = start; posting < end; posting++)
        {
            uint32_t term_id = chunk->term_ids[posting];
            uint32_t delta = term_id - previous_term;
            long code = lround(
                chunk->contributions[posting] / scales[document]
            );

            if (!evoke_semantic_forward_add_size(
                    &row_data_size,
                    evoke_semantic_forward_varint_size(delta),
                    1) ||
                !evoke_semantic_forward_add_size(
                    &row_data_size,
                    1,
                    sizeof(int8_t)))
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            if (code > maximum_code)
            {
                code = maximum_code;
            }
            else if (code < -maximum_code)
            {
                code = -maximum_code;
            }
            codes[posting] = (int8_t) code;
            if (term_counts[term_id] == UINT32_MAX)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            term_counts[term_id]++;
            if (posting > start &&
                chunk->term_ids[posting - 1U] == term_id)
            {
                dense_terms[term_id] = UINT8_MAX;
            }
            previous_term = term_id;
        }
    }
    for (uint32_t term_id = 0;
         term_id < chunk->vocab_size;
         term_id++)
    {
        uint64_t sparse_bytes =
            (uint64_t) term_counts[term_id] *
                EVOKE_SEMANTIC_FORWARD_TRANSPOSE_ENTRY_SIZE;
        uint64_t dense_bytes =
            (uint64_t) chunk->document_count + sizeof(uint32_t);

        if (dense_terms[term_id] != UINT8_MAX && term_counts[term_id] > 0 &&
            dense_bytes <= sparse_bytes)
        {
            dense_terms[term_id] = 1;
            term_dense_indexes[term_id] = dense_term_count++;
        }
        else
        {
            dense_terms[term_id] = 0;
            term_dense_indexes[term_id] = UINT32_MAX;
            if (UINT32_MAX - sparse_posting_count < term_counts[term_id])
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            sparse_posting_count += term_counts[term_id];
        }
    }
    if (row_data_size > UINT32_MAX || size > SIZE_MAX - row_data_size)
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    size += row_data_size;
    transpose_offset = size;
    transpose_scales_offset = size;
    if (!evoke_semantic_forward_add_size(
            &size,
            chunk->document_count,
            sizeof(float)))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    transpose_term_offsets_offset = size;
    if (!evoke_semantic_forward_add_size(
            &size,
            (uint64_t) chunk->vocab_size + 1U,
            sizeof(uint32_t)))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    transpose_postings_offset = size;
    if (!evoke_semantic_forward_add_size(
            &size,
            sparse_posting_count,
            EVOKE_SEMANTIC_FORWARD_TRANSPOSE_ENTRY_SIZE))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    transpose_dense_term_ids_offset = size;
    if (!evoke_semantic_forward_add_size(
            &size,
            dense_term_count,
            sizeof(uint32_t)))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    transpose_dense_codes_offset = size;
    if (!evoke_semantic_forward_add_size(
            &size,
            (uint64_t) dense_term_count * chunk->document_count,
            sizeof(int8_t)))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    bytes = calloc(size, 1);
    if (bytes == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    evoke_semantic_forward_write_u32(bytes + 0, EVOKE_SEMANTIC_FORWARD_MAGIC);
    evoke_semantic_forward_write_u16(bytes + 4, EVOKE_SEMANTIC_FORWARD_VERSION);
    evoke_semantic_forward_write_u16(
        bytes + 6,
        EVOKE_SEMANTIC_FORWARD_HEADER_SIZE
    );
    evoke_semantic_forward_write_u32(bytes + 8, chunk->first_document);
    evoke_semantic_forward_write_u32(bytes + 12, chunk->document_count);
    evoke_semantic_forward_write_u32(bytes + 16, chunk->posting_count);
    evoke_semantic_forward_write_u32(
        bytes + 20,
        EVOKE_SEMANTIC_FORWARD_QUANTIZATION_BITS
    );
    evoke_semantic_forward_write_u64(
        bytes + 24,
        chunk->source_authority_checksum
    );
    evoke_semantic_forward_write_u64(bytes + 32, size);
    evoke_semantic_forward_write_u64(bytes + 48, transpose_offset);
    evoke_semantic_forward_write_u32(bytes + 56, chunk->vocab_size);
    evoke_semantic_forward_write_u32(
        bytes + 60,
        EVOKE_SEMANTIC_FORWARD_TRANSPOSE_ENTRY_SIZE
    );
    evoke_semantic_forward_write_u32(bytes + 64, sparse_posting_count);
    evoke_semantic_forward_write_u32(bytes + 68, dense_term_count);
    cursor = EVOKE_SEMANTIC_FORWARD_HEADER_SIZE;
    for (uint32_t document = 0;
         document < chunk->document_count;
         document++)
    {
        uint32_t start = chunk->row_offsets[document];
        uint32_t end = chunk->row_offsets[document + 1U];
        uint32_t previous_term = 0;

        evoke_semantic_forward_write_u32(bytes + cursor, (uint32_t) row_cursor);
        cursor += sizeof(uint32_t);
        if (start == end)
        {
            continue;
        }
        evoke_semantic_forward_write_f32(
            bytes + row_data_offset + row_cursor,
            scales[document]
        );
        row_cursor += sizeof(float);
        for (uint32_t posting = start; posting < end; posting++)
        {
            uint32_t term_id = chunk->term_ids[posting];
            uint32_t delta = term_id - previous_term;
            row_cursor += evoke_semantic_forward_write_varint(
                bytes + row_data_offset + row_cursor,
                delta
            );
            bytes[row_data_offset + row_cursor] =
                (uint8_t) codes[posting];
            row_cursor += sizeof(int8_t);
            previous_term = term_id;
        }
    }
    evoke_semantic_forward_write_u32(bytes + cursor, (uint32_t) row_cursor);
    cursor += sizeof(uint32_t);
    if (cursor != row_data_offset || row_cursor != row_data_size)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    for (uint32_t document = 0;
         document < chunk->document_count;
         document++)
    {
        evoke_semantic_forward_write_f32(
            bytes + transpose_scales_offset +
                (size_t) document * sizeof(float),
            scales[document]
        );
    }
    {
        uint32_t posting_cursor = 0;

        for (uint32_t term_id = 0;
             term_id < chunk->vocab_size;
             term_id++)
        {
            evoke_semantic_forward_write_u32(
                bytes + transpose_term_offsets_offset +
                    (size_t) term_id * sizeof(uint32_t),
                posting_cursor
            );
            term_cursors[term_id] = posting_cursor;
            if (dense_terms[term_id] == 0 &&
                UINT32_MAX - posting_cursor < term_counts[term_id])
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            if (dense_terms[term_id] == 0)
            {
                posting_cursor += term_counts[term_id];
            }
        }
        evoke_semantic_forward_write_u32(
            bytes + transpose_term_offsets_offset +
                (size_t) chunk->vocab_size * sizeof(uint32_t),
            posting_cursor
        );
        if (posting_cursor != sparse_posting_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
    }
    {
        uint32_t dense_index = 0;

        for (uint32_t term_id = 0;
             term_id < chunk->vocab_size;
             term_id++)
        {
            if (dense_terms[term_id] == 0)
            {
                continue;
            }
            if (dense_index != term_dense_indexes[term_id])
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            evoke_semantic_forward_write_u32(
                bytes + transpose_dense_term_ids_offset +
                    (size_t) dense_index * sizeof(uint32_t),
                term_id
            );
            dense_index++;
        }
        if (dense_index != dense_term_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
    }
    for (uint32_t document = 0;
         document < chunk->document_count;
         document++)
    {
        uint32_t start = chunk->row_offsets[document];
        uint32_t end = chunk->row_offsets[document + 1U];

        if (document > UINT16_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        for (uint32_t posting = start; posting < end; posting++)
        {
            uint32_t term_id = chunk->term_ids[posting];
            uint32_t dense_index = term_dense_indexes[term_id];

            if (dense_index != UINT32_MAX)
            {
                size_t code_offset = transpose_dense_codes_offset +
                    (size_t) dense_index * chunk->document_count + document;

                bytes[code_offset] = (uint8_t) codes[posting];
            }
            else
            {
                uint32_t transpose_posting = term_cursors[term_id]++;
                size_t entry_offset = transpose_postings_offset +
                    (size_t) transpose_posting *
                        EVOKE_SEMANTIC_FORWARD_TRANSPOSE_ENTRY_SIZE;

                evoke_semantic_forward_write_u16(
                    bytes + entry_offset,
                    (uint16_t) document
                );
                bytes[entry_offset + sizeof(uint16_t)] =
                    (uint8_t) codes[posting];
            }
        }
    }
    evoke_semantic_forward_write_u64(
        bytes + EVOKE_SEMANTIC_FORWARD_CHECKSUM_OFFSET,
        evoke_semantic_forward_checksum(bytes, size)
    );
    *bytes_out = bytes;
    *size_out = size;
    bytes = NULL;

cleanup:
    free(bytes);
    free(term_dense_indexes);
    free(term_cursors);
    free(term_counts);
    free(codes);
    free(scales);
    free(dense_terms);
    return status;
}

evoke_status
evoke_semantic_forward_chunk_deserialize(
    const uint8_t *bytes,
    size_t size,
    uint64_t expected_source_authority_checksum,
    evoke_semantic_forward_chunk *chunk_out
)
{
    evoke_semantic_forward_chunk chunk;
    evoke_semantic_forward_header header;
    uint32_t posting_cursor = 0;
    evoke_status status;

    if (bytes == NULL || chunk_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (size < EVOKE_SEMANTIC_FORWARD_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_semantic_forward_chunk_init(&chunk);
    status = evoke_semantic_forward_header_deserialize(
        bytes,
        size,
        size,
        expected_source_authority_checksum,
        &header
    );
    if (status != EVOKE_OK ||
        evoke_semantic_forward_read_u64(
            bytes + EVOKE_SEMANTIC_FORWARD_CHECKSUM_OFFSET) !=
            evoke_semantic_forward_checksum(bytes, size))
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_semantic_forward_validate_transpose_equivalence(
        bytes,
        &header
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    chunk.first_document = header.first_document;
    chunk.document_count = header.document_count;
    chunk.posting_count = header.posting_count;
    chunk.vocab_size = header.vocab_size == 0
        ? UINT32_MAX
        : header.vocab_size;
    chunk.source_authority_checksum = header.source_authority_checksum;
    chunk.row_offsets = calloc(
        (size_t) chunk.document_count + 1U,
        sizeof(*chunk.row_offsets)
    );
    if (chunk.posting_count > 0)
    {
        chunk.term_ids = malloc(
            (size_t) chunk.posting_count * sizeof(*chunk.term_ids)
        );
        chunk.operations = malloc(
            (size_t) chunk.posting_count * sizeof(*chunk.operations)
        );
        chunk.contributions = malloc(
            (size_t) chunk.posting_count * sizeof(*chunk.contributions)
        );
    }
    if (chunk.row_offsets == NULL ||
        (chunk.posting_count > 0 &&
         (chunk.term_ids == NULL || chunk.operations == NULL ||
          chunk.contributions == NULL)))
    {
        evoke_semantic_forward_chunk_free(&chunk);
        return EVOKE_ERR_NOMEM;
    }
    if (evoke_semantic_forward_read_u32(
            bytes + header.row_offsets_offset) != 0 ||
        evoke_semantic_forward_read_u32(
            bytes + header.row_offsets_offset +
                (size_t) chunk.document_count * sizeof(uint32_t)) !=
            header.row_data_end - header.row_data_offset)
    {
        evoke_semantic_forward_chunk_free(&chunk);
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t document = 0;
         document < chunk.document_count;
         document++)
    {
        uint32_t row_start = evoke_semantic_forward_read_u32(
            bytes + header.row_offsets_offset +
                (size_t) document * sizeof(uint32_t)
        );
        uint32_t row_end = evoke_semantic_forward_read_u32(
            bytes + header.row_offsets_offset +
                ((size_t) document + 1U) * sizeof(uint32_t)
        );
        size_t row_size;
        size_t row_cursor = 0;
        const size_t code_width = sizeof(int8_t);
        uint32_t previous_term = 0;
        float scale;

        chunk.row_offsets[document] = posting_cursor;
        if (row_start > row_end ||
            row_end > header.row_data_end - header.row_data_offset)
        {
            status = EVOKE_ERR_FORMAT;
            break;
        }
        row_size = (size_t) row_end - row_start;
        if (row_size == 0)
        {
            continue;
        }
        if (row_size < sizeof(float))
        {
            status = EVOKE_ERR_FORMAT;
            break;
        }
        scale = evoke_semantic_forward_read_f32(
            bytes + header.row_data_offset + row_start
        );
        if (!isfinite(scale) || scale <= 0.0f)
        {
            status = EVOKE_ERR_FORMAT;
            break;
        }
        row_cursor = sizeof(float);
        while (row_cursor < row_size)
        {
            uint32_t delta;
            uint32_t term_id;
            int32_t code;

            if (posting_cursor >= chunk.posting_count ||
                !evoke_semantic_forward_read_varint(
                    bytes + header.row_data_offset + row_start,
                    row_size,
                    &row_cursor,
                    &delta
                ) ||
                row_cursor > row_size - code_width ||
                previous_term > UINT32_MAX - delta)
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
            term_id = previous_term + delta;
            code = (int8_t) bytes[
                header.row_data_offset + row_start + row_cursor
            ];
            row_cursor += code_width;
            if (code == INT8_MIN)
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
            chunk.term_ids[posting_cursor] = term_id;
            chunk.operations[posting_cursor] =
                EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT;
            chunk.contributions[posting_cursor] = (double) code * scale;
            previous_term = term_id;
            posting_cursor++;
        }
        if (status != EVOKE_OK)
        {
            break;
        }
    }
    chunk.row_offsets[chunk.document_count] = posting_cursor;
    if (status == EVOKE_OK && posting_cursor != chunk.posting_count)
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status == EVOKE_OK)
    {
        status = evoke_semantic_forward_chunk_validate(&chunk);
    }
    if (status != EVOKE_OK)
    {
        evoke_semantic_forward_chunk_free(&chunk);
        return status;
    }
    evoke_semantic_forward_chunk_free(chunk_out);
    *chunk_out = chunk;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_forward_score_row_sorted(
    const uint32_t *term_ids,
    const uint8_t *operations,
    const double *contributions,
    uint32_t posting_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out
)
{
    uint32_t posting;
    size_t query = 0;
    float score = 0.0f;

    if (query_ids == NULL || query_weights == NULL || score_out == NULL ||
        (posting_count > 0 &&
         (term_ids == NULL || operations == NULL || contributions == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    posting = 0;
    while (posting < posting_count && query < query_count)
    {
        if ((posting > 0 && term_ids[posting - 1U] > term_ids[posting]) ||
            (operations[posting] !=
                EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT &&
             operations[posting] != EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT) ||
            !isfinite(contributions[posting]))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (term_ids[posting] < query_ids[query])
        {
            posting++;
            continue;
        }
        if (term_ids[posting] > query_ids[query])
        {
            query++;
            continue;
        }
        do
        {
            if (operations[posting] ==
                EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT)
            {
                score += (float) (
                    contributions[posting] * (double) query_weights[query]
                );
            }
            else
            {
                float impact = (float) contributions[posting];

                score += query_weights[query] * impact;
            }
            posting++;
        }
        while (posting < posting_count &&
               term_ids[posting] == query_ids[query]);
        query++;
    }
    while (posting < posting_count)
    {
        if ((posting > 0 && term_ids[posting - 1U] > term_ids[posting]) ||
            (operations[posting] !=
                EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT &&
             operations[posting] != EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT) ||
            !isfinite(contributions[posting]))
        {
            return EVOKE_ERR_FORMAT;
        }
        posting++;
    }
    if (!isfinite(score) || score > FLT_MAX || score < -FLT_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    *score_out = score;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_forward_score_row(
    const uint32_t *term_ids,
    const uint8_t *operations,
    const double *contributions,
    uint32_t posting_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out
)
{
    bool query_is_sorted = true;
    float score = 0.0f;

    if (query_ids == NULL || query_weights == NULL || score_out == NULL ||
        (posting_count > 0 &&
         (term_ids == NULL || operations == NULL || contributions == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    for (size_t query = 0; query < query_count; query++)
    {
        if (!isfinite(query_weights[query]))
        {
            return EVOKE_ERR_INVALID;
        }
        if (query > 0 && query_ids[query - 1U] >= query_ids[query])
        {
            query_is_sorted = false;
        }
    }
    if (query_is_sorted)
    {
        return evoke_semantic_forward_score_row_sorted(
            term_ids,
            operations,
            contributions,
            posting_count,
            query_ids,
            query_weights,
            query_count,
            score_out
        );
    }
    for (uint32_t posting = 0; posting < posting_count; posting++)
    {
        if ((posting > 0 && term_ids[posting - 1U] > term_ids[posting]) ||
            (operations[posting] !=
                EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT &&
             operations[posting] != EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT) ||
            !isfinite(contributions[posting]))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    for (size_t query = 0; query < query_count; query++)
    {
        uint32_t left = 0;
        uint32_t right = posting_count;
        uint32_t posting;

        while (left < right)
        {
            uint32_t middle = left + (right - left) / 2U;

            if (term_ids[middle] < query_ids[query])
            {
                left = middle + 1U;
            }
            else
            {
                right = middle;
            }
        }
        posting = left;
        while (posting < posting_count &&
               term_ids[posting] == query_ids[query])
        {
            if (operations[posting] ==
                EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT)
            {
                score += (float) (
                    contributions[posting] * (double) query_weights[query]
                );
            }
            else
            {
                score += query_weights[query] *
                    (float) contributions[posting];
            }
            posting++;
        }
    }
    if (!isfinite(score) || score > FLT_MAX || score < -FLT_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    *score_out = score;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_forward_score_serialized_row_sorted(
    const uint8_t *bytes,
    size_t size,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out,
    uint32_t *posting_count_out
)
{
    size_t cursor = 0;
    size_t query = 0;
    const size_t code_width = sizeof(int8_t);
    uint32_t previous_term = 0;
    uint32_t posting_count = 0;
    double score = 0.0;
    float scale;

    if (query_ids == NULL || query_weights == NULL || score_out == NULL ||
        posting_count_out == NULL ||
        (size > 0 && bytes == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    for (size_t index = 0; index < query_count; index++)
    {
        if (!isfinite(query_weights[index]) ||
            (index > 0 && query_ids[index - 1U] >= query_ids[index]))
        {
            return EVOKE_ERR_INVALID;
        }
    }
    if (size == 0)
    {
        *score_out = 0.0f;
        *posting_count_out = 0;
        return EVOKE_OK;
    }
    if (size < sizeof(float))
    {
        return EVOKE_ERR_FORMAT;
    }
    scale = evoke_semantic_forward_read_f32(bytes);
    if (!isfinite(scale) || scale <= 0.0f)
    {
        return EVOKE_ERR_FORMAT;
    }
    cursor = sizeof(float);
    while (cursor < size)
    {
        uint32_t delta;
        uint32_t term_id;
        int32_t code;

        if (!evoke_semantic_forward_read_varint(
                bytes,
                size,
                &cursor,
                &delta
            ) ||
            cursor > size - code_width ||
            previous_term > UINT32_MAX - delta ||
            posting_count == UINT32_MAX)
        {
            return EVOKE_ERR_FORMAT;
        }
        term_id = previous_term + delta;
        code = (int8_t) bytes[cursor];
        cursor += code_width;
        if (code == INT8_MIN)
        {
            return EVOKE_ERR_FORMAT;
        }
        while (query < query_count && query_ids[query] < term_id)
        {
            query++;
        }
        if (query < query_count && query_ids[query] == term_id)
        {
            score += (double) code * scale * query_weights[query];
        }
        previous_term = term_id;
        posting_count++;
    }
    if (!isfinite(score) || score > FLT_MAX || score < -FLT_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    *score_out = (float) score;
    *posting_count_out = posting_count;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_forward_score_transposed_sorted(
    const uint8_t *bytes,
    size_t size,
    uint64_t expected_source_authority_checksum,
    const uint8_t *allowed_document_bitmap,
    size_t allowed_document_bitmap_size,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *scores_out,
    size_t score_count,
    uint64_t *postings_examined_out,
    uint64_t *logical_bytes_read_out
)
{
    evoke_semantic_forward_header header;
    double *scores = NULL;
    uint64_t postings_examined = 0;
    uint64_t logical_bytes = 0;
    uint32_t previous_offset = 0;
    evoke_status status;

    if (bytes == NULL || query_ids == NULL || query_weights == NULL ||
        scores_out == NULL || postings_examined_out == NULL ||
        logical_bytes_read_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *postings_examined_out = 0;
    *logical_bytes_read_out = 0;
    status = evoke_semantic_forward_header_deserialize(
        bytes,
        size,
        size,
        expected_source_authority_checksum,
        &header
    );
    if (status != EVOKE_OK ||
        header.format_version != EVOKE_SEMANTIC_FORWARD_VERSION ||
        evoke_semantic_forward_read_u64(
            bytes + EVOKE_SEMANTIC_FORWARD_CHECKSUM_OFFSET) !=
            evoke_semantic_forward_checksum(bytes, size) ||
        score_count < header.document_count ||
        (allowed_document_bitmap != NULL &&
         allowed_document_bitmap_size <
            ((size_t) header.document_count + 7U) / 8U))
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    for (size_t query = 0; query < query_count; query++)
    {
        if (!isfinite(query_weights[query]) ||
            query_ids[query] >= header.vocab_size ||
            (query > 0 && query_ids[query - 1U] >= query_ids[query]))
        {
            return EVOKE_ERR_INVALID;
        }
    }
    for (uint32_t term_id = 0; term_id <= header.vocab_size; term_id++)
    {
        uint32_t offset = evoke_semantic_forward_read_u32(
            bytes + header.transpose_term_offsets_offset +
                (size_t) term_id * sizeof(uint32_t)
        );

        if (offset < previous_offset ||
            offset > header.transpose_sparse_posting_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        previous_offset = offset;
    }
    if (previous_offset != header.transpose_sparse_posting_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t term_id = 0; term_id < header.vocab_size; term_id++)
    {
        uint32_t start = evoke_semantic_forward_read_u32(
            bytes + header.transpose_term_offsets_offset +
                (size_t) term_id * sizeof(uint32_t)
        );
        uint32_t end = evoke_semantic_forward_read_u32(
            bytes + header.transpose_term_offsets_offset +
                ((size_t) term_id + 1U) * sizeof(uint32_t)
        );
        uint16_t previous_document = 0;

        for (uint32_t posting = start; posting < end; posting++)
        {
            size_t entry_offset = header.transpose_postings_offset +
                (size_t) posting * header.transpose_entry_size;
            uint16_t document = evoke_semantic_forward_read_u16(
                bytes + entry_offset
            );
            int8_t code = (int8_t) bytes[
                entry_offset + sizeof(uint16_t)
            ];

            if (document >= header.document_count || code == INT8_MIN ||
                (posting > start && document < previous_document))
            {
                return EVOKE_ERR_FORMAT;
            }
            previous_document = document;
        }
    }
    for (uint32_t dense_index = 0;
         dense_index < header.transpose_dense_term_count;
         dense_index++)
    {
        uint32_t term_id = evoke_semantic_forward_read_u32(
            bytes + header.transpose_dense_term_ids_offset +
                (size_t) dense_index * sizeof(uint32_t)
        );
        uint32_t start;
        uint32_t end;

        if (term_id >= header.vocab_size ||
            (dense_index > 0 &&
             term_id <= evoke_semantic_forward_read_u32(
                bytes + header.transpose_dense_term_ids_offset +
                    (size_t) (dense_index - 1U) * sizeof(uint32_t))))
        {
            return EVOKE_ERR_FORMAT;
        }
        start = evoke_semantic_forward_read_u32(
            bytes + header.transpose_term_offsets_offset +
                (size_t) term_id * sizeof(uint32_t)
        );
        end = evoke_semantic_forward_read_u32(
            bytes + header.transpose_term_offsets_offset +
                ((size_t) term_id + 1U) * sizeof(uint32_t)
        );
        if (start != end)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t document = 0;
             document < header.document_count;
             document++)
        {
            int8_t code = (int8_t) bytes[
                header.transpose_dense_codes_offset +
                    (size_t) dense_index * header.document_count + document
            ];

            if (code == INT8_MIN)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    scores = calloc(header.document_count, sizeof(*scores));
    if (scores == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    memset(scores_out, 0, header.document_count * sizeof(*scores_out));
    logical_bytes = header.row_offsets_offset;
    logical_bytes += (uint64_t) header.document_count * sizeof(float);
    logical_bytes +=
        ((uint64_t) header.vocab_size + 1U) * sizeof(uint32_t);
    logical_bytes +=
        (uint64_t) header.transpose_dense_term_count * sizeof(uint32_t);
    for (size_t query = 0; query < query_count; query++)
    {
        uint32_t term_id = query_ids[query];
        uint32_t dense_low = 0;
        uint32_t dense_high = header.transpose_dense_term_count;
        uint32_t dense_index = UINT32_MAX;
        uint32_t start = evoke_semantic_forward_read_u32(
            bytes + header.transpose_term_offsets_offset +
                (size_t) term_id * sizeof(uint32_t)
        );
        uint32_t end = evoke_semantic_forward_read_u32(
            bytes + header.transpose_term_offsets_offset +
                ((size_t) term_id + 1U) * sizeof(uint32_t)
        );

        while (dense_low < dense_high)
        {
            uint32_t middle = dense_low + (dense_high - dense_low) / 2U;
            uint32_t dense_term = evoke_semantic_forward_read_u32(
                bytes + header.transpose_dense_term_ids_offset +
                    (size_t) middle * sizeof(uint32_t)
            );

            if (dense_term < term_id)
            {
                dense_low = middle + 1U;
            }
            else
            {
                dense_high = middle;
            }
        }
        if (dense_low < header.transpose_dense_term_count &&
            evoke_semantic_forward_read_u32(
                bytes + header.transpose_dense_term_ids_offset +
                    (size_t) dense_low * sizeof(uint32_t)) == term_id)
        {
            dense_index = dense_low;
        }
        if (dense_index != UINT32_MAX)
        {
            const uint8_t *dense_codes =
                bytes + header.transpose_dense_codes_offset +
                    (size_t) dense_index * header.document_count;

            if (UINT64_MAX - postings_examined < header.document_count ||
                UINT64_MAX - logical_bytes < header.document_count)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            postings_examined += header.document_count;
            logical_bytes += header.document_count;
            for (uint32_t document = 0;
                 document < header.document_count;
                 document++)
            {
                int8_t code = (int8_t) dense_codes[document];
                float scale;

                if (allowed_document_bitmap != NULL &&
                    (allowed_document_bitmap[document >> 3] &
                     (uint8_t) (UINT8_C(1) << (document & 7U))) == 0)
                {
                    continue;
                }
                if (code == 0)
                {
                    continue;
                }
                scale = evoke_semantic_forward_read_f32(
                    bytes + header.transpose_scales_offset +
                        (size_t) document * sizeof(float)
                );
                if (!isfinite(scale) || scale <= 0.0f)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto cleanup;
                }
                scores[document] +=
                    (double) code * scale * query_weights[query];
                if (!isfinite(scores[document]) ||
                    scores[document] > FLT_MAX ||
                    scores[document] < -FLT_MAX)
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
            }
            continue;
        }
        if (UINT64_MAX - postings_examined < (uint64_t) end - start ||
            UINT64_MAX - logical_bytes <
                ((uint64_t) end - start) * header.transpose_entry_size)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        postings_examined += (uint64_t) end - start;
        logical_bytes +=
            ((uint64_t) end - start) * header.transpose_entry_size;
        for (uint32_t posting = start; posting < end; posting++)
        {
            size_t entry_offset = header.transpose_postings_offset +
                (size_t) posting * header.transpose_entry_size;
            uint16_t document = evoke_semantic_forward_read_u16(
                bytes + entry_offset
            );
            int8_t code = (int8_t) bytes[
                entry_offset + sizeof(uint16_t)
            ];
            float scale;

            if (allowed_document_bitmap != NULL &&
                (allowed_document_bitmap[document >> 3] &
                 (uint8_t) (UINT8_C(1) << (document & 7U))) == 0)
            {
                continue;
            }
            scale = evoke_semantic_forward_read_f32(
                bytes + header.transpose_scales_offset +
                    (size_t) document * sizeof(float)
            );
            if (!isfinite(scale) || scale <= 0.0f)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            scores[document] +=
                (double) code * scale * query_weights[query];
            if (!isfinite(scores[document]) ||
                scores[document] > FLT_MAX ||
                scores[document] < -FLT_MAX)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
        }
    }
    for (uint32_t document = 0;
         document < header.document_count;
         document++)
    {
        scores_out[document] = (float) scores[document];
    }
    *postings_examined_out = postings_examined;
    *logical_bytes_read_out = logical_bytes;
    status = EVOKE_OK;

cleanup:
    free(scores);
    return status;
}

evoke_status
evoke_semantic_forward_chunk_score_sorted(
    const evoke_semantic_forward_chunk *chunk,
    uint32_t document_id,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out
)
{
    uint32_t row;
    uint32_t row_start;
    uint32_t row_end;

    if (chunk == NULL || query_ids == NULL || query_weights == NULL ||
        score_out == NULL ||
        document_id < chunk->first_document ||
        document_id >= chunk->first_document + chunk->document_count)
    {
        return EVOKE_ERR_INVALID;
    }
    row = document_id - chunk->first_document;
    row_start = chunk->row_offsets[row];
    row_end = chunk->row_offsets[row + 1U];
    return evoke_semantic_forward_score_row_sorted(
        row_end > row_start ? chunk->term_ids + row_start : NULL,
        row_end > row_start ? chunk->operations + row_start : NULL,
        row_end > row_start ? chunk->contributions + row_start : NULL,
        row_end - row_start,
        query_ids,
        query_weights,
        query_count,
        score_out
    );
}

evoke_status
evoke_semantic_forward_chunk_score(
    const evoke_semantic_forward_chunk *chunk,
    uint32_t document_id,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out
)
{
    uint32_t row;
    uint32_t row_start;
    uint32_t row_end;
    bool query_is_sorted = true;
    float score = 0.0f;

    if (chunk == NULL || query_ids == NULL || query_weights == NULL ||
        score_out == NULL ||
        document_id < chunk->first_document ||
        document_id >= chunk->first_document + chunk->document_count)
    {
        return EVOKE_ERR_INVALID;
    }
    row = document_id - chunk->first_document;
    row_start = chunk->row_offsets[row];
    row_end = chunk->row_offsets[row + 1U];
    for (size_t query = 0; query < query_count; query++)
    {
        if (!isfinite(query_weights[query]))
        {
            return EVOKE_ERR_INVALID;
        }
        if (query > 0 && query_ids[query - 1U] >= query_ids[query])
        {
            query_is_sorted = false;
        }
    }
    if (query_is_sorted)
    {
        return evoke_semantic_forward_chunk_score_sorted(
            chunk,
            document_id,
            query_ids,
            query_weights,
            query_count,
            score_out
        );
    }
    for (size_t query = 0; query < query_count; query++)
    {
        uint32_t left = row_start;
        uint32_t right = row_end;
        uint32_t posting;

        while (left < right)
        {
            uint32_t middle = left + (right - left) / 2U;

            if (chunk->term_ids[middle] < query_ids[query])
            {
                left = middle + 1U;
            }
            else
            {
                right = middle;
            }
        }
        posting = left;
        while (posting < row_end &&
               chunk->term_ids[posting] == query_ids[query])
        {
            if (chunk->operations[posting] ==
                EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT)
            {
                score += (float) (
                    chunk->contributions[posting] *
                    (double) query_weights[query]
                );
            }
            else
            {
                float impact = (float) chunk->contributions[posting];

                score += query_weights[query] * impact;
            }
            posting++;
        }
    }
    if (!isfinite(score) || score > FLT_MAX || score < -FLT_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    *score_out = (float) score;
    return EVOKE_OK;
}
