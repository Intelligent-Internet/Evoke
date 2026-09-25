#include "evoke_storage.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_MAGIC 0x424D3235U
#define EVOKE_FLAG_HAS_NONOCCURRENCE 0x0001U
#define EVOKE_FLAG_HAS_VOCAB 0x0002U
#define EVOKE_FLAG_HAS_EMPTY_TOKEN 0x0004U
#define EVOKE_FLAG_HAS_EXACT_STATS 0x0008U
#define EVOKE_FLAG_KNOWN_MASK                                           \
    (EVOKE_FLAG_HAS_NONOCCURRENCE | EVOKE_FLAG_HAS_VOCAB |         \
     EVOKE_FLAG_HAS_EMPTY_TOKEN | EVOKE_FLAG_HAS_EXACT_STATS)
#define EVOKE_STREAM_SCRATCH_BYTES 65536U

static void
evoke_write_u16_le(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t) (value & 0xFFU);
    dst[1] = (uint8_t) ((value >> 8) & 0xFFU);
}

static void
evoke_write_u32_le(uint8_t *dst, uint32_t value)
{
    dst[0] = (uint8_t) (value & 0xFFU);
    dst[1] = (uint8_t) ((value >> 8) & 0xFFU);
    dst[2] = (uint8_t) ((value >> 16) & 0xFFU);
    dst[3] = (uint8_t) ((value >> 24) & 0xFFU);
}

static void
evoke_write_u64_le(uint8_t *dst, uint64_t value)
{
    size_t i;

    for (i = 0; i < 8; i++)
    {
        dst[i] = (uint8_t) ((value >> (i * 8)) & 0xFFU);
    }
}

static void
evoke_write_f32_le(uint8_t *dst, float value)
{
    union
    {
        float f;
        uint32_t u;
    } conv;

    conv.f = value;
    evoke_write_u32_le(dst, conv.u);
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

    for (i = 0; i < 8; i++)
    {
        value |= ((uint64_t) src[i]) << (i * 8);
    }

    return value;
}

static float
evoke_read_f32_le(const uint8_t *src)
{
    union
    {
        float f;
        uint32_t u;
    } conv;

    conv.u = evoke_read_u32_le(src);
    return conv.f;
}

static bool
evoke_checked_add_size(size_t a, size_t b, size_t *out)
{
    if (a > ((size_t) -1) - b)
    {
        return false;
    }

    *out = a + b;
    return true;
}

static bool
evoke_checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (a == 0 || b == 0)
    {
        *out = 0;
        return true;
    }

    if (a > ((size_t) -1) / b)
    {
        return false;
    }

    *out = a * b;
    return true;
}

static bool
evoke_is_valid_method(evoke_method method)
{
    switch (method)
    {
        case EVOKE_METHOD_ROBERTSON:
        case EVOKE_METHOD_LUCENE:
        case EVOKE_METHOD_ATIRE:
        case EVOKE_METHOD_BM25L:
        case EVOKE_METHOD_BM25PLUS:
            return true;
    }

    return false;
}

static evoke_status
evoke_validate_header_values(
    const evoke_index *index,
    uint16_t flags
)
{
    if ((flags & ~EVOKE_FLAG_KNOWN_MASK) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (!evoke_is_valid_method(index->params.method) ||
        !evoke_is_valid_method(index->params.idf_method))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (!isfinite(index->params.k1) ||
        !isfinite(index->params.b) ||
        !isfinite(index->params.delta) ||
        index->params.k1 < 0.0f ||
        index->params.b < 0.0f ||
        index->params.b > 1.0f ||
        index->params.delta < 0.0f)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (index->data_len >= (uint64_t) UINT32_MAX)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (index->vocab_size == UINT32_MAX)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (index->has_empty_token && index->empty_token_id >= index->vocab_size)
    {
        return EVOKE_ERR_FORMAT;
    }

    return EVOKE_OK;
}

static evoke_status
evoke_validate_postings(const evoke_index *index)
{
    size_t i;
    size_t indptr_len = (size_t) index->vocab_size + 1;
    uint64_t last = 0;

    if (index->indptr == NULL ||
        (index->data_len > 0 &&
         (index->data == NULL || index->indices == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    if (index->indptr[0] != 0)
    {
        return EVOKE_ERR_FORMAT;
    }

    for (i = 0; i < indptr_len; i++)
    {
        uint64_t current = index->indptr[i];

        if (current < last || current > index->data_len)
        {
            return EVOKE_ERR_FORMAT;
        }
        last = current;
    }
    if (last != index->data_len)
    {
        return EVOKE_ERR_FORMAT;
    }

    for (i = 0; i < index->data_len; i++)
    {
        if (index->indices[i] >= index->num_docs)
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    return EVOKE_OK;
}

static bool
evoke_has_exact_stats(const evoke_index *index)
{
    bool has_any;

    has_any =
        index->term_frequencies != NULL ||
        index->doc_lengths != NULL ||
        index->doc_frequencies != NULL;
    return has_any &&
           (index->data_len == 0 || index->term_frequencies != NULL) &&
           (index->num_docs == 0 || index->doc_lengths != NULL) &&
           (index->vocab_size == 0 || index->doc_frequencies != NULL);
}

static evoke_status
evoke_validate_payload_values(const evoke_index *index, uint16_t flags)
{
    bool has_exact_stats;
    bool has_any_exact_stats;
    uint64_t i;

    if (index == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    has_exact_stats = evoke_has_exact_stats(index);
    has_any_exact_stats =
        index->term_frequencies != NULL ||
        index->doc_lengths != NULL ||
        index->doc_frequencies != NULL;
    if ((has_any_exact_stats && !has_exact_stats) ||
        (((flags & EVOKE_FLAG_HAS_EXACT_STATS) != 0) != has_exact_stats))
    {
        return EVOKE_ERR_INVALID;
    }
    if (((flags & EVOKE_FLAG_HAS_NONOCCURRENCE) != 0) !=
        (index->nonoccurrence != NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    if (((flags & EVOKE_FLAG_HAS_VOCAB) != 0) !=
        (index->vocab != NULL))
    {
        return EVOKE_ERR_INVALID;
    }

    for (i = 0; i < index->data_len; i++)
    {
        if (!isfinite(index->data[i]) ||
            (has_exact_stats && index->term_frequencies[i] == 0))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    for (i = 0; i < index->vocab_size; i++)
    {
        if (index->nonoccurrence != NULL &&
            !isfinite(index->nonoccurrence[i]))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (has_exact_stats &&
            (index->doc_frequencies[i] > index->num_docs ||
             index->doc_frequencies[i] !=
             index->indptr[i + 1] - index->indptr[i]))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (index->vocab != NULL && index->vocab[i] == NULL)
        {
            return EVOKE_ERR_INVALID;
        }
    }

    return EVOKE_OK;
}

static uint16_t
evoke_serialized_flags(const evoke_index *index)
{
    uint16_t flags = 0;

    if (index->nonoccurrence != NULL)
    {
        flags |= EVOKE_FLAG_HAS_NONOCCURRENCE;
    }
    if (index->vocab != NULL)
    {
        flags |= EVOKE_FLAG_HAS_VOCAB;
    }
    if (index->has_empty_token)
    {
        flags |= EVOKE_FLAG_HAS_EMPTY_TOKEN;
    }
    if (evoke_has_exact_stats(index))
    {
        flags |= EVOKE_FLAG_HAS_EXACT_STATS;
    }
    return flags;
}

static evoke_status
evoke_stream_f32_array(
    evoke_stream_write_cb write_cb,
    void *write_ctx,
    const float *values,
    uint64_t count
)
{
    uint8_t scratch[EVOKE_STREAM_SCRATCH_BYTES];
    uint64_t pos = 0;

    while (pos < count)
    {
        size_t i;
        size_t chunk = (size_t) (count - pos);
        evoke_status status;

        if (chunk > sizeof(scratch) / sizeof(float))
        {
            chunk = sizeof(scratch) / sizeof(float);
        }
        for (i = 0; i < chunk; i++)
        {
            evoke_write_f32_le(
                scratch + i * sizeof(float),
                values[pos + i]
            );
        }
        status = write_cb(write_ctx, scratch, chunk * sizeof(float));
        if (status != EVOKE_OK)
        {
            return status;
        }
        pos += chunk;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_stream_u32_array(
    evoke_stream_write_cb write_cb,
    void *write_ctx,
    const uint32_t *values,
    uint64_t count
)
{
    uint8_t scratch[EVOKE_STREAM_SCRATCH_BYTES];
    uint64_t pos = 0;

    while (pos < count)
    {
        size_t i;
        size_t chunk = (size_t) (count - pos);
        evoke_status status;

        if (chunk > sizeof(scratch) / sizeof(uint32_t))
        {
            chunk = sizeof(scratch) / sizeof(uint32_t);
        }
        for (i = 0; i < chunk; i++)
        {
            evoke_write_u32_le(
                scratch + i * sizeof(uint32_t),
                values[pos + i]
            );
        }
        status = write_cb(write_ctx, scratch, chunk * sizeof(uint32_t));
        if (status != EVOKE_OK)
        {
            return status;
        }
        pos += chunk;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_stream_u64_array(
    evoke_stream_write_cb write_cb,
    void *write_ctx,
    const uint64_t *values,
    uint64_t count
)
{
    uint8_t scratch[EVOKE_STREAM_SCRATCH_BYTES];
    uint64_t pos = 0;

    while (pos < count)
    {
        size_t i;
        size_t chunk = (size_t) (count - pos);
        evoke_status status;

        if (chunk > sizeof(scratch) / sizeof(uint64_t))
        {
            chunk = sizeof(scratch) / sizeof(uint64_t);
        }
        for (i = 0; i < chunk; i++)
        {
            evoke_write_u64_le(
                scratch + i * sizeof(uint64_t),
                values[pos + i]
            );
        }
        status = write_cb(write_ctx, scratch, chunk * sizeof(uint64_t));
        if (status != EVOKE_OK)
        {
            return status;
        }
        pos += chunk;
    }
    return EVOKE_OK;
}

evoke_status
evoke_serialized_index_size(
    const evoke_index *index,
    size_t *len_out
)
{
    uint16_t flags;
    size_t len = EVOKE_HEADER_SIZE;
    size_t chunk = 0;
    uint32_t i;

    if (index == NULL || len_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *len_out = 0;

    flags = evoke_serialized_flags(index);
    if (evoke_validate_header_values(index, flags) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    {
        evoke_status status = evoke_validate_postings(index);

        if (status != EVOKE_OK)
        {
            return status;
        }
        status = evoke_validate_payload_values(index, flags);
        if (status != EVOKE_OK)
        {
            return status;
        }
    }

    if (!evoke_checked_mul_size(
            (size_t) index->data_len,
            sizeof(*index->data),
            &chunk) ||
        !evoke_checked_add_size(len, chunk, &len))
    {
        return EVOKE_ERR_RANGE;
    }
    if (!evoke_checked_mul_size(
            (size_t) index->data_len,
            sizeof(*index->indices),
            &chunk) ||
        !evoke_checked_add_size(len, chunk, &len))
    {
        return EVOKE_ERR_RANGE;
    }
    if (!evoke_checked_mul_size(
            (size_t) index->vocab_size + 1,
            sizeof(*index->indptr),
            &chunk) ||
        !evoke_checked_add_size(len, chunk, &len))
    {
        return EVOKE_ERR_RANGE;
    }
    if (index->nonoccurrence != NULL)
    {
        if (!evoke_checked_mul_size(
                (size_t) index->vocab_size,
                sizeof(*index->nonoccurrence),
                &chunk) ||
            !evoke_checked_add_size(len, chunk, &len))
        {
            return EVOKE_ERR_RANGE;
        }
    }
    if ((flags & EVOKE_FLAG_HAS_EXACT_STATS) != 0)
    {
        if (!evoke_checked_mul_size(
                (size_t) index->data_len,
                sizeof(*index->term_frequencies),
                &chunk) ||
            !evoke_checked_add_size(len, chunk, &len))
        {
            return EVOKE_ERR_RANGE;
        }
        if (!evoke_checked_mul_size(
                (size_t) index->num_docs,
                sizeof(*index->doc_lengths),
                &chunk) ||
            !evoke_checked_add_size(len, chunk, &len))
        {
            return EVOKE_ERR_RANGE;
        }
        if (!evoke_checked_mul_size(
                (size_t) index->vocab_size,
                sizeof(*index->doc_frequencies),
                &chunk) ||
            !evoke_checked_add_size(len, chunk, &len))
        {
            return EVOKE_ERR_RANGE;
        }
    }
    if (index->vocab != NULL)
    {
        for (i = 0; i < index->vocab_size; i++)
        {
            size_t token_len = strlen(index->vocab[i]);
            if (token_len > UINT32_MAX ||
                !evoke_checked_add_size(len, sizeof(uint32_t), &len) ||
                !evoke_checked_add_size(len, token_len, &len))
            {
                return EVOKE_ERR_RANGE;
            }
        }
    }

    *len_out = len;
    return EVOKE_OK;
}

static evoke_status
evoke_parse_header(
    const uint8_t *bytes,
    size_t len,
    uint16_t *flags_out,
    evoke_index *index_out
)
{
    uint32_t magic;
    uint16_t version;

    if (len < EVOKE_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }

    magic = evoke_read_u32_le(bytes);
    version = evoke_read_u16_le(bytes + 4);
    if (magic != EVOKE_MAGIC ||
        version != EVOKE_STORAGE_CURRENT_VERSION)
    {
        return EVOKE_ERR_FORMAT;
    }

    *flags_out = evoke_read_u16_le(bytes + 6);
    index_out->params.method =
        (evoke_method) evoke_read_u32_le(bytes + 8);
    index_out->params.idf_method =
        (evoke_method) evoke_read_u32_le(bytes + 12);
    index_out->params.k1 = evoke_read_f32_le(bytes + 16);
    index_out->params.b = evoke_read_f32_le(bytes + 20);
    index_out->params.delta = evoke_read_f32_le(bytes + 24);
    index_out->num_docs = evoke_read_u32_le(bytes + 28);
    index_out->vocab_size = evoke_read_u32_le(bytes + 32);
    index_out->data_len = evoke_read_u64_le(bytes + 36);
    index_out->empty_token_id = evoke_read_u32_le(bytes + 44);
    index_out->has_empty_token =
        ((*flags_out & EVOKE_FLAG_HAS_EMPTY_TOKEN) != 0);

    return evoke_validate_header_values(index_out, *flags_out);
}

typedef struct evoke_buffer_writer
{
    uint8_t *ptr;
} evoke_buffer_writer;

static evoke_status
evoke_write_to_buffer(void *ctx, const uint8_t *bytes, size_t len)
{
    evoke_buffer_writer *writer = ctx;

    memcpy(writer->ptr, bytes, len);
    writer->ptr += len;
    return EVOKE_OK;
}

evoke_status
evoke_serialize_index(
    const evoke_index *index,
    uint8_t **bytes_out,
    size_t *len_out
)
{
    size_t len = 0;
    uint8_t *bytes;
    evoke_buffer_writer writer;
    evoke_status status;

    if (index == NULL || bytes_out == NULL || len_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    *bytes_out = NULL;
    *len_out = 0;

    status = evoke_serialized_index_size(index, &len);
    if (status != EVOKE_OK)
    {
        return status;
    }
    bytes = malloc(len);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    memset(bytes, 0, len);

    writer.ptr = bytes;
    status = evoke_serialize_index_stream(
        index,
        evoke_write_to_buffer,
        &writer,
        &len
    );
    if (status != EVOKE_OK)
    {
        free(bytes);
        return status;
    }

    *bytes_out = bytes;
    *len_out = len;
    return EVOKE_OK;
}

evoke_status
evoke_serialize_index_stream(
    const evoke_index *index,
    evoke_stream_write_cb write_cb,
    void *write_ctx,
    size_t *len_out
)
{
    uint16_t flags;
    uint8_t scratch[EVOKE_HEADER_SIZE];
    uint32_t i;
    evoke_status status;
    size_t len = 0;

    if (index == NULL || write_cb == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_serialized_index_size(index, &len);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (len_out != NULL)
    {
        *len_out = len;
    }

    flags = evoke_serialized_flags(index);
    memset(scratch, 0, sizeof(scratch));
    evoke_write_u32_le(scratch, EVOKE_MAGIC);
    evoke_write_u16_le(scratch + 4, EVOKE_STORAGE_CURRENT_VERSION);
    evoke_write_u16_le(scratch + 6, flags);
    evoke_write_u32_le(scratch + 8, (uint32_t) index->params.method);
    evoke_write_u32_le(scratch + 12, (uint32_t) index->params.idf_method);
    evoke_write_f32_le(scratch + 16, index->params.k1);
    evoke_write_f32_le(scratch + 20, index->params.b);
    evoke_write_f32_le(scratch + 24, index->params.delta);
    evoke_write_u32_le(scratch + 28, index->num_docs);
    evoke_write_u32_le(scratch + 32, index->vocab_size);
    evoke_write_u64_le(scratch + 36, index->data_len);
    evoke_write_u32_le(scratch + 44, index->empty_token_id);
    status = write_cb(write_ctx, scratch, EVOKE_HEADER_SIZE);
    if (status != EVOKE_OK)
    {
        return status;
    }

    status = evoke_stream_f32_array(
        write_cb,
        write_ctx,
        index->data,
        index->data_len
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_stream_u32_array(
        write_cb,
        write_ctx,
        index->indices,
        index->data_len
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_stream_u64_array(
        write_cb,
        write_ctx,
        index->indptr,
        (uint64_t) index->vocab_size + 1
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if ((flags & EVOKE_FLAG_HAS_EXACT_STATS) != 0)
    {
        status = evoke_stream_u32_array(
            write_cb,
            write_ctx,
            index->term_frequencies,
            index->data_len
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        status = evoke_stream_u32_array(
            write_cb,
            write_ctx,
            index->doc_lengths,
            index->num_docs
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        status = evoke_stream_u32_array(
            write_cb,
            write_ctx,
            index->doc_frequencies,
            index->vocab_size
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    if (index->nonoccurrence != NULL)
    {
        status = evoke_stream_f32_array(
            write_cb,
            write_ctx,
            index->nonoccurrence,
            index->vocab_size
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    if (index->vocab != NULL)
    {
        for (i = 0; i < index->vocab_size; i++)
        {
            size_t token_len = strlen(index->vocab[i]);
            evoke_write_u32_le(scratch, (uint32_t) token_len);
            status = write_cb(write_ctx, scratch, sizeof(uint32_t));
            if (status != EVOKE_OK)
            {
                return status;
            }
            status = write_cb(
                write_ctx,
                (const uint8_t *) index->vocab[i],
                token_len
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
        }
    }

    return EVOKE_OK;
}

evoke_status
evoke_deserialize_index(
    const uint8_t *bytes,
    size_t len,
    evoke_index *index_out
)
{
    uint16_t flags = 0;
    const uint8_t *ptr;
    size_t remaining;
    size_t chunk;
    uint32_t i;
    evoke_status status;

    if (bytes == NULL || index_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    evoke_index_init(index_out);
    status = evoke_parse_header(bytes, len, &flags, index_out);
    if (status != EVOKE_OK)
    {
        return status;
    }

    ptr = bytes + EVOKE_HEADER_SIZE;
    remaining = len - EVOKE_HEADER_SIZE;

    if (!evoke_checked_mul_size(
            (size_t) index_out->data_len,
            sizeof(*index_out->data),
            &chunk) ||
        chunk > remaining)
    {
        evoke_index_free(index_out);
        return EVOKE_ERR_FORMAT;
    }
    if (chunk > 0)
    {
        index_out->data = malloc(chunk);
        if (index_out->data == NULL)
        {
            evoke_index_free(index_out);
            return EVOKE_ERR_NOMEM;
        }
        for (i = 0; i < index_out->data_len; i++)
        {
            index_out->data[i] = evoke_read_f32_le(ptr);
            ptr += sizeof(float);
        }
        remaining -= chunk;
    }

    if (!evoke_checked_mul_size(
            (size_t) index_out->data_len,
            sizeof(*index_out->indices),
            &chunk) ||
        chunk > remaining)
    {
        evoke_index_free(index_out);
        return EVOKE_ERR_FORMAT;
    }
    if (chunk > 0)
    {
        index_out->indices = malloc(chunk);
        if (index_out->indices == NULL)
        {
            evoke_index_free(index_out);
            return EVOKE_ERR_NOMEM;
        }
        for (i = 0; i < index_out->data_len; i++)
        {
            index_out->indices[i] = evoke_read_u32_le(ptr);
            ptr += sizeof(uint32_t);
        }
        remaining -= chunk;
    }

    if (!evoke_checked_mul_size(
            (size_t) index_out->vocab_size + 1,
            sizeof(*index_out->indptr),
            &chunk) ||
        chunk > remaining)
    {
        evoke_index_free(index_out);
        return EVOKE_ERR_FORMAT;
    }
    index_out->indptr = malloc(chunk);
    if (index_out->indptr == NULL)
    {
        evoke_index_free(index_out);
        return EVOKE_ERR_NOMEM;
    }
    for (i = 0; i < (size_t) index_out->vocab_size + 1; i++)
    {
        index_out->indptr[i] = evoke_read_u64_le(ptr);
        ptr += sizeof(uint64_t);
    }
    remaining -= chunk;

    status = evoke_validate_postings(index_out);
    if (status != EVOKE_OK)
    {
        evoke_index_free(index_out);
        return status;
    }

    if ((flags & EVOKE_FLAG_HAS_EXACT_STATS) != 0)
    {
        if (!evoke_checked_mul_size(
                (size_t) index_out->data_len,
                sizeof(*index_out->term_frequencies),
                &chunk) ||
            chunk > remaining)
        {
            evoke_index_free(index_out);
            return EVOKE_ERR_FORMAT;
        }
        if (chunk > 0)
        {
            index_out->term_frequencies = malloc(chunk);
            if (index_out->term_frequencies == NULL)
            {
                evoke_index_free(index_out);
                return EVOKE_ERR_NOMEM;
            }
            for (i = 0; i < index_out->data_len; i++)
            {
                index_out->term_frequencies[i] = evoke_read_u32_le(ptr);
                ptr += sizeof(uint32_t);
            }
            remaining -= chunk;
        }

        if (!evoke_checked_mul_size(
                (size_t) index_out->num_docs,
                sizeof(*index_out->doc_lengths),
                &chunk) ||
            chunk > remaining)
        {
            evoke_index_free(index_out);
            return EVOKE_ERR_FORMAT;
        }
        if (chunk > 0)
        {
            index_out->doc_lengths = malloc(chunk);
            if (index_out->doc_lengths == NULL)
            {
                evoke_index_free(index_out);
                return EVOKE_ERR_NOMEM;
            }
            for (i = 0; i < index_out->num_docs; i++)
            {
                index_out->doc_lengths[i] = evoke_read_u32_le(ptr);
                ptr += sizeof(uint32_t);
            }
            remaining -= chunk;
        }

        if (!evoke_checked_mul_size(
                (size_t) index_out->vocab_size,
                sizeof(*index_out->doc_frequencies),
                &chunk) ||
            chunk > remaining)
        {
            evoke_index_free(index_out);
            return EVOKE_ERR_FORMAT;
        }
        if (chunk > 0)
        {
            index_out->doc_frequencies = malloc(chunk);
            if (index_out->doc_frequencies == NULL)
            {
                evoke_index_free(index_out);
                return EVOKE_ERR_NOMEM;
            }
            for (i = 0; i < index_out->vocab_size; i++)
            {
                index_out->doc_frequencies[i] = evoke_read_u32_le(ptr);
                ptr += sizeof(uint32_t);
            }
            remaining -= chunk;
        }
    }

    if ((flags & EVOKE_FLAG_HAS_NONOCCURRENCE) != 0)
    {
        if (!evoke_checked_mul_size(
                (size_t) index_out->vocab_size,
                sizeof(*index_out->nonoccurrence),
                &chunk) ||
            chunk > remaining)
        {
            evoke_index_free(index_out);
            return EVOKE_ERR_FORMAT;
        }

        if (chunk > 0)
        {
            index_out->nonoccurrence = malloc(chunk);
            if (index_out->nonoccurrence == NULL)
            {
                evoke_index_free(index_out);
                return EVOKE_ERR_NOMEM;
            }
            for (i = 0; i < index_out->vocab_size; i++)
            {
                index_out->nonoccurrence[i] = evoke_read_f32_le(ptr);
                ptr += sizeof(float);
            }
            remaining -= chunk;
        }
    }

    if ((flags & EVOKE_FLAG_HAS_VOCAB) != 0)
    {
        index_out->vocab = calloc(index_out->vocab_size, sizeof(*index_out->vocab));
        if (index_out->vocab == NULL)
        {
            evoke_index_free(index_out);
            return EVOKE_ERR_NOMEM;
        }

        for (i = 0; i < index_out->vocab_size; i++)
        {
            uint32_t token_len;

            if (remaining < sizeof(uint32_t))
            {
                evoke_index_free(index_out);
                return EVOKE_ERR_FORMAT;
            }
            token_len = evoke_read_u32_le(ptr);
            ptr += sizeof(uint32_t);
            remaining -= sizeof(uint32_t);

            if (token_len > remaining)
            {
                evoke_index_free(index_out);
                return EVOKE_ERR_FORMAT;
            }
            if (memchr(ptr, '\0', token_len) != NULL)
            {
                evoke_index_free(index_out);
                return EVOKE_ERR_FORMAT;
            }

            index_out->vocab[i] = malloc((size_t) token_len + 1);
            if (index_out->vocab[i] == NULL)
            {
                evoke_index_free(index_out);
                return EVOKE_ERR_NOMEM;
            }
            memcpy(index_out->vocab[i], ptr, token_len);
            index_out->vocab[i][token_len] = '\0';
            ptr += token_len;
            remaining -= token_len;
        }
    }

    if (remaining != 0)
    {
        evoke_index_free(index_out);
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_validate_payload_values(index_out, flags);
    if (status != EVOKE_OK)
    {
        evoke_index_free(index_out);
        return EVOKE_ERR_FORMAT;
    }

    return EVOKE_OK;
}

evoke_status
evoke_peek_serialized_meta(
    const uint8_t *bytes,
    size_t len,
    uint32_t *num_docs_out,
    uint32_t *vocab_size_out,
    uint64_t *data_len_out
)
{
    evoke_index meta;
    uint16_t flags = 0;
    evoke_status status;

    if (bytes == NULL || num_docs_out == NULL || vocab_size_out == NULL ||
        data_len_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }

    evoke_index_init(&meta);
    status = evoke_parse_header(bytes, len, &flags, &meta);
    if (status != EVOKE_OK)
    {
        return status;
    }

    *num_docs_out = meta.num_docs;
    *vocab_size_out = meta.vocab_size;
    *data_len_out = meta.data_len;
    return EVOKE_OK;
}
