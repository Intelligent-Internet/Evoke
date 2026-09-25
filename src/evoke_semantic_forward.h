#ifndef EVOKE_SEMANTIC_FORWARD_H
#define EVOKE_SEMANTIC_FORWARD_H

#include <stddef.h>
#include <stdint.h>

#include "evoke_core.h"

#define EVOKE_SEMANTIC_FORWARD_TRANSPOSE_VERSION UINT16_C(6)
#define EVOKE_SEMANTIC_FORWARD_HEADER_BYTES 72U
#define EVOKE_SEMANTIC_FORWARD_TRANSPOSE_ENTRY_SIZE UINT32_C(3)

typedef enum evoke_semantic_forward_operation
{
    EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT = 1,
    EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT = 2
} evoke_semantic_forward_operation;

typedef struct evoke_semantic_forward_chunk
{
    uint64_t source_authority_checksum;
    uint32_t first_document;
    uint32_t document_count;
    uint32_t posting_count;
    uint32_t vocab_size;
    uint32_t *row_offsets;
    uint32_t *term_ids;
    uint8_t *operations;
    double *contributions;
} evoke_semantic_forward_chunk;

typedef struct evoke_semantic_forward_header
{
    uint64_t source_authority_checksum;
    size_t object_size;
    size_t row_offsets_offset;
    size_t row_data_offset;
    size_t row_data_end;
    size_t transpose_scales_offset;
    size_t transpose_term_offsets_offset;
    size_t transpose_postings_offset;
    size_t transpose_dense_term_ids_offset;
    size_t transpose_dense_codes_offset;
    uint32_t first_document;
    uint32_t document_count;
    uint32_t posting_count;
    uint32_t quantization_bits;
    uint32_t vocab_size;
    uint32_t transpose_entry_size;
    uint32_t transpose_sparse_posting_count;
    uint32_t transpose_dense_term_count;
    uint16_t format_version;
} evoke_semantic_forward_header;

void evoke_semantic_forward_chunk_init(
    evoke_semantic_forward_chunk *chunk
);

void evoke_semantic_forward_chunk_free(
    evoke_semantic_forward_chunk *chunk
);

evoke_status evoke_semantic_forward_chunk_validate(
    const evoke_semantic_forward_chunk *chunk
);

evoke_status evoke_semantic_forward_chunk_serialize(
    const evoke_semantic_forward_chunk *chunk,
    uint8_t **bytes_out,
    size_t *size_out
);

/* chunk_out must be initialized before this call. */
evoke_status evoke_semantic_forward_chunk_deserialize(
    const uint8_t *bytes,
    size_t size,
    uint64_t expected_source_authority_checksum,
    evoke_semantic_forward_chunk *chunk_out
);

size_t evoke_semantic_forward_header_size(void);

/*
 * Validate the fixed header and derive the serialized array offsets. The
 * enclosing page object remains responsible for authenticating partial range
 * reads; this function validates the semantic layout against object_size.
 */
evoke_status evoke_semantic_forward_header_deserialize(
    const uint8_t *bytes,
    size_t size,
    size_t object_size,
    uint64_t expected_source_authority_checksum,
    evoke_semantic_forward_header *header_out
);

/* Score one independently loaded serialized row. */
evoke_status evoke_semantic_forward_score_row_sorted(
    const uint32_t *term_ids,
    const uint8_t *operations,
    const double *contributions,
    uint32_t posting_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out
);

evoke_status evoke_semantic_forward_score_row(
    const uint32_t *term_ids,
    const uint8_t *operations,
    const double *contributions,
    uint32_t posting_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out
);

/* Score one current delta-varint/int8 row without materializing it. */
evoke_status evoke_semantic_forward_score_serialized_row_sorted(
    const uint8_t *bytes,
    size_t size,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out,
    uint32_t *posting_count_out
);

/*
 * Score a current v6 block-local term directory. High-DF terms use dense
 * int8 lanes only when that is no larger than sparse (document, code) pairs.
 * The bitmap uses chunk-local document IDs; NULL admits every document. The
 * result array contains one score per chunk-local document and is zeroed by
 * this function.
 */
evoke_status evoke_semantic_forward_score_transposed_sorted(
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
);

/*
 * Exact sparse dot product over root-stable unified additive contributions.
 * Query terms are accumulated in caller order, and same-term contributions
 * retain root authority order. Strictly increasing query IDs use a linear
 * merge; arbitrary caller order retains the exact binary-search path.
 * Contributions may be signed.
 */
evoke_status evoke_semantic_forward_chunk_score(
    const evoke_semantic_forward_chunk *chunk,
    uint32_t document_id,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out
);

/*
 * Fast path for a caller-validated query. IDs must be strictly increasing and
 * weights finite. This avoids repeating query validation for every candidate
 * document while retaining the same accumulation order as the generic API.
 */
evoke_status evoke_semantic_forward_chunk_score_sorted(
    const evoke_semantic_forward_chunk *chunk,
    uint32_t document_id,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out
);

#endif
