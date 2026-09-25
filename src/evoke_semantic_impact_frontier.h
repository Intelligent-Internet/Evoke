#ifndef EVOKE_SEMANTIC_IMPACT_FRONTIER_H
#define EVOKE_SEMANTIC_IMPACT_FRONTIER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "evoke_core.h"
#include "evoke_semantic_bmp.h"

/* Research codec only. It is not registered as a segment root object. */
#define EVOKE_SEMANTIC_IMPACT_FRONTIER_BLOCK_SHIFT UINT16_C(6)
#define EVOKE_SEMANTIC_IMPACT_FRONTIER_BLOCK_SIZE UINT32_C(64)
#define EVOKE_SEMANTIC_IMPACT_FRONTIER_TERMS_PER_SHARD UINT32_C(64)
#define EVOKE_SEMANTIC_IMPACT_FRONTIER_HEADER_SIZE 64U
#define EVOKE_SEMANTIC_IMPACT_FRONTIER_LOCAL_DOCUMENT_SIZE 1U

typedef struct evoke_semantic_impact_frontier_summary
{
    uint64_t source_authority_checksum;
    size_t object_size;
    size_t offsets_offset;
    size_t payload_offset;
    uint32_t first_term;
    uint32_t term_count;
    uint32_t document_count;
    uint32_t vocab_size;
    uint16_t block_shift;
    evoke_semantic_impact_precision impact_precision;
} evoke_semantic_impact_frontier_summary;

typedef struct evoke_semantic_impact_frontier_cursor
{
    const uint8_t *bytes;
    size_t size;
    size_t offset;
    uint32_t block_count;
    uint32_t previous_block_id;
    uint8_t posting_width;
    evoke_semantic_impact_precision impact_precision;
    bool has_previous;
} evoke_semantic_impact_frontier_cursor;

typedef struct evoke_semantic_impact_frontier_block
{
    uint32_t block_id;
    uint32_t posting_count;
    const uint8_t *postings;
    uint8_t posting_width;
    evoke_semantic_impact_precision impact_precision;
} evoke_semantic_impact_frontier_block;

evoke_status evoke_semantic_impact_frontier_round_up(
    double impact,
    float *impact_out
);

size_t evoke_semantic_impact_frontier_block_size(
    uint32_t block_delta,
    uint32_t posting_count,
    evoke_semantic_impact_precision impact_precision
);

evoke_status evoke_semantic_impact_frontier_encode_block(
    uint8_t *bytes,
    size_t capacity,
    uint32_t block_delta,
    const uint8_t *local_documents,
    const double *impacts,
    uint32_t posting_count,
    evoke_semantic_impact_precision impact_precision,
    size_t *size_out
);

evoke_status evoke_semantic_impact_frontier_shard_serialize(
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
);

evoke_status evoke_semantic_impact_frontier_header_deserialize(
    const uint8_t *bytes,
    size_t size,
    size_t object_size,
    uint64_t expected_source_authority_checksum,
    evoke_semantic_impact_frontier_summary *summary_out
);

evoke_status evoke_semantic_impact_frontier_term_slice(
    const evoke_semantic_impact_frontier_summary *summary,
    const uint8_t *offset_bytes,
    size_t offset_size,
    uint32_t term_id,
    size_t *offset_out,
    size_t *size_out
);

evoke_status evoke_semantic_impact_frontier_cursor_init(
    evoke_semantic_impact_frontier_cursor *cursor,
    const uint8_t *bytes,
    size_t size,
    uint32_t block_count,
    evoke_semantic_impact_precision impact_precision
);

evoke_status evoke_semantic_impact_frontier_cursor_next(
    evoke_semantic_impact_frontier_cursor *cursor,
    evoke_semantic_impact_frontier_block *block_out,
    bool *has_block_out
);

evoke_status evoke_semantic_impact_frontier_block_posting(
    const evoke_semantic_impact_frontier_block *block,
    uint32_t posting_index,
    uint8_t *local_document_out,
    float *impact_out
);

#endif
