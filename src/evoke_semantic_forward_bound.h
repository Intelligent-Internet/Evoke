#ifndef EVOKE_SEMANTIC_FORWARD_BOUND_H
#define EVOKE_SEMANTIC_FORWARD_BOUND_H

#include <stddef.h>
#include <stdint.h>

#include "evoke_core.h"

#define EVOKE_SEMANTIC_FORWARD_BOUND_BLOCK_SHIFT UINT16_C(3)
#define EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD UINT32_C(64)
#define EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE 64U

typedef struct evoke_semantic_forward_bound_summary
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
} evoke_semantic_forward_bound_summary;

typedef evoke_status (*evoke_semantic_forward_bound_visitor)(
    void *context,
    uint32_t block_id,
    float maximum
);

evoke_status evoke_semantic_forward_bound_shard_serialize(
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
);

evoke_status evoke_semantic_forward_bound_header_deserialize(
    const uint8_t *bytes,
    size_t size,
    size_t object_size,
    uint64_t expected_source_authority_checksum,
    evoke_semantic_forward_bound_summary *summary_out
);

evoke_status evoke_semantic_forward_bound_term_slice(
    const evoke_semantic_forward_bound_summary *summary,
    const uint8_t *offset_bytes,
    size_t offset_size,
    uint32_t term_id,
    size_t *offset_out,
    size_t *size_out
);

size_t evoke_semantic_forward_bound_entry_size(
    uint32_t block_delta
);

evoke_status evoke_semantic_forward_bound_encode_entry(
    uint8_t *bytes,
    size_t capacity,
    uint32_t block_delta,
    float maximum,
    size_t *size_out
);

evoke_status evoke_semantic_forward_bound_visit(
    const uint8_t *bytes,
    size_t size,
    uint32_t block_count,
    evoke_semantic_forward_bound_visitor visitor,
    void *visitor_context,
    uint32_t *entry_count_out
);

#endif
