#ifndef EVOKE_AM_BUILD_H
#define EVOKE_AM_BUILD_H

#include "postgres.h"

#include "common/relpath.h"
#include "storage/buffile.h"
#include "storage/itemptr.h"
#include "utils/rel.h"

#include "evoke_core.h"
#include "evoke_semantic.h"
#include "evoke_segments.h"

typedef enum evoke_am_rebuild_builder
{
    EVOKE_AM_REBUILD_BUILDER_STANDARD = 0,
    EVOKE_AM_REBUILD_BUILDER_COMPACT,
    EVOKE_AM_REBUILD_BUILDER_SPILL,
    EVOKE_AM_REBUILD_BUILDER_SEMANTIC_STREAM
} evoke_am_rebuild_builder;

typedef struct evoke_am_rebuild_workload
{
    uint64 live_payload_bytes;
    uint64 identity_source_bytes;
    uint64 mutation_bytes;
} evoke_am_rebuild_workload;

typedef struct evoke_am_rebuild_output
{
    Oid source_type;
    ItemPointerData *doc_tids;
    size_t num_docs;
    uint8_t *index_bytes;
    size_t index_bytes_len;
    char semantic_signature[EVOKE_AM_SEMANTIC_SIGNATURE_LEN + 1];
    evoke_index index;
    bool index_valid;
    bool segment_sae;
    BufFile *segment_semantic_postings;
    size_t segment_semantic_posting_count;
    uint8 *segment_semantic_input_fingerprints;
    size_t segment_semantic_input_fingerprint_count;
    double heap_tuples;
    double index_tuples;
} evoke_am_rebuild_output;

void evoke_am_rebuild_output_release(
    evoke_am_rebuild_output *output
);

uint64 evoke_am_prepare_replacement_relation(
    Relation index_relation
);

void evoke_am_publish_replacement_segments(
    Relation index_relation,
    ForkNumber fork_number,
    evoke_am_rebuild_output *replacement,
    const uint8 contract_hash[EVOKE_SEGMENT_CONTRACT_HASH_BYTES],
    uint16 cache_epoch,
    uint16 flags,
    uint64 rebuild_count
);

bool evoke_am_rebuild_memory_budget_choose(
    bool semantic_streaming,
    const evoke_am_rebuild_workload *workload,
    int rebuild_memory_budget_mb,
    int work_mem_kb,
    evoke_am_rebuild_builder *builder_out,
    uint64 *standard_estimated_bytes_out,
    uint64 *compact_estimated_bytes_out,
    uint64 *spill_estimated_bytes_out,
    uint64 *budget_bytes_out
);

const char *evoke_am_rebuild_builder_name(
    evoke_am_rebuild_builder builder
);

#endif
