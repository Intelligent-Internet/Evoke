#ifndef EVOKE_SEMANTIC_BMP_H
#define EVOKE_SEMANTIC_BMP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "evoke_core.h"

#define EVOKE_SEMANTIC_BMP_BLOCK_SHIFT 4U
#define EVOKE_SEMANTIC_BMP_SUPERBLOCK_SHIFT 8U
#define EVOKE_SEMANTIC_BMP_FORMAT_VERSION 2U
#define EVOKE_SEMANTIC_BMP_HEADER_SIZE 136U
#define EVOKE_SEMANTIC_BMP_TERM_SIZE 28U
#define EVOKE_SEMANTIC_BMP_REF_SIZE 8U
#define EVOKE_SEMANTIC_BMP_SUPER_REF_SIZE 16U
#define EVOKE_SEMANTIC_BMP_BLOCK_SIZE 12U
#define EVOKE_SEMANTIC_BMP_RECORD_SIZE 12U
#define EVOKE_SEMANTIC_BMP_PACKED_TERM_SIZE 52U
#define EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT 6U
#define EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT 10U
#define EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS 64U
#define EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE 11U
#define EVOKE_SEMANTIC_BMP_PACKED_SUPER_REF_SIZE 16U
#define EVOKE_SEMANTIC_BMP_PACKED_FORMAT_VERSION 3U

typedef enum evoke_semantic_impact_precision
{
    EVOKE_SEMANTIC_IMPACT_PRECISION_F32 = 32,
    EVOKE_SEMANTIC_IMPACT_PRECISION_FP16 = 16,
    EVOKE_SEMANTIC_IMPACT_PRECISION_U8 = 8
} evoke_semantic_impact_precision;

typedef struct evoke_semantic_bmp_disk_header
{
    uint32_t document_count;
    uint32_t block_shift;
    uint32_t block_count;
    uint32_t superblock_shift;
    uint32_t superblock_count;
    uint32_t active_block_count;
    uint32_t term_count;
    uint32_t ref_count;
    uint32_t super_ref_count;
    uint32_t record_count;
    uint64_t posting_count;
    uint64_t terms_offset;
    uint64_t super_refs_offset;
    uint64_t refs_offset;
    uint64_t blocks_offset;
    uint64_t records_offset;
    uint64_t impacts_offset;
    uint64_t total_size;
    uint64_t checksum;
} evoke_semantic_bmp_disk_header;

typedef struct evoke_semantic_bmp_posting
{
    uint32_t term_id;
    uint32_t document_id;
    float impact;
} evoke_semantic_bmp_posting;

/* One term-local, document-ordered source run for bounded-memory builds. */
typedef struct evoke_semantic_bmp_run
{
    uint32_t term_id;
    uint64_t posting_count;
    const uint32_t *local_document_ids;
    const evoke_posting_value *values;
    const uint32_t *document_id_map;
    uint32_t document_id_base;
    uint32_t local_document_count;
} evoke_semantic_bmp_run;

typedef struct evoke_semantic_bmp_term
{
    uint32_t term_id;
    uint32_t first_ref;
    uint32_t ref_count;
    uint32_t first_super_ref;
    uint32_t super_ref_count;
    float min_impact;
    float max_impact;
} evoke_semantic_bmp_term;

typedef struct evoke_semantic_bmp_ref
{
    uint32_t block_id;
    uint32_t record_index;
    float min_impact;
    float max_impact;
} evoke_semantic_bmp_ref;

typedef struct evoke_semantic_bmp_super_ref
{
    uint32_t superblock_id;
    uint32_t first_ref;
    uint16_t ref_count;
    uint16_t reserved;
    float min_impact;
    float max_impact;
} evoke_semantic_bmp_super_ref;

typedef struct evoke_semantic_bmp_block
{
    uint32_t block_id;
    uint32_t first_record;
    uint32_t record_count;
} evoke_semantic_bmp_block;

typedef struct evoke_semantic_bmp_record
{
    uint32_t term_id;
    uint32_t first_impact;
    uint64_t document_mask;
    uint16_t impact_count;
} evoke_semantic_bmp_record;

typedef struct evoke_semantic_bmp_index
{
    uint32_t document_count;
    uint32_t block_shift;
    uint32_t block_count;
    uint32_t superblock_shift;
    uint32_t superblock_count;
    uint32_t active_block_count;
    uint32_t term_count;
    uint32_t ref_count;
    uint32_t super_ref_count;
    uint32_t record_count;
    uint64_t posting_count;
    evoke_semantic_bmp_term *terms;
    evoke_semantic_bmp_super_ref *super_refs;
    evoke_semantic_bmp_ref *refs;
    evoke_semantic_bmp_block *blocks;
    evoke_semantic_bmp_record *records;
    float *impacts;
} evoke_semantic_bmp_index;

typedef struct evoke_semantic_bmp_stats
{
    uint64_t bound_entries_visited;
    uint64_t super_bound_entries_visited;
    uint64_t superblocks_with_positive_bound;
    uint64_t superblocks_scored;
    uint64_t superblocks_skipped;
    uint64_t blocks_with_positive_bound;
    uint64_t blocks_scored;
    uint64_t blocks_skipped;
    uint64_t forward_records_examined;
    uint64_t postings_examined;
    uint64_t adaptive_fallbacks;
    uint64_t seed_documents_offered;
    uint64_t approximate_prune_events;
    uint64_t norm_bound_reductions;
    float max_approximate_skipped_bound;
    float final_kth_score;
} evoke_semantic_bmp_stats;

/* Exact document scores used only to establish an initial top-k threshold. */
typedef struct evoke_semantic_bmp_seed
{
    uint32_t document_id;
    float score;
} evoke_semantic_bmp_seed;

/* Optional exact Hölder bounds used by the representation oracle. */
typedef struct evoke_semantic_bmp_norm_bounds
{
    const float *block_max_l1;
    const float *block_max_l2;
    const float *superblock_max_l1;
    const float *superblock_max_l2;
    uint32_t block_count;
    uint32_t superblock_count;
} evoke_semantic_bmp_norm_bounds;

typedef struct evoke_semantic_bmp_packed_term
{
    uint32_t term_id;
    uint32_t first_ref;
    uint32_t ref_count;
    uint32_t first_super_ref;
    uint32_t super_ref_count;
    uint32_t first_doc_byte;
    uint32_t posting_count;
    uint32_t first_document;
    uint32_t doc_delta_width;
    uint32_t first_block_membership_byte;
    uint32_t block_membership_bytes;
    float min_impact;
    float max_impact;
} evoke_semantic_bmp_packed_term;

typedef struct evoke_semantic_bmp_packed_disk_header
{
    uint32_t document_count;
    uint32_t block_count;
    uint32_t superblock_count;
    uint32_t term_count;
    uint32_t ref_count;
    uint32_t super_ref_count;
    uint32_t record_count;
    uint32_t super_ref_bytes;
    uint32_t ref_bytes;
    uint32_t block_membership_bytes;
    uint32_t doc_delta_bytes;
    uint64_t posting_count;
    uint64_t terms_offset;
    uint64_t super_refs_offset;
    uint64_t refs_offset;
    uint64_t block_membership_offset;
    uint64_t doc_deltas_offset;
    uint64_t impacts_offset;
    uint64_t total_size;
    uint64_t checksum;
    evoke_semantic_impact_precision impact_precision;
} evoke_semantic_bmp_packed_disk_header;

typedef struct evoke_semantic_bmp_packed_super_ref
{
    uint32_t superblock_id;
    uint32_t first_ref;
    uint32_t first_impact;
    uint16_t ref_count;
    float min_impact;
    float max_impact;
} evoke_semantic_bmp_packed_super_ref;

typedef struct evoke_semantic_bmp_packed_ref
{
    uint8_t local_block_id;
    uint64_t document_mask;
    float min_impact;
    float max_impact;
} evoke_semantic_bmp_packed_ref;

typedef struct evoke_semantic_bmp_packed_index
{
    uint32_t document_count;
    uint32_t block_count;
    uint32_t superblock_count;
    uint32_t term_count;
    uint32_t ref_count;
    uint32_t super_ref_count;
    uint32_t record_count;
    uint64_t posting_count;
    evoke_semantic_bmp_packed_term *terms;
    uint8_t *super_refs;
    uint32_t super_ref_bytes;
    uint8_t *refs;
    uint32_t ref_bytes;
    uint8_t *block_membership;
    uint32_t block_membership_bytes;
    uint8_t *doc_deltas;
    uint32_t doc_delta_bytes;
    float *impacts;
    evoke_semantic_impact_precision impact_precision;
} evoke_semantic_bmp_packed_index;

void evoke_semantic_bmp_index_init(evoke_semantic_bmp_index *index);

void evoke_semantic_bmp_index_free(evoke_semantic_bmp_index *index);

evoke_status evoke_semantic_bmp_index_build(
    uint32_t document_count,
    const evoke_semantic_bmp_posting *postings,
    size_t posting_count,
    evoke_semantic_bmp_index *index_out
);

evoke_status evoke_semantic_bmp_index_build_runs(
    uint32_t document_count,
    const evoke_semantic_bmp_run *runs,
    size_t run_count,
    evoke_semantic_bmp_index *index_out
);

evoke_status evoke_semantic_bmp_index_validate(
    const evoke_semantic_bmp_index *index
);

evoke_status evoke_semantic_bmp_serialized_size(
    const evoke_semantic_bmp_index *index,
    size_t *size_out
);

evoke_status evoke_semantic_bmp_serialize(
    const evoke_semantic_bmp_index *index,
    uint8_t **bytes_out,
    size_t *size_out
);

evoke_status evoke_semantic_bmp_serialize_into(
    const evoke_semantic_bmp_index *index,
    uint8_t *bytes,
    size_t size
);

evoke_status evoke_semantic_bmp_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_index *index_out
);

evoke_status evoke_semantic_bmp_disk_header_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_disk_header *header_out
);

evoke_status evoke_semantic_bmp_term_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_term *term_out
);

evoke_status evoke_semantic_bmp_ref_decode(
    const uint8_t *bytes,
    size_t size,
    float term_min_impact,
    float term_max_impact,
    evoke_semantic_bmp_ref *ref_out
);

evoke_status evoke_semantic_bmp_super_ref_decode(
    const uint8_t *bytes,
    size_t size,
    float term_min_impact,
    float term_max_impact,
    evoke_semantic_bmp_super_ref *ref_out
);

evoke_status evoke_semantic_bmp_block_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_block *block_out
);

evoke_status evoke_semantic_bmp_record_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_record *record_out
);

evoke_status evoke_semantic_bmp_impacts_decode(
    const uint8_t *bytes,
    size_t size,
    uint16_t impact_count,
    float *impacts_out
);

evoke_status evoke_semantic_bmp_packed_impact_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_impact_precision impact_precision,
    float term_min_impact,
    float term_max_impact,
    float *impact_out
);

evoke_status evoke_semantic_bmp_packed_impacts_decode(
    const uint8_t *bytes,
    size_t size,
    uint16_t impact_count,
    evoke_semantic_impact_precision impact_precision,
    float term_min_impact,
    float term_max_impact,
    float *impacts_out
);

float evoke_semantic_bmp_contribution_bound(
    float weight,
    float min_impact,
    float max_impact
);

evoke_status evoke_semantic_bmp_topk(
    const evoke_semantic_bmp_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
);

void evoke_semantic_bmp_packed_index_init(
    evoke_semantic_bmp_packed_index *index
);

void evoke_semantic_bmp_packed_index_free(
    evoke_semantic_bmp_packed_index *index
);

evoke_status evoke_semantic_bmp_packed_index_build(
    const evoke_semantic_bmp_index *source,
    evoke_semantic_bmp_packed_index *index_out
);

/*
 * Build the compact authority directly from term-local posting runs. This
 * avoids constructing the expanded block-forward representation first.
 */
evoke_status evoke_semantic_bmp_packed_index_build_runs(
    uint32_t document_count,
    const evoke_semantic_bmp_run *runs,
    size_t run_count,
    evoke_semantic_bmp_packed_index *index_out
);

evoke_status evoke_semantic_bmp_packed_index_build_runs_with_precision(
    uint32_t document_count,
    const evoke_semantic_bmp_run *runs,
    size_t run_count,
    evoke_semantic_impact_precision impact_precision,
    evoke_semantic_bmp_packed_index *index_out
);

size_t evoke_semantic_bmp_impact_width(
    evoke_semantic_impact_precision impact_precision
);

evoke_status evoke_semantic_impact_encode(
    uint8_t *bytes,
    size_t size,
    evoke_semantic_impact_precision impact_precision,
    float impact,
    float *quantized_out
);

evoke_status evoke_semantic_impact_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_impact_precision impact_precision,
    float *impact_out
);

evoke_status evoke_semantic_bmp_packed_index_validate(
    const evoke_semantic_bmp_packed_index *index
);

evoke_status evoke_semantic_bmp_packed_serialized_size(
    const evoke_semantic_bmp_packed_index *index,
    size_t *size_out
);

evoke_status evoke_semantic_bmp_packed_serialize(
    const evoke_semantic_bmp_packed_index *index,
    uint8_t **bytes_out,
    size_t *size_out
);

evoke_status evoke_semantic_bmp_packed_serialize_into(
    const evoke_semantic_bmp_packed_index *index,
    uint8_t *bytes,
    size_t size
);

evoke_status evoke_semantic_bmp_packed_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_packed_index *index_out
);

evoke_status evoke_semantic_bmp_packed_disk_header_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_packed_disk_header *header_out
);

evoke_status evoke_semantic_bmp_packed_term_decode(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_bmp_packed_term *term_out
);

evoke_status evoke_semantic_bmp_packed_super_ref_decode(
    const uint8_t *bytes,
    size_t size,
    float term_min_impact,
    float term_max_impact,
    evoke_semantic_bmp_packed_super_ref *ref_out
);

evoke_status evoke_semantic_bmp_packed_ref_decode(
    const uint8_t *bytes,
    size_t size,
    float term_min_impact,
    float term_max_impact,
    evoke_semantic_bmp_packed_ref *ref_out
);

evoke_status evoke_semantic_bmp_packed_term_materialize(
    const evoke_semantic_bmp_packed_index *index,
    uint32_t term_index,
    uint32_t *document_ids_out,
    evoke_posting_value *values_out,
    size_t capacity
);

evoke_status evoke_semantic_bmp_packed_topk(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
);

/*
 * Run the same exact packed traversal after offering exact, unique seed
 * scores. Seed document IDs may be unordered; the executor suppresses their
 * later block offers so each document appears at most once in the result.
 */
evoke_status evoke_semantic_bmp_packed_topk_seeded(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    const evoke_semantic_bmp_seed *seeds,
    size_t seed_count,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
);

/* Experimental oracle for measuring bounded top-k boundary trade-offs. */
evoke_status evoke_semantic_bmp_packed_topk_seeded_bounded(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    const evoke_semantic_bmp_seed *seeds,
    size_t seed_count,
    float max_boundary_error,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
);

/* Experimental exact traversal with block-level norm-envelope bounds. */
evoke_status evoke_semantic_bmp_packed_topk_norm(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    const evoke_semantic_bmp_norm_bounds *norm_bounds,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
);

/* Exact term-at-a-time fallback over the same packed authority. */
evoke_status evoke_semantic_bmp_packed_taat_topk(
    const evoke_semantic_bmp_packed_index *index,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_count,
    size_t k,
    evoke_topk_result *result_out,
    evoke_semantic_bmp_stats *stats_out
);

#endif
