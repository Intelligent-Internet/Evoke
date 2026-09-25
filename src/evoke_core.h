#ifndef EVOKE_CORE_H
#define EVOKE_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum evoke_status
{
    EVOKE_OK = 0,
    EVOKE_ERR_NOMEM = 1,
    EVOKE_ERR_INVALID = 2,
    EVOKE_ERR_RANGE = 3,
    EVOKE_ERR_FORMAT = 4
} evoke_status;

uint32_t evoke_u32_saturating_add(uint32_t left, uint32_t right);
uint64_t evoke_u64_saturating_add(uint64_t left, uint64_t right);
uint64_t evoke_u64_saturating_mul(uint64_t left, uint64_t right);

typedef enum evoke_method
{
    EVOKE_METHOD_ROBERTSON = 0,
    EVOKE_METHOD_LUCENE = 1,
    EVOKE_METHOD_ATIRE = 2,
    EVOKE_METHOD_BM25L = 3,
    EVOKE_METHOD_BM25PLUS = 4
} evoke_method;

typedef struct evoke_doc_ids
{
    uint32_t *token_ids;
    size_t len;
} evoke_doc_ids;

typedef struct evoke_doc_tokens
{
    const char **tokens;
    size_t len;
} evoke_doc_tokens;

typedef struct evoke_term_entry
{
    uint32_t token_id;
    uint32_t doc_id;
    uint32_t tf;
} evoke_term_entry;

typedef evoke_status (*evoke_term_entry_reader_cb)(
    void *ctx,
    evoke_term_entry *entry_out
);

typedef evoke_status (*evoke_term_entry_rewind_cb)(void *ctx);

typedef struct evoke_params
{
    float k1;
    float b;
    float delta;
    evoke_method method;
    evoke_method idf_method;
} evoke_params;

typedef struct evoke_index
{
    evoke_params params;
    uint32_t num_docs;
    uint32_t vocab_size;
    uint64_t data_len;
    bool has_empty_token;
    uint32_t empty_token_id;
    float *data;
    uint32_t *indices;
    uint64_t *indptr;
    uint32_t *term_frequencies;
    uint32_t *doc_lengths;
    uint32_t *doc_frequencies;
    float *nonoccurrence;
    char **vocab;
} evoke_index;

typedef struct evoke_topk_result
{
    uint32_t *doc_ids;
    float *scores;
    size_t len;
} evoke_topk_result;

typedef struct evoke_topk_item
{
    float score;
    uint32_t doc_id;
    uint64_t tie_break_key;
} evoke_topk_item;

/* One-shot streaming top-k state whose allocation is proportional to k. */
typedef struct evoke_topk_accumulator
{
    evoke_topk_item *heap;
    size_t len;
    size_t capacity;
    bool finalized;
} evoke_topk_accumulator;

typedef enum evoke_posting_extent_kind
{
    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL = 1,
    EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT = 2,
    EVOKE_POSTING_EXTENT_LEXICAL_IMPACT = 3
} evoke_posting_extent_kind;

typedef union evoke_posting_value
{
    float impact;
    uint32_t term_frequency;
} evoke_posting_value;

#define EVOKE_DEFAULT_POSTING_BLOCK_SHIFT 7U

/*
 * Persistent, query-independent metadata for one posting slice inside a
 * stable global document-slot block. Document-length extrema are deliberately
 * not duplicated per term: the query runtime derives them once from the
 * global document-block directory.
 */
typedef struct evoke_posting_block_record
{
    uint64_t posting_offset;
    uint32_t posting_count;
    uint32_t block_id;
    uint32_t first_document_id;
    uint32_t last_document_id;
    uint32_t min_term_frequency;
    uint32_t max_term_frequency;
    float min_impact;
    float max_impact;
    evoke_posting_extent_kind kind;
} evoke_posting_block_record;

typedef struct evoke_posting_extent
{
    const float *data;
    const uint32_t *indices;
    const uint32_t *term_frequencies;
    const evoke_posting_value *values;
    const uint32_t *document_id_map;
    const evoke_posting_block_record *blocks;
    uint64_t len;
    uint32_t document_id_base;
    uint32_t local_document_count;
    uint32_t block_count;
    uint32_t block_shift;
    evoke_posting_extent_kind kind;
} evoke_posting_extent;

typedef struct evoke_term_extent_list
{
    const evoke_posting_extent *extents;
    size_t len;
} evoke_term_extent_list;

/*
 * One query-independent upper-bound unit aligned to the stable global
 * document-slot space. Every extent and fold uses the same block identity, so
 * bounds from folded prefixes and exact tails can later enter one global
 * pruning schedule.
 */
typedef struct evoke_posting_block_bound
{
    uint64_t posting_offset;
    uint32_t posting_count;
    uint32_t block_id;
    uint32_t first_document_id;
    uint32_t last_document_id;
    uint32_t min_term_frequency;
    uint32_t max_term_frequency;
    uint32_t min_document_length;
    uint32_t max_document_length;
    float min_impact;
    float max_impact;
    evoke_posting_extent_kind kind;
} evoke_posting_block_bound;

typedef struct evoke_document_block_extrema
{
    uint64_t document_count;
    uint32_t min_document_length;
    uint32_t max_document_length;
} evoke_document_block_extrema;

typedef struct evoke_blockmax_stats
{
    uint64_t blocks_considered;
    uint64_t blocks_scored;
    uint64_t blocks_skipped;
    uint64_t postings_scored;
    uint64_t zero_score_documents_considered;
} evoke_blockmax_stats;

typedef struct evoke_corpus_stats
{
    uint64_t document_count;
    uint64_t total_document_length;
    const uint32_t *doc_frequencies;
    size_t vocab_size;
} evoke_corpus_stats;

/*
 * Validate an extent once when it is attached to a trusted read view. Scoring
 * assumes this validation has succeeded and does not repeat per-posting range
 * checks in the hot loop.
 */
evoke_status evoke_posting_extent_validate_layout(
    const evoke_index *index,
    const evoke_posting_extent *extent
);

evoke_status evoke_posting_extent_build_block_records(
    const evoke_posting_extent *extent,
    uint32_t block_shift,
    evoke_posting_block_record **records_out,
    size_t *record_count_out
);

void evoke_posting_block_records_free(
    evoke_posting_block_record *records
);

evoke_status evoke_posting_extent_validate_block_records(
    const evoke_posting_extent *extent,
    uint32_t block_shift,
    const evoke_posting_block_record *records,
    size_t record_count
);

evoke_status evoke_posting_extent_build_block_bounds(
    const evoke_index *index,
    const evoke_posting_extent *extent,
    uint32_t block_shift,
    evoke_posting_block_bound **bounds_out,
    size_t *bound_count_out
);

void evoke_posting_block_bounds_free(evoke_posting_block_bound *bounds);

evoke_status evoke_posting_block_score_upper_bound(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    uint32_t live_document_frequency,
    const evoke_posting_block_bound *bound,
    float query_weight,
    float *score_upper_bound_out
);

const char *evoke_strerror(evoke_status status);
bool evoke_params_are_valid(const evoke_params *params);
bool evoke_parse_method(
    const char *name,
    evoke_method *method_out
);
const char *evoke_method_name(evoke_method method);
bool evoke_method_requires_nonoccurrence(evoke_method method);
double evoke_score_tfc(
    evoke_method method,
    double tf,
    double doc_length,
    double average_doc_length,
    double k1,
    double b,
    double delta
);
double evoke_score_idf(
    evoke_method method,
    double document_frequency,
    double document_count
);
const char *evoke_active_simd_path(void);
void evoke_index_init(evoke_index *index);
void evoke_index_free(evoke_index *index);
void evoke_topk_result_free(evoke_topk_result *result);

evoke_status evoke_build_index_from_ids(
    const evoke_doc_ids *docs,
    size_t num_docs,
    const evoke_params *params,
    bool create_empty_token,
    evoke_index *index_out
);

evoke_status evoke_build_index_from_ids_compact(
    const evoke_doc_ids *docs,
    size_t num_docs,
    const evoke_params *params,
    bool create_empty_token,
    evoke_index *index_out
);

evoke_status evoke_build_index_from_term_entries(
    const evoke_term_entry *entries,
    uint64_t num_entries,
    const uint32_t *doc_lengths,
    uint32_t num_docs,
    uint32_t vocab_size,
    const evoke_params *params,
    bool create_empty_token,
    bool has_empty_token,
    uint32_t empty_token_id,
    const char *const *vocab,
    evoke_index *index_out
);

evoke_status evoke_build_index_from_term_entry_reader(
    uint64_t num_entries,
    evoke_term_entry_reader_cb read_cb,
    evoke_term_entry_rewind_cb rewind_cb,
    void *reader_ctx,
    const uint32_t *doc_lengths,
    uint32_t num_docs,
    uint32_t vocab_size,
    const evoke_params *params,
    bool create_empty_token,
    bool has_empty_token,
    uint32_t empty_token_id,
    const char *const *vocab,
    evoke_index *index_out
);

evoke_status evoke_build_index_from_tokens(
    const evoke_doc_tokens *docs,
    size_t num_docs,
    const evoke_params *params,
    evoke_index *index_out
);

evoke_status evoke_build_index_from_tokens_compact(
    const evoke_doc_tokens *docs,
    size_t num_docs,
    const evoke_params *params,
    evoke_index *index_out
);

evoke_status evoke_query_token_ids(
    const evoke_index *index,
    const char **tokens,
    size_t num_tokens,
    uint32_t **query_ids_out,
    size_t *query_len_out
);

evoke_status evoke_scores_from_ids(
    const evoke_index *index,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
);

evoke_status evoke_scores_from_ids_extents(
    const evoke_index *index,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
);

evoke_status evoke_scores_from_ids_exact_stats(
    const evoke_index *index,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
);

evoke_status evoke_scores_from_ids_neutral(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
);

evoke_status evoke_scores_from_ids_mixed(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
);

evoke_status evoke_scores_from_ids_mixed_retired(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint32_t *query_ids,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
);

/*
 * Resolve the exact live lexical document frequency for one mixed read-view
 * term. This is the shared authority for query scoring and epoch-bound impact
 * compilation; semantic extents do not contribute to lexical DF.
 */
evoke_status evoke_term_live_document_frequency(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    uint32_t term_id,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    uint32_t *live_document_frequency_out
);

evoke_status evoke_scores_from_weighted_ids_mixed_retired(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    const float *weight_mask,
    float **scores_out
);

evoke_status evoke_topk_from_weighted_ids_mixed_retired_blockmax(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const evoke_document_block_extrema *document_blocks,
    size_t document_block_count,
    uint32_t block_shift,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *blockmax_stats_out
);

evoke_status
evoke_topk_from_weighted_ids_mixed_retired_blockmax_with_tie_breaks(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const evoke_document_block_extrema *document_blocks,
    size_t document_block_count,
    uint32_t block_shift,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint64_t *tie_break_keys,
    const uint32_t *tie_break_order,
    size_t tie_break_order_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *blockmax_stats_out
);

evoke_status
evoke_topk_from_weighted_ids_mixed_retired_blockmax_filtered_with_tie_breaks(
    const evoke_index *index,
    const evoke_corpus_stats *stats,
    const evoke_term_extent_list *term_extents,
    size_t num_term_extent_lists,
    const evoke_document_block_extrema *document_blocks,
    size_t document_block_count,
    uint32_t block_shift,
    const uint32_t *retired_document_ids,
    size_t retired_document_count,
    const uint64_t *tie_break_keys,
    const uint32_t *tie_break_order,
    size_t tie_break_order_count,
    const uint8_t *allowed_document_bitmap,
    size_t allowed_document_count,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *blockmax_stats_out
);

evoke_status evoke_topk(
    const float *scores,
    size_t num_scores,
    size_t k,
    bool sorted,
    evoke_topk_result *result_out
);

evoke_status evoke_topk_with_tie_breaks(
    const float *scores,
    size_t num_scores,
    size_t k,
    bool sorted,
    const uint64_t *tie_break_keys,
    evoke_topk_result *result_out
);

evoke_status evoke_topk_subset(
    const float *scores,
    const uint32_t *candidate_doc_ids,
    size_t num_candidate_doc_ids,
    size_t k,
    bool sorted,
    bool positive_only,
    evoke_topk_result *result_out
);

evoke_status evoke_topk_subset_with_tie_breaks(
    const float *scores,
    const uint32_t *candidate_doc_ids,
    size_t num_candidate_doc_ids,
    size_t k,
    bool sorted,
    bool positive_only,
    const uint64_t *tie_break_keys,
    evoke_topk_result *result_out
);

evoke_status evoke_topk_accumulator_init(
    evoke_topk_accumulator *accumulator,
    size_t capacity
);

evoke_status evoke_topk_accumulator_offer(
    evoke_topk_accumulator *accumulator,
    float score,
    uint32_t doc_id,
    uint64_t tie_break_key
);

evoke_status evoke_topk_accumulator_finish(
    evoke_topk_accumulator *accumulator,
    bool sorted,
    evoke_topk_result *result_out
);

void evoke_topk_accumulator_free(
    evoke_topk_accumulator *accumulator
);

#endif
