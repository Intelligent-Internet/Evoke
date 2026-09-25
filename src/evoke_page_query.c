#include "postgres.h"

#include "miscadmin.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "utils/elog.h"
#include "utils/memutils.h"

#include "evoke_page_query.h"
#include "evoke_weighted_space_saving.h"

#define EVOKE_PAGE_QUERY_MATERIALIZATION_LIMIT \
    (UINT64_C(8) * UINT64_C(1024) * UINT64_C(1024))
#define EVOKE_PAGE_QUERY_ORDERED_BLOCK_LIMIT \
    (UINT64_C(32) * UINT64_C(1024) * UINT64_C(1024))
#define EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT \
    (UINT64_C(64) * UINT64_C(1024) * UINT64_C(1024))
#define EVOKE_PAGE_QUERY_BMP_REF_WINDOW UINT32_C(4096)
#define EVOKE_PAGE_QUERY_BMP_SUPER_REF_WINDOW UINT32_C(4096)
#define EVOKE_PAGE_QUERY_BMP_MAX_UNPRUNED_BATCHES UINT32_C(1)
#define EVOKE_PAGE_QUERY_BMP_MAX_SUPER_REF_DENSITY UINT64_C(8)
#define EVOKE_PAGE_QUERY_INTERRUPT_MASK UINT64_C(0x1f)
#define EVOKE_PAGE_QUERY_ACCELERATOR_STALE_OVERFETCH_FLOOR UINT64_C(64)
#define EVOKE_PAGE_QUERY_ACCELERATOR_STALE_OVERFETCH_MAX UINT64_C(4096)

bool evoke_test_disable_semantic_bmp = false;
bool evoke_test_force_semantic_bmp = false;
bool evoke_test_disable_fused_semantic_taat = false;
int evoke_test_filtered_forward_route = EVOKE_FILTERED_FORWARD_ROUTE_AUTO;
double evoke_test_query_max_df_ratio = 1.0;
int evoke_test_query_semantic_work_target_postings = 0;
double evoke_test_query_semantic_error_budget_ratio = 0.0;
double evoke_test_query_semantic_impact_floor_ratio = 0.0;
double evoke_test_query_semantic_min_support_ratio = 0.0;
int evoke_test_query_semantic_bmp_super_batch =
    (int) EVOKE_PAGE_QUERY_BMP_DEFAULT_SUPER_BATCH;
bool evoke_semantic_accelerator_disabled = false;
bool evoke_test_semantic_accelerator_seed_bmp = false;
double evoke_semantic_accelerator_heap_factor = 0.7;
int evoke_semantic_accelerator_candidate_multiplier = 64;
bool evoke_semantic_accelerator_bound_residual_candidates = true;
bool evoke_semantic_accelerator_accumulate_residual_candidates = true;
bool evoke_test_semantic_accelerator_summarize_residual_candidates = false;
int evoke_test_semantic_accelerator_summary_multiplier = 4;

typedef struct evoke_page_query_l0_build_event
{
    evoke_l0_record_kind kind;
    uint64 sequence;
    uint64 document_slot;
    uint32 document_length;
    uint32 heap_block;
    uint16 heap_offset;
} evoke_page_query_l0_build_event;

typedef struct evoke_page_query_l0_build_contribution
{
    uint32 document_slot;
    evoke_page_query_l0_contribution contribution;
} evoke_page_query_l0_build_contribution;

typedef struct evoke_page_query_l0_build_document
{
    uint32 document_slot;
    uint32 document_length;
    uint64 born_sequence;
    uint64 initial_retirement_sequence;
    uint32 heap_block;
    uint16 heap_offset;
    uint64 immutable_last_sequence;
    uint64 last_applied_sequence;
    evoke_l0_record_kind last_applied_kind;
    bool initial_live;
    bool live;
    bool visible_event;
    bool shadows_immutable;
    bool l0_placeholder;
} evoke_page_query_l0_build_document;

typedef struct evoke_page_query_l0_build
{
    Relation index_relation;
    const evoke_segment_query_context *context;
    const evoke_page_query_l0_input *input;
    uint32 expected_record_count;
    uint32 record_index;
    uint32 *record_xids;
    uint8 *visibility;
    uint32 *upsert_slots;
    size_t upsert_count;
    evoke_page_query_l0_build_event *events;
    size_t event_count;
    evoke_page_query_l0_build_contribution *contributions;
    size_t contribution_count;
    size_t contribution_capacity;
    size_t *query_token_lengths;
    uint32 max_numeric_term_plus_one;
    uint32 max_semantic_term_plus_one;
    bool snapshot_sensitive;
} evoke_page_query_l0_build;

typedef struct evoke_page_query_term
{
    evoke_segment_query_term_plan plan;
    float query_weight;
    uint32 query_index;
    uint32 immutable_live_document_frequency;
    uint32 live_document_frequency;
    uint32 first_cursor;
    uint32 cursor_count;
    double idf;
    float nonoccurrence;
    bool lexical_neutral_seen;
    bool lexical_impact_seen;
} evoke_page_query_term;

typedef struct evoke_page_query_cursor
{
    uint32 term_index;
    uint32 run_index;
    uint32 block_index;
    uint32 posting_index;
    uint32 previous_document_slot;
    bool have_previous_document;
    bool exhausted;
    evoke_segment_query_block block;
} evoke_page_query_cursor;

typedef struct evoke_page_query_block_cursor
{
    evoke_posting_block_record record;
    evoke_posting_block_record record_window[
        EVOKE_SEGMENT_QUERY_BLOCK_RECORD_WINDOW
    ];
    uint32 term_index;
    uint32 run_index;
    uint32 block_index;
    uint32 window_first_block_index;
    uint32 window_count;
    uint32 previous_block_id;
    bool have_previous_block;
    bool exhausted;
} evoke_page_query_block_cursor;

typedef struct evoke_page_query_block_reference
{
    uint8 bytes[6];
} evoke_page_query_block_reference;

typedef struct evoke_page_query_ordered_block
{
    uint64 posting_count;
    uint32 first_contribution;
    uint32 contribution_count;
    uint32 block_id;
    float upper_bound;
} evoke_page_query_ordered_block;

typedef struct evoke_page_query_bmp_block
{
    uint32 block_id;
    float upper_bound;
} evoke_page_query_bmp_block;

typedef struct evoke_page_query_bmp_run_cache
{
    evoke_segment_query_bmp_cached_super_ref *super_refs;
    uint32 super_ref_count;
} evoke_page_query_bmp_run_cache;

typedef struct evoke_page_query_bmp_superblock_range
{
    uint32 first_superblock;
    uint32 superblock_limit;
} evoke_page_query_bmp_superblock_range;

typedef struct evoke_page_query_cleanup
{
    evoke_page_query_term *terms;
    evoke_page_query_cursor *cursors;
    evoke_page_query_block_cursor *block_cursors;
    uint32 *block_cursor_heap;
    uint32 *block_cursor_group;
    size_t block_cursor_heap_count;
    size_t block_cursor_group_count;
    uint32 *query_term_map;
    evoke_segment_query_term *materialized_terms;
    uint32 *materialized_block_ids;
    float *term_at_a_time_scores;
    const uint32 *term_at_a_time_document_lengths;
    uint32 *term_at_a_time_owned_document_lengths;
    uint32 *term_at_a_time_document_slots;
    evoke_posting_value *term_at_a_time_values;
    size_t materialized_term_count;
    size_t materialized_block_count;
    bool *projection_document_seen;
    evoke_segment_query_document_reader document_reader;
    evoke_segment_query_document_block document_block;
    evoke_topk_accumulator accumulator;
    Relation index_relation;
    const evoke_segment_query_context *context;
    const evoke_page_query_filter *filter;
    const evoke_page_query_l0_projection *projection;
    uint64 minimum_root_sequence;
} evoke_page_query_cleanup;

typedef struct evoke_page_query_ordered_cleanup
{
    MemoryContextCallback callback;
    bool active;
    evoke_page_query_block_reference *contributions;
    evoke_page_query_ordered_block *blocks;
    uint8 *block_scored;
} evoke_page_query_ordered_cleanup;

typedef struct evoke_page_query_bmp_cleanup
{
    MemoryContextCallback callback;
    bool active;
    bool pruning_initialized;
    float *scores;
    float *lexical_max;
    float *super_lexical_max;
    float *super_upper_bounds;
    float *batch_upper_bounds;
    float *flat_upper_bounds;
    int32 *selected_superblocks;
    uint8 *allowed_blocks;
    uint8 *allowed_block_bits;
    uint8 *allowed_superblocks;
    uint8 *block_membership_scratch;
    bool *exact_blocks;
    uint32 *document_slots;
    evoke_posting_value *values;
    uint32 *owned_document_lengths;
    uint32 *filtered_document_block_offsets;
    uint32 *filtered_document_lengths;
    uint8 *liveness;
    evoke_semantic_bmp_ref *refs;
    evoke_semantic_bmp_super_ref *super_refs;
    evoke_segment_query_bmp_cached_super_ref *cached_super_refs;
    evoke_page_query_bmp_run_cache *run_caches;
    size_t run_cache_count;
    evoke_page_query_bmp_superblock_range *allowed_superblock_ranges;
    evoke_page_query_bmp_block *ranked_superblocks;
    evoke_page_query_bmp_block *ranked_batch_blocks;
    evoke_page_query_bmp_block *ranked_flat_blocks;
    evoke_topk_accumulator pruning_accumulator;
} evoke_page_query_bmp_cleanup;

typedef struct evoke_page_query_zero_candidate
{
    uint64 born_sequence;
    uint32 document_slot;
} evoke_page_query_zero_candidate;

typedef struct evoke_page_query_document_length_build
{
    uint32 *document_lengths;
    uint64 document_slot_count;
    uint64 next_document_slot;
    bool valid;
} evoke_page_query_document_length_build;

static bool
evoke_page_query_document_is_live(
    const evoke_document_cow_record *record
)
{
    return record != NULL &&
        (record->version.flags &
         EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) == 0 &&
        record->retirement.retirement_sequence == 0;
}

static bool
evoke_page_query_document_allowed(
    const evoke_page_query_filter *filter,
    uint32 document_slot
)
{
    uint8 mask;

    if (filter == NULL)
    {
        return true;
    }
    if (filter->allows_document != NULL)
    {
        return filter->allows_document(
            filter->allows_document_context,
            document_slot
        );
    }
    if ((uint64) document_slot >= filter->document_slot_count)
    {
        return false;
    }
    mask = (uint8) (UINT8_C(1) << (document_slot & UINT32_C(7)));
    return (filter->allowed_document_bitmap[document_slot >> 3] & mask) != 0;
}

static bool
evoke_page_query_block_has_allowed_document(
    const evoke_page_query_filter *filter,
    uint32 block_id,
    uint32 block_shift
)
{
    uint64 first_document_slot;
    uint64 block_document_count;
    uint64 remaining;
    uint64 document_count;
    uint64 last_document_slot;
    uint64 first_byte;
    uint64 last_byte;

    if (filter == NULL || filter->allowed_document_bitmap == NULL)
    {
        return true;
    }
    if (block_shift >= UINT32_C(32))
    {
        return false;
    }
    first_document_slot = (uint64) block_id << block_shift;
    if (first_document_slot >= filter->document_slot_count)
    {
        return false;
    }
    block_document_count = UINT64_C(1) << block_shift;
    remaining = filter->document_slot_count - first_document_slot;
    document_count = remaining > block_document_count
        ? block_document_count
        : remaining;
    last_document_slot = first_document_slot + document_count - UINT64_C(1);
    first_byte = first_document_slot >> 3;
    last_byte = last_document_slot >> 3;
    if (first_byte == last_byte)
    {
        uint8 first_mask = (uint8) (
            UINT8_MAX << (first_document_slot & UINT64_C(7))
        );
        uint8 last_mask = (uint8) (
            UINT8_MAX >> (UINT64_C(7) -
                (last_document_slot & UINT64_C(7)))
        );

        return (filter->allowed_document_bitmap[first_byte] &
            first_mask & last_mask) != 0;
    }
    if ((filter->allowed_document_bitmap[first_byte] &
         (uint8) (UINT8_MAX <<
             (first_document_slot & UINT64_C(7)))) != 0)
    {
        return true;
    }
    for (uint64 byte_index = first_byte + UINT64_C(1);
         byte_index < last_byte;
         byte_index++)
    {
        if (filter->allowed_document_bitmap[byte_index] != 0)
        {
            return true;
        }
    }
    return (filter->allowed_document_bitmap[last_byte] &
        (uint8) (UINT8_MAX >> (UINT64_C(7) -
            (last_document_slot & UINT64_C(7))))) != 0;
}

static bool
evoke_page_query_root_document_allowed(
    const evoke_page_query_cleanup *cleanup,
    uint32 document_slot
)
{
    evoke_document_cow_record record;
    const evoke_page_query_l0_document *projected_document =
        cleanup == NULL
            ? NULL
            : evoke_page_query_l0_find_document(
                cleanup->projection,
                document_slot
            );

    if (!evoke_page_query_document_allowed(
            cleanup == NULL ? NULL : cleanup->filter,
            document_slot) ||
        (projected_document != NULL &&
         projected_document->shadows_immutable))
    {
        return false;
    }
    if (cleanup == NULL || cleanup->minimum_root_sequence == 0)
    {
        return true;
    }
    evoke_segment_pages_load_document_record(
        cleanup->index_relation,
        &cleanup->context->root,
        &cleanup->context->manifest,
        document_slot,
        &record
    );
    return evoke_page_query_document_is_live(&record) &&
        record.version.born_sequence > cleanup->minimum_root_sequence;
}

static evoke_status
evoke_page_query_offer_root(
    const evoke_page_query_cleanup *cleanup,
    evoke_topk_accumulator *accumulator,
    float score,
    uint32 document_slot,
    uint64 tie_break_key
)
{
    if (!evoke_page_query_root_document_allowed(cleanup, document_slot))
    {
        return EVOKE_OK;
    }
    return evoke_topk_accumulator_offer(
        accumulator,
        score,
        document_slot,
        tie_break_key
    );
}

static evoke_status
evoke_page_query_offer_projection(
    const evoke_page_query_cleanup *cleanup,
    evoke_topk_accumulator *accumulator,
    float score,
    uint32 document_slot,
    uint64 tie_break_key
)
{
    if (!evoke_page_query_document_allowed(
            cleanup == NULL ? NULL : cleanup->filter,
            document_slot))
    {
        return EVOKE_OK;
    }
    return evoke_topk_accumulator_offer(
        accumulator,
        score,
        document_slot,
        tie_break_key
    );
}

static void
evoke_page_query_collect_document_length(
    void *context,
    const evoke_document_cow_record *document
)
{
    evoke_page_query_document_length_build *build = context;

    if (build == NULL || document == NULL || !build->valid)
    {
        return;
    }
    if (build->next_document_slot >= build->document_slot_count ||
        document->version.document_slot != build->next_document_slot)
    {
        build->valid = false;
        return;
    }
    build->document_lengths[build->next_document_slot] =
        document->version.document_length;
    build->next_document_slot++;
}

void
evoke_page_query_l0_projection_init(
    evoke_page_query_l0_projection *projection
)
{
    if (projection != NULL)
    {
        memset(projection, 0, sizeof(*projection));
    }
}

void
evoke_page_query_l0_projection_free(
    evoke_page_query_l0_projection *projection
)
{
    if (projection == NULL)
    {
        return;
    }
    if (projection->contributions != NULL)
    {
        pfree(projection->contributions);
    }
    if (projection->documents != NULL)
    {
        pfree(projection->documents);
    }
    if (projection->terms != NULL)
    {
        pfree(projection->terms);
    }
    memset(projection, 0, sizeof(*projection));
}

static int
evoke_page_query_cmp_u32(const void *left, const void *right)
{
    uint32 left_value = *((const uint32 *) left);
    uint32 right_value = *((const uint32 *) right);

    if (left_value < right_value)
    {
        return -1;
    }
    return left_value > right_value;
}

static int
evoke_page_query_cmp_l0_contribution(
    const void *left,
    const void *right
)
{
    const evoke_page_query_l0_build_contribution *left_value = left;
    const evoke_page_query_l0_build_contribution *right_value = right;

    if (left_value->document_slot != right_value->document_slot)
    {
        return left_value->document_slot < right_value->document_slot
            ? -1
            : 1;
    }
    if (left_value->contribution.query_index !=
        right_value->contribution.query_index)
    {
        return left_value->contribution.query_index <
            right_value->contribution.query_index
            ? -1
            : 1;
    }
    if (left_value->contribution.kind != right_value->contribution.kind)
    {
        return left_value->contribution.kind < right_value->contribution.kind
            ? -1
            : 1;
    }
    return 0;
}

static evoke_page_query_l0_build_document *
evoke_page_query_find_build_document(
    evoke_page_query_l0_build_document *documents,
    size_t document_count,
    uint32 document_slot
)
{
    size_t low = 0;
    size_t high = document_count;

    while (low < high)
    {
        size_t middle = low + (high - low) / 2;

        if (documents[middle].document_slot < document_slot)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= document_count ||
        documents[low].document_slot != document_slot)
    {
        return NULL;
    }
    return &documents[low];
}

const evoke_page_query_l0_document *
evoke_page_query_l0_find_document(
    const evoke_page_query_l0_projection *projection,
    uint32 document_slot
)
{
    size_t low = 0;
    size_t high;

    if (projection == NULL || projection->documents == NULL)
    {
        return NULL;
    }
    high = projection->document_count;
    while (low < high)
    {
        size_t middle = low + (high - low) / 2;

        if (projection->documents[middle].document_slot < document_slot)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= projection->document_count ||
        projection->documents[low].document_slot != document_slot)
    {
        return NULL;
    }
    return &projection->documents[low];
}

static void
evoke_page_query_l0_reserve_contributions(
    evoke_page_query_l0_build *build,
    size_t additional
)
{
    size_t required;
    size_t capacity;
    Size bytes;

    if (additional > SIZE_MAX - build->contribution_count)
    {
        ereport(ERROR, (errmsg("evoke query L0 projection is too large")));
    }
    required = build->contribution_count + additional;
    if (required <= build->contribution_capacity)
    {
        return;
    }
    capacity = build->contribution_capacity > 0
        ? build->contribution_capacity
        : 16;
    while (capacity < required)
    {
        if (capacity > SIZE_MAX / 2)
        {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    if (capacity > MaxAllocSize / sizeof(*build->contributions))
    {
        ereport(ERROR, (errmsg("evoke query L0 projection is too large")));
    }
    bytes = (Size) capacity * sizeof(*build->contributions);
    build->contributions = build->contributions == NULL
        ? palloc(bytes)
        : repalloc(build->contributions, bytes);
    build->contribution_capacity = capacity;
}

static void
evoke_page_query_l0_append_contribution(
    evoke_page_query_l0_build *build,
    uint32 document_slot,
    uint32 query_index,
    evoke_posting_extent_kind kind,
    evoke_posting_value value
)
{
    evoke_page_query_l0_build_contribution *contribution;

    evoke_page_query_l0_reserve_contributions(build, 1);
    contribution = &build->contributions[build->contribution_count++];
    memset(contribution, 0, sizeof(*contribution));
    contribution->document_slot = document_slot;
    contribution->contribution.query_index = query_index;
    contribution->contribution.kind = kind;
    contribution->contribution.value = value;
}

static void
evoke_page_query_l0_validate_record(
    const evoke_page_query_l0_build *build,
    const evoke_l0_record_view *record
)
{
    bool upsert_encoding_valid;

    if (build == NULL || record == NULL ||
        record->document_slot >= build->context->root.next_document_slot ||
        record->document_slot > UINT32_MAX)
    {
        ereport(ERROR, (errmsg("invalid evoke query L0 record")));
    }
    upsert_encoding_valid = build->input->numeric_source
        ? record->term_encoding == EVOKE_L0_TERM_ENCODING_NUMERIC
        : record->term_encoding == EVOKE_L0_TERM_ENCODING_UTF8;
    if ((record->kind == EVOKE_L0_RECORD_UPSERT &&
         !upsert_encoding_valid) ||
        (record->kind == EVOKE_L0_RECORD_RETIRE &&
         record->term_encoding != EVOKE_L0_TERM_ENCODING_NONE) ||
        (record->kind == EVOKE_L0_RECORD_SEMANTIC_COMPLETE &&
         record->term_encoding != EVOKE_L0_TERM_ENCODING_NUMERIC) ||
        (record->kind == EVOKE_L0_RECORD_SEMANTIC_QUARANTINE &&
         record->term_encoding != EVOKE_L0_TERM_ENCODING_NONE) ||
        (record->kind != EVOKE_L0_RECORD_UPSERT &&
         record->kind != EVOKE_L0_RECORD_RETIRE &&
         record->kind != EVOKE_L0_RECORD_SEMANTIC_COMPLETE &&
         record->kind != EVOKE_L0_RECORD_SEMANTIC_QUARANTINE))
    {
        ereport(ERROR, (errmsg("invalid evoke query L0 record kind")));
    }
}

static void
evoke_page_query_l0_collect_xid(
    void *context,
    const evoke_l0_record_view *record
)
{
    evoke_page_query_l0_build *build = context;

    if (build == NULL || build->record_index >= build->expected_record_count)
    {
        ereport(ERROR, (errmsg("invalid evoke query L0 record count")));
    }
    evoke_page_query_l0_validate_record(build, record);
    build->record_xids[build->record_index++] = record->record_xid;
    if (record->kind == EVOKE_L0_RECORD_UPSERT)
    {
        build->upsert_slots[build->upsert_count++] =
            (uint32) record->document_slot;
    }
}

static bool
evoke_page_query_l0_atom_matches(
    const evoke_page_query_l0_build *build,
    const evoke_l0_lexical_atom *atom,
    size_t query_index
)
{
    if (build->input->numeric_source)
    {
        return atom->term_id == build->input->query_ids[query_index];
    }
    if (build->input->query_prefixes != NULL &&
        build->input->query_prefixes[query_index])
    {
        return atom->term_bytes_len >=
                build->query_token_lengths[query_index] &&
            memcmp(
                atom->term_bytes,
                build->input->query_tokens[query_index],
                build->query_token_lengths[query_index]
            ) == 0;
    }
    return atom->term_bytes_len == build->query_token_lengths[query_index] &&
        memcmp(
            atom->term_bytes,
            build->input->query_tokens[query_index],
            atom->term_bytes_len
        ) == 0;
}

static void
evoke_page_query_l0_collect_visible(
    void *context,
    const evoke_l0_record_view *record
)
{
    evoke_page_query_l0_build *build = context;
    evoke_page_query_l0_build_event *event;
    uint8 visibility;

    if (build == NULL || build->record_index >= build->expected_record_count ||
        build->record_xids[build->record_index] != record->record_xid)
    {
        ereport(ERROR, (errmsg("evoke query L0 stream changed during read")));
    }
    evoke_page_query_l0_validate_record(build, record);
    visibility = build->visibility[build->record_index++];
    if ((visibility &
         ~(EVOKE_PAGE_QUERY_L0_VISIBILITY_INCLUDE |
           EVOKE_PAGE_QUERY_L0_VISIBILITY_SNAPSHOT_SENSITIVE)) != 0)
    {
        ereport(ERROR, (errmsg("invalid evoke query L0 visibility state")));
    }
    if ((visibility &
         EVOKE_PAGE_QUERY_L0_VISIBILITY_SNAPSHOT_SENSITIVE) != 0)
    {
        build->snapshot_sensitive = true;
    }
    if ((visibility & EVOKE_PAGE_QUERY_L0_VISIBILITY_INCLUDE) == 0)
    {
        return;
    }
    event = &build->events[build->event_count++];
    memset(event, 0, sizeof(*event));
    event->kind = record->kind;
    event->sequence = record->sequence;
    event->document_slot = record->document_slot;
    event->document_length = record->document_length;
    event->heap_block = record->heap_block;
    event->heap_offset = record->heap_offset;

    if (record->kind == EVOKE_L0_RECORD_UPSERT)
    {
        evoke_l0_lexical_atom *atoms = NULL;
        evoke_status status;

        if (record->atom_count > MaxAllocSize / sizeof(*atoms))
        {
            ereport(ERROR, (errmsg("evoke query L0 atom record is too large")));
        }
        if (record->atom_count > 0)
        {
            atoms = palloc(sizeof(*atoms) * (Size) record->atom_count);
        }
        status = evoke_l0_record_view_decode_atoms(
            record,
            atoms,
            record->atom_count
        );
        if (status != EVOKE_OK)
        {
            ereport(
                ERROR,
                (
                    errmsg("failed to decode evoke query L0 atoms"),
                    errdetail("Validation failed: %s.", evoke_strerror(status))
                )
            );
        }
        for (uint32 atom_index = 0;
             atom_index < record->atom_count;
             atom_index++)
        {
            evoke_posting_value value;

            if (build->input->numeric_source)
            {
                if (atoms[atom_index].term_id == UINT32_MAX)
                {
                    ereport(ERROR, (errmsg("evoke query L0 term is too large")));
                }
                build->max_numeric_term_plus_one = Max(
                    build->max_numeric_term_plus_one,
                    atoms[atom_index].term_id + 1
                );
            }
            value.term_frequency = atoms[atom_index].term_frequency;
            for (size_t query_index = 0;
                 query_index < build->input->query_len;
                 query_index++)
            {
                if (evoke_page_query_l0_atom_matches(
                        build,
                        &atoms[atom_index],
                        query_index))
                {
                    evoke_page_query_l0_append_contribution(
                        build,
                        (uint32) record->document_slot,
                        (uint32) query_index,
                        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL,
                        value
                    );
                }
            }
        }
        if (atoms != NULL)
        {
            pfree(atoms);
        }
    }
    else if (record->kind == EVOKE_L0_RECORD_SEMANTIC_COMPLETE)
    {
        evoke_l0_semantic_atom *atoms = NULL;
        evoke_status status;

        if (record->atom_count > MaxAllocSize / sizeof(*atoms))
        {
            ereport(
                ERROR,
                (errmsg("evoke query semantic L0 record is too large"))
            );
        }
        if (record->atom_count > 0)
        {
            atoms = palloc(sizeof(*atoms) * (Size) record->atom_count);
        }
        status = evoke_l0_record_view_decode_semantic_atoms(
            record,
            atoms,
            record->atom_count
        );
        if (status != EVOKE_OK)
        {
            ereport(
                ERROR,
                (
                    errmsg("failed to decode evoke query semantic L0 atoms"),
                    errdetail("Validation failed: %s.", evoke_strerror(status))
                )
            );
        }
        for (uint32 atom_index = 0;
             atom_index < record->atom_count;
             atom_index++)
        {
            evoke_posting_value value;

            if (atoms[atom_index].term_id == UINT32_MAX)
            {
                ereport(ERROR, (errmsg("evoke semantic L0 term is too large")));
            }
            build->max_semantic_term_plus_one = Max(
                build->max_semantic_term_plus_one,
                atoms[atom_index].term_id + 1
            );
            value.impact = atoms[atom_index].impact;
            for (size_t query_index = 0;
                 query_index < build->input->query_len;
                 query_index++)
            {
                if (build->input->query_ids != NULL &&
                    atoms[atom_index].term_id ==
                        build->input->query_ids[query_index])
                {
                    evoke_page_query_l0_append_contribution(
                        build,
                        (uint32) record->document_slot,
                        (uint32) query_index,
                        EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT,
                        value
                    );
                }
            }
        }
        if (atoms != NULL)
        {
            pfree(atoms);
        }
    }
}

void
evoke_page_query_build_l0_projection(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_input *input,
    evoke_page_query_l0_visibility_resolver visibility_resolver,
    void *visibility_context,
    evoke_page_query_l0_projection *projection_out
)
{
    evoke_page_query_l0_projection projection;
    evoke_page_query_l0_build build;
    evoke_page_query_l0_build_document *documents = NULL;
    uint32 *document_slots = NULL;
    size_t document_slot_count = 0;
    size_t document_count = 0;
    uint64 visible_document_count;
    uint64 total_document_length;
    uint64 old_document_count;
    MemoryContext output_context;
    MemoryContext temporary_context;
    MemoryContext old_context;
    evoke_l0_visit_stats first_stats;
    evoke_l0_visit_stats second_stats;

    if (index_relation == NULL || context == NULL || input == NULL ||
        projection_out == NULL || input->query_len > UINT32_MAX ||
        context->root.next_document_slot > UINT32_MAX ||
        (input->query_len > 0 && input->numeric_source &&
         input->query_ids == NULL) ||
        (input->query_len > 0 && !input->numeric_source &&
         input->query_tokens == NULL))
    {
        ereport(ERROR, (errmsg("invalid evoke query L0 projection request")));
    }
    if (context->root.pending_l0.record_count >
        UINT32_MAX - context->root.active_l0.record_count)
    {
        ereport(ERROR, (errmsg("evoke query L0 record count is too large")));
    }

    evoke_page_query_l0_projection_init(&projection);
    memset(&build, 0, sizeof(build));
    memset(&first_stats, 0, sizeof(first_stats));
    memset(&second_stats, 0, sizeof(second_stats));
    build.index_relation = index_relation;
    build.context = context;
    build.input = input;
    build.expected_record_count = context->root.pending_l0.record_count +
        context->root.active_l0.record_count;
    output_context = CurrentMemoryContext;
    temporary_context = AllocSetContextCreate(
        output_context,
        "evoke page query L0 projection",
        ALLOCSET_SMALL_SIZES
    );
    old_context = MemoryContextSwitchTo(temporary_context);

    PG_TRY();
    {
        if (!input->numeric_source && input->query_len > 0)
        {
            build.query_token_lengths = palloc(
                sizeof(*build.query_token_lengths) * input->query_len
            );
            for (size_t query_index = 0;
                 query_index < input->query_len;
                 query_index++)
            {
                if (input->query_tokens[query_index] == NULL)
                {
                    ereport(
                        ERROR,
                        (errmsg("invalid evoke query L0 token"))
                    );
                }
                build.query_token_lengths[query_index] =
                    strlen(input->query_tokens[query_index]);
            }
        }
        if (build.expected_record_count > 0)
        {
            Size record_bytes;

            if (visibility_resolver == NULL ||
                build.expected_record_count >
                    MaxAllocSize / sizeof(*build.record_xids) ||
                build.expected_record_count >
                    MaxAllocSize / sizeof(*build.events) ||
                build.expected_record_count >
                    MaxAllocSize / sizeof(*build.upsert_slots))
            {
                ereport(
                    ERROR,
                    (errmsg("evoke query L0 projection is too large"))
                );
            }
            record_bytes = (Size) build.expected_record_count;
            build.record_xids = palloc(
                sizeof(*build.record_xids) * record_bytes
            );
            build.visibility = palloc0(
                sizeof(*build.visibility) * record_bytes
            );
            build.upsert_slots = palloc(
                sizeof(*build.upsert_slots) * record_bytes
            );
            build.events = palloc(
                sizeof(*build.events) * record_bytes
            );

            evoke_segment_pages_visit_l0_records(
                index_relation,
                &context->root,
                true,
                true,
                evoke_page_query_l0_collect_xid,
                &build,
                &first_stats
            );
            if (build.record_index != build.expected_record_count ||
                first_stats.record_count != build.expected_record_count)
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke query L0 record closure"))
                );
            }
            visibility_resolver(
                visibility_context,
                build.record_xids,
                build.visibility,
                build.expected_record_count
            );
            build.record_index = 0;
            evoke_segment_pages_visit_l0_records(
                index_relation,
                &context->root,
                true,
                true,
                evoke_page_query_l0_collect_visible,
                &build,
                &second_stats
            );
            if (build.record_index != build.expected_record_count ||
                second_stats.record_count != first_stats.record_count ||
                second_stats.page_count != first_stats.page_count ||
                second_stats.payload_bytes != first_stats.payload_bytes)
            {
                ereport(
                    ERROR,
                    (errmsg("evoke query L0 stream changed during projection"))
                );
            }
        }

        old_document_count = context->manifest.document_slot_count;
        if (build.upsert_count > 1)
        {
            qsort(
                build.upsert_slots,
                build.upsert_count,
                sizeof(*build.upsert_slots),
                evoke_page_query_cmp_u32
            );
        }
        {
            uint64 tail_document_count = 0;

            for (size_t upsert_index = 0;
                 upsert_index < build.upsert_count;
                 upsert_index++)
            {
                uint32 document_slot = build.upsert_slots[upsert_index];

                if (upsert_index > 0 &&
                    build.upsert_slots[upsert_index - 1] == document_slot)
                {
                    ereport(
                        ERROR,
                        (errmsg("duplicate evoke query L0 upsert"))
                    );
                }
                if ((uint64) document_slot < old_document_count)
                {
                    continue;
                }
                if ((uint64) document_slot !=
                    old_document_count + tail_document_count)
                {
                    ereport(
                        ERROR,
                        (errmsg("invalid evoke query L0 tail document range"))
                    );
                }
                tail_document_count++;
            }
            if (old_document_count + tail_document_count !=
                context->root.next_document_slot)
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke query L0 document high water"))
                );
            }
        }
        if (build.max_semantic_term_plus_one > Max(
                context->manifest.vocab_size,
                build.max_numeric_term_plus_one))
        {
            ereport(ERROR, (errmsg("invalid evoke semantic L0 term")));
        }

        if (build.event_count > SIZE_MAX - build.upsert_count ||
            build.event_count + build.upsert_count >
                MaxAllocSize / sizeof(*document_slots))
        {
            ereport(ERROR, (errmsg("evoke query L0 document set is too large")));
        }
        document_slot_count = build.event_count + build.upsert_count;
        if (document_slot_count > 0)
        {
            document_slots = palloc(
                sizeof(*document_slots) * document_slot_count
            );
        }
        for (size_t event_index = 0;
             event_index < build.event_count;
             event_index++)
        {
            document_slots[event_index] =
                (uint32) build.events[event_index].document_slot;
        }
        if (build.upsert_count > 0)
        {
            memcpy(
                document_slots + build.event_count,
                build.upsert_slots,
                sizeof(*document_slots) * build.upsert_count
            );
        }
        if (document_slot_count > 1)
        {
            qsort(
                document_slots,
                document_slot_count,
                sizeof(*document_slots),
                evoke_page_query_cmp_u32
            );
        }
        for (size_t slot_index = 0;
             slot_index < document_slot_count;
             slot_index++)
        {
            if (slot_index == 0 ||
                document_slots[slot_index - 1] != document_slots[slot_index])
            {
                document_slots[document_count++] =
                    document_slots[slot_index];
            }
        }
        if (document_count > 0)
        {
            documents = palloc0(sizeof(*documents) * document_count);
        }
        for (size_t document_index = 0;
             document_index < document_count;
             document_index++)
        {
            evoke_page_query_l0_build_document *document =
                &documents[document_index];

            document->document_slot = document_slots[document_index];
            if ((uint64) document->document_slot < old_document_count)
            {
                evoke_document_cow_record record;

                evoke_segment_pages_load_document_record(
                    index_relation,
                    &context->root,
                    &context->manifest,
                    document->document_slot,
                    &record
                );
                if (record.version.document_slot != document->document_slot)
                {
                    ereport(
                        ERROR,
                        (errmsg("invalid evoke query L0 document record"))
                    );
                }
                document->initial_live =
                    evoke_page_query_document_is_live(&record);
                document->live = document->initial_live;
                document->document_length = record.version.document_length;
                document->born_sequence = record.version.born_sequence;
                document->initial_retirement_sequence =
                    record.retirement.retirement_sequence;
                document->heap_block = record.version.heap_block;
                document->heap_offset = record.version.heap_offset;
                document->immutable_last_sequence =
                    evoke_document_cow_record_last_sequence(&record);
                document->l0_placeholder =
                    evoke_document_cow_record_is_l0_owned(&record) &&
                    (record.version.flags &
                     EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) != 0;
            }
        }
        for (size_t upsert_index = 0;
             upsert_index < build.upsert_count;
             upsert_index++)
        {
            evoke_page_query_l0_build_document *document =
                evoke_page_query_find_build_document(
                    documents,
                    document_count,
                    build.upsert_slots[upsert_index]
                );

            if (document == NULL || document->initial_live)
            {
                ereport(ERROR, (errmsg("invalid evoke query L0 upsert slot")));
            }
        }

        visible_document_count = context->manifest.visible_document_count;
        total_document_length = context->manifest.total_document_length;
        for (size_t event_index = 0;
             event_index < build.event_count;
             event_index++)
        {
            const evoke_page_query_l0_build_event *event =
                &build.events[event_index];
            evoke_page_query_l0_build_document *document =
                evoke_page_query_find_build_document(
                    documents,
                    document_count,
                    (uint32) event->document_slot
                );

            if (document == NULL)
            {
                ereport(ERROR, (errmsg("invalid evoke query L0 document")));
            }
            if (!document->l0_placeholder &&
                event->sequence <= document->immutable_last_sequence)
            {
                continue;
            }
            document->visible_event = true;
            if (event->kind == EVOKE_L0_RECORD_UPSERT)
            {
                if (document->live || visible_document_count == UINT64_MAX ||
                    total_document_length >
                        UINT64_MAX - event->document_length)
                {
                    ereport(ERROR, (errmsg("invalid evoke query L0 upsert")));
                }
                document->live = true;
                document->shadows_immutable = true;
                document->document_length = event->document_length;
                document->born_sequence = event->sequence;
                document->heap_block = event->heap_block;
                document->heap_offset = event->heap_offset;
                visible_document_count++;
                total_document_length += event->document_length;
            }
            else if (event->kind == EVOKE_L0_RECORD_RETIRE)
            {
                if (!document->live ||
                    document->document_length != event->document_length ||
                    visible_document_count == 0 ||
                    total_document_length < event->document_length)
                {
                    ereport(
                        ERROR,
                        (
                            errmsg(
                                "invalid evoke query L0 retirement target"
                            ),
                            errdetail(
                                "slot=%u event_sequence=" UINT64_FORMAT
                                " event_length=%u live=%s "
                                "document_length=%u born_sequence="
                                UINT64_FORMAT " retirement_sequence="
                                UINT64_FORMAT " immutable_sequence="
                                UINT64_FORMAT " last_event_kind=%u "
                                "last_event_sequence=" UINT64_FORMAT " "
                                "document_tid=%u/%u placeholder=%s "
                                "manifest=" UINT64_FORMAT
                                " manifest_sequence=" UINT64_FORMAT
                                " pending_segment=" UINT64_FORMAT
                                " pending_records=%u active_segment="
                                UINT64_FORMAT " active_records=%u.",
                                document->document_slot,
                                event->sequence,
                                event->document_length,
                                document->live ? "true" : "false",
                                document->document_length,
                                document->born_sequence,
                                document->initial_retirement_sequence,
                                document->immutable_last_sequence,
                                (unsigned int) document->last_applied_kind,
                                document->last_applied_sequence,
                                document->heap_block,
                                (unsigned int) document->heap_offset,
                                document->l0_placeholder
                                    ? "true"
                                    : "false",
                                context->manifest.manifest_id,
                                context->manifest.max_sequence,
                                context->root.pending_l0.segment_id,
                                context->root.pending_l0.record_count,
                                context->root.active_l0.segment_id,
                                context->root.active_l0.record_count
                            )
                        )
                    );
                }
                document->live = false;
                document->shadows_immutable = true;
                visible_document_count--;
                total_document_length -= event->document_length;
            }
            document->last_applied_kind = event->kind;
            document->last_applied_sequence = event->sequence;
        }
        if (visible_document_count > context->root.next_document_slot)
        {
            ereport(ERROR, (errmsg("invalid evoke query L0 corpus state")));
        }

        if (build.contribution_count > 1)
        {
            qsort(
                build.contributions,
                build.contribution_count,
                sizeof(*build.contributions),
                evoke_page_query_cmp_l0_contribution
            );
        }
        projection.visible_document_count = visible_document_count;
        projection.total_document_length = total_document_length;
        projection.term_count = input->query_len;
        projection.visited_record_count = build.expected_record_count;
        projection.visible_record_count = (uint32) build.event_count;
        projection.snapshot_sensitive = build.snapshot_sensitive;
        if (projection.term_count > 0)
        {
            projection.terms = MemoryContextAllocZero(
                output_context,
                sizeof(*projection.terms) * projection.term_count
            );
        }
        for (size_t document_index = 0;
             document_index < document_count;
             document_index++)
        {
            if (documents[document_index].visible_event)
            {
                projection.document_count++;
            }
        }
        if (projection.document_count > 0)
        {
            projection.documents = MemoryContextAllocZero(
                output_context,
                sizeof(*projection.documents) * projection.document_count
            );
        }
        {
            size_t output_document_index = 0;

            for (size_t document_index = 0;
                 document_index < document_count;
                 document_index++)
            {
                const evoke_page_query_l0_build_document *source =
                    &documents[document_index];
                evoke_page_query_l0_document *target;

                if (!source->visible_event)
                {
                    continue;
                }
                target = &projection.documents[output_document_index++];
                target->document_slot = source->document_slot;
                target->document_length = source->document_length;
                target->born_sequence = source->born_sequence;
                target->heap_block = source->heap_block;
                target->heap_offset = source->heap_offset;
                target->live = source->live;
                target->shadows_immutable = source->shadows_immutable;
            }
        }

        for (size_t contribution_index = 0;
             contribution_index < build.contribution_count;
             contribution_index++)
        {
            const evoke_page_query_l0_build_contribution *source =
                &build.contributions[contribution_index];
            evoke_page_query_l0_build_document *document =
                evoke_page_query_find_build_document(
                    documents,
                    document_count,
                    source->document_slot
                );

            if (document != NULL && document->live &&
                document->visible_event)
            {
                projection.contribution_count++;
            }
        }
        if (projection.contribution_count > 0)
        {
            projection.contributions = MemoryContextAllocZero(
                output_context,
                sizeof(*projection.contributions) *
                    projection.contribution_count
            );
        }
        {
            size_t output_index = 0;
            uint32 previous_lexical_document = UINT32_MAX;
            uint32 previous_lexical_query = UINT32_MAX;

            for (size_t contribution_index = 0;
                 contribution_index < build.contribution_count;
                 contribution_index++)
            {
                const evoke_page_query_l0_build_contribution *source =
                    &build.contributions[contribution_index];
                evoke_page_query_l0_build_document *build_document =
                    evoke_page_query_find_build_document(
                        documents,
                        document_count,
                        source->document_slot
                    );
                const evoke_page_query_l0_document *public_document;
                size_t public_document_index;

                if (build_document == NULL || !build_document->live ||
                    !build_document->visible_event)
                {
                    continue;
                }
                public_document = evoke_page_query_l0_find_document(
                    &projection,
                    source->document_slot
                );
                if (public_document == NULL ||
                    source->contribution.query_index >= projection.term_count)
                {
                    ereport(
                        ERROR,
                        (errmsg("invalid evoke query L0 contribution"))
                    );
                }
                public_document_index = (size_t) (
                    public_document - projection.documents
                );
                if (projection.documents[public_document_index].
                        contribution_count == 0)
                {
                    projection.documents[public_document_index].
                        first_contribution = output_index;
                }
                projection.contributions[output_index++] =
                    source->contribution;
                projection.documents[public_document_index].
                    contribution_count++;

                if (source->contribution.kind ==
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    evoke_page_query_l0_term_stats *term =
                        &projection.terms[
                            source->contribution.query_index
                        ];

                    term->lexical_seen = true;
                    if (previous_lexical_document == source->document_slot &&
                        previous_lexical_query ==
                            source->contribution.query_index)
                    {
                        ereport(
                            ERROR,
                            (errmsg("duplicate evoke query L0 lexical atom"))
                        );
                    }
                    if (term->lexical_document_frequency == UINT32_MAX)
                    {
                        ereport(
                            ERROR,
                            (errmsg("evoke query L0 frequency overflow"))
                        );
                    }
                    term->lexical_document_frequency++;
                    previous_lexical_document = source->document_slot;
                    previous_lexical_query =
                        source->contribution.query_index;
                }
                else if (source->contribution.kind ==
                         EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
                {
                    projection.terms[source->contribution.query_index].
                        semantic_seen = true;
                }
                else
                {
                    ereport(
                        ERROR,
                        (errmsg("invalid evoke query L0 posting kind"))
                    );
                }
            }
            if (output_index != projection.contribution_count)
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke query L0 contribution closure"))
                );
            }
        }

        MemoryContextSwitchTo(old_context);
        MemoryContextDelete(temporary_context);
        temporary_context = NULL;
        evoke_page_query_l0_projection_free(projection_out);
        *projection_out = projection;
        evoke_page_query_l0_projection_init(&projection);
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(old_context);
        if (temporary_context != NULL)
        {
            MemoryContextDelete(temporary_context);
        }
        evoke_page_query_l0_projection_free(&projection);
        PG_RE_THROW();
    }
    PG_END_TRY();
}

static evoke_status
evoke_page_query_cursor_load_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_term *term,
    evoke_page_query_cursor *cursor,
    evoke_page_query_stats *stats
)
{
    const evoke_segment_query_run *run;

    if (index_relation == NULL || context == NULL || term == NULL ||
        cursor == NULL || stats == NULL ||
        cursor->run_index >= term->plan.run_count)
    {
        return EVOKE_ERR_INVALID;
    }
    run = &term->plan.runs[cursor->run_index];
    while (cursor->block_index < run->block_count)
    {
        evoke_posting_block_record record;

        evoke_segment_pages_load_query_term_block_record(
            index_relation,
            context,
            &term->plan,
            cursor->run_index,
            cursor->block_index,
            &record
        );
        if (record.posting_count == 0)
        {
            cursor->block_index++;
            continue;
        }
        evoke_segment_pages_load_query_term_block_from_record(
            index_relation,
            context,
            &term->plan,
            cursor->run_index,
            cursor->block_index,
            &record,
            &cursor->block
        );
        break;
    }
    if (cursor->block_index >= run->block_count)
    {
        cursor->exhausted = true;
        return EVOKE_OK;
    }
    stats->posting_block_reads++;
    if (cursor->block.view.len == 0 ||
        cursor->block.view.len >
            EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS ||
        (cursor->have_previous_document &&
         cursor->block.document_slots[0] <=
            cursor->previous_document_slot))
    {
        return EVOKE_ERR_FORMAT;
    }
    cursor->posting_index = 0;
    cursor->exhausted = false;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_cursor_advance(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_term *term,
    evoke_page_query_cursor *cursor,
    evoke_page_query_stats *stats
)
{
    if (cursor == NULL || cursor->exhausted ||
        cursor->posting_index >= cursor->block.view.len)
    {
        return EVOKE_ERR_INVALID;
    }
    cursor->previous_document_slot =
        cursor->block.document_slots[cursor->posting_index];
    cursor->have_previous_document = true;
    cursor->posting_index++;
    if (cursor->posting_index < cursor->block.view.len)
    {
        if (cursor->block.document_slots[cursor->posting_index] <=
            cursor->previous_document_slot)
        {
            return EVOKE_ERR_FORMAT;
        }
        return EVOKE_OK;
    }
    cursor->block_index++;
    return evoke_page_query_cursor_load_block(
        index_relation,
        context,
        term,
        cursor,
        stats
    );
}

static bool
evoke_page_query_run_needs_live_frequency(
    const evoke_segment_query_context *context,
    const evoke_segment_query_run *run
)
{
    return run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL ||
        (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT &&
         evoke_method_requires_nonoccurrence(
             context->query_contract.params.method));
}

static evoke_status
evoke_page_query_reset_cursors(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    size_t cursor_capacity,
    bool frequency_only,
    size_t *cursor_count_out,
    evoke_page_query_stats *stats
)
{
    size_t cursor_index = 0;

    if (cursor_count_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *cursor_count_out = 0;
    memset(
        cleanup->cursors,
        0,
        cursor_capacity * sizeof(*cleanup->cursors)
    );
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        evoke_page_query_term *term = &cleanup->terms[term_index];

        term->first_cursor = (uint32) cursor_index;
        term->cursor_count = 0;
        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            evoke_page_query_cursor *cursor;
            evoke_status status;

            if (frequency_only &&
                !evoke_page_query_run_needs_live_frequency(
                    context,
                    &term->plan.runs[run_index]))
            {
                continue;
            }
            if (cursor_index >= cursor_capacity)
            {
                return EVOKE_ERR_RANGE;
            }
            cursor = &cleanup->cursors[cursor_index++];
            term->cursor_count++;
            cursor->term_index = (uint32) term_index;
            cursor->run_index = run_index;
            status = evoke_page_query_cursor_load_block(
                index_relation,
                context,
                term,
                cursor,
                stats
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
        }
    }
    *cursor_count_out = cursor_index;
    return !frequency_only && cursor_index != cursor_capacity
        ? EVOKE_ERR_FORMAT
        : EVOKE_OK;
}

static bool
evoke_page_query_next_document(
    const evoke_page_query_cursor *cursors,
    size_t cursor_count,
    uint32 *document_slot_out
)
{
    uint32 document_slot = UINT32_MAX;
    bool found = false;

    for (size_t cursor_index = 0;
         cursor_index < cursor_count;
         cursor_index++)
    {
        const evoke_page_query_cursor *cursor =
            &cursors[cursor_index];
        uint32 candidate;

        if (cursor->exhausted)
        {
            continue;
        }
        candidate = cursor->block.document_slots[
            cursor->posting_index
        ];
        if (!found || candidate < document_slot)
        {
            document_slot = candidate;
            found = true;
        }
    }
    if (found)
    {
        *document_slot_out = document_slot;
    }
    return found;
}

static evoke_status
evoke_page_query_load_document(
    Relation index_relation,
    const evoke_segment_query_context *context,
    uint32 document_slot,
    evoke_page_query_cleanup *cleanup,
    bool *document_block_loaded,
    const evoke_document_cow_record **record_out,
    evoke_page_query_stats *stats
)
{
    uint32 block_id = document_slot >>
        context->query_contract.block_shift;
    uint32 local_slot;

    if (!*document_block_loaded ||
        cleanup->document_block.block_id != block_id)
    {
        /*
         * Ranked accelerator and BMP candidates can revisit immutable
         * document blocks out of order. The COW reader supports arbitrary
         * block loads; only the cached block identity determines a hit.
         */
        evoke_segment_pages_load_query_document_block(
            index_relation,
            context,
            &cleanup->document_reader,
            block_id,
            &cleanup->document_block
        );
        *document_block_loaded = true;
        stats->document_block_reads++;
        if (cleanup->document_block.record_count >
            stats->max_document_block_records)
        {
            stats->max_document_block_records =
                cleanup->document_block.record_count;
        }
    }
    if (document_slot <
        cleanup->document_block.first_document_slot)
    {
        return EVOKE_ERR_FORMAT;
    }
    local_slot = document_slot -
        cleanup->document_block.first_document_slot;
    if (local_slot >= cleanup->document_block.record_count ||
        cleanup->document_block.records[
            local_slot
        ].version.document_slot != document_slot)
    {
        return EVOKE_ERR_FORMAT;
    }
    *record_out = &cleanup->document_block.records[local_slot];
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_count_live_frequencies(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    size_t cursor_count,
    evoke_page_query_stats *stats
)
{
    bool document_block_loaded = false;
    uint32 document_slot;
    uint64 interrupt_counter = 0;
    size_t frequency_cursor_count = 0;
    evoke_status status;

    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        cleanup->terms[term_index].immutable_live_document_frequency = 0;
        cleanup->terms[term_index].live_document_frequency = 0;
    }
    status = evoke_page_query_reset_cursors(
        index_relation,
        context,
        cleanup,
        term_count,
        cursor_count,
        true,
        &frequency_cursor_count,
        stats
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    while (evoke_page_query_next_document(
        cleanup->cursors,
        frequency_cursor_count,
        &document_slot))
    {
        const evoke_document_cow_record *record;
        bool live;

        if ((interrupt_counter++ & EVOKE_PAGE_QUERY_INTERRUPT_MASK) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
        status = evoke_page_query_load_document(
            index_relation,
            context,
            document_slot,
            cleanup,
            &document_block_loaded,
            &record,
            stats
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        live = evoke_page_query_document_is_live(record);
        if (live && projection != NULL)
        {
            const evoke_page_query_l0_document *projected_document =
                evoke_page_query_l0_find_document(
                    projection,
                    document_slot
                );

            if (projected_document != NULL &&
                projected_document->shadows_immutable)
            {
                live = false;
            }
        }
        stats->documents_examined++;
        for (size_t cursor_index = 0;
             cursor_index < frequency_cursor_count;
             cursor_index++)
        {
            evoke_page_query_cursor *cursor =
                &cleanup->cursors[cursor_index];
            evoke_page_query_term *term;
            const evoke_segment_query_run *run;

            if (cursor->exhausted ||
                cursor->block.document_slots[
                    cursor->posting_index
                ] != document_slot)
            {
                continue;
            }
            term = &cleanup->terms[cursor->term_index];
            run = &term->plan.runs[cursor->run_index];
            stats->postings_examined++;
            if (live &&
                (run->kind ==
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL ||
                 run->kind ==
                    EVOKE_POSTING_EXTENT_LEXICAL_IMPACT))
            {
                if (term->live_document_frequency == UINT32_MAX)
                {
                    return EVOKE_ERR_RANGE;
                }
                term->immutable_live_document_frequency++;
                term->live_document_frequency++;
            }
            status = evoke_page_query_cursor_advance(
                index_relation,
                context,
                term,
                cursor,
                stats
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
        }
    }
    if (projection != NULL)
    {
        for (size_t term_index = 0;
             term_index < term_count;
             term_index++)
        {
            evoke_page_query_term *term = &cleanup->terms[term_index];
            uint32 added = term->query_index == UINT32_MAX
                ? 0
                : projection->terms[term->query_index].
                    lexical_document_frequency;

            if (term->live_document_frequency > UINT32_MAX - added)
            {
                return EVOKE_ERR_RANGE;
            }
            term->live_document_frequency += added;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_prepare_term_statistics(
    const evoke_segment_query_context *context,
    uint64 visible_document_count,
    double average_document_length,
    evoke_page_query_cleanup *cleanup,
    size_t term_count
)
{
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        evoke_page_query_term *term = &cleanup->terms[term_index];

        if (term->immutable_live_document_frequency >
            term->plan.raw_document_frequency ||
        term->live_document_frequency >
            visible_document_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (!term->lexical_neutral_seen &&
            !(term->lexical_impact_seen &&
              evoke_method_requires_nonoccurrence(
                  context->query_contract.params.method)))
        {
            continue;
        }
        if (term->live_document_frequency == 0)
        {
            if (term->lexical_impact_seen)
            {
                return EVOKE_ERR_FORMAT;
            }
            continue;
        }
        term->idf = evoke_score_idf(
            context->query_contract.params.idf_method,
            (double) term->live_document_frequency,
            (double) visible_document_count
        );
        if (evoke_method_requires_nonoccurrence(
                context->query_contract.params.method))
        {
            term->nonoccurrence = (float) (
                term->idf * evoke_score_tfc(
                    context->query_contract.params.method,
                    0.0,
                    average_document_length,
                    average_document_length,
                    context->query_contract.params.k1,
                    context->query_contract.params.b,
                    context->query_contract.params.delta
                )
            );
            if (!isfinite(term->nonoccurrence))
            {
                return EVOKE_ERR_RANGE;
            }
        }
    }
    return EVOKE_OK;
}

static bool
evoke_page_query_semantic_only_postings(
    const evoke_page_query_term *term,
    uint64 *postings_out
)
{
    bool semantic_seen = false;
    uint64 postings = 0;

    for (uint32 run_index = 0;
         run_index < term->plan.run_count;
         run_index++)
    {
        evoke_posting_extent_kind kind = term->plan.runs[run_index].kind;

        if (kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL ||
            kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
        {
            return false;
        }
        if (kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            continue;
        }
        semantic_seen = true;
        if (postings > UINT64_MAX -
            term->plan.runs[run_index].posting_count)
        {
            postings = UINT64_MAX;
        }
        else
        {
            postings += term->plan.runs[run_index].posting_count;
        }
    }
    if (semantic_seen && postings_out != NULL)
    {
        *postings_out = postings;
    }
    return semantic_seen;
}

static size_t
evoke_page_query_apply_test_df_limit(
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    uint64 visible_document_count,
    const evoke_page_query_l0_projection *projection,
    evoke_page_query_stats *stats
)
{
    size_t retained = 0;
    uint64 semantic_postings = 0;

    if (evoke_test_query_max_df_ratio >= 1.0 ||
        projection != NULL || visible_document_count == 0)
    {
        return term_count;
    }
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        uint64 term_postings = 0;

        if (!evoke_page_query_semantic_only_postings(
                &cleanup->terms[term_index],
                &term_postings
            ))
        {
            continue;
        }
        if (semantic_postings > UINT64_MAX - term_postings)
        {
            semantic_postings = UINT64_MAX;
        }
        else
        {
            semantic_postings += term_postings;
        }
    }
    if (evoke_test_query_semantic_work_target_postings > 0 &&
        semantic_postings <=
            (uint64) evoke_test_query_semantic_work_target_postings)
    {
        return term_count;
    }
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        evoke_page_query_term *term = &cleanup->terms[term_index];
        uint64 term_postings = 0;
        bool semantic_only = evoke_page_query_semantic_only_postings(
            term,
            &term_postings
        );
        bool preserve = !semantic_only ||
            term->query_index == UINT32_MAX ||
            (double) term->plan.raw_document_frequency <=
                evoke_test_query_max_df_ratio *
                    (double) visible_document_count;

        if (!preserve)
        {
            if (stats->query_df_pruned_term_count < UINT32_MAX)
            {
                stats->query_df_pruned_term_count++;
            }
            if (stats->query_df_pruned_postings >
                UINT64_MAX - term_postings)
            {
                stats->query_df_pruned_postings = UINT64_MAX;
            }
            else
            {
                stats->query_df_pruned_postings += term_postings;
            }
            cleanup->query_term_map[term->query_index] = UINT32_MAX;
            continue;
        }
        if (retained != term_index)
        {
            cleanup->terms[retained] = *term;
        }
        if (cleanup->terms[retained].query_index != UINT32_MAX)
        {
            cleanup->query_term_map[
                cleanup->terms[retained].query_index
            ] = (uint32) retained;
        }
        retained++;
    }
    return retained;
}

typedef struct evoke_page_query_error_budget_candidate
{
    size_t term_index;
    double absolute_bound;
    uint64 postings;
} evoke_page_query_error_budget_candidate;

static bool
evoke_page_query_semantic_plan_absolute_bound(
    const evoke_segment_query_term_plan *plan,
    float query_weight,
    double *bound_out,
    uint64 *postings_out
)
{
    bool semantic_seen = false;
    double bound = 0.0;
    uint64 postings = 0;

    for (uint32 run_index = 0;
         run_index < plan->run_count;
         run_index++)
    {
        evoke_posting_extent_kind kind = plan->runs[run_index].kind;

        if (kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL ||
            kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
        {
            return false;
        }
        if (kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            continue;
        }
        if (!plan->runs[run_index].semantic_bmp.available)
        {
            return false;
        }
        semantic_seen = true;
        bound += Max(
            fabs(
                (double) query_weight *
                (double) plan->runs[run_index].semantic_bmp.min_impact
            ),
            fabs(
                (double) query_weight *
                (double) plan->runs[run_index].semantic_bmp.max_impact
            )
        );
        if (!isfinite(bound))
        {
            return false;
        }
        if (postings > UINT64_MAX -
            plan->runs[run_index].posting_count)
        {
            postings = UINT64_MAX;
        }
        else
        {
            postings += plan->runs[run_index].posting_count;
        }
    }
    if (semantic_seen && bound_out != NULL)
    {
        *bound_out = bound;
    }
    if (semantic_seen && postings_out != NULL)
    {
        *postings_out = postings;
    }
    return semantic_seen;
}

static bool
evoke_page_query_semantic_term_absolute_bound(
    const evoke_page_query_term *term,
    double *bound_out,
    uint64 *postings_out
)
{
    return evoke_page_query_semantic_plan_absolute_bound(
        &term->plan,
        term->query_weight,
        bound_out,
        postings_out
    );
}

static int
evoke_page_query_compare_error_budget_candidates(
    const void *left_ptr,
    const void *right_ptr
)
{
    const evoke_page_query_error_budget_candidate *left = left_ptr;
    const evoke_page_query_error_budget_candidate *right = right_ptr;
    long double left_yield;
    long double right_yield;

    if (left->absolute_bound == 0.0 || right->absolute_bound == 0.0)
    {
        if (left->absolute_bound == 0.0 && right->absolute_bound != 0.0)
        {
            return -1;
        }
        if (left->absolute_bound != 0.0 && right->absolute_bound == 0.0)
        {
            return 1;
        }
    }
    left_yield = (long double) left->postings *
        (long double) right->absolute_bound;
    right_yield = (long double) right->postings *
        (long double) left->absolute_bound;
    if (left_yield > right_yield)
    {
        return -1;
    }
    if (left_yield < right_yield)
    {
        return 1;
    }
    if (left->postings > right->postings)
    {
        return -1;
    }
    if (left->postings < right->postings)
    {
        return 1;
    }
    if (left->term_index < right->term_index)
    {
        return -1;
    }
    if (left->term_index > right->term_index)
    {
        return 1;
    }
    return 0;
}

static size_t
evoke_page_query_apply_test_error_budget(
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    const evoke_page_query_l0_projection *projection,
    evoke_page_query_stats *stats
)
{
    evoke_page_query_error_budget_candidate *candidates;
    bool *omit;
    size_t candidate_count = 0;
    size_t omitted_count = 0;
    size_t omit_limit;
    size_t retained = 0;
    double total_bound = 0.0;
    double omitted_bound = 0.0;
    double budget;

    /*
     * For every omitted semantic term i, absolute_bound_i is an upper
     * bound on |q_i * d_i| for any document.  The triangle inequality then
     * guarantees that the per-document semantic score error is no greater
     * than the sum admitted below.
     */

    if (term_count == 0 ||
        evoke_test_query_semantic_error_budget_ratio <= 0.0 ||
        projection != NULL)
    {
        return term_count;
    }
    candidates = palloc(sizeof(*candidates) * term_count);
    omit = palloc0(sizeof(*omit) * term_count);
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        const evoke_page_query_term *term = &cleanup->terms[term_index];
        double bound;
        uint64 postings;

        if (term->query_index != UINT32_MAX &&
            evoke_page_query_semantic_term_absolute_bound(
                term,
                &bound,
                &postings
            ))
        {
            if (!isfinite(total_bound + bound))
            {
                pfree(omit);
                pfree(candidates);
                return term_count;
            }
            total_bound += bound;
            candidates[candidate_count].term_index = term_index;
            candidates[candidate_count].absolute_bound = bound;
            candidates[candidate_count].postings = postings;
            candidate_count++;
        }
    }
    if (candidate_count == 0 || total_bound == 0.0)
    {
        pfree(omit);
        pfree(candidates);
        return term_count;
    }
    budget = evoke_test_query_semantic_error_budget_ratio * total_bound;
    omit_limit = candidate_count;
    if (evoke_test_query_semantic_min_support_ratio > 0.0)
    {
        size_t retain_count = (size_t) ceil(
            evoke_test_query_semantic_min_support_ratio *
            (double) candidate_count
        );

        retain_count = Min(retain_count, candidate_count);
        omit_limit = candidate_count - retain_count;
    }
    qsort(
        candidates,
        candidate_count,
        sizeof(*candidates),
        evoke_page_query_compare_error_budget_candidates
    );
    for (size_t candidate_index = 0;
         candidate_index < candidate_count;
         candidate_index++)
    {
        const evoke_page_query_error_budget_candidate *candidate =
            &candidates[candidate_index];
        double next_omitted_bound =
            omitted_bound + candidate->absolute_bound;

        if (omitted_count >= omit_limit)
        {
            break;
        }
        if (!isfinite(next_omitted_bound) || next_omitted_bound > budget)
        {
            continue;
        }
        omit[candidate->term_index] = true;
        omitted_count++;
        omitted_bound = next_omitted_bound;
        if (stats->query_error_budget_pruned_term_count < UINT32_MAX)
        {
            stats->query_error_budget_pruned_term_count++;
        }
        if (stats->query_error_budget_pruned_postings >
            UINT64_MAX - candidate->postings)
        {
            stats->query_error_budget_pruned_postings = UINT64_MAX;
        }
        else
        {
            stats->query_error_budget_pruned_postings +=
                candidate->postings;
        }
    }
    stats->query_semantic_total_absolute_bound = total_bound;
    stats->query_semantic_omitted_absolute_bound = omitted_bound;
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        evoke_page_query_term *term = &cleanup->terms[term_index];
        bool preserve = !omit[term_index];

        if (!preserve)
        {
            cleanup->query_term_map[term->query_index] = UINT32_MAX;
            continue;
        }
        if (retained != term_index)
        {
            cleanup->terms[retained] = *term;
        }
        if (cleanup->terms[retained].query_index != UINT32_MAX)
        {
            cleanup->query_term_map[
                cleanup->terms[retained].query_index
            ] = (uint32) retained;
        }
        retained++;
    }
    pfree(omit);
    pfree(candidates);
    return retained;
}

static bool
evoke_page_query_test_omit_semantic_impact(
    const evoke_segment_query_run *run,
    float impact,
    evoke_page_query_stats *stats
)
{
    double max_absolute_impact;
    double floor;

    if (evoke_test_query_semantic_impact_floor_ratio <= 0.0 ||
        run == NULL || stats == NULL ||
        run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        !run->semantic_bmp.available)
    {
        return false;
    }
    max_absolute_impact = Max(
        fabs((double) run->semantic_bmp.min_impact),
        fabs((double) run->semantic_bmp.max_impact)
    );
    floor = evoke_test_query_semantic_impact_floor_ratio *
        max_absolute_impact;
    if (fabs((double) impact) > floor)
    {
        return false;
    }
    if (stats->query_impact_floor_omitted_postings < UINT64_MAX)
    {
        stats->query_impact_floor_omitted_postings++;
    }
    return true;
}

static evoke_status
evoke_page_query_score_l0_document(
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const evoke_page_query_l0_document *document,
    evoke_page_query_cleanup *cleanup,
    double average_document_length,
    float *score_out
)
{
    float score = 0.0f;

    if (context == NULL || projection == NULL || document == NULL ||
        cleanup == NULL || score_out == NULL || !document->live ||
        document->first_contribution > projection->contribution_count ||
        document->contribution_count >
            projection->contribution_count - document->first_contribution)
    {
        return EVOKE_ERR_INVALID;
    }
    for (size_t local_index = 0;
         local_index < document->contribution_count;
         local_index++)
    {
        const evoke_page_query_l0_contribution *contribution =
            &projection->contributions[
                document->first_contribution + local_index
            ];
        uint32 term_index;
        evoke_page_query_term *term;

        if (contribution->query_index >= projection->term_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        term_index = cleanup->query_term_map[contribution->query_index];
        if (term_index == UINT32_MAX)
        {
            return EVOKE_ERR_FORMAT;
        }
        term = &cleanup->terms[term_index];
        if (contribution->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
        {
            double tfc;

            if (term->live_document_frequency == 0)
            {
                return EVOKE_ERR_FORMAT;
            }
            tfc = evoke_score_tfc(
                context->query_contract.params.method,
                (double) contribution->value.term_frequency,
                (double) document->document_length,
                average_document_length,
                context->query_contract.params.k1,
                context->query_contract.params.b,
                context->query_contract.params.delta
            );
            score += (float) (
                (double) term->query_weight *
                (term->idf * tfc - (double) term->nonoccurrence)
            );
        }
        else if (contribution->kind ==
                 EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
        {
            score += term->query_weight * contribution->value.impact;
        }
        else
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    *score_out = score;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_block_cursor_load(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_term *term,
    evoke_page_query_block_cursor *cursor,
    evoke_page_query_stats *stats
)
{
    const evoke_segment_query_run *run;

    if (index_relation == NULL || context == NULL || term == NULL ||
        cursor == NULL || stats == NULL ||
        cursor->run_index >= term->plan.run_count)
    {
        return EVOKE_ERR_INVALID;
    }
    run = &term->plan.runs[cursor->run_index];
    while (cursor->block_index < run->block_count)
    {
        if (cursor->window_count == 0 ||
            cursor->block_index < cursor->window_first_block_index ||
            cursor->block_index >=
                cursor->window_first_block_index + cursor->window_count)
        {
            cursor->window_first_block_index = cursor->block_index;
            cursor->window_count =
                evoke_segment_pages_load_query_term_block_records(
                    index_relation,
                    context,
                    &term->plan,
                    cursor->run_index,
                    cursor->block_index,
                    cursor->record_window,
                    EVOKE_SEGMENT_QUERY_BLOCK_RECORD_WINDOW
                );
            stats->posting_block_metadata_reads++;
        }
        cursor->record = cursor->record_window[
            cursor->block_index - cursor->window_first_block_index
        ];
        if (cursor->record.posting_count != 0)
        {
            break;
        }
        cursor->block_index++;
    }
    if (cursor->block_index >= run->block_count)
    {
        cursor->exhausted = true;
        return EVOKE_OK;
    }
    if (cursor->have_previous_block &&
        cursor->record.block_id <= cursor->previous_block_id)
    {
        return EVOKE_ERR_FORMAT;
    }
    cursor->exhausted = false;
    return EVOKE_OK;
}

static bool
evoke_page_query_block_cursor_precedes(
    const evoke_page_query_cleanup *cleanup,
    uint32 left_index,
    uint32 right_index
)
{
    const evoke_page_query_block_cursor *left =
        &cleanup->block_cursors[left_index];
    const evoke_page_query_block_cursor *right =
        &cleanup->block_cursors[right_index];

    return left->record.block_id < right->record.block_id ||
        (left->record.block_id == right->record.block_id &&
         left_index < right_index);
}

static void
evoke_page_query_block_cursor_heap_push(
    evoke_page_query_cleanup *cleanup,
    uint32 cursor_index
)
{
    size_t slot = cleanup->block_cursor_heap_count++;

    while (slot > 0)
    {
        size_t parent = (slot - 1) / 2;
        uint32 parent_index = cleanup->block_cursor_heap[parent];

        if (!evoke_page_query_block_cursor_precedes(
                cleanup,
                cursor_index,
                parent_index))
        {
            break;
        }
        cleanup->block_cursor_heap[slot] = parent_index;
        slot = parent;
    }
    cleanup->block_cursor_heap[slot] = cursor_index;
}

static uint32
evoke_page_query_block_cursor_heap_pop(evoke_page_query_cleanup *cleanup)
{
    uint32 result = cleanup->block_cursor_heap[0];
    uint32 replacement;
    size_t slot = 0;

    cleanup->block_cursor_heap_count--;
    if (cleanup->block_cursor_heap_count == 0)
    {
        return result;
    }
    replacement = cleanup->block_cursor_heap[
        cleanup->block_cursor_heap_count
    ];
    while (true)
    {
        size_t left = slot * 2 + 1;
        size_t right = left + 1;
        size_t child;

        if (left >= cleanup->block_cursor_heap_count)
        {
            break;
        }
        child = left;
        if (right < cleanup->block_cursor_heap_count &&
            evoke_page_query_block_cursor_precedes(
                cleanup,
                cleanup->block_cursor_heap[right],
                cleanup->block_cursor_heap[left]))
        {
            child = right;
        }
        if (!evoke_page_query_block_cursor_precedes(
                cleanup,
                cleanup->block_cursor_heap[child],
                replacement))
        {
            break;
        }
        cleanup->block_cursor_heap[slot] =
            cleanup->block_cursor_heap[child];
        slot = child;
    }
    cleanup->block_cursor_heap[slot] = replacement;
    return result;
}

static evoke_status
evoke_page_query_prepare_next_block_group(
    evoke_page_query_cleanup *cleanup,
    uint32 *block_id_out,
    bool *found_out
)
{
    uint32 block_id;

    if (cleanup == NULL || block_id_out == NULL || found_out == NULL ||
        cleanup->block_cursor_group_count != 0)
    {
        return EVOKE_ERR_INVALID;
    }
    if (cleanup->block_cursor_heap_count == 0)
    {
        *found_out = false;
        return EVOKE_OK;
    }
    block_id = cleanup->block_cursors[
        cleanup->block_cursor_heap[0]
    ].record.block_id;
    do
    {
        cleanup->block_cursor_group[
            cleanup->block_cursor_group_count++
        ] = evoke_page_query_block_cursor_heap_pop(cleanup);
    } while (cleanup->block_cursor_heap_count > 0 &&
             cleanup->block_cursors[
                 cleanup->block_cursor_heap[0]
             ].record.block_id == block_id);
    *block_id_out = block_id;
    *found_out = true;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_reset_block_cursors(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    size_t cursor_count,
    evoke_page_query_stats *stats
)
{
    size_t cursor_index = 0;

    if (cursor_count == 0)
    {
        return EVOKE_OK;
    }
    free(cleanup->block_cursors);
    free(cleanup->block_cursor_heap);
    free(cleanup->block_cursor_group);
    cleanup->block_cursors = NULL;
    cleanup->block_cursor_heap = NULL;
    cleanup->block_cursor_group = NULL;
    cleanup->block_cursor_heap_count = 0;
    cleanup->block_cursor_group_count = 0;
    cleanup->block_cursors = calloc(
        cursor_count,
        sizeof(*cleanup->block_cursors)
    );
    cleanup->block_cursor_heap = calloc(
        cursor_count,
        sizeof(*cleanup->block_cursor_heap)
    );
    cleanup->block_cursor_group = calloc(
        cursor_count,
        sizeof(*cleanup->block_cursor_group)
    );
    if (cleanup->block_cursors == NULL ||
        cleanup->block_cursor_heap == NULL ||
        cleanup->block_cursor_group == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    stats->posting_block_metadata_cache_bytes = Max(
        stats->posting_block_metadata_cache_bytes,
        (uint64) cursor_count *
            sizeof(cleanup->block_cursors[0].record_window)
    );
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        const evoke_page_query_term *term = &cleanup->terms[term_index];

        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            evoke_page_query_block_cursor *cursor;
            evoke_status status;

            if (cursor_index >= cursor_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            cursor = &cleanup->block_cursors[cursor_index++];
            cursor->term_index = (uint32) term_index;
            cursor->run_index = run_index;
            status = evoke_page_query_block_cursor_load(
                index_relation,
                context,
                term,
                cursor,
                stats
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
            if (!cursor->exhausted)
            {
                evoke_page_query_block_cursor_heap_push(
                    cleanup,
                    (uint32) (cursor_index - 1)
                );
            }
        }
    }
    return cursor_index == cursor_count
        ? EVOKE_OK
        : EVOKE_ERR_FORMAT;
}

static evoke_status
evoke_page_query_block_upper_bound(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    uint32 block_id,
    uint64 visible_document_count,
    uint64 total_document_length,
    const evoke_document_cow_length_extrema *document_extrema_hint,
    float *upper_bound_out
)
{
    evoke_document_cow_length_extrema document_extrema = {0};
    evoke_index index_shape;
    evoke_corpus_stats corpus_stats;
    double total_upper_bound = 0.0;
    bool document_extrema_loaded = false;

    if (upper_bound_out == NULL || visible_document_count == 0 ||
        total_document_length == 0 ||
        context->manifest.document_slot_count > UINT32_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&index_shape, 0, sizeof(index_shape));
    index_shape.num_docs =
        (uint32) context->manifest.document_slot_count;
    index_shape.params = context->query_contract.params;
    memset(&corpus_stats, 0, sizeof(corpus_stats));
    corpus_stats.document_count = visible_document_count;
    corpus_stats.total_document_length = total_document_length;

    for (size_t group_index = 0;
         group_index < cleanup->block_cursor_group_count;
         group_index++)
    {
        uint32 cursor_index = cleanup->block_cursor_group[group_index];
        const evoke_page_query_block_cursor *cursor =
            &cleanup->block_cursors[cursor_index];
        const evoke_page_query_term *term;
        evoke_posting_block_bound bound;
        float contribution_bound;
        evoke_status status;

        if (cursor->exhausted || cursor->record.block_id != block_id)
        {
            continue;
        }
        term = &cleanup->terms[cursor->term_index];
        if (cursor->record.kind ==
                EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
            !document_extrema_loaded)
        {
            if (document_extrema_hint != NULL)
            {
                document_extrema = *document_extrema_hint;
            }
            else
            {
                evoke_segment_pages_load_document_block_extrema(
                    index_relation,
                    &context->root,
                    &context->manifest,
                    &context->query_contract,
                    block_id,
                    &document_extrema
                );
            }
            if (document_extrema.document_count == 0)
            {
                return EVOKE_ERR_FORMAT;
            }
            document_extrema_loaded = true;
        }
        memset(&bound, 0, sizeof(bound));
        bound.posting_offset = cursor->record.posting_offset;
        bound.posting_count = cursor->record.posting_count;
        bound.block_id = cursor->record.block_id;
        bound.first_document_id = cursor->record.first_document_id;
        bound.last_document_id = cursor->record.last_document_id;
        bound.min_term_frequency = cursor->record.min_term_frequency;
        bound.max_term_frequency = cursor->record.max_term_frequency;
        bound.min_impact = cursor->record.min_impact;
        bound.max_impact = cursor->record.max_impact;
        bound.kind = cursor->record.kind;
        if (document_extrema_loaded)
        {
            bound.min_document_length =
                document_extrema.min_document_length;
            bound.max_document_length =
                document_extrema.max_document_length;
        }
        /*
         * A term run may still contain retired postings until reclamation.
         * Frequency counting has already proved that such a lexical run has
         * no visible contribution, so its safe upper bound is zero.
         */
        if (bound.kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
            term->live_document_frequency == 0)
        {
            continue;
        }
        status = evoke_posting_block_score_upper_bound(
            &index_shape,
            &corpus_stats,
            term->live_document_frequency,
            &bound,
            term->query_weight,
            &contribution_bound
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        total_upper_bound = nextafter(
            total_upper_bound + (double) contribution_bound,
            INFINITY
        );
    }
    if (!isfinite(total_upper_bound) || total_upper_bound > FLT_MAX)
    {
        *upper_bound_out = INFINITY;
    }
    else
    {
        *upper_bound_out = nextafterf(
            (float) total_upper_bound,
            INFINITY
        );
    }
    return EVOKE_OK;
}

static bool
evoke_page_query_block_record_matches(
    const evoke_posting_block_record *left,
    const evoke_segment_query_block *right
)
{
    bool impact_bounds_match;

    if (left->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
    {
        /*
         * Compact semantic records carry outward-quantized bounds for
         * pruning. The decoded block recomputes exact extrema, so equality
         * is neither required nor generally possible.
         */
        impact_bounds_match =
            left->min_impact <= right->record.min_impact &&
            left->max_impact >= right->record.max_impact;
    }
    else
    {
        impact_bounds_match =
            left->min_impact == right->record.min_impact &&
            left->max_impact == right->record.max_impact;
    }
    return left->posting_offset == right->source_posting_offset &&
        left->posting_count == right->record.posting_count &&
        left->block_id == right->record.block_id &&
        left->first_document_id == right->record.first_document_id &&
        left->last_document_id == right->record.last_document_id &&
        left->min_term_frequency == right->record.min_term_frequency &&
        left->max_term_frequency == right->record.max_term_frequency &&
        impact_bounds_match &&
        left->kind == right->record.kind;
}

static bool
evoke_page_query_block_requires_document_scores(
    const evoke_page_query_cleanup *cleanup,
    uint32 block_id
)
{
    for (size_t group_index = 0;
         group_index < cleanup->block_cursor_group_count;
         group_index++)
    {
        uint32 cursor_index = cleanup->block_cursor_group[group_index];
        const evoke_page_query_block_cursor *cursor =
            &cleanup->block_cursors[cursor_index];

        if (!cursor->exhausted && cursor->record.block_id == block_id &&
            cursor->record.kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
        {
            return true;
        }
    }
    return false;
}

static bool
evoke_page_query_score_may_enter_topk(
    const evoke_topk_accumulator *accumulator,
    float score
)
{
    if (accumulator == NULL || accumulator->capacity == 0 ||
        accumulator->finalized)
    {
        return false;
    }
    if (accumulator->len < accumulator->capacity)
    {
        return true;
    }

    /* Equal scores still need the persisted tie-break key. */
    return score >= accumulator->heap[0].score;
}

static evoke_status
evoke_page_query_score_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    uint32 block_id,
    double average_document_length,
    evoke_page_query_stats *stats
)
{
    float scores[EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS] = {0};
    bool touched[EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS] = {0};
    bool document_block_loaded = false;
    bool requires_document_scores;
    uint64 first_document_slot;
    uint64 remaining;
    uint32 document_count;

    CHECK_FOR_INTERRUPTS();
    first_document_slot = (uint64) block_id <<
        context->query_contract.block_shift;
    if (first_document_slot >= context->manifest.document_slot_count ||
        first_document_slot > UINT32_MAX)
    {
        return EVOKE_ERR_FORMAT;
    }
    remaining = context->manifest.document_slot_count -
        first_document_slot;
    document_count = remaining > EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS
        ? EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS
        : (uint32) remaining;
    requires_document_scores =
        evoke_page_query_block_requires_document_scores(cleanup, block_id);
    if (requires_document_scores)
    {
        evoke_segment_pages_load_query_document_block(
            index_relation,
            context,
            &cleanup->document_reader,
            block_id,
            &cleanup->document_block
        );
        document_block_loaded = true;
        stats->document_block_reads++;
        stats->max_document_block_records = Max(
            stats->max_document_block_records,
            cleanup->document_block.record_count
        );
    }

    for (size_t group_index = 0;
         group_index < cleanup->block_cursor_group_count;
         group_index++)
    {
        uint32 cursor_index = cleanup->block_cursor_group[group_index];
        evoke_page_query_block_cursor *cursor =
            &cleanup->block_cursors[cursor_index];
        evoke_page_query_term *term;
        const evoke_segment_query_run *run;
        evoke_segment_query_block block;

        if (cursor->exhausted || cursor->record.block_id != block_id)
        {
            continue;
        }
        term = &cleanup->terms[cursor->term_index];
        run = &term->plan.runs[cursor->run_index];
        evoke_segment_pages_load_query_term_block_from_record(
            index_relation,
            context,
            &term->plan,
            cursor->run_index,
            cursor->block_index,
            &cursor->record,
            &block
        );
        stats->posting_block_reads++;
        if (!evoke_page_query_block_record_matches(
                &cursor->record,
                &block))
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32 posting_index = 0;
             posting_index < block.view.len;
             posting_index++)
        {
            uint32 document_slot = block.document_slots[posting_index];
            uint32 local_slot;
            const evoke_document_cow_record *record = NULL;

            if ((uint64) document_slot < first_document_slot)
            {
                return EVOKE_ERR_FORMAT;
            }
            local_slot = (uint32) (
                (uint64) document_slot - first_document_slot
            );
            if (local_slot >= document_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            stats->postings_examined++;
            if (!evoke_page_query_document_allowed(
                    cleanup->filter,
                    document_slot))
            {
                continue;
            }
            if (requires_document_scores)
            {
                record = &cleanup->document_block.records[local_slot];
                if (record->version.document_slot != document_slot)
                {
                    return EVOKE_ERR_FORMAT;
                }
            }
            touched[local_slot] = true;
            if (requires_document_scores &&
                !evoke_page_query_document_is_live(record))
            {
                continue;
            }
            if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                double tfc;

                if (term->live_document_frequency == 0)
                {
                    return EVOKE_ERR_FORMAT;
                }
                tfc = evoke_score_tfc(
                    context->query_contract.params.method,
                    (double) block.values[
                        posting_index
                    ].term_frequency,
                    (double) record->version.document_length,
                    average_document_length,
                    context->query_contract.params.k1,
                    context->query_contract.params.b,
                    context->query_contract.params.delta
                );
                scores[local_slot] += (float) (
                    (double) term->query_weight *
                    (term->idf * tfc - (double) term->nonoccurrence)
                );
            }
            else if (run->kind ==
                         EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
                     run->kind ==
                         EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
            {
                scores[local_slot] += term->query_weight *
                    block.values[posting_index].impact;
            }
            else
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    for (uint32 local_slot = 0;
         local_slot < document_count;
         local_slot++)
    {
        const evoke_document_cow_record *record;
        uint32 document_slot = (uint32) first_document_slot + local_slot;
        evoke_status status;

        if (!touched[local_slot])
        {
            continue;
        }
        stats->documents_examined++;
        if (scores[local_slot] <= 0.0f)
        {
            continue;
        }
        if (!requires_document_scores &&
            !evoke_page_query_score_may_enter_topk(
                &cleanup->accumulator,
                scores[local_slot]
            ))
        {
            continue;
        }
        if (!document_block_loaded)
        {
            evoke_segment_pages_load_query_document_block(
                index_relation,
                context,
                &cleanup->document_reader,
                block_id,
                &cleanup->document_block
            );
            document_block_loaded = true;
            stats->document_block_reads++;
            stats->max_document_block_records = Max(
                stats->max_document_block_records,
                cleanup->document_block.record_count
            );
        }
        record = &cleanup->document_block.records[local_slot];
        if (record->version.document_slot != document_slot)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (!evoke_page_query_document_is_live(record))
        {
            continue;
        }
        stats->positive_document_count++;
        status = evoke_page_query_offer_root(
            cleanup,
            &cleanup->accumulator,
            scores[local_slot],
            document_slot,
            record->version.born_sequence
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_advance_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    uint32 block_id,
    evoke_page_query_stats *stats
)
{
    for (size_t group_index = 0;
         group_index < cleanup->block_cursor_group_count;
         group_index++)
    {
        uint32 cursor_index = cleanup->block_cursor_group[group_index];
        evoke_page_query_block_cursor *cursor =
            &cleanup->block_cursors[cursor_index];
        evoke_page_query_term *term;
        evoke_status status;

        if (cursor->exhausted || cursor->record.block_id != block_id)
        {
            continue;
        }
        if (cursor->block_index == UINT32_MAX)
        {
            return EVOKE_ERR_RANGE;
        }
        cursor->previous_block_id = block_id;
        cursor->have_previous_block = true;
        cursor->block_index++;
        term = &cleanup->terms[cursor->term_index];
        status = evoke_page_query_block_cursor_load(
            index_relation,
            context,
            term,
            cursor,
            stats
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        if (!cursor->exhausted)
        {
            evoke_page_query_block_cursor_heap_push(
                cleanup,
                cursor_index
            );
        }
    }
    cleanup->block_cursor_group_count = 0;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_score_positive_blocks(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    size_t cursor_count,
    uint64 visible_document_count,
    uint64 total_document_length,
    double average_document_length,
    evoke_page_query_stats *stats
)
{
    uint32 block_id;
    bool found;
    uint64 initial_blocks_considered = stats->blocks_considered;
    uint64 initial_blocks_scored = stats->blocks_scored;
    uint64 initial_blocks_skipped = stats->blocks_skipped;
    evoke_status status;

    status = evoke_page_query_reset_block_cursors(
        index_relation,
        context,
        cleanup,
        term_count,
        cursor_count,
        stats
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    while (true)
    {
        float upper_bound;
        bool skip;

        status = evoke_page_query_prepare_next_block_group(
            cleanup,
            &block_id,
            &found
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        if (!found)
        {
            break;
        }
        CHECK_FOR_INTERRUPTS();
        stats->blocks_considered++;
        if (!evoke_page_query_block_has_allowed_document(
                cleanup->filter,
                block_id,
                context->query_contract.block_shift))
        {
            stats->blocks_skipped++;
            for (size_t group_index = 0;
                 group_index < cleanup->block_cursor_group_count;
                 group_index++)
            {
                uint32 cursor_index =
                    cleanup->block_cursor_group[group_index];
                const evoke_page_query_block_cursor *cursor =
                    &cleanup->block_cursors[cursor_index];

                stats->postings_skipped += cursor->record.posting_count;
            }
            status = evoke_page_query_advance_block(
                index_relation,
                context,
                cleanup,
                block_id,
                stats
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
            continue;
        }
        status = evoke_page_query_block_upper_bound(
            index_relation,
            context,
            cleanup,
            block_id,
            visible_document_count,
            total_document_length,
            NULL,
            &upper_bound
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        skip = cleanup->accumulator.len ==
                cleanup->accumulator.capacity &&
            upper_bound < cleanup->accumulator.heap[0].score;
        if (skip)
        {
            stats->blocks_skipped++;
            for (size_t group_index = 0;
                 group_index < cleanup->block_cursor_group_count;
                 group_index++)
            {
                uint32 cursor_index =
                    cleanup->block_cursor_group[group_index];
                const evoke_page_query_block_cursor *cursor =
                    &cleanup->block_cursors[cursor_index];

                stats->postings_skipped += cursor->record.posting_count;
            }
        }
        else
        {
            status = evoke_page_query_score_block(
                index_relation,
                context,
                cleanup,
                block_id,
                average_document_length,
                stats
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
            stats->blocks_scored++;
        }
        status = evoke_page_query_advance_block(
            index_relation,
            context,
            cleanup,
            block_id,
            stats
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    return stats->blocks_considered >= initial_blocks_considered &&
        stats->blocks_scored >= initial_blocks_scored &&
        stats->blocks_skipped >= initial_blocks_skipped &&
        stats->blocks_scored - initial_blocks_scored +
            stats->blocks_skipped - initial_blocks_skipped ==
            stats->blocks_considered - initial_blocks_considered
        ? EVOKE_OK
        : EVOKE_ERR_FORMAT;
}

static int
evoke_page_query_compare_ordered_blocks(const void *left, const void *right)
{
    const evoke_page_query_ordered_block *a = left;
    const evoke_page_query_ordered_block *b = right;

    if (a->upper_bound > b->upper_bound)
    {
        return -1;
    }
    if (a->upper_bound < b->upper_bound)
    {
        return 1;
    }
    return a->block_id < b->block_id
        ? -1
        : a->block_id > b->block_id ? 1 : 0;
}

static void
evoke_page_query_ordered_cleanup_release(
    evoke_page_query_ordered_cleanup *resources
)
{
    if (resources == NULL || !resources->active)
    {
        return;
    }
    resources->active = false;
    free(resources->block_scored);
    free(resources->blocks);
    free(resources->contributions);
    resources->block_scored = NULL;
    resources->blocks = NULL;
    resources->contributions = NULL;
}

static void
evoke_page_query_ordered_context_reset(void *arg)
{
    evoke_page_query_ordered_cleanup *resources = arg;

    evoke_page_query_ordered_cleanup_release(resources);
}

static bool
evoke_page_query_ordered_block_budget(
    const evoke_segment_query_context *context,
    const evoke_page_query_cleanup *cleanup,
    size_t term_count,
    size_t cursor_count,
    size_t *contribution_capacity_out,
    size_t *block_capacity_out,
    uint64 *bytes_out
)
{
    uint64 contribution_capacity = 0;
    uint64 block_capacity;
    uint64 block_size;
    uint64 bytes;

    if (context == NULL || cleanup == NULL ||
        contribution_capacity_out == NULL || block_capacity_out == NULL ||
        bytes_out == NULL || term_count == 0 || cursor_count == 0 ||
        cursor_count > UINT16_MAX ||
        context->query_contract.block_shift >= 63)
    {
        return false;
    }
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        const evoke_segment_query_term_plan *plan =
            &cleanup->terms[term_index].plan;

        for (uint32 run_index = 0;
             run_index < plan->run_count;
             run_index++)
        {
            if (contribution_capacity >
                UINT64_MAX - plan->runs[run_index].block_count)
            {
                return false;
            }
            contribution_capacity += plan->runs[run_index].block_count;
        }
    }
    if (contribution_capacity == 0 ||
        contribution_capacity > UINT32_MAX ||
        contribution_capacity >
            UINT64_MAX / sizeof(evoke_page_query_block_reference))
    {
        return false;
    }
    block_size = UINT64_C(1) << context->query_contract.block_shift;
    block_capacity = context->manifest.document_slot_count / block_size;
    if (context->manifest.document_slot_count % block_size != 0)
    {
        block_capacity++;
    }
    if (block_capacity == 0 || block_capacity > SIZE_MAX ||
        block_capacity >
            UINT64_MAX / sizeof(evoke_page_query_ordered_block))
    {
        return false;
    }
    bytes = contribution_capacity *
        sizeof(evoke_page_query_block_reference);
    if (bytes > UINT64_MAX -
        block_capacity * sizeof(evoke_page_query_ordered_block))
    {
        return false;
    }
    bytes += block_capacity * sizeof(evoke_page_query_ordered_block);
    if (bytes > UINT64_MAX - block_capacity * sizeof(uint8))
    {
        return false;
    }
    bytes += block_capacity * sizeof(uint8);
    if (bytes > EVOKE_PAGE_QUERY_ORDERED_BLOCK_LIMIT)
    {
        return false;
    }
    *contribution_capacity_out = (size_t) contribution_capacity;
    *block_capacity_out = (size_t) block_capacity;
    *bytes_out = bytes;
    return true;
}

static void
evoke_page_query_block_reference_set(
    evoke_page_query_block_reference *reference,
    uint16 cursor_index,
    uint32 block_index
)
{
    memcpy(reference->bytes, &cursor_index, sizeof(cursor_index));
    memcpy(
        reference->bytes + sizeof(cursor_index),
        &block_index,
        sizeof(block_index)
    );
}

static void
evoke_page_query_block_reference_get(
    const evoke_page_query_block_reference *reference,
    uint16 *cursor_index_out,
    uint32 *block_index_out
)
{
    memcpy(cursor_index_out, reference->bytes, sizeof(*cursor_index_out));
    memcpy(
        block_index_out,
        reference->bytes + sizeof(*cursor_index_out),
        sizeof(*block_index_out)
    );
}

static evoke_status
evoke_page_query_try_score_ordered_blocks(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    size_t cursor_count,
    uint64 visible_document_count,
    uint64 total_document_length,
    double average_document_length,
    evoke_page_query_stats *stats,
    bool *used_out
)
{
    evoke_page_query_ordered_cleanup *resources;
    evoke_page_query_block_reference *contributions;
    evoke_page_query_ordered_block *blocks;
    uint8 *block_scored;
    size_t contribution_capacity;
    size_t block_capacity;
    size_t contribution_count = 0;
    size_t block_count = 0;
    uint64 metadata_bytes;
    uint64 initial_blocks_considered = stats->blocks_considered;
    uint64 initial_blocks_scored = stats->blocks_scored;
    uint64 initial_blocks_skipped = stats->blocks_skipped;
    evoke_document_cow_length_extrema global_document_extrema = {0};
    bool requires_document_extrema = false;
    evoke_status status = EVOKE_OK;

    if (used_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *used_out = false;
    if (!evoke_page_query_ordered_block_budget(
            context,
            cleanup,
            term_count,
            cursor_count,
            &contribution_capacity,
            &block_capacity,
            &metadata_bytes))
    {
        return EVOKE_OK;
    }
    resources = palloc0(sizeof(*resources));
    resources->active = true;
    resources->callback.func = evoke_page_query_ordered_context_reset;
    resources->callback.arg = resources;
    MemoryContextRegisterResetCallback(
        CurrentMemoryContext,
        &resources->callback
    );
    resources->contributions = calloc(
        contribution_capacity,
        sizeof(*contributions)
    );
    resources->blocks = calloc(block_capacity, sizeof(*blocks));
    resources->block_scored = calloc(
        block_capacity,
        sizeof(*block_scored)
    );
    contributions = resources->contributions;
    blocks = resources->blocks;
    block_scored = resources->block_scored;
    if (blocks == NULL || contributions == NULL || block_scored == NULL)
    {
        /* The exact streaming path remains available under memory pressure. */
        status = EVOKE_OK;
        goto done;
    }
    if (stats->posting_block_metadata_cache_bytes <=
        UINT64_MAX - metadata_bytes)
    {
        stats->posting_block_metadata_cache_bytes += metadata_bytes;
    }
    else
    {
        stats->posting_block_metadata_cache_bytes = UINT64_MAX;
    }
    for (size_t term_index = 0;
         term_index < term_count && !requires_document_extrema;
         term_index++)
    {
        const evoke_segment_query_term_plan *plan =
            &cleanup->terms[term_index].plan;

        for (uint32 run_index = 0;
             run_index < plan->run_count;
             run_index++)
        {
            if (plan->runs[run_index].kind ==
                EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                requires_document_extrema = true;
                break;
            }
        }
    }
    if (requires_document_extrema)
    {
        evoke_segment_pages_load_document_length_extrema(
            index_relation,
            &context->root,
            &context->manifest,
            &global_document_extrema
        );
    }
    status = evoke_page_query_reset_block_cursors(
        index_relation,
        context,
        cleanup,
        term_count,
        cursor_count,
        stats
    );
    if (status != EVOKE_OK)
    {
        goto done;
    }
    while (true)
    {
        evoke_page_query_ordered_block *ordered;
        uint32 block_id;
        bool found;

        if ((block_count & EVOKE_PAGE_QUERY_INTERRUPT_MASK) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
        status = evoke_page_query_prepare_next_block_group(
            cleanup,
            &block_id,
            &found
        );
        if (status != EVOKE_OK || !found)
        {
            break;
        }
        if (block_count >= block_capacity)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        ordered = &blocks[block_count++];
        if (block_id >= block_capacity)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        ordered->block_id = block_id;
        ordered->first_contribution = (uint32) contribution_count;
        if (cleanup->block_cursor_group_count > UINT32_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto done;
        }
        ordered->contribution_count =
            (uint32) cleanup->block_cursor_group_count;
        status = evoke_page_query_block_upper_bound(
            index_relation,
            context,
            cleanup,
            block_id,
            visible_document_count,
            total_document_length,
            requires_document_extrema ? &global_document_extrema : NULL,
            &ordered->upper_bound
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
        for (size_t group_index = 0;
             group_index < cleanup->block_cursor_group_count;
             group_index++)
        {
            uint32 cursor_index =
                cleanup->block_cursor_group[group_index];
            const evoke_page_query_block_cursor *cursor =
                &cleanup->block_cursors[cursor_index];

            if (cursor_index > UINT16_MAX ||
                contribution_count >= contribution_capacity)
            {
                status = EVOKE_ERR_RANGE;
                goto done;
            }
            if (ordered->posting_count >
                UINT64_MAX - cursor->record.posting_count)
            {
                status = EVOKE_ERR_RANGE;
                goto done;
            }
            ordered->posting_count += cursor->record.posting_count;
            evoke_page_query_block_reference_set(
                &contributions[contribution_count++],
                (uint16) cursor_index,
                cursor->block_index
            );
        }
        status = evoke_page_query_advance_block(
            index_relation,
            context,
            cleanup,
            block_id,
            stats
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
    }
    if (status != EVOKE_OK || block_count == 0 ||
        contribution_count == 0 ||
        contribution_count > contribution_capacity)
    {
        status = status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        goto done;
    }
    qsort(
        blocks,
        block_count,
        sizeof(*blocks),
        evoke_page_query_compare_ordered_blocks
    );
    stats->blocks_considered += block_count;
    for (size_t order_index = 0;
         order_index < block_count;
         order_index++)
    {
        const evoke_page_query_ordered_block *ordered =
            &blocks[order_index];

        CHECK_FOR_INTERRUPTS();
        if (cleanup->accumulator.len == cleanup->accumulator.capacity &&
            ordered->upper_bound < cleanup->accumulator.heap[0].score)
        {
            break;
        }
        if (ordered->contribution_count > cursor_count ||
            ordered->first_contribution > contribution_count ||
            ordered->contribution_count >
                contribution_count - ordered->first_contribution)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        for (uint32 local_index = 0;
             local_index < ordered->contribution_count;
             local_index++)
        {
            const evoke_page_query_block_reference *reference =
                &contributions[
                    ordered->first_contribution + local_index
                ];
            evoke_page_query_block_cursor *cursor;
            const evoke_page_query_term *term;
            uint16 cursor_index;
            uint32 block_index;

            evoke_page_query_block_reference_get(
                reference,
                &cursor_index,
                &block_index
            );
            if ((size_t) cursor_index >= cursor_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto done;
            }
            cursor = &cleanup->block_cursors[cursor_index];
            term = &cleanup->terms[cursor->term_index];
            if (cursor->run_index >= term->plan.run_count ||
                block_index >=
                    term->plan.runs[cursor->run_index].block_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto done;
            }
            cursor->block_index = block_index;
            evoke_segment_pages_load_query_term_block_record(
                index_relation,
                context,
                &term->plan,
                cursor->run_index,
                block_index,
                &cursor->record
            );
            stats->posting_block_metadata_reads++;
            if (cursor->record.block_id != ordered->block_id)
            {
                status = EVOKE_ERR_FORMAT;
                goto done;
            }
            cursor->exhausted = false;
            cleanup->block_cursor_group[local_index] = cursor_index;
        }
        cleanup->block_cursor_group_count = ordered->contribution_count;
        if (cleanup->accumulator.len == cleanup->accumulator.capacity &&
            requires_document_extrema)
        {
            float refined_upper_bound;

            status = evoke_page_query_block_upper_bound(
                index_relation,
                context,
                cleanup,
                ordered->block_id,
                visible_document_count,
                total_document_length,
                NULL,
                &refined_upper_bound
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }
            if (refined_upper_bound < cleanup->accumulator.heap[0].score)
            {
                cleanup->block_cursor_group_count = 0;
                continue;
            }
        }
        status = evoke_page_query_score_block(
            index_relation,
            context,
            cleanup,
            ordered->block_id,
            average_document_length,
            stats
        );
        cleanup->block_cursor_group_count = 0;
        if (status != EVOKE_OK)
        {
            goto done;
        }
        stats->blocks_scored++;
        block_scored[order_index] = 1;
    }
    for (size_t order_index = 0;
         order_index < block_count;
         order_index++)
    {
        const evoke_page_query_ordered_block *ordered =
            &blocks[order_index];

        if (block_scored[order_index] != 0)
        {
            continue;
        }
        stats->blocks_skipped++;
        if (stats->postings_skipped <=
            UINT64_MAX - ordered->posting_count)
        {
            stats->postings_skipped += ordered->posting_count;
        }
        else
        {
            stats->postings_skipped = UINT64_MAX;
        }
    }
    if (stats->blocks_considered < initial_blocks_considered ||
        stats->blocks_scored < initial_blocks_scored ||
        stats->blocks_skipped < initial_blocks_skipped ||
        stats->blocks_scored - initial_blocks_scored +
            stats->blocks_skipped - initial_blocks_skipped !=
            stats->blocks_considered - initial_blocks_considered)
    {
        status = EVOKE_ERR_FORMAT;
        goto done;
    }
    *used_out = true;
    stats->ordered_block_query_path = true;

done:
    cleanup->block_cursor_group_count = 0;
    evoke_page_query_ordered_cleanup_release(resources);
    return status;
}

static bool
evoke_page_query_bounded_add(
    uint64 *total,
    uint64 count,
    uint64 item_size,
    uint64 limit
)
{
    uint64 bytes;

    if (total == NULL ||
        (count != 0 && item_size > UINT64_MAX / count))
    {
        return false;
    }
    bytes = count * item_size;
    if (*total > UINT64_MAX - bytes)
    {
        return false;
    }
    *total += bytes;
    return *total <= limit;
}

static int
evoke_page_query_compare_bmp_blocks(
    const void *left,
    const void *right
)
{
    const evoke_page_query_bmp_block *a = left;
    const evoke_page_query_bmp_block *b = right;

    if (a->upper_bound != b->upper_bound)
    {
        return a->upper_bound > b->upper_bound ? -1 : 1;
    }
    return a->block_id < b->block_id
        ? -1
        : a->block_id > b->block_id;
}

static void
evoke_page_query_bmp_run_caches_free(
    evoke_page_query_bmp_run_cache *caches,
    size_t cache_count
)
{
    if (caches == NULL)
    {
        return;
    }
    for (size_t cache_index = 0;
         cache_index < cache_count;
         cache_index++)
    {
        free(caches[cache_index].super_refs);
    }
    free(caches);
}

static void
evoke_page_query_bmp_cleanup_release(evoke_page_query_bmp_cleanup *resources)
{
    if (resources == NULL || !resources->active)
    {
        return;
    }
    resources->active = false;
    if (resources->pruning_initialized)
    {
        evoke_topk_accumulator_free(&resources->pruning_accumulator);
    }
    evoke_page_query_bmp_run_caches_free(
        resources->run_caches,
        resources->run_cache_count
    );
    free(resources->super_refs);
    free(resources->cached_super_refs);
    free(resources->refs);
    free(resources->values);
    free(resources->document_slots);
    free(resources->owned_document_lengths);
    free(resources->filtered_document_lengths);
    free(resources->filtered_document_block_offsets);
    free(resources->liveness);
    free(resources->ranked_flat_blocks);
    free(resources->ranked_batch_blocks);
    free(resources->ranked_superblocks);
    free(resources->batch_upper_bounds);
    free(resources->flat_upper_bounds);
    free(resources->allowed_superblock_ranges);
    free(resources->block_membership_scratch);
    free(resources->allowed_superblocks);
    free(resources->allowed_block_bits);
    free(resources->allowed_blocks);
    free(resources->selected_superblocks);
    free(resources->super_upper_bounds);
    free(resources->super_lexical_max);
    free(resources->exact_blocks);
    free(resources->lexical_max);
    free(resources->scores);
    memset(
        &resources->pruning_accumulator,
        0,
        sizeof(resources->pruning_accumulator)
    );
    resources->pruning_initialized = false;
    resources->run_caches = NULL;
    resources->run_cache_count = 0;
}

static void
evoke_page_query_bmp_context_reset(void *arg)
{
    evoke_page_query_bmp_cleanup *resources = arg;

    evoke_page_query_bmp_cleanup_release(resources);
}

static const evoke_segment_query_bmp_cached_super_ref *
evoke_page_query_bmp_run_cache_find_super_ref(
    const evoke_page_query_bmp_run_cache *cache,
    uint32 superblock_id
)
{
    uint32 low = 0;
    uint32 high;

    if (cache == NULL || cache->super_refs == NULL)
    {
        return NULL;
    }
    high = cache->super_ref_count;
    while (low < high)
    {
        uint32 middle = low + (high - low) / UINT32_C(2);

        if (cache->super_refs[middle].superblock_id < superblock_id)
        {
            low = middle + UINT32_C(1);
        }
        else
        {
            high = middle;
        }
    }
    return low < cache->super_ref_count &&
        cache->super_refs[low].superblock_id == superblock_id
        ? &cache->super_refs[low]
        : NULL;
}

static evoke_status
evoke_page_query_bmp_lower_bound_super_ref(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    uint32 target_superblock,
    evoke_page_query_stats *stats,
    uint32 *ref_index_out
)
{
    const evoke_segment_query_run *run;
    evoke_segment_query_bmp_cached_super_ref ref;
    uint32 low = 0;
    uint32 high;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        stats == NULL || ref_index_out == NULL ||
        run_index >= plan->run_count)
    {
        return EVOKE_ERR_INVALID;
    }
    run = &plan->runs[run_index];
    if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        !run->semantic_bmp.available)
    {
        return EVOKE_ERR_INVALID;
    }
    high = run->semantic_bmp.super_ref_count;
    while (low < high)
    {
        uint32 middle = low + (high - low) / UINT32_C(2);
        uint32 loaded =
            evoke_segment_pages_load_query_bmp_cached_super_ref_window(
                index_relation,
                context,
                plan,
                run_index,
                middle,
                &ref,
                UINT32_C(1)
            );

        if (loaded != UINT32_C(1))
        {
            return EVOKE_ERR_FORMAT;
        }
        stats->semantic_bmp_super_ref_reads++;
        if (ref.superblock_id < target_superblock)
        {
            low = middle + UINT32_C(1);
        }
        else
        {
            high = middle;
        }
    }
    *ref_index_out = low;
    return EVOKE_OK;
}

static bool
evoke_page_query_bmp_prefer_sequential_super_refs(
    uint32 super_ref_count,
    size_t allowed_superblock_count,
    size_t allowed_superblock_range_count
)
{
    uint64 binary_search_span = (uint64) super_ref_count + UINT64_C(1);
    uint64 binary_search_reads = 0;
    uint64 estimated_selective_reads;

    if (super_ref_count == 0 || allowed_superblock_range_count == 0)
    {
        return false;
    }
    while (binary_search_span > UINT64_C(1))
    {
        binary_search_span =
            (binary_search_span + UINT64_C(1)) >> UINT64_C(1);
        binary_search_reads++;
    }
    if (allowed_superblock_range_count >
        (UINT64_MAX - Min(
            (uint64) allowed_superblock_count,
            (uint64) super_ref_count
        )) / binary_search_reads)
    {
        return true;
    }
    estimated_selective_reads = Min(
        (uint64) allowed_superblock_count,
        (uint64) super_ref_count
    ) + (uint64) allowed_superblock_range_count * binary_search_reads;
    return estimated_selective_reads >= (uint64) super_ref_count;
}

static evoke_status
evoke_page_query_bmp_cache_filtered_super_refs_sequential(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_segment_query_term_plan *plan,
    uint32 run_index,
    const uint8 *allowed_superblocks,
    evoke_segment_query_bmp_cached_super_ref *window,
    evoke_page_query_bmp_run_cache *run_cache,
    evoke_page_query_stats *stats
)
{
    const evoke_segment_query_run *run;
    uint32 retained_ref_count = 0;
    uint32 retained_ref_capacity;

    if (index_relation == NULL || context == NULL || plan == NULL ||
        allowed_superblocks == NULL || window == NULL ||
        run_cache == NULL || stats == NULL ||
        run_index >= plan->run_count)
    {
        return EVOKE_ERR_INVALID;
    }
    run = &plan->runs[run_index];
    if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        !run->semantic_bmp.available)
    {
        return EVOKE_ERR_INVALID;
    }
    retained_ref_capacity = run_cache->super_ref_count;
    for (uint32 first_ref = 0;
         first_ref < run->semantic_bmp.super_ref_count;)
    {
        uint32 loaded =
            evoke_segment_pages_load_query_bmp_cached_super_ref_window(
                index_relation,
                context,
                plan,
                run_index,
                first_ref,
                window,
                Min(
                    EVOKE_PAGE_QUERY_BMP_SUPER_REF_WINDOW,
                    run->semantic_bmp.super_ref_count - first_ref
                )
            );

        if (loaded == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        stats->semantic_bmp_super_ref_reads += loaded;
        for (uint32 ref_index = 0;
             ref_index < loaded;
             ref_index++)
        {
            const evoke_segment_query_bmp_cached_super_ref *ref =
                &window[ref_index];

            if (allowed_superblocks[ref->superblock_id] == 0)
            {
                continue;
            }
            if (retained_ref_count >= retained_ref_capacity)
            {
                return EVOKE_ERR_RANGE;
            }
            run_cache->super_refs[retained_ref_count++] = *ref;
        }
        first_ref += loaded;
        CHECK_FOR_INTERRUPTS();
    }
    run_cache->super_ref_count = retained_ref_count;
    stats->semantic_bmp_filtered_sequential_scans++;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_bmp_score_block_offset(
    const uint32 *filtered_block_offsets,
    size_t filtered_block_count,
    uint32 block_id,
    size_t *score_offset_out
)
{
    size_t compact_block;

    if (score_offset_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    compact_block = filtered_block_offsets == NULL
        ? block_id
        : filtered_block_offsets[block_id];
    if ((filtered_block_offsets != NULL &&
         compact_block >= filtered_block_count) ||
        compact_block > SIZE_MAX /
            EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS)
    {
        return EVOKE_ERR_FORMAT;
    }
    *score_offset_out = compact_block *
        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_score_semantic_bmp_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_cleanup *cleanup,
    const evoke_page_query_bmp_run_cache *run_caches,
    size_t term_count,
    uint64 document_slot_count,
    uint32 block_id,
    const uint32 *filtered_block_offsets,
    size_t filtered_block_count,
    float *scores,
    evoke_page_query_stats *stats
)
{
    uint32 first_document = block_id <<
        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
    size_t score_block_offset;
    evoke_status status;

    status = evoke_page_query_bmp_score_block_offset(
        filtered_block_offsets,
        filtered_block_count,
        block_id,
        &score_block_offset
    );
    if (status != EVOKE_OK)
    {
        return status;
    }

    for (size_t term_index = 0; term_index < term_count; term_index++)
    {
        const evoke_page_query_term *term = &cleanup->terms[term_index];

        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            const evoke_segment_query_run *run =
                &term->plan.runs[run_index];
            const evoke_page_query_bmp_run_cache *run_cache;
            const evoke_segment_query_bmp_cached_super_ref *super_ref;
            evoke_semantic_bmp_record record;
            float impacts[EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS];
            uint64 mask;
            uint16 impact_index = 0;

            if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
            {
                continue;
            }
            run_cache = &run_caches[
                term_index * EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS + run_index
            ];
            if (run_cache->super_refs != NULL)
            {
                super_ref = evoke_page_query_bmp_run_cache_find_super_ref(
                    run_cache,
                    block_id >> UINT32_C(4)
                );
                if (super_ref == NULL)
                {
                    continue;
                }
                if (!evoke_segment_pages_load_query_bmp_cached_record(
                        index_relation,
                        context,
                        &term->plan,
                        run_index,
                        super_ref,
                        block_id,
                        &record,
                        impacts))
                {
                    continue;
                }
            }
            else if (!evoke_segment_pages_load_query_bmp_record(
                    index_relation,
                    context,
                    &term->plan,
                    run_index,
                    block_id,
                    &record,
                    impacts))
            {
                continue;
            }
            stats->semantic_bmp_record_reads++;
            mask = record.document_mask;
            for (uint32 local_document = 0;
                 mask != 0;
                 local_document++, mask >>= 1)
            {
                uint32 document_slot;

                if ((mask & UINT64_C(1)) == 0)
                {
                    continue;
                }
                document_slot = first_document + local_document;
                if (document_slot >= document_slot_count)
                {
                    return EVOKE_ERR_FORMAT;
                }
                if (evoke_page_query_test_omit_semantic_impact(
                        run,
                        impacts[impact_index],
                        stats))
                {
                    impact_index++;
                    continue;
                }
                if (!evoke_page_query_document_allowed(
                        cleanup->filter,
                        document_slot))
                {
                    impact_index++;
                    continue;
                }
                scores[score_block_offset + local_document] +=
                    term->query_weight * impacts[impact_index++];
                if (!isfinite(scores[score_block_offset + local_document]))
                {
                    return EVOKE_ERR_RANGE;
                }
                stats->semantic_bmp_postings_examined++;
            }
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_accumulate_bmp_fine_bounds(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_term *term,
    uint32 run_index,
    uint32 superblock_id,
    uint32 first_ref,
    uint16 ref_count,
    const int32 *selected_superblocks,
    float *batch_upper_bounds,
    evoke_semantic_bmp_ref *refs,
    evoke_page_query_stats *stats
)
{
    const evoke_segment_query_run *run = &term->plan.runs[run_index];
    int32 batch_index = selected_superblocks[superblock_id];
    uint32 first_child_ref;
    uint32 loaded_ref_count;

    if (batch_index < 0)
    {
        return EVOKE_OK;
    }
    if (first_ref < run->semantic_bmp.first_ref)
    {
        return EVOKE_ERR_FORMAT;
    }
    first_child_ref = first_ref - run->semantic_bmp.first_ref;
    loaded_ref_count = evoke_segment_pages_load_query_bmp_ref_window(
        index_relation,
        context,
        &term->plan,
        run_index,
        first_child_ref,
        refs,
        ref_count
    );
    if (loaded_ref_count != ref_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    stats->semantic_bmp_ref_reads += loaded_ref_count;
    for (uint32 ref_index = 0;
         ref_index < loaded_ref_count;
         ref_index++)
    {
        const evoke_semantic_bmp_ref *ref = &refs[ref_index];
        uint32 expected_first_block = superblock_id << UINT32_C(4);
        uint32 child;
        size_t bound_index;
        float contribution;
        double next;

        if (ref->block_id < expected_first_block ||
            ref->block_id >= expected_first_block + UINT32_C(16))
        {
            return EVOKE_ERR_FORMAT;
        }
        child = ref->block_id - expected_first_block;
        bound_index = (size_t) batch_index * UINT32_C(16) + child;
        contribution = evoke_semantic_bmp_contribution_bound(
            term->query_weight,
            ref->min_impact,
            ref->max_impact
        );
        next = nextafter(
            (double) batch_upper_bounds[bound_index] + contribution,
            INFINITY
        );
        batch_upper_bounds[bound_index] = next > FLT_MAX
            ? INFINITY
            : nextafterf((float) next, INFINITY);
    }
    return EVOKE_OK;
}

#define EVOKE_PAGE_QUERY_LIVENESS_UNKNOWN UINT8_C(0)
#define EVOKE_PAGE_QUERY_LIVENESS_LIVE UINT8_C(1)
#define EVOKE_PAGE_QUERY_LIVENESS_DEAD UINT8_C(2)

static evoke_status
evoke_page_query_load_bmp_liveness(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    uint32 bmp_block_id,
    uint8 *liveness,
    evoke_page_query_stats *stats
)
{
    uint32 first_document = bmp_block_id <<
        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
    uint32 document_block_id = first_document >>
        context->query_contract.block_shift;

    if (liveness[first_document] != EVOKE_PAGE_QUERY_LIVENESS_UNKNOWN)
    {
        return EVOKE_OK;
    }
    evoke_segment_pages_load_query_document_block(
        index_relation,
        context,
        &cleanup->document_reader,
        document_block_id,
        &cleanup->document_block
    );
    stats->document_block_reads++;
    stats->max_document_block_records = Max(
        stats->max_document_block_records,
        cleanup->document_block.record_count
    );
    for (uint32 local_slot = 0;
         local_slot < cleanup->document_block.record_count;
         local_slot++)
    {
        const evoke_document_cow_record *document =
            &cleanup->document_block.records[local_slot];
        uint32 document_slot =
            cleanup->document_block.first_document_slot + local_slot;

        if (document_slot >= context->manifest.document_slot_count ||
            document->version.document_slot != document_slot)
        {
            ereport(
                ERROR,
                (
                    errmsg("invalid evoke BMP liveness document block"),
                    errdetail(
                        "bmp_block=%u document_block=%u first=%u "
                        "local=%u slot=%u record_slot=%llu slots=%llu.",
                        bmp_block_id,
                        document_block_id,
                        cleanup->document_block.first_document_slot,
                        local_slot,
                        document_slot,
                        (unsigned long long)
                            document->version.document_slot,
                        (unsigned long long)
                            context->manifest.document_slot_count
                    )
                )
            );
        }
        liveness[document_slot] =
            evoke_page_query_document_is_live(document)
            ? EVOKE_PAGE_QUERY_LIVENESS_LIVE
            : EVOKE_PAGE_QUERY_LIVENESS_DEAD;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_try_score_semantic_bmp(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    double average_document_length,
    size_t k,
    const evoke_topk_result *seed,
    evoke_page_query_stats *stats,
    bool *used_out
)
{
    uint64 document_slot_count = context->manifest.document_slot_count;
    uint64 block_count64;
    uint64 superblock_count64;
    uint64 memory_bytes;
    float *scores = NULL;
    float *lexical_max = NULL;
    float *super_lexical_max = NULL;
    float *super_upper_bounds = NULL;
    float *batch_upper_bounds = NULL;
    float *flat_upper_bounds = NULL;
    int32 *selected_superblocks = NULL;
    uint8 *allowed_blocks = NULL;
    uint8 *allowed_block_bits = NULL;
    uint8 *allowed_superblocks = NULL;
    uint8 *block_membership_scratch = NULL;
    bool *exact_blocks = NULL;
    uint32 *document_slots = NULL;
    evoke_posting_value *values = NULL;
    uint32 *owned_document_lengths = NULL;
    uint32 *filtered_document_block_offsets = NULL;
    uint32 *filtered_document_lengths = NULL;
    const uint32 *document_lengths = NULL;
    uint8 *liveness = NULL;
    evoke_semantic_bmp_ref *refs = NULL;
    evoke_semantic_bmp_super_ref *super_refs = NULL;
    evoke_segment_query_bmp_cached_super_ref *cached_super_refs = NULL;
    evoke_page_query_bmp_run_cache *run_caches = NULL;
    evoke_page_query_bmp_superblock_range *allowed_superblock_ranges = NULL;
    evoke_page_query_bmp_block *ranked_superblocks = NULL;
    evoke_page_query_bmp_block *ranked_batch_blocks = NULL;
    evoke_page_query_bmp_block *ranked_flat_blocks = NULL;
    evoke_topk_accumulator pruning_accumulator;
    evoke_page_query_bmp_cleanup *resources = NULL;
    size_t ranked_superblock_count = 0;
    size_t exact_block_count = 0;
    size_t allowed_block_count = 0;
    size_t filtered_document_block_count = 0;
    size_t allowed_superblock_count = 0;
    size_t allowed_superblock_range_count = 0;
    size_t run_cache_count;
    size_t semantic_run_count = 0;
    uint64 semantic_posting_count = 0;
    uint64 semantic_super_ref_count = 0;
    uint64 score_slot_count;
    bool has_semantic = false;
    bool needs_document_lengths = false;
    bool needs_liveness = false;
    bool needs_flat_fallback = false;
    bool cache_super_refs = false;
    bool direct_flat_filtered = false;
    bool lexical_document_block_loaded = false;
    bool materialize_document_lengths = false;
    bool materialize_filtered_document_lengths = false;
    bool restrict_to_allowed_blocks = false;
    bool use_compact_filtered_scores = false;
    uint32 super_batch = (uint32) Min(
        Max(evoke_test_query_semantic_bmp_super_batch, 1),
        (int) EVOKE_PAGE_QUERY_BMP_SUPER_BATCH
    );
    evoke_status status = EVOKE_OK;

    if (cleanup == NULL || used_out == NULL || stats == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    restrict_to_allowed_blocks = cleanup->filter != NULL &&
        cleanup->filter->allowed_document_bitmap != NULL;
    *used_out = false;
    memset(&pruning_accumulator, 0, sizeof(pruning_accumulator));
    if (evoke_test_disable_semantic_bmp ||
        document_slot_count == 0 || document_slot_count > UINT32_MAX)
    {
        return EVOKE_OK;
    }
    needs_liveness = context->manifest.visible_document_count !=
        document_slot_count;
    block_count64 = (document_slot_count + 63) >>
        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
    superblock_count64 = (document_slot_count + 1023) >>
        EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT;
    if (term_count > SIZE_MAX / EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS)
    {
        return EVOKE_ERR_RANGE;
    }
    run_cache_count = term_count * EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS;
    memory_bytes = 0;
    if (block_count64 == 0 || block_count64 > UINT32_MAX ||
        superblock_count64 == 0 || superblock_count64 > UINT32_MAX ||
        !evoke_page_query_bounded_add(
            &memory_bytes,
            block_count64,
            sizeof(float) + sizeof(bool),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        ) ||
        !evoke_page_query_bounded_add(
            &memory_bytes,
            superblock_count64,
            sizeof(float) * 2 + sizeof(int32) +
                sizeof(evoke_page_query_bmp_block),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        ) ||
        !evoke_page_query_bounded_add(
            &memory_bytes,
            EVOKE_PAGE_QUERY_BMP_SUPER_BATCH * UINT32_C(16),
            sizeof(float) + sizeof(evoke_page_query_bmp_block),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        ) ||
        !evoke_page_query_bounded_add(
            &memory_bytes,
            EVOKE_SEGMENT_QUERY_POSTING_WINDOW,
            sizeof(uint32) + sizeof(evoke_posting_value),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        ) ||
        !evoke_page_query_bounded_add(
            &memory_bytes,
            EVOKE_PAGE_QUERY_BMP_REF_WINDOW,
            sizeof(evoke_semantic_bmp_ref),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        ) ||
        !evoke_page_query_bounded_add(
            &memory_bytes,
            EVOKE_PAGE_QUERY_BMP_SUPER_REF_WINDOW,
            sizeof(evoke_semantic_bmp_super_ref),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        ) ||
        !evoke_page_query_bounded_add(
            &memory_bytes,
            EVOKE_PAGE_QUERY_BMP_SUPER_REF_WINDOW,
            sizeof(evoke_segment_query_bmp_cached_super_ref),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        ) ||
        !evoke_page_query_bounded_add(
            &memory_bytes,
            run_cache_count,
            sizeof(evoke_page_query_bmp_run_cache),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        ) ||
        (restrict_to_allowed_blocks &&
         (!evoke_page_query_bounded_add(
              &memory_bytes,
              block_count64,
              sizeof(*allowed_blocks),
              EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
          ) ||
          !evoke_page_query_bounded_add(
              &memory_bytes,
              (block_count64 + UINT64_C(7)) >> 3,
              sizeof(*allowed_block_bits),
              EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
          ) ||
          !evoke_page_query_bounded_add(
              &memory_bytes,
              (block_count64 + UINT64_C(7)) >> 3,
              sizeof(*block_membership_scratch),
              EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
          ) ||
          !evoke_page_query_bounded_add(
              &memory_bytes,
              superblock_count64,
              sizeof(*allowed_superblocks),
              EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
          ))) ||
        (needs_liveness && !evoke_page_query_bounded_add(
            &memory_bytes,
            document_slot_count,
            sizeof(uint8),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        )))
    {
        return EVOKE_OK;
    }
    for (size_t term_index = 0; term_index < term_count; term_index++)
    {
        const evoke_page_query_term *term = &cleanup->terms[term_index];

        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            const evoke_segment_query_run *run =
                &term->plan.runs[run_index];

            if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                needs_document_lengths = true;
            }
            if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
            {
                continue;
            }
            has_semantic = true;
            semantic_run_count++;
            if (!run->semantic_bmp.available ||
                run->semantic_bmp.document_count > document_slot_count ||
                run->semantic_bmp.block_count > block_count64 ||
                run->semantic_bmp.superblock_count > superblock_count64 ||
                (run->semantic_bmp.block_membership_bytes != 0 &&
                 run->semantic_bmp.block_membership_bytes !=
                    ((run->semantic_bmp.block_count + UINT32_C(7)) >> 3)))
            {
                return EVOKE_OK;
            }
            if (semantic_posting_count >
                UINT64_MAX - run->posting_count)
            {
                return EVOKE_ERR_RANGE;
            }
            semantic_posting_count += run->posting_count;
            if (semantic_super_ref_count > UINT64_MAX -
                run->semantic_bmp.super_ref_count)
            {
                return EVOKE_ERR_RANGE;
            }
            semantic_super_ref_count +=
                run->semantic_bmp.super_ref_count;
        }
    }
    if (!has_semantic)
    {
        return EVOKE_OK;
    }
    stats->semantic_bmp_query_super_ref_count = semantic_super_ref_count;
    /*
     * Dense term-by-superblock metadata predicts the quarter-corpus fallback
     * that makes the BMP preflight slower than the exact packed stream.  This
     * skips only the strategy, so exactness is unchanged.  Tests may force the
     * BMP path to continue validating its representation.
     */
    if (!evoke_test_force_semantic_bmp &&
        !restrict_to_allowed_blocks &&
        semantic_super_ref_count >
            superblock_count64 *
                EVOKE_PAGE_QUERY_BMP_MAX_SUPER_REF_DENSITY)
    {
        stats->semantic_bmp_admission_skipped = true;
        return EVOKE_OK;
    }
    if (needs_document_lengths &&
        (context->resident_document_lengths == NULL ||
         context->resident_document_length_count != document_slot_count))
    {
        uint64 materialized_memory_bytes = memory_bytes;

        materialize_document_lengths = evoke_page_query_bounded_add(
            &materialized_memory_bytes,
            document_slot_count,
            sizeof(uint32),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        );
        if (materialize_document_lengths)
        {
            memory_bytes = materialized_memory_bytes;
        }
    }
    if (restrict_to_allowed_blocks)
    {
        uint64 bitmap_byte_count =
            (Min(
                cleanup->filter->document_slot_count,
                document_slot_count
            ) + UINT64_C(7)) >> 3;
        uint32 previous_block = UINT32_MAX;
        uint32 previous_superblock = UINT32_MAX;

        for (uint64 byte_index = 0;
             byte_index < bitmap_byte_count;
             byte_index++)
        {
            uint8 candidates =
                cleanup->filter->allowed_document_bitmap[byte_index];

            while (candidates != 0)
            {
                unsigned bit_index = (unsigned) __builtin_ctz(
                    (unsigned) candidates
                );
                uint64 document_slot =
                    byte_index * UINT64_C(8) + bit_index;
                uint32 block_id;
                uint32 superblock_id;

                candidates &= (uint8) (candidates - UINT8_C(1));
                if (document_slot >= document_slot_count)
                {
                    continue;
                }
                block_id = (uint32) (
                    document_slot >>
                        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT
                );
                superblock_id = (uint32) (
                    document_slot >>
                        EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT
                );
                if (block_id != previous_block)
                {
                    allowed_block_count++;
                    previous_block = block_id;
                }
                if (superblock_id != previous_superblock)
                {
                    allowed_superblock_count++;
                    previous_superblock = superblock_id;
                }
            }
        }
        if (allowed_block_count == 0 || allowed_superblock_count == 0)
        {
            return EVOKE_OK;
        }
        stats->filtered_bmp_allowed_blocks = allowed_block_count;
        stats->filtered_bmp_allowed_superblocks = allowed_superblock_count;
        use_compact_filtered_scores =
            (uint64) allowed_block_count *
                EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS *
                sizeof(*scores) +
                block_count64 *
                    sizeof(*filtered_document_block_offsets) <
            document_slot_count * sizeof(*scores);
        if (use_compact_filtered_scores &&
            !evoke_page_query_bounded_add(
                &memory_bytes,
                block_count64,
                sizeof(*filtered_document_block_offsets),
                EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT))
        {
            return EVOKE_OK;
        }
        if (!evoke_page_query_bounded_add(
                &memory_bytes,
                allowed_superblock_count,
                sizeof(*allowed_superblock_ranges),
                EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT))
        {
            return EVOKE_OK;
        }
        if (needs_document_lengths &&
            !materialize_document_lengths &&
            (context->resident_document_lengths == NULL ||
             context->resident_document_length_count !=
                document_slot_count))
        {
            uint64 filtered_length_memory_bytes = memory_bytes;

            materialize_filtered_document_lengths =
                (use_compact_filtered_scores ||
                 evoke_page_query_bounded_add(
                    &filtered_length_memory_bytes,
                    block_count64,
                    sizeof(*filtered_document_block_offsets),
                    EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT)) &&
                evoke_page_query_bounded_add(
                    &filtered_length_memory_bytes,
                    (uint64) allowed_block_count *
                        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS,
                    sizeof(*filtered_document_lengths),
                    EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
                );
            if (materialize_filtered_document_lengths)
            {
                memory_bytes = filtered_length_memory_bytes;
            }
        }
    }
    score_slot_count = use_compact_filtered_scores
        ? (uint64) allowed_block_count *
            EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS
        : document_slot_count;
    if (!evoke_page_query_bounded_add(
            &memory_bytes,
            score_slot_count,
            sizeof(*scores),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT))
    {
        return EVOKE_OK;
    }
    {
        uint64 cached_memory_bytes = memory_bytes;
        uint64 cache_ref_count = semantic_super_ref_count;

        if (restrict_to_allowed_blocks)
        {
            uint64 selective_capacity;

            if (semantic_run_count >
                UINT64_MAX / allowed_superblock_count)
            {
                return EVOKE_ERR_RANGE;
            }
            selective_capacity =
                semantic_run_count * allowed_superblock_count;
            cache_ref_count = Min(cache_ref_count, selective_capacity);
        }

        cache_super_refs = evoke_page_query_bounded_add(
            &cached_memory_bytes,
            cache_ref_count,
            sizeof(evoke_segment_query_bmp_cached_super_ref),
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
        );
        if (cache_super_refs)
        {
            memory_bytes = cached_memory_bytes;
        }
    }
    /*
     * A scattered filter can touch nearly every 1024-document superblock while
     * retaining only a small fraction of the 64-document blocks.  The
     * hierarchical path spends more work materializing coarse bounds than
     * it saves.  Build fine-block bounds once instead; this preserves the same
     * sign-aware upper bound and exact scoring contract without repeated
     * metadata work.  Unfiltered queries retain hierarchical pruning.
     */
    direct_flat_filtered = restrict_to_allowed_blocks;
    /*
     * The flat route skips coarse bound accumulation, not record addressing.
     * Retain the bounded selected-superblock cache so exact record reads do
     * not repeat page-native super-ref searches for every scored block.
     */
    stats->semantic_bmp_direct_flat_filtered = direct_flat_filtered;
    stats->semantic_bmp_query_bytes = memory_bytes;

    resources = palloc0(sizeof(*resources));
    resources->active = true;
    resources->run_cache_count = run_cache_count;
    resources->callback.func = evoke_page_query_bmp_context_reset;
    resources->callback.arg = resources;
    MemoryContextRegisterResetCallback(
        CurrentMemoryContext,
        &resources->callback
    );
    scores = calloc((size_t) score_slot_count, sizeof(*scores));
    lexical_max = calloc((size_t) block_count64, sizeof(*lexical_max));
    exact_blocks = calloc((size_t) block_count64, sizeof(*exact_blocks));
    super_lexical_max = calloc(
        (size_t) superblock_count64,
        sizeof(*super_lexical_max)
    );
    super_upper_bounds = calloc(
        (size_t) superblock_count64,
        sizeof(*super_upper_bounds)
    );
    selected_superblocks = malloc(
        (size_t) superblock_count64 * sizeof(*selected_superblocks)
    );
    batch_upper_bounds = malloc(
        EVOKE_PAGE_QUERY_BMP_SUPER_BATCH * UINT32_C(16) *
            sizeof(*batch_upper_bounds)
    );
    ranked_superblocks = malloc(
        (size_t) superblock_count64 * sizeof(*ranked_superblocks)
    );
    ranked_batch_blocks = malloc(
        EVOKE_PAGE_QUERY_BMP_SUPER_BATCH * UINT32_C(16) *
            sizeof(*ranked_batch_blocks)
    );
    document_slots = malloc(
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW * sizeof(*document_slots)
    );
    values = malloc(
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW * sizeof(*values)
    );
    refs = malloc(EVOKE_PAGE_QUERY_BMP_REF_WINDOW * sizeof(*refs));
    super_refs = malloc(
        EVOKE_PAGE_QUERY_BMP_SUPER_REF_WINDOW * sizeof(*super_refs)
    );
    cached_super_refs = malloc(
        EVOKE_PAGE_QUERY_BMP_SUPER_REF_WINDOW *
            sizeof(*cached_super_refs)
    );
    run_caches = calloc(run_cache_count, sizeof(*run_caches));
    if (restrict_to_allowed_blocks)
    {
        allowed_blocks = calloc(
            (size_t) block_count64,
            sizeof(*allowed_blocks)
        );
        allowed_block_bits = calloc(
            (size_t) ((block_count64 + UINT64_C(7)) >> 3),
            sizeof(*allowed_block_bits)
        );
        block_membership_scratch = malloc(
            (size_t) ((block_count64 + UINT64_C(7)) >> 3)
        );
        allowed_superblocks = calloc(
            (size_t) superblock_count64,
            sizeof(*allowed_superblocks)
        );
        allowed_superblock_ranges = malloc(
            allowed_superblock_count *
                sizeof(*allowed_superblock_ranges)
        );
        if (use_compact_filtered_scores ||
            materialize_filtered_document_lengths)
        {
            filtered_document_block_offsets = malloc(
                (size_t) block_count64 *
                    sizeof(*filtered_document_block_offsets)
            );
        }
        if (materialize_filtered_document_lengths)
        {
            filtered_document_lengths = calloc(
                allowed_block_count *
                    EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS,
                sizeof(*filtered_document_lengths)
            );
        }
    }
    if (needs_liveness)
    {
        liveness = calloc((size_t) document_slot_count, sizeof(*liveness));
    }
    resources->scores = scores;
    resources->lexical_max = lexical_max;
    resources->exact_blocks = exact_blocks;
    resources->super_lexical_max = super_lexical_max;
    resources->super_upper_bounds = super_upper_bounds;
    resources->selected_superblocks = selected_superblocks;
    resources->batch_upper_bounds = batch_upper_bounds;
    resources->ranked_superblocks = ranked_superblocks;
    resources->ranked_batch_blocks = ranked_batch_blocks;
    resources->document_slots = document_slots;
    resources->values = values;
    resources->refs = refs;
    resources->super_refs = super_refs;
    resources->cached_super_refs = cached_super_refs;
    resources->run_caches = run_caches;
    resources->allowed_blocks = allowed_blocks;
    resources->allowed_block_bits = allowed_block_bits;
    resources->block_membership_scratch = block_membership_scratch;
    resources->allowed_superblocks = allowed_superblocks;
    resources->allowed_superblock_ranges = allowed_superblock_ranges;
    resources->filtered_document_block_offsets =
        filtered_document_block_offsets;
    resources->filtered_document_lengths = filtered_document_lengths;
    resources->liveness = liveness;
    if (scores == NULL || lexical_max == NULL || exact_blocks == NULL ||
        super_lexical_max == NULL || super_upper_bounds == NULL ||
        selected_superblocks == NULL || batch_upper_bounds == NULL ||
        ranked_superblocks == NULL || ranked_batch_blocks == NULL ||
        document_slots == NULL || values == NULL ||
        refs == NULL || super_refs == NULL || cached_super_refs == NULL ||
        run_caches == NULL ||
        (restrict_to_allowed_blocks &&
         (allowed_blocks == NULL || allowed_block_bits == NULL ||
          block_membership_scratch == NULL ||
          allowed_superblocks == NULL ||
          allowed_superblock_ranges == NULL ||
          ((use_compact_filtered_scores ||
            materialize_filtered_document_lengths) &&
           filtered_document_block_offsets == NULL) ||
          (materialize_filtered_document_lengths &&
           filtered_document_lengths == NULL))) ||
        (needs_liveness && liveness == NULL))
    {
        status = EVOKE_ERR_NOMEM;
        goto done;
    }
    stats->semantic_bmp_attempted = true;
    if (restrict_to_allowed_blocks)
    {
        uint64 bitmap_byte_count =
            (Min(
                cleanup->filter->document_slot_count,
                document_slot_count
            ) + UINT64_C(7)) >> 3;

        if (filtered_document_block_offsets != NULL)
        {
            memset(
                filtered_document_block_offsets,
                0xff,
                (size_t) block_count64 *
                    sizeof(*filtered_document_block_offsets)
            );
        }

        for (uint64 byte_index = 0;
             byte_index < bitmap_byte_count;
             byte_index++)
        {
            uint8 candidates =
                cleanup->filter->allowed_document_bitmap[byte_index];

            while (candidates != 0)
            {
                unsigned bit_index = (unsigned) __builtin_ctz(
                    (unsigned) candidates
                );
                uint64 document_slot =
                    byte_index * UINT64_C(8) + bit_index;
                uint32 block_id = (uint32) (
                    document_slot >>
                        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT
                );
                uint32 superblock_id = (uint32) (
                    document_slot >>
                        EVOKE_SEMANTIC_BMP_PACKED_SUPERBLOCK_SHIFT
                );

                candidates &= (uint8) (candidates - UINT8_C(1));
                if (document_slot >= document_slot_count)
                {
                    continue;
                }
                if (allowed_blocks[block_id] == 0)
                {
                    allowed_blocks[block_id] = UINT8_C(1);
                    allowed_block_bits[block_id >> 3U] |=
                        (uint8) (UINT8_C(1) <<
                            (block_id & UINT32_C(7)));
                    if (filtered_document_block_offsets != NULL)
                    {
                        filtered_document_block_offsets[block_id] =
                            (uint32) filtered_document_block_count++;
                    }
                }
                allowed_superblocks[superblock_id] = UINT8_C(1);
            }
        }
        if (allowed_block_count == 0)
        {
            goto done;
        }
        if (filtered_document_block_offsets != NULL &&
            filtered_document_block_count != allowed_block_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        for (uint32 superblock_id = 0;
             superblock_id < (uint32) superblock_count64;)
        {
            uint32 first_superblock;

            if (allowed_superblocks[superblock_id] == 0)
            {
                superblock_id++;
                continue;
            }
            first_superblock = superblock_id;
            do
            {
                superblock_id++;
            }
            while (superblock_id < (uint32) superblock_count64 &&
                   allowed_superblocks[superblock_id] != 0);
            if (allowed_superblock_range_count >=
                allowed_superblock_count)
            {
                status = EVOKE_ERR_RANGE;
                goto done;
            }
            allowed_superblock_ranges[
                allowed_superblock_range_count
            ].first_superblock = first_superblock;
            allowed_superblock_ranges[
                allowed_superblock_range_count
            ].superblock_limit = superblock_id;
            allowed_superblock_range_count++;
        }
        if (allowed_superblock_range_count == 0)
        {
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
    }
    memset(
        selected_superblocks,
        0xff,
        (size_t) superblock_count64 * sizeof(*selected_superblocks)
    );
    for (size_t term_index = 0; term_index < term_count; term_index++)
    {
        const evoke_page_query_term *term = &cleanup->terms[term_index];

        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            const evoke_segment_query_run *run =
                &term->plan.runs[run_index];
            evoke_page_query_bmp_run_cache *run_cache =
                &run_caches[
                    term_index * EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS +
                        run_index
                ];

            if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
            {
                continue;
            }
            if (!cache_super_refs)
            {
                continue;
            }
            run_cache->super_ref_count = restrict_to_allowed_blocks
                ? Min(
                    run->semantic_bmp.super_ref_count,
                    (uint32) allowed_superblock_count
                )
                : run->semantic_bmp.super_ref_count;
            run_cache->super_refs = malloc(
                (size_t) run_cache->super_ref_count *
                    sizeof(*run_cache->super_refs)
            );
            if (run_cache->super_ref_count != 0 &&
                run_cache->super_refs == NULL)
            {
                status = EVOKE_ERR_NOMEM;
                goto done;
            }
            if (restrict_to_allowed_blocks)
            {
                uint32 retained_ref_count = 0;

                if (evoke_page_query_bmp_prefer_sequential_super_refs(
                        run->semantic_bmp.super_ref_count,
                        allowed_superblock_count,
                        allowed_superblock_range_count))
                {
                    status =
                        evoke_page_query_bmp_cache_filtered_super_refs_sequential(
                            index_relation,
                            context,
                            &term->plan,
                            run_index,
                            allowed_superblocks,
                            cached_super_refs,
                            run_cache,
                            stats
                        );
                    if (status != EVOKE_OK)
                    {
                        goto done;
                    }
                    goto validate_cached_super_refs;
                }

                for (size_t range_index = 0;
                     range_index < allowed_superblock_range_count;
                     range_index++)
                {
                    const evoke_page_query_bmp_superblock_range *range =
                        &allowed_superblock_ranges[range_index];
                    uint32 first_ref;

                    status = evoke_page_query_bmp_lower_bound_super_ref(
                        index_relation,
                        context,
                        &term->plan,
                        run_index,
                        range->first_superblock,
                        stats,
                        &first_ref
                    );
                    if (status != EVOKE_OK)
                    {
                        goto done;
                    }
                    while (first_ref < run->semantic_bmp.super_ref_count)
                    {
                        uint32 range_width =
                            range->superblock_limit -
                            range->first_superblock;
                        uint32 loaded =
                            evoke_segment_pages_load_query_bmp_cached_super_ref_window(
                                index_relation,
                                context,
                                &term->plan,
                                run_index,
                                first_ref,
                                cached_super_refs,
                                Min(
                                    EVOKE_PAGE_QUERY_BMP_SUPER_REF_WINDOW,
                                    Min(
                                        range_width,
                                        run->semantic_bmp.super_ref_count -
                                            first_ref
                                    )
                                )
                            );
                        bool range_complete = false;

                        if (loaded == 0)
                        {
                            status = EVOKE_ERR_FORMAT;
                            goto done;
                        }
                        stats->semantic_bmp_super_ref_reads += loaded;
                        for (uint32 ref_index = 0;
                             ref_index < loaded;
                             ref_index++)
                        {
                            const evoke_segment_query_bmp_cached_super_ref
                                *ref = &cached_super_refs[ref_index];

                            if (ref->superblock_id >=
                                range->superblock_limit)
                            {
                                range_complete = true;
                                break;
                            }
                            if (ref->superblock_id <
                                range->first_superblock)
                            {
                                continue;
                            }
                            if (retained_ref_count >=
                                run_cache->super_ref_count)
                            {
                                status = EVOKE_ERR_RANGE;
                                goto done;
                            }
                            run_cache->super_refs[
                                retained_ref_count++
                            ] = *ref;
                        }
                        if (range_complete)
                        {
                            break;
                        }
                        first_ref += loaded;
                    }
                }
                run_cache->super_ref_count = retained_ref_count;
            }
            else
            {
                for (uint32 first_ref = 0;
                     first_ref < run_cache->super_ref_count;)
                {
                    uint32 loaded =
                        evoke_segment_pages_load_query_bmp_cached_super_ref_window(
                            index_relation,
                            context,
                            &term->plan,
                            run_index,
                            first_ref,
                            run_cache->super_refs + first_ref,
                            Min(
                                EVOKE_PAGE_QUERY_BMP_SUPER_REF_WINDOW,
                                run_cache->super_ref_count - first_ref
                            )
                        );

                    if (loaded == 0)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto done;
                    }
                    first_ref += loaded;
                    stats->semantic_bmp_super_ref_reads += loaded;
                }
            }
validate_cached_super_refs:
            for (uint32 ref_index = 1;
                 ref_index < run_cache->super_ref_count;
                 ref_index++)
            {
                if (run_cache->super_refs[ref_index - 1].superblock_id >=
                    run_cache->super_refs[ref_index].superblock_id)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto done;
                }
            }
        }
    }
    if (needs_document_lengths)
    {
        if (context->resident_document_lengths != NULL &&
            context->resident_document_length_count == document_slot_count)
        {
            document_lengths = context->resident_document_lengths;
        }
        else if (materialize_document_lengths)
        {
            owned_document_lengths = calloc(
                (size_t) document_slot_count *
                    sizeof(*owned_document_lengths),
                1
            );
            resources->owned_document_lengths = owned_document_lengths;
            if (owned_document_lengths == NULL)
            {
                status = EVOKE_ERR_NOMEM;
                goto done;
            }
            if (restrict_to_allowed_blocks)
            {
                uint64 bitmap_byte_count =
                    (Min(
                        cleanup->filter->document_slot_count,
                        document_slot_count
                    ) + UINT64_C(7)) >> 3;

                for (uint64 byte_index = 0;
                     byte_index < bitmap_byte_count;
                     byte_index++)
                {
                    uint8 candidates = cleanup->filter->
                        allowed_document_bitmap[byte_index];

                    while (candidates != 0)
                    {
                        const evoke_document_cow_record *document = NULL;
                        unsigned bit_index = (unsigned) __builtin_ctz(
                            (unsigned) candidates
                        );
                        uint64 document_slot_u64 =
                            byte_index * UINT64_C(8) + bit_index;
                        uint32 document_slot;

                        candidates &= (uint8) (
                            candidates - UINT8_C(1)
                        );
                        if (document_slot_u64 >= document_slot_count)
                        {
                            continue;
                        }
                        document_slot = (uint32) document_slot_u64;
                        status = evoke_page_query_load_document(
                            index_relation,
                            context,
                            document_slot,
                            cleanup,
                            &lexical_document_block_loaded,
                            &document,
                            stats
                        );
                        if (status != EVOKE_OK)
                        {
                            goto done;
                        }
                        if (!evoke_page_query_document_is_live(document))
                        {
                            continue;
                        }
                        owned_document_lengths[document_slot] =
                            document->version.document_length;
                    }
                    if ((byte_index & UINT64_C(4095)) == 0)
                    {
                        CHECK_FOR_INTERRUPTS();
                    }
                }
            }
            else if (!evoke_segment_pages_load_query_document_lengths(
                         index_relation,
                         context,
                         owned_document_lengths))
            {
                evoke_page_query_document_length_build length_build;

                memset(&length_build, 0, sizeof(length_build));
                length_build.document_lengths = owned_document_lengths;
                length_build.document_slot_count = document_slot_count;
                length_build.valid = true;
                evoke_segment_pages_visit_query_document_records(
                    index_relation,
                    context,
                    &cleanup->document_reader,
                    evoke_page_query_collect_document_length,
                    &length_build
                );
                if (!length_build.valid ||
                    length_build.next_document_slot != document_slot_count)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto done;
                }
            }
            document_lengths = owned_document_lengths;
        }
        else if (materialize_filtered_document_lengths)
        {
            uint64 bitmap_byte_count =
                (Min(
                    cleanup->filter->document_slot_count,
                    document_slot_count
                ) + UINT64_C(7)) >> 3;

            for (uint64 byte_index = 0;
                 byte_index < bitmap_byte_count;
                 byte_index++)
            {
                uint8 candidates = cleanup->filter->
                    allowed_document_bitmap[byte_index];

                while (candidates != 0)
                {
                    const evoke_document_cow_record *document = NULL;
                    unsigned bit_index = (unsigned) __builtin_ctz(
                        (unsigned) candidates
                    );
                    uint64 document_slot_u64 =
                        byte_index * UINT64_C(8) + bit_index;
                    uint32 document_slot;
                    uint32 block_id;
                    uint32 compact_block;

                    candidates &= (uint8) (candidates - UINT8_C(1));
                    if (document_slot_u64 >= document_slot_count)
                    {
                        continue;
                    }
                    document_slot = (uint32) document_slot_u64;
                    block_id = document_slot >>
                        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
                    compact_block =
                        filtered_document_block_offsets[block_id];
                    if (compact_block >= filtered_document_block_count)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto done;
                    }
                    status = evoke_page_query_load_document(
                        index_relation,
                        context,
                        document_slot,
                        cleanup,
                        &lexical_document_block_loaded,
                        &document,
                        stats
                    );
                    if (status != EVOKE_OK)
                    {
                        goto done;
                    }
                    if (!evoke_page_query_document_is_live(document))
                    {
                        continue;
                    }
                    filtered_document_lengths[
                        (size_t) compact_block *
                            EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS +
                            (document_slot &
                                (EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS -
                                 UINT32_C(1)))
                    ] = document->version.document_length;
                }
                if ((byte_index & UINT64_C(4095)) == 0)
                {
                    CHECK_FOR_INTERRUPTS();
                }
            }
        }
    }

    for (size_t term_index = 0; term_index < term_count; term_index++)
    {
        const evoke_page_query_term *term = &cleanup->terms[term_index];

        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            const evoke_segment_query_run *run =
                &term->plan.runs[run_index];

            if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                term->live_document_frequency == 0)
            {
                continue;
            }
            if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
            {
                const evoke_page_query_bmp_run_cache *run_cache =
                    &run_caches[
                        term_index * EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS +
                            run_index
                    ];

                if (direct_flat_filtered)
                {
                    continue;
                }
                if (run_cache->super_refs != NULL)
                {
                    for (uint32 ref_index = 0;
                         ref_index < run_cache->super_ref_count;
                         ref_index++)
                    {
                        const evoke_segment_query_bmp_cached_super_ref *ref =
                            &run_cache->super_refs[ref_index];
                        float contribution =
                            evoke_semantic_bmp_contribution_bound(
                                term->query_weight,
                                ref->min_impact,
                                ref->max_impact
                            );
                        double next = nextafter(
                            (double) super_upper_bounds[ref->superblock_id] +
                                contribution,
                            INFINITY
                        );

                        if (allowed_superblocks != NULL &&
                            allowed_superblocks[ref->superblock_id] == 0)
                        {
                            continue;
                        }

                        super_upper_bounds[ref->superblock_id] =
                            next > FLT_MAX
                            ? INFINITY
                            : nextafterf((float) next, INFINITY);
                    }
                }
                else
                {
                    uint32 first_ref = 0;

                    while (first_ref < run->semantic_bmp.super_ref_count)
                    {
                        uint32 ref_count =
                            evoke_segment_pages_load_query_bmp_super_ref_window(
                                index_relation,
                                context,
                                &term->plan,
                                run_index,
                                first_ref,
                                super_refs,
                                EVOKE_PAGE_QUERY_BMP_SUPER_REF_WINDOW
                            );

                        if (ref_count == 0)
                        {
                            status = EVOKE_ERR_FORMAT;
                            goto done;
                        }

                        for (uint32 ref_index = 0;
                             ref_index < ref_count;
                             ref_index++)
                        {
                            const evoke_semantic_bmp_super_ref *ref =
                                &super_refs[ref_index];
                            float contribution =
                                evoke_semantic_bmp_contribution_bound(
                                    term->query_weight,
                                    ref->min_impact,
                                    ref->max_impact
                                );
                            double next = nextafter(
                                (double) super_upper_bounds[
                                    ref->superblock_id
                                ] + contribution,
                                INFINITY
                            );

                            if (allowed_superblocks != NULL &&
                                allowed_superblocks[
                                    ref->superblock_id
                                ] == 0)
                            {
                                continue;
                            }

                            super_upper_bounds[ref->superblock_id] =
                                next > FLT_MAX
                                ? INFINITY
                                : nextafterf((float) next, INFINITY);
                        }
                        first_ref += ref_count;
                        stats->semantic_bmp_super_ref_reads += ref_count;
                    }
                }
                continue;
            }

            for (uint64 first_posting = 0;
                 first_posting < run->posting_count;)
            {
                uint32 posting_count =
                    evoke_segment_pages_load_query_term_posting_window(
                        index_relation,
                        context,
                        &term->plan,
                        run_index,
                        first_posting,
                        document_slots,
                        values,
                        EVOKE_SEGMENT_QUERY_POSTING_WINDOW
                    );

                for (uint32 posting_index = 0;
                     posting_index < posting_count;
                     posting_index++)
                {
                    uint32 document_slot = document_slots[posting_index];
                    uint32 block_id = document_slot >>
                        EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
                    size_t score_block_offset;
                    size_t score_index;
                    float contribution;

                    stats->postings_examined++;
                    if (allowed_blocks != NULL &&
                        allowed_blocks[block_id] == 0)
                    {
                        continue;
                    }
                    if (!evoke_page_query_document_allowed(
                            cleanup->filter,
                            document_slot))
                    {
                        continue;
                    }
                    status = evoke_page_query_bmp_score_block_offset(
                        use_compact_filtered_scores
                            ? filtered_document_block_offsets
                            : NULL,
                        use_compact_filtered_scores
                            ? filtered_document_block_count
                            : 0,
                        block_id,
                        &score_block_offset
                    );
                    if (status != EVOKE_OK)
                    {
                        goto done;
                    }
                    score_index = score_block_offset +
                        (document_slot &
                            (EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS -
                             UINT32_C(1)));

                    if (run->kind ==
                            EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                    {
                        const evoke_document_cow_record *document = NULL;
                        uint32 document_length;
                        double tfc;

                        if (document_lengths != NULL)
                        {
                            document_length =
                                document_lengths[document_slot];
                        }
                        else if (filtered_document_lengths != NULL)
                        {
                            uint32 compact_block =
                                filtered_document_block_offsets[block_id];

                            if (compact_block >=
                                filtered_document_block_count)
                            {
                                status = EVOKE_ERR_FORMAT;
                                goto done;
                            }
                            document_length = filtered_document_lengths[
                                (size_t) compact_block *
                                    EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS +
                                    (document_slot &
                                        (EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS -
                                         UINT32_C(1)))
                            ];
                        }
                        else
                        {
                            status = evoke_page_query_load_document(
                                index_relation,
                                context,
                                document_slot,
                                cleanup,
                                &lexical_document_block_loaded,
                                &document,
                                stats
                            );
                            if (status != EVOKE_OK)
                            {
                                goto done;
                            }
                            document_length =
                                document->version.document_length;
                        }
                        tfc = evoke_score_tfc(
                            context->query_contract.params.method,
                            (double) values[posting_index].term_frequency,
                            (double) document_length,
                            average_document_length,
                            context->query_contract.params.k1,
                            context->query_contract.params.b,
                            context->query_contract.params.delta
                        );
                        contribution = (float) (
                            (double) term->query_weight *
                            (term->idf * tfc -
                             (double) term->nonoccurrence)
                        );
                    }
                    else if (run->kind ==
                                 EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
                    {
                        contribution = term->query_weight *
                            values[posting_index].impact;
                    }
                    else
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto done;
                    }
                    scores[score_index] += contribution;
                    if (!isfinite(scores[score_index]))
                    {
                        status = EVOKE_ERR_RANGE;
                        goto done;
                    }
                    if (lexical_max[block_id] < scores[score_index])
                    {
                        lexical_max[block_id] = scores[score_index];
                    }
                }
                first_posting += posting_count;
            }
        }
    }
    for (uint32 block_id = 0;
         block_id < (uint32) block_count64;
         block_id++)
    {
        uint32 superblock_id = block_id >> UINT32_C(4);

        if (allowed_blocks != NULL && allowed_blocks[block_id] == 0)
        {
            continue;
        }
        if (super_lexical_max[superblock_id] < lexical_max[block_id])
        {
            super_lexical_max[superblock_id] = lexical_max[block_id];
        }
    }
    for (uint32 superblock_id = 0;
         superblock_id < (uint32) superblock_count64;
         superblock_id++)
    {
        double combined = nextafter(
            (double) super_lexical_max[superblock_id] +
                super_upper_bounds[superblock_id],
            INFINITY
        );

        if (allowed_superblocks != NULL &&
            allowed_superblocks[superblock_id] == 0)
        {
            continue;
        }
        if (combined <= 0.0)
        {
            continue;
        }
        ranked_superblocks[ranked_superblock_count].block_id =
            superblock_id;
        ranked_superblocks[ranked_superblock_count].upper_bound =
            combined > FLT_MAX
                ? INFINITY
                : nextafterf((float) combined, INFINITY);
        ranked_superblock_count++;
    }
    qsort(
        ranked_superblocks,
        ranked_superblock_count,
        sizeof(*ranked_superblocks),
        evoke_page_query_compare_bmp_blocks
    );
    needs_flat_fallback = direct_flat_filtered;
    status = evoke_topk_accumulator_init(&pruning_accumulator, k);
    if (status != EVOKE_OK)
    {
        goto done;
    }
    resources->pruning_accumulator = pruning_accumulator;
    resources->pruning_initialized = true;
    if (seed != NULL)
    {
        for (size_t result_index = 0;
             result_index < seed->len;
             result_index++)
        {
            status = evoke_page_query_offer_root(
                cleanup,
                &pruning_accumulator,
                seed->scores[result_index],
                seed->doc_ids[result_index],
                seed->doc_ids[result_index]
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }
        }
    }

    for (size_t batch_start = 0;
         !direct_flat_filtered && batch_start < ranked_superblock_count;)
    {
        size_t batch_count = Min(
            (size_t) super_batch,
            ranked_superblock_count - batch_start
        );
        size_t ranked_batch_block_count = 0;

        CHECK_FOR_INTERRUPTS();
        if (pruning_accumulator.len == pruning_accumulator.capacity &&
            ranked_superblocks[batch_start].upper_bound <
                pruning_accumulator.heap[0].score)
        {
            break;
        }
        memset(
            selected_superblocks,
            0xff,
            (size_t) superblock_count64 *
                sizeof(*selected_superblocks)
        );
        memset(
            batch_upper_bounds,
            0,
            batch_count * UINT32_C(16) *
                sizeof(*batch_upper_bounds)
        );
        for (size_t batch_index = 0;
             batch_index < batch_count;
             batch_index++)
        {
            uint32 superblock_id =
                ranked_superblocks[batch_start + batch_index].block_id;
            uint32 first_block = superblock_id << UINT32_C(4);

            selected_superblocks[superblock_id] = (int32) batch_index;
            for (uint32 child = 0; child < UINT32_C(16); child++)
            {
                uint32 block_id = first_block + child;

                if (block_id < block_count64 &&
                    (allowed_blocks == NULL ||
                     allowed_blocks[block_id] != 0))
                {
                    batch_upper_bounds[
                        batch_index * UINT32_C(16) + child
                    ] = lexical_max[block_id];
                }
            }
        }
        for (size_t term_index = 0; term_index < term_count; term_index++)
        {
            const evoke_page_query_term *term =
                &cleanup->terms[term_index];

            for (uint32 run_index = 0;
                 run_index < term->plan.run_count;
                 run_index++)
            {
                const evoke_segment_query_run *run =
                    &term->plan.runs[run_index];
                evoke_page_query_bmp_run_cache *run_cache =
                    &run_caches[
                        term_index * EVOKE_SEGMENT_QUERY_TERM_MAX_RUNS +
                            run_index
                    ];

                if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
                {
                    continue;
                }
                if (run_cache->super_refs != NULL)
                {
                    for (size_t batch_index = 0;
                         batch_index < batch_count;
                         batch_index++)
                    {
                        const evoke_segment_query_bmp_cached_super_ref
                            *super_ref;
                        uint32 superblock_id = ranked_superblocks[
                            batch_start + batch_index
                        ].block_id;

                        /*
                         * The cache is sorted by superblock.  Scanning every
                         * cached ref for every ranked batch makes fragmented
                         * filters quadratic in the number of allowed blocks.
                         */
                        super_ref =
                            evoke_page_query_bmp_run_cache_find_super_ref(
                                run_cache,
                                superblock_id
                            );
                        if (super_ref == NULL)
                        {
                            continue;
                        }

                        status = evoke_page_query_accumulate_bmp_fine_bounds(
                            index_relation,
                            context,
                            term,
                            run_index,
                            super_ref->superblock_id,
                            super_ref->first_ref,
                            super_ref->ref_count,
                            selected_superblocks,
                            batch_upper_bounds,
                            refs,
                            stats
                        );
                        if (status != EVOKE_OK)
                        {
                            goto done;
                        }
                    }
                }
                else
                {
                    uint32 first_super_ref = 0;

                    while (first_super_ref <
                           run->semantic_bmp.super_ref_count)
                    {
                        uint32 super_ref_count =
                            evoke_segment_pages_load_query_bmp_super_ref_window(
                                index_relation,
                                context,
                                &term->plan,
                                run_index,
                                first_super_ref,
                                super_refs,
                                EVOKE_PAGE_QUERY_BMP_SUPER_REF_WINDOW
                            );

                        if (super_ref_count == 0)
                        {
                            status = EVOKE_ERR_FORMAT;
                            goto done;
                        }

                        stats->semantic_bmp_super_ref_reads +=
                            super_ref_count;
                        for (uint32 super_ref_index = 0;
                             super_ref_index < super_ref_count;
                             super_ref_index++)
                        {
                            const evoke_semantic_bmp_super_ref *super_ref =
                                &super_refs[super_ref_index];

                            status =
                                evoke_page_query_accumulate_bmp_fine_bounds(
                                    index_relation,
                                    context,
                                    term,
                                    run_index,
                                    super_ref->superblock_id,
                                    super_ref->first_ref,
                                    super_ref->ref_count,
                                    selected_superblocks,
                                    batch_upper_bounds,
                                    refs,
                                    stats
                                );
                            if (status != EVOKE_OK)
                            {
                                goto done;
                            }
                        }
                        first_super_ref += super_ref_count;
                    }
                }
            }
        }
        for (size_t batch_index = 0;
             batch_index < batch_count;
             batch_index++)
        {
            uint32 superblock_id =
                ranked_superblocks[batch_start + batch_index].block_id;
            uint32 first_block = superblock_id << UINT32_C(4);

            for (uint32 child = 0; child < UINT32_C(16); child++)
            {
                uint32 block_id = first_block + child;
                float upper_bound = batch_upper_bounds[
                    batch_index * UINT32_C(16) + child
                ];

                if (block_id < block_count64 && upper_bound > 0.0f &&
                    (allowed_blocks == NULL ||
                     allowed_blocks[block_id] != 0))
                {
                    ranked_batch_blocks[
                        ranked_batch_block_count
                    ].block_id = block_id;
                    ranked_batch_blocks[
                        ranked_batch_block_count
                    ].upper_bound = upper_bound;
                    ranked_batch_block_count++;
                }
            }
        }
        qsort(
            ranked_batch_blocks,
            ranked_batch_block_count,
            sizeof(*ranked_batch_blocks),
            evoke_page_query_compare_bmp_blocks
        );
        for (size_t ranked_index = 0;
             ranked_index < ranked_batch_block_count;
             ranked_index++)
        {
            const evoke_page_query_bmp_block *ranked =
                &ranked_batch_blocks[ranked_index];
            uint32 first_document = ranked->block_id <<
                EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
            uint32 document_limit = Min(
                first_document +
                    EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS,
                (uint32) document_slot_count
            );
            size_t score_block_offset;

            CHECK_FOR_INTERRUPTS();
            stats->blocks_considered++;
            if (pruning_accumulator.len ==
                    pruning_accumulator.capacity &&
                ranked->upper_bound < pruning_accumulator.heap[0].score)
            {
                stats->blocks_skipped +=
                    ranked_batch_block_count - ranked_index;
                break;
            }
            status = evoke_page_query_bmp_score_block_offset(
                use_compact_filtered_scores
                    ? filtered_document_block_offsets
                    : NULL,
                use_compact_filtered_scores
                    ? filtered_document_block_count
                    : 0,
                ranked->block_id,
                &score_block_offset
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }
            status = evoke_page_query_score_semantic_bmp_block(
                index_relation,
                context,
                cleanup,
                run_caches,
                term_count,
                document_slot_count,
                ranked->block_id,
                use_compact_filtered_scores
                    ? filtered_document_block_offsets
                    : NULL,
                use_compact_filtered_scores
                    ? filtered_document_block_count
                    : 0,
                scores,
                stats
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }
            if (liveness != NULL)
            {
                status = evoke_page_query_load_bmp_liveness(
                    index_relation,
                    context,
                    cleanup,
                    ranked->block_id,
                    liveness,
                    stats
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
            }
            exact_blocks[ranked->block_id] = true;
            exact_block_count++;
            for (uint32 document_slot = first_document;
                 document_slot < document_limit;
                 document_slot++)
            {
                float score = scores[
                    score_block_offset +
                        (document_slot &
                            (EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS -
                             UINT32_C(1)))
                ];

                if (score > 0.0f &&
                    (liveness == NULL ||
                     liveness[document_slot] ==
                         EVOKE_PAGE_QUERY_LIVENESS_LIVE))
                {
                    status = evoke_page_query_offer_root(
                        cleanup,
                        &pruning_accumulator,
                        score,
                        document_slot,
                        document_slot
                    );
                    if (status != EVOKE_OK)
                    {
                        goto done;
                    }
                }
            }
            stats->blocks_scored++;
        }
        batch_start += batch_count;
        if (batch_start < ranked_superblock_count &&
            pruning_accumulator.len == pruning_accumulator.capacity &&
            ranked_superblocks[batch_start].upper_bound <
                pruning_accumulator.heap[0].score)
        {
            break;
        }
        if (!restrict_to_allowed_blocks &&
            batch_start < ranked_superblock_count &&
            batch_start >=
                (size_t) super_batch *
                    EVOKE_PAGE_QUERY_BMP_MAX_UNPRUNED_BATCHES)
        {
            needs_flat_fallback = true;
            break;
        }
    }

    if (needs_flat_fallback)
    {
        size_t ranked_flat_count = 0;
        size_t competitive_flat_count = 0;
        uint64 fallback_memory_bytes = memory_bytes;

        if (!evoke_page_query_bounded_add(
                &fallback_memory_bytes,
                block_count64,
                sizeof(*flat_upper_bounds),
                EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT))
        {
            stats->semantic_bmp_fallback = true;
            status = EVOKE_OK;
            goto done;
        }
        flat_upper_bounds = calloc(
            (size_t) block_count64,
            sizeof(*flat_upper_bounds)
        );
        resources->flat_upper_bounds = flat_upper_bounds;
        if (flat_upper_bounds == NULL)
        {
            status = EVOKE_ERR_NOMEM;
            goto done;
        }

        for (size_t term_index = 0;
             term_index < term_count;
             term_index++)
        {
            const evoke_page_query_term *term =
                &cleanup->terms[term_index];

            for (uint32 run_index = 0;
                 run_index < term->plan.run_count;
                 run_index++)
            {
                const evoke_segment_query_run *run =
                    &term->plan.runs[run_index];
                uint32 first_ref = 0;
                uint32 previous_matching_superblock = UINT32_MAX;

                if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
                {
                    continue;
                }
                if (allowed_block_bits != NULL &&
                    run->semantic_bmp.block_membership_bytes != 0)
                {
                    uint32 relative_ref = 0;
                    uint32 packed_window_first_ref = 0;
                    uint32 packed_window_ref_count = 0;
                    uint8 *packed_ref_bytes = (uint8 *) refs;
                    uint32 membership_bytes =
                        evoke_segment_pages_load_query_bmp_block_membership(
                            index_relation,
                            context,
                            &term->plan,
                            run_index,
                            block_membership_scratch,
                            (uint32) ((block_count64 + UINT64_C(7)) >> 3)
                        );

                    for (uint32 byte_index = 0;
                         byte_index < membership_bytes;
                         byte_index++)
                    {
                        uint8 present = block_membership_scratch[byte_index];
                        uint8 matching = present &
                            allowed_block_bits[byte_index];

                        if ((byte_index & UINT32_C(4095)) == 0)
                        {
                            CHECK_FOR_INTERRUPTS();
                        }
                        while (matching != 0)
                        {
                            unsigned bit_index = (unsigned) __builtin_ctz(
                                (unsigned) matching
                            );
                            uint8 lower_mask = (uint8) (
                                (UINT32_C(1) << bit_index) - UINT32_C(1)
                            );
                            uint32 ref_rank = relative_ref +
                                (uint32) __builtin_popcount(
                                    (unsigned) (present & lower_mask)
                                );
                            uint32 block_id = byte_index * UINT32_C(8) +
                                bit_index;
                            uint32 superblock_id = block_id >> UINT32_C(4);
                            evoke_semantic_bmp_packed_ref packed_ref;
                            float contribution;
                            double next;

                            matching &= (uint8) (
                                matching - UINT8_C(1)
                            );
                            if (packed_window_ref_count == 0 ||
                                ref_rank < packed_window_first_ref ||
                                ref_rank - packed_window_first_ref >=
                                    packed_window_ref_count)
                            {
                                packed_window_first_ref = ref_rank;
                                packed_window_ref_count =
                                    evoke_segment_pages_load_query_bmp_packed_ref_page(
                                        index_relation,
                                        context,
                                        &term->plan,
                                        run_index,
                                        ref_rank,
                                        packed_ref_bytes,
                                        EVOKE_PAGE_QUERY_BMP_REF_WINDOW
                                    );
                                if (packed_window_ref_count == 0)
                                {
                                    status = EVOKE_ERR_FORMAT;
                                    goto done;
                                }
                                stats->posting_block_metadata_reads++;
                                stats->semantic_bmp_ref_reads +=
                                    packed_window_ref_count;
                            }
                            status = evoke_semantic_bmp_packed_ref_decode(
                                packed_ref_bytes +
                                    (Size) (
                                        ref_rank - packed_window_first_ref
                                    ) * EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
                                EVOKE_SEMANTIC_BMP_PACKED_REF_SIZE,
                                run->semantic_bmp.min_impact,
                                run->semantic_bmp.max_impact,
                                &packed_ref
                            );
                            if (status != EVOKE_OK ||
                                packed_ref.local_block_id !=
                                    (block_id & UINT32_C(15)))
                            {
                                status = status == EVOKE_OK
                                    ? EVOKE_ERR_FORMAT
                                    : status;
                                goto done;
                            }
                            contribution =
                                evoke_semantic_bmp_contribution_bound(
                                    term->query_weight,
                                    packed_ref.min_impact,
                                    packed_ref.max_impact
                                );
                            next = nextafter(
                                (double) flat_upper_bounds[block_id] +
                                    contribution,
                                INFINITY
                            );
                            flat_upper_bounds[block_id] = next > FLT_MAX
                                ? INFINITY
                                : nextafterf((float) next, INFINITY);
                            stats->filtered_bmp_matching_ref_count++;
                            if (superblock_id !=
                                previous_matching_superblock)
                            {
                                stats->
                                    filtered_bmp_matching_super_ref_count++;
                                previous_matching_superblock = superblock_id;
                            }
                        }
                        relative_ref += (uint32) __builtin_popcount(
                            (unsigned) present
                        );
                    }
                    if (relative_ref != run->semantic_bmp.ref_count)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto done;
                    }
                    continue;
                }
                while (first_ref < run->semantic_bmp.ref_count)
                {
                    uint32 ref_count =
                        evoke_segment_pages_load_query_bmp_ref_window(
                            index_relation,
                            context,
                            &term->plan,
                            run_index,
                            first_ref,
                            refs,
                            EVOKE_PAGE_QUERY_BMP_REF_WINDOW
                        );

                    if (ref_count == 0)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto done;
                    }

                    stats->semantic_bmp_ref_reads += ref_count;
                    for (uint32 ref_index = 0;
                         ref_index < ref_count;
                         ref_index++)
                    {
                        const evoke_semantic_bmp_ref *ref =
                            &refs[ref_index];
                        float contribution =
                            evoke_semantic_bmp_contribution_bound(
                                term->query_weight,
                                ref->min_impact,
                                ref->max_impact
                            );
                        double next = nextafter(
                            (double) flat_upper_bounds[ref->block_id] +
                                contribution,
                            INFINITY
                        );

                        if (allowed_blocks != NULL &&
                            allowed_blocks[ref->block_id] == 0)
                        {
                            continue;
                        }
                        if (allowed_blocks != NULL)
                        {
                            uint32 superblock_id = ref->block_id >>
                                UINT32_C(4);

                            stats->filtered_bmp_matching_ref_count++;
                            if (superblock_id !=
                                previous_matching_superblock)
                            {
                                stats->
                                    filtered_bmp_matching_super_ref_count++;
                                previous_matching_superblock = superblock_id;
                            }
                        }

                        flat_upper_bounds[ref->block_id] = next > FLT_MAX
                            ? INFINITY
                            : nextafterf((float) next, INFINITY);
                    }
                    first_ref += ref_count;
                }
            }
        }
        for (uint32 block_id = 0;
             block_id < (uint32) block_count64;
             block_id++)
        {
            double combined;

            if (exact_blocks[block_id] ||
                (allowed_blocks != NULL && allowed_blocks[block_id] == 0))
            {
                continue;
            }
            combined = nextafter(
                (double) lexical_max[block_id] +
                    flat_upper_bounds[block_id],
                INFINITY
            );
            if (combined <= 0.0)
            {
                continue;
            }
            ranked_flat_count++;
        }
        if (!evoke_page_query_bounded_add(
                &fallback_memory_bytes,
                ranked_flat_count,
                sizeof(*ranked_flat_blocks),
                EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT))
        {
            stats->semantic_bmp_fallback = true;
            status = EVOKE_OK;
            goto done;
        }
        if (ranked_flat_count > 0)
        {
            ranked_flat_blocks = malloc(
                ranked_flat_count * sizeof(*ranked_flat_blocks)
            );
            resources->ranked_flat_blocks = ranked_flat_blocks;
            if (ranked_flat_blocks == NULL)
            {
                status = EVOKE_ERR_NOMEM;
                goto done;
            }
        }
        stats->semantic_bmp_query_bytes = fallback_memory_bytes;
        ranked_flat_count = 0;
        for (uint32 block_id = 0;
             block_id < (uint32) block_count64;
             block_id++)
        {
            double combined;

            if (exact_blocks[block_id] ||
                (allowed_blocks != NULL && allowed_blocks[block_id] == 0))
            {
                continue;
            }
            combined = nextafter(
                (double) lexical_max[block_id] +
                    flat_upper_bounds[block_id],
                INFINITY
            );
            if (combined <= 0.0)
            {
                continue;
            }
            ranked_flat_blocks[ranked_flat_count].block_id = block_id;
            ranked_flat_blocks[ranked_flat_count].upper_bound =
                combined > FLT_MAX
                    ? INFINITY
                    : nextafterf((float) combined, INFINITY);
            ranked_flat_count++;
        }
        qsort(
            ranked_flat_blocks,
            ranked_flat_count,
            sizeof(*ranked_flat_blocks),
            evoke_page_query_compare_bmp_blocks
        );
        if (pruning_accumulator.len == pruning_accumulator.capacity)
        {
            while (competitive_flat_count < ranked_flat_count &&
                   ranked_flat_blocks[
                       competitive_flat_count
                   ].upper_bound >= pruning_accumulator.heap[0].score)
            {
                competitive_flat_count++;
            }
        }
        else
        {
            competitive_flat_count = ranked_flat_count;
        }
        if (allowed_blocks == NULL &&
            (exact_block_count > block_count64 / UINT32_C(4) ||
             competitive_flat_count >
                 block_count64 / UINT32_C(4) - exact_block_count))
        {
            stats->semantic_bmp_fallback = true;
            status = EVOKE_OK;
            goto done;
        }
        for (size_t ranked_index = 0;
             ranked_index < ranked_flat_count;
             ranked_index++)
        {
            const evoke_page_query_bmp_block *ranked =
                &ranked_flat_blocks[ranked_index];
            uint32 first_document = ranked->block_id <<
                EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
            uint32 document_limit = Min(
                first_document +
                    EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS,
                (uint32) document_slot_count
            );
            size_t score_block_offset;

            CHECK_FOR_INTERRUPTS();
            stats->blocks_considered++;
            if (pruning_accumulator.len ==
                    pruning_accumulator.capacity &&
                ranked->upper_bound < pruning_accumulator.heap[0].score)
            {
                stats->blocks_skipped +=
                    ranked_flat_count - ranked_index;
                break;
            }
            status = evoke_page_query_bmp_score_block_offset(
                use_compact_filtered_scores
                    ? filtered_document_block_offsets
                    : NULL,
                use_compact_filtered_scores
                    ? filtered_document_block_count
                    : 0,
                ranked->block_id,
                &score_block_offset
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }
            status = evoke_page_query_score_semantic_bmp_block(
                index_relation,
                context,
                cleanup,
                run_caches,
                term_count,
                document_slot_count,
                ranked->block_id,
                use_compact_filtered_scores
                    ? filtered_document_block_offsets
                    : NULL,
                use_compact_filtered_scores
                    ? filtered_document_block_count
                    : 0,
                scores,
                stats
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }
            if (liveness != NULL)
            {
                status = evoke_page_query_load_bmp_liveness(
                    index_relation,
                    context,
                    cleanup,
                    ranked->block_id,
                    liveness,
                    stats
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
            }
            exact_blocks[ranked->block_id] = true;
            exact_block_count++;
            for (uint32 document_slot = first_document;
                 document_slot < document_limit;
                 document_slot++)
            {
                float score = scores[
                    score_block_offset +
                        (document_slot &
                            (EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS -
                             UINT32_C(1)))
                ];

                if (score > 0.0f &&
                    (liveness == NULL ||
                     liveness[document_slot] ==
                         EVOKE_PAGE_QUERY_LIVENESS_LIVE))
                {
                    status = evoke_page_query_offer_root(
                        cleanup,
                        &pruning_accumulator,
                        score,
                        document_slot,
                        document_slot
                    );
                    if (status != EVOKE_OK)
                    {
                        goto done;
                    }
                }
            }
            stats->blocks_scored++;
        }
    }

    {
        bool document_block_loaded = false;

        for (uint32 block_id = 0;
             block_id < (uint32) block_count64;
             block_id++)
        {
            uint32 first_document = block_id <<
                EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
            uint32 document_limit = Min(
                first_document +
                    EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS,
                (uint32) document_slot_count
            );
            size_t score_block_offset;
            bool competitive = false;

            if (!exact_blocks[block_id])
            {
                continue;
            }
            status = evoke_page_query_bmp_score_block_offset(
                use_compact_filtered_scores
                    ? filtered_document_block_offsets
                    : NULL,
                use_compact_filtered_scores
                    ? filtered_document_block_count
                    : 0,
                block_id,
                &score_block_offset
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }

            for (uint32 document_slot = first_document;
                 document_slot < document_limit;
                 document_slot++)
            {
                float score = scores[
                    score_block_offset +
                        (document_slot &
                            (EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS -
                             UINT32_C(1)))
                ];

                if (score != 0.0f)
                {
                    stats->documents_examined++;
                }
                if (score > 0.0f &&
                    (liveness == NULL ||
                     liveness[document_slot] ==
                         EVOKE_PAGE_QUERY_LIVENESS_LIVE) &&
                    evoke_page_query_score_may_enter_topk(
                        &cleanup->accumulator,
                        score))
                {
                    competitive = true;
                }
            }
            if (!competitive)
            {
                continue;
            }
            for (uint32 document_slot = first_document;
                 document_slot < document_limit;
                 document_slot++)
            {
                const evoke_document_cow_record *document;
                float score = scores[
                    score_block_offset +
                        (document_slot &
                            (EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS -
                             UINT32_C(1)))
                ];

                if (score <= 0.0f ||
                    (liveness != NULL &&
                     liveness[document_slot] !=
                         EVOKE_PAGE_QUERY_LIVENESS_LIVE) ||
                    !evoke_page_query_score_may_enter_topk(
                        &cleanup->accumulator,
                        score))
                {
                    continue;
                }
                status = evoke_page_query_load_document(
                    index_relation,
                    context,
                    document_slot,
                    cleanup,
                    &document_block_loaded,
                    &document,
                    stats
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
                if (!evoke_page_query_document_is_live(document) ||
                    !evoke_page_query_root_document_allowed(
                        cleanup,
                        document_slot))
                {
                    continue;
                }
                stats->positive_document_count++;
                status = evoke_page_query_offer_root(
                    cleanup,
                    &cleanup->accumulator,
                    score,
                    document_slot,
                    document->version.born_sequence
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
            }
        }
        if (seed != NULL)
        {
            for (size_t result_index = 0;
                 result_index < seed->len;
                 result_index++)
            {
                const evoke_document_cow_record *document;
                uint32 document_slot = seed->doc_ids[result_index];
                uint32 block_id = document_slot >>
                    EVOKE_SEMANTIC_BMP_PACKED_BLOCK_SHIFT;
                float score = seed->scores[result_index];

                if (document_slot >= document_slot_count ||
                    block_id >= block_count64 || exact_blocks[block_id] ||
                    score <= 0.0f ||
                    !evoke_page_query_score_may_enter_topk(
                        &cleanup->accumulator,
                        score))
                {
                    continue;
                }
                status = evoke_page_query_load_document(
                    index_relation,
                    context,
                    document_slot,
                    cleanup,
                    &document_block_loaded,
                    &document,
                    stats
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
                if (!evoke_page_query_document_is_live(document) ||
                    !evoke_page_query_root_document_allowed(
                        cleanup,
                        document_slot))
                {
                    continue;
                }
                stats->positive_document_count++;
                status = evoke_page_query_offer_root(
                    cleanup,
                    &cleanup->accumulator,
                    score,
                    document_slot,
                    document->version.born_sequence
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
            }
        }
    }
    stats->semantic_bmp_query_path = true;
    if (semantic_posting_count >= stats->semantic_bmp_postings_examined)
    {
        stats->postings_skipped += semantic_posting_count -
            stats->semantic_bmp_postings_examined;
    }
    *used_out = true;

done:
    evoke_page_query_bmp_cleanup_release(resources);
    return status;
}

static bool
evoke_page_query_term_at_a_time_budget(
    const evoke_segment_query_context *context,
    size_t term_count,
    uint64 *bytes_out
)
{
    uint64 slot_count;

    if (context == NULL || bytes_out == NULL || term_count <= 8 ||
        context->root.active_l0.record_count != 0 ||
        context->root.pending_l0.record_count != 0 ||
        context->manifest.visible_document_count !=
            context->manifest.document_slot_count)
    {
        return false;
    }
    slot_count = context->manifest.document_slot_count;
    if (slot_count == 0 || slot_count > UINT32_MAX ||
        slot_count > UINT64_MAX /
            (sizeof(float) + sizeof(uint32)))
    {
        return false;
    }
    *bytes_out = slot_count * sizeof(float);
    if (context->resident_document_lengths == NULL ||
        context->resident_document_length_count != slot_count)
    {
        if (!evoke_page_query_bounded_add(
                bytes_out,
                slot_count,
                sizeof(uint32),
                EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT))
        {
            return false;
        }
    }
    if (*bytes_out > UINT64_MAX -
            EVOKE_SEGMENT_QUERY_POSTING_WINDOW *
                (sizeof(uint32) + sizeof(evoke_posting_value)))
    {
        return false;
    }
    *bytes_out += EVOKE_SEGMENT_QUERY_POSTING_WINDOW *
        (sizeof(uint32) + sizeof(evoke_posting_value));
    return *bytes_out <= EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT &&
        *bytes_out <= SIZE_MAX;
}

static evoke_status
evoke_page_query_score_term_at_a_time(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    double average_document_length,
    evoke_page_query_stats *stats
)
{
    uint64 document_slot_count = context->manifest.document_slot_count;
    uint64 document_block_count;
    evoke_page_query_document_length_build length_build;

    document_block_count =
        (document_slot_count + EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS - 1) >>
        context->query_contract.block_shift;
    if (cleanup->term_at_a_time_owned_document_lengths != NULL &&
        !evoke_segment_pages_load_query_document_lengths(
            index_relation,
            context,
            cleanup->term_at_a_time_owned_document_lengths))
    {
        memset(&length_build, 0, sizeof(length_build));
        length_build.document_lengths =
            cleanup->term_at_a_time_owned_document_lengths;
        length_build.document_slot_count = document_slot_count;
        length_build.valid = true;
        evoke_segment_pages_visit_query_document_records(
            index_relation,
            context,
            &cleanup->document_reader,
            evoke_page_query_collect_document_length,
            &length_build
        );
        if (!length_build.valid ||
            length_build.next_document_slot != document_slot_count)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    stats->document_block_reads += document_block_count;
    stats->max_document_block_records = Max(
        stats->max_document_block_records,
        (uint32) EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS
    );

    for (size_t term_index = 0; term_index < term_count; term_index++)
    {
        const evoke_page_query_term *term = &cleanup->terms[term_index];

        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            const evoke_segment_query_run *run =
                &term->plan.runs[run_index];
            evoke_segment_query_posting_cursor posting_cursor;
            uint32 previous_document_slot = 0;
            bool have_previous_document = false;

            evoke_segment_query_posting_cursor_init(&posting_cursor);
            stats->posting_block_reads += run->block_count;
            while (posting_cursor.next_posting_index < run->posting_count)
            {
                uint64 first_posting_index =
                    posting_cursor.next_posting_index;
                uint32 posting_count;

                CHECK_FOR_INTERRUPTS();
                if (!evoke_test_disable_fused_semantic_taat &&
                    run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT &&
                    run->semantic_bmp.available)
                {
                    double impact_floor = 0.0;

                    if (evoke_test_query_semantic_impact_floor_ratio > 0.0)
                    {
                        impact_floor =
                            evoke_test_query_semantic_impact_floor_ratio * Max(
                                fabs((double)
                                    run->semantic_bmp.min_impact),
                                fabs((double)
                                    run->semantic_bmp.max_impact)
                            );
                    }
                    posting_count =
                        evoke_segment_pages_accumulate_query_semantic_stream_window(
                            index_relation,
                            context,
                            &term->plan,
                            run_index,
                            &posting_cursor,
                            (uint8 *) cleanup->
                                term_at_a_time_document_slots,
                            (uint8 *) cleanup->term_at_a_time_values,
                            EVOKE_SEGMENT_QUERY_POSTING_WINDOW,
                            term->query_weight,
                            cleanup->term_at_a_time_scores,
                            document_slot_count,
                            impact_floor,
                            &stats->query_impact_floor_omitted_postings
                        );
                    stats->postings_examined += posting_count;
                    previous_document_slot =
                        posting_cursor.previous_document_slot;
                    have_previous_document = true;
                    if (posting_count == 0 ||
                        posting_count >
                            run->posting_count - first_posting_index ||
                        posting_cursor.next_posting_index !=
                            first_posting_index + posting_count)
                    {
                        return EVOKE_ERR_FORMAT;
                    }
                    continue;
                }
                posting_count =
                    evoke_segment_pages_load_query_term_posting_stream_window(
                        index_relation,
                        context,
                        &term->plan,
                        run_index,
                        &posting_cursor,
                        cleanup->term_at_a_time_document_slots,
                        cleanup->term_at_a_time_values,
                        EVOKE_SEGMENT_QUERY_POSTING_WINDOW
                    );
                if (posting_count == 0 ||
                    posting_count >
                        run->posting_count - first_posting_index ||
                    posting_cursor.next_posting_index !=
                        first_posting_index + posting_count)
                {
                    return EVOKE_ERR_FORMAT;
                }
                for (uint32 posting_index = 0;
                     posting_index < posting_count;
                     posting_index++)
                {
                    uint32 document_slot =
                        cleanup->term_at_a_time_document_slots[
                            posting_index
                        ];
                    float contribution;

                    if ((uint64) document_slot >= document_slot_count ||
                        (have_previous_document &&
                         document_slot <= previous_document_slot))
                    {
                        return EVOKE_ERR_FORMAT;
                    }
                    previous_document_slot = document_slot;
                    have_previous_document = true;
                    stats->postings_examined++;
                    if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                    {
                        double tfc;

                        if (term->live_document_frequency == 0)
                        {
                            return EVOKE_ERR_FORMAT;
                        }
                        tfc = evoke_score_tfc(
                            context->query_contract.params.method,
                            (double) cleanup->term_at_a_time_values[
                                posting_index
                            ].term_frequency,
                            (double) cleanup->
                                term_at_a_time_document_lengths[
                                    document_slot
                                ],
                            average_document_length,
                            context->query_contract.params.k1,
                            context->query_contract.params.b,
                            context->query_contract.params.delta
                        );
                        contribution = (float) (
                            (double) term->query_weight *
                            (term->idf * tfc -
                             (double) term->nonoccurrence)
                        );
                        cleanup->term_at_a_time_scores[document_slot] +=
                            contribution;
                    }
                    else if (run->kind ==
                                 EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
                             run->kind ==
                                 EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
                    {
                        if (evoke_page_query_test_omit_semantic_impact(
                                run,
                                cleanup->term_at_a_time_values[
                                    posting_index
                                ].impact,
                                stats))
                        {
                            continue;
                        }
                        cleanup->term_at_a_time_scores[document_slot] +=
                            term->query_weight *
                            cleanup->term_at_a_time_values[
                                posting_index
                            ].impact;
                    }
                    else
                    {
                        return EVOKE_ERR_FORMAT;
                    }
                }
            }
        }
    }

    for (uint64 block_id = 0;
         block_id < document_block_count;
         block_id++)
    {
        uint64 first_document_slot = block_id <<
            context->query_contract.block_shift;
        uint32 record_count = (uint32) Min(
            document_slot_count - first_document_slot,
            (uint64) EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS
        );
        bool competitive = false;

        if ((block_id & EVOKE_PAGE_QUERY_INTERRUPT_MASK) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
        stats->blocks_considered++;
        for (uint32 local_slot = 0;
             local_slot < record_count;
             local_slot++)
        {
            float score = cleanup->term_at_a_time_scores[
                first_document_slot + local_slot
            ];

            if (score != 0.0f)
            {
                stats->documents_examined++;
            }
            if (score > 0.0f && evoke_page_query_score_may_enter_topk(
                    &cleanup->accumulator,
                    score))
            {
                competitive = true;
            }
        }
        if (!competitive)
        {
            stats->blocks_skipped++;
            continue;
        }
        evoke_segment_pages_load_query_document_block(
            index_relation,
            context,
            &cleanup->document_reader,
            (uint32) block_id,
            &cleanup->document_block
        );
        stats->document_block_reads++;
        stats->max_document_block_records = Max(
            stats->max_document_block_records,
            cleanup->document_block.record_count
        );
        if (cleanup->document_block.record_count != record_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32 local_slot = 0;
             local_slot < record_count;
             local_slot++)
        {
            const evoke_document_cow_record *document =
                &cleanup->document_block.records[local_slot];
            uint32 document_slot =
                (uint32) first_document_slot + local_slot;
            float score = cleanup->term_at_a_time_scores[document_slot];
            evoke_status status;

            if (score <= 0.0f ||
                !evoke_page_query_score_may_enter_topk(
                    &cleanup->accumulator,
                    score))
            {
                continue;
            }
            if (document->version.document_slot != document_slot)
            {
                return EVOKE_ERR_FORMAT;
            }
            if (!evoke_page_query_document_is_live(document))
            {
                continue;
            }
            stats->positive_document_count++;
            status = evoke_page_query_offer_root(
                cleanup,
                &cleanup->accumulator,
                score,
                document_slot,
                document->version.born_sequence
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
        }
        stats->blocks_scored++;
    }
    stats->term_at_a_time_query_path = true;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_try_score_term_at_a_time(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    double average_document_length,
    evoke_page_query_stats *stats,
    bool *used_out
)
{
    uint64 memory_bytes;
    uint64 score_bytes;
    uint64 length_bytes;
    bool use_resident_document_lengths;
    evoke_status status;

    if (used_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *used_out = false;
    if (!evoke_page_query_term_at_a_time_budget(
            context,
            term_count,
            &memory_bytes))
    {
        return EVOKE_OK;
    }
    score_bytes = context->manifest.document_slot_count * sizeof(float);
    length_bytes = context->manifest.document_slot_count * sizeof(uint32);
    use_resident_document_lengths =
        context->resident_document_lengths != NULL &&
        context->resident_document_length_count ==
            context->manifest.document_slot_count;
    cleanup->term_at_a_time_scores = calloc(1, (size_t) score_bytes);
    if (use_resident_document_lengths)
    {
        cleanup->term_at_a_time_document_lengths =
            context->resident_document_lengths;
    }
    else
    {
        cleanup->term_at_a_time_owned_document_lengths = malloc(
            (size_t) length_bytes
        );
        cleanup->term_at_a_time_document_lengths =
            cleanup->term_at_a_time_owned_document_lengths;
    }
    cleanup->term_at_a_time_document_slots = malloc(
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW *
            sizeof(*cleanup->term_at_a_time_document_slots)
    );
    cleanup->term_at_a_time_values = malloc(
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW *
            sizeof(*cleanup->term_at_a_time_values)
    );
    if (cleanup->term_at_a_time_scores == NULL ||
        cleanup->term_at_a_time_document_lengths == NULL ||
        cleanup->term_at_a_time_document_slots == NULL ||
        cleanup->term_at_a_time_values == NULL)
    {
        free(cleanup->term_at_a_time_scores);
        free(cleanup->term_at_a_time_owned_document_lengths);
        free(cleanup->term_at_a_time_document_slots);
        free(cleanup->term_at_a_time_values);
        cleanup->term_at_a_time_scores = NULL;
        cleanup->term_at_a_time_document_lengths = NULL;
        cleanup->term_at_a_time_owned_document_lengths = NULL;
        cleanup->term_at_a_time_document_slots = NULL;
        cleanup->term_at_a_time_values = NULL;
        return EVOKE_OK;
    }
    stats->term_at_a_time_score_bytes = score_bytes;
    stats->term_at_a_time_document_length_bytes =
        use_resident_document_lengths ? 0 : length_bytes;
    status = evoke_page_query_score_term_at_a_time(
        index_relation,
        context,
        cleanup,
        term_count,
        average_document_length,
        stats
    );
    if (status == EVOKE_OK)
    {
        *used_out = true;
    }
    return status;
}

static bool
evoke_page_query_materialization_add(
    uint64 *total,
    uint64 count,
    uint64 item_size
)
{
    return evoke_page_query_bounded_add(
        total,
        count,
        item_size,
        EVOKE_PAGE_QUERY_MATERIALIZATION_LIMIT
    );
}

static bool
evoke_page_query_materialization_budget(
    const evoke_segment_query_context *context,
    const evoke_page_query_cleanup *cleanup,
    size_t term_count,
    uint64 *bytes_out,
    size_t *block_count_out
)
{
    uint64 bytes = 0;
    uint64 block_count = 0;

    if (context == NULL || cleanup == NULL || bytes_out == NULL ||
        block_count_out == NULL || term_count <= 8 ||
        context->root.active_l0.record_count != 0 ||
        context->root.pending_l0.record_count != 0 ||
        context->manifest.visible_document_count !=
            context->manifest.document_slot_count ||
        !evoke_page_query_materialization_add(
            &bytes,
            term_count,
            sizeof(evoke_segment_query_term)))
    {
        return false;
    }
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        const evoke_segment_query_term_plan *plan =
            &cleanup->terms[term_index].plan;
        uint64 extent_capacity = plan->run_count == 0 ? 0 : 4;

        while (extent_capacity < plan->run_count)
        {
            if (extent_capacity > UINT64_MAX / 2)
            {
                return false;
            }
            extent_capacity *= 2;
        }
        if (!evoke_page_query_materialization_add(
                &bytes,
                extent_capacity,
                sizeof(evoke_segment_query_extent)))
        {
            return false;
        }

        for (uint32 run_index = 0;
             run_index < plan->run_count;
             run_index++)
        {
            const evoke_segment_query_run *run =
                &plan->runs[run_index];

            /*
             * Packed semantic runs have no expanded extent authority.
             * Their bounded fallbacks stream the packed posting window or
             * synthesize one block cursor at a time instead.
             */
            if (run->kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
            {
                return false;
            }
            if (!evoke_page_query_materialization_add(
                    &bytes,
                    run->posting_count,
                    sizeof(uint32) + sizeof(evoke_posting_value)) ||
                !evoke_page_query_materialization_add(
                    &bytes,
                    run->block_count,
                    sizeof(evoke_posting_block_record) +
                        sizeof(uint32)) ||
                block_count > SIZE_MAX - run->block_count)
            {
                return false;
            }
            block_count += run->block_count;
        }
    }
    if (block_count == 0 || block_count > SIZE_MAX)
    {
        return false;
    }
    *bytes_out = bytes;
    *block_count_out = (size_t) block_count;
    return true;
}

static int
evoke_page_query_compare_block_ids(const void *left, const void *right)
{
    uint32 a = *(const uint32 *) left;
    uint32 b = *(const uint32 *) right;

    return a < b ? -1 : a > b ? 1 : 0;
}

static const evoke_posting_block_record *
evoke_page_query_find_materialized_block(
    const evoke_posting_extent *extent,
    uint32 block_id
)
{
    uint32 low = 0;
    uint32 high;

    if (extent == NULL || extent->blocks == NULL)
    {
        return NULL;
    }
    high = extent->block_count;
    while (low < high)
    {
        uint32 middle = low + (high - low) / 2;
        uint32 candidate = extent->blocks[middle].block_id;

        if (candidate < block_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low < extent->block_count &&
        extent->blocks[low].block_id == block_id
        ? &extent->blocks[low]
        : NULL;
}

static evoke_status
evoke_page_query_score_materialized_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    uint32 block_id,
    double average_document_length,
    evoke_page_query_stats *stats
)
{
    float scores[EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS] = {0};
    bool touched[EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS] = {0};
    uint64 first_document_slot;
    uint64 remaining;
    uint32 document_count;

    first_document_slot = (uint64) block_id <<
        context->query_contract.block_shift;
    if (first_document_slot >= context->manifest.document_slot_count ||
        first_document_slot > UINT32_MAX)
    {
        return EVOKE_ERR_FORMAT;
    }
    remaining = context->manifest.document_slot_count -
        first_document_slot;
    document_count = remaining > EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS
        ? EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS
        : (uint32) remaining;

    evoke_segment_pages_load_query_document_block(
        index_relation,
        context,
        &cleanup->document_reader,
        block_id,
        &cleanup->document_block
    );
    stats->document_block_reads++;
    stats->max_document_block_records = Max(
        stats->max_document_block_records,
        cleanup->document_block.record_count
    );

    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        const evoke_page_query_term *term = &cleanup->terms[term_index];
        const evoke_segment_query_term *materialized =
            &cleanup->materialized_terms[term_index];

        for (uint32 extent_index = 0;
             extent_index < materialized->extent_count;
             extent_index++)
        {
            const evoke_posting_extent *extent =
                &materialized->extents[extent_index].view;
            const evoke_posting_block_record *record =
                evoke_page_query_find_materialized_block(
                    extent,
                    block_id
                );

            if (record == NULL)
            {
                continue;
            }
            if (record->posting_offset > extent->len ||
                record->posting_count >
                    extent->len - record->posting_offset)
            {
                return EVOKE_ERR_FORMAT;
            }
            for (uint32 local_posting = 0;
                 local_posting < record->posting_count;
                 local_posting++)
            {
                uint64 posting_index =
                    record->posting_offset + local_posting;
                uint32 document_slot = extent->indices[posting_index];
                uint32 local_slot;
                const evoke_document_cow_record *document;

                if ((uint64) document_slot < first_document_slot)
                {
                    return EVOKE_ERR_FORMAT;
                }
                local_slot = (uint32) (
                    (uint64) document_slot - first_document_slot
                );
                if (local_slot >= document_count)
                {
                    return EVOKE_ERR_FORMAT;
                }
                document = &cleanup->document_block.records[local_slot];
                if (document->version.document_slot != document_slot)
                {
                    return EVOKE_ERR_FORMAT;
                }
                stats->postings_examined++;
                touched[local_slot] = true;
                if (!evoke_page_query_document_is_live(document))
                {
                    return EVOKE_ERR_FORMAT;
                }
                if (extent->kind ==
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    double tfc;

                    if (term->live_document_frequency == 0)
                    {
                        return EVOKE_ERR_FORMAT;
                    }
                    if (document == NULL)
                    {
                        return EVOKE_ERR_FORMAT;
                    }
                    tfc = evoke_score_tfc(
                        context->query_contract.params.method,
                        (double) extent->values[
                            posting_index
                        ].term_frequency,
                        (double) document->version.document_length,
                        average_document_length,
                        context->query_contract.params.k1,
                        context->query_contract.params.b,
                        context->query_contract.params.delta
                    );
                    scores[local_slot] += (float) (
                        (double) term->query_weight *
                        (term->idf * tfc -
                         (double) term->nonoccurrence)
                    );
                }
                else if (extent->kind ==
                             EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
                         extent->kind ==
                             EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
                {
                    scores[local_slot] += term->query_weight *
                        extent->values[posting_index].impact;
                }
                else
                {
                    return EVOKE_ERR_FORMAT;
                }
            }
        }
    }
    for (uint32 local_slot = 0;
         local_slot < document_count;
         local_slot++)
    {
        const evoke_document_cow_record *document =
            &cleanup->document_block.records[local_slot];
        evoke_status status;

        if (!touched[local_slot])
        {
            continue;
        }
        stats->documents_examined++;
        if (scores[local_slot] <= 0.0f)
        {
            continue;
        }
        if (document->version.document_slot !=
                first_document_slot + local_slot ||
            !evoke_page_query_document_is_live(document))
        {
            return EVOKE_ERR_FORMAT;
        }
        stats->positive_document_count++;
        status = evoke_page_query_offer_root(
            cleanup,
            &cleanup->accumulator,
            scores[local_slot],
            (uint32) document->version.document_slot,
            document->version.born_sequence
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_try_materialized_scoring(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    double average_document_length,
    evoke_page_query_stats *stats,
    bool *used_out
)
{
    uint64 materialized_bytes;
    size_t block_capacity;
    size_t block_count = 0;

    if (used_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *used_out = false;
    if (!evoke_page_query_materialization_budget(
            context,
            cleanup,
            term_count,
            &materialized_bytes,
            &block_capacity))
    {
        return EVOKE_OK;
    }
    cleanup->materialized_terms = palloc0(
        term_count * sizeof(*cleanup->materialized_terms)
    );
    cleanup->materialized_term_count = term_count;
    cleanup->materialized_block_ids = palloc(
        block_capacity * sizeof(*cleanup->materialized_block_ids)
    );

    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        evoke_segment_query_term *materialized =
            &cleanup->materialized_terms[term_index];

        evoke_segment_query_term_init(materialized);
        evoke_segment_pages_load_query_term_from_plan(
            index_relation,
            context,
            &cleanup->terms[term_index].plan,
            materialized
        );
        if (materialized->term_id !=
                cleanup->terms[term_index].plan.term_id ||
            materialized->raw_document_frequency !=
                cleanup->terms[term_index].plan.raw_document_frequency)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32 extent_index = 0;
             extent_index < materialized->extent_count;
             extent_index++)
        {
            const evoke_posting_extent *extent =
                &materialized->extents[extent_index].view;

            stats->posting_block_reads += extent->block_count;
            for (uint32 record_index = 0;
                 record_index < extent->block_count;
                 record_index++)
            {
                if (block_count >= block_capacity)
                {
                    return EVOKE_ERR_FORMAT;
                }
                cleanup->materialized_block_ids[block_count++] =
                    extent->blocks[record_index].block_id;
            }
        }
    }
    qsort(
        cleanup->materialized_block_ids,
        block_count,
        sizeof(*cleanup->materialized_block_ids),
        evoke_page_query_compare_block_ids
    );
    cleanup->materialized_block_count = 0;
    for (size_t block_index = 0;
         block_index < block_count;
         block_index++)
    {
        if (cleanup->materialized_block_count == 0 ||
            cleanup->materialized_block_ids[block_index] !=
                cleanup->materialized_block_ids[
                    cleanup->materialized_block_count - 1])
        {
            cleanup->materialized_block_ids[
                cleanup->materialized_block_count++] =
                    cleanup->materialized_block_ids[block_index];
        }
    }
    for (size_t block_index = 0;
         block_index < cleanup->materialized_block_count;
         block_index++)
    {
        evoke_status status;

        CHECK_FOR_INTERRUPTS();
        stats->blocks_considered++;
        status = evoke_page_query_score_materialized_block(
            index_relation,
            context,
            cleanup,
            term_count,
            cleanup->materialized_block_ids[block_index],
            average_document_length,
            stats
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        stats->blocks_scored++;
    }
    stats->materialized_query_bytes = materialized_bytes;
    stats->materialized_query_path = true;
    *used_out = true;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_score_positive_documents(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    evoke_page_query_cleanup *cleanup,
    size_t term_count,
    size_t cursor_count,
    double average_document_length,
    evoke_page_query_stats *stats
)
{
    bool document_block_loaded = false;
    uint32 document_slot;
    uint64 interrupt_counter = 0;
    size_t scoring_cursor_count = 0;
    evoke_status status;

    status = evoke_page_query_reset_cursors(
        index_relation,
        context,
        cleanup,
        term_count,
        cursor_count,
        false,
        &scoring_cursor_count,
        stats
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    while (evoke_page_query_next_document(
        cleanup->cursors,
        scoring_cursor_count,
        &document_slot))
    {
        const evoke_document_cow_record *record;
        const evoke_page_query_l0_document *projected_document = NULL;
        float score = 0.0f;
        bool live;

        if ((interrupt_counter++ & EVOKE_PAGE_QUERY_INTERRUPT_MASK) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
        status = evoke_page_query_load_document(
            index_relation,
            context,
            document_slot,
            cleanup,
            &document_block_loaded,
            &record,
            stats
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        live = evoke_page_query_document_is_live(record);
        if (projection != NULL)
        {
            projected_document = evoke_page_query_l0_find_document(
                projection,
                document_slot
            );
            if (live && projected_document != NULL &&
                projected_document->shadows_immutable)
            {
                live = false;
            }
        }
        stats->documents_examined++;
        for (size_t cursor_index = 0;
             cursor_index < scoring_cursor_count;
             cursor_index++)
        {
            evoke_page_query_cursor *cursor =
                &cleanup->cursors[cursor_index];
            evoke_page_query_term *term;
            const evoke_segment_query_run *run;

            if (cursor->exhausted ||
                cursor->block.document_slots[
                    cursor->posting_index
                ] != document_slot)
            {
                continue;
            }
            term = &cleanup->terms[cursor->term_index];
            run = &term->plan.runs[cursor->run_index];
            stats->postings_examined++;
            if (live)
            {
                if (run->kind ==
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    double tfc;

                    if (term->live_document_frequency == 0)
                    {
                        return EVOKE_ERR_FORMAT;
                    }
                    tfc = evoke_score_tfc(
                        context->query_contract.params.method,
                        (double) cursor->block.values[
                            cursor->posting_index
                        ].term_frequency,
                        (double) record->version.document_length,
                        average_document_length,
                        context->query_contract.params.k1,
                        context->query_contract.params.b,
                        context->query_contract.params.delta
                    );
                    score += (float) (
                        (double) term->query_weight *
                        (term->idf * tfc -
                         (double) term->nonoccurrence)
                    );
                }
                else if (run->kind ==
                         EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
                         run->kind ==
                         EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
                {
                    score += term->query_weight *
                        cursor->block.values[
                            cursor->posting_index
                        ].impact;
                }
                else
                {
                    return EVOKE_ERR_FORMAT;
                }
            }
            status = evoke_page_query_cursor_advance(
                index_relation,
                context,
                term,
                cursor,
                stats
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
        }
        if (live && projected_document != NULL &&
            projected_document->live &&
            projected_document->contribution_count > 0)
        {
            float overlay_score = 0.0f;
            size_t projected_index = (size_t) (
                projected_document - projection->documents
            );

            status = evoke_page_query_score_l0_document(
                context,
                projection,
                projected_document,
                cleanup,
                average_document_length,
                &overlay_score
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
            score += overlay_score;
            cleanup->projection_document_seen[projected_index] = true;
        }
        if (live && score > 0.0f)
        {
            stats->positive_document_count++;
            status = evoke_page_query_offer_root(
                cleanup,
                &cleanup->accumulator,
                score,
                document_slot,
                record->version.born_sequence
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_score_remaining_l0_documents(
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    evoke_page_query_cleanup *cleanup,
    double average_document_length,
    evoke_page_query_stats *stats
)
{
    if (projection == NULL)
    {
        return EVOKE_OK;
    }
    for (size_t document_index = 0;
         document_index < projection->document_count;
         document_index++)
    {
        const evoke_page_query_l0_document *document =
            &projection->documents[document_index];
        float score = 0.0f;
        evoke_status status;

        if ((document_index & EVOKE_PAGE_QUERY_INTERRUPT_MASK) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
        if (!document->live || document->contribution_count == 0 ||
            cleanup->projection_document_seen[document_index])
        {
            continue;
        }
        status = evoke_page_query_score_l0_document(
            context,
            projection,
            document,
            cleanup,
            average_document_length,
            &score
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        cleanup->projection_document_seen[document_index] = true;
        stats->documents_examined++;
        if (score <= 0.0f)
        {
            continue;
        }
        stats->positive_document_count++;
        status = evoke_page_query_offer_projection(
            cleanup,
            &cleanup->accumulator,
            score,
            document->document_slot,
            document->born_sequence
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    return EVOKE_OK;
}

static void
evoke_page_query_cleanup_free(evoke_page_query_cleanup *cleanup)
{
    if (cleanup == NULL)
    {
        return;
    }
    evoke_topk_accumulator_free(&cleanup->accumulator);
    if (cleanup->materialized_terms != NULL)
    {
        for (size_t term_index = 0;
             term_index < cleanup->materialized_term_count;
             term_index++)
        {
            evoke_segment_query_term_free(
                &cleanup->materialized_terms[term_index]
            );
        }
        pfree(cleanup->materialized_terms);
    }
    if (cleanup->materialized_block_ids != NULL)
    {
        pfree(cleanup->materialized_block_ids);
    }
    free(cleanup->term_at_a_time_scores);
    free(cleanup->term_at_a_time_owned_document_lengths);
    free(cleanup->term_at_a_time_document_slots);
    free(cleanup->term_at_a_time_values);
    free(cleanup->projection_document_seen);
    free(cleanup->query_term_map);
    free(cleanup->block_cursor_group);
    free(cleanup->block_cursor_heap);
    free(cleanup->block_cursors);
    free(cleanup->cursors);
    free(cleanup->terms);
    pfree(cleanup);
}

typedef struct evoke_page_query_accelerator_term
{
    uint32 term_id;
    float weight;
    evoke_segment_query_term_plan plan;
    bool accelerated;
} evoke_page_query_accelerator_term;

typedef struct evoke_page_query_residual_merge
{
    evoke_page_query_term *terms;
    evoke_page_query_cursor *cursors;
    uint32 *heap;
    size_t term_count;
    size_t cursor_count;
    size_t heap_count;
} evoke_page_query_residual_merge;

static size_t
evoke_page_query_apply_accelerator_error_budget(
    evoke_page_query_accelerator_term *terms,
    size_t term_count,
    evoke_page_query_stats *stats
)
{
    evoke_page_query_error_budget_candidate *candidates;
    bool *omit;
    size_t candidate_count = 0;
    size_t omitted_count = 0;
    size_t omit_limit;
    size_t retained = 0;
    double total_bound = 0.0;
    double omitted_bound = 0.0;
    double budget;

    if (terms == NULL || stats == NULL || term_count == 0 ||
        evoke_test_query_semantic_error_budget_ratio <= 0.0)
    {
        return term_count;
    }
    candidates = palloc(sizeof(*candidates) * term_count);
    omit = palloc0(sizeof(*omit) * term_count);
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        double bound;
        uint64 postings;

        if (terms[term_index].accelerated)
        {
            continue;
        }
        if (!evoke_page_query_semantic_plan_absolute_bound(
                &terms[term_index].plan,
                terms[term_index].weight,
                &bound,
                &postings))
        {
            continue;
        }
        if (!isfinite(total_bound + bound))
        {
            pfree(omit);
            pfree(candidates);
            return term_count;
        }
        total_bound += bound;
        candidates[candidate_count].term_index = term_index;
        candidates[candidate_count].absolute_bound = bound;
        candidates[candidate_count].postings = postings;
        candidate_count++;
    }
    if (candidate_count == 0 || total_bound == 0.0)
    {
        pfree(omit);
        pfree(candidates);
        return term_count;
    }
    budget = evoke_test_query_semantic_error_budget_ratio * total_bound;
    omit_limit = candidate_count;
    if (evoke_test_query_semantic_min_support_ratio > 0.0)
    {
        size_t retain_count = (size_t) ceil(
            evoke_test_query_semantic_min_support_ratio *
            (double) candidate_count
        );

        retain_count = Min(retain_count, candidate_count);
        omit_limit = candidate_count - retain_count;
    }
    qsort(
        candidates,
        candidate_count,
        sizeof(*candidates),
        evoke_page_query_compare_error_budget_candidates
    );
    for (size_t candidate_index = 0;
         candidate_index < candidate_count;
         candidate_index++)
    {
        const evoke_page_query_error_budget_candidate *candidate =
            &candidates[candidate_index];
        double next_omitted_bound =
            omitted_bound + candidate->absolute_bound;

        if (omitted_count >= omit_limit)
        {
            break;
        }
        if (!isfinite(next_omitted_bound) || next_omitted_bound > budget)
        {
            continue;
        }
        omit[candidate->term_index] = true;
        omitted_count++;
        omitted_bound = next_omitted_bound;
        if (stats->query_error_budget_pruned_term_count < UINT32_MAX)
        {
            stats->query_error_budget_pruned_term_count++;
        }
        if (stats->query_error_budget_pruned_postings >
            UINT64_MAX - candidate->postings)
        {
            stats->query_error_budget_pruned_postings = UINT64_MAX;
        }
        else
        {
            stats->query_error_budget_pruned_postings +=
                candidate->postings;
        }
    }
    stats->query_semantic_total_absolute_bound = total_bound;
    stats->query_semantic_omitted_absolute_bound = omitted_bound;
    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        if (omit[term_index])
        {
            continue;
        }
        if (retained != term_index)
        {
            terms[retained] = terms[term_index];
        }
        retained++;
    }
    pfree(omit);
    pfree(candidates);
    return retained;
}

typedef struct evoke_page_query_forward_score_context
{
    Relation index_relation;
    const evoke_segment_query_context *query_context;
    const evoke_semantic_accelerator_directory *directory;
    const evoke_page_query_filter *candidate_filter;
    evoke_semantic_forward_chunk chunk;
    evoke_segment_forward_row_reader row_reader;
    evoke_segment_forward_transpose_reader transpose_reader;
    const uint32 *query_ids;
    const float *query_weights;
    const uint32 *allowed_forward_chunk_counts;
    const uint8 *direct_forward_chunks;
    size_t query_count;
    bool query_is_sorted;
    bool direct_rows;
    uint8 *scored_documents;
    float base_score;
    uint64 chunk_reads;
    uint64 row_reads;
    uint64 postings_examined;
    uint64 bytes_read;
    uint64 scratch_extra_peak_bytes;
    evoke_topk_accumulator filtered_topk;
} evoke_page_query_forward_score_context;

typedef struct evoke_page_query_forward_bound_block
{
    float upper_bound;
    uint32 block_id;
} evoke_page_query_forward_bound_block;

typedef struct evoke_page_query_forward_bound_accumulation
{
    double *upper_bounds;
    const uint8 *allowed_blocks;
    uint32 block_count;
    float query_weight;
} evoke_page_query_forward_bound_accumulation;

typedef struct evoke_page_query_accelerator_cleanup
{
    MemoryContextCallback callback;
    bool active;
    evoke_topk_result *result_out;
    evoke_semantic_accelerator_directory directory;
    evoke_page_query_accelerator_term *terms;
    evoke_semantic_accelerator_index *owned_indexes;
    const evoke_semantic_accelerator_index **indexes;
    evoke_page_query_forward_score_context score_context;
    evoke_topk_result accelerated_result;
    evoke_topk_result residual_result;
    evoke_topk_result term_candidate_result;
    evoke_topk_result residual_candidate_result;
    evoke_topk_accumulator residual_topk;
    evoke_topk_accumulator term_candidate_topk;
    evoke_topk_accumulator residual_candidate_topk;
    evoke_topk_accumulator merged_topk;
    evoke_page_query_residual_merge exact_residual_merge;
    evoke_topk_accumulator exact_residual_topk;
    evoke_weighted_space_saving residual_summary;
    uint32 *canonical_ids;
    float *canonical_weights;
    uint32 *ordered_ids;
    float *ordered_weights;
    uint32 *allowed_forward_chunk_counts;
    uint32 *hybrid_transpose_chunk_counts;
    uint8 *hybrid_direct_chunks;
    uint8 *accelerated_documents;
    uint8 *residual_documents;
    float *residual_candidate_scores;
    uint32 *document_window;
    evoke_posting_value *value_window;
    evoke_semantic_accelerator_query_scratch *accelerator_scratch;
    size_t index_count;
} evoke_page_query_accelerator_cleanup;

static uint64
evoke_page_query_memory_add(
    uint64 bytes,
    uint64 count,
    size_t element_size
)
{
    return evoke_u64_saturating_add(
        bytes,
        evoke_u64_saturating_mul(count, (uint64) element_size)
    );
}

static uint64
evoke_page_query_accelerator_index_bytes(
    const evoke_semantic_accelerator_index *index
)
{
    uint64 bytes = 0;

    if (index == NULL)
    {
        return 0;
    }
    bytes = evoke_page_query_memory_add(
        bytes,
        index->term_count,
        sizeof(*index->terms)
    );
    bytes = evoke_page_query_memory_add(
        bytes,
        index->cluster_count,
        sizeof(*index->clusters)
    );
    bytes = evoke_page_query_memory_add(
        bytes,
        index->document_ref_count,
        sizeof(*index->document_ids)
    );
    return evoke_page_query_memory_add(
        bytes,
        index->summary_count,
        sizeof(*index->summaries)
    );
}

static uint64
evoke_page_query_topk_result_bytes(const evoke_topk_result *result)
{
    if (result == NULL)
    {
        return 0;
    }
    return evoke_page_query_memory_add(
        0,
        result->len,
        sizeof(*result->doc_ids) + sizeof(*result->scores)
    );
}

static uint64
evoke_page_query_topk_accumulator_bytes(
    const evoke_topk_accumulator *accumulator
)
{
    if (accumulator == NULL || accumulator->heap == NULL)
    {
        return 0;
    }
    return evoke_page_query_memory_add(
        0,
        accumulator->capacity,
        sizeof(*accumulator->heap)
    );
}

static uint64
evoke_page_query_summary_bytes(
    const evoke_weighted_space_saving *summary
)
{
    uint64 bytes = 0;

    if (summary == NULL)
    {
        return 0;
    }
    if (summary->counters != NULL)
    {
        bytes = evoke_page_query_memory_add(
            bytes,
            summary->capacity,
            sizeof(*summary->counters)
        );
    }
    if (summary->hash_slots != NULL)
    {
        bytes = evoke_page_query_memory_add(
            bytes,
            summary->hash_capacity,
            sizeof(*summary->hash_slots)
        );
    }
    if (summary->heap != NULL)
    {
        bytes = evoke_page_query_memory_add(
            bytes,
            summary->capacity,
            sizeof(*summary->heap)
        );
    }
    return bytes;
}

static uint64
evoke_page_query_forward_scratch_bytes(
    const evoke_page_query_forward_score_context *context
)
{
    const evoke_semantic_forward_chunk *chunk;
    const evoke_segment_forward_row_reader *row_reader;
    const evoke_segment_forward_transpose_reader *transpose_reader;
    uint64 bytes;

    if (context == NULL)
    {
        return 0;
    }
    chunk = &context->chunk;
    row_reader = &context->row_reader;
    transpose_reader = &context->transpose_reader;
    bytes = context->scratch_extra_peak_bytes;
    if (chunk->row_offsets != NULL)
    {
        bytes = evoke_page_query_memory_add(
            bytes,
            (uint64) chunk->document_count + 1U,
            sizeof(*chunk->row_offsets)
        );
    }
    if (chunk->term_ids != NULL)
    {
        bytes = evoke_page_query_memory_add(
            bytes,
            chunk->posting_count,
            sizeof(*chunk->term_ids)
        );
    }
    if (chunk->operations != NULL)
    {
        bytes = evoke_page_query_memory_add(
            bytes,
            chunk->posting_count,
            sizeof(*chunk->operations)
        );
    }
    if (chunk->contributions != NULL)
    {
        bytes = evoke_page_query_memory_add(
            bytes,
            chunk->posting_count,
            sizeof(*chunk->contributions)
        );
    }
    if (row_reader->row_offsets != NULL)
    {
        bytes = evoke_u64_saturating_add(
            bytes,
            (uint64) row_reader->row_offset_capacity
        );
    }
    if (row_reader->row_bytes != NULL)
    {
        bytes = evoke_u64_saturating_add(
            bytes,
            (uint64) row_reader->row_capacity
        );
    }
    if (transpose_reader->scales != NULL)
    {
        bytes = evoke_u64_saturating_add(
            bytes,
            (uint64) transpose_reader->scales_capacity
        );
    }
    if (transpose_reader->dense_term_ids != NULL)
    {
        bytes = evoke_u64_saturating_add(
            bytes,
            (uint64) transpose_reader->dense_term_ids_capacity
        );
    }
    if (transpose_reader->dense_codes != NULL)
    {
        bytes = evoke_u64_saturating_add(
            bytes,
            (uint64) transpose_reader->dense_codes_capacity
        );
    }
    if (transpose_reader->term_offsets != NULL)
    {
        bytes = evoke_u64_saturating_add(
            bytes,
            (uint64) transpose_reader->term_offsets_capacity
        );
    }
    if (transpose_reader->query_ranges != NULL)
    {
        bytes = evoke_u64_saturating_add(
            bytes,
            (uint64) transpose_reader->query_ranges_capacity
        );
    }
    if (transpose_reader->query_dense_indexes != NULL)
    {
        bytes = evoke_u64_saturating_add(
            bytes,
            (uint64) transpose_reader->query_dense_indexes_capacity
        );
    }
    if (transpose_reader->postings != NULL)
    {
        bytes = evoke_u64_saturating_add(
            bytes,
            (uint64) transpose_reader->postings_capacity
        );
    }
    if (transpose_reader->scores != NULL)
    {
        bytes = evoke_u64_saturating_add(
            bytes,
            (uint64) transpose_reader->score_capacity
        );
    }
    return bytes;
}

static void
evoke_page_query_residual_merge_free(
    evoke_page_query_residual_merge *merge
)
{
    if (merge == NULL)
    {
        return;
    }
    free(merge->heap);
    free(merge->cursors);
    free(merge->terms);
    memset(merge, 0, sizeof(*merge));
}

static uint32
evoke_page_query_residual_merge_document(
    const evoke_page_query_residual_merge *merge,
    uint32 cursor_index
)
{
    const evoke_page_query_cursor *cursor = &merge->cursors[cursor_index];

    return cursor->block.document_slots[cursor->posting_index];
}

static bool
evoke_page_query_residual_merge_precedes(
    const evoke_page_query_residual_merge *merge,
    uint32 left_index,
    uint32 right_index
)
{
    uint32 left_document = evoke_page_query_residual_merge_document(
        merge,
        left_index
    );
    uint32 right_document = evoke_page_query_residual_merge_document(
        merge,
        right_index
    );

    return left_document < right_document ||
        (left_document == right_document && left_index < right_index);
}

static void
evoke_page_query_residual_merge_heap_push(
    evoke_page_query_residual_merge *merge,
    uint32 cursor_index
)
{
    size_t slot = merge->heap_count++;

    while (slot > 0)
    {
        size_t parent = (slot - 1U) / 2U;
        uint32 parent_index = merge->heap[parent];

        if (!evoke_page_query_residual_merge_precedes(
                merge,
                cursor_index,
                parent_index))
        {
            break;
        }
        merge->heap[slot] = parent_index;
        slot = parent;
    }
    merge->heap[slot] = cursor_index;
}

static uint32
evoke_page_query_residual_merge_heap_pop(
    evoke_page_query_residual_merge *merge
)
{
    uint32 result = merge->heap[0];
    uint32 replacement;
    size_t slot = 0;

    merge->heap_count--;
    if (merge->heap_count == 0)
    {
        return result;
    }
    replacement = merge->heap[merge->heap_count];
    while (true)
    {
        size_t left = slot * 2U + 1U;
        size_t right = left + 1U;
        size_t child;

        if (left >= merge->heap_count)
        {
            break;
        }
        child = left;
        if (right < merge->heap_count &&
            evoke_page_query_residual_merge_precedes(
                merge,
                merge->heap[right],
                merge->heap[left]))
        {
            child = right;
        }
        if (!evoke_page_query_residual_merge_precedes(
                merge,
                merge->heap[child],
                replacement))
        {
            break;
        }
        merge->heap[slot] = merge->heap[child];
        slot = child;
    }
    merge->heap[slot] = replacement;
    return result;
}

static evoke_status
evoke_page_query_residual_posting_contribution(
    const evoke_segment_query_context *context,
    evoke_posting_extent_kind run_kind,
    float query_weight,
    double idf,
    uint32 document,
    const evoke_posting_value *value,
    double average_document_length,
    double *score_out
)
{
    double score;

    if (run_kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
        run_kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
    {
        score = (double) query_weight * value->impact;
    }
    else if (run_kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
    {
        double tfc = evoke_score_tfc(
            context->query_contract.params.method,
            value->term_frequency,
            context->resident_document_lengths[document],
            average_document_length,
            context->query_contract.params.k1,
            context->query_contract.params.b,
            context->query_contract.params.delta
        );

        score = (double) query_weight * idf * tfc;
    }
    else
    {
        return EVOKE_ERR_FORMAT;
    }
    if (!isfinite(score) || score > FLT_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    *score_out = score;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_residual_candidate_contribution(
    const evoke_segment_query_context *context,
    const evoke_page_query_term *term,
    const evoke_page_query_cursor *cursor,
    double average_document_length,
    double *score_out
)
{
    const evoke_segment_query_run *run =
        &term->plan.runs[cursor->run_index];

    return evoke_page_query_residual_posting_contribution(
        context,
        run->kind,
        term->query_weight,
        term->idf,
        cursor->block.document_slots[cursor->posting_index],
        &cursor->block.values[cursor->posting_index],
        average_document_length,
        score_out
    );
}

static evoke_status
evoke_page_query_accumulate_residual_candidates_dense(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_accelerator_term *accelerator_terms,
    size_t accelerator_term_count,
    double average_document_length,
    size_t candidate_capacity,
    evoke_page_query_accelerator_cleanup *resources,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats,
    uint64 scratch_peak_bytes,
    uint64 *scratch_peak_bytes_out,
    bool *available_out
)
{
    evoke_topk_accumulator *topk;
    uint64 document_count;
    uint64 documents_scanned = 0;
    evoke_status status = EVOKE_OK;

    if (resources == NULL || available_out == NULL ||
        scratch_peak_bytes_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    topk = &resources->exact_residual_topk;
    document_count = resources->directory.document_count;
    *scratch_peak_bytes_out = scratch_peak_bytes;
    *available_out = false;
    resources->residual_candidate_scores = calloc(
        (size_t) document_count,
        sizeof(*resources->residual_candidate_scores)
    );
    resources->document_window = malloc(
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW *
            sizeof(*resources->document_window)
    );
    resources->value_window = malloc(
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW *
            sizeof(*resources->value_window)
    );
    if (resources->residual_candidate_scores == NULL ||
        resources->document_window == NULL ||
        resources->value_window == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    status = evoke_topk_accumulator_init(topk, candidate_capacity);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    for (size_t term_index = 0;
         term_index < accelerator_term_count;
         term_index++)
    {
        const evoke_page_query_accelerator_term *term =
            &accelerator_terms[term_index];
        double residual_idf = 0.0;

        if (term->accelerated)
        {
            continue;
        }
        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            if (term->plan.runs[run_index].kind ==
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                residual_idf = evoke_score_idf(
                    context->query_contract.params.idf_method,
                    term->plan.raw_document_frequency,
                    context->manifest.visible_document_count
                );
                if (!isfinite(residual_idf))
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
                break;
            }
        }
        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            evoke_segment_query_posting_cursor cursor;
            const evoke_segment_query_run *run =
                &term->plan.runs[run_index];

            evoke_segment_query_posting_cursor_init(&cursor);
            while (cursor.next_posting_index < run->posting_count)
            {
                uint32 loaded =
                    evoke_segment_pages_load_query_term_posting_stream_window(
                        index_relation,
                        context,
                        &term->plan,
                        run_index,
                        &cursor,
                        resources->document_window,
                        resources->value_window,
                        EVOKE_SEGMENT_QUERY_POSTING_WINDOW
                    );

                stats->accelerator_residual_postings =
                    evoke_u64_saturating_add(
                        stats->accelerator_residual_postings,
                        loaded
                    );
                for (uint32 posting_index = 0;
                     posting_index < loaded;
                     posting_index++)
                {
                    uint32 document =
                        resources->document_window[posting_index];
                    const evoke_posting_value *value =
                        &resources->value_window[posting_index];
                    double contribution;
                    double accumulated;

                    if (document >= document_count)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto cleanup;
                    }
                    status =
                        evoke_page_query_residual_posting_contribution(
                            context,
                            run->kind,
                            term->weight,
                            residual_idf,
                            document,
                            value,
                            average_document_length,
                            &contribution
                        );
                    if (status != EVOKE_OK)
                    {
                        goto cleanup;
                    }
                    if (contribution <= 0.0)
                    {
                        continue;
                    }
                    accumulated =
                        resources->residual_candidate_scores[document] +
                        contribution;
                    if (!isfinite(accumulated) || accumulated > FLT_MAX)
                    {
                        status = EVOKE_ERR_RANGE;
                        goto cleanup;
                    }
                    resources->residual_candidate_scores[document] =
                        (float) accumulated;
                }
                CHECK_FOR_INTERRUPTS();
            }
        }
    }
    for (uint32 document = 0;
         document < (uint32) document_count;
         document++)
    {
        float score = resources->residual_candidate_scores[document];

        if (score > 0.0f)
        {
            status = evoke_topk_accumulator_offer(
                topk,
                score,
                document,
                document
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
        }
        documents_scanned++;
        if ((documents_scanned & UINT64_C(4095)) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
    }
    status = evoke_topk_accumulator_finish(topk, false, result_out);
    if (status == EVOKE_OK)
    {
        *available_out = true;
    }

cleanup:
    if (status != EVOKE_OK)
    {
        evoke_topk_result_free(result_out);
    }
    evoke_topk_accumulator_free(topk);
    free(resources->value_window);
    resources->value_window = NULL;
    free(resources->document_window);
    resources->document_window = NULL;
    free(resources->residual_candidate_scores);
    resources->residual_candidate_scores = NULL;
    return status;
}

static bool
evoke_page_query_prefer_dense_residual_accumulation(
    const evoke_page_query_accelerator_term *terms,
    size_t term_count,
    uint64 document_count,
    size_t candidate_capacity,
    uint64 *scratch_peak_bytes_out
)
{
    uint64 posting_count = 0;
    uint64 cursor_count = 0;
    uint64 heap_levels = 0;
    uint64 heap_span;
    uint64 dense_work;
    uint64 merge_work;
    uint64 scratch_bytes = 0;

    if (terms == NULL || scratch_peak_bytes_out == NULL ||
        document_count == 0 || document_count > UINT32_MAX)
    {
        return false;
    }
    for (size_t term_index = 0; term_index < term_count; term_index++)
    {
        if (terms[term_index].accelerated)
        {
            continue;
        }
        for (uint32 run_index = 0;
             run_index < terms[term_index].plan.run_count;
             run_index++)
        {
            if (terms[term_index]
                    .plan.runs[run_index]
                    .posting_count > 0)
            {
                cursor_count = evoke_u64_saturating_add(
                    cursor_count,
                    1
                );
            }
            posting_count = evoke_u64_saturating_add(
                posting_count,
                terms[term_index].plan.runs[run_index].posting_count
            );
        }
    }
    scratch_bytes = evoke_page_query_memory_add(
        scratch_bytes,
        document_count,
        sizeof(float)
    );
    scratch_bytes = evoke_page_query_memory_add(
        scratch_bytes,
        candidate_capacity,
        sizeof(evoke_topk_item) + sizeof(uint32) + sizeof(float)
    );
    scratch_bytes = evoke_page_query_memory_add(
        scratch_bytes,
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW,
        sizeof(uint32) + sizeof(evoke_posting_value)
    );
    *scratch_peak_bytes_out = scratch_bytes;
    if (scratch_bytes > EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT ||
        cursor_count < 2 || posting_count == 0)
    {
        return false;
    }
    heap_span = cursor_count;
    while (heap_span > 1)
    {
        heap_span = (heap_span + 1) >> 1;
        heap_levels++;
    }
    dense_work = evoke_u64_saturating_add(
        posting_count,
        document_count
    );
    merge_work = evoke_u64_saturating_mul(
        posting_count,
        heap_levels
    );
    return dense_work <= merge_work;
}

static evoke_status
evoke_page_query_accumulate_residual_candidates_merge(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_accelerator_term *accelerator_terms,
    size_t accelerator_term_count,
    double average_document_length,
    size_t candidate_capacity,
    evoke_page_query_accelerator_cleanup *resources,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats,
    uint64 *scratch_peak_bytes_out,
    bool *available_out
)
{
    evoke_page_query_residual_merge *merge;
    evoke_topk_accumulator *topk;
    size_t residual_term_count = 0;
    size_t residual_cursor_count = 0;
    size_t term_index = 0;
    size_t cursor_index = 0;
    uint64 documents_processed = 0;
    evoke_status status = EVOKE_OK;

    if (resources == NULL || available_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    merge = &resources->exact_residual_merge;
    topk = &resources->exact_residual_topk;
    *scratch_peak_bytes_out = 0;
    *available_out = false;
    for (size_t index = 0; index < accelerator_term_count; index++)
    {
        if (accelerator_terms[index].accelerated)
        {
            continue;
        }
        if (residual_cursor_count > SIZE_MAX -
                accelerator_terms[index].plan.run_count)
        {
            return EVOKE_ERR_RANGE;
        }
        residual_cursor_count += accelerator_terms[index].plan.run_count;
        residual_term_count++;
    }
    if (residual_cursor_count > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    merge->term_count = residual_term_count;
    merge->cursor_count = residual_cursor_count;
    *scratch_peak_bytes_out = evoke_page_query_memory_add(
        *scratch_peak_bytes_out,
        merge->term_count,
        sizeof(*merge->terms)
    );
    *scratch_peak_bytes_out = evoke_page_query_memory_add(
        *scratch_peak_bytes_out,
        merge->cursor_count,
        sizeof(*merge->cursors)
    );
    *scratch_peak_bytes_out = evoke_page_query_memory_add(
        *scratch_peak_bytes_out,
        merge->cursor_count,
        sizeof(*merge->heap)
    );
    *scratch_peak_bytes_out = evoke_page_query_memory_add(
        *scratch_peak_bytes_out,
        candidate_capacity,
        sizeof(*topk->heap)
    );
    if (*scratch_peak_bytes_out > EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT)
    {
        return EVOKE_OK;
    }
    if (residual_term_count > 0)
    {
        merge->terms = calloc(residual_term_count, sizeof(*merge->terms));
    }
    if (residual_cursor_count > 0)
    {
        merge->cursors = calloc(
            residual_cursor_count,
            sizeof(*merge->cursors)
        );
        merge->heap = calloc(residual_cursor_count, sizeof(*merge->heap));
    }
    if ((residual_term_count > 0 && merge->terms == NULL) ||
        (residual_cursor_count > 0 &&
         (merge->cursors == NULL || merge->heap == NULL)))
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    status = evoke_topk_accumulator_init(topk, candidate_capacity);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    for (size_t index = 0; index < accelerator_term_count; index++)
    {
        evoke_page_query_term *term;
        bool lexical_neutral = false;

        if (accelerator_terms[index].accelerated)
        {
            continue;
        }
        term = &merge->terms[term_index];
        term->plan = accelerator_terms[index].plan;
        term->query_weight = accelerator_terms[index].weight;
        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            evoke_page_query_cursor *cursor =
                &merge->cursors[cursor_index];

            lexical_neutral = lexical_neutral ||
                term->plan.runs[run_index].kind ==
                    EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
            cursor->term_index = (uint32) term_index;
            cursor->run_index = run_index;
            status = evoke_page_query_cursor_load_block(
                index_relation,
                context,
                term,
                cursor,
                stats
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            if (!cursor->exhausted)
            {
                evoke_page_query_residual_merge_heap_push(
                    merge,
                    (uint32) cursor_index
                );
            }
            cursor_index++;
        }
        if (lexical_neutral)
        {
            term->idf = evoke_score_idf(
                context->query_contract.params.idf_method,
                term->plan.raw_document_frequency,
                context->manifest.visible_document_count
            );
            if (!isfinite(term->idf))
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
        }
        term_index++;
    }
    while (merge->heap_count > 0)
    {
        uint32 document = evoke_page_query_residual_merge_document(
            merge,
            merge->heap[0]
        );
        float accumulated_score = 0.0f;

        do
        {
            uint32 current_cursor_index =
                evoke_page_query_residual_merge_heap_pop(merge);
            evoke_page_query_cursor *cursor =
                &merge->cursors[current_cursor_index];
            const evoke_page_query_term *term =
                &merge->terms[cursor->term_index];
            double contribution;

            status = evoke_page_query_residual_candidate_contribution(
                context,
                term,
                cursor,
                average_document_length,
                &contribution
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            stats->accelerator_residual_postings =
                evoke_u64_saturating_add(
                    stats->accelerator_residual_postings,
                    1
                );
            if (contribution > 0.0)
            {
                double accumulated =
                    (double) accumulated_score + contribution;

                if (!isfinite(accumulated) || accumulated > FLT_MAX)
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
                accumulated_score = (float) accumulated;
            }
            status = evoke_page_query_cursor_advance(
                index_relation,
                context,
                term,
                cursor,
                stats
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            if (!cursor->exhausted)
            {
                evoke_page_query_residual_merge_heap_push(
                    merge,
                    current_cursor_index
                );
            }
        } while (merge->heap_count > 0 &&
                 evoke_page_query_residual_merge_document(
                     merge,
                     merge->heap[0]
                 ) == document);
        if (accumulated_score > 0.0f)
        {
            status = evoke_topk_accumulator_offer(
                topk,
                accumulated_score,
                document,
                document
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
        }
        documents_processed++;
        if ((documents_processed & UINT64_C(4095)) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
    }
    status = evoke_topk_accumulator_finish(topk, false, result_out);
    if (status == EVOKE_OK)
    {
        *available_out = true;
    }

cleanup:
    if (status != EVOKE_OK)
    {
        evoke_topk_result_free(result_out);
    }
    evoke_topk_accumulator_free(topk);
    evoke_page_query_residual_merge_free(merge);
    return status;
}

static evoke_status
evoke_page_query_accumulate_residual_candidates_exact(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_accelerator_term *accelerator_terms,
    size_t accelerator_term_count,
    double average_document_length,
    size_t candidate_capacity,
    evoke_page_query_accelerator_cleanup *resources,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats,
    uint64 *scratch_peak_bytes_out,
    bool *available_out
)
{
    uint64 dense_scratch_bytes = 0;

    if (evoke_page_query_prefer_dense_residual_accumulation(
            accelerator_terms,
            accelerator_term_count,
            resources->directory.document_count,
            candidate_capacity,
            &dense_scratch_bytes))
    {
        stats->semantic_accelerator_residual_dense_accumulation = true;
        return evoke_page_query_accumulate_residual_candidates_dense(
            index_relation,
            context,
            accelerator_terms,
            accelerator_term_count,
            average_document_length,
            candidate_capacity,
            resources,
            result_out,
            stats,
            dense_scratch_bytes,
            scratch_peak_bytes_out,
            available_out
        );
    }
    return evoke_page_query_accumulate_residual_candidates_merge(
        index_relation,
        context,
        accelerator_terms,
        accelerator_term_count,
        average_document_length,
        candidate_capacity,
        resources,
        result_out,
        stats,
        scratch_peak_bytes_out,
        available_out
    );
}

#define EVOKE_PAGE_QUERY_EXACT_ROWS_PER_FORWARD_CHUNK UINT32_C(1)
#define EVOKE_PAGE_QUERY_EXACT_DATA_ROWS_PER_FORWARD_CHUNK UINT32_C(256)
#define EVOKE_PAGE_QUERY_FILTER_TRANSPOSE_LANE_BUDGET \
    EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT
#define EVOKE_PAGE_QUERY_FILTER_TRANSPOSE_FIXED_SCRATCH \
    (UINT64_C(512) * UINT64_C(1024))
#define EVOKE_PAGE_QUERY_FILTER_PREFETCH_MIN_DOCUMENTS UINT64_C(1024)
#define EVOKE_PAGE_QUERY_FILTER_PREFETCH_BATCH_DOCUMENTS UINT32_C(1024)
static bool
evoke_page_query_prefer_direct_forward_rows(
    uint64 maximum_forward_document_count,
    uint64 estimated_direct_bytes,
    uint64 estimated_transpose_bytes,
    uint64 *transpose_lane_bytes_out,
    bool *transpose_budget_exceeded_out
)
{
    uint64 transpose_lane_bytes;

    if (transpose_lane_bytes_out == NULL ||
        transpose_budget_exceeded_out == NULL ||
        maximum_forward_document_count == 0 ||
        estimated_direct_bytes == 0 || estimated_transpose_bytes == 0)
    {
        return false;
    }
    transpose_lane_bytes = evoke_u64_saturating_add(
        evoke_u64_saturating_mul(
            maximum_forward_document_count,
            sizeof(float) + sizeof(double) + sizeof(float) + sizeof(int8)
        ),
        EVOKE_PAGE_QUERY_FILTER_TRANSPOSE_FIXED_SCRATCH
    );
    *transpose_lane_bytes_out = transpose_lane_bytes;
    *transpose_budget_exceeded_out =
        transpose_lane_bytes >
            EVOKE_PAGE_QUERY_FILTER_TRANSPOSE_LANE_BUDGET;

    /* Compare publication-derived physical bytes, not a global cost factor. */
    return estimated_direct_bytes <= estimated_transpose_bytes ||
        *transpose_budget_exceeded_out;
}

static bool
evoke_page_query_prefer_bounded_forward_rows(
    bool prefer_direct_rows,
    uint64 estimated_transpose_bytes,
    uint64 estimated_bound_bytes
)
{
    if (prefer_direct_rows || estimated_transpose_bytes == 0 ||
        estimated_bound_bytes == 0 ||
        estimated_bound_bytes == UINT64_MAX)
    {
        return false;
    }

    /*
     * Bound probing has a fixed publication/read cost and can lose on compact
     * roots. Admit it only when the normal transpose stream exceeds the same
     * statement working-set budget used by the forward executors and the
     * published bound stream is physically smaller. This keeps the decision
     * independent of corpus identity, filter density, and fitted constants.
     */
    return estimated_transpose_bytes >
            EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT &&
        estimated_bound_bytes < estimated_transpose_bytes;
}

static uint64
evoke_page_query_estimate_direct_forward_work(
    const evoke_semantic_accelerator_forward_entry *entry,
    uint32 allowed_document_count
)
{
    uint64 scaled_postings;
    uint64 estimate;

    if (entry == NULL || entry->document_count == 0 ||
        allowed_document_count == 0 ||
        allowed_document_count > entry->document_count)
    {
        return UINT64_MAX;
    }
    scaled_postings = evoke_u64_saturating_mul(
        entry->posting_count,
        allowed_document_count
    );
    if (scaled_postings == UINT64_MAX)
    {
        return UINT64_MAX;
    }
    estimate = scaled_postings / entry->document_count;
    if (scaled_postings % entry->document_count != 0)
    {
        estimate++;
    }

    /* Every selected row has fixed lookup/merge work even when it is empty. */
    return Max(estimate, (uint64) allowed_document_count);
}

static uint64
evoke_page_query_scale_bytes(
    uint64 total_bytes,
    uint64 selected,
    uint64 total
)
{
    uint64 scaled;

    if (selected == 0 || total == 0 || selected > total)
    {
        return UINT64_MAX;
    }
    scaled = evoke_u64_saturating_mul(total_bytes, selected);
    if (scaled == UINT64_MAX)
    {
        return UINT64_MAX;
    }
    return scaled / total + (scaled % total != 0);
}

static uint64
evoke_page_query_estimate_direct_forward_bytes(
    const evoke_semantic_accelerator_forward_entry *entry,
    uint64 row_data_bytes,
    uint32 allowed_document_count
)
{
    uint64 fixed_bytes;
    uint64 selected_row_bytes;

    if (entry == NULL || entry->document_count == 0 ||
        allowed_document_count == 0 ||
        allowed_document_count > entry->document_count)
    {
        return UINT64_MAX;
    }
    fixed_bytes = evoke_u64_saturating_add(
        EVOKE_SEMANTIC_FORWARD_HEADER_BYTES,
        ((uint64) entry->document_count + 1U) * sizeof(uint32)
    );
    selected_row_bytes = evoke_page_query_scale_bytes(
        row_data_bytes,
        allowed_document_count,
        entry->document_count
    );
    return evoke_u64_saturating_add(fixed_bytes, selected_row_bytes);
}

static uint64
evoke_page_query_estimate_transpose_chunk_bytes(
    const evoke_semantic_accelerator_directory *directory,
    uint32 chunk_index,
    const uint32 *query_ids,
    size_t query_count
)
{
    const evoke_semantic_accelerator_forward_entry *entry;
    uint64 estimate;

    if (directory == NULL || query_ids == NULL || query_count == 0 ||
        chunk_index >= directory->forward_chunk_count ||
        directory->document_count == 0)
    {
        return UINT64_MAX;
    }
    entry = &directory->forward_chunks[chunk_index];
    estimate = directory->forward_transpose_fixed_bytes[chunk_index];
    for (size_t query_index = 0;
         query_index < query_count;
         query_index++)
    {
        uint32 term_id = query_ids[query_index];
        uint64 term_bytes;

        if (term_id >= directory->vocab_size)
        {
            return UINT64_MAX;
        }
        term_bytes = evoke_page_query_scale_bytes(
            directory->forward_term_bytes[term_id],
            entry->document_count,
            directory->document_count
        );
        estimate = evoke_u64_saturating_add(estimate, term_bytes);
    }
    return estimate;
}

static uint64
evoke_page_query_estimate_forward_bound_bytes(
    const evoke_semantic_accelerator_directory *directory,
    const uint32 *query_ids,
    size_t query_count
)
{
    uint64 bytes = 0;
    uint32 previous_shard = UINT32_MAX;

    if (directory == NULL || query_ids == NULL || query_count == 0 ||
        directory->forward_bound_term_bytes == NULL)
    {
        return UINT64_MAX;
    }
    for (size_t query = 0; query < query_count; query++)
    {
        uint32 term_id = query_ids[query];
        uint32 shard;

        if (term_id >= directory->vocab_size)
        {
            return UINT64_MAX;
        }
        shard = term_id /
            EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD;
        if (shard != previous_shard)
        {
            uint32 first_term = shard *
                EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD;
            uint32 term_count = Min(
                EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD,
                directory->vocab_size - first_term
            );

            bytes = evoke_u64_saturating_add(
                bytes,
                EVOKE_SEMANTIC_FORWARD_BOUND_HEADER_SIZE +
                    ((uint64) term_count + 1U) * sizeof(uint64)
            );
            previous_shard = shard;
        }
        bytes = evoke_u64_saturating_add(
            bytes,
            directory->forward_bound_term_bytes[term_id]
        );
    }
    return bytes;
}

static evoke_status
evoke_page_query_count_allowed_forward_chunks(
    const evoke_page_query_filter *filter,
    const evoke_semantic_accelerator_directory *directory,
    uint32 **counts_out,
    evoke_page_query_stats *stats
)
{
    uint32 *counts;
    uint64 bitmap_byte_count;
    uint64 previous_block = UINT64_MAX;
    uint64 previous_b64_range = UINT64_MAX;
    uint64 previous_b512_range = UINT64_MAX;
    uint64 previous_b4096_range = UINT64_MAX;

    if (filter == NULL || directory == NULL || counts_out == NULL ||
        filter->allowed_document_bitmap == NULL ||
        directory->forward_chunk_count == 0 ||
        directory->forward_document_shift == 0 ||
        directory->forward_document_shift >= 32)
    {
        return EVOKE_ERR_INVALID;
    }
    *counts_out = NULL;
    counts = calloc(
        directory->forward_chunk_count,
        sizeof(*counts)
    );
    if (counts == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    bitmap_byte_count =
        (Min(
            filter->document_slot_count,
            (uint64) directory->document_count
        ) + UINT64_C(7)) >> 3;
    for (uint64 byte_index = 0;
         byte_index < bitmap_byte_count;
         byte_index++)
    {
        uint8 candidates = filter->allowed_document_bitmap[byte_index];

        if (candidates != 0 && stats != NULL)
        {
            uint64 b64_range = byte_index >> 3;
            uint64 b512_range = byte_index >> 6;
            uint64 b4096_range = byte_index >> 9;

            if (stats->filtered_forward_allowed_b8_blocks == 0)
            {
                stats->filtered_forward_allowed_b8_first = byte_index;
            }
            stats->filtered_forward_allowed_b8_last = byte_index;
            stats->filtered_forward_allowed_b8_blocks++;
            if (previous_block == UINT64_MAX ||
                byte_index != previous_block + UINT64_C(1))
            {
                stats->filtered_forward_allowed_b8_runs++;
            }
            if (b64_range != previous_b64_range)
            {
                stats->filtered_forward_allowed_b64_ranges++;
                previous_b64_range = b64_range;
            }
            if (b512_range != previous_b512_range)
            {
                stats->filtered_forward_allowed_b512_ranges++;
                previous_b512_range = b512_range;
            }
            if (b4096_range != previous_b4096_range)
            {
                stats->filtered_forward_allowed_b4096_ranges++;
                previous_b4096_range = b4096_range;
            }
            previous_block = byte_index;
        }
        while (candidates != 0)
        {
            unsigned bit_index = (unsigned) __builtin_ctz(
                (unsigned) candidates
            );
            uint64 document_slot =
                byte_index * UINT64_C(8) + bit_index;
            uint64 chunk_index;

            candidates &= (uint8) (candidates - UINT8_C(1));
            if (document_slot >= directory->document_count)
            {
                continue;
            }
            chunk_index = document_slot >>
                directory->forward_document_shift;
            if (chunk_index >= directory->forward_chunk_count ||
                counts[chunk_index] == UINT32_MAX)
            {
                free(counts);
                return EVOKE_ERR_FORMAT;
            }
            counts[chunk_index]++;
        }
    }
    if (stats != NULL && stats->filtered_forward_allowed_b8_blocks != 0)
    {
        stats->filtered_forward_allowed_b8_span =
            stats->filtered_forward_allowed_b8_last -
            stats->filtered_forward_allowed_b8_first + UINT64_C(1);
    }
    *counts_out = counts;
    return EVOKE_OK;
}

static int
evoke_page_query_compare_accelerator_term(
    const void *left,
    const void *right
)
{
    const evoke_page_query_accelerator_term *a = left;
    const evoke_page_query_accelerator_term *b = right;

    return (a->term_id > b->term_id) - (a->term_id < b->term_id);
}

static int
evoke_page_query_compare_document_id(const void *left, const void *right)
{
    const uint32 a = *(const uint32 *) left;
    const uint32 b = *(const uint32 *) right;

    return (a > b) - (a < b);
}

static evoke_status
evoke_page_query_score_forward_document(
    void *context,
    uint32 document_id,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_count,
    float *score_out
)
{
    evoke_page_query_forward_score_context *score_context = context;
    float score;
    evoke_status status;

    (void) query_ids;
    (void) query_weights;
    (void) query_count;
    if (score_context == NULL || score_out == NULL ||
        document_id >= score_context->directory->document_count ||
        score_context->query_ids == NULL ||
        score_context->query_weights == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (score_context->candidate_filter != NULL &&
        !evoke_page_query_document_allowed(
            score_context->candidate_filter,
            document_id))
    {
        *score_out = 0.0f;
        return EVOKE_OK;
    }
    if (score_context->direct_rows)
    {
        uint64 postings_examined = 0;
        uint64 bytes_read = 0;

        if (score_context->allowed_forward_chunk_counts != NULL)
        {
            uint32 chunk_index = document_id >>
                score_context->directory->forward_document_shift;
            uint32 allowed_count;

            if (chunk_index >=
                score_context->directory->forward_chunk_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            allowed_count =
                score_context->allowed_forward_chunk_counts[chunk_index];
            evoke_segment_forward_row_reader_set_exact_reads(
                &score_context->row_reader,
                allowed_count <=
                    EVOKE_PAGE_QUERY_EXACT_ROWS_PER_FORWARD_CHUNK
            );
            evoke_segment_forward_row_reader_set_exact_data_reads(
                &score_context->row_reader,
                allowed_count <=
                    EVOKE_PAGE_QUERY_EXACT_DATA_ROWS_PER_FORWARD_CHUNK
            );
        }

        evoke_segment_pages_score_semantic_accelerator_forward_row(
            score_context->index_relation,
            score_context->query_context,
            score_context->directory,
            &score_context->row_reader,
            document_id,
            score_context->query_ids,
            score_context->query_weights,
            score_context->query_count,
            score_context->query_is_sorted,
            &score,
            &postings_examined,
            &bytes_read
        );
        score_context->row_reads++;
        if (UINT64_MAX - score_context->postings_examined <
            postings_examined)
        {
            return EVOKE_ERR_RANGE;
        }
        score_context->postings_examined += postings_examined;
        if (UINT64_MAX - score_context->bytes_read < bytes_read)
        {
            return EVOKE_ERR_RANGE;
        }
        score_context->bytes_read += bytes_read;
        status = EVOKE_OK;
    }
    else if (score_context->chunk.document_count == 0 ||
        document_id < score_context->chunk.first_document ||
        document_id >= score_context->chunk.first_document +
            score_context->chunk.document_count)
    {
        evoke_semantic_forward_chunk_free(&score_context->chunk);
        evoke_segment_pages_load_semantic_accelerator_forward(
            score_context->index_relation,
            &score_context->query_context->root,
            &score_context->query_context->manifest,
            score_context->directory,
            document_id,
            &score_context->chunk
        );
        score_context->chunk_reads++;
    }
    if (!score_context->direct_rows && score_context->query_is_sorted)
    {
        status = evoke_semantic_forward_chunk_score_sorted(
            &score_context->chunk,
            document_id,
            score_context->query_ids,
            score_context->query_weights,
            score_context->query_count,
            &score
        );
    }
    else if (!score_context->direct_rows)
    {
        status = evoke_semantic_forward_chunk_score(
            &score_context->chunk,
            document_id,
            score_context->query_ids,
            score_context->query_weights,
            score_context->query_count,
            &score
        );
    }
    if (status != EVOKE_OK)
    {
        return status;
    }
    if ((double) score + score_context->base_score > FLT_MAX ||
        (double) score + score_context->base_score < -FLT_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    *score_out = score + score_context->base_score;
    if (score_context->scored_documents != NULL)
    {
        score_context->scored_documents[document_id >> 3] |=
            (uint8) (UINT8_C(1) << (document_id & 7U));
    }
    return EVOKE_OK;
}

static bool
evoke_page_query_accelerator_result_contains(
    const evoke_topk_result *result,
    uint32 document_id
)
{
    for (size_t index = 0; index < result->len; index++)
    {
        if (result->doc_ids[index] == document_id)
        {
            return true;
        }
    }
    return false;
}

static evoke_status
evoke_page_query_accumulate_forward_bound(
    void *context,
    uint32 block_id,
    float maximum
)
{
    evoke_page_query_forward_bound_accumulation *accumulation = context;
    double contribution;
    double upper_bound;

    if (accumulation == NULL || accumulation->upper_bounds == NULL ||
        accumulation->allowed_blocks == NULL ||
        block_id >= accumulation->block_count ||
        !isfinite(maximum) || maximum <= 0.0f ||
        !isfinite(accumulation->query_weight) ||
        accumulation->query_weight <= 0.0f)
    {
        return EVOKE_ERR_INVALID;
    }
    if (accumulation->allowed_blocks[block_id] == 0)
    {
        return EVOKE_OK;
    }
    contribution = (double) maximum * accumulation->query_weight;
    if (!isfinite(contribution))
    {
        return EVOKE_ERR_RANGE;
    }
    contribution = nextafter(contribution, INFINITY);
    upper_bound = accumulation->upper_bounds[block_id] + contribution;
    if (!isfinite(upper_bound))
    {
        return EVOKE_ERR_RANGE;
    }
    accumulation->upper_bounds[block_id] = nextafter(
        upper_bound,
        INFINITY
    );
    return EVOKE_OK;
}

static int
evoke_page_query_compare_forward_bound_blocks(
    const void *left,
    const void *right
)
{
    const evoke_page_query_forward_bound_block *a = left;
    const evoke_page_query_forward_bound_block *b = right;

    if (a->upper_bound > b->upper_bound)
    {
        return -1;
    }
    if (a->upper_bound < b->upper_bound)
    {
        return 1;
    }
    return a->block_id < b->block_id
        ? -1
        : a->block_id > b->block_id ? 1 : 0;
}

static evoke_status
evoke_page_query_score_filtered_forward_bounded(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    const evoke_page_query_filter *filter,
    size_t k,
    evoke_page_query_forward_score_context *score_context,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats,
    uint64 *documents_scored_out,
    bool *available_out
)
{
    evoke_segment_forward_bound_reader bound_reader;
    evoke_page_query_forward_bound_accumulation accumulation;
    evoke_page_query_forward_bound_block *blocks = NULL;
    double *upper_bounds = NULL;
    uint64 bitmap_byte_count;
    uint64 block_count_u64;
    uint64 allowed_block_count = 0;
    uint64 memory_bytes;
    uint64 documents_scored = 0;
    uint64 blocks_scored = 0;
    size_t candidate_count = 0;
    evoke_status status = EVOKE_OK;

    if (index_relation == NULL || context == NULL || directory == NULL ||
        filter == NULL || score_context == NULL || result_out == NULL ||
        stats == NULL || documents_scored_out == NULL ||
        available_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *documents_scored_out = 0;
    *available_out = false;
    if (!filter->live_membership_verified ||
        filter->allowed_document_bitmap == NULL ||
        score_context->query_ids == NULL ||
        score_context->query_weights == NULL ||
        score_context->query_count == 0 ||
        !score_context->query_is_sorted ||
        directory->forward_row_offsets == NULL ||
        !evoke_semantic_accelerator_directory_has_complete_forward_bounds(
            directory))
    {
        return EVOKE_OK;
    }
    block_count_u64 =
        ((uint64) directory->document_count + UINT64_C(7)) >> 3;
    bitmap_byte_count =
        (Min(
            filter->document_slot_count,
            (uint64) directory->document_count
        ) + UINT64_C(7)) >> 3;
    if (block_count_u64 == 0 || block_count_u64 > UINT32_MAX ||
        bitmap_byte_count > block_count_u64)
    {
        return EVOKE_ERR_RANGE;
    }
    for (uint64 block = 0; block < bitmap_byte_count; block++)
    {
        if (filter->allowed_document_bitmap[block] != 0)
        {
            allowed_block_count++;
        }
    }
    memory_bytes = evoke_u64_saturating_add(
        evoke_u64_saturating_mul(
            block_count_u64,
            sizeof(*upper_bounds)
        ),
        evoke_u64_saturating_mul(
            allowed_block_count,
            sizeof(*blocks)
        )
    );
    if (memory_bytes > EVOKE_PAGE_QUERY_TERM_AT_A_TIME_LIMIT ||
        block_count_u64 > MaxAllocSize / sizeof(*upper_bounds) ||
        allowed_block_count > MaxAllocSize / sizeof(*blocks))
    {
        stats->filtered_forward_bound_budget_exceeded = true;
        return EVOKE_OK;
    }
    upper_bounds = palloc0(
        (Size) block_count_u64 * sizeof(*upper_bounds)
    );
    if (allowed_block_count > 0)
    {
        blocks = palloc(
            (Size) allowed_block_count * sizeof(*blocks)
        );
    }
    score_context->scratch_extra_peak_bytes = Max(
        score_context->scratch_extra_peak_bytes,
        memory_bytes
    );
    evoke_segment_forward_bound_reader_init(&bound_reader);
    memset(&accumulation, 0, sizeof(accumulation));
    accumulation.upper_bounds = upper_bounds;
    accumulation.allowed_blocks = filter->allowed_document_bitmap;
    accumulation.block_count = (uint32) block_count_u64;
    for (size_t query = 0; query < score_context->query_count; query++)
    {
        uint32 entry_count = 0;
        uint64 bytes_read = 0;

        accumulation.query_weight = score_context->query_weights[query];
        evoke_segment_pages_visit_semantic_accelerator_forward_bound(
            index_relation,
            context,
            directory,
            &bound_reader,
            score_context->query_ids[query],
            evoke_page_query_accumulate_forward_bound,
            &accumulation,
            &entry_count,
            &bytes_read
        );
        stats->filtered_forward_bound_entries = evoke_u64_saturating_add(
            stats->filtered_forward_bound_entries,
            entry_count
        );
        stats->filtered_forward_bound_bytes = evoke_u64_saturating_add(
            stats->filtered_forward_bound_bytes,
            bytes_read
        );
        score_context->bytes_read = evoke_u64_saturating_add(
            score_context->bytes_read,
            bytes_read
        );
    }
    for (uint64 block = 0; block < bitmap_byte_count; block++)
    {
        double complete_upper_bound;
        float upper_bound;

        if (filter->allowed_document_bitmap[block] == 0)
        {
            continue;
        }
        complete_upper_bound = upper_bounds[block] +
            (double) score_context->base_score;
        if (!isfinite(complete_upper_bound))
        {
            status = EVOKE_ERR_RANGE;
            goto done;
        }
        if (complete_upper_bound <= 0.0)
        {
            continue;
        }
        upper_bound = complete_upper_bound >= FLT_MAX
            ? FLT_MAX
            : (float) complete_upper_bound;
        if ((double) upper_bound < complete_upper_bound)
        {
            upper_bound = nextafterf(upper_bound, INFINITY);
        }
        blocks[candidate_count].upper_bound = upper_bound;
        blocks[candidate_count].block_id = (uint32) block;
        candidate_count++;
    }
    stats->filtered_forward_bound_blocks_considered = candidate_count;
    if (candidate_count > 1)
    {
        qsort(
            blocks,
            candidate_count,
            sizeof(*blocks),
            evoke_page_query_compare_forward_bound_blocks
        );
    }
    status = evoke_topk_accumulator_init(
        &score_context->filtered_topk,
        k
    );
    if (status != EVOKE_OK)
    {
        goto done;
    }
    for (size_t candidate = 0; candidate < candidate_count; candidate++)
    {
        uint32 block_id = blocks[candidate].block_id;
        uint8 allowed = filter->allowed_document_bitmap[block_id];
        float block_scores[8];
        uint32 block_documents_scored = 0;
        uint64 block_postings_examined = 0;
        uint64 block_bytes_read = 0;

        if (score_context->filtered_topk.len ==
                score_context->filtered_topk.capacity &&
            blocks[candidate].upper_bound <
                score_context->filtered_topk.heap[0].score)
        {
            break;
        }
        evoke_segment_pages_score_semantic_accelerator_forward_block(
            index_relation,
            context,
            directory,
            &score_context->row_reader,
            block_id,
            allowed,
            score_context->query_ids,
            score_context->query_weights,
            score_context->query_count,
            block_scores,
            &block_documents_scored,
            &block_postings_examined,
            &block_bytes_read
        );
        score_context->row_reads = evoke_u64_saturating_add(
            score_context->row_reads,
            block_documents_scored
        );
        score_context->postings_examined = evoke_u64_saturating_add(
            score_context->postings_examined,
            block_postings_examined
        );
        score_context->bytes_read = evoke_u64_saturating_add(
            score_context->bytes_read,
            block_bytes_read
        );
        documents_scored = evoke_u64_saturating_add(
            documents_scored,
            block_documents_scored
        );
        for (unsigned bit = 0; bit < 8U; bit++)
        {
            uint32 document;
            float score;

            if ((allowed & (uint8) (UINT8_C(1) << bit)) == 0)
            {
                continue;
            }
            document = block_id * UINT32_C(8) + bit;
            if (document >= directory->document_count)
            {
                continue;
            }
            if ((double) block_scores[bit] + score_context->base_score >
                    FLT_MAX ||
                (double) block_scores[bit] + score_context->base_score <
                    -FLT_MAX)
            {
                status = EVOKE_ERR_RANGE;
                goto done;
            }
            score = block_scores[bit] + score_context->base_score;
            if (score_context->scored_documents != NULL)
            {
                score_context->scored_documents[document >> 3] |=
                    (uint8) (UINT8_C(1) << (document & 7U));
            }
            if (score > 0.0f)
            {
                status = evoke_topk_accumulator_offer(
                    &score_context->filtered_topk,
                    score,
                    document,
                    document
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
            }
        }
        blocks_scored++;
        if ((blocks_scored & EVOKE_PAGE_QUERY_INTERRUPT_MASK) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
    }
    status = evoke_topk_accumulator_finish(
        &score_context->filtered_topk,
        true,
        result_out
    );
    if (status == EVOKE_OK)
    {
        *documents_scored_out = documents_scored;
        *available_out = true;
        stats->filtered_forward_bound_blocks_scored = blocks_scored;
    }

done:
    evoke_topk_accumulator_free(&score_context->filtered_topk);
    if (blocks != NULL)
    {
        pfree(blocks);
    }
    pfree(upper_bounds);
    return status;
}

static evoke_status
evoke_page_query_score_filtered_forward_transposed(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    const evoke_page_query_filter *filter,
    size_t k,
    evoke_page_query_forward_score_context *score_context,
    evoke_topk_result *result_out,
    uint64 *documents_scored_out,
    bool *available_out
)
{
    float *scores = NULL;
    Size scores_capacity;
    Size allowed_bitmap_size;
    uint64 documents_scored = 0;
    bool saw_transposed_chunk = false;
    evoke_status status;

    if (index_relation == NULL || context == NULL || directory == NULL ||
        filter == NULL || score_context == NULL || result_out == NULL ||
        documents_scored_out == NULL || available_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *documents_scored_out = 0;
    *available_out = false;
    if (!filter->live_membership_verified ||
        filter->allowed_document_bitmap == NULL ||
        score_context->allowed_forward_chunk_counts == NULL ||
        directory->forward_document_shift == 0 ||
        directory->forward_document_shift >= 32)
    {
        return EVOKE_OK;
    }
    scores_capacity = (Size) UINT32_C(1) <<
        directory->forward_document_shift;
    if (scores_capacity > MaxAllocSize / sizeof(*scores))
    {
        return EVOKE_ERR_RANGE;
    }
    scores = palloc(scores_capacity * sizeof(*scores));
    score_context->scratch_extra_peak_bytes = Max(
        score_context->scratch_extra_peak_bytes,
        (uint64) scores_capacity * sizeof(*scores)
    );
    allowed_bitmap_size =
        ((Size) directory->document_count + 7U) / 8U;
    status = evoke_topk_accumulator_init(
        &score_context->filtered_topk,
        k
    );
    if (status != EVOKE_OK)
    {
        pfree(scores);
        return status;
    }
    score_context->scratch_extra_peak_bytes = Max(
        score_context->scratch_extra_peak_bytes,
        evoke_u64_saturating_add(
            (uint64) scores_capacity * sizeof(*scores),
            evoke_page_query_topk_accumulator_bytes(
                &score_context->filtered_topk
            )
        )
    );
    for (uint32 chunk_index = 0;
         chunk_index < directory->forward_chunk_count;
         chunk_index++)
    {
        const evoke_semantic_accelerator_forward_entry *entry;
        uint64 postings_examined = 0;
        uint64 bytes_read = 0;
        if (score_context->allowed_forward_chunk_counts[chunk_index] == 0)
        {
            continue;
        }
        entry = &directory->forward_chunks[chunk_index];
        evoke_segment_pages_score_semantic_accelerator_forward_transposed(
            index_relation,
            context,
            directory,
            &score_context->transpose_reader,
            chunk_index,
            filter->allowed_document_bitmap,
            allowed_bitmap_size,
            score_context->query_ids,
            score_context->query_weights,
            score_context->query_count,
            scores,
            scores_capacity,
            &postings_examined,
            &bytes_read
        );
        saw_transposed_chunk = true;
        score_context->chunk_reads++;
        if (UINT64_MAX - score_context->bytes_read < bytes_read)
        {
            status = EVOKE_ERR_RANGE;
            goto done;
        }
        if (UINT64_MAX - score_context->postings_examined <
            postings_examined)
        {
            status = EVOKE_ERR_RANGE;
            goto done;
        }
        score_context->bytes_read += bytes_read;
        score_context->postings_examined += postings_examined;
        for (uint32 local_document = 0;
             local_document < entry->document_count;
             local_document++)
        {
            uint32 document = entry->first_document + local_document;
            double total_score;

            if ((filter->allowed_document_bitmap[document >> 3] &
                 (uint8) (UINT8_C(1) << (document & 7U))) == 0)
            {
                continue;
            }
            total_score = (double) scores[local_document] +
                score_context->base_score;
            if (!isfinite(total_score) || total_score > FLT_MAX ||
                total_score < -FLT_MAX)
            {
                status = EVOKE_ERR_RANGE;
                goto done;
            }
            documents_scored++;
            if (score_context->scored_documents != NULL)
            {
                score_context->scored_documents[document >> 3] |=
                    (uint8) (UINT8_C(1) << (document & 7U));
            }
            if (total_score > 0.0)
            {
                status = evoke_topk_accumulator_offer(
                    &score_context->filtered_topk,
                    (float) total_score,
                    document,
                    document
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
            }
        }
        CHECK_FOR_INTERRUPTS();
    }
    status = evoke_topk_accumulator_finish(
        &score_context->filtered_topk,
        true,
        result_out
    );
    if (status == EVOKE_OK)
    {
        *documents_scored_out = documents_scored;
        *available_out = saw_transposed_chunk;
    }

done:
    if (!*available_out)
    {
        evoke_topk_result_free(result_out);
        score_context->bytes_read = 0;
    }
    evoke_topk_accumulator_free(&score_context->filtered_topk);
    pfree(scores);
    return status;
}

static evoke_status
evoke_page_query_score_filtered_forward(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_semantic_accelerator_directory *directory,
    const evoke_page_query_filter *filter,
    size_t k,
    evoke_page_query_forward_score_context *score_context,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats,
    uint64 *documents_scored_out
)
{
    evoke_segment_query_document_reader document_reader;
    evoke_segment_query_document_block document_block;
    uint64 document_block_count;
    uint64 documents_scored = 0;
    evoke_status status;

    memset(&document_reader, 0, sizeof(document_reader));
    memset(&document_block, 0, sizeof(document_block));
    *documents_scored_out = 0;
    document_block_count =
        ((uint64) directory->document_count +
            EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS - 1U) /
        EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS;
    status = evoke_topk_accumulator_init(
        &score_context->filtered_topk,
        k
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    score_context->scratch_extra_peak_bytes = Max(
        score_context->scratch_extra_peak_bytes,
        evoke_page_query_topk_accumulator_bytes(
            &score_context->filtered_topk
        )
    );
    if (filter->live_membership_verified)
    {
        uint64 bitmap_byte_count =
            (Min(
                filter->document_slot_count,
                (uint64) directory->document_count
            ) + UINT64_C(7)) >> 3;
        uint32 prefetch_documents[
            EVOKE_PAGE_QUERY_FILTER_PREFETCH_BATCH_DOCUMENTS
        ];
        uint32 prefetch_document_count = 0;
        bool use_prefetch = filter->allowed_document_count >=
            EVOKE_PAGE_QUERY_FILTER_PREFETCH_MIN_DOCUMENTS;

        for (uint64 byte_index = 0;
             byte_index < bitmap_byte_count;
             byte_index++)
        {
            uint8 candidates = filter->allowed_document_bitmap[byte_index];

            while (candidates != 0)
            {
                unsigned bit_index = (unsigned) __builtin_ctz(
                    (unsigned) candidates
                );
                uint64 document_slot_u64 =
                    byte_index * UINT64_C(8) + bit_index;
                uint32 document_slot;
                float score;

                candidates &= (uint8) (candidates - UINT8_C(1));
                if (document_slot_u64 >= directory->document_count)
                {
                    continue;
                }
                document_slot = (uint32) document_slot_u64;
                if (score_context->direct_forward_chunks != NULL)
                {
                    uint32 chunk_index = document_slot >>
                        directory->forward_document_shift;

                    if (chunk_index >= directory->forward_chunk_count)
                    {
                        status = EVOKE_ERR_FORMAT;
                        goto done;
                    }
                    if (score_context->direct_forward_chunks[
                            chunk_index] == 0)
                    {
                        continue;
                    }
                }
                if (use_prefetch)
                {
                    prefetch_documents[prefetch_document_count++] =
                        document_slot;
                    if (prefetch_document_count <
                        EVOKE_PAGE_QUERY_FILTER_PREFETCH_BATCH_DOCUMENTS)
                    {
                        continue;
                    }
                    evoke_segment_pages_prefetch_semantic_accelerator_forward_rows(
                        index_relation,
                        context,
                        directory,
                        prefetch_documents,
                        prefetch_document_count
                    );
                    for (uint32 prefetch_index = 0;
                         prefetch_index < prefetch_document_count;
                         prefetch_index++)
                    {
                        status = evoke_page_query_score_forward_document(
                            score_context,
                            prefetch_documents[prefetch_index],
                            score_context->query_ids,
                            score_context->query_weights,
                            score_context->query_count,
                            &score
                        );
                        if (status != EVOKE_OK)
                        {
                            goto done;
                        }
                        documents_scored++;
                        if (score > 0.0f)
                        {
                            status = evoke_topk_accumulator_offer(
                                &score_context->filtered_topk,
                                score,
                                prefetch_documents[prefetch_index],
                                prefetch_documents[prefetch_index]
                            );
                            if (status != EVOKE_OK)
                            {
                                goto done;
                            }
                        }
                    }
                    prefetch_document_count = 0;
                    continue;
                }
                status = evoke_page_query_score_forward_document(
                    score_context,
                    document_slot,
                    score_context->query_ids,
                    score_context->query_weights,
                    score_context->query_count,
                    &score
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
                documents_scored++;
                if (score > 0.0f)
                {
                    status = evoke_topk_accumulator_offer(
                        &score_context->filtered_topk,
                        score,
                        document_slot,
                        document_slot
                    );
                    if (status != EVOKE_OK)
                    {
                        goto done;
                    }
                }
            }
            if ((byte_index & UINT64_C(4095)) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
        }
        if (use_prefetch && prefetch_document_count > 0)
        {
            evoke_segment_pages_prefetch_semantic_accelerator_forward_rows(
                index_relation,
                context,
                directory,
                prefetch_documents,
                prefetch_document_count
            );
            for (uint32 prefetch_index = 0;
                 prefetch_index < prefetch_document_count;
                 prefetch_index++)
            {
                float score;

                status = evoke_page_query_score_forward_document(
                    score_context,
                    prefetch_documents[prefetch_index],
                    score_context->query_ids,
                    score_context->query_weights,
                    score_context->query_count,
                    &score
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
                documents_scored++;
                if (score > 0.0f)
                {
                    status = evoke_topk_accumulator_offer(
                        &score_context->filtered_topk,
                        score,
                        prefetch_documents[prefetch_index],
                        prefetch_documents[prefetch_index]
                    );
                    if (status != EVOKE_OK)
                    {
                        goto done;
                    }
                }
            }
        }
        status = evoke_topk_accumulator_finish(
            &score_context->filtered_topk,
            true,
            result_out
        );
        if (status == EVOKE_OK)
        {
            *documents_scored_out = documents_scored;
        }
        goto done;
    }
    for (uint64 block_index = 0;
         block_index < document_block_count;
         block_index++)
    {
        uint32 block_id = (uint32) block_index;

        if (!evoke_page_query_block_has_allowed_document(
                filter,
                block_id,
                context->query_contract.block_shift))
        {
            continue;
        }
        evoke_segment_pages_load_query_document_block(
            index_relation,
            context,
            &document_reader,
            block_id,
            &document_block
        );
        stats->document_block_reads++;
        stats->max_document_block_records = Max(
            stats->max_document_block_records,
            document_block.record_count
        );
        for (uint32 local_slot = 0;
             local_slot < document_block.record_count;
             local_slot++)
        {
            const evoke_document_cow_record *record =
                &document_block.records[local_slot];
            uint32 document_slot =
                document_block.first_document_slot + local_slot;
            float score;

            if (record->version.document_slot != document_slot)
            {
                status = EVOKE_ERR_FORMAT;
                goto done;
            }
            if (!evoke_page_query_document_allowed(filter, document_slot) ||
                !evoke_page_query_document_is_live(record))
            {
                continue;
            }
            if (score_context->direct_forward_chunks != NULL)
            {
                uint32 chunk_index = document_slot >>
                    directory->forward_document_shift;

                if (chunk_index >= directory->forward_chunk_count)
                {
                    status = EVOKE_ERR_FORMAT;
                    goto done;
                }
                if (score_context->direct_forward_chunks[chunk_index] == 0)
                {
                    continue;
                }
            }
            status = evoke_page_query_score_forward_document(
                score_context,
                document_slot,
                score_context->query_ids,
                score_context->query_weights,
                score_context->query_count,
                &score
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }
            documents_scored++;
            if (score > 0.0f)
            {
                status = evoke_topk_accumulator_offer(
                    &score_context->filtered_topk,
                    score,
                    document_slot,
                    document_slot
                );
                if (status != EVOKE_OK)
                {
                    goto done;
                }
            }
        }
        CHECK_FOR_INTERRUPTS();
    }
    status = evoke_topk_accumulator_finish(
        &score_context->filtered_topk,
        true,
        result_out
    );
    if (status == EVOKE_OK)
    {
        *documents_scored_out = documents_scored;
    }

done:
    evoke_topk_accumulator_free(&score_context->filtered_topk);
    return status;
}

static evoke_status
evoke_page_query_merge_filtered_partitions(
    const evoke_topk_result *left,
    const evoke_topk_result *right,
    size_t k,
    evoke_topk_accumulator *accumulator,
    evoke_topk_result *result_out
)
{
    evoke_status status;

    if (left == NULL || right == NULL || accumulator == NULL ||
        result_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_topk_accumulator_init(accumulator, k);
    if (status != EVOKE_OK)
    {
        return status;
    }
    for (size_t result_index = 0;
         result_index < left->len;
         result_index++)
    {
        status = evoke_topk_accumulator_offer(
            accumulator,
            left->scores[result_index],
            left->doc_ids[result_index],
            left->doc_ids[result_index]
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    for (size_t result_index = 0;
         result_index < right->len;
         result_index++)
    {
        status = evoke_topk_accumulator_offer(
            accumulator,
            right->scores[result_index],
            right->doc_ids[result_index],
            right->doc_ids[result_index]
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    return evoke_topk_accumulator_finish(accumulator, true, result_out);
}

static void
evoke_page_query_accelerator_cleanup_release(
    evoke_page_query_accelerator_cleanup *resources,
    bool release_result
)
{
    if (resources == NULL || !resources->active)
    {
        return;
    }
    resources->active = false;
    if (release_result && resources->result_out != NULL)
    {
        evoke_topk_result_free(resources->result_out);
    }
    evoke_topk_accumulator_free(&resources->merged_topk);
    evoke_weighted_space_saving_free(&resources->residual_summary);
    evoke_topk_accumulator_free(&resources->residual_candidate_topk);
    evoke_topk_accumulator_free(&resources->term_candidate_topk);
    evoke_topk_accumulator_free(&resources->residual_topk);
    evoke_topk_accumulator_free(&resources->exact_residual_topk);
    evoke_page_query_residual_merge_free(
        &resources->exact_residual_merge
    );
    evoke_topk_result_free(&resources->term_candidate_result);
    evoke_topk_result_free(&resources->residual_candidate_result);
    evoke_topk_result_free(&resources->residual_result);
    evoke_topk_result_free(&resources->accelerated_result);
    evoke_topk_accumulator_free(
        &resources->score_context.filtered_topk
    );
    evoke_semantic_forward_chunk_free(&resources->score_context.chunk);
    evoke_segment_forward_row_reader_reset(
        &resources->score_context.row_reader
    );
    evoke_segment_forward_transpose_reader_reset(
        &resources->score_context.transpose_reader
    );
    for (size_t index = 0; index < resources->index_count; index++)
    {
        evoke_semantic_accelerator_index_free(
            &resources->owned_indexes[index]
        );
    }
    evoke_semantic_accelerator_directory_free(&resources->directory);
    evoke_semantic_accelerator_query_scratch_destroy(
        resources->accelerator_scratch
    );
    free(resources->value_window);
    free(resources->document_window);
    free(resources->residual_documents);
    free(resources->residual_candidate_scores);
    free(resources->accelerated_documents);
    free(resources->indexes);
    free(resources->owned_indexes);
    free(resources->canonical_weights);
    free(resources->canonical_ids);
    free(resources->hybrid_direct_chunks);
    free(resources->hybrid_transpose_chunk_counts);
    free(resources->allowed_forward_chunk_counts);
    free(resources->ordered_weights);
    free(resources->ordered_ids);
    free(resources->terms);
}

static void
evoke_page_query_accelerator_context_reset(void *arg)
{
    evoke_page_query_accelerator_cleanup *resources = arg;

    evoke_page_query_accelerator_cleanup_release(resources, true);
}

static evoke_status
evoke_page_query_try_semantic_accelerator(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const evoke_page_query_filter *filter,
    const evoke_page_query_filter *candidate_filter,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats,
    bool *used_out
)
{
    evoke_semantic_accelerator_options options;
    evoke_semantic_accelerator_stats accelerator_stats;
    size_t term_count = 0;
    size_t ordered_count = 0;
    size_t residual_term_count = 0;
    size_t residual_document_count = 0;
    size_t residual_documents_already_scored = 0;
    size_t residual_candidate_capacity = 0;
    size_t residual_candidates_per_term = 0;
    size_t residual_summary_capacity = 0;
    uint64 residual_term_candidate_peak_bytes = 0;
    uint64 residual_merge_peak_bytes = 0;
    uint64 initial_error_budget_pruned_postings =
        stats->query_error_budget_pruned_postings;
    uint32 initial_error_budget_pruned_term_count =
        stats->query_error_budget_pruned_term_count;
    double initial_semantic_total_absolute_bound =
        stats->query_semantic_total_absolute_bound;
    double initial_semantic_omitted_absolute_bound =
        stats->query_semantic_omitted_absolute_bound;
    double average_document_length;
    double base_score = 0.0;
    uint64 filtered_documents_scored = 0;
    bool bound_residual_candidates = false;
    bool accumulate_residual_candidates = false;
    bool summarize_residual_candidates = false;
    bool residual_merge_available = false;
    bool residual_lexical_neutral = false;
    bool transposed_available = false;
    bool bounded_available = false;
    bool prefer_direct_rows = false;
    bool prefer_hybrid_rows = false;
    bool prefer_bounded_rows = false;
    bool baseline_delta_mode;
    uint32 hybrid_direct_chunk_count = 0;
    uint32 hybrid_transpose_chunk_count = 0;
    uint64 maximum_forward_document_count = 0;
    uint64 active_forward_document_count = 0;
    uint64 direct_forward_work = 0;
    uint64 transpose_forward_work = 0;
    uint64 direct_forward_estimated_bytes = 0;
    uint64 transpose_forward_estimated_bytes = 0;
    uint64 bound_forward_estimated_bytes = UINT64_MAX;
    uint64 transpose_planning_bytes = 0;
    uint64 transpose_stream_bytes = 0;
    uint64 transpose_forward_lane_bytes = 0;
    bool transpose_forward_budget_exceeded = false;
    evoke_status status = EVOKE_OK;
    evoke_page_query_accelerator_cleanup *resources;

    *used_out = false;
    if (evoke_semantic_accelerator_disabled)
    {
        return EVOKE_OK;
    }
    stats->semantic_accelerator_attempted = true;
    if (evoke_semantic_accelerator_heap_factor <= 0.0)
    {
        /* The compact forward stream is bounded-approximate, never exact. */
        stats->semantic_accelerator_fallback = true;
        return EVOKE_OK;
    }
    if (query_len == 0 || k == 0 ||
        evoke_method_requires_nonoccurrence(
            context->query_contract.params.method) ||
        !context->semantic_accelerator_compatible ||
        !evoke_segment_manifest_semantic_accelerator_eligible(
            &context->root,
            &context->manifest))
    {
        stats->semantic_accelerator_fallback = true;
        return EVOKE_OK;
    }
    baseline_delta_mode = projection != NULL ||
        evoke_segment_manifest_semantic_accelerator_baseline_sequence(
            &context->manifest
        ) < context->manifest.max_sequence;
    resources = palloc0(sizeof(*resources));
    resources->active = true;
    resources->result_out = result_out;
    resources->callback.func =
        evoke_page_query_accelerator_context_reset;
    resources->callback.arg = resources;
    MemoryContextRegisterResetCallback(
        CurrentMemoryContext,
        &resources->callback
    );
    evoke_semantic_accelerator_directory_init(&resources->directory);
    evoke_weighted_space_saving_init_empty(&resources->residual_summary);
    memset(&resources->score_context, 0, sizeof(resources->score_context));
    evoke_semantic_forward_chunk_init(&resources->score_context.chunk);
    evoke_segment_forward_row_reader_init(&resources->score_context.row_reader);
    evoke_segment_forward_transpose_reader_init(
        &resources->score_context.transpose_reader
    );
    memset(&accelerator_stats, 0, sizeof(accelerator_stats));

    resources->terms = calloc(query_len, sizeof(*resources->terms));
    resources->canonical_ids = malloc(query_len * sizeof(*resources->canonical_ids));
    resources->canonical_weights = malloc(query_len * sizeof(*resources->canonical_weights));
    resources->ordered_ids = malloc(query_len * sizeof(*resources->ordered_ids));
    resources->ordered_weights = malloc(query_len * sizeof(*resources->ordered_weights));
    if (resources->terms == NULL || resources->canonical_ids == NULL || resources->canonical_weights == NULL ||
        resources->ordered_ids == NULL || resources->ordered_weights == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (size_t query_index = 0; query_index < query_len; query_index++)
    {
        float weight = query_weights == NULL
            ? 1.0f
            : query_weights[query_index];

        if (!isfinite(weight) || weight < 0.0f)
        {
            stats->semantic_accelerator_fallback = true;
            goto cleanup;
        }
        if (weight == 0.0f ||
            query_ids[query_index] >= context->manifest.vocab_size)
        {
            continue;
        }
        resources->terms[term_count].term_id = query_ids[query_index];
        resources->terms[term_count].weight = weight;
        term_count++;
    }
    if (term_count == 0)
    {
        stats->semantic_accelerator_fallback = true;
        goto cleanup;
    }
    qsort(
        resources->terms,
        term_count,
        sizeof(*resources->terms),
        evoke_page_query_compare_accelerator_term
    );
    {
        size_t unique_count = 0;

        for (size_t term_index = 0; term_index < term_count; term_index++)
        {
            if (unique_count > 0 &&
                resources->terms[unique_count - 1U].term_id == resources->terms[term_index].term_id)
            {
                resources->terms[unique_count - 1U].weight += resources->terms[term_index].weight;
                if (!isfinite(resources->terms[unique_count - 1U].weight))
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
            }
            else
            {
                resources->terms[unique_count++] = resources->terms[term_index];
            }
        }
        term_count = unique_count;
    }
    stats->query_term_count = (uint32) term_count;

    if (filter != NULL)
    {
        evoke_segment_pages_load_semantic_accelerator_forward_directory(
            index_relation,
            context,
            &resources->directory
        );
    }
    else
    {
        evoke_segment_pages_load_semantic_accelerator_directory(
            index_relation,
            &context->root,
            &context->manifest,
            &resources->directory
        );
    }
    stats->accelerator_directory_term_count =
        resources->directory.term_count;
    if (resources->directory.forward_chunk_count == 0 ||
        resources->directory.document_count == 0 ||
        resources->directory.document_count >
            context->manifest.document_slot_count)
    {
        stats->semantic_accelerator_fallback = true;
        goto cleanup;
    }
    stats->accelerator_baseline_document_slots =
        resources->directory.document_count;
    if (baseline_delta_mode)
    {
        size_t retained_term_count = 0;

        for (size_t term_index = 0;
             term_index < term_count;
             term_index++)
        {
            if (resources->terms[term_index].term_id >=
                resources->directory.vocab_size)
            {
                continue;
            }
            resources->terms[retained_term_count++] =
                resources->terms[term_index];
        }
        term_count = retained_term_count;
        stats->query_term_count = (uint32) term_count;
        if (term_count == 0)
        {
            stats->semantic_accelerator_fallback = true;
            goto cleanup;
        }
    }
    k = Min(k, (size_t) resources->directory.document_count);
    if (filter != NULL)
    {
        for (size_t term_index = 0;
             term_index < term_count;
             term_index++)
        {
            resources->ordered_ids[term_index] = resources->terms[term_index].term_id;
            resources->ordered_weights[term_index] = resources->terms[term_index].weight;
        }
        resources->score_context.index_relation = index_relation;
        resources->score_context.query_context = context;
        resources->score_context.directory = &resources->directory;
        resources->score_context.query_ids = resources->ordered_ids;
        resources->score_context.query_weights = resources->ordered_weights;
        resources->score_context.query_count = term_count;
        resources->score_context.query_is_sorted = true;
        resources->score_context.direct_rows = true;
        if (filter->allowed_document_bitmap != NULL)
        {
            status = evoke_page_query_count_allowed_forward_chunks(
                filter,
                &resources->directory,
                &resources->allowed_forward_chunk_counts,
                stats
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
        }
        resources->score_context.allowed_forward_chunk_counts =
            resources->allowed_forward_chunk_counts;
        if (resources->allowed_forward_chunk_counts != NULL)
        {
            uint64 active_chunk_count = 0;

            for (uint32 chunk_index = 0;
                 chunk_index < resources->directory.forward_chunk_count;
                 chunk_index++)
            {
                uint32 allowed_chunk_documents =
                    resources->allowed_forward_chunk_counts[chunk_index];

                if (allowed_chunk_documents == 0)
                {
                    continue;
                }
                active_chunk_count++;
                maximum_forward_document_count = Max(
                    maximum_forward_document_count,
                    resources->directory.forward_chunks[
                        chunk_index
                    ].document_count
                );
                active_forward_document_count = evoke_u64_saturating_add(
                    active_forward_document_count,
                    resources->directory.forward_chunks[
                        chunk_index
                    ].document_count
                );
                transpose_stream_bytes = evoke_u64_saturating_add(
                    transpose_stream_bytes,
                    resources->directory.forward_chunks[
                        chunk_index
                    ].forward_object.object_bytes
                );
                direct_forward_work = evoke_u64_saturating_add(
                    direct_forward_work,
                    evoke_page_query_estimate_direct_forward_work(
                        &resources->directory.forward_chunks[chunk_index],
                        allowed_chunk_documents
                    )
                );
                direct_forward_estimated_bytes = evoke_u64_saturating_add(
                    direct_forward_estimated_bytes,
                    evoke_page_query_estimate_direct_forward_bytes(
                        &resources->directory.forward_chunks[chunk_index],
                        resources->directory.forward_row_data_bytes[
                            chunk_index
                        ],
                        allowed_chunk_documents
                    )
                );
                transpose_forward_estimated_bytes =
                    evoke_u64_saturating_add(
                        transpose_forward_estimated_bytes,
                        resources->directory.forward_transpose_fixed_bytes[
                            chunk_index
                        ]
                    );
            }
            if (active_chunk_count > 0)
            {
                for (size_t term_index = 0;
                     term_index < term_count;
                     term_index++)
                {
                    uint64 scaled_work = evoke_u64_saturating_mul(
                        resources->directory.forward_term_work[
                            resources->ordered_ids[term_index]
                        ],
                        active_forward_document_count
                    );

                    if (scaled_work == UINT64_MAX)
                    {
                        transpose_forward_work = UINT64_MAX;
                        break;
                    }
                    scaled_work =
                        scaled_work /
                            resources->directory.document_count +
                        (scaled_work %
                            resources->directory.document_count != 0);
                    transpose_forward_work = evoke_u64_saturating_add(
                        transpose_forward_work,
                        scaled_work
                    );
                    transpose_forward_estimated_bytes =
                        evoke_u64_saturating_add(
                            transpose_forward_estimated_bytes,
                            evoke_page_query_scale_bytes(
                                resources->directory.forward_term_bytes[
                                    resources->ordered_ids[term_index]
                                ],
                                active_forward_document_count,
                                resources->directory.document_count
                            )
                        );
                }
                /*
                 * Transpose initializes and reduces one score lane for every
                 * document in each active chunk before query-term ranges are
                 * accumulated. Charge that executor work once, independently
                 * of query-term posting density.
                 */
                transpose_forward_work = evoke_u64_saturating_add(
                    transpose_forward_work,
                    active_forward_document_count
                );
                transpose_planning_bytes = evoke_u64_saturating_mul(
                    resources->directory.vocab_size,
                    3U * sizeof(uint64)
                );
                transpose_planning_bytes = evoke_u64_saturating_add(
                    transpose_planning_bytes,
                    evoke_u64_saturating_mul(
                        resources->directory.forward_chunk_count,
                        2U * sizeof(uint64)
                    )
                );
                bound_forward_estimated_bytes =
                    evoke_page_query_estimate_forward_bound_bytes(
                        &resources->directory,
                        resources->ordered_ids,
                        term_count
                    );
                prefer_direct_rows =
                    evoke_page_query_prefer_direct_forward_rows(
                        maximum_forward_document_count,
                        direct_forward_estimated_bytes,
                        transpose_forward_estimated_bytes,
                        &transpose_forward_lane_bytes,
                        &transpose_forward_budget_exceeded
                    );
                prefer_bounded_rows =
                    evoke_page_query_prefer_bounded_forward_rows(
                        prefer_direct_rows,
                        transpose_forward_estimated_bytes,
                        bound_forward_estimated_bytes
                    );
            }
        }
        stats->filtered_forward_direct_work = direct_forward_work;
        stats->filtered_forward_transpose_work = transpose_forward_work;
        stats->filtered_forward_direct_estimated_bytes =
            direct_forward_estimated_bytes;
        stats->filtered_forward_transpose_estimated_bytes =
            transpose_forward_estimated_bytes;
        stats->filtered_forward_bound_estimated_bytes =
            bound_forward_estimated_bytes == UINT64_MAX
                ? 0
                : bound_forward_estimated_bytes;
        stats->filtered_forward_transpose_planning_bytes =
            transpose_planning_bytes;
        stats->filtered_forward_transpose_stream_bytes =
            transpose_stream_bytes;
        stats->filtered_forward_transpose_lane_bytes =
            transpose_forward_lane_bytes;
        stats->filtered_forward_transpose_budget_exceeded =
            transpose_forward_budget_exceeded;
        if (evoke_test_filtered_forward_route ==
            EVOKE_FILTERED_FORWARD_ROUTE_DIRECT)
        {
            prefer_direct_rows = true;
            prefer_bounded_rows = false;
        }
        else if (evoke_test_filtered_forward_route ==
                 EVOKE_FILTERED_FORWARD_ROUTE_TRANSPOSE)
        {
            prefer_direct_rows = false;
            prefer_bounded_rows = false;
        }
        else if (evoke_test_filtered_forward_route ==
                 EVOKE_FILTERED_FORWARD_ROUTE_HYBRID)
        {
            size_t chunk_count =
                resources->directory.forward_chunk_count;

            prefer_bounded_rows = false;
            resources->hybrid_direct_chunks = calloc(
                chunk_count,
                sizeof(*resources->hybrid_direct_chunks)
            );
            resources->hybrid_transpose_chunk_counts = calloc(
                chunk_count,
                sizeof(*resources->hybrid_transpose_chunk_counts)
            );
            if (resources->hybrid_direct_chunks == NULL ||
                resources->hybrid_transpose_chunk_counts == NULL)
            {
                status = EVOKE_ERR_NOMEM;
                goto cleanup;
            }
            for (uint32 chunk_index = 0;
                 chunk_index < chunk_count;
                 chunk_index++)
            {
                const evoke_semantic_accelerator_forward_entry *entry;
                uint32 allowed_count =
                    resources->allowed_forward_chunk_counts[chunk_index];
                uint64 direct_bytes;
                uint64 transpose_bytes;

                if (allowed_count == 0)
                {
                    continue;
                }
                entry = &resources->directory.forward_chunks[chunk_index];
                direct_bytes = evoke_page_query_estimate_direct_forward_bytes(
                    entry,
                    resources->directory.forward_row_data_bytes[chunk_index],
                    allowed_count
                );
                transpose_bytes =
                    evoke_page_query_estimate_transpose_chunk_bytes(
                        &resources->directory,
                        chunk_index,
                        resources->ordered_ids,
                        term_count
                    );
                if (transpose_forward_budget_exceeded ||
                    direct_bytes <= transpose_bytes)
                {
                    resources->hybrid_direct_chunks[chunk_index] = 1;
                    hybrid_direct_chunk_count++;
                }
                else
                {
                    resources->hybrid_transpose_chunk_counts[chunk_index] =
                        allowed_count;
                    hybrid_transpose_chunk_count++;
                }
            }
            prefer_direct_rows = hybrid_transpose_chunk_count == 0;
            prefer_hybrid_rows = hybrid_direct_chunk_count > 0 &&
                hybrid_transpose_chunk_count > 0;
        }
        else if (evoke_test_filtered_forward_route ==
                 EVOKE_FILTERED_FORWARD_ROUTE_BOUND)
        {
            prefer_bounded_rows = true;
        }
        if (prefer_bounded_rows)
        {
            status = evoke_page_query_score_filtered_forward_bounded(
                index_relation,
                context,
                &resources->directory,
                filter,
                k,
                &resources->score_context,
                result_out,
                stats,
                &filtered_documents_scored,
                &bounded_available
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            if (!bounded_available)
            {
                stats->filtered_forward_bound_probe_fallback = true;
            }
        }
        if (bounded_available)
        {
            stats->accelerator_forward_chunk_reads =
                resources->score_context.chunk_reads;
            stats->accelerator_forward_row_reads =
                resources->score_context.row_reads;
            stats->accelerator_forward_postings_examined =
                resources->score_context.postings_examined;
            stats->accelerator_forward_bytes =
                resources->score_context.bytes_read;
            stats->semantic_accelerator_forward_bounded = true;
            stats->documents_examined = filtered_documents_scored;
            stats->accelerator_document_refs_examined =
                filtered_documents_scored;
            stats->accelerator_documents_scored =
                filtered_documents_scored;
            stats->positive_document_count = result_out->len;
            stats->positive_topk_complete = true;
            stats->semantic_accelerator_query_path = true;
            *used_out = true;
            goto cleanup;
        }
        resources->score_context.bytes_read = evoke_u64_saturating_add(
            resources->score_context.bytes_read,
            transpose_planning_bytes
        );
        if (prefer_hybrid_rows)
        {
            uint64 transposed_documents_scored = 0;
            uint64 direct_documents_scored = 0;

            resources->score_context.allowed_forward_chunk_counts =
                resources->hybrid_transpose_chunk_counts;
            status = evoke_page_query_score_filtered_forward_transposed(
                index_relation,
                context,
                &resources->directory,
                filter,
                k,
                &resources->score_context,
                &resources->accelerated_result,
                &transposed_documents_scored,
                &transposed_available
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            if (!transposed_available)
            {
                evoke_topk_result_free(&resources->accelerated_result);
                resources->score_context.allowed_forward_chunk_counts =
                    resources->allowed_forward_chunk_counts;
                resources->score_context.direct_forward_chunks = NULL;
                prefer_hybrid_rows = false;
                prefer_direct_rows = true;
            }
            else
            {
                resources->score_context.allowed_forward_chunk_counts =
                    resources->allowed_forward_chunk_counts;
                resources->score_context.direct_forward_chunks =
                    resources->hybrid_direct_chunks;
                status = evoke_page_query_score_filtered_forward(
                    index_relation,
                    context,
                    &resources->directory,
                    filter,
                    k,
                    &resources->score_context,
                    &resources->residual_result,
                    stats,
                    &direct_documents_scored
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                resources->score_context.direct_forward_chunks = NULL;
                status = evoke_page_query_merge_filtered_partitions(
                    &resources->accelerated_result,
                    &resources->residual_result,
                    k,
                    &resources->merged_topk,
                    result_out
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                filtered_documents_scored = evoke_u64_saturating_add(
                    transposed_documents_scored,
                    direct_documents_scored
                );
                stats->accelerator_forward_chunk_reads =
                    resources->score_context.chunk_reads;
                stats->accelerator_forward_row_reads =
                    resources->score_context.row_reads;
                stats->accelerator_forward_postings_examined =
                    resources->score_context.postings_examined;
                stats->accelerator_forward_bytes =
                    resources->score_context.bytes_read;
                stats->semantic_accelerator_forward_direct_rows = true;
                stats->semantic_accelerator_forward_transposed = true;
                stats->documents_examined = filtered_documents_scored;
                stats->accelerator_document_refs_examined =
                    filtered_documents_scored;
                stats->accelerator_documents_scored =
                    filtered_documents_scored;
                stats->positive_document_count = result_out->len;
                stats->positive_topk_complete = true;
                stats->semantic_accelerator_query_path = true;
                *used_out = true;
                goto cleanup;
            }
        }
        if (!prefer_direct_rows)
        {
            status = evoke_page_query_score_filtered_forward_transposed(
                index_relation,
                context,
                &resources->directory,
                filter,
                k,
                &resources->score_context,
                result_out,
                &filtered_documents_scored,
                &transposed_available
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
        }
        if (prefer_direct_rows || !transposed_available)
        {
            status = evoke_page_query_score_filtered_forward(
                index_relation,
                context,
                &resources->directory,
                filter,
                k,
                &resources->score_context,
                result_out,
                stats,
                &filtered_documents_scored
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
        }
        stats->accelerator_forward_chunk_reads = resources->score_context.chunk_reads;
        stats->accelerator_forward_row_reads = resources->score_context.row_reads;
        stats->accelerator_forward_postings_examined =
            resources->score_context.postings_examined;
        stats->accelerator_forward_bytes = resources->score_context.bytes_read;
        if (prefer_direct_rows || !transposed_available)
        {
            stats->semantic_accelerator_forward_direct_rows = true;
        }
        else
        {
            stats->semantic_accelerator_forward_transposed = true;
        }
        stats->documents_examined = filtered_documents_scored;
        stats->accelerator_document_refs_examined =
            filtered_documents_scored;
        stats->accelerator_documents_scored = filtered_documents_scored;
        stats->positive_document_count = result_out->len;
        stats->positive_topk_complete = true;
        stats->semantic_accelerator_query_path = true;
        *used_out = true;
        goto cleanup;
    }
    for (size_t term_index = 0; term_index < term_count; term_index++)
    {
        if (!baseline_delta_mode)
        {
            evoke_segment_pages_load_query_term_plan(
                index_relation,
                context,
                resources->terms[term_index].term_id,
                &resources->terms[term_index].plan
            );
        }
        resources->terms[term_index].accelerated =
            evoke_semantic_accelerator_directory_find(
                &resources->directory,
                resources->terms[term_index].term_id
            ) != NULL;
        if (resources->terms[term_index].accelerated)
        {
            stats->accelerator_matched_term_count++;
        }
    }
    if (!baseline_delta_mode)
    {
        term_count = evoke_page_query_apply_accelerator_error_budget(
            resources->terms,
            term_count,
            stats
        );
    }
    if (term_count == 0)
    {
        stats->semantic_accelerator_fallback = true;
        goto cleanup;
    }
    if (!baseline_delta_mode &&
        evoke_semantic_accelerator_bound_residual_candidates)
    {
        for (size_t term_index = 0;
             term_index < term_count && !residual_lexical_neutral;
             term_index++)
        {
            if (resources->terms[term_index].accelerated)
            {
                continue;
            }
            for (uint32 run_index = 0;
                 run_index < resources->terms[term_index].plan.run_count;
                 run_index++)
            {
                if (resources->terms[term_index].plan.runs[run_index].kind ==
                        EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    residual_lexical_neutral = true;
                    break;
                }
            }
        }
        if (residual_lexical_neutral &&
            (context->resident_document_lengths == NULL ||
             context->resident_document_length_count !=
                 context->manifest.document_slot_count))
        {
            stats->semantic_accelerator_fallback = true;
            goto cleanup;
        }
    }

    resources->owned_indexes = calloc(term_count, sizeof(*resources->owned_indexes));
    resources->indexes = calloc(term_count, sizeof(*resources->indexes));
    if (resources->owned_indexes == NULL || resources->indexes == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    average_document_length =
        (double) context->manifest.total_document_length /
        (double) context->manifest.visible_document_count;
    for (size_t term_index = 0; term_index < term_count; term_index++)
    {
        bool lexical_seen = false;

        resources->canonical_ids[term_index] = resources->terms[term_index].term_id;
        resources->canonical_weights[term_index] = resources->terms[term_index].weight;
        resources->ordered_ids[ordered_count] = resources->terms[term_index].term_id;
        resources->ordered_weights[ordered_count] = resources->terms[term_index].weight;
        ordered_count++;
        for (uint32 run_index = 0;
             run_index < resources->terms[term_index].plan.run_count;
             run_index++)
        {
            evoke_posting_extent_kind kind =
                resources->terms[term_index].plan.runs[run_index].kind;

            if (kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                lexical_seen = true;
            }
            else if (kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
            {
                lexical_seen = true;
            }
            else if (kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
        }
        if (lexical_seen && evoke_method_requires_nonoccurrence(
                context->query_contract.params.method) &&
            resources->terms[term_index].plan.raw_document_frequency > 0)
        {
            double idf = evoke_score_idf(
                context->query_contract.params.idf_method,
                resources->terms[term_index].plan.raw_document_frequency,
                context->manifest.visible_document_count
            );
            double nonoccurrence = idf * evoke_score_tfc(
                context->query_contract.params.method,
                0.0,
                average_document_length,
                average_document_length,
                context->query_contract.params.k1,
                context->query_contract.params.b,
                context->query_contract.params.delta
            );

            base_score += resources->terms[term_index].weight * nonoccurrence;
        }
        if (resources->terms[term_index].accelerated)
        {
            size_t owned_index = resources->index_count++;

            evoke_semantic_accelerator_index_init(
                &resources->owned_indexes[owned_index]
            );
            evoke_segment_pages_load_semantic_accelerator_term(
                index_relation,
                &context->root,
                &context->manifest,
                &resources->directory,
                resources->terms[term_index].term_id,
                &resources->owned_indexes[owned_index]
            );
            resources->indexes[owned_index] =
                &resources->owned_indexes[owned_index];
        }
    }
    if (!isfinite(base_score) ||
        base_score > FLT_MAX || base_score < -FLT_MAX)
    {
        stats->semantic_accelerator_fallback = true;
        goto cleanup;
    }
    resources->score_context.index_relation = index_relation;
    resources->score_context.query_context = context;
    resources->score_context.directory = &resources->directory;
    resources->score_context.query_ids = resources->ordered_ids;
    resources->score_context.query_weights = resources->ordered_weights;
    resources->score_context.query_count = ordered_count;
    resources->score_context.query_is_sorted = true;
    for (size_t query_index = 1;
         query_index < ordered_count;
         query_index++)
    {
        if (resources->ordered_ids[query_index - 1U] >= resources->ordered_ids[query_index])
        {
            resources->score_context.query_is_sorted = false;
            break;
        }
    }
    resources->score_context.base_score = (float) base_score;
    resources->score_context.candidate_filter = candidate_filter;
    if (resources->index_count == 0)
    {
        stats->semantic_accelerator_fallback = true;
        goto cleanup;
    }
    resources->accelerated_documents = calloc(
        ((size_t) resources->directory.document_count + 7U) / 8U,
        1
    );
    if (resources->accelerated_documents == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    resources->score_context.scored_documents = resources->accelerated_documents;
    options.query_cut = term_count;
    options.candidate_multiplier =
        (uint32) evoke_semantic_accelerator_candidate_multiplier;
    options.document_shift = resources->directory.forward_document_shift;
    options.heap_factor =
        (float) evoke_semantic_accelerator_heap_factor;
    resources->score_context.direct_rows =
        evoke_semantic_accelerator_bound_residual_candidates &&
        options.heap_factor > 0.0f;
    resources->accelerator_scratch =
        evoke_semantic_accelerator_query_scratch_create();
    if (resources->accelerator_scratch == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    status = evoke_semantic_accelerator_topk_many_owned(
        resources->indexes,
        resources->index_count,
        resources->canonical_ids,
        resources->canonical_weights,
        term_count,
        k,
        &options,
        evoke_page_query_score_forward_document,
        &resources->score_context,
        &resources->accelerated_result,
        &accelerator_stats,
        resources->accelerator_scratch
    );
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }

    /*
     * Bounded residual selection changes candidate generation only. h0 keeps
     * the complete residual union so it remains the exact accelerator oracle.
     * Every retained candidate still receives its complete unified score.
     */
    bound_residual_candidates =
        evoke_semantic_accelerator_bound_residual_candidates &&
        options.heap_factor > 0.0f;
    stats->semantic_accelerator_residual_candidates_bounded =
        bound_residual_candidates;
    for (size_t term_index = 0; term_index < term_count; term_index++)
    {
        if (!baseline_delta_mode)
        {
            residual_term_count +=
                !resources->terms[term_index].accelerated;
        }
    }
    if (bound_residual_candidates && residual_term_count > 0)
    {
        if (k > SIZE_MAX / options.candidate_multiplier)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        residual_candidate_capacity =
            k * (size_t) options.candidate_multiplier;
        if (residual_candidate_capacity > resources->directory.document_count)
        {
            residual_candidate_capacity = resources->directory.document_count;
        }
        residual_candidates_per_term =
            (residual_candidate_capacity + residual_term_count - 1U) /
            residual_term_count;
        accumulate_residual_candidates =
            evoke_semantic_accelerator_accumulate_residual_candidates;
        summarize_residual_candidates =
            accumulate_residual_candidates &&
            evoke_test_semantic_accelerator_summarize_residual_candidates;
        if (summarize_residual_candidates)
        {
            if (residual_candidate_capacity > SIZE_MAX /
                    (size_t) evoke_test_semantic_accelerator_summary_multiplier)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            residual_summary_capacity = residual_candidate_capacity *
                (size_t) evoke_test_semantic_accelerator_summary_multiplier;
            if (residual_summary_capacity > resources->directory.document_count)
            {
                residual_summary_capacity = resources->directory.document_count;
            }
        }
        stats->semantic_accelerator_residual_candidates_accumulated =
            accumulate_residual_candidates;
        stats->semantic_accelerator_residual_candidates_summarized =
            summarize_residual_candidates;
    }
    if (residual_term_count > 0)
    {
        if (!accumulate_residual_candidates)
        {
            resources->residual_documents = calloc(
                ((size_t) resources->directory.document_count + 7U) / 8U,
                1
            );
        }
        if (!accumulate_residual_candidates ||
            summarize_residual_candidates)
        {
            resources->document_window = malloc(
                EVOKE_SEGMENT_QUERY_POSTING_WINDOW * sizeof(*resources->document_window)
            );
            resources->value_window = malloc(
                EVOKE_SEGMENT_QUERY_POSTING_WINDOW * sizeof(*resources->value_window)
            );
        }
        if ((!accumulate_residual_candidates &&
                resources->residual_documents == NULL) ||
            ((!accumulate_residual_candidates ||
              summarize_residual_candidates) &&
             (resources->document_window == NULL || resources->value_window == NULL)))
        {
            status = EVOKE_ERR_NOMEM;
            goto cleanup;
        }
        if (accumulate_residual_candidates &&
            !summarize_residual_candidates)
        {
            status = evoke_page_query_accumulate_residual_candidates_exact(
                index_relation,
                context,
                resources->terms,
                term_count,
                average_document_length,
                residual_candidate_capacity,
                resources,
                &resources->residual_candidate_result,
                stats,
                &residual_merge_peak_bytes,
                &residual_merge_available
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            if (!residual_merge_available)
            {
                stats->semantic_accelerator_fallback = true;
                goto cleanup;
            }
        }
        else if (summarize_residual_candidates)
        {
            status = evoke_weighted_space_saving_init(
                &resources->residual_summary,
                residual_summary_capacity
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
        }
        for (size_t term_index = 0;
             term_index < term_count &&
                 (!accumulate_residual_candidates ||
                  summarize_residual_candidates);
             term_index++)
        {
            const evoke_segment_query_term_plan *plan =
                &resources->terms[term_index].plan;
            double residual_idf = 0.0;

            if (resources->terms[term_index].accelerated)
            {
                continue;
            }
            if (bound_residual_candidates)
            {
                if (!accumulate_residual_candidates)
                {
                    status = evoke_topk_accumulator_init(
                        &resources->term_candidate_topk,
                        residual_candidates_per_term
                    );
                    if (status != EVOKE_OK)
                    {
                        goto cleanup;
                    }
                    residual_term_candidate_peak_bytes = Max(
                        residual_term_candidate_peak_bytes,
                        evoke_page_query_topk_accumulator_bytes(
                            &resources->term_candidate_topk
                        )
                    );
                }
                residual_idf = evoke_score_idf(
                    context->query_contract.params.idf_method,
                    plan->raw_document_frequency,
                    context->manifest.visible_document_count
                );
                if (!isfinite(residual_idf))
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
            }
            for (uint32 run_index = 0;
                 run_index < plan->run_count;
                 run_index++)
            {
                evoke_segment_query_posting_cursor cursor;
                const evoke_segment_query_run *run = &plan->runs[run_index];

                evoke_segment_query_posting_cursor_init(&cursor);
                while (cursor.next_posting_index < run->posting_count)
                {
                    uint32 loaded =
                        evoke_segment_pages_load_query_term_posting_stream_window(
                            index_relation,
                            context,
                            plan,
                            run_index,
                            &cursor,
                            resources->document_window,
                            resources->value_window,
                            EVOKE_SEGMENT_QUERY_POSTING_WINDOW
                        );

                    stats->accelerator_residual_postings += loaded;
                    for (uint32 posting_index = 0;
                         posting_index < loaded;
                         posting_index++)
                    {
                        uint32 document = resources->document_window[posting_index];
                        uint8 bit;
                        uint8 *slot;

                        if (document >= resources->directory.document_count)
                        {
                            status = EVOKE_ERR_FORMAT;
                            goto cleanup;
                        }
                        if (bound_residual_candidates)
                        {
                            double candidate_score;

                            if (run->kind ==
                                    EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
                                run->kind ==
                                    EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
                            {
                                candidate_score =
                                    (double) resources->terms[term_index].weight *
                                    resources->value_window[posting_index].impact;
                            }
                            else if (run->kind ==
                                     EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                            {
                                double tfc = evoke_score_tfc(
                                    context->query_contract.params.method,
                                    resources->value_window[
                                        posting_index
                                    ].term_frequency,
                                    context->resident_document_lengths[
                                        document
                                    ],
                                    average_document_length,
                                    context->query_contract.params.k1,
                                    context->query_contract.params.b,
                                    context->query_contract.params.delta
                                );

                                candidate_score =
                                    (double) resources->terms[term_index].weight *
                                    residual_idf * tfc;
                            }
                            else
                            {
                                status = EVOKE_ERR_FORMAT;
                                goto cleanup;
                            }
                            if (!isfinite(candidate_score) ||
                                candidate_score > FLT_MAX)
                            {
                                status = EVOKE_ERR_RANGE;
                                goto cleanup;
                            }
                            if (candidate_score > 0.0)
                            {
                                if (accumulate_residual_candidates)
                                {
                                    if (summarize_residual_candidates)
                                    {
                                        status =
                                            evoke_weighted_space_saving_offer(
                                                &resources->residual_summary,
                                                document,
                                                candidate_score
                                            );
                                        if (status != EVOKE_OK)
                                        {
                                            goto cleanup;
                                        }
                                    }
                                    else
                                    {
                                        status = EVOKE_ERR_INVALID;
                                        goto cleanup;
                                    }
                                }
                                else
                                {
                                    status = evoke_topk_accumulator_offer(
                                        &resources->term_candidate_topk,
                                        (float) candidate_score,
                                        document,
                                        document
                                    );
                                    if (status != EVOKE_OK)
                                    {
                                        goto cleanup;
                                    }
                                }
                            }
                        }
                        else
                        {
                            bit = (uint8) (
                                UINT8_C(1) << (document & 7U)
                            );
                            slot = &resources->residual_documents[document >> 3];
                            if ((*slot & bit) == 0)
                            {
                                *slot |= bit;
                                residual_document_count++;
                            }
                        }
                    }
                    CHECK_FOR_INTERRUPTS();
                }
            }
            if (bound_residual_candidates && !accumulate_residual_candidates)
            {
                status = evoke_topk_accumulator_finish(
                    &resources->term_candidate_topk,
                    false,
                    &resources->term_candidate_result
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                residual_term_candidate_peak_bytes = Max(
                    residual_term_candidate_peak_bytes,
                    evoke_u64_saturating_add(
                        evoke_page_query_topk_accumulator_bytes(
                            &resources->term_candidate_topk
                        ),
                        evoke_page_query_topk_result_bytes(
                            &resources->term_candidate_result
                        )
                    )
                );
                for (size_t candidate_index = 0;
                     candidate_index < resources->term_candidate_result.len;
                     candidate_index++)
                {
                    uint32 document =
                        resources->term_candidate_result.doc_ids[candidate_index];
                    uint8 bit = (uint8) (
                        UINT8_C(1) << (document & 7U)
                    );
                    uint8 *slot = &resources->residual_documents[document >> 3];

                    if ((*slot & bit) == 0)
                    {
                        *slot |= bit;
                        residual_document_count++;
                    }
                }
                evoke_topk_result_free(&resources->term_candidate_result);
                evoke_topk_accumulator_free(&resources->term_candidate_topk);
            }
        }
        status = evoke_topk_accumulator_init(&resources->residual_topk, k);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        if (accumulate_residual_candidates)
        {
            if (summarize_residual_candidates)
            {
                status = evoke_topk_accumulator_init(
                    &resources->residual_candidate_topk,
                    residual_candidate_capacity
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                for (size_t counter_index = 0;
                     counter_index < resources->residual_summary.len;
                     counter_index++)
                {
                    const evoke_weighted_space_saving_counter *counter =
                        &resources->residual_summary.counters[counter_index];
                    float candidate_score;

                    if (counter->estimate > FLT_MAX)
                    {
                        status = EVOKE_ERR_RANGE;
                        goto cleanup;
                    }
                    candidate_score = (float) counter->estimate;
                    status = evoke_topk_accumulator_offer(
                        &resources->residual_candidate_topk,
                        candidate_score,
                        counter->item,
                        counter->item
                    );
                    if (status != EVOKE_OK)
                    {
                        goto cleanup;
                    }
                }
                status = evoke_topk_accumulator_finish(
                    &resources->residual_candidate_topk,
                    false,
                    &resources->residual_candidate_result
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
            }
            /*
             * Candidate scores have served their only purpose once the
             * bounded set is selected. Complete rows in physical document
             * order so offset and column pages are consumed monotonically.
             */
            qsort(
                resources->residual_candidate_result.doc_ids,
                resources->residual_candidate_result.len,
                sizeof(*resources->residual_candidate_result.doc_ids),
                evoke_page_query_compare_document_id
            );
            if (resources->residual_candidate_result.len >=
                EVOKE_PAGE_QUERY_FILTER_PREFETCH_MIN_DOCUMENTS)
            {
                evoke_segment_pages_prefetch_semantic_accelerator_forward_rows(
                    index_relation,
                    context,
                    &resources->directory,
                    resources->residual_candidate_result.doc_ids,
                    resources->residual_candidate_result.len
                );
            }
            for (size_t candidate_index = 0;
                 candidate_index < resources->residual_candidate_result.len;
                 candidate_index++)
            {
                uint32 document =
                    resources->residual_candidate_result.doc_ids[candidate_index];
                float score;

                residual_document_count++;
                if ((resources->accelerated_documents[document >> 3] & (uint8) (
                        UINT8_C(1) << (document & 7U))) != 0)
                {
                    residual_documents_already_scored++;
                    continue;
                }
                status = evoke_page_query_score_forward_document(
                    &resources->score_context,
                    document,
                    resources->canonical_ids,
                    resources->canonical_weights,
                    term_count,
                    &score
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                if (score > 0.0f)
                {
                    status = evoke_topk_accumulator_offer(
                        &resources->residual_topk,
                        score,
                        document,
                        document
                    );
                    if (status != EVOKE_OK)
                    {
                        goto cleanup;
                    }
                }
                CHECK_FOR_INTERRUPTS();
            }
        }
        else
        {
            for (uint32 document = 0;
                 document < resources->directory.document_count;
                 document++)
            {
                uint8 bit = (uint8) (UINT8_C(1) << (document & 7U));
                float score;

                if ((resources->residual_documents[document >> 3] & bit) == 0)
                {
                    continue;
                }
                if ((resources->accelerated_documents[document >> 3] & bit) != 0)
                {
                    /* Accelerator already computed the complete score. */
                    residual_documents_already_scored++;
                    continue;
                }
                status = evoke_page_query_score_forward_document(
                    &resources->score_context,
                    document,
                    resources->canonical_ids,
                    resources->canonical_weights,
                    term_count,
                    &score
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                if (score > 0.0f)
                {
                    status = evoke_topk_accumulator_offer(
                        &resources->residual_topk,
                        score,
                        document,
                        document
                    );
                    if (status != EVOKE_OK)
                    {
                        goto cleanup;
                    }
                }
                if ((document & EVOKE_PAGE_QUERY_INTERRUPT_MASK) == 0)
                {
                    CHECK_FOR_INTERRUPTS();
                }
            }
        }
        status = evoke_topk_accumulator_finish(
            &resources->residual_topk,
            true,
            &resources->residual_result
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    status = evoke_topk_accumulator_init(&resources->merged_topk, k);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    for (size_t result_index = 0;
         result_index < resources->accelerated_result.len;
         result_index++)
    {
        status = evoke_topk_accumulator_offer(
            &resources->merged_topk,
            resources->accelerated_result.scores[result_index],
            resources->accelerated_result.doc_ids[result_index],
            resources->accelerated_result.doc_ids[result_index]
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    for (size_t result_index = 0;
         result_index < resources->residual_result.len;
         result_index++)
    {
        uint32 document = resources->residual_result.doc_ids[result_index];

        if (evoke_page_query_accelerator_result_contains(
                &resources->accelerated_result,
                document))
        {
            continue;
        }
        status = evoke_topk_accumulator_offer(
            &resources->merged_topk,
            resources->residual_result.scores[result_index],
            document,
            document
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    status = evoke_topk_accumulator_finish(
        &resources->merged_topk,
        true,
        result_out
    );
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    if (!baseline_delta_mode &&
        result_out->len < k && options.heap_factor > 0.0f)
    {
        evoke_topk_result_free(result_out);
        stats->semantic_accelerator_fallback = true;
        goto cleanup;
    }
    stats->accelerator_forward_chunk_reads = resources->score_context.chunk_reads;
    stats->accelerator_forward_row_reads = resources->score_context.row_reads;
    stats->accelerator_forward_bytes = resources->score_context.bytes_read;
    stats->accelerator_clusters_opened = accelerator_stats.clusters_opened;
    stats->accelerator_clusters_skipped = accelerator_stats.clusters_skipped;
    stats->accelerator_document_refs_examined =
        accelerator_stats.document_refs_examined;
    stats->accelerator_documents_scored =
        accelerator_stats.documents_scored + residual_document_count -
        residual_documents_already_scored;
    stats->accelerator_residual_documents = residual_document_count;
    stats->accelerator_residual_summary_capacity = residual_summary_capacity;
    stats->accelerator_residual_summary_counters = resources->residual_summary.len;
    stats->accelerator_residual_summary_replacements =
        resources->residual_summary.replacements;
    stats->accelerator_residual_summary_total_weight =
        resources->residual_summary.total_weight;
    stats->accelerator_residual_summary_maximum_error =
        resources->residual_summary.maximum_error;
    stats->semantic_accelerator_block_major =
        accelerator_stats.block_major_query;
    stats->positive_document_count = result_out->len;
    stats->positive_topk_complete = true;
    stats->semantic_accelerator_query_path = true;
    *used_out = true;

cleanup:
    {
        uint64 owned_index_bytes = 0;
        uint64 membership_bytes = 0;
        uint64 candidate_scratch_bytes =
            accelerator_stats.query_scratch_peak_bytes;
        uint64 forward_scratch_bytes;
        uint64 residual_scratch_bytes = 0;

        stats->filtered_forward_sparse_oracle_full_bytes =
            resources->score_context.transpose_reader
                .sparse_oracle_full_bytes;
        stats->filtered_forward_sparse_oracle_selected_bytes =
            resources->score_context.transpose_reader
                .sparse_oracle_selected_bytes;
        stats->filtered_forward_sparse_oracle_ranges =
            resources->score_context.transpose_reader.sparse_oracle_ranges;
        stats->filtered_forward_sparse_oracle_full_pages =
            resources->score_context.transpose_reader
                .sparse_oracle_full_pages;
        stats->filtered_forward_sparse_oracle_selected_pages =
            resources->score_context.transpose_reader
                .sparse_oracle_selected_pages;

        if (resources->owned_indexes != NULL)
        {
            owned_index_bytes = evoke_page_query_memory_add(
                owned_index_bytes,
                term_count,
                sizeof(*resources->owned_indexes)
            );
        }
        if (resources->indexes != NULL)
        {
            owned_index_bytes = evoke_page_query_memory_add(
                owned_index_bytes,
                term_count,
                sizeof(*resources->indexes)
            );
        }
        for (size_t index = 0; index < resources->index_count; index++)
        {
            owned_index_bytes = evoke_u64_saturating_add(
                owned_index_bytes,
                evoke_page_query_accelerator_index_bytes(
                    &resources->owned_indexes[index]
                )
            );
        }
        if (resources->directory.terms != NULL)
        {
            owned_index_bytes = evoke_page_query_memory_add(
                owned_index_bytes,
                resources->directory.term_count,
                sizeof(*resources->directory.terms)
            );
        }
        if (resources->directory.forward_chunks != NULL)
        {
            owned_index_bytes = evoke_page_query_memory_add(
                owned_index_bytes,
                resources->directory.forward_chunk_count,
                sizeof(*resources->directory.forward_chunks)
            );
        }
        if (resources->accelerated_documents != NULL)
        {
            membership_bytes = evoke_page_query_memory_add(
                membership_bytes,
                ((uint64) resources->directory.document_count + 7U) / 8U,
                sizeof(*resources->accelerated_documents)
            );
        }
        if (resources->residual_documents != NULL)
        {
            membership_bytes = evoke_page_query_memory_add(
                membership_bytes,
                ((uint64) resources->directory.document_count + 7U) / 8U,
                sizeof(*resources->residual_documents)
            );
        }
        if (resources->document_window != NULL)
        {
            residual_scratch_bytes = evoke_page_query_memory_add(
                residual_scratch_bytes,
                EVOKE_SEGMENT_QUERY_POSTING_WINDOW,
                sizeof(*resources->document_window)
            );
        }
        if (resources->value_window != NULL)
        {
            residual_scratch_bytes = evoke_page_query_memory_add(
                residual_scratch_bytes,
                EVOKE_SEGMENT_QUERY_POSTING_WINDOW,
                sizeof(*resources->value_window)
            );
        }
        residual_scratch_bytes = evoke_u64_saturating_add(
            residual_scratch_bytes,
            evoke_page_query_summary_bytes(&resources->residual_summary)
        );
        residual_scratch_bytes = evoke_u64_saturating_add(
            residual_scratch_bytes,
            evoke_page_query_topk_accumulator_bytes(&resources->residual_topk)
        );
        residual_scratch_bytes = evoke_u64_saturating_add(
            residual_scratch_bytes,
            evoke_page_query_topk_accumulator_bytes(
                &resources->residual_candidate_topk
            )
        );
        residual_scratch_bytes = evoke_u64_saturating_add(
            residual_scratch_bytes,
            evoke_page_query_topk_result_bytes(&resources->residual_result)
        );
        residual_scratch_bytes = evoke_u64_saturating_add(
            residual_scratch_bytes,
            evoke_page_query_topk_result_bytes(
                &resources->residual_candidate_result
            )
        );
        residual_scratch_bytes = evoke_u64_saturating_add(
            residual_scratch_bytes,
            residual_term_candidate_peak_bytes
        );
        residual_scratch_bytes = evoke_u64_saturating_add(
            residual_scratch_bytes,
            residual_merge_peak_bytes
        );

        if (resources->allowed_forward_chunk_counts != NULL)
        {
            membership_bytes = evoke_page_query_memory_add(
                membership_bytes,
                resources->directory.forward_chunk_count,
                sizeof(*resources->allowed_forward_chunk_counts)
            );
        }

        forward_scratch_bytes =
            evoke_page_query_forward_scratch_bytes(&resources->score_context);
        stats->accelerator_owned_index_bytes = owned_index_bytes;
        stats->accelerator_membership_bytes = membership_bytes;
        stats->accelerator_candidate_scratch_bytes =
            candidate_scratch_bytes;
        stats->accelerator_forward_scratch_bytes = forward_scratch_bytes;
        stats->accelerator_residual_scratch_bytes =
            residual_scratch_bytes;
        stats->semantic_accelerator_query_bytes =
            evoke_u64_saturating_add(
                evoke_u64_saturating_add(
                    owned_index_bytes,
                    membership_bytes
                ),
                evoke_u64_saturating_add(
                    candidate_scratch_bytes,
                    evoke_u64_saturating_add(
                        forward_scratch_bytes,
                        residual_scratch_bytes
                    )
                )
            );
    }
    if (!*used_out)
    {
        stats->query_error_budget_pruned_postings =
            initial_error_budget_pruned_postings;
        stats->query_error_budget_pruned_term_count =
            initial_error_budget_pruned_term_count;
        stats->query_semantic_total_absolute_bound =
            initial_semantic_total_absolute_bound;
        stats->query_semantic_omitted_absolute_bound =
            initial_semantic_omitted_absolute_bound;
    }
    if (status != EVOKE_OK)
    {
        evoke_topk_result_free(result_out);
    }
    evoke_page_query_accelerator_cleanup_release(resources, false);
    return status;
}

static size_t
evoke_page_query_stale_accelerator_k(
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    uint64 baseline_sequence,
    size_t k
)
{
    uint64 changed_documents = projection == NULL
        ? 0
        : projection->document_count;
    uint64 overfetch_limit;
    uint64 requested;

    for (uint32 segment_index = 0;
         segment_index < context->manifest.segment_count;
         segment_index++)
    {
        const evoke_segment_descriptor *descriptor =
            &context->manifest.segments[segment_index];

        if (descriptor->max_sequence > baseline_sequence)
        {
            changed_documents = evoke_u64_saturating_add(
                changed_documents,
                descriptor->document_count
            );
        }
    }
    /*
     * A stale baseline is an approximate availability bridge, not permission
     * to make foreground work linear in accumulated mutation debt. Allow up
     * to one extra result window for ordinary K values and reserve
     * up to 64 slots for small K, while keeping one accelerator execution
     * bounded even when maintenance falls behind.
     */
    overfetch_limit = Max(
        (uint64) k,
        EVOKE_PAGE_QUERY_ACCELERATOR_STALE_OVERFETCH_FLOOR
    );
    overfetch_limit = Min(
        overfetch_limit,
        EVOKE_PAGE_QUERY_ACCELERATOR_STALE_OVERFETCH_MAX
    );
    changed_documents = Min(changed_documents, overfetch_limit);
    requested = evoke_u64_saturating_add((uint64) k, changed_documents);
    requested = Min(requested, context->manifest.document_slot_count);
    requested = Min(requested, (uint64) SIZE_MAX);
    return (size_t) requested;
}

static void
evoke_page_query_filter_accelerator_baseline(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const evoke_page_query_filter *filter,
    uint64 baseline_sequence,
    evoke_topk_result *result,
    evoke_page_query_stats *stats
)
{
    size_t write_index = 0;

    for (size_t read_index = 0;
         read_index < result->len;
         read_index++)
    {
        uint32 document_slot = result->doc_ids[read_index];
        const evoke_page_query_l0_document *projected_document =
            evoke_page_query_l0_find_document(projection, document_slot);
        evoke_document_cow_record record;

        if (!evoke_page_query_document_allowed(filter, document_slot) ||
            (projected_document != NULL &&
             projected_document->shadows_immutable))
        {
            stats->accelerator_stale_candidates_rejected++;
            continue;
        }
        evoke_segment_pages_load_document_record(
            index_relation,
            &context->root,
            &context->manifest,
            document_slot,
            &record
        );
        if (record.version.document_slot != document_slot ||
            !evoke_page_query_document_is_live(&record) ||
            record.version.born_sequence > baseline_sequence)
        {
            stats->accelerator_stale_candidates_rejected++;
            continue;
        }
        result->doc_ids[write_index] = document_slot;
        result->scores[write_index] = result->scores[read_index];
        write_index++;
    }
    result->len = write_index;
}

evoke_status
evoke_page_query_positive_topk_filtered(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const evoke_page_query_filter *filter,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
)
{
    evoke_page_query_cleanup *cleanup;
    evoke_page_query_stats stats;
    evoke_topk_result accelerator_seed = {0};
    size_t term_capacity;
    size_t term_count = 0;
    size_t cursor_count = 0;
    bool live_frequency_required = false;
    uint64 visible_document_count;
    uint64 total_document_length;
    uint64 accelerator_baseline_sequence = 0;
    double average_document_length;
    double nonoccurrence_base_score = 0.0;
    evoke_status result_status = EVOKE_OK;

    visible_document_count = projection == NULL
        ? context != NULL
            ? context->manifest.visible_document_count
            : 0
        : projection->visible_document_count;
    total_document_length = projection == NULL
        ? context != NULL
            ? context->manifest.total_document_length
            : 0
        : projection->total_document_length;
    if (index_relation == NULL || context == NULL || result_out == NULL ||
        stats_out == NULL || (query_len > 0 && query_ids == NULL) ||
        query_len > UINT32_MAX ||
        k > visible_document_count ||
        (filter != NULL &&
          ((filter->allowed_document_bitmap == NULL &&
            filter->allows_document == NULL) ||
          filter->document_slot_count !=
              context->root.next_document_slot ||
          filter->allowed_document_count >
              filter->document_slot_count ||
          k > filter->allowed_document_count)) ||
        (projection != NULL && projection->term_count != query_len))
    {
        return EVOKE_ERR_INVALID;
    }
    memset(result_out, 0, sizeof(*result_out));
    memset(stats_out, 0, sizeof(*stats_out));
    memset(&stats, 0, sizeof(stats));
    if (k == 0 || visible_document_count == 0)
    {
        stats_out->positive_topk_complete = k == 0;
        return EVOKE_OK;
    }
    if (total_document_length == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    accelerator_baseline_sequence =
        evoke_segment_manifest_semantic_accelerator_baseline_sequence(
            &context->manifest
        );
    {
        bool accelerator_used = false;
        bool accelerator_is_current =
            projection == NULL &&
            accelerator_baseline_sequence == context->manifest.max_sequence;
        size_t accelerator_k = accelerator_is_current
            ? k
            : evoke_page_query_stale_accelerator_k(
                context,
                projection,
                accelerator_baseline_sequence,
                k
            );

        if (accelerator_k > 0)
        {
            result_status = evoke_page_query_try_semantic_accelerator(
                index_relation,
                context,
                projection,
                filter,
                NULL,
                query_ids,
                query_weights,
                query_len,
                accelerator_k,
                result_out,
                &stats,
                &accelerator_used
            );
        }
        if (result_status != EVOKE_OK)
        {
            return result_status;
        }
        if (accelerator_used)
        {
            stats.accelerator_baseline_sequence =
                accelerator_baseline_sequence;
            if (accelerator_is_current &&
                !(filter == NULL &&
                  evoke_test_semantic_accelerator_seed_bmp))
            {
                *stats_out = stats;
                return EVOKE_OK;
            }
            if (!accelerator_is_current)
            {
                evoke_page_query_filter_accelerator_baseline(
                    index_relation,
                    context,
                    projection,
                    filter,
                    accelerator_baseline_sequence,
                    result_out,
                    &stats
                );
                result_out->len = Min(result_out->len, k);
                stats.semantic_accelerator_stale_baseline = true;
                stats.positive_document_count = result_out->len;
                stats.positive_topk_complete = true;
                *stats_out = stats;
                return EVOKE_OK;
            }
            accelerator_seed = *result_out;
            memset(result_out, 0, sizeof(*result_out));
            stats.positive_topk_complete = false;
            stats.semantic_accelerator_query_path = false;
            stats.positive_document_count = 0;
        }
    }

    term_capacity = query_len > 0 ? query_len : 1;
    cleanup = palloc0(sizeof(*cleanup));
    cleanup->index_relation = index_relation;
    cleanup->context = context;
    cleanup->filter = filter;
    cleanup->projection = projection;
    cleanup->minimum_root_sequence = 0;
    cleanup->terms = calloc(
        term_capacity,
        sizeof(*cleanup->terms)
    );
    if (cleanup->terms == NULL)
    {
        evoke_topk_result_free(&accelerator_seed);
        evoke_page_query_cleanup_free(cleanup);
        return EVOKE_ERR_NOMEM;
    }
    cleanup->query_term_map = malloc(
        sizeof(*cleanup->query_term_map) * term_capacity
    );
    if (cleanup->query_term_map == NULL)
    {
        evoke_topk_result_free(&accelerator_seed);
        evoke_page_query_cleanup_free(cleanup);
        return EVOKE_ERR_NOMEM;
    }
    for (size_t query_index = 0;
         query_index < term_capacity;
         query_index++)
    {
        cleanup->query_term_map[query_index] = UINT32_MAX;
    }
    if (projection != NULL && projection->document_count > 0)
    {
        cleanup->projection_document_seen = calloc(
            projection->document_count,
            sizeof(*cleanup->projection_document_seen)
        );
        if (cleanup->projection_document_seen == NULL)
        {
            evoke_topk_result_free(&accelerator_seed);
            evoke_page_query_cleanup_free(cleanup);
            return EVOKE_ERR_NOMEM;
        }
    }
    for (size_t query_index = 0; query_index < query_len; query_index++)
    {
        float query_weight = query_weights == NULL
            ? 1.0f
            : query_weights[query_index];

        if (!isfinite(query_weight))
        {
            result_status = EVOKE_ERR_INVALID;
            goto done;
        }
    }

    PG_TRY();
    {
        for (size_t query_index = 0;
             query_index < query_len;
             query_index++)
        {
            evoke_page_query_term *term;
            bool immutable_seen =
                query_ids[query_index] < context->manifest.vocab_size;
            bool projected_seen = projection != NULL &&
                (projection->terms[query_index].lexical_seen ||
                 projection->terms[query_index].semantic_seen);

            if (!immutable_seen && !projected_seen)
            {
                continue;
            }
            cleanup->query_term_map[query_index] = (uint32) term_count;
            term = &cleanup->terms[term_count++];
            term->query_index = (uint32) query_index;
            term->query_weight = query_weights == NULL
                ? 1.0f
                : query_weights[query_index];
            if (immutable_seen)
            {
                if (cleanup->minimum_root_sequence > 0)
                {
                    evoke_segment_pages_load_query_term_plan_after_sequence(
                        index_relation,
                        context,
                        query_ids[query_index],
                        cleanup->minimum_root_sequence,
                        &term->plan
                    );
                }
                else
                {
                    evoke_segment_pages_load_query_term_plan(
                        index_relation,
                        context,
                        query_ids[query_index],
                        &term->plan
                    );
                }
            }
            else
            {
                term->plan.term_id = query_ids[query_index];
            }
        }
        if (term_count == 0 &&
            (context->query_contract.flags &
             EVOKE_QUERY_CONTRACT_FLAG_EMPTY_TOKEN) != 0)
        {
            evoke_page_query_term *term = &cleanup->terms[term_count++];

            term->query_index = UINT32_MAX;
            term->query_weight = 1.0f;
            if (cleanup->minimum_root_sequence > 0)
            {
                evoke_segment_pages_load_query_term_plan_after_sequence(
                    index_relation,
                    context,
                    context->query_contract.empty_token_id,
                    cleanup->minimum_root_sequence,
                    &term->plan
                );
            }
            else
            {
                evoke_segment_pages_load_query_term_plan(
                    index_relation,
                    context,
                    context->query_contract.empty_token_id,
                    &term->plan
                );
            }
        }
    }
    PG_CATCH();
    {
        evoke_topk_result_free(&accelerator_seed);
        evoke_page_query_cleanup_free(cleanup);
        PG_RE_THROW();
    }
    PG_END_TRY();

    term_count = evoke_page_query_apply_test_df_limit(
        cleanup,
        term_count,
        visible_document_count,
        projection,
        &stats
    );
    term_count = evoke_page_query_apply_test_error_budget(
        cleanup,
        term_count,
        projection,
        &stats
    );

    for (size_t term_index = 0;
         term_index < term_count;
         term_index++)
    {
        evoke_page_query_term *term = &cleanup->terms[term_index];

        if (projection != NULL && term->query_index != UINT32_MAX &&
            projection->terms[term->query_index].lexical_seen)
        {
            term->lexical_neutral_seen = true;
        }

        if ((size_t) term->plan.run_count > SIZE_MAX - cursor_count ||
            cursor_count + term->plan.run_count > UINT32_MAX)
        {
            result_status = EVOKE_ERR_RANGE;
            goto done;
        }
        cursor_count += term->plan.run_count;
        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            evoke_posting_extent_kind kind =
                term->plan.runs[run_index].kind;

            if (kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                term->lexical_neutral_seen = true;
            }
            if (kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
            {
                term->lexical_impact_seen = true;
            }
        }
        live_frequency_required = live_frequency_required ||
            term->lexical_neutral_seen ||
            (term->lexical_impact_seen &&
             evoke_method_requires_nonoccurrence(
                 context->query_contract.params.method));
    }
    if (cursor_count > 0)
    {
        cleanup->cursors = calloc(
            cursor_count,
            sizeof(*cleanup->cursors)
        );
        if (cleanup->cursors == NULL)
        {
            result_status = EVOKE_ERR_NOMEM;
            goto done;
        }
    }
    stats.query_term_count = (uint32) term_count;
    stats.query_run_count = (uint32) cursor_count;
    result_status = evoke_topk_accumulator_init(
        &cleanup->accumulator,
        k
    );
    if (result_status != EVOKE_OK)
    {
        goto done;
    }
    average_document_length =
        (double) total_document_length /
        (double) visible_document_count;

    PG_TRY();
    {
        if (cursor_count > 0 && live_frequency_required)
        {
            if (cleanup->minimum_root_sequence > 0)
            {
                for (size_t term_index = 0;
                     term_index < term_count;
                     term_index++)
                {
                    evoke_page_query_term *term =
                        &cleanup->terms[term_index];
                    uint64 approximate_live_frequency =
                        term->plan.raw_document_frequency;

                    term->immutable_live_document_frequency =
                        term->plan.raw_document_frequency;
                    if (projection != NULL &&
                        term->query_index != UINT32_MAX)
                    {
                        approximate_live_frequency =
                            evoke_u64_saturating_add(
                                approximate_live_frequency,
                                projection->terms[term->query_index].
                                    lexical_document_frequency
                            );
                    }
                    /*
                     * Keep delta scoring independent from a baseline scan.
                     * L0 replacements can temporarily over-count the sealed
                     * version, which lowers rather than inflates their IDF.
                     */
                    term->live_document_frequency = (uint32) Min(
                        approximate_live_frequency,
                        visible_document_count
                    );
                }
            }
            else if (projection == NULL &&
                context->manifest.visible_document_count ==
                    context->manifest.document_slot_count)
            {
                for (size_t term_index = 0;
                     term_index < term_count;
                     term_index++)
                {
                    cleanup->terms[term_index].
                        immutable_live_document_frequency =
                            cleanup->terms[term_index].
                                plan.raw_document_frequency;
                    cleanup->terms[term_index].live_document_frequency =
                        cleanup->terms[term_index].
                            plan.raw_document_frequency;
                }
            }
            else
            {
                result_status = evoke_page_query_count_live_frequencies(
                    index_relation,
                    context,
                    projection,
                    cleanup,
                    term_count,
                    cursor_count,
                    &stats
                );
            }
        }
        else if (projection != NULL)
        {
            for (size_t term_index = 0;
                 term_index < term_count;
                 term_index++)
            {
                evoke_page_query_term *term = &cleanup->terms[term_index];

                if (term->query_index != UINT32_MAX)
                {
                    term->live_document_frequency =
                        projection->terms[term->query_index].
                            lexical_document_frequency;
                }
            }
        }
        if (result_status == EVOKE_OK)
        {
            result_status = evoke_page_query_prepare_term_statistics(
                context,
                visible_document_count,
                average_document_length,
                cleanup,
                term_count
            );
        }
        if (result_status == EVOKE_OK &&
            evoke_method_requires_nonoccurrence(
                context->query_contract.params.method))
        {
            for (size_t term_index = 0;
                 term_index < term_count;
                 term_index++)
            {
                const evoke_page_query_term *term =
                    &cleanup->terms[term_index];

                nonoccurrence_base_score +=
                    (double) term->query_weight *
                    (double) term->nonoccurrence;
            }
            if (!isfinite(nonoccurrence_base_score) ||
                nonoccurrence_base_score > FLT_MAX ||
                nonoccurrence_base_score < -FLT_MAX)
            {
                result_status = EVOKE_ERR_RANGE;
            }
            else
            {
                stats.nonoccurrence_base_score =
                    (float) nonoccurrence_base_score;
            }
        }
        if (result_status == EVOKE_OK && cursor_count > 0)
        {
            if (projection != NULL)
            {
                bool scored = false;

                result_status =
                    evoke_page_query_try_score_semantic_bmp(
                        index_relation,
                        context,
                        cleanup,
                        term_count,
                        average_document_length,
                        k,
                        NULL,
                        &stats,
                        &scored
                    );
                if (result_status == EVOKE_OK && !scored)
                {
                    result_status =
                        evoke_page_query_score_positive_documents(
                            index_relation,
                            context,
                            projection,
                            cleanup,
                            term_count,
                            cursor_count,
                            average_document_length,
                            &stats
                        );
                }
            }
            else if (filter != NULL)
            {
                bool scored = false;

                result_status =
                    evoke_page_query_try_score_semantic_bmp(
                        index_relation,
                        context,
                        cleanup,
                        term_count,
                        average_document_length,
                        k,
                        NULL,
                        &stats,
                        &scored
                    );
                if (result_status == EVOKE_OK && !scored)
                {
                    result_status = evoke_page_query_score_positive_blocks(
                        index_relation,
                        context,
                        cleanup,
                        term_count,
                        cursor_count,
                        visible_document_count,
                        total_document_length,
                        average_document_length,
                        &stats
                    );
                }
            }
            else if (projection == NULL)
            {
                bool scored = false;

                result_status =
                    evoke_page_query_try_score_semantic_bmp(
                        index_relation,
                        context,
                        cleanup,
                        term_count,
                        average_document_length,
                        k,
                        accelerator_seed.len > 0
                            ? &accelerator_seed
                            : NULL,
                        &stats,
                        &scored
                    );
                if (result_status == EVOKE_OK && !scored)
                {
                    result_status =
                    evoke_page_query_try_materialized_scoring(
                        index_relation,
                        context,
                        cleanup,
                        term_count,
                        average_document_length,
                        &stats,
                        &scored
                    );
                }
                if (result_status == EVOKE_OK && !scored)
                {
                    result_status =
                        evoke_page_query_try_score_term_at_a_time(
                            index_relation,
                            context,
                            cleanup,
                            term_count,
                            average_document_length,
                            &stats,
                            &scored
                        );
                }
                if (result_status == EVOKE_OK && !scored)
                {
                    result_status =
                        evoke_page_query_try_score_ordered_blocks(
                            index_relation,
                            context,
                            cleanup,
                            term_count,
                            cursor_count,
                            visible_document_count,
                            total_document_length,
                            average_document_length,
                            &stats,
                            &scored
                        );
                }
                if (result_status == EVOKE_OK && !scored)
                {
                    result_status =
                        evoke_page_query_score_positive_blocks(
                            index_relation,
                            context,
                            cleanup,
                            term_count,
                            cursor_count,
                            visible_document_count,
                            total_document_length,
                            average_document_length,
                            &stats
                        );
                }
            }
        }
        if (result_status == EVOKE_OK)
        {
            result_status = evoke_page_query_score_remaining_l0_documents(
                context,
                projection,
                cleanup,
                average_document_length,
                &stats
            );
        }
        if (result_status == EVOKE_OK)
        {
            result_status = evoke_topk_accumulator_finish(
                &cleanup->accumulator,
                true,
                result_out
            );
        }
        if (result_status == EVOKE_OK &&
            stats.nonoccurrence_base_score != 0.0f)
        {
            for (size_t result_index = 0;
                 result_index < result_out->len;
                 result_index++)
            {
                result_out->scores[result_index] +=
                    stats.nonoccurrence_base_score;
            }
        }
    }
    PG_CATCH();
    {
        evoke_topk_result_free(&accelerator_seed);
        evoke_page_query_cleanup_free(cleanup);
        PG_RE_THROW();
    }
    PG_END_TRY();

done:
    *stats_out = stats;
    if (result_status == EVOKE_OK)
    {
        stats.positive_topk_complete = result_out->len == k;
        *stats_out = stats;
    }
    else
    {
        evoke_topk_result_free(result_out);
    }
    evoke_topk_result_free(&accelerator_seed);
    evoke_page_query_cleanup_free(cleanup);
    return result_status;
}

evoke_status
evoke_page_query_positive_topk_projected(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
)
{
    return evoke_page_query_positive_topk_filtered(
        index_relation,
        context,
        projection,
        NULL,
        query_ids,
        query_weights,
        query_len,
        k,
        result_out,
        stats_out
    );
}

evoke_status
evoke_page_query_positive_topk(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
)
{
    return evoke_page_query_positive_topk_projected(
        index_relation,
        context,
        NULL,
        query_ids,
        query_weights,
        query_len,
        k,
        result_out,
        stats_out
    );
}

static bool
evoke_page_query_result_contains(
    const evoke_topk_result *result,
    uint32 document_slot
)
{
    for (size_t index = 0; index < result->len; index++)
    {
        if (result->doc_ids[index] == document_slot)
        {
            return true;
        }
    }
    return false;
}

static int
evoke_page_query_cmp_zero_candidate(const void *left, const void *right)
{
    const evoke_page_query_zero_candidate *a = left;
    const evoke_page_query_zero_candidate *b = right;

    if (a->born_sequence < b->born_sequence)
    {
        return -1;
    }
    if (a->born_sequence > b->born_sequence)
    {
        return 1;
    }
    if (a->document_slot < b->document_slot)
    {
        return -1;
    }
    if (a->document_slot > b->document_slot)
    {
        return 1;
    }
    return 0;
}

static bool
evoke_page_query_zero_candidate_precedes(
    const evoke_page_query_zero_candidate *left,
    const evoke_page_query_zero_candidate *right
)
{
    return evoke_page_query_cmp_zero_candidate(left, right) < 0;
}

static void
evoke_page_query_keep_bounded_zero_candidate(
    evoke_page_query_zero_candidate *candidates,
    size_t capacity,
    size_t *count_inout,
    const evoke_page_query_zero_candidate *candidate
)
{
    size_t count = *count_inout;
    size_t worst = 0;

    if (capacity == 0)
    {
        return;
    }
    if (count < capacity)
    {
        candidates[count] = *candidate;
        *count_inout = count + 1U;
        return;
    }
    for (size_t index = 1; index < count; index++)
    {
        if (evoke_page_query_zero_candidate_precedes(
                &candidates[worst],
                &candidates[index]))
        {
            worst = index;
        }
    }
    if (evoke_page_query_zero_candidate_precedes(
            candidate,
            &candidates[worst]))
    {
        candidates[worst] = *candidate;
    }
}

static void
evoke_page_query_free_zero_work(
    evoke_document_cow_record *cow_records,
    evoke_page_query_zero_candidate *candidates
)
{
    if (candidates != NULL)
    {
        pfree(candidates);
    }
    if (cow_records != NULL)
    {
        pfree(cow_records);
    }
}

typedef struct evoke_page_query_zero_filter_context
{
    const evoke_page_query_filter *filter;
    const evoke_page_query_l0_projection *projection;
    const evoke_topk_result *result;
} evoke_page_query_zero_filter_context;

static bool
evoke_page_query_root_zero_candidate_matches(
    void *context,
    const evoke_document_cow_record *record
)
{
    const evoke_page_query_zero_filter_context *filter_context = context;
    uint32 document_slot = (uint32) record->version.document_slot;
    const evoke_page_query_l0_document *projected_document =
        evoke_page_query_l0_find_document(
            filter_context->projection,
            document_slot
        );

    return evoke_page_query_document_allowed(
            filter_context->filter,
            document_slot) &&
        !evoke_page_query_result_contains(
            filter_context->result,
            document_slot) &&
        (projected_document == NULL ||
         !projected_document->shadows_immutable);
}

static bool
evoke_page_query_block_has_filtered_zero_candidate(
    const evoke_page_query_filter *filter,
    const evoke_page_query_l0_projection *projection,
    const evoke_topk_result *result,
    uint32 block_id,
    uint32 block_shift,
    uint64 document_count
)
{
    uint64 first_document = (uint64) block_id << block_shift;
    uint64 last_document = Min(
        document_count,
        first_document + (UINT64_C(1) << block_shift)
    );

    if (!evoke_page_query_block_has_allowed_document(
            filter,
            block_id,
            block_shift))
    {
        return false;
    }
    for (uint64 document_slot = first_document;
         document_slot < last_document;
         document_slot++)
    {
        const evoke_page_query_l0_document *projected_document;

        if (!evoke_page_query_document_allowed(
                filter,
                (uint32) document_slot) ||
            evoke_page_query_result_contains(
                result,
                (uint32) document_slot))
        {
            continue;
        }
        projected_document = evoke_page_query_l0_find_document(
            projection,
            (uint32) document_slot
        );
        if (projected_document == NULL ||
            !projected_document->shadows_immutable)
        {
            return true;
        }
    }
    return false;
}

static bool
evoke_page_query_filtered_zero_block_scan_is_lower(
    const evoke_segment_query_context *context,
    const evoke_page_query_filter *filter,
    const evoke_page_query_l0_projection *projection,
    const evoke_topk_result *result,
    size_t needed
)
{
    uint64 document_count;
    uint64 document_block_count;
    uint64 direct_record_work = 0;
    uint64 eligible_document_count;
    uint64 born_record_work;

    if (context == NULL || filter == NULL || result == NULL || needed == 0 ||
        filter->allowed_document_bitmap == NULL ||
        context->query_contract.block_shift !=
            EVOKE_DEFAULT_POSTING_BLOCK_SHIFT ||
        result->len > filter->allowed_document_count)
    {
        return false;
    }
    document_count = context->manifest.document_slot_count;
    eligible_document_count =
        filter->allowed_document_count - result->len;
    if (document_count == 0 || eligible_document_count < needed)
    {
        return false;
    }
    document_block_count =
        (document_count + EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS - 1U) /
        EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS;
    for (uint64 block_index = 0;
         block_index < document_block_count;
         block_index++)
    {
        uint64 first_document_slot;
        uint64 block_record_count;

        if (!evoke_page_query_block_has_filtered_zero_candidate(
                filter,
                projection,
                result,
                (uint32) block_index,
                context->query_contract.block_shift,
                document_count))
        {
            continue;
        }
        first_document_slot = block_index <<
            context->query_contract.block_shift;
        block_record_count = Min(
            (uint64) EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS,
            document_count - first_document_slot
        );
        direct_record_work = evoke_u64_saturating_add(
            direct_record_work,
            block_record_count
        );
    }

    /*
     * A matching born-prefix cursor expects to inspect approximately
     * needed * corpus / eligible records.  Allowed-block reads instead own
     * every record in each touched score block.  Compare those two physical
     * record workloads before choosing; both paths remain exact and use the
     * same immutable COW authority.
     */
    born_record_work = evoke_u64_saturating_mul(
        (uint64) needed,
        document_count
    );
    born_record_work = born_record_work / eligible_document_count +
        (born_record_work % eligible_document_count != 0 ? 1U : 0U);
    return direct_record_work < born_record_work;
}

static evoke_status
evoke_page_query_collect_filtered_zero_scores_by_block(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const evoke_page_query_filter *filter,
    const evoke_topk_result *result,
    evoke_page_query_zero_candidate *candidates,
    size_t candidate_capacity,
    size_t *candidate_count_out,
    evoke_page_query_stats *stats
)
{
    evoke_segment_query_document_reader document_reader;
    evoke_segment_query_document_block document_block;
    uint64 document_count = context->manifest.document_slot_count;
    uint64 document_block_count =
        (document_count + EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS - 1U) /
        EVOKE_SEGMENT_QUERY_BLOCK_MAX_POSTINGS;
    size_t candidate_count = 0;

    memset(&document_reader, 0, sizeof(document_reader));
    memset(&document_block, 0, sizeof(document_block));
    *candidate_count_out = 0;
    for (uint64 block_index = 0;
         block_index < document_block_count;
         block_index++)
    {
        if (!evoke_page_query_block_has_filtered_zero_candidate(
                filter,
                projection,
                result,
                (uint32) block_index,
                context->query_contract.block_shift,
                document_count))
        {
            continue;
        }
        evoke_segment_pages_load_query_document_block(
            index_relation,
            context,
            &document_reader,
            (uint32) block_index,
            &document_block
        );
        stats->document_block_reads = evoke_u64_saturating_add(
            stats->document_block_reads,
            1
        );
        stats->zero_score_cow_records_examined =
            evoke_u64_saturating_add(
                stats->zero_score_cow_records_examined,
                document_block.record_count
            );
        for (uint32 local_slot = 0;
             local_slot < document_block.record_count;
             local_slot++)
        {
            const evoke_document_cow_record *record =
                &document_block.records[local_slot];
            uint32 document_slot =
                document_block.first_document_slot + local_slot;
            const evoke_page_query_l0_document *projected_document;
            evoke_page_query_zero_candidate candidate;

            if (record->version.document_slot != document_slot)
            {
                return EVOKE_ERR_FORMAT;
            }
            projected_document = evoke_page_query_l0_find_document(
                projection,
                document_slot
            );
            if (!evoke_page_query_document_allowed(filter, document_slot) ||
                !evoke_page_query_document_is_live(record) ||
                evoke_page_query_result_contains(result, document_slot) ||
                (projected_document != NULL &&
                 projected_document->shadows_immutable))
            {
                continue;
            }
            candidate.born_sequence = record->version.born_sequence;
            candidate.document_slot = document_slot;
            evoke_page_query_keep_bounded_zero_candidate(
                candidates,
                candidate_capacity,
                &candidate_count,
                &candidate
            );
        }
        CHECK_FOR_INTERRUPTS();
    }
    *candidate_count_out = candidate_count;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_complete_filtered_zero_scores(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const evoke_page_query_filter *filter,
    size_t k,
    evoke_topk_result *result,
    evoke_page_query_stats *stats
)
{
    evoke_document_cow_record *root_records = NULL;
    evoke_page_query_zero_candidate *candidates = NULL;
    evoke_page_query_zero_filter_context filter_context;
    evoke_document_cow_born_prefix_stats born_stats;
    size_t root_record_count = 0;
    size_t projected_candidate_count = 0;
    size_t candidate_count = 0;
    size_t needed;
    bool use_block_scan;
    uint32 *doc_ids;
    float *scores;
    evoke_status status;

    if (result->len >= k)
    {
        stats->topk_complete = true;
        return EVOKE_OK;
    }
    needed = k - result->len;
    if (needed == 0 || needed > filter->allowed_document_count ||
        needed > MaxAllocSize / sizeof(*root_records) ||
        needed > SIZE_MAX / 2U ||
        needed * 2U > MaxAllocSize / sizeof(*candidates))
    {
        return EVOKE_ERR_RANGE;
    }
    candidates = palloc(sizeof(*candidates) * needed * 2U);
    filter_context.filter = filter;
    filter_context.projection = projection;
    filter_context.result = result;
    memset(&born_stats, 0, sizeof(born_stats));
    use_block_scan = evoke_page_query_filtered_zero_block_scan_is_lower(
        context,
        filter,
        projection,
        result,
        needed
    );
    if (use_block_scan)
    {
        status = evoke_page_query_collect_filtered_zero_scores_by_block(
            index_relation,
            context,
            projection,
            filter,
            result,
            candidates,
            needed,
            &root_record_count,
            stats
        );
        if (status != EVOKE_OK)
        {
            evoke_page_query_free_zero_work(NULL, candidates);
            return status;
        }
    }
    else
    {
        root_records = palloc(sizeof(*root_records) * needed);
        evoke_segment_pages_load_matching_live_born_prefix(
            index_relation,
            &context->root,
            &context->manifest,
            needed,
            evoke_page_query_root_zero_candidate_matches,
            &filter_context,
            root_records,
            needed,
            &root_record_count,
            &born_stats
        );
        for (size_t index = 0; index < root_record_count; index++)
        {
            candidates[index].born_sequence =
                root_records[index].version.born_sequence;
            candidates[index].document_slot =
                (uint32) root_records[index].version.document_slot;
        }
    }
    candidate_count = root_record_count;
    if (projection != NULL)
    {
        for (size_t index = 0; index < projection->document_count; index++)
        {
            const evoke_page_query_l0_document *document =
                &projection->documents[index];
            evoke_page_query_zero_candidate candidate;

            if ((index & EVOKE_PAGE_QUERY_INTERRUPT_MASK) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
            if (!document->live || !document->shadows_immutable ||
                !evoke_page_query_document_allowed(
                    filter,
                    document->document_slot) ||
                evoke_page_query_result_contains(
                    result,
                    document->document_slot))
            {
                continue;
            }
            candidate.born_sequence = document->born_sequence;
            candidate.document_slot = document->document_slot;
            evoke_page_query_keep_bounded_zero_candidate(
                &candidates[root_record_count],
                needed,
                &projected_candidate_count,
                &candidate
            );
        }
    }
    candidate_count = root_record_count + projected_candidate_count;
    qsort(
        candidates,
        candidate_count,
        sizeof(*candidates),
        evoke_page_query_cmp_zero_candidate
    );

    doc_ids = realloc(result->doc_ids, sizeof(*doc_ids) * k);
    if (doc_ids == NULL)
    {
        evoke_page_query_free_zero_work(root_records, candidates);
        return EVOKE_ERR_NOMEM;
    }
    result->doc_ids = doc_ids;
    scores = realloc(result->scores, sizeof(*scores) * k);
    if (scores == NULL)
    {
        evoke_page_query_free_zero_work(root_records, candidates);
        return EVOKE_ERR_NOMEM;
    }
    result->scores = scores;
    for (size_t index = 0;
         index < candidate_count && result->len < k;
         index++)
    {
        result->doc_ids[result->len] = candidates[index].document_slot;
        result->scores[result->len] = stats->nonoccurrence_base_score;
        result->len++;
        stats->zero_score_documents_added++;
    }
    stats->zero_score_cow_objects_loaded = evoke_u64_saturating_add(
        stats->zero_score_cow_objects_loaded,
        born_stats.objects_loaded
    );
    stats->zero_score_cow_records_examined = evoke_u64_saturating_add(
        stats->zero_score_cow_records_examined,
        born_stats.records_examined
    );
    stats->zero_score_heap_peak = Max(
        stats->zero_score_heap_peak,
        born_stats.heap_peak
    );
    stats->topk_complete = result->len == k;
    evoke_page_query_free_zero_work(root_records, candidates);
    return stats->topk_complete ? EVOKE_OK : EVOKE_ERR_FORMAT;
}

static evoke_status
evoke_page_query_complete_zero_scores(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const evoke_page_query_filter *filter,
    size_t k,
    evoke_topk_result *result,
    evoke_page_query_stats *stats
)
{
    evoke_document_cow_record *cow_records = NULL;
    evoke_page_query_zero_candidate *candidates = NULL;
    evoke_document_cow_born_prefix_stats born_stats;
    uint64 visible_document_count = projection == NULL
        ? context->manifest.visible_document_count
        : projection->visible_document_count;
    size_t projection_count = projection == NULL
        ? 0
        : projection->document_count;
    size_t needed;
    size_t prefix_limit;
    size_t cow_record_count = 0;
    size_t candidate_capacity;
    size_t candidate_count = 0;
    uint32 *doc_ids;
    float *scores;

    if (filter != NULL)
    {
        return evoke_page_query_complete_filtered_zero_scores(
            index_relation,
            context,
            projection,
            filter,
            k,
            result,
            stats
        );
    }

    if (result->len >= k)
    {
        stats->topk_complete = true;
        return EVOKE_OK;
    }
    needed = k - result->len;
    if (needed > SIZE_MAX - result->len ||
        needed + result->len > SIZE_MAX - projection_count)
    {
        return EVOKE_ERR_RANGE;
    }
    prefix_limit = needed + result->len + projection_count;
    if ((uint64) prefix_limit > context->manifest.visible_document_count)
    {
        prefix_limit = (size_t) context->manifest.visible_document_count;
    }
    if (prefix_limit > 0)
    {
        if (prefix_limit > MaxAllocSize / sizeof(*cow_records))
        {
            return EVOKE_ERR_RANGE;
        }
        cow_records = palloc(sizeof(*cow_records) * prefix_limit);
    }
    memset(&born_stats, 0, sizeof(born_stats));
    evoke_segment_pages_load_live_born_prefix(
        index_relation,
        &context->root,
        &context->manifest,
        prefix_limit,
        cow_records,
        prefix_limit,
        &cow_record_count,
        &born_stats
    );
    if (cow_record_count > SIZE_MAX - projection_count)
    {
        evoke_page_query_free_zero_work(cow_records, NULL);
        return EVOKE_ERR_RANGE;
    }
    candidate_capacity = cow_record_count + projection_count;
    if (candidate_capacity > 0)
    {
        if (candidate_capacity > MaxAllocSize / sizeof(*candidates))
        {
            evoke_page_query_free_zero_work(cow_records, NULL);
            return EVOKE_ERR_RANGE;
        }
        candidates = palloc(sizeof(*candidates) * candidate_capacity);
    }

    for (size_t index = 0; index < cow_record_count; index++)
    {
        const evoke_document_cow_record *record = &cow_records[index];
        uint32 document_slot = (uint32) record->version.document_slot;
        const evoke_page_query_l0_document *projected_document =
            evoke_page_query_l0_find_document(projection, document_slot);

        if ((index & EVOKE_PAGE_QUERY_INTERRUPT_MASK) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
        if ((projected_document != NULL &&
             projected_document->shadows_immutable) ||
            evoke_page_query_result_contains(result, document_slot))
        {
            continue;
        }
        candidates[candidate_count].born_sequence =
            record->version.born_sequence;
        candidates[candidate_count].document_slot = document_slot;
        candidate_count++;
    }
    if (projection != NULL)
    {
        for (size_t index = 0; index < projection->document_count; index++)
        {
            const evoke_page_query_l0_document *document =
                &projection->documents[index];

            if ((index & EVOKE_PAGE_QUERY_INTERRUPT_MASK) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
            if (!document->live || !document->shadows_immutable ||
                evoke_page_query_result_contains(
                    result,
                    document->document_slot))
            {
                continue;
            }
            candidates[candidate_count].born_sequence =
                document->born_sequence;
            candidates[candidate_count].document_slot =
                document->document_slot;
            candidate_count++;
        }
    }
    if (candidate_count > 1)
    {
        qsort(
            candidates,
            candidate_count,
            sizeof(*candidates),
            evoke_page_query_cmp_zero_candidate
        );
    }

    doc_ids = realloc(result->doc_ids, sizeof(*doc_ids) * k);
    if (doc_ids == NULL)
    {
        evoke_page_query_free_zero_work(cow_records, candidates);
        return EVOKE_ERR_NOMEM;
    }
    result->doc_ids = doc_ids;
    scores = realloc(result->scores, sizeof(*scores) * k);
    if (scores == NULL)
    {
        evoke_page_query_free_zero_work(cow_records, candidates);
        return EVOKE_ERR_NOMEM;
    }
    result->scores = scores;
    for (size_t index = 0;
         index < candidate_count && result->len < k;
         index++)
    {
        result->doc_ids[result->len] = candidates[index].document_slot;
        result->scores[result->len] = stats->nonoccurrence_base_score;
        result->len++;
        stats->zero_score_documents_added++;
    }
    stats->zero_score_cow_objects_loaded = born_stats.objects_loaded;
    stats->zero_score_cow_records_examined = born_stats.records_examined;
    stats->zero_score_heap_peak = born_stats.heap_peak;
    stats->topk_complete = result->len == k && k <= visible_document_count;
    evoke_page_query_free_zero_work(cow_records, candidates);
    return stats->topk_complete ? EVOKE_OK : EVOKE_ERR_FORMAT;
}

evoke_status
evoke_page_query_topk_filtered(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const evoke_page_query_filter *filter,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
)
{
    evoke_status status = evoke_page_query_positive_topk_filtered(
        index_relation,
        context,
        projection,
        filter,
        query_ids,
        query_weights,
        query_len,
        k,
        result_out,
        stats_out
    );

    if (status != EVOKE_OK)
    {
        return status;
    }
    if (result_out->len == k)
    {
        stats_out->topk_complete = true;
        return EVOKE_OK;
    }
    status = evoke_page_query_complete_zero_scores(
        index_relation,
        context,
        projection,
        filter,
        k,
        result_out,
        stats_out
    );
    if (status != EVOKE_OK)
    {
        evoke_topk_result_free(result_out);
    }
    return status;
}

evoke_status
evoke_page_query_topk_projected(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_l0_projection *projection,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
)
{
    return evoke_page_query_topk_filtered(
        index_relation,
        context,
        projection,
        NULL,
        query_ids,
        query_weights,
        query_len,
        k,
        result_out,
        stats_out
    );
}

evoke_status
evoke_page_query_topk(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_page_query_stats *stats_out
)
{
    return evoke_page_query_topk_projected(
        index_relation,
        context,
        NULL,
        query_ids,
        query_weights,
        query_len,
        k,
        result_out,
        stats_out
    );
}

typedef struct evoke_page_query_cost_work
{
    double *upper_bounds;
    float *term_maxima;
    uint32 *term_epochs;
    uint32 *touched_blocks;
    uint32 *term_counts;
    uint32 *posting_counts;
    uint32 epoch;
    size_t touched_count;
} evoke_page_query_cost_work;

typedef struct evoke_page_query_cost_semantic_term
{
    size_t term_index;
    double global_cap;
    double global_floor;
    double global_absolute_cap;
    uint64 super_ref_count;
    uint64 ref_count;
    uint64 posting_count;
} evoke_page_query_cost_semantic_term;

static int
evoke_page_query_cost_semantic_term_compare(const void *left, const void *right)
{
    const evoke_page_query_cost_semantic_term *first = left;
    const evoke_page_query_cost_semantic_term *second = right;

    if (first->global_cap > second->global_cap)
    {
        return -1;
    }
    if (first->global_cap < second->global_cap)
    {
        return 1;
    }
    if (first->term_index < second->term_index)
    {
        return -1;
    }
    return first->term_index > second->term_index ? 1 : 0;
}

static uint32
evoke_page_query_cost_projection_points(
    uint32 semantic_term_count,
    uint32 *points_out
)
{
    static const uint32 candidates[] = {
        0, 1, 2, 4, 8, 12, 16, 20, 24, 32,
        40, 48, 56, 64, 72, 96, 128, 192, 256
    };
    uint32 point_count = 0;

    for (size_t index = 0; index < lengthof(candidates); index++)
    {
        if (candidates[index] > semantic_term_count)
        {
            break;
        }
        points_out[point_count++] = candidates[index];
    }
    if (point_count == 0 || points_out[point_count - 1] != semantic_term_count)
    {
        if (point_count >= EVOKE_PAGE_QUERY_COST_MAX_ESSENTIAL_PROJECTIONS)
        {
            point_count--;
        }
        points_out[point_count++] = semantic_term_count;
    }
    return point_count;
}

static void
evoke_page_query_cost_heap_offer(
    float *heap,
    size_t *heap_len,
    size_t heap_capacity,
    float value
)
{
    size_t index;

    if (*heap_len < heap_capacity)
    {
        index = (*heap_len)++;
        heap[index] = value;
        while (index > 0)
        {
            size_t parent = (index - 1U) / 2U;
            float swap;

            if (heap[parent] <= heap[index])
            {
                break;
            }
            swap = heap[parent];
            heap[parent] = heap[index];
            heap[index] = swap;
            index = parent;
        }
        return;
    }
    if (value <= heap[0])
    {
        return;
    }
    heap[0] = value;
    index = 0;
    for (;;)
    {
        size_t left = index * 2U + 1U;
        size_t right = left + 1U;
        size_t smallest = index;
        float swap;

        if (left < heap_capacity && heap[left] < heap[smallest])
        {
            smallest = left;
        }
        if (right < heap_capacity && heap[right] < heap[smallest])
        {
            smallest = right;
        }
        if (smallest == index)
        {
            break;
        }
        swap = heap[index];
        heap[index] = heap[smallest];
        heap[smallest] = swap;
        index = smallest;
    }
}

static evoke_status
evoke_page_query_cost_kth_lower_bound(
    const double *partial_scores,
    const double *partial_absolute_sums,
    uint64 document_slot_count,
    size_t k,
    uint32 addition_count_bound,
    double residual_global_floor,
    double residual_global_absolute_cap,
    float *lower_bound_out
)
{
    double unit_roundoff = (double) FLT_EPSILON / 2.0;
    double error_product = addition_count_bound * unit_roundoff;
    double gamma;
    float *heap;
    size_t heap_len = 0;

    if (partial_scores == NULL || partial_absolute_sums == NULL ||
        lower_bound_out == NULL || k == 0 || k > document_slot_count ||
        !isfinite(residual_global_floor) ||
        !isfinite(residual_global_absolute_cap) ||
        residual_global_floor > 0.0 ||
        residual_global_absolute_cap < 0.0 || error_product >= 1.0 ||
        k > MaxAllocSize / sizeof(float))
    {
        return EVOKE_ERR_INVALID;
    }
    gamma = error_product / (1.0 - error_product);
    heap = palloc(sizeof(*heap) * k);
    for (uint64 document = 0; document < document_slot_count; document++)
    {
        double error_bound = gamma * (
            partial_absolute_sums[document] +
            residual_global_absolute_cap
        );
        double lower = partial_scores[document] +
            residual_global_floor - error_bound;
        float value;

        if (!isfinite(lower) || !isfinite(error_bound))
        {
            pfree(heap);
            return EVOKE_ERR_RANGE;
        }
        if (lower <= 0.0)
        {
            value = 0.0f;
        }
        else if (lower >= (double) FLT_MAX)
        {
            value = nextafterf(FLT_MAX, -INFINITY);
        }
        else
        {
            value = nextafterf((float) lower, -INFINITY);
            if (value < 0.0f)
            {
                value = 0.0f;
            }
        }
        evoke_page_query_cost_heap_offer(&heap[0], &heap_len, k, value);
    }
    if (heap_len != k)
    {
        pfree(heap);
        return EVOKE_ERR_FORMAT;
    }
    *lower_bound_out = heap[0];
    pfree(heap);
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_cost_capture_essential_projection(
    uint64 document_slot_count,
    size_t k,
    uint32 addition_count_bound,
    float full_kth_score,
    const evoke_page_query_cost_level *level,
    const evoke_page_query_cost_work *work,
    const double *partial_scores,
    const double *partial_absolute_sums,
    uint32 essential_semantic_terms,
    uint32 semantic_term_count,
    double residual_global_cap,
    double residual_global_floor,
    double residual_global_absolute_cap,
    uint64 semantic_super_refs,
    uint64 semantic_refs,
    uint64 semantic_postings,
    evoke_page_query_cost_essential_projection *projection,
    uint8 *competitive_bitmap
)
{
    uint64 block_size = UINT64_C(1) << level->block_shift;
    evoke_status status;

    memset(projection, 0, sizeof(*projection));
    projection->essential_semantic_terms = essential_semantic_terms;
    projection->residual_semantic_terms = semantic_term_count -
        essential_semantic_terms;
    projection->essential_semantic_super_refs = semantic_super_refs;
    projection->essential_semantic_refs = semantic_refs;
    projection->essential_semantic_postings = semantic_postings;
    projection->residual_global_cap = residual_global_cap <= 0.0
        ? 0.0f
        : residual_global_cap > FLT_MAX
            ? INFINITY
            : nextafterf((float) residual_global_cap, INFINITY);
    projection->residual_global_floor = residual_global_floor >= 0.0
        ? 0.0f
        : residual_global_floor < -(double) FLT_MAX
            ? -INFINITY
            : nextafterf((float) residual_global_floor, -INFINITY);
    projection->residual_global_absolute_cap =
        residual_global_absolute_cap <= 0.0
            ? 0.0f
            : residual_global_absolute_cap > FLT_MAX
                ? INFINITY
                : nextafterf(
                    (float) residual_global_absolute_cap,
                    INFINITY
                );
    status = evoke_page_query_cost_kth_lower_bound(
        partial_scores,
        partial_absolute_sums,
        document_slot_count,
        k,
        addition_count_bound,
        residual_global_floor,
        residual_global_absolute_cap,
        &projection->kth_lower_bound
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    projection->kth_lower_bound_safe =
        projection->kth_lower_bound <= full_kth_score;

    for (uint64 block_id = 0; block_id < level->block_count; block_id++)
    {
        double upper_bound = nextafter(
            work->upper_bounds[block_id] + residual_global_cap,
            INFINITY
        );
        uint64 first_document_slot;
        uint64 document_count;

        if (upper_bound < (double) projection->kth_lower_bound)
        {
            continue;
        }
        competitive_bitmap[block_id >> 3] |=
            (uint8) (UINT8_C(1) << (block_id & 7U));
        first_document_slot = block_id * block_size;
        document_count = Min(
            block_size,
            document_slot_count - first_document_slot
        );
        if (projection->competitive_blocks == UINT64_MAX ||
            projection->competitive_document_slots >
                UINT64_MAX - document_count)
        {
            return EVOKE_ERR_RANGE;
        }
        projection->competitive_blocks++;
        projection->competitive_document_slots += document_count;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_cost_finalize_essential_projections(
    float kth_score,
    const evoke_page_query_cost_level *level,
    const evoke_page_query_cost_work *work,
    evoke_page_query_cost_audit *audit,
    uint8 **competitive_bitmaps
)
{
    for (uint32 projection_index = 0;
         projection_index < audit->essential_projection_count;
         projection_index++)
    {
        evoke_page_query_cost_essential_projection *projection =
            &audit->essential_projections[projection_index];
        const uint8 *bitmap = competitive_bitmaps[projection_index];

        for (uint64 block_id = 0; block_id < level->block_count; block_id++)
        {
            bool selected = (bitmap[block_id >> 3] &
                (uint8) (UINT8_C(1) << (block_id & 7U))) != 0;

            if (selected)
            {
                if (projection->competitive_postings >
                    UINT64_MAX - work->posting_counts[block_id])
                {
                    return EVOKE_ERR_RANGE;
                }
                projection->competitive_postings +=
                    work->posting_counts[block_id];
            }
            if (work->upper_bounds[block_id] >= (double) kth_score &&
                !selected)
            {
                projection->missed_full_competitive_blocks++;
            }
        }
    }
    return EVOKE_OK;
}

static void
evoke_page_query_cost_work_free(evoke_page_query_cost_work *work)
{
    if (work == NULL)
    {
        return;
    }
    if (work->upper_bounds != NULL)
    {
        pfree(work->upper_bounds);
    }
    if (work->term_maxima != NULL)
    {
        pfree(work->term_maxima);
    }
    if (work->term_epochs != NULL)
    {
        pfree(work->term_epochs);
    }
    if (work->touched_blocks != NULL)
    {
        pfree(work->touched_blocks);
    }
    if (work->term_counts != NULL)
    {
        pfree(work->term_counts);
    }
    if (work->posting_counts != NULL)
    {
        pfree(work->posting_counts);
    }
    memset(work, 0, sizeof(*work));
}

static evoke_status
evoke_page_query_cost_work_init(
    uint64 document_slot_count,
    evoke_page_query_cost_level *level,
    evoke_page_query_cost_work *work
)
{
    uint64 block_size;
    uint64 block_count;

    if (document_slot_count == 0 || level == NULL || work == NULL ||
        level->block_shift >= 63)
    {
        return EVOKE_ERR_INVALID;
    }
    block_size = UINT64_C(1) << level->block_shift;
    block_count = document_slot_count / block_size;
    if (document_slot_count % block_size != 0)
    {
        block_count++;
    }
    if (block_count == 0 || block_count > MaxAllocSize / sizeof(double) ||
        block_count > MaxAllocSize / sizeof(float) ||
        block_count > MaxAllocSize / sizeof(uint32))
    {
        return EVOKE_ERR_RANGE;
    }
    memset(work, 0, sizeof(*work));
    level->block_count = block_count;
    work->upper_bounds = palloc0(sizeof(double) * (Size) block_count);
    work->term_maxima = palloc(sizeof(float) * (Size) block_count);
    work->term_epochs = palloc0(sizeof(uint32) * (Size) block_count);
    work->touched_blocks = palloc(sizeof(uint32) * (Size) block_count);
    work->term_counts = palloc0(sizeof(uint32) * (Size) block_count);
    work->posting_counts = palloc0(sizeof(uint32) * (Size) block_count);
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_cost_record_posting(
    uint32 document_slot,
    float contribution,
    evoke_page_query_cost_level *levels,
    evoke_page_query_cost_work *work
)
{
    for (size_t level_index = 0;
         level_index < EVOKE_PAGE_QUERY_COST_LEVEL_COUNT;
         level_index++)
    {
        evoke_page_query_cost_level *level = &levels[level_index];
        evoke_page_query_cost_work *level_work = &work[level_index];
        uint32 block_id = document_slot >> level->block_shift;

        if ((uint64) block_id >= level->block_count ||
            level_work->posting_counts[block_id] == UINT32_MAX)
        {
            return EVOKE_ERR_RANGE;
        }
        level_work->posting_counts[block_id]++;
        if (level_work->term_epochs[block_id] != level_work->epoch)
        {
            if (level_work->touched_count >= level->block_count)
            {
                return EVOKE_ERR_RANGE;
            }
            level_work->term_epochs[block_id] = level_work->epoch;
            level_work->term_maxima[block_id] = 0.0f;
            level_work->touched_blocks[level_work->touched_count++] =
                block_id;
        }
        if (contribution > level_work->term_maxima[block_id])
        {
            level_work->term_maxima[block_id] = contribution;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_cost_finish_term(
    evoke_page_query_cost_level *levels,
    evoke_page_query_cost_work *work
)
{
    for (size_t level_index = 0;
         level_index < EVOKE_PAGE_QUERY_COST_LEVEL_COUNT;
         level_index++)
    {
        evoke_page_query_cost_level *level = &levels[level_index];
        evoke_page_query_cost_work *level_work = &work[level_index];

        if (level->touched_term_blocks >
            UINT64_MAX - level_work->touched_count)
        {
            return EVOKE_ERR_RANGE;
        }
        level->touched_term_blocks += level_work->touched_count;
        for (size_t touched_index = 0;
             touched_index < level_work->touched_count;
             touched_index++)
        {
            uint32 block_id = level_work->touched_blocks[touched_index];

            if (level_work->term_counts[block_id] == UINT32_MAX)
            {
                return EVOKE_ERR_RANGE;
            }
            level_work->term_counts[block_id]++;
            level_work->upper_bounds[block_id] = nextafter(
                level_work->upper_bounds[block_id] +
                    (double) level_work->term_maxima[block_id],
                INFINITY
            );
        }
        level_work->touched_count = 0;
        if (level_work->epoch == UINT32_MAX)
        {
            memset(
                level_work->term_epochs,
                0,
                sizeof(uint32) * (Size) level->block_count
            );
            level_work->epoch = 1;
        }
        else
        {
            level_work->epoch++;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_cost_process_term(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_term *term,
    const uint32 *document_lengths,
    double average_document_length,
    uint64 document_slot_count,
    double *partial_scores,
    double *partial_absolute_sums,
    uint32 *posting_document_slots,
    evoke_posting_value *posting_values,
    evoke_page_query_cost_audit *audit,
    evoke_page_query_cost_work *work
)
{
    if (partial_scores == NULL || partial_absolute_sums == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint32 run_index = 0;
         run_index < term->plan.run_count;
         run_index++)
    {
        const evoke_segment_query_run *run = &term->plan.runs[run_index];
        uint64 first_posting_index = 0;

        while (first_posting_index < run->posting_count)
        {
            uint32 posting_count;

            CHECK_FOR_INTERRUPTS();
            posting_count = evoke_segment_pages_load_query_term_posting_window(
                index_relation,
                context,
                &term->plan,
                run_index,
                first_posting_index,
                posting_document_slots,
                posting_values,
                EVOKE_SEGMENT_QUERY_POSTING_WINDOW
            );
            if (posting_count == 0 ||
                posting_count > run->posting_count - first_posting_index)
            {
                return EVOKE_ERR_FORMAT;
            }
            for (uint32 posting_index = 0;
                 posting_index < posting_count;
                 posting_index++)
            {
                uint32 document_slot =
                    posting_document_slots[posting_index];
                float contribution;
                evoke_status status;

                if ((uint64) document_slot >= document_slot_count)
                {
                    return EVOKE_ERR_FORMAT;
                }
                if (run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
                {
                    double tfc = evoke_score_tfc(
                        context->query_contract.params.method,
                        (double) posting_values[
                            posting_index
                        ].term_frequency,
                        (double) document_lengths[document_slot],
                        average_document_length,
                        context->query_contract.params.k1,
                        context->query_contract.params.b,
                        context->query_contract.params.delta
                    );

                    contribution = (float) (
                        (double) term->query_weight * term->idf * tfc
                    );
                }
                else
                {
                    contribution = term->query_weight *
                        posting_values[posting_index].impact;
                }
                if (!isfinite(contribution))
                {
                    return EVOKE_ERR_RANGE;
                }
                partial_scores[document_slot] += (double) contribution;
                partial_absolute_sums[document_slot] +=
                    fabs((double) contribution);
                if (!isfinite(partial_scores[document_slot]) ||
                    !isfinite(partial_absolute_sums[document_slot]))
                {
                    return EVOKE_ERR_RANGE;
                }
                status = evoke_page_query_cost_record_posting(
                    document_slot,
                    contribution,
                    audit->levels,
                    work
                );
                if (status != EVOKE_OK)
                {
                    return status;
                }
                if (audit->query_postings == UINT64_MAX)
                {
                    return EVOKE_ERR_RANGE;
                }
                audit->query_postings++;
            }
            first_posting_index += posting_count;
        }
    }
    return evoke_page_query_cost_finish_term(audit->levels, work);
}

static evoke_status
evoke_page_query_cost_finalize_hierarchy(
    float kth_score,
    uint32 query_term_count,
    evoke_page_query_cost_audit *audit,
    evoke_page_query_cost_work *work
)
{
    size_t top_level = EVOKE_PAGE_QUERY_COST_LEVEL_COUNT - 1;

    for (size_t reverse_index = 0;
         reverse_index < EVOKE_PAGE_QUERY_COST_LEVEL_COUNT;
         reverse_index++)
    {
        size_t level_index = top_level - reverse_index;
        const evoke_page_query_cost_level *level =
            &audit->levels[level_index];
        const evoke_page_query_cost_work *level_work = &work[level_index];
        const evoke_page_query_cost_work *parent_work = level_index == top_level
            ? NULL
            : &work[level_index + 1];
        uint8 parent_shift = level_index == top_level
            ? 0
            : audit->levels[level_index + 1].block_shift;

        for (uint64 block_id = 0;
             block_id < level->block_count;
             block_id++)
        {
            bool visited = parent_work == NULL;

            if (parent_work != NULL)
            {
                uint8 shift_delta = parent_shift - level->block_shift;
                uint64 parent_id = block_id >> shift_delta;

                if (parent_id >= audit->levels[
                        level_index + 1
                    ].block_count)
                {
                    return EVOKE_ERR_FORMAT;
                }
                visited = parent_work->upper_bounds[parent_id] >=
                    (double) kth_score;
            }
            if (!visited)
            {
                continue;
            }
            if (audit->hierarchy_nodes_read == UINT64_MAX ||
                audit->hierarchy_sparse_term_hits >
                    UINT64_MAX - level_work->term_counts[block_id] ||
                audit->hierarchy_dense_term_probes >
                    UINT64_MAX - query_term_count)
            {
                return EVOKE_ERR_RANGE;
            }
            audit->hierarchy_nodes_read++;
            audit->hierarchy_sparse_term_hits +=
                level_work->term_counts[block_id];
            audit->hierarchy_dense_term_probes += query_term_count;
            if (level_index == 0 &&
                level_work->upper_bounds[block_id] >=
                    (double) kth_score)
            {
                if (audit->hierarchy_leaf_blocks == UINT64_MAX ||
                    audit->hierarchy_leaf_postings >
                        UINT64_MAX -
                            level_work->posting_counts[block_id])
                {
                    return EVOKE_ERR_RANGE;
                }
                audit->hierarchy_leaf_blocks++;
                audit->hierarchy_leaf_postings +=
                    level_work->posting_counts[block_id];
            }
        }
    }
    return EVOKE_OK;
}

static bool
evoke_page_query_cost_plan_is_better(
    uint64 candidate_cost,
    uint64 candidate_nodes,
    uint64 current_cost,
    uint64 current_nodes
)
{
    return candidate_cost < current_cost ||
        (candidate_cost == current_cost && candidate_nodes < current_nodes);
}

static evoke_status
evoke_page_query_cost_finalize_optimal_plans(
    float kth_score,
    uint32 query_term_count,
    evoke_page_query_cost_audit *audit,
    evoke_page_query_cost_work *work
)
{
    uint64 sparse_cost[EVOKE_PAGE_QUERY_COST_LEVEL_COUNT];
    uint64 dense_cost[EVOKE_PAGE_QUERY_COST_LEVEL_COUNT];
    uint64 sparse_nodes[EVOKE_PAGE_QUERY_COST_LEVEL_COUNT];
    uint64 dense_nodes[EVOKE_PAGE_QUERY_COST_LEVEL_COUNT];
    uint16 sparse_mask[EVOKE_PAGE_QUERY_COST_LEVEL_COUNT];
    uint16 dense_mask[EVOKE_PAGE_QUERY_COST_LEVEL_COUNT];

    for (size_t level_index = 0;
         level_index < EVOKE_PAGE_QUERY_COST_LEVEL_COUNT;
         level_index++)
    {
        const evoke_page_query_cost_level *level =
            &audit->levels[level_index];

        if (query_term_count > 0 &&
            level->block_count > UINT64_MAX / query_term_count)
        {
            return EVOKE_ERR_RANGE;
        }
        sparse_cost[level_index] = level->touched_term_blocks;
        dense_cost[level_index] = level->block_count * query_term_count;
        sparse_nodes[level_index] = level->block_count;
        dense_nodes[level_index] = level->block_count;
        sparse_mask[level_index] = (uint16) (UINT16_C(1) << level_index);
        dense_mask[level_index] = sparse_mask[level_index];
    }
    for (size_t reverse_index = 0;
         reverse_index < EVOKE_PAGE_QUERY_COST_LEVEL_COUNT;
         reverse_index++)
    {
        size_t parent_index = EVOKE_PAGE_QUERY_COST_LEVEL_COUNT -
            reverse_index - 1;
        const evoke_page_query_cost_level *parent =
            &audit->levels[parent_index];
        const evoke_page_query_cost_work *parent_work =
            &work[parent_index];

        for (size_t child_index = 0;
             child_index < parent_index;
             child_index++)
        {
            const evoke_page_query_cost_level *child =
                &audit->levels[child_index];
            const evoke_page_query_cost_work *child_work =
                &work[child_index];
            uint8 shift_delta = parent->block_shift - child->block_shift;
            uint64 transition_nodes = 0;
            uint64 transition_sparse = 0;
            uint64 transition_dense;
            uint64 candidate_sparse;
            uint64 candidate_dense;
            uint64 candidate_sparse_nodes;
            uint64 candidate_dense_nodes;

            for (uint64 block_id = 0;
                 block_id < child->block_count;
                 block_id++)
            {
                uint64 parent_id = block_id >> shift_delta;

                if (parent_id >= parent->block_count)
                {
                    return EVOKE_ERR_FORMAT;
                }
                if (parent_work->upper_bounds[parent_id] <
                    (double) kth_score)
                {
                    continue;
                }
                if (transition_nodes == UINT64_MAX ||
                    transition_sparse > UINT64_MAX -
                        child_work->term_counts[block_id])
                {
                    return EVOKE_ERR_RANGE;
                }
                transition_nodes++;
                transition_sparse += child_work->term_counts[block_id];
            }
            if (query_term_count > 0 &&
                transition_nodes > UINT64_MAX / query_term_count)
            {
                return EVOKE_ERR_RANGE;
            }
            transition_dense = transition_nodes * query_term_count;
            if (sparse_cost[parent_index] > UINT64_MAX -
                    transition_sparse ||
                dense_cost[parent_index] > UINT64_MAX -
                    transition_dense ||
                sparse_nodes[parent_index] > UINT64_MAX -
                    transition_nodes ||
                dense_nodes[parent_index] > UINT64_MAX -
                    transition_nodes)
            {
                return EVOKE_ERR_RANGE;
            }
            candidate_sparse = sparse_cost[parent_index] +
                transition_sparse;
            candidate_dense = dense_cost[parent_index] + transition_dense;
            candidate_sparse_nodes = sparse_nodes[parent_index] +
                transition_nodes;
            candidate_dense_nodes = dense_nodes[parent_index] +
                transition_nodes;
            if (evoke_page_query_cost_plan_is_better(
                    candidate_sparse,
                    candidate_sparse_nodes,
                    sparse_cost[child_index],
                    sparse_nodes[child_index]))
            {
                sparse_cost[child_index] = candidate_sparse;
                sparse_nodes[child_index] = candidate_sparse_nodes;
                sparse_mask[child_index] = sparse_mask[parent_index] |
                    (uint16) (UINT16_C(1) << child_index);
            }
            if (evoke_page_query_cost_plan_is_better(
                    candidate_dense,
                    candidate_dense_nodes,
                    dense_cost[child_index],
                    dense_nodes[child_index]))
            {
                dense_cost[child_index] = candidate_dense;
                dense_nodes[child_index] = candidate_dense_nodes;
                dense_mask[child_index] = dense_mask[parent_index] |
                    (uint16) (UINT16_C(1) << child_index);
            }
        }
    }
    audit->optimal_sparse_plan.nodes_read = sparse_nodes[0];
    audit->optimal_sparse_plan.sparse_term_hits = sparse_cost[0];
    audit->optimal_sparse_plan.dense_term_probes = 0;
    audit->optimal_sparse_plan.level_mask = sparse_mask[0];
    audit->optimal_dense_plan.nodes_read = dense_nodes[0];
    audit->optimal_dense_plan.sparse_term_hits = 0;
    audit->optimal_dense_plan.dense_term_probes = dense_cost[0];
    audit->optimal_dense_plan.level_mask = dense_mask[0];
    return EVOKE_OK;
}

static void
evoke_page_query_cost_finalize_levels(
    uint64 document_slot_count,
    float kth_score,
    evoke_page_query_cost_level *levels,
    evoke_page_query_cost_work *work
)
{
    for (size_t level_index = 0;
         level_index < EVOKE_PAGE_QUERY_COST_LEVEL_COUNT;
         level_index++)
    {
        evoke_page_query_cost_level *level = &levels[level_index];
        evoke_page_query_cost_work *level_work = &work[level_index];
        uint64 block_size = UINT64_C(1) << level->block_shift;

        level->bound_bytes = level->touched_term_blocks *
            (sizeof(uint32) + sizeof(float));
        for (uint64 block_id = 0;
             block_id < level->block_count;
             block_id++)
        {
            uint64 first_document_slot;
            uint64 document_count;

            level->total_postings +=
                level_work->posting_counts[block_id];
            if (level_work->posting_counts[block_id] == 0 ||
                level_work->upper_bounds[block_id] < (double) kth_score)
            {
                continue;
            }
            first_document_slot = block_id * block_size;
            document_count = Min(
                block_size,
                document_slot_count - first_document_slot
            );
            level->competitive_blocks++;
            level->competitive_postings +=
                level_work->posting_counts[block_id];
            level->competitive_document_slots += document_count;
        }
    }
}

evoke_status
evoke_page_query_cost_audit_run(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_page_query_cost_audit *audit_out
)
{
    static const uint8 block_shifts[EVOKE_PAGE_QUERY_COST_LEVEL_COUNT] = {
        3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22
    };
    evoke_page_query_cost_work work[EVOKE_PAGE_QUERY_COST_LEVEL_COUNT];
    evoke_page_query_cleanup statistics_cleanup;
    evoke_segment_query_document_reader document_reader;
    evoke_page_query_term *terms = NULL;
    evoke_topk_result topk;
    evoke_page_query_stats topk_stats;
    uint32 *document_lengths = NULL;
    double *partial_scores = NULL;
    double *partial_absolute_sums = NULL;
    uint32 *posting_document_slots = NULL;
    evoke_posting_value *posting_values = NULL;
    size_t *mandatory_term_indices = NULL;
    evoke_page_query_cost_semantic_term *semantic_terms = NULL;
    double *semantic_suffix_caps = NULL;
    double *semantic_suffix_floors = NULL;
    double *semantic_suffix_absolute_caps = NULL;
    uint8 *essential_bitmaps[
        EVOKE_PAGE_QUERY_COST_MAX_ESSENTIAL_PROJECTIONS
    ];
    uint32 projection_points[
        EVOKE_PAGE_QUERY_COST_MAX_ESSENTIAL_PROJECTIONS
    ];
    uint64 document_slot_count;
    uint64 essential_bitmap_bytes = 0;
    uint64 semantic_super_refs = 0;
    uint64 semantic_refs = 0;
    uint64 semantic_postings = 0;
    double average_document_length;
    size_t term_count = 0;
    size_t mandatory_term_count = 0;
    size_t semantic_term_count = 0;
    uint32 projection_cursor = 0;
    bool needs_document_lengths = false;
    evoke_status status = EVOKE_OK;

    memset(work, 0, sizeof(work));
    memset(essential_bitmaps, 0, sizeof(essential_bitmaps));
    memset(&statistics_cleanup, 0, sizeof(statistics_cleanup));
    memset(&document_reader, 0, sizeof(document_reader));
    memset(&topk, 0, sizeof(topk));
    memset(&topk_stats, 0, sizeof(topk_stats));
    if (index_relation == NULL || context == NULL || audit_out == NULL ||
        query_ids == NULL || query_len == 0 || k == 0 ||
        k > context->manifest.visible_document_count ||
        context->root.active_l0.record_count != 0 ||
        context->root.pending_l0.record_count != 0 ||
        context->manifest.visible_document_count !=
            context->manifest.document_slot_count ||
        evoke_method_requires_nonoccurrence(
            context->query_contract.params.method))
    {
        return EVOKE_ERR_INVALID;
    }
    memset(audit_out, 0, sizeof(*audit_out));
    document_slot_count = context->manifest.document_slot_count;
    audit_out->document_slot_count = document_slot_count;
    for (size_t level_index = 0;
         level_index < EVOKE_PAGE_QUERY_COST_LEVEL_COUNT;
         level_index++)
    {
        audit_out->levels[level_index].block_shift =
            block_shifts[level_index];
        status = evoke_page_query_cost_work_init(
            document_slot_count,
            &audit_out->levels[level_index],
            &work[level_index]
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
        work[level_index].epoch = 1;
    }

    status = evoke_page_query_topk(
        index_relation,
        context,
        query_ids,
        query_weights,
        query_len,
        k,
        &topk,
        &topk_stats
    );
    if (status != EVOKE_OK || topk.len != k)
    {
        status = status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        goto done;
    }
    audit_out->kth_score = topk.scores[k - 1];

    terms = palloc0(sizeof(*terms) * query_len);
    for (size_t query_index = 0; query_index < query_len; query_index++)
    {
        evoke_page_query_term *term;
        float query_weight = query_weights == NULL
            ? 1.0f
            : query_weights[query_index];

        if (!isfinite(query_weight) || query_weight < 0.0f)
        {
            status = EVOKE_ERR_INVALID;
            goto done;
        }
        if (query_ids[query_index] >= context->manifest.vocab_size)
        {
            continue;
        }
        term = &terms[term_count++];
        term->query_index = (uint32) query_index;
        term->query_weight = query_weight;
        evoke_segment_pages_load_query_term_plan(
            index_relation,
            context,
            query_ids[query_index],
            &term->plan
        );
        term->immutable_live_document_frequency =
            term->plan.raw_document_frequency;
        term->live_document_frequency = term->plan.raw_document_frequency;
        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            evoke_posting_extent_kind kind =
                term->plan.runs[run_index].kind;

            audit_out->query_run_count++;
            if (kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
            {
                term->lexical_neutral_seen = true;
                needs_document_lengths = true;
            }
            else if (kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
            {
                term->lexical_impact_seen = true;
            }
            else if (kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
            {
                status = EVOKE_ERR_FORMAT;
                goto done;
            }
        }
    }
    audit_out->query_term_count = (uint32) term_count;
    average_document_length =
        (double) context->manifest.total_document_length /
        (double) context->manifest.visible_document_count;
    statistics_cleanup.terms = terms;
    status = evoke_page_query_prepare_term_statistics(
        context,
        context->manifest.visible_document_count,
        average_document_length,
        &statistics_cleanup,
        term_count
    );
    if (status != EVOKE_OK)
    {
        goto done;
    }
    mandatory_term_indices = palloc(sizeof(*mandatory_term_indices) *
        term_count);
    semantic_terms = palloc0(sizeof(*semantic_terms) * term_count);
    for (size_t term_index = 0; term_index < term_count; term_index++)
    {
        const evoke_page_query_term *term = &terms[term_index];
        evoke_page_query_cost_semantic_term candidate;
        bool projectable = true;

        memset(&candidate, 0, sizeof(candidate));
        candidate.term_index = term_index;
        for (uint32 run_index = 0;
             run_index < term->plan.run_count;
             run_index++)
        {
            const evoke_segment_query_run *run = &term->plan.runs[run_index];
            float run_cap;
            float first_contribution;
            float second_contribution;
            float run_floor;
            float run_absolute_cap;

            if (run->kind != EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT ||
                !run->semantic_bmp.available)
            {
                projectable = false;
                break;
            }
            run_cap = evoke_semantic_bmp_contribution_bound(
                term->query_weight,
                run->semantic_bmp.min_impact,
                run->semantic_bmp.max_impact
            );
            /* Match the scorer's float product before bounding its sum. */
            first_contribution = term->query_weight *
                run->semantic_bmp.min_impact;
            second_contribution = term->query_weight *
                run->semantic_bmp.max_impact;
            run_floor = Min(
                0.0f,
                Min(first_contribution, second_contribution)
            );
            run_absolute_cap = Max(
                fabsf(first_contribution),
                fabsf(second_contribution)
            );
            candidate.global_cap = nextafter(
                candidate.global_cap + (double) run_cap,
                INFINITY
            );
            candidate.global_floor = nextafter(
                candidate.global_floor + (double) run_floor,
                -INFINITY
            );
            candidate.global_absolute_cap = nextafter(
                candidate.global_absolute_cap +
                    (double) run_absolute_cap,
                INFINITY
            );
            if (!isfinite(candidate.global_cap) ||
                !isfinite(candidate.global_floor) ||
                !isfinite(candidate.global_absolute_cap))
            {
                status = EVOKE_ERR_RANGE;
                goto done;
            }
            if (candidate.super_ref_count >
                    UINT64_MAX - run->semantic_bmp.super_ref_count ||
                candidate.ref_count >
                    UINT64_MAX - run->semantic_bmp.ref_count ||
                candidate.posting_count >
                    UINT64_MAX - run->posting_count)
            {
                status = EVOKE_ERR_RANGE;
                goto done;
            }
            candidate.super_ref_count +=
                run->semantic_bmp.super_ref_count;
            candidate.ref_count += run->semantic_bmp.ref_count;
            candidate.posting_count += run->posting_count;
        }
        if (projectable)
        {
            semantic_terms[semantic_term_count++] = candidate;
        }
        else
        {
            mandatory_term_indices[mandatory_term_count++] = term_index;
        }
    }
    qsort(
        semantic_terms,
        semantic_term_count,
        sizeof(*semantic_terms),
        evoke_page_query_cost_semantic_term_compare
    );
    semantic_suffix_caps = palloc0(
        sizeof(*semantic_suffix_caps) * (semantic_term_count + 1)
    );
    semantic_suffix_floors = palloc0(
        sizeof(*semantic_suffix_floors) * (semantic_term_count + 1)
    );
    semantic_suffix_absolute_caps = palloc0(
        sizeof(*semantic_suffix_absolute_caps) *
            (semantic_term_count + 1)
    );
    for (size_t reverse_index = semantic_term_count;
         reverse_index > 0;
         reverse_index--)
    {
        semantic_suffix_caps[reverse_index - 1] = nextafter(
            semantic_suffix_caps[reverse_index] +
                semantic_terms[reverse_index - 1].global_cap,
            INFINITY
        );
        semantic_suffix_floors[reverse_index - 1] = nextafter(
            semantic_suffix_floors[reverse_index] +
                semantic_terms[reverse_index - 1].global_floor,
            -INFINITY
        );
        semantic_suffix_absolute_caps[reverse_index - 1] = nextafter(
            semantic_suffix_absolute_caps[reverse_index] +
                semantic_terms[reverse_index - 1].global_absolute_cap,
            INFINITY
        );
    }
    audit_out->mandatory_term_count = (uint32) mandatory_term_count;
    audit_out->projectable_semantic_term_count =
        (uint32) semantic_term_count;
    audit_out->essential_projection_count =
        evoke_page_query_cost_projection_points(
            (uint32) semantic_term_count,
            projection_points
        );
    essential_bitmap_bytes =
        (audit_out->levels[1].block_count + 7U) / 8U;
    if (essential_bitmap_bytes == 0 ||
        essential_bitmap_bytes > MaxAllocSize)
    {
        status = EVOKE_ERR_RANGE;
        goto done;
    }
    if (needs_document_lengths)
    {
        evoke_page_query_document_length_build length_build;

        if (document_slot_count > MaxAllocSize / sizeof(uint32))
        {
            status = EVOKE_ERR_RANGE;
            goto done;
        }
        document_lengths = palloc(
            sizeof(*document_lengths) * (Size) document_slot_count
        );
        if (!evoke_segment_pages_load_query_document_lengths(
                index_relation,
                context,
                document_lengths))
        {
            memset(&length_build, 0, sizeof(length_build));
            length_build.document_lengths = document_lengths;
            length_build.document_slot_count = document_slot_count;
            length_build.valid = true;
            evoke_segment_pages_visit_query_document_records(
                index_relation,
                context,
                &document_reader,
                evoke_page_query_collect_document_length,
                &length_build
            );
            if (!length_build.valid ||
                length_build.next_document_slot != document_slot_count)
            {
                status = EVOKE_ERR_FORMAT;
                goto done;
            }
        }
    }
    posting_document_slots = palloc(
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW *
            sizeof(*posting_document_slots)
    );
    posting_values = palloc(
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW * sizeof(*posting_values)
    );
    if (document_slot_count > MaxAllocSize / sizeof(*partial_scores) ||
        document_slot_count >
            MaxAllocSize / sizeof(*partial_absolute_sums))
    {
        status = EVOKE_ERR_RANGE;
        goto done;
    }
    partial_scores = palloc0(
        sizeof(*partial_scores) * (Size) document_slot_count
    );
    partial_absolute_sums = palloc0(
        sizeof(*partial_absolute_sums) * (Size) document_slot_count
    );

    for (size_t mandatory_index = 0;
         mandatory_index < mandatory_term_count;
         mandatory_index++)
    {
        CHECK_FOR_INTERRUPTS();
        status = evoke_page_query_cost_process_term(
            index_relation,
            context,
            &terms[mandatory_term_indices[mandatory_index]],
            document_lengths,
            average_document_length,
            document_slot_count,
            partial_scores,
            partial_absolute_sums,
            posting_document_slots,
            posting_values,
            audit_out,
            work
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
    }
    audit_out->mandatory_touched_b16_blocks =
        audit_out->levels[1].touched_term_blocks;
    if (projection_cursor < audit_out->essential_projection_count &&
        projection_points[projection_cursor] == 0)
    {
        essential_bitmaps[projection_cursor] =
            palloc0((Size) essential_bitmap_bytes);
        status = evoke_page_query_cost_capture_essential_projection(
            document_slot_count,
            k,
            audit_out->query_run_count,
            audit_out->kth_score,
            &audit_out->levels[1],
            &work[1],
            partial_scores,
            partial_absolute_sums,
            0,
            (uint32) semantic_term_count,
            semantic_suffix_caps[0],
            semantic_suffix_floors[0],
            semantic_suffix_absolute_caps[0],
            0,
            0,
            0,
            &audit_out->essential_projections[projection_cursor],
            essential_bitmaps[projection_cursor]
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
        projection_cursor++;
    }
    for (size_t semantic_index = 0;
         semantic_index < semantic_term_count;
         semantic_index++)
    {
        const evoke_page_query_cost_semantic_term *semantic_term =
            &semantic_terms[semantic_index];

        CHECK_FOR_INTERRUPTS();
        status = evoke_page_query_cost_process_term(
            index_relation,
            context,
            &terms[semantic_term->term_index],
            document_lengths,
            average_document_length,
            document_slot_count,
            partial_scores,
            partial_absolute_sums,
            posting_document_slots,
            posting_values,
            audit_out,
            work
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
        if (semantic_super_refs >
                UINT64_MAX - semantic_term->super_ref_count ||
            semantic_refs > UINT64_MAX - semantic_term->ref_count ||
            semantic_postings >
                UINT64_MAX - semantic_term->posting_count)
        {
            status = EVOKE_ERR_RANGE;
            goto done;
        }
        semantic_super_refs += semantic_term->super_ref_count;
        semantic_refs += semantic_term->ref_count;
        semantic_postings += semantic_term->posting_count;
        if (projection_cursor < audit_out->essential_projection_count &&
            projection_points[projection_cursor] == semantic_index + 1)
        {
            essential_bitmaps[projection_cursor] =
                palloc0((Size) essential_bitmap_bytes);
            status = evoke_page_query_cost_capture_essential_projection(
                document_slot_count,
                k,
                audit_out->query_run_count,
                audit_out->kth_score,
                &audit_out->levels[1],
                &work[1],
                partial_scores,
                partial_absolute_sums,
                (uint32) semantic_index + 1,
                (uint32) semantic_term_count,
                semantic_suffix_caps[semantic_index + 1],
                semantic_suffix_floors[semantic_index + 1],
                semantic_suffix_absolute_caps[semantic_index + 1],
                semantic_super_refs,
                semantic_refs,
                semantic_postings,
                &audit_out->essential_projections[projection_cursor],
                essential_bitmaps[projection_cursor]
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }
            projection_cursor++;
        }
    }
    if (projection_cursor != audit_out->essential_projection_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto done;
    }
    evoke_page_query_cost_finalize_levels(
        document_slot_count,
        audit_out->kth_score,
        audit_out->levels,
        work
    );
    status = evoke_page_query_cost_finalize_hierarchy(
        audit_out->kth_score,
        audit_out->query_term_count,
        audit_out,
        work
    );
    if (status == EVOKE_OK)
    {
        status = evoke_page_query_cost_finalize_essential_projections(
            audit_out->kth_score,
            &audit_out->levels[1],
            &work[1],
            audit_out,
            essential_bitmaps
        );
    }
    if (status == EVOKE_OK)
    {
        status = evoke_page_query_cost_finalize_optimal_plans(
            audit_out->kth_score,
            audit_out->query_term_count,
            audit_out,
            work
        );
    }

done:
    for (size_t projection_index = 0;
         projection_index < lengthof(essential_bitmaps);
         projection_index++)
    {
        if (essential_bitmaps[projection_index] != NULL)
        {
            pfree(essential_bitmaps[projection_index]);
        }
    }
    if (semantic_suffix_caps != NULL)
    {
        pfree(semantic_suffix_caps);
    }
    if (semantic_suffix_floors != NULL)
    {
        pfree(semantic_suffix_floors);
    }
    if (semantic_suffix_absolute_caps != NULL)
    {
        pfree(semantic_suffix_absolute_caps);
    }
    if (semantic_terms != NULL)
    {
        pfree(semantic_terms);
    }
    if (mandatory_term_indices != NULL)
    {
        pfree(mandatory_term_indices);
    }
    if (posting_values != NULL)
    {
        pfree(posting_values);
    }
    if (posting_document_slots != NULL)
    {
        pfree(posting_document_slots);
    }
    if (document_lengths != NULL)
    {
        pfree(document_lengths);
    }
    if (partial_absolute_sums != NULL)
    {
        pfree(partial_absolute_sums);
    }
    if (partial_scores != NULL)
    {
        pfree(partial_scores);
    }
    if (terms != NULL)
    {
        pfree(terms);
    }
    evoke_topk_result_free(&topk);
    for (size_t level_index = 0;
         level_index < EVOKE_PAGE_QUERY_COST_LEVEL_COUNT;
         level_index++)
    {
        evoke_page_query_cost_work_free(&work[level_index]);
    }
    return status;
}

typedef struct evoke_page_query_forward_bound_work
{
    double *term_maxima;
    uint32 *allowed_documents;
    uint64 *allowed_postings;
    uint64 *allowed_row_bytes;
    float *published_term_maxima;
    uint32 *published_touched_terms;
    uint32 *published_previous_blocks;
    uint32 published_touched_count;
    uint32 published_current_block;
    uint32 vocab_size;
} evoke_page_query_forward_bound_work;

static evoke_status
evoke_page_query_forward_bound_estimate_bytes(
    uint64 document_count,
    size_t query_len,
    const uint8 *block_shifts,
    size_t level_count,
    uint32 vocab_size,
    uint64 *bytes_out
)
{
    uint64 total = 0;

    if (document_count == 0 || query_len == 0 || block_shifts == NULL ||
        vocab_size == 0 || bytes_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    for (size_t level_index = 0;
         level_index < level_count;
         level_index++)
    {
        uint64 block_size = UINT64_C(1) << block_shifts[level_index];
        uint64 block_count =
            (document_count + block_size - 1U) / block_size;
        uint64 per_block_bytes;

        if (block_count == 0 || query_len > UINT64_MAX / block_count ||
            (uint64) query_len * block_count >
                UINT64_MAX / sizeof(double))
        {
            return EVOKE_ERR_RANGE;
        }
        per_block_bytes = (uint64) query_len * sizeof(double);
        if (per_block_bytes >
            UINT64_MAX - sizeof(uint32) - 2U * sizeof(uint64))
        {
            return EVOKE_ERR_RANGE;
        }
        per_block_bytes += sizeof(uint32) + 2U * sizeof(uint64);
        if (block_count > UINT64_MAX / per_block_bytes ||
            total > UINT64_MAX - block_count * per_block_bytes)
        {
            return EVOKE_ERR_RANGE;
        }
        total += block_count * per_block_bytes;
        if ((uint64) vocab_size >
                UINT64_MAX /
                    (sizeof(float) + 2U * sizeof(uint32)) ||
            total > UINT64_MAX -
                (uint64) vocab_size *
                    (sizeof(float) + 2U * sizeof(uint32)))
        {
            return EVOKE_ERR_RANGE;
        }
        total += (uint64) vocab_size *
            (sizeof(float) + 2U * sizeof(uint32));
    }
    *bytes_out = total;
    return EVOKE_OK;
}

static size_t
evoke_page_query_forward_varint_size(uint32 value)
{
    size_t size = 1;

    while (value >= UINT32_C(0x80))
    {
        value >>= 7;
        size++;
    }
    return size;
}

static void
evoke_page_query_forward_bound_work_free(
    evoke_page_query_forward_bound_work *work
)
{
    if (work == NULL)
    {
        return;
    }
    free(work->term_maxima);
    free(work->allowed_documents);
    free(work->allowed_postings);
    free(work->allowed_row_bytes);
    free(work->published_term_maxima);
    free(work->published_touched_terms);
    free(work->published_previous_blocks);
    memset(work, 0, sizeof(*work));
}

static bool
evoke_page_query_forward_bound_document_allowed(
    const evoke_page_query_filter *filter,
    uint32 document
)
{
    if (filter == NULL)
    {
        return true;
    }
    return filter->allowed_document_bitmap != NULL &&
        document < filter->document_slot_count &&
        (filter->allowed_document_bitmap[document >> 3] &
         (uint8) (UINT8_C(1) << (document & 7U))) != 0;
}

static evoke_status
evoke_page_query_forward_bound_work_init(
    uint64 document_count,
    size_t query_len,
    uint32 vocab_size,
    uint8 block_shift,
    evoke_page_query_forward_bound_level *level,
    evoke_page_query_forward_bound_work *work
)
{
    uint64 block_size = UINT64_C(1) << block_shift;
    uint64 block_count = (document_count + block_size - 1U) / block_size;
    size_t term_block_count;

    memset(level, 0, sizeof(*level));
    memset(work, 0, sizeof(*work));
    if (block_count == 0 || block_count > SIZE_MAX || query_len == 0 ||
        vocab_size == 0 ||
        query_len > SIZE_MAX / (size_t) block_count)
    {
        return EVOKE_ERR_RANGE;
    }
    term_block_count = query_len * (size_t) block_count;
    if (term_block_count > SIZE_MAX / sizeof(*work->term_maxima) ||
        (size_t) block_count >
            SIZE_MAX / sizeof(*work->allowed_documents) ||
        (size_t) block_count >
            SIZE_MAX / sizeof(*work->allowed_postings) ||
        (size_t) block_count >
            SIZE_MAX / sizeof(*work->allowed_row_bytes))
    {
        return EVOKE_ERR_RANGE;
    }
    work->term_maxima = calloc(
        term_block_count,
        sizeof(*work->term_maxima)
    );
    work->allowed_documents = calloc(
        (size_t) block_count,
        sizeof(*work->allowed_documents)
    );
    work->allowed_postings = calloc(
        (size_t) block_count,
        sizeof(*work->allowed_postings)
    );
    work->allowed_row_bytes = calloc(
        (size_t) block_count,
        sizeof(*work->allowed_row_bytes)
    );
    work->published_term_maxima = calloc(
        vocab_size,
        sizeof(*work->published_term_maxima)
    );
    work->published_touched_terms = malloc(
        (size_t) vocab_size * sizeof(*work->published_touched_terms)
    );
    work->published_previous_blocks = calloc(
        vocab_size,
        sizeof(*work->published_previous_blocks)
    );
    if (work->term_maxima == NULL ||
        work->allowed_documents == NULL ||
        work->allowed_postings == NULL ||
        work->allowed_row_bytes == NULL ||
        work->published_term_maxima == NULL ||
        work->published_touched_terms == NULL ||
        work->published_previous_blocks == NULL)
    {
        evoke_page_query_forward_bound_work_free(work);
        return EVOKE_ERR_NOMEM;
    }
    level->block_shift = block_shift;
    level->block_count = block_count;
    level->topk_contained = true;
    work->published_current_block = UINT32_MAX;
    work->vocab_size = vocab_size;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_forward_bound_flush_published_block(
    evoke_page_query_forward_bound_level *level,
    evoke_page_query_forward_bound_work *work
)
{
    if (work->published_current_block == UINT32_MAX)
    {
        return EVOKE_OK;
    }
    for (uint32 index = 0;
         index < work->published_touched_count;
         index++)
    {
        uint32 term_id = work->published_touched_terms[index];
        float maximum = work->published_term_maxima[term_id];
        uint32 delta = work->published_current_block -
            work->published_previous_blocks[term_id];
        uint64 entry_bytes =
            evoke_page_query_forward_varint_size(delta) + sizeof(float);

        if (!(maximum > 0.0f) || !isfinite(maximum) ||
            level->published_bound_entries == UINT64_MAX ||
            level->published_bound_bytes > UINT64_MAX - entry_bytes)
        {
            return EVOKE_ERR_RANGE;
        }
        level->published_bound_entries++;
        level->published_bound_bytes += entry_bytes;
        work->published_previous_blocks[term_id] =
            work->published_current_block;
        work->published_term_maxima[term_id] = 0.0f;
    }
    work->published_touched_count = 0;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_forward_bound_start_published_block(
    uint32 document,
    evoke_page_query_forward_bound_level *level,
    evoke_page_query_forward_bound_work *work
)
{
    uint32 block = document >> level->block_shift;
    evoke_status status;

    if (work->published_current_block == block)
    {
        return EVOKE_OK;
    }
    if (work->published_current_block != UINT32_MAX &&
        block <= work->published_current_block)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_page_query_forward_bound_flush_published_block(
        level,
        work
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    work->published_current_block = block;
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_forward_bound_record_published_term(
    uint32 term_id,
    double contribution,
    evoke_page_query_forward_bound_work *work
)
{
    float stored_bound;

    if (term_id >= work->vocab_size || !isfinite(contribution))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (contribution <= 0.0)
    {
        return EVOKE_OK;
    }
    if (contribution > FLT_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    stored_bound = nextafterf((float) contribution, INFINITY);
    if (!isfinite(stored_bound))
    {
        return EVOKE_ERR_RANGE;
    }
    if (work->published_term_maxima[term_id] == 0.0f)
    {
        if (work->published_touched_count >= work->vocab_size)
        {
            return EVOKE_ERR_RANGE;
        }
        work->published_touched_terms[
            work->published_touched_count++
        ] = term_id;
    }
    if (stored_bound > work->published_term_maxima[term_id])
    {
        work->published_term_maxima[term_id] = stored_bound;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_forward_bound_record_row(
    const evoke_semantic_forward_chunk *chunk,
    uint32 local_document,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_len,
    bool allowed,
    evoke_page_query_forward_bound_level *levels,
    evoke_page_query_forward_bound_work *work,
    evoke_topk_accumulator *topk
)
{
    uint32 document = chunk->first_document + local_document;
    uint32 row_start = chunk->row_offsets[local_document];
    uint32 row_end = chunk->row_offsets[local_document + 1U];
    uint32 posting = row_start;
    size_t query = 0;
    size_t row_bytes = row_start == row_end ? 0 : sizeof(float);
    uint32 previous_term = 0;
    double score = 0.0;
    evoke_status status;

    for (size_t level_index = 0;
         level_index < EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT;
         level_index++)
    {
        status = evoke_page_query_forward_bound_start_published_block(
            document,
            &levels[level_index],
            &work[level_index]
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }

    for (uint32 row_posting = row_start;
         row_posting < row_end;
         row_posting++)
    {
        uint32 term_id = chunk->term_ids[row_posting];
        uint32 delta = term_id - previous_term;

        if (row_bytes > SIZE_MAX -
                evoke_page_query_forward_varint_size(delta) - 1U)
        {
            return EVOKE_ERR_RANGE;
        }
        row_bytes += evoke_page_query_forward_varint_size(delta) + 1U;
        previous_term = term_id;
    }
    for (uint32 row_posting = row_start;
         row_posting < row_end;)
    {
        uint32 term_id = chunk->term_ids[row_posting];
        double contribution = 0.0;

        do
        {
            contribution += chunk->contributions[row_posting];
            row_posting++;
        }
        while (row_posting < row_end &&
               chunk->term_ids[row_posting] == term_id);
        for (size_t level_index = 0;
             level_index < EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT;
             level_index++)
        {
            status = evoke_page_query_forward_bound_record_published_term(
                term_id,
                contribution,
                &work[level_index]
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
        }
    }
    while (posting < row_end && query < query_len)
    {
        uint32 term_id = chunk->term_ids[posting];
        double contribution = 0.0;
        double weighted_contribution;

        if (term_id < query_ids[query])
        {
            posting++;
            continue;
        }
        if (term_id > query_ids[query])
        {
            query++;
            continue;
        }
        do
        {
            contribution += chunk->contributions[posting];
            posting++;
        }
        while (posting < row_end &&
               chunk->term_ids[posting] == term_id);
        weighted_contribution = contribution *
            (double) query_weights[query];
        if (!isfinite(contribution) || contribution > FLT_MAX ||
            !isfinite(weighted_contribution))
        {
            return EVOKE_ERR_RANGE;
        }
        score += weighted_contribution;
        for (size_t level_index = 0;
             level_index < EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT;
             level_index++)
        {
            uint64 block = (uint64) document >>
                levels[level_index].block_shift;
            size_t offset = query *
                (size_t) levels[level_index].block_count +
                (size_t) block;

            if (contribution > work[level_index].term_maxima[offset])
            {
                float stored_bound = nextafterf(
                    (float) contribution,
                    INFINITY
                );

                if (!isfinite(stored_bound))
                {
                    return EVOKE_ERR_RANGE;
                }
                work[level_index].term_maxima[offset] = stored_bound;
            }
        }
        query++;
    }
    if (!isfinite(score) || score > FLT_MAX || score < -FLT_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    if (allowed)
    {
        uint64 posting_count = row_end - row_start;

        for (size_t level_index = 0;
             level_index < EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT;
             level_index++)
        {
            uint64 block = (uint64) document >>
                levels[level_index].block_shift;

            if (work[level_index].allowed_documents[block] == UINT32_MAX ||
                work[level_index].allowed_postings[block] >
                    UINT64_MAX - posting_count ||
                work[level_index].allowed_row_bytes[block] >
                    UINT64_MAX - row_bytes)
            {
                return EVOKE_ERR_RANGE;
            }
            work[level_index].allowed_documents[block]++;
            work[level_index].allowed_postings[block] += posting_count;
            work[level_index].allowed_row_bytes[block] += row_bytes;
        }
        if (score > 0.0)
        {
            return evoke_topk_accumulator_offer(
                topk,
                (float) score,
                document,
                document
            );
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_forward_bound_finalize_level(
    const evoke_topk_result *topk,
    const float *query_weights,
    size_t query_len,
    float kth_score,
    evoke_page_query_forward_bound_level *level,
    const evoke_page_query_forward_bound_work *work
)
{
    const uint64 bounds_per_window = UINT64_C(4096) / sizeof(float);
    uint64 membership_bytes = (level->block_count + 7U) / 8U;
    uint64 projected_bound_bytes = 0;

    if (level->block_count > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    for (size_t query = 0; query < query_len; query++)
    {
        uint32 previous_block = 0;
        uint64 query_bound_entries = 0;
        uint64 query_bound_bytes = 0;

        for (uint64 block = 0; block < level->block_count; block++)
        {
            double maximum = work->term_maxima[
                query * (size_t) level->block_count + (size_t) block
            ];
            uint32 block_delta;
            uint64 entry_bytes;

            if (maximum <= 0.0 || query_weights[query] <= 0.0f)
            {
                continue;
            }
            block_delta = (uint32) block - previous_block;
            entry_bytes =
                evoke_page_query_forward_varint_size(block_delta) +
                sizeof(float);
            if (level->bound_entries == UINT64_MAX ||
                projected_bound_bytes > UINT64_MAX - entry_bytes)
            {
                return EVOKE_ERR_RANGE;
            }
            level->bound_entries++;
            query_bound_entries++;
            projected_bound_bytes += entry_bytes;
            query_bound_bytes += entry_bytes;
            previous_block = (uint32) block;
        }
        if (query_bound_entries == 0)
        {
            continue;
        }
        if (query_bound_entries >
                (UINT64_MAX - membership_bytes) / sizeof(float) ||
            level->addressable_storage_bytes >
                UINT64_MAX - membership_bytes -
                    query_bound_entries * sizeof(float))
        {
            return EVOKE_ERR_RANGE;
        }
        if (membership_bytes + query_bound_entries * sizeof(float) <
            query_bound_bytes)
        {
            uint64 present_rank = 0;
            uint64 previous_window = UINT64_MAX;

            level->addressable_dense_term_count++;
            level->addressable_storage_bytes += membership_bytes +
                query_bound_entries * sizeof(float);
            if (level->addressable_query_metadata_bytes >
                UINT64_MAX - membership_bytes)
            {
                return EVOKE_ERR_RANGE;
            }
            level->addressable_query_metadata_bytes += membership_bytes;
            for (uint64 block = 0; block < level->block_count; block++)
            {
                double maximum = work->term_maxima[
                    query * (size_t) level->block_count + (size_t) block
                ];

                if (maximum <= 0.0 || query_weights[query] <= 0.0f)
                {
                    continue;
                }
                if (work->allowed_documents[block] != 0)
                {
                    uint64 window = present_rank / bounds_per_window;

                    if (level->addressable_selected_bound_entries ==
                        UINT64_MAX)
                    {
                        return EVOKE_ERR_RANGE;
                    }
                    level->addressable_selected_bound_entries++;
                    if (window != previous_window)
                    {
                        uint64 first_entry = window * bounds_per_window;
                        uint64 remaining_entries = query_bound_entries -
                            first_entry;
                        uint64 window_entries = remaining_entries <
                            bounds_per_window
                            ? remaining_entries
                            : bounds_per_window;
                        uint64 window_bytes = window_entries * sizeof(float);

                        if (level->addressable_selected_bound_windows ==
                                UINT64_MAX ||
                            level->addressable_selected_window_bytes >
                                UINT64_MAX - window_bytes)
                        {
                            return EVOKE_ERR_RANGE;
                        }
                        level->addressable_selected_bound_windows++;
                        level->addressable_selected_window_bytes +=
                            window_bytes;
                        previous_window = window;
                    }
                }
                present_rank++;
            }
        }
        else
        {
            level->addressable_sparse_term_count++;
            if (level->addressable_storage_bytes >
                    UINT64_MAX - query_bound_bytes ||
                level->addressable_query_metadata_bytes >
                    UINT64_MAX - query_bound_bytes)
            {
                return EVOKE_ERR_RANGE;
            }
            level->addressable_storage_bytes += query_bound_bytes;
            level->addressable_query_metadata_bytes += query_bound_bytes;
        }
    }
    if (level->addressable_selected_bound_entries >
            UINT64_MAX / sizeof(float) ||
        level->addressable_query_metadata_bytes >
            UINT64_MAX - level->addressable_selected_window_bytes)
    {
        return EVOKE_ERR_RANGE;
    }
    level->addressable_selected_bound_bytes =
        level->addressable_selected_bound_entries * sizeof(float);
    level->addressable_query_bytes =
        level->addressable_query_metadata_bytes +
        level->addressable_selected_window_bytes;
    for (uint64 block = 0; block < level->block_count; block++)
    {
        double upper_bound = 0.0;

        for (size_t query = 0; query < query_len; query++)
        {
            double maximum = work->term_maxima[
                query * (size_t) level->block_count + (size_t) block
            ];

            if (maximum > 0.0 && query_weights[query] > 0.0f)
            {
                upper_bound = nextafter(
                    upper_bound + maximum *
                        (double) query_weights[query],
                    INFINITY
                );
            }
        }
        if (work->allowed_documents[block] == 0)
        {
            continue;
        }
        level->allowed_blocks++;
        level->allowed_documents += work->allowed_documents[block];
        level->allowed_postings += work->allowed_postings[block];
        level->allowed_row_bytes += work->allowed_row_bytes[block];
        if (upper_bound >= (double) kth_score)
        {
            level->competitive_blocks++;
            level->competitive_documents += work->allowed_documents[block];
            level->competitive_postings += work->allowed_postings[block];
            level->competitive_row_bytes += work->allowed_row_bytes[block];
        }
    }
    level->projected_bound_bytes = projected_bound_bytes;
    for (size_t rank = 0; rank < topk->len; rank++)
    {
        uint64 block = (uint64) topk->doc_ids[rank] >> level->block_shift;
        double upper_bound = 0.0;

        if (block >= level->block_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (size_t query = 0; query < query_len; query++)
        {
            double maximum = work->term_maxima[
                query * (size_t) level->block_count + (size_t) block
            ];

            if (maximum > 0.0 && query_weights[query] > 0.0f)
            {
                upper_bound = nextafter(
                    upper_bound + maximum *
                        (double) query_weights[query],
                    INFINITY
                );
            }
        }
        if (upper_bound < (double) kth_score)
        {
            level->topk_contained = false;
            break;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_page_query_term_suffix_ceiling_from_directory(
    const evoke_semantic_accelerator_directory *directory,
    const evoke_segment_query_context *context,
    const uint32_t *query_ids,
    size_t query_len,
    evoke_page_query_term_suffix_ceiling *ceiling_out
)
{
    if (directory == NULL || context == NULL || query_ids == NULL ||
        query_len == 0 || ceiling_out == NULL ||
        directory->builder_policy_id !=
            EVOKE_SEMANTIC_ACCELERATOR_CURRENT_POLICY ||
        directory->retained_document_cap !=
            EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP ||
        directory->forward_term_work == NULL ||
        directory->forward_term_bytes == NULL ||
        directory->document_count !=
            context->manifest.document_slot_count)
    {
        return EVOKE_ERR_FORMAT;
    }

    memset(ceiling_out, 0, sizeof(*ceiling_out));
    for (size_t query = 0; query < query_len; query++)
    {
        if (query_ids[query] >= directory->vocab_size ||
            (query > 0 && query_ids[query - 1U] >= query_ids[query]))
        {
            return EVOKE_ERR_INVALID;
        }
    }

    ceiling_out->document_count = directory->document_count;
    ceiling_out->forward_chunk_count = directory->forward_chunk_count;
    ceiling_out->vocab_size = directory->vocab_size;
    ceiling_out->query_term_count = (uint32) query_len;
    ceiling_out->maximum_query_term = query_ids[query_len - 1U];
    for (uint32 term_id = 0; term_id < directory->vocab_size; term_id++)
    {
        uint64 term_work = directory->forward_term_work[term_id];
        uint64 term_bytes = directory->forward_term_bytes[term_id];

        if ((term_id & UINT32_C(4095)) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
        if (ceiling_out->forward_term_work_total >
                UINT64_MAX - term_work ||
            ceiling_out->forward_term_bytes_total >
                UINT64_MAX - term_bytes)
        {
            return EVOKE_ERR_RANGE;
        }
        ceiling_out->forward_term_work_total += term_work;
        ceiling_out->forward_term_bytes_total += term_bytes;
        if (term_id <= ceiling_out->maximum_query_term)
        {
            continue;
        }
        if (ceiling_out->forward_term_work_suffix >
                UINT64_MAX - term_work ||
            ceiling_out->forward_term_bytes_suffix >
                UINT64_MAX - term_bytes)
        {
            return EVOKE_ERR_RANGE;
        }
        ceiling_out->forward_term_work_suffix += term_work;
        ceiling_out->forward_term_bytes_suffix += term_bytes;
    }
    return EVOKE_OK;
}

evoke_status
evoke_page_query_term_suffix_ceiling_run(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const uint32_t *query_ids,
    size_t query_len,
    evoke_page_query_term_suffix_ceiling *ceiling_out
)
{
    evoke_semantic_accelerator_directory directory;
    evoke_status status;

    if (index_relation == NULL || context == NULL || query_ids == NULL ||
        query_len == 0 || ceiling_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_semantic_accelerator_directory_init(&directory);
    PG_TRY();
    {
        evoke_segment_pages_load_semantic_accelerator_forward_directory(
            index_relation,
            context,
            &directory
        );
        status = evoke_page_query_term_suffix_ceiling_from_directory(
            &directory,
            context,
            query_ids,
            query_len,
            ceiling_out
        );
    }
    PG_FINALLY();
    {
        evoke_semantic_accelerator_directory_free(&directory);
    }
    PG_END_TRY();
    return status;
}

evoke_status
evoke_page_query_forward_bound_audit_run(
    Relation index_relation,
    const evoke_segment_query_context *context,
    const evoke_page_query_filter *filter,
    const uint32_t *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_page_query_forward_bound_audit *audit_out
)
{
    static const uint8 block_shifts[
        EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT
    ] = {3, 4, 6};
    evoke_page_query_forward_bound_work work[
        EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT
    ];
    evoke_semantic_accelerator_directory directory;
    evoke_semantic_forward_chunk chunk;
    evoke_page_query_term_suffix_ceiling ceiling;
    evoke_topk_accumulator accumulator;
    evoke_topk_result topk;
    uint64 expected_first_document = 0;
    uint64 estimated_work_bytes = 0;
    evoke_status status = EVOKE_OK;

    memset(work, 0, sizeof(work));
    memset(&accumulator, 0, sizeof(accumulator));
    memset(&topk, 0, sizeof(topk));
    memset(&ceiling, 0, sizeof(ceiling));
    evoke_semantic_accelerator_directory_init(&directory);
    evoke_semantic_forward_chunk_init(&chunk);
    if (index_relation == NULL || context == NULL || audit_out == NULL ||
        query_ids == NULL || query_weights == NULL || query_len == 0 ||
        k == 0 || context->root.active_l0.record_count != 0 ||
        context->root.pending_l0.record_count != 0 ||
        context->manifest.visible_document_count !=
            context->manifest.document_slot_count ||
        (filter != NULL &&
         (!filter->live_membership_verified ||
          filter->allowed_document_bitmap == NULL ||
          filter->document_slot_count <
              context->manifest.document_slot_count ||
          filter->allowed_document_count < k)))
    {
        return EVOKE_ERR_INVALID;
    }
    memset(audit_out, 0, sizeof(*audit_out));
    for (size_t query = 0; query < query_len; query++)
    {
        if (!isfinite(query_weights[query]) || query_weights[query] < 0.0f ||
            query_ids[query] >= context->manifest.vocab_size ||
            (query > 0 && query_ids[query - 1U] >= query_ids[query]))
        {
            return EVOKE_ERR_INVALID;
        }
    }
    evoke_segment_pages_load_semantic_accelerator_forward_directory(
        index_relation,
        context,
        &directory
    );
    if (directory.builder_policy_id !=
            EVOKE_SEMANTIC_ACCELERATOR_CURRENT_POLICY ||
        directory.retained_document_cap !=
            EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP ||
        directory.forward_term_work == NULL ||
        directory.forward_term_bytes == NULL ||
        !evoke_semantic_accelerator_directory_has_tid_lookup(&directory))
    {
        audit_out->failure_stage = 11;
        status = EVOKE_ERR_FORMAT;
        goto done;
    }
    if (directory.forward_chunk_count == 0 ||
        directory.forward_document_shift == 0 ||
        directory.forward_document_shift >= 32)
    {
        audit_out->failure_stage = 12;
        status = EVOKE_ERR_FORMAT;
        goto done;
    }
    if (directory.document_count !=
        context->manifest.document_slot_count)
    {
        audit_out->failure_stage = 13;
        status = EVOKE_ERR_FORMAT;
        goto done;
    }
    status = evoke_page_query_term_suffix_ceiling_from_directory(
        &directory,
        context,
        query_ids,
        query_len,
        &ceiling
    );
    if (status != EVOKE_OK)
    {
        audit_out->failure_stage = 14;
        goto done;
    }
    audit_out->document_count = ceiling.document_count;
    audit_out->allowed_document_count = filter == NULL
        ? directory.document_count
        : filter->allowed_document_count;
    audit_out->forward_chunk_count = ceiling.forward_chunk_count;
    audit_out->query_term_count = ceiling.query_term_count;
    audit_out->maximum_query_term = ceiling.maximum_query_term;
    audit_out->forward_term_work_total =
        ceiling.forward_term_work_total;
    audit_out->forward_term_work_suffix =
        ceiling.forward_term_work_suffix;
    audit_out->forward_term_bytes_total =
        ceiling.forward_term_bytes_total;
    audit_out->forward_term_bytes_suffix =
        ceiling.forward_term_bytes_suffix;
    status = evoke_page_query_forward_bound_estimate_bytes(
        directory.document_count,
        query_len,
        block_shifts,
        EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT,
        directory.vocab_size,
        &estimated_work_bytes
    );
    if (status != EVOKE_OK)
    {
        goto done;
    }
    if (estimated_work_bytes >
        EVOKE_PAGE_QUERY_FORWARD_BOUND_AUDIT_MAX_BYTES)
    {
        status = EVOKE_ERR_RANGE;
        goto done;
    }
    for (size_t level_index = 0;
         level_index < EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT;
         level_index++)
    {
        status = evoke_page_query_forward_bound_work_init(
            directory.document_count,
            query_len,
            directory.vocab_size,
            block_shifts[level_index],
            &audit_out->levels[level_index],
            &work[level_index]
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
    }
    status = evoke_topk_accumulator_init(&accumulator, k);
    if (status != EVOKE_OK)
    {
        goto done;
    }
    for (uint32 chunk_index = 0;
         chunk_index < directory.forward_chunk_count;
         chunk_index++)
    {
        const evoke_semantic_accelerator_forward_entry *entry =
            &directory.forward_chunks[chunk_index];

        CHECK_FOR_INTERRUPTS();
        if (entry->first_document != expected_first_document ||
            entry->document_count == 0 ||
            expected_first_document >
                directory.document_count - entry->document_count ||
            audit_out->forward_object_bytes >
                UINT64_MAX - entry->forward_object.object_bytes)
        {
            audit_out->failure_stage = 2;
            status = EVOKE_ERR_FORMAT;
            goto done;
        }
        audit_out->forward_object_bytes +=
            entry->forward_object.object_bytes;
        evoke_segment_pages_load_semantic_accelerator_forward(
            index_relation,
            &context->root,
            &context->manifest,
            &directory,
            entry->first_document,
            &chunk
        );
        for (uint32 local_document = 0;
             local_document < chunk.document_count;
             local_document++)
        {
            uint32 document = chunk.first_document + local_document;

            status = evoke_page_query_forward_bound_record_row(
                &chunk,
                local_document,
                query_ids,
                query_weights,
                query_len,
                evoke_page_query_forward_bound_document_allowed(
                    filter,
                    document
                ),
                audit_out->levels,
                work,
                &accumulator
            );
            if (status != EVOKE_OK)
            {
                goto done;
            }
        }
        expected_first_document += entry->document_count;
        evoke_semantic_forward_chunk_free(&chunk);
    }
    if (expected_first_document != directory.document_count)
    {
        audit_out->failure_stage = 3;
        status = EVOKE_ERR_FORMAT;
        goto done;
    }
    status = evoke_topk_accumulator_finish(&accumulator, true, &topk);
    if (status != EVOKE_OK)
    {
        goto done;
    }
    audit_out->topk_count = (uint32) topk.len;
    audit_out->kth_score = topk.len == k ? topk.scores[k - 1U] : 0.0f;
    for (size_t level_index = 0;
         level_index < EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT;
         level_index++)
    {
        status = evoke_page_query_forward_bound_flush_published_block(
            &audit_out->levels[level_index],
            &work[level_index]
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
        status = evoke_page_query_forward_bound_finalize_level(
            &topk,
            query_weights,
            query_len,
            audit_out->kth_score,
            &audit_out->levels[level_index],
            &work[level_index]
        );
        if (status != EVOKE_OK)
        {
            goto done;
        }
    }

done:
    evoke_topk_result_free(&topk);
    evoke_topk_accumulator_free(&accumulator);
    evoke_semantic_forward_chunk_free(&chunk);
    evoke_semantic_accelerator_directory_free(&directory);
    for (size_t level_index = 0;
         level_index < EVOKE_PAGE_QUERY_FORWARD_BOUND_LEVEL_COUNT;
         level_index++)
    {
        evoke_page_query_forward_bound_work_free(&work[level_index]);
    }
    return status;
}
