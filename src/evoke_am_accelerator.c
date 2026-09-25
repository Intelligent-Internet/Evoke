#include "postgres.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "access/table.h"
#include "catalog/pg_collation_d.h"
#include "catalog/pg_type_d.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "parser/parse_oper.h"
#include "storage/buffile.h"
#include "storage/lmgr.h"
#include "utils/backend_status.h"
#include "utils/array.h"
#include "utils/logtape.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/snapmgr.h"
#include "utils/tuplesort.h"

#include "evoke_am_accelerator.h"
#include "evoke_am_reclamation.h"
#include "evoke_document_tid_lookup.h"
#include "evoke_scope_pg.h"
#include "evoke_segment_pages.h"
#include "evoke_semantic_accelerator_builder.h"
#include "evoke_semantic_forward_bound.h"

#define EVOKE_AM_ACCELERATOR_FORWARD_TARGET_BYTES \
    (UINT64_C(8) * 1024 * 1024)
#define EVOKE_AM_ACCELERATOR_FORWARD_HEADER_BYTES 64U
#define EVOKE_AM_ACCELERATOR_CENTROID_FRACTION 0.02f
#define EVOKE_AM_ACCELERATOR_MINIMUM_CLUSTER_SIZE 2U
#define EVOKE_AM_ACCELERATOR_DOCUMENT_CUT 15U
#define EVOKE_AM_ACCELERATOR_SUMMARY_ENERGY 0.10f
#define EVOKE_AM_ACCELERATOR_RANDOM_SEED 1142U

double evoke_test_semantic_accelerator_posting_mass = 0.30;
int evoke_test_semantic_accelerator_forward_document_shift = 0;

typedef struct evoke_am_accelerator_forward_entry
{
    uint64 sort_key;
    double contribution;
} evoke_am_accelerator_forward_entry;

typedef struct evoke_am_accelerator_retained_document
{
    uint32 document_id;
    float impact;
} evoke_am_accelerator_retained_document;

StaticAssertDecl(
    sizeof(evoke_am_accelerator_forward_entry) == 16,
    "semantic accelerator transpose tuple must remain compact"
);

#define EVOKE_AM_ACCELERATOR_FORWARD_ORDER_MAX UINT32_C(0x3fffffff)

static uint64
evoke_am_accelerator_forward_key(
    uint32 term_id,
    uint32 contribution_order,
    uint8 operation
)
{
    return ((uint64) term_id << 32) |
        ((uint64) contribution_order << 2) |
        (uint64) operation;
}

static uint32
evoke_am_accelerator_forward_term(
    const evoke_am_accelerator_forward_entry *entry
)
{
    return (uint32) (entry->sort_key >> 32);
}

static uint8
evoke_am_accelerator_forward_operation(
    const evoke_am_accelerator_forward_entry *entry
)
{
    return (uint8) (entry->sort_key & UINT64_C(3));
}

typedef struct evoke_am_accelerator_source
{
    uint64 authority_checksum;
    uint32 document_count;
    uint32 visible_document_count;
    uint32 vocab_size;
    uint32 block_count;
    uint32 document_shift;
    uint32 documents_per_chunk;
    uint32 selected_count;
    uint32 forward_chunk_count;
    uint64 *document_offsets;
    uint32 *selected_terms;
    uint64 *selected_offsets;
    uint32 *selected_documents;
    uint32 *view_term_ids;
    float *view_impacts;
    uint32 view_capacity;
    uint8 *term_bytes;
    size_t term_size;
    uint8 *forward_bytes;
    size_t forward_size;
    uint8 *forward_bound_bytes;
    size_t forward_bound_size;
    BufFile *forward_file;
    BufFile *forward_bound_file;
    int *document_file_numbers;
    off_t *document_file_offsets;
    int *forward_bound_file_numbers;
    off_t *forward_bound_file_offsets;
    uint64 *forward_bound_term_sizes;
    evoke_am_accelerator_forward_entry *forward_entries;
    uint64 forward_capacity;
    uint32 *chunk_offsets;
    uint32 *chunk_term_ids;
    uint8 *chunk_operations;
    double *chunk_contributions;
    ItemPointerData *document_tids;
    uint8 *tid_lookup_bytes;
    size_t tid_lookup_size;
    uint8 *scope_bytes;
    size_t scope_size;
} evoke_am_accelerator_source;

static void
evoke_am_accelerator_source_init(evoke_am_accelerator_source *source)
{
    memset(source, 0, sizeof(*source));
}

static void
evoke_am_accelerator_source_free(evoke_am_accelerator_source *source)
{
    if (source == NULL)
    {
        return;
    }
    free(source->document_offsets);
    free(source->selected_terms);
    free(source->selected_offsets);
    free(source->selected_documents);
    free(source->view_term_ids);
    free(source->view_impacts);
    free(source->term_bytes);
    free(source->forward_bytes);
    free(source->forward_bound_bytes);
    if (source->forward_file != NULL)
    {
        BufFileClose(source->forward_file);
    }
    if (source->forward_bound_file != NULL)
    {
        BufFileClose(source->forward_bound_file);
    }
    free(source->document_file_numbers);
    free(source->document_file_offsets);
    free(source->forward_bound_file_numbers);
    free(source->forward_bound_file_offsets);
    free(source->forward_bound_term_sizes);
    free(source->forward_entries);
    free(source->chunk_offsets);
    free(source->chunk_term_ids);
    free(source->chunk_operations);
    free(source->chunk_contributions);
    free(source->document_tids);
    free(source->tid_lookup_bytes);
    free(source->scope_bytes);
    evoke_am_accelerator_source_init(source);
}

static bool
evoke_am_accelerator_retained_is_worse(
    const evoke_am_accelerator_retained_document *left,
    const evoke_am_accelerator_retained_document *right
)
{
    return left->impact < right->impact ||
        (left->impact == right->impact &&
         left->document_id > right->document_id);
}

static void
evoke_am_accelerator_retained_swap(
    evoke_am_accelerator_retained_document *left,
    evoke_am_accelerator_retained_document *right
)
{
    evoke_am_accelerator_retained_document temporary = *left;

    *left = *right;
    *right = temporary;
}

static void
evoke_am_accelerator_retained_sift_up(
    evoke_am_accelerator_retained_document *heap,
    uint32 position
)
{
    while (position > 0)
    {
        uint32 parent = (position - 1U) / 2U;

        if (!evoke_am_accelerator_retained_is_worse(
                &heap[position],
                &heap[parent]))
        {
            break;
        }
        evoke_am_accelerator_retained_swap(
            &heap[position],
            &heap[parent]
        );
        position = parent;
    }
}

static void
evoke_am_accelerator_retained_sift_down(
    evoke_am_accelerator_retained_document *heap,
    uint32 count
)
{
    uint32 position = 0;

    for (;;)
    {
        uint32 left = position * 2U + 1U;
        uint32 right = left + 1U;
        uint32 worst = position;

        if (left < count && evoke_am_accelerator_retained_is_worse(
                &heap[left],
                &heap[worst]))
        {
            worst = left;
        }
        if (right < count && evoke_am_accelerator_retained_is_worse(
                &heap[right],
                &heap[worst]))
        {
            worst = right;
        }
        if (worst == position)
        {
            break;
        }
        evoke_am_accelerator_retained_swap(
            &heap[position],
            &heap[worst]
        );
        position = worst;
    }
}

static void
evoke_am_accelerator_retained_offer(
    evoke_am_accelerator_retained_document *heap,
    uint32 *count,
    uint32 document_id,
    double contribution
)
{
    evoke_am_accelerator_retained_document candidate;

    candidate.document_id = document_id;
    candidate.impact = contribution > FLT_MAX
        ? FLT_MAX
        : (float) contribution;
    if (*count < EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP)
    {
        heap[*count] = candidate;
        evoke_am_accelerator_retained_sift_up(heap, *count);
        (*count)++;
        return;
    }
    if (!evoke_am_accelerator_retained_is_worse(&heap[0], &candidate))
    {
        return;
    }
    heap[0] = candidate;
    evoke_am_accelerator_retained_sift_down(heap, *count);
}

static int
evoke_am_accelerator_compare_retained_document(
    const void *left,
    const void *right
)
{
    const evoke_am_accelerator_retained_document *a = left;
    const evoke_am_accelerator_retained_document *b = right;

    return a->document_id < b->document_id
        ? -1
        : a->document_id > b->document_id ? 1 : 0;
}

static evoke_status
evoke_am_accelerator_next_forward_term(
    const evoke_am_accelerator_source *source,
    uint64 end,
    uint64 *position_io,
    uint32 *term_id_out,
    double *contribution_out
)
{
    uint64 position;
    uint32 term_id;
    double contribution = 0.0;

    if (source == NULL || position_io == NULL || term_id_out == NULL ||
        contribution_out == NULL || *position_io >= end)
    {
        return EVOKE_ERR_INVALID;
    }
    position = *position_io;
    term_id = evoke_am_accelerator_forward_term(
        &source->forward_entries[position]
    );
    do
    {
        contribution += source->forward_entries[position].contribution;
        position++;
    }
    while (position < end &&
           evoke_am_accelerator_forward_term(
               &source->forward_entries[position]
           ) == term_id);
    if (!isfinite(contribution))
    {
        return EVOKE_ERR_RANGE;
    }
    *position_io = position;
    *term_id_out = term_id;
    *contribution_out = contribution;
    return EVOKE_OK;
}

static evoke_status
evoke_am_accelerator_next_quantized_forward_term(
    const evoke_am_accelerator_source *source,
    uint64 end,
    uint64 *position_io,
    float scale,
    uint32 *term_id_out,
    double *contribution_out,
    double *quantized_contribution_out
)
{
    uint64 position;
    uint32 term_id;
    double contribution = 0.0;
    double quantized_contribution = 0.0;

    if (source == NULL || position_io == NULL || term_id_out == NULL ||
        contribution_out == NULL || quantized_contribution_out == NULL ||
        *position_io >= end || !isfinite(scale) || scale <= 0.0f)
    {
        return EVOKE_ERR_INVALID;
    }
    position = *position_io;
    term_id = evoke_am_accelerator_forward_term(
        &source->forward_entries[position]
    );
    do
    {
        double raw = source->forward_entries[position].contribution;
        long code = lround(raw / scale);

        if (code > INT8_MAX)
        {
            code = INT8_MAX;
        }
        else if (code < -INT8_MAX)
        {
            code = -INT8_MAX;
        }
        contribution += raw;
        quantized_contribution += (double) code * scale;
        position++;
    }
    while (position < end &&
           evoke_am_accelerator_forward_term(
               &source->forward_entries[position]
           ) == term_id);
    if (!isfinite(contribution) || !isfinite(quantized_contribution))
    {
        return EVOKE_ERR_RANGE;
    }
    *position_io = position;
    *term_id_out = term_id;
    *contribution_out = contribution;
    *quantized_contribution_out = quantized_contribution;
    return EVOKE_OK;
}

static evoke_status
evoke_am_accelerator_choose_document_shift(
    evoke_am_accelerator_source *source
)
{
    uint32 minimum_shift = 1;
    uint32 maximum_shift = EVOKE_AM_ACCELERATOR_MAX_DOCUMENT_SHIFT;
    uint32 shift;

    if (source == NULL || source->document_count == 0 ||
        source->document_offsets == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (evoke_test_semantic_accelerator_forward_document_shift > 0)
    {
        minimum_shift = (uint32)
            evoke_test_semantic_accelerator_forward_document_shift;
        maximum_shift = minimum_shift;
    }
    for (shift = maximum_shift;
         shift >= minimum_shift;
         shift--)
    {
        uint32 documents_per_chunk = UINT32_C(1) << shift;
        bool fits = true;

        for (uint32 first_document = 0;
             first_document < source->document_count;)
        {
            uint32 document_count = Min(
                documents_per_chunk,
                source->document_count - first_document
            );
            uint64 posting_count =
                source->document_offsets[first_document + document_count] -
                source->document_offsets[first_document];
            uint64 object_bytes =
                EVOKE_AM_ACCELERATOR_FORWARD_HEADER_BYTES +
                ((uint64) document_count + 1U) * sizeof(uint32) +
                posting_count * (
                    sizeof(uint32) + sizeof(uint8) + sizeof(double)
                );

            if (object_bytes > EVOKE_AM_ACCELERATOR_FORWARD_TARGET_BYTES ||
                object_bytes > (uint64) MaxAllocSize)
            {
                fits = false;
                break;
            }
            first_document += document_count;
        }
        if (fits)
        {
            uint64 block_count =
                ((uint64) source->document_count +
                 documents_per_chunk - 1U) >> shift;

            if (block_count == 0 || block_count > UINT32_MAX)
            {
                return EVOKE_ERR_RANGE;
            }
            source->document_shift = shift;
            source->documents_per_chunk = documents_per_chunk;
            source->block_count = (uint32) block_count;
            source->forward_chunk_count = source->block_count;
            return EVOKE_OK;
        }
    }
    return EVOKE_ERR_RANGE;
}

static evoke_status
evoke_am_accelerator_load_forward_range(
    evoke_am_accelerator_source *source,
    uint32 first_document,
    uint32 document_count,
    uint64 *posting_count_out
)
{
    uint64 first_posting;
    uint64 last_posting;
    uint64 posting_count;

    if (source == NULL || source->forward_file == NULL ||
        source->document_file_numbers == NULL ||
        source->document_file_offsets == NULL || posting_count_out == NULL ||
        first_document >= source->document_count || document_count == 0 ||
        document_count > source->document_count - first_document)
    {
        return EVOKE_ERR_INVALID;
    }
    first_posting = source->document_offsets[first_document];
    last_posting = source->document_offsets[first_document + document_count];
    if (last_posting < first_posting ||
        last_posting - first_posting > SIZE_MAX /
            sizeof(*source->forward_entries))
    {
        return EVOKE_ERR_RANGE;
    }
    posting_count = last_posting - first_posting;
    if (posting_count > source->forward_capacity)
    {
        evoke_am_accelerator_forward_entry *next_entries = realloc(
            source->forward_entries,
            (size_t) posting_count * sizeof(*next_entries)
        );

        if (next_entries == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        source->forward_entries = next_entries;
        source->forward_capacity = posting_count;
    }
    if (BufFileSeek(
            source->forward_file,
            source->document_file_numbers[first_document],
            source->document_file_offsets[first_document],
            SEEK_SET) != 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (posting_count > 0)
    {
        BufFileReadExact(
            source->forward_file,
            source->forward_entries,
            (size_t) posting_count * sizeof(*source->forward_entries)
        );
    }
    *posting_count_out = posting_count;
    return EVOKE_OK;
}

static evoke_status
evoke_am_accelerator_flush_forward_bound_block(
    evoke_am_accelerator_source *source,
    LogicalTape **term_tapes,
    uint32 *previous_blocks,
    const float *maxima,
    const uint32 *touched_terms,
    uint32 touched_count,
    uint32 block_id
)
{
    if (source == NULL || term_tapes == NULL || previous_blocks == NULL ||
        source->forward_bound_term_sizes == NULL || maxima == NULL ||
        (touched_count > 0 && touched_terms == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint32 touched = 0; touched < touched_count; touched++)
    {
        uint32 term_id = touched_terms[touched];
        float maximum = maxima[term_id];
        uint32 delta;
        uint8 encoded[9];
        size_t encoded_size;
        evoke_status status;

        if (term_id >= source->vocab_size || term_tapes[term_id] == NULL ||
            !isfinite(maximum) || maximum <= 0.0f)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (source->forward_bound_term_sizes[term_id] > 0 &&
            block_id <= previous_blocks[term_id])
        {
            return EVOKE_ERR_FORMAT;
        }
        delta = source->forward_bound_term_sizes[term_id] == 0
            ? block_id
            : block_id - previous_blocks[term_id];
        status = evoke_semantic_forward_bound_encode_entry(
            encoded,
            sizeof(encoded),
            delta,
            maximum,
            &encoded_size
        );
        if (status != EVOKE_OK ||
            UINT64_MAX - source->forward_bound_term_sizes[term_id] <
                encoded_size)
        {
            return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
        }
        LogicalTapeWrite(term_tapes[term_id], encoded, encoded_size);
        source->forward_bound_term_sizes[term_id] += encoded_size;
        previous_blocks[term_id] = block_id;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_am_accelerator_write_forward_bound_file(
    evoke_am_accelerator_source *source,
    LogicalTape **term_tapes
)
{
    uint8 copy_buffer[BLCKSZ];

    if (source == NULL || term_tapes == NULL || source->vocab_size == 0 ||
        source->block_count == 0 ||
        source->forward_bound_term_sizes == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    source->forward_bound_file_numbers = calloc(
        source->vocab_size,
        sizeof(*source->forward_bound_file_numbers)
    );
    source->forward_bound_file_offsets = calloc(
        source->vocab_size,
        sizeof(*source->forward_bound_file_offsets)
    );
    if (source->forward_bound_file_numbers == NULL ||
        source->forward_bound_file_offsets == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    source->forward_bound_file = BufFileCreateTemp(false);
    for (uint32 term_id = 0; term_id < source->vocab_size; term_id++)
    {
        uint64 remaining = source->forward_bound_term_sizes[term_id];

        BufFileTell(
            source->forward_bound_file,
            &source->forward_bound_file_numbers[term_id],
            &source->forward_bound_file_offsets[term_id]
        );
        if (remaining == 0)
        {
            continue;
        }
        LogicalTapeRewindForRead(term_tapes[term_id], BLCKSZ);
        while (remaining > 0)
        {
            size_t requested = (size_t) Min(
                remaining,
                (uint64) sizeof(copy_buffer)
            );
            size_t read_size = LogicalTapeRead(
                term_tapes[term_id],
                copy_buffer,
                requested
            );

            if (read_size != requested)
            {
                return EVOKE_ERR_FORMAT;
            }
            BufFileWrite(source->forward_bound_file, copy_buffer, read_size);
            remaining -= read_size;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_am_accelerator_finalize_forward(
    evoke_am_accelerator_source *source,
    const uint32 *document_frequencies
)
{
    int32 *selected_map = NULL;
    evoke_am_accelerator_retained_document *retained_documents = NULL;
    uint32 *retained_counts = NULL;
    float *bound_maxima = NULL;
    uint32 *bound_touched_terms = NULL;
    LogicalTapeSet *bound_tape_set = NULL;
    LogicalTape **bound_term_tapes = NULL;
    uint32 *bound_previous_blocks = NULL;
    uint64 retained_capacity = 0;
    uint32 bound_touched_count = 0;
    uint32 current_bound_block = 0;
    bool have_bound_block = false;
    float actual_mass = 0.0f;
    evoke_status status = EVOKE_OK;

    if (source == NULL || document_frequencies == NULL ||
        source->forward_file == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_am_accelerator_choose_document_shift(source);
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    status = evoke_semantic_accelerator_select_terms(
        document_frequencies,
        source->vocab_size,
        (float) evoke_test_semantic_accelerator_posting_mass,
        &source->selected_terms,
        &source->selected_count,
        &actual_mass
    );
    if (status != EVOKE_OK || source->selected_count == 0)
    {
        status = status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
        goto cleanup;
    }
    selected_map = palloc_mul_extended(
        (size_t) source->vocab_size,
        sizeof(*selected_map),
        MCXT_ALLOC_HUGE
    );
    source->selected_offsets = calloc(
        (size_t) source->selected_count + 1U,
        sizeof(*source->selected_offsets)
    );
    if (source->selected_offsets == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (uint32 term = 0; term < source->vocab_size; term++)
    {
        selected_map[term] = -1;
    }
    retained_capacity = (uint64) source->selected_count *
        EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP;
    if (retained_capacity > SIZE_MAX / sizeof(*retained_documents))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    retained_documents = palloc_mul_extended(
        (size_t) retained_capacity,
        sizeof(*retained_documents),
        MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO
    );
    retained_counts = palloc_mul_extended(
        (size_t) source->selected_count,
        sizeof(*retained_counts),
        MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO
    );
    for (uint32 selected = 0; selected < source->selected_count; selected++)
    {
        uint32 term_id = source->selected_terms[selected];

        selected_map[term_id] = (int32) selected;
    }
    bound_maxima = calloc(source->vocab_size, sizeof(*bound_maxima));
    bound_touched_terms = malloc(
        (size_t) source->vocab_size * sizeof(*bound_touched_terms)
    );
    if (bound_maxima == NULL || bound_touched_terms == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    source->forward_bound_term_sizes = calloc(
        source->vocab_size,
        sizeof(*source->forward_bound_term_sizes)
    );
    bound_previous_blocks = calloc(
        source->vocab_size,
        sizeof(*bound_previous_blocks)
    );
    bound_term_tapes = calloc(
        source->vocab_size,
        sizeof(*bound_term_tapes)
    );
    if (source->forward_bound_term_sizes == NULL ||
        bound_previous_blocks == NULL || bound_term_tapes == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    /* Hundreds of sparse term tapes must not reserve sort-sized extents. */
    bound_tape_set = LogicalTapeSetCreate(false, NULL, -1);
    for (uint32 term_id = 0; term_id < source->vocab_size; term_id++)
    {
        bound_term_tapes[term_id] = LogicalTapeCreate(bound_tape_set);
    }
    for (uint32 first_document = 0;
         first_document < source->document_count;
         first_document += source->documents_per_chunk)
    {
        uint32 document_count = Min(
            source->documents_per_chunk,
            source->document_count - first_document
        );
        uint64 first_posting = source->document_offsets[first_document];
        uint64 posting_count;

        status = evoke_am_accelerator_load_forward_range(
            source,
            first_document,
            document_count,
            &posting_count
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        for (uint32 document_offset = 0;
             document_offset < document_count;
             document_offset++)
        {
            uint32 document = first_document + document_offset;
            uint32 bound_block = document >>
                EVOKE_SEMANTIC_FORWARD_BOUND_BLOCK_SHIFT;
            uint64 position = source->document_offsets[document] -
                first_posting;
            uint64 end = source->document_offsets[document + 1U] -
                first_posting;
            double maximum_absolute = 0.0;
            float scale;

            if (!have_bound_block)
            {
                current_bound_block = bound_block;
                have_bound_block = true;
            }
            else if (bound_block != current_bound_block)
            {
                status = evoke_am_accelerator_flush_forward_bound_block(
                    source,
                    bound_term_tapes,
                    bound_previous_blocks,
                    bound_maxima,
                    bound_touched_terms,
                    bound_touched_count,
                    current_bound_block
                );
                if (status != EVOKE_OK)
                {
                    goto cleanup;
                }
                for (uint32 touched = 0;
                     touched < bound_touched_count;
                     touched++)
                {
                    bound_maxima[bound_touched_terms[touched]] = 0.0f;
                }
                bound_touched_count = 0;
                current_bound_block = bound_block;
            }
            if (end < position || end > posting_count)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            for (uint64 posting = position; posting < end; posting++)
            {
                double absolute = fabs(
                    source->forward_entries[posting].contribution
                );

                if (absolute > maximum_absolute)
                {
                    maximum_absolute = absolute;
                }
            }
            scale = maximum_absolute == 0.0
                ? 1.0f
                : (float) (maximum_absolute / (double) INT8_MAX);
            if (!isfinite(scale) || scale <= 0.0f)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            while (position < end)
            {
                uint32 term_id;
                double contribution;
                double quantized_contribution;
                int32 selected;

                status = evoke_am_accelerator_next_quantized_forward_term(
                    source,
                    end,
                    &position,
                    scale,
                    &term_id,
                    &contribution,
                    &quantized_contribution
                );
                if (status != EVOKE_OK || term_id >= source->vocab_size)
                {
                    status = status == EVOKE_OK
                        ? EVOKE_ERR_RANGE
                        : status;
                    goto cleanup;
                }
                if (quantized_contribution > 0.0)
                {
                    float maximum;

                    if (quantized_contribution >= FLT_MAX)
                    {
                        maximum = FLT_MAX;
                    }
                    else
                    {
                        maximum = (float) quantized_contribution;
                        if ((double) maximum < quantized_contribution)
                        {
                            maximum = nextafterf(maximum, INFINITY);
                        }
                    }

                    if (bound_maxima[term_id] == 0.0f)
                    {
                        if (bound_touched_count >= source->vocab_size)
                        {
                            status = EVOKE_ERR_RANGE;
                            goto cleanup;
                        }
                        bound_touched_terms[bound_touched_count++] = term_id;
                    }
                    if (maximum > bound_maxima[term_id])
                    {
                        bound_maxima[term_id] = maximum;
                    }
                }
                selected = selected_map[term_id];
                if (selected >= 0 && contribution > 0.0)
                {
                    evoke_am_accelerator_retained_offer(
                        &retained_documents[
                            (size_t) selected *
                            EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP
                        ],
                        &retained_counts[selected],
                        document,
                        contribution
                    );
                }
            }
        }
        CHECK_FOR_INTERRUPTS();
    }
    if (have_bound_block)
    {
        status = evoke_am_accelerator_flush_forward_bound_block(
            source,
            bound_term_tapes,
            bound_previous_blocks,
            bound_maxima,
            bound_touched_terms,
            bound_touched_count,
            current_bound_block
        );
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    status = evoke_am_accelerator_write_forward_bound_file(
        source,
        bound_term_tapes
    );
    if (status != EVOKE_OK)
    {
        goto cleanup;
    }
    for (uint32 selected = 0; selected < source->selected_count; selected++)
    {
        if (UINT64_MAX - source->selected_offsets[selected] <
            retained_counts[selected])
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        source->selected_offsets[selected + 1U] =
            source->selected_offsets[selected] + retained_counts[selected];
    }
    if (source->selected_offsets[source->selected_count] >
        SIZE_MAX / sizeof(*source->selected_documents))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    source->selected_documents = malloc(
        (size_t) source->selected_offsets[source->selected_count] *
            sizeof(*source->selected_documents)
    );
    if (source->selected_documents == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    for (uint32 selected = 0; selected < source->selected_count; selected++)
    {
        evoke_am_accelerator_retained_document *heap =
            &retained_documents[
                (size_t) selected *
                EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP
            ];

        qsort(
            heap,
            retained_counts[selected],
            sizeof(*heap),
            evoke_am_accelerator_compare_retained_document
        );
        for (uint32 item = 0; item < retained_counts[selected]; item++)
        {
            source->selected_documents[
                source->selected_offsets[selected] + item
            ] = heap[item].document_id;
        }
    }
cleanup:
    if (bound_term_tapes != NULL)
    {
        for (uint32 term_id = 0; term_id < source->vocab_size; term_id++)
        {
            if (bound_term_tapes[term_id] != NULL)
            {
                LogicalTapeClose(bound_term_tapes[term_id]);
            }
        }
    }
    if (bound_tape_set != NULL)
    {
        LogicalTapeSetClose(bound_tape_set);
    }
    free(bound_term_tapes);
    free(bound_previous_blocks);
    free(bound_maxima);
    free(bound_touched_terms);
    if (selected_map != NULL)
    {
        pfree(selected_map);
    }
    if (retained_documents != NULL)
    {
        pfree(retained_documents);
    }
    if (retained_counts != NULL)
    {
        pfree(retained_counts);
    }
    return status;
}

static evoke_status
evoke_am_accelerator_read_document(
    void *context,
    uint32 document_id,
    evoke_semantic_accelerator_document_view *view_out
)
{
    evoke_am_accelerator_source *source = context;
    uint64 position;
    uint64 end;
    uint64 required;
    uint64 loaded;
    uint32 count = 0;
    evoke_status status;

    if (source == NULL || view_out == NULL ||
        document_id >= source->document_count)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(view_out, 0, sizeof(*view_out));
    position = source->document_offsets[document_id];
    end = source->document_offsets[document_id + 1U];
    if (end < position || end - position > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    required = end - position;
    status = evoke_am_accelerator_load_forward_range(
        source,
        document_id,
        1,
        &loaded
    );
    if (status != EVOKE_OK || loaded != required)
    {
        return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
    }
    position = 0;
    end = loaded;
    if (required > source->view_capacity)
    {
        uint32 *next_term_ids;
        float *next_impacts;

        next_term_ids = realloc(
            source->view_term_ids,
            (size_t) required * sizeof(*next_term_ids)
        );
        if (next_term_ids == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        source->view_term_ids = next_term_ids;
        next_impacts = realloc(
            source->view_impacts,
            (size_t) required * sizeof(*next_impacts)
        );
        if (next_impacts == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        source->view_impacts = next_impacts;
        source->view_capacity = (uint32) required;
    }
    while (position < end)
    {
        uint32 term_id;
        double contribution;
        float impact;
        status = evoke_am_accelerator_next_forward_term(
            source,
            end,
            &position,
            &term_id,
            &contribution
        );

        if (status != EVOKE_OK || term_id >= source->vocab_size)
        {
            return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
        }
        if (contribution <= 0.0)
        {
            continue;
        }
        impact = contribution > FLT_MAX ? FLT_MAX : (float) contribution;
        if (!isfinite(impact) || impact <= 0.0f)
        {
            continue;
        }
        source->view_term_ids[count] = term_id;
        source->view_impacts[count] = impact;
        count++;
    }
    if (count == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    view_out->term_ids = source->view_term_ids;
    view_out->impacts = source->view_impacts;
    view_out->term_count = count;
    return EVOKE_OK;
}

static evoke_status
evoke_am_accelerator_posting_contribution(
    const evoke_segment_query_context *context,
    const uint32 *document_lengths,
    evoke_posting_extent_kind kind,
    uint32 document,
    const evoke_posting_value *value,
    double average_document_length,
    double idf,
    double nonoccurrence,
    double *contribution_out
)
{
    double contribution;

    if (context == NULL || document_lengths == NULL || value == NULL ||
        contribution_out == NULL ||
        document >= context->manifest.document_slot_count)
    {
        return EVOKE_ERR_INVALID;
    }
    if (kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL)
    {
        uint32 term_frequency = value->term_frequency;
        double tfc;

        if (term_frequency == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        tfc = evoke_score_tfc(
            context->query_contract.params.method,
            (double) term_frequency,
            document_lengths[document],
            average_document_length,
            context->query_contract.params.k1,
            context->query_contract.params.b,
            context->query_contract.params.delta
        );
        contribution = idf * tfc;
        if (evoke_method_requires_nonoccurrence(
                context->query_contract.params.method))
        {
            contribution -= nonoccurrence;
        }
    }
    else if (kind == EVOKE_POSTING_EXTENT_LEXICAL_IMPACT ||
             kind == EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT)
    {
        contribution = value->impact;
    }
    else
    {
        return EVOKE_ERR_FORMAT;
    }
    if (!isfinite(contribution))
    {
        return EVOKE_ERR_RANGE;
    }
    *contribution_out = contribution;
    return EVOKE_OK;
}

typedef struct evoke_am_accelerator_document_metadata
{
    uint32 *document_lengths;
    uint8 *unavailable_documents;
    ItemPointerData *document_tids;
    uint64 next_document;
    uint64 visible_documents;
    uint64 total_document_length;
    uint64 document_count;
    evoke_status status;
} evoke_am_accelerator_document_metadata;

static bool
evoke_am_accelerator_document_unavailable(
    const uint8 *unavailable_documents,
    uint32 document
)
{
    return (unavailable_documents[document >> 3] &
            (UINT8_C(1) << (document & 7U))) != 0;
}

static void
evoke_am_accelerator_collect_document_metadata(
    void *context,
    const evoke_document_cow_record *record
)
{
    evoke_am_accelerator_document_metadata *metadata = context;
    const evoke_document_version_record *version;
    uint32 document;
    bool unavailable;

    if (metadata == NULL || record == NULL ||
        metadata->status != EVOKE_OK)
    {
        return;
    }
    version = &record->version;
    if (version->document_slot != metadata->next_document ||
        version->document_slot >= metadata->document_count)
    {
        metadata->status = EVOKE_ERR_FORMAT;
        return;
    }
    document = (uint32) version->document_slot;
    metadata->next_document++;
    metadata->document_lengths[document] = version->document_length;
    unavailable =
        (version->flags & EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) != 0 ||
        record->retirement.retirement_sequence != 0;
    if (unavailable)
    {
        metadata->unavailable_documents[document >> 3] |=
            (uint8) (UINT8_C(1) << (document & 7U));
        return;
    }
    if (version->heap_offset == InvalidOffsetNumber)
    {
        metadata->status = EVOKE_ERR_FORMAT;
        return;
    }
    ItemPointerSet(
        &metadata->document_tids[document],
        (BlockNumber) version->heap_block,
        (OffsetNumber) version->heap_offset
    );
    if (UINT64_MAX - metadata->total_document_length <
        version->document_length)
    {
        metadata->status = EVOKE_ERR_RANGE;
        return;
    }
    metadata->visible_documents++;
    metadata->total_document_length += version->document_length;
}

static evoke_status
evoke_am_accelerator_source_build_query_context(
    Relation index_relation,
    const evoke_segment_query_context *context,
    evoke_am_accelerator_source *source
)
{
    AttrNumber sort_attnums[4] = {1, 2, 3, 4};
    Oid sort_operators[4];
    Oid sort_collations[4] = {
        InvalidOid,
        InvalidOid,
        InvalidOid,
        InvalidOid
    };
    bool nulls_first[4] = {false, false, false, false};
    evoke_am_accelerator_document_metadata metadata;
    TupleDesc sort_desc = NULL;
    TupleTableSlot *input_slot = NULL;
    TupleTableSlot *output_slot = NULL;
    Tuplesortstate *sort = NULL;
    uint32 *document_frequencies = NULL;
    uint32 *seen_document_terms = NULL;
    uint32 *posting_documents = NULL;
    evoke_posting_value *posting_values = NULL;
    uint8 *unavailable_documents = NULL;
    uint32 *document_lengths = NULL;
    uint64 output_count = 0;
    uint64 next_document = 0;
    uint32 group_document = 0;
    uint32 group_term = 0;
    double group_contribution = 0.0;
    bool have_group = false;
    double average_document_length;
    evoke_status status = EVOKE_OK;

    if (index_relation == NULL || context == NULL || source == NULL ||
        context->manifest.document_slot_count == 0 ||
        context->manifest.document_slot_count > UINT32_MAX ||
        context->manifest.vocab_size == 0 ||
        context->manifest.vocab_size == UINT32_MAX ||
        context->manifest.visible_document_count == 0 ||
        context->manifest.total_document_length == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    source->document_count =
        (uint32) context->manifest.document_slot_count;
    source->visible_document_count =
        (uint32) context->manifest.visible_document_count;
    source->vocab_size = context->manifest.vocab_size;
    status = evoke_segment_manifest_authority_checksum(
        &context->manifest,
        &source->authority_checksum
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    document_lengths = palloc_mul_extended(
        (size_t) source->document_count,
        sizeof(*document_lengths),
        MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO
    );
    unavailable_documents = palloc_extended(
        ((size_t) source->document_count + 7U) / 8U,
        MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO
    );
    document_frequencies = palloc_mul_extended(
        (size_t) source->vocab_size,
        sizeof(*document_frequencies),
        MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO
    );
    if (context->manifest.visible_document_count !=
        context->manifest.document_slot_count)
    {
        seen_document_terms = palloc_mul_extended(
            (size_t) source->document_count,
            sizeof(*seen_document_terms),
            MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO
        );
    }
    posting_documents = palloc(
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW * sizeof(*posting_documents)
    );
    posting_values = palloc(
        EVOKE_SEGMENT_QUERY_POSTING_WINDOW * sizeof(*posting_values)
    );
    source->document_offsets = calloc(
        (size_t) source->document_count + 1U,
        sizeof(*source->document_offsets)
    );
    source->document_file_numbers = calloc(
        source->document_count,
        sizeof(*source->document_file_numbers)
    );
    source->document_file_offsets = calloc(
        source->document_count,
        sizeof(*source->document_file_offsets)
    );
    source->document_tids = calloc(
        source->document_count,
        sizeof(*source->document_tids)
    );
    if (source->document_offsets == NULL ||
        source->document_file_numbers == NULL ||
        source->document_file_offsets == NULL ||
        source->document_tids == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    memset(&metadata, 0, sizeof(metadata));
    metadata.document_lengths = document_lengths;
    metadata.unavailable_documents = unavailable_documents;
    metadata.document_tids = source->document_tids;
    metadata.document_count = source->document_count;
    metadata.status = EVOKE_OK;
    evoke_segment_pages_visit_document_records(
        index_relation,
        &context->root,
        &context->manifest,
        evoke_am_accelerator_collect_document_metadata,
        &metadata
    );
    if (metadata.status != EVOKE_OK ||
        metadata.next_document != source->document_count ||
        metadata.visible_documents !=
            context->manifest.visible_document_count ||
        metadata.total_document_length !=
            context->manifest.total_document_length)
    {
        status = metadata.status == EVOKE_OK
            ? EVOKE_ERR_FORMAT
            : metadata.status;
        goto cleanup;
    }
    average_document_length =
        (double) context->manifest.total_document_length /
        (double) context->manifest.visible_document_count;

    sort_desc = CreateTemplateTupleDesc(5);
    TupleDescInitEntry(sort_desc, 1, "document_id", INT8OID, -1, 0);
    TupleDescInitEntry(sort_desc, 2, "term_id", INT8OID, -1, 0);
    TupleDescInitEntry(sort_desc, 3, "contribution_order", INT8OID, -1, 0);
    TupleDescInitEntry(sort_desc, 4, "operation", INT8OID, -1, 0);
    TupleDescInitEntry(sort_desc, 5, "contribution", FLOAT8OID, -1, 0);
    BlessTupleDesc(sort_desc);
    input_slot = MakeSingleTupleTableSlot(sort_desc, &TTSOpsVirtual);
    output_slot = MakeSingleTupleTableSlot(sort_desc, &TTSOpsMinimalTuple);
    get_sort_group_operators(
        INT8OID,
        true,
        false,
        false,
        &sort_operators[0],
        NULL,
        NULL,
        NULL
    );
    sort_operators[1] = sort_operators[0];
    sort_operators[2] = sort_operators[0];
    sort_operators[3] = sort_operators[0];
    sort = tuplesort_begin_heap(
        sort_desc,
        4,
        sort_attnums,
        sort_operators,
        sort_collations,
        nulls_first,
        Min(
            maintenance_work_mem,
            (int) EVOKE_AM_ACCELERATOR_SORT_MEMORY_MAX_KB
        ),
        NULL,
        TUPLESORT_NONE
    );

    /*
     * Type and sort setup may acquire a catalog snapshot.  The remaining
     * transpose reads only the immutable index generation, so retaining that
     * snapshot would pin vacuum for the full accelerator build.
     */
    InvalidateCatalogSnapshot();
    pgstat_report_activity(
        STATE_RUNNING,
        "evoke maintenance: prepare semantic query accelerator"
    );

    for (uint32 term_id = 0; term_id < source->vocab_size; term_id++)
    {
        evoke_segment_query_term_plan plan;
        uint32 live_document_frequency;
        uint32 seen_marker = term_id + 1U;
        double idf = 0.0;
        double nonoccurrence = 0.0;

        evoke_segment_pages_load_query_term_plan(
            index_relation,
            context,
            term_id,
            &plan
        );
        live_document_frequency = plan.raw_document_frequency;
        if (seen_document_terms != NULL)
        {
            live_document_frequency = 0;
            for (uint32 run_index = 0;
                 run_index < plan.run_count;
                 run_index++)
            {
                const evoke_segment_query_run *run = &plan.runs[run_index];
                evoke_segment_query_posting_cursor cursor;
                uint32 posting_count;

                if (run->kind != EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL &&
                    run->kind != EVOKE_POSTING_EXTENT_LEXICAL_IMPACT)
                {
                    continue;
                }
                evoke_segment_query_posting_cursor_init(&cursor);
                while (cursor.next_posting_index < run->posting_count)
                {
                    posting_count =
                        evoke_segment_pages_load_query_term_posting_stream_window(
                            index_relation,
                            context,
                            &plan,
                            run_index,
                            &cursor,
                            posting_documents,
                            posting_values,
                            EVOKE_SEGMENT_QUERY_POSTING_WINDOW);
                    for (uint32 posting = 0;
                         posting < posting_count;
                         posting++)
                    {
                        uint32 document = posting_documents[posting];

                        if (document >= source->document_count)
                        {
                            status = EVOKE_ERR_RANGE;
                            goto cleanup;
                        }
                        if (!evoke_am_accelerator_document_unavailable(
                                unavailable_documents,
                                document) &&
                            seen_document_terms[document] != seen_marker)
                        {
                            if (live_document_frequency == UINT32_MAX)
                            {
                                status = EVOKE_ERR_RANGE;
                                goto cleanup;
                            }
                            seen_document_terms[document] = seen_marker;
                            live_document_frequency++;
                        }
                    }
                }
            }
        }
        if (live_document_frequency > 0)
        {
            idf = evoke_score_idf(
                context->query_contract.params.idf_method,
                live_document_frequency,
                context->manifest.visible_document_count
            );
            if (evoke_method_requires_nonoccurrence(
                    context->query_contract.params.method))
            {
                nonoccurrence = idf * evoke_score_tfc(
                    context->query_contract.params.method,
                    0.0,
                    0.0,
                    average_document_length,
                    context->query_contract.params.k1,
                    context->query_contract.params.b,
                    context->query_contract.params.delta
                );
            }
        }
        for (uint32 run_index = 0;
             run_index < plan.run_count;
             run_index++)
        {
            const evoke_segment_query_run *run = &plan.runs[run_index];
            evoke_segment_query_posting_cursor cursor;
            uint32 posting_count;
            uint8 operation =
                run->kind == EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL
                    ? EVOKE_SEMANTIC_FORWARD_DOUBLE_PRODUCT
                    : EVOKE_SEMANTIC_FORWARD_FLOAT_IMPACT;

            if (run_index > EVOKE_AM_ACCELERATOR_FORWARD_ORDER_MAX)
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
            evoke_segment_query_posting_cursor_init(&cursor);
            while (cursor.next_posting_index < run->posting_count)
            {
                posting_count =
                    evoke_segment_pages_load_query_term_posting_stream_window(
                        index_relation,
                        context,
                        &plan,
                        run_index,
                        &cursor,
                        posting_documents,
                        posting_values,
                        EVOKE_SEGMENT_QUERY_POSTING_WINDOW);
                for (uint32 posting = 0;
                     posting < posting_count;
                     posting++)
                {
                    uint32 document = posting_documents[posting];
                    double contribution;

                    status = evoke_am_accelerator_posting_contribution(
                        context,
                        document_lengths,
                        run->kind,
                        document,
                        &posting_values[posting],
                        average_document_length,
                        idf,
                        nonoccurrence,
                        &contribution
                    );
                    if (status != EVOKE_OK)
                    {
                        goto cleanup;
                    }
                    if (evoke_am_accelerator_document_unavailable(
                            unavailable_documents,
                            document) ||
                        contribution == 0.0)
                    {
                        continue;
                    }
                    ExecClearTuple(input_slot);
                    input_slot->tts_values[0] =
                        Int64GetDatum((int64) document);
                    input_slot->tts_values[1] =
                        Int64GetDatum((int64) term_id);
                    input_slot->tts_values[2] =
                        Int64GetDatum((int64) run_index);
                    input_slot->tts_values[3] =
                        Int64GetDatum((int64) operation);
                    input_slot->tts_values[4] =
                        Float8GetDatum(contribution);
                    for (uint32 attribute = 0;
                         attribute < 5;
                         attribute++)
                    {
                        input_slot->tts_isnull[attribute] = false;
                    }
                    ExecStoreVirtualTuple(input_slot);
                    tuplesort_puttupleslot(sort, input_slot);
                }
            }
        }
        if ((term_id & UINT32_C(1023)) == 0)
        {
            CHECK_FOR_INTERRUPTS();
        }
    }

    source->forward_file = BufFileCreateTemp(false);
    tuplesort_performsort(sort);
    while (tuplesort_gettupleslot(
        sort,
        true,
        false,
        output_slot,
        NULL
    ))
    {
        bool isnull[5];
        int64 document_value;
        int64 term_value;
        int64 order_value;
        int64 operation_value;
        double contribution;
        evoke_am_accelerator_forward_entry entry;
        uint32 document;
        uint32 term_id;

        Datum document_datum = slot_getattr(
            output_slot,
            1,
            &isnull[0]
        );
        Datum term_datum = slot_getattr(output_slot, 2, &isnull[1]);
        Datum order_datum = slot_getattr(output_slot, 3, &isnull[2]);
        Datum operation_datum = slot_getattr(
            output_slot,
            4,
            &isnull[3]
        );
        Datum contribution_datum = slot_getattr(
            output_slot,
            5,
            &isnull[4]
        );

        if (isnull[0] || isnull[1] || isnull[2] || isnull[3] ||
            isnull[4])
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        document_value = DatumGetInt64(document_datum);
        term_value = DatumGetInt64(term_datum);
        order_value = DatumGetInt64(order_datum);
        operation_value = DatumGetInt64(operation_datum);
        contribution = DatumGetFloat8(contribution_datum);
        if (document_value < 0 ||
            (uint64) document_value >= source->document_count ||
            term_value < 0 || (uint64) term_value >= source->vocab_size ||
            order_value < 0 ||
            (uint64) order_value > EVOKE_AM_ACCELERATOR_FORWARD_ORDER_MAX ||
            operation_value < 0 || operation_value > 3 ||
            !isfinite(contribution) || contribution == 0.0)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        document = (uint32) document_value;
        term_id = (uint32) term_value;
        while (next_document <= document)
        {
            source->document_offsets[next_document] = output_count;
            BufFileTell(
                source->forward_file,
                &source->document_file_numbers[next_document],
                &source->document_file_offsets[next_document]
            );
            next_document++;
        }
        if (have_group &&
            (group_document != document || group_term != term_id))
        {
            if (group_contribution > 0.0)
            {
                if (document_frequencies[group_term] == UINT32_MAX)
                {
                    status = EVOKE_ERR_RANGE;
                    goto cleanup;
                }
                document_frequencies[group_term]++;
            }
            group_document = document;
            group_term = term_id;
            group_contribution = contribution;
        }
        else if (!have_group)
        {
            group_document = document;
            group_term = term_id;
            group_contribution = contribution;
            have_group = true;
        }
        else
        {
            group_contribution += contribution;
            if (!isfinite(group_contribution))
            {
                status = EVOKE_ERR_RANGE;
                goto cleanup;
            }
        }
        entry.sort_key = evoke_am_accelerator_forward_key(
            term_id,
            (uint32) order_value,
            (uint8) operation_value
        );
        entry.contribution = contribution;
        BufFileWrite(source->forward_file, &entry, sizeof(entry));
        if (output_count == UINT64_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        output_count++;
        ExecClearTuple(output_slot);
    }
    if (have_group && group_contribution > 0.0)
    {
        if (document_frequencies[group_term] == UINT32_MAX)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        document_frequencies[group_term]++;
    }
    while (next_document < source->document_count)
    {
        source->document_offsets[next_document] = output_count;
        BufFileTell(
            source->forward_file,
            &source->document_file_numbers[next_document],
            &source->document_file_offsets[next_document]
        );
        next_document++;
    }
    source->document_offsets[source->document_count] = output_count;
    tuplesort_end(sort);
    sort = NULL;
    status = evoke_am_accelerator_finalize_forward(
        source,
        document_frequencies
    );

cleanup:
    if (sort != NULL)
    {
        tuplesort_end(sort);
    }
    if (input_slot != NULL)
    {
        ExecDropSingleTupleTableSlot(input_slot);
    }
    if (output_slot != NULL)
    {
        ExecDropSingleTupleTableSlot(output_slot);
    }
    if (sort_desc != NULL)
    {
        FreeTupleDesc(sort_desc);
    }
    if (document_lengths != NULL)
    {
        pfree(document_lengths);
    }
    if (unavailable_documents != NULL)
    {
        pfree(unavailable_documents);
    }
    if (document_frequencies != NULL)
    {
        pfree(document_frequencies);
    }
    if (seen_document_terms != NULL)
    {
        pfree(seen_document_terms);
    }
    if (posting_documents != NULL)
    {
        pfree(posting_documents);
    }
    if (posting_values != NULL)
    {
        pfree(posting_values);
    }
    if (status != EVOKE_OK)
    {
        evoke_am_accelerator_source_free(source);
    }
    return status;
}

static evoke_status
evoke_am_accelerator_produce_term(
    void *context,
    uint32 artifact_index,
    evoke_semantic_accelerator_term_artifact *artifact_out
)
{
    evoke_am_accelerator_source *source = context;
    evoke_semantic_accelerator_builder_options options;
    evoke_semantic_accelerator_owned_term term;
    uint64 start;
    uint64 end;
    evoke_status status;

    if (source == NULL || artifact_out == NULL ||
        artifact_index >= source->selected_count)
    {
        return EVOKE_ERR_INVALID;
    }
    free(source->term_bytes);
    source->term_bytes = NULL;
    source->term_size = 0;
    start = source->selected_offsets[artifact_index];
    end = source->selected_offsets[artifact_index + 1U];
    if (end <= start || end - start > UINT32_MAX)
    {
        return end <= start ? EVOKE_ERR_FORMAT : EVOKE_ERR_RANGE;
    }
    evoke_semantic_accelerator_owned_term_init(&term);
    options.centroid_fraction = EVOKE_AM_ACCELERATOR_CENTROID_FRACTION;
    options.minimum_cluster_size =
        EVOKE_AM_ACCELERATOR_MINIMUM_CLUSTER_SIZE;
    options.document_cut = EVOKE_AM_ACCELERATOR_DOCUMENT_CUT;
    options.summary_energy = EVOKE_AM_ACCELERATOR_SUMMARY_ENERGY;
    options.random_seed = EVOKE_AM_ACCELERATOR_RANDOM_SEED;
    status = evoke_semantic_accelerator_build_term(
        source->selected_terms[artifact_index],
        &source->selected_documents[start],
        (uint32) (end - start),
        &options,
        evoke_am_accelerator_read_document,
        source,
        &term
    );
    if (status != EVOKE_OK)
    {
        evoke_semantic_accelerator_owned_term_free(&term);
        return status;
    }
    status = evoke_semantic_accelerator_term_serialize(
        source->authority_checksum,
        source->document_count,
        &term.input,
        &source->term_bytes,
        &source->term_size
    );
    evoke_semantic_accelerator_owned_term_free(&term);
    if (status != EVOKE_OK || source->term_size > MaxAllocSize)
    {
        free(source->term_bytes);
        source->term_bytes = NULL;
        source->term_size = 0;
        return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
    }
    artifact_out->term_id = source->selected_terms[artifact_index];
    artifact_out->bytes = source->term_bytes;
    artifact_out->size = (Size) source->term_size;
    return EVOKE_OK;
}

static evoke_status
evoke_am_accelerator_produce_forward(
    void *context,
    uint32 artifact_index,
    evoke_semantic_forward_artifact *artifact_out
)
{
    evoke_am_accelerator_source *source = context;
    evoke_semantic_forward_chunk chunk;
    uint32 first_document;
    uint32 document_count;
    uint64 first_posting;
    uint64 last_posting;
    uint32 posting_count;
    evoke_status status;

    if (source == NULL || artifact_out == NULL ||
        artifact_index >= source->forward_chunk_count)
    {
        return EVOKE_ERR_INVALID;
    }
    free(source->forward_bytes);
    free(source->chunk_offsets);
    free(source->chunk_term_ids);
    free(source->chunk_operations);
    free(source->chunk_contributions);
    source->forward_bytes = NULL;
    source->chunk_offsets = NULL;
    source->chunk_term_ids = NULL;
    source->chunk_operations = NULL;
    source->chunk_contributions = NULL;
    source->forward_size = 0;
    first_document = artifact_index << source->document_shift;
    document_count = Min(
        source->documents_per_chunk,
        source->document_count - first_document
    );
    first_posting = source->document_offsets[first_document];
    last_posting = source->document_offsets[
        first_document + document_count
    ];
    if (last_posting < first_posting ||
        last_posting - first_posting > UINT32_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    posting_count = (uint32) (last_posting - first_posting);
    {
        uint64 loaded;

        status = evoke_am_accelerator_load_forward_range(
            source,
            first_document,
            document_count,
            &loaded
        );
        if (status != EVOKE_OK || loaded != posting_count)
        {
            return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
        }
    }
    source->chunk_offsets = malloc(
        ((size_t) document_count + 1U) *
        sizeof(*source->chunk_offsets)
    );
    if (posting_count > 0)
    {
        source->chunk_term_ids = malloc(
            (size_t) posting_count * sizeof(*source->chunk_term_ids)
        );
        source->chunk_operations = malloc(
            (size_t) posting_count * sizeof(*source->chunk_operations)
        );
        source->chunk_contributions = malloc(
            (size_t) posting_count * sizeof(*source->chunk_contributions)
        );
    }
    if (source->chunk_offsets == NULL ||
        (posting_count > 0 &&
         (source->chunk_term_ids == NULL ||
          source->chunk_operations == NULL ||
          source->chunk_contributions == NULL)))
    {
        return EVOKE_ERR_NOMEM;
    }
    for (uint32 document = 0; document <= document_count; document++)
    {
        uint64 offset = source->document_offsets[
            first_document + document
        ] - first_posting;

        if (offset > UINT32_MAX)
        {
            return EVOKE_ERR_RANGE;
        }
        source->chunk_offsets[document] = (uint32) offset;
    }
    for (uint32 posting = 0; posting < posting_count; posting++)
    {
        const evoke_am_accelerator_forward_entry *entry =
            &source->forward_entries[posting];

        source->chunk_term_ids[posting] =
            evoke_am_accelerator_forward_term(entry);
        source->chunk_operations[posting] =
            evoke_am_accelerator_forward_operation(entry);
        source->chunk_contributions[posting] = entry->contribution;
    }
    evoke_semantic_forward_chunk_init(&chunk);
    chunk.source_authority_checksum = source->authority_checksum;
    chunk.first_document = first_document;
    chunk.document_count = document_count;
    chunk.posting_count = posting_count;
    chunk.vocab_size = source->vocab_size;
    chunk.row_offsets = source->chunk_offsets;
    chunk.term_ids = source->chunk_term_ids;
    chunk.operations = source->chunk_operations;
    chunk.contributions = source->chunk_contributions;
    status = evoke_semantic_forward_chunk_serialize(
        &chunk,
        &source->forward_bytes,
        &source->forward_size
    );
    if (status != EVOKE_OK || source->forward_size > MaxAllocSize)
    {
        free(source->forward_bytes);
        source->forward_bytes = NULL;
        source->forward_size = 0;
        return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
    }
    artifact_out->first_document = first_document;
    artifact_out->document_count = document_count;
    artifact_out->bytes = source->forward_bytes;
    artifact_out->size = (Size) source->forward_size;
    return EVOKE_OK;
}

static evoke_status
evoke_am_accelerator_produce_forward_bound(
    void *context,
    uint32 artifact_index,
    evoke_semantic_forward_bound_artifact *artifact_out
)
{
    evoke_am_accelerator_source *source = context;
    uint64 payload_offsets[
        EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD + 1U
    ] = {0};
    uint8 *payload = NULL;
    uint32 shard_count;
    uint32 first_term;
    uint32 term_count;
    size_t payload_size;
    evoke_status status;

    if (source == NULL || artifact_out == NULL ||
        source->forward_bound_file == NULL ||
        source->forward_bound_file_numbers == NULL ||
        source->forward_bound_file_offsets == NULL ||
        source->forward_bound_term_sizes == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    shard_count = (source->vocab_size - 1U) /
        EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD + 1U;
    if (artifact_index >= shard_count)
    {
        return EVOKE_ERR_INVALID;
    }
    free(source->forward_bound_bytes);
    source->forward_bound_bytes = NULL;
    source->forward_bound_size = 0;
    first_term = artifact_index *
        EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD;
    term_count = Min(
        EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD,
        source->vocab_size - first_term
    );
    for (uint32 local_term = 0; local_term < term_count; local_term++)
    {
        uint64 term_size = source->forward_bound_term_sizes[
            first_term + local_term
        ];

        if (term_size > SIZE_MAX ||
            payload_offsets[local_term] > UINT64_MAX - term_size)
        {
            return EVOKE_ERR_RANGE;
        }
        payload_offsets[local_term + 1U] =
            payload_offsets[local_term] + term_size;
    }
    if (payload_offsets[term_count] > MaxAllocSize)
    {
        return EVOKE_ERR_RANGE;
    }
    payload_size = (size_t) payload_offsets[term_count];
    if (payload_size > 0)
    {
        payload = malloc(payload_size);
        if (payload == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
    }
    for (uint32 local_term = 0; local_term < term_count; local_term++)
    {
        uint32 term_id = first_term + local_term;
        size_t term_size =
            (size_t) source->forward_bound_term_sizes[term_id];

        if (BufFileSeek(
                source->forward_bound_file,
                source->forward_bound_file_numbers[term_id],
                source->forward_bound_file_offsets[term_id],
                SEEK_SET) != 0)
        {
            free(payload);
            return EVOKE_ERR_FORMAT;
        }
        if (term_size > 0)
        {
            BufFileReadExact(
                source->forward_bound_file,
                payload + payload_offsets[local_term],
                term_size
            );
        }
    }
    status = evoke_semantic_forward_bound_shard_serialize(
        source->authority_checksum,
        source->document_count,
        source->vocab_size,
        first_term,
        term_count,
        payload_offsets,
        payload,
        payload_size,
        &source->forward_bound_bytes,
        &source->forward_bound_size
    );
    free(payload);
    if (status != EVOKE_OK || source->forward_bound_size > MaxAllocSize)
    {
        free(source->forward_bound_bytes);
        source->forward_bound_bytes = NULL;
        source->forward_bound_size = 0;
        return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
    }
    artifact_out->first_term = first_term;
    artifact_out->term_count = term_count;
    artifact_out->bytes = source->forward_bound_bytes;
    artifact_out->size = (Size) source->forward_bound_size;
    return EVOKE_OK;
}

typedef struct evoke_am_accelerator_tid_pair
{
    uint64 tid_key;
    uint32 document_slot;
} evoke_am_accelerator_tid_pair;

static int
evoke_am_accelerator_compare_tid_pairs(
    const void *left,
    const void *right
)
{
    const evoke_am_accelerator_tid_pair *a = left;
    const evoke_am_accelerator_tid_pair *b = right;

    if (a->tid_key != b->tid_key)
    {
        return a->tid_key < b->tid_key ? -1 : 1;
    }
    return a->document_slot < b->document_slot
        ? -1
        : a->document_slot > b->document_slot;
}

static evoke_status
evoke_am_accelerator_build_tid_lookup(evoke_am_accelerator_source *source)
{
    evoke_am_accelerator_tid_pair *pairs = NULL;
    uint64 *keys = NULL;
    uint32 *slots = NULL;
    uint32 pair_count = 0;
    evoke_status status = EVOKE_OK;

    if (source == NULL || source->authority_checksum == 0 ||
        source->document_count == 0 ||
        source->visible_document_count == 0 ||
        source->visible_document_count > source->document_count ||
        source->document_tids == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    pairs = palloc_mul_extended(
        (size_t) source->visible_document_count,
        sizeof(*pairs),
        MCXT_ALLOC_HUGE
    );
    for (uint32 document = 0;
         document < source->document_count;
         document++)
    {
        const ItemPointerData *tid = &source->document_tids[document];

        if (!ItemPointerIsValid(tid))
        {
            continue;
        }
        if (pair_count >= source->visible_document_count)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        pairs[pair_count].tid_key =
            ((uint64) ItemPointerGetBlockNumber(tid) << 16) |
            (uint64) ItemPointerGetOffsetNumber(tid);
        pairs[pair_count].document_slot = document;
        pair_count++;
    }
    if (pair_count != source->visible_document_count)
    {
        status = EVOKE_ERR_FORMAT;
        goto cleanup;
    }
    qsort(
        pairs,
        pair_count,
        sizeof(*pairs),
        evoke_am_accelerator_compare_tid_pairs
    );
    keys = palloc_mul_extended(
        (size_t) pair_count,
        sizeof(*keys),
        MCXT_ALLOC_HUGE
    );
    slots = palloc_mul_extended(
        (size_t) pair_count,
        sizeof(*slots),
        MCXT_ALLOC_HUGE
    );
    for (uint32 index = 0; index < pair_count; index++)
    {
        if (index > 0 && pairs[index - 1U].tid_key == pairs[index].tid_key)
        {
            status = EVOKE_ERR_FORMAT;
            goto cleanup;
        }
        keys[index] = pairs[index].tid_key;
        slots[index] = pairs[index].document_slot;
    }
    pfree(pairs);
    pairs = NULL;
    free(source->tid_lookup_bytes);
    source->tid_lookup_bytes = NULL;
    source->tid_lookup_size = 0;
    status = evoke_document_tid_lookup_serialize(
        source->authority_checksum,
        keys,
        slots,
        pair_count,
        source->document_count,
        &source->tid_lookup_bytes,
        &source->tid_lookup_size
    );

cleanup:
    if (slots != NULL)
    {
        pfree(slots);
    }
    if (keys != NULL)
    {
        pfree(keys);
    }
    if (pairs != NULL)
    {
        pfree(pairs);
    }
    return status;
}

bool
evoke_am_prepare_accelerator_baseline(
    Relation index_relation,
    const evoke_segment_read_root *root,
    const evoke_segment_query_context *context,
    evoke_segment_page_reuse_arena *reuse_arena,
    uint64 scope_output_budget,
    volatile bool *reader_fence_locked_out,
    bool *memory_blocked_out,
    uint64 *scope_required_bytes_out,
    evoke_segment_manifest *next_manifest,
    evoke_segment_cow_result *result_out
)
{
    evoke_am_accelerator_source source;
    evoke_segment_cow_write_outcome write_outcome;
    evoke_status status;
    Oid scope_heap_oid = InvalidOid;
    volatile bool scope_heap_locked = false;
    size_t scope_required_size = 0;
    bool prepared = false;

    if (index_relation == NULL || root == NULL || context == NULL ||
        reuse_arena == NULL || reader_fence_locked_out == NULL ||
        memory_blocked_out == NULL || scope_required_bytes_out == NULL ||
        next_manifest == NULL || result_out == NULL ||
        root->root_id != context->root.root_id ||
        !evoke_segment_object_ref_equal(
            &root->manifest,
            &context->root.manifest) ||
        (context->manifest.flags &
         EVOKE_SEGMENT_MANIFEST_FLAG_SAE) == 0)
    {
        return false;
    }
    *reader_fence_locked_out = false;
    *memory_blocked_out = false;
    *scope_required_bytes_out = 0;
    evoke_am_accelerator_source_init(&source);
    PG_TRY();
    {
        pgstat_report_activity(
            STATE_RUNNING,
            "evoke maintenance: inspect semantic accelerator metadata"
        );
        status = evoke_am_accelerator_source_build_query_context(
            index_relation,
            context,
            &source
        );
        if (status != EVOKE_OK)
        {
            ereport(
                ERROR,
                (
                    errmsg(
                        "failed to prepare evoke accelerator baseline"
                    ),
                    errdetail(
                        "Validation failed: %s.",
                        evoke_strerror(status)
                    )
                )
            );
        }
        status = evoke_am_accelerator_build_tid_lookup(&source);
        if (status != EVOKE_OK)
        {
            ereport(
                ERROR,
                (
                    errmsg(
                        "failed to prepare evoke accelerator TID lookup"
                    ),
                    errdetail(
                        "Validation failed: %s.",
                        evoke_strerror(status)
                    )
                )
            );
        }
        if (index_relation->rd_index->indnatts >
            index_relation->rd_index->indnkeyatts)
        {
            scope_heap_oid = index_relation->rd_index->indrelid;
            if (!ConditionalLockRelationOid(
                    scope_heap_oid,
                    AccessShareLock))
            {
                goto accelerator_done;
            }
            scope_heap_locked = true;
        }
        pgstat_report_activity(
            STATE_RUNNING,
            "evoke maintenance: inspect semantic scope metadata"
        );
        status = evoke_scope_build_for_index(
            index_relation,
            source.document_tids,
            source.document_count,
            source.authority_checksum,
            scope_output_budget > SIZE_MAX
                ? SIZE_MAX
                : (size_t) scope_output_budget,
            &scope_required_size,
            &source.scope_bytes,
            &source.scope_size
        );
        *scope_required_bytes_out = scope_required_size;
        if (scope_heap_locked)
        {
            UnlockRelationOid(scope_heap_oid, AccessShareLock);
            scope_heap_locked = false;
        }
        if (status == EVOKE_ERR_NOMEM &&
            scope_required_size > scope_output_budget)
        {
            *memory_blocked_out = true;
            goto accelerator_done;
        }
        if (status != EVOKE_OK)
        {
            ereport(
                ERROR,
                (
                    errmsg(
                        "failed to prepare evoke accelerator scope postings"
                    ),
                    errdetail(
                        "Validation failed: %s.",
                        evoke_strerror(status)
                    )
                )
            );
        }
        InvalidateCatalogSnapshot();
        pgstat_report_activity(
            STATE_RUNNING,
            "evoke maintenance: publish semantic query accelerator"
        );
        write_outcome =
            evoke_segment_pages_write_semantic_accelerator_complete_stream_fork(
                index_relation,
                MAIN_FORKNUM,
                root,
                &context->manifest,
                source.selected_count,
                evoke_am_accelerator_produce_term,
                &source,
                source.forward_chunk_count,
                source.document_shift,
                evoke_am_accelerator_produce_forward,
                &source,
                (source.vocab_size - 1U) /
                    EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD + 1U,
                evoke_am_accelerator_produce_forward_bound,
                &source,
                source.scope_bytes,
                source.scope_size,
                source.tid_lookup_bytes,
                source.tid_lookup_size,
                NULL,
                next_manifest,
                result_out
            );
        if (write_outcome ==
                EVOKE_SEGMENT_COW_WRITE_PREPARED_READER_FENCE_REQUIRED)
        {
            *reader_fence_locked_out =
                evoke_am_acquire_convergent_reader_fence(
                    index_relation,
                    root,
                    &context->manifest,
                    reuse_arena
            );
            if (!*reader_fence_locked_out)
            {
                evoke_segment_pages_abandon_staged_write(
                    index_relation,
                    next_manifest->manifest_id,
                    result_out
                );
                goto accelerator_done;
            }
            write_outcome = EVOKE_SEGMENT_COW_WRITE_WRITTEN;
        }
        if (write_outcome != EVOKE_SEGMENT_COW_WRITE_WRITTEN)
        {
            ereport(
                ERROR,
                (errmsg("evoke accelerator reader-fenced publication failed"))
            );
        }
        prepared = true;

accelerator_done:
        ;
    }
    PG_FINALLY();
    {
        if (scope_heap_locked)
        {
            UnlockRelationOid(scope_heap_oid, AccessShareLock);
        }
        evoke_am_accelerator_source_free(&source);
    }
    PG_END_TRY();
    return prepared;
}
