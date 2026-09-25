#include "evoke_segments.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct evoke_initial_fold_stream
{
    const evoke_index *lexical_index;
    size_t semantic_posting_count;
    size_t semantic_cursor;
    evoke_segment_semantic_posting_reader semantic_reader;
    evoke_segment_semantic_posting_rewind semantic_rewind;
    void *semantic_reader_context;
    evoke_segment_semantic_posting semantic_posting;
    evoke_segment_semantic_posting previous_semantic_posting;
    evoke_term_fold_bundle pending_term;
    uint64_t owner_manifest_id;
    uint64_t coverage_sequence;
    size_t target_bytes;
    uint32_t lexical_term_cursor;
    bool have_semantic_posting;
    bool have_previous_semantic_posting;
    bool have_pending_term;
};

typedef struct evoke_initial_fold_group_builder
{
    evoke_term_fold_bundle bundle;
    size_t run_capacity;
    uint64_t posting_capacity;
} evoke_initial_fold_group_builder;

static evoke_status
evoke_initial_fold_stream_read_semantic(
    evoke_initial_fold_stream *stream
)
{
    evoke_segment_semantic_posting posting;
    evoke_status status;

    if (stream->semantic_cursor >= stream->semantic_posting_count)
    {
        stream->have_semantic_posting = false;
        return EVOKE_OK;
    }
    status = stream->semantic_reader(
        stream->semantic_reader_context,
        &posting
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (posting.term_id >= stream->lexical_index->vocab_size ||
        posting.document_slot >= stream->lexical_index->num_docs ||
        !isfinite(posting.impact) || posting.impact < 0.0f ||
        (stream->have_previous_semantic_posting &&
         (stream->previous_semantic_posting.term_id > posting.term_id ||
          (stream->previous_semantic_posting.term_id == posting.term_id &&
           stream->previous_semantic_posting.document_slot >=
                posting.document_slot))))
    {
        return EVOKE_ERR_FORMAT;
    }
    stream->semantic_posting = posting;
    stream->previous_semantic_posting = posting;
    stream->have_semantic_posting = true;
    stream->have_previous_semantic_posting = true;
    stream->semantic_cursor++;
    return EVOKE_OK;
}

static void
evoke_initial_fold_stream_advance_lexical(
    evoke_initial_fold_stream *stream
)
{
    while (stream->lexical_term_cursor <
            stream->lexical_index->vocab_size &&
           stream->lexical_index->indptr[
               stream->lexical_term_cursor] ==
            stream->lexical_index->indptr[
                stream->lexical_term_cursor + 1])
    {
        stream->lexical_term_cursor++;
    }
}

static evoke_status
evoke_initial_fold_stream_grow_semantic(
    uint32_t **slots,
    evoke_posting_value **values,
    size_t *capacity
)
{
    size_t next_capacity = *capacity == 0 ? 256 : *capacity * 2;
    uint32_t *next_slots;
    evoke_posting_value *next_values;

    if (next_capacity < *capacity ||
        next_capacity > SIZE_MAX / sizeof(**slots) ||
        next_capacity > SIZE_MAX / sizeof(**values))
    {
        return EVOKE_ERR_RANGE;
    }
    next_slots = realloc(*slots, next_capacity * sizeof(**slots));
    if (next_slots == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    *slots = next_slots;
    next_values = realloc(*values, next_capacity * sizeof(**values));
    if (next_values == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    *values = next_values;
    *capacity = next_capacity;
    return EVOKE_OK;
}

static evoke_status
evoke_initial_fold_stream_next_term(
    evoke_initial_fold_stream *stream,
    evoke_term_fold_bundle *term_out,
    bool *done_out
)
{
    const evoke_index *index = stream->lexical_index;
    evoke_posting_value *semantic_values = NULL;
    uint32_t *semantic_slots = NULL;
    size_t semantic_capacity = 0;
    size_t semantic_count = 0;
    uint64_t lexical_start = 0;
    uint64_t lexical_count = 0;
    uint64_t posting_count;
    uint32_t lexical_term;
    uint32_t semantic_term;
    uint32_t term_id;
    uint32_t run_cursor = 0;
    evoke_status status = EVOKE_OK;

    *done_out = false;
    evoke_initial_fold_stream_advance_lexical(stream);
    lexical_term = stream->lexical_term_cursor < index->vocab_size
        ? stream->lexical_term_cursor
        : UINT32_MAX;
    semantic_term = stream->have_semantic_posting
        ? stream->semantic_posting.term_id
        : UINT32_MAX;
    term_id = lexical_term < semantic_term ? lexical_term : semantic_term;
    if (term_id == UINT32_MAX)
    {
        *done_out = true;
        return EVOKE_OK;
    }

    if (lexical_term == term_id)
    {
        lexical_start = index->indptr[term_id];
        lexical_count = index->indptr[term_id + 1] - lexical_start;
        stream->lexical_term_cursor++;
    }
    while (stream->have_semantic_posting &&
           stream->semantic_posting.term_id == term_id)
    {
        if (semantic_count == semantic_capacity)
        {
            status = evoke_initial_fold_stream_grow_semantic(
                &semantic_slots,
                &semantic_values,
                &semantic_capacity
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
        }
        semantic_slots[semantic_count] =
            stream->semantic_posting.document_slot;
        semantic_values[semantic_count].impact =
            stream->semantic_posting.impact;
        semantic_count++;
        status = evoke_initial_fold_stream_read_semantic(stream);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
    }
    if (lexical_count > UINT64_MAX - semantic_count)
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    posting_count = lexical_count + semantic_count;
    if (posting_count == 0 || posting_count > SIZE_MAX ||
        posting_count > SIZE_MAX / sizeof(*term_out->document_slots) ||
        posting_count > SIZE_MAX / sizeof(*term_out->values))
    {
        status = EVOKE_ERR_RANGE;
        goto cleanup;
    }
    term_out->run_count =
        (lexical_count > 0 ? 1 : 0) + (semantic_count > 0 ? 1 : 0);
    term_out->runs = calloc(term_out->run_count, sizeof(*term_out->runs));
    term_out->document_slots = malloc(
        (size_t) posting_count * sizeof(*term_out->document_slots)
    );
    term_out->values = calloc(
        (size_t) posting_count,
        sizeof(*term_out->values)
    );
    if (term_out->runs == NULL || term_out->document_slots == NULL ||
        term_out->values == NULL)
    {
        status = EVOKE_ERR_NOMEM;
        goto cleanup;
    }
    term_out->object_kind = EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    term_out->owner_manifest_id = stream->owner_manifest_id;
    term_out->posting_count = posting_count;

    if (lexical_count > 0)
    {
        evoke_term_fold_run *run = &term_out->runs[run_cursor++];

        run->term_id = term_id;
        run->kind = EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL;
        run->coverage_sequence = stream->coverage_sequence;
        run->posting_count = lexical_count;
        for (uint64_t posting_index = 0;
             posting_index < lexical_count;
             posting_index++)
        {
            uint64_t source_index = lexical_start + posting_index;
            uint32_t document_slot = index->indices[source_index];
            uint32_t term_frequency = index->term_frequencies[source_index];

            if (document_slot >= index->num_docs ||
                term_frequency == 0 ||
                (posting_index > 0 &&
                 term_out->document_slots[posting_index - 1] >=
                    document_slot))
            {
                status = EVOKE_ERR_FORMAT;
                goto cleanup;
            }
            term_out->document_slots[posting_index] = document_slot;
            term_out->values[posting_index].term_frequency = term_frequency;
        }
    }
    if (semantic_count > 0)
    {
        evoke_term_fold_run *run = &term_out->runs[run_cursor++];

        run->term_id = term_id;
        run->kind = EVOKE_POSTING_EXTENT_SEMANTIC_IMPACT;
        run->coverage_sequence = stream->coverage_sequence;
        run->posting_offset = lexical_count;
        run->posting_count = semantic_count;
        memcpy(
            &term_out->document_slots[lexical_count],
            semantic_slots,
            semantic_count * sizeof(*semantic_slots)
        );
        memcpy(
            &term_out->values[lexical_count],
            semantic_values,
            semantic_count * sizeof(*semantic_values)
        );
    }
    status = evoke_term_fold_bundle_validate(term_out);

cleanup:
    free(semantic_values);
    free(semantic_slots);
    if (status != EVOKE_OK)
    {
        evoke_term_fold_bundle_free(term_out);
    }
    return status;
}

static evoke_status
evoke_initial_fold_term_size(
    const evoke_term_fold_bundle *term,
    size_t *size_out
)
{
    size_t size = 0;

    if (term == NULL || size_out == NULL || term->run_count == 0)
    {
        return EVOKE_ERR_INVALID;
    }
    for (uint32_t run_index = 0;
         run_index < term->run_count;
         run_index++)
    {
        const evoke_term_fold_run *run = &term->runs[run_index];
        uint64_t posting_end = run->posting_offset + run->posting_count;
        uint64_t block_count = 0;
        uint32_t prior_block = 0;
        size_t run_size;

        for (uint64_t posting_index = run->posting_offset;
             posting_index < posting_end;
             posting_index++)
        {
            uint32_t block_id = term->document_slots[posting_index] >>
                EVOKE_DEFAULT_POSTING_BLOCK_SHIFT;

            if (posting_index == run->posting_offset ||
                block_id != prior_block)
            {
                block_count++;
                prior_block = block_id;
            }
        }
        if (run->posting_count >
                (SIZE_MAX - EVOKE_TERM_FOLD_RUN_SIZE) /
                    (sizeof(uint32_t) * 2) ||
            block_count >
                (SIZE_MAX - EVOKE_TERM_FOLD_RUN_SIZE -
                 (size_t) run->posting_count *
                    (sizeof(uint32_t) * 2)) /
                    EVOKE_POSTING_BLOCK_RECORD_SIZE)
        {
            return EVOKE_ERR_RANGE;
        }
        run_size = EVOKE_TERM_FOLD_RUN_SIZE +
            (size_t) run->posting_count * (sizeof(uint32_t) * 2) +
            (size_t) block_count * EVOKE_POSTING_BLOCK_RECORD_SIZE;
        if (size > SIZE_MAX - run_size)
        {
            return EVOKE_ERR_RANGE;
        }
        size += run_size;
    }
    *size_out = size;
    return EVOKE_OK;
}

static evoke_status
evoke_initial_fold_group_reserve(
    evoke_initial_fold_group_builder *builder,
    size_t required_runs,
    uint64_t required_postings
)
{
    size_t run_capacity = builder->run_capacity;
    uint64_t posting_capacity = builder->posting_capacity;
    evoke_term_fold_run *runs;
    uint32_t *document_slots;
    evoke_posting_value *values;

    if (required_postings > SIZE_MAX)
    {
        return EVOKE_ERR_RANGE;
    }
    while (run_capacity < required_runs)
    {
        size_t next = run_capacity == 0 ? 16 : run_capacity * 2;

        if (next < run_capacity)
        {
            return EVOKE_ERR_RANGE;
        }
        run_capacity = next;
    }
    while (posting_capacity < required_postings)
    {
        uint64_t next = posting_capacity == 0
            ? 4096
            : posting_capacity * 2;

        if (next < posting_capacity || next > SIZE_MAX)
        {
            posting_capacity = required_postings;
            break;
        }
        posting_capacity = next;
    }
    if (run_capacity > SIZE_MAX / sizeof(*runs) ||
        posting_capacity > SIZE_MAX / sizeof(*document_slots) ||
        posting_capacity > SIZE_MAX / sizeof(*values))
    {
        return EVOKE_ERR_RANGE;
    }
    runs = realloc(
        builder->bundle.runs,
        run_capacity * sizeof(*runs)
    );
    if (runs == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    builder->bundle.runs = runs;
    document_slots = realloc(
        builder->bundle.document_slots,
        (size_t) posting_capacity * sizeof(*document_slots)
    );
    if (document_slots == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    builder->bundle.document_slots = document_slots;
    values = realloc(
        builder->bundle.values,
        (size_t) posting_capacity * sizeof(*values)
    );
    if (values == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    builder->bundle.values = values;
    builder->run_capacity = run_capacity;
    builder->posting_capacity = posting_capacity;
    return EVOKE_OK;
}

static evoke_status
evoke_initial_fold_group_append(
    evoke_initial_fold_group_builder *builder,
    const evoke_term_fold_bundle *term
)
{
    uint32_t old_run_count = builder->bundle.run_count;
    uint64_t old_posting_count = builder->bundle.posting_count;
    evoke_status status;

    if (term->run_count > UINT32_MAX - old_run_count ||
        term->posting_count > UINT64_MAX - old_posting_count)
    {
        return EVOKE_ERR_RANGE;
    }
    status = evoke_initial_fold_group_reserve(
        builder,
        (size_t) old_run_count + term->run_count,
        old_posting_count + term->posting_count
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    memcpy(
        &builder->bundle.document_slots[old_posting_count],
        term->document_slots,
        (size_t) term->posting_count *
            sizeof(*term->document_slots)
    );
    memcpy(
        &builder->bundle.values[old_posting_count],
        term->values,
        (size_t) term->posting_count * sizeof(*term->values)
    );
    for (uint32_t run_index = 0;
         run_index < term->run_count;
         run_index++)
    {
        evoke_term_fold_run run = term->runs[run_index];

        run.posting_offset += old_posting_count;
        builder->bundle.runs[old_run_count + run_index] = run;
    }
    builder->bundle.run_count += term->run_count;
    builder->bundle.posting_count += term->posting_count;
    return EVOKE_OK;
}

evoke_status
evoke_initial_fold_stream_create(
    const evoke_index *lexical_index,
    size_t semantic_posting_count,
    evoke_segment_semantic_posting_reader semantic_reader,
    evoke_segment_semantic_posting_rewind semantic_rewind,
    void *semantic_reader_context,
    uint64_t owner_manifest_id,
    uint64_t coverage_sequence,
    size_t target_bytes,
    evoke_initial_fold_stream **stream_out
)
{
    evoke_initial_fold_stream *stream;
    evoke_status status;

    if (lexical_index == NULL || stream_out == NULL ||
        owner_manifest_id == 0 || coverage_sequence == 0 ||
        target_bytes <= EVOKE_TERM_FOLD_HEADER_SIZE ||
        (lexical_index->vocab_size > 0 &&
         lexical_index->indptr == NULL) ||
        (lexical_index->data_len > 0 &&
         (lexical_index->indices == NULL ||
          lexical_index->term_frequencies == NULL)) ||
        (semantic_posting_count > 0 &&
         (semantic_reader == NULL || semantic_rewind == NULL ||
          semantic_reader_context == NULL)))
    {
        return EVOKE_ERR_INVALID;
    }
    *stream_out = NULL;
    if (lexical_index->vocab_size > 0 &&
        (lexical_index->indptr[0] != 0 ||
         lexical_index->indptr[lexical_index->vocab_size] !=
            lexical_index->data_len))
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t term_id = 0;
         term_id < lexical_index->vocab_size;
         term_id++)
    {
        if (lexical_index->indptr[term_id] >
            lexical_index->indptr[term_id + 1])
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    stream = calloc(1, sizeof(*stream));
    if (stream == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    stream->lexical_index = lexical_index;
    stream->semantic_posting_count = semantic_posting_count;
    stream->semantic_reader = semantic_reader;
    stream->semantic_rewind = semantic_rewind;
    stream->semantic_reader_context = semantic_reader_context;
    stream->owner_manifest_id = owner_manifest_id;
    stream->coverage_sequence = coverage_sequence;
    stream->target_bytes = target_bytes;
    evoke_term_fold_bundle_init(&stream->pending_term);
    if (semantic_posting_count > 0)
    {
        status = semantic_rewind(semantic_reader_context);
        if (status == EVOKE_OK)
        {
            status = evoke_initial_fold_stream_read_semantic(stream);
        }
        if (status != EVOKE_OK)
        {
            evoke_initial_fold_stream_free(stream);
            return status;
        }
    }
    *stream_out = stream;
    return EVOKE_OK;
}

evoke_status
evoke_initial_fold_stream_next(
    evoke_initial_fold_stream *stream,
    evoke_term_fold_bundle *bundle_out,
    bool *done_out
)
{
    evoke_initial_fold_group_builder builder;
    size_t group_size = EVOKE_TERM_FOLD_HEADER_SIZE;
    evoke_status status = EVOKE_OK;

    if (stream == NULL || bundle_out == NULL || done_out == NULL ||
        bundle_out->runs != NULL || bundle_out->blocks != NULL ||
        bundle_out->document_slots != NULL || bundle_out->values != NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&builder, 0, sizeof(builder));
    evoke_term_fold_bundle_init(&builder.bundle);
    builder.bundle.object_kind = EVOKE_SEGMENT_OBJECT_NEUTRAL_FOLD;
    builder.bundle.owner_manifest_id = stream->owner_manifest_id;
    *done_out = false;

    while (true)
    {
        evoke_term_fold_bundle term;
        size_t term_size;
        bool source_done = false;

        evoke_term_fold_bundle_init(&term);
        if (stream->have_pending_term)
        {
            term = stream->pending_term;
            evoke_term_fold_bundle_init(&stream->pending_term);
            stream->have_pending_term = false;
        }
        else
        {
            status = evoke_initial_fold_stream_next_term(
                stream,
                &term,
                &source_done
            );
            if (status != EVOKE_OK)
            {
                goto cleanup;
            }
            if (source_done)
            {
                if (builder.bundle.run_count == 0)
                {
                    *done_out = true;
                }
                break;
            }
        }
        status = evoke_initial_fold_term_size(&term, &term_size);
        if (status != EVOKE_OK)
        {
            evoke_term_fold_bundle_free(&term);
            goto cleanup;
        }
        if (builder.bundle.run_count > 0 &&
            (group_size >= stream->target_bytes ||
             term_size > stream->target_bytes - group_size))
        {
            stream->pending_term = term;
            stream->have_pending_term = true;
            break;
        }
        status = evoke_initial_fold_group_append(&builder, &term);
        evoke_term_fold_bundle_free(&term);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        if (group_size > SIZE_MAX - term_size)
        {
            status = EVOKE_ERR_RANGE;
            goto cleanup;
        }
        group_size += term_size;
        if (group_size >= stream->target_bytes)
        {
            break;
        }
    }
    if (!*done_out)
    {
        status = evoke_term_fold_bundle_validate(&builder.bundle);
        if (status != EVOKE_OK)
        {
            goto cleanup;
        }
        *bundle_out = builder.bundle;
        evoke_term_fold_bundle_init(&builder.bundle);
    }

cleanup:
    evoke_term_fold_bundle_free(&builder.bundle);
    return status;
}

void
evoke_initial_fold_stream_free(evoke_initial_fold_stream *stream)
{
    if (stream == NULL)
    {
        return;
    }
    evoke_term_fold_bundle_free(&stream->pending_term);
    free(stream);
}
