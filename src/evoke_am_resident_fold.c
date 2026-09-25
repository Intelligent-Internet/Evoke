#include "postgres.h"

#include <math.h>

#include "utils/memutils.h"

#include "evoke_am_resident_fold.h"

#define EVOKE_AM_RESIDENT_FOLD_MAGIC UINT64_C(0x49343252464f4c44)
#define EVOKE_AM_RESIDENT_FOLD_VERSION 2
#define EVOKE_AM_RESIDENT_FOLD_HAS_VOCAB UINT32_C(1)
#define EVOKE_AM_RESIDENT_FOLD_TIE_ORDER_IDENTITY UINT32_C(2)
#define EVOKE_AM_RESIDENT_FOLD_HAS_EMPTY_TOKEN UINT32_C(4)

typedef struct evoke_am_resident_fold_header
{
    uint64 magic;
    uint32 version;
    uint32 flags;
    uint64 total_size;
    uint64 reserved;
    evoke_segment_read_root root;
    evoke_params params;
    uint64 corpus_document_count;
    uint64 total_document_length;
    uint32 num_docs;
    uint32 vocab_size;
    uint32 extent_count;
    uint32 document_block_count;
    uint32 retired_document_count;
    uint32 tie_break_order_count;
    uint32 empty_token_id;
    uint32 block_shift;
    uint8 contract_hash[EVOKE_SEGMENT_CONTRACT_HASH_BYTES];
    uint64 terms_offset;
    uint64 extents_offset;
    uint64 doc_lengths_offset;
    uint64 doc_frequencies_offset;
    uint64 document_tids_offset;
    uint64 document_block_extrema_offset;
    uint64 retired_document_ids_offset;
    uint64 tie_break_keys_offset;
    uint64 tie_break_order_offset;
    uint64 vocab_offsets_offset;
    uint64 sorted_vocab_ids_offset;
    uint64 vocab_bytes_offset;
    uint64 vocab_bytes_size;
} evoke_am_resident_fold_header;

typedef struct evoke_am_resident_fold_term
{
    uint32 first_extent;
    uint32 extent_count;
} evoke_am_resident_fold_term;

typedef struct evoke_am_resident_fold_extent
{
    uint64 data_offset;
    uint64 indices_offset;
    uint64 term_frequencies_offset;
    uint64 values_offset;
    uint64 document_id_map_offset;
    uint64 blocks_offset;
    uint64 posting_count;
    uint32 document_id_base;
    uint32 local_document_count;
    uint32 block_count;
    uint32 block_shift;
    uint32 kind;
    uint32 reserved;
} evoke_am_resident_fold_extent;

typedef struct evoke_am_resident_fold_vocab_item
{
    const char *token;
    uint32 term_id;
} evoke_am_resident_fold_vocab_item;

typedef struct evoke_am_resident_fold_query
{
    evoke_index index;
    evoke_corpus_stats stats;
    evoke_term_extent_list *terms;
    evoke_posting_extent *extents;
    uint32 *doc_frequencies;
    uint32 *mapped_query_ids;
    uint32 *original_term_ids;
    size_t unique_term_count;
    size_t extent_count;
} evoke_am_resident_fold_query;

static bool
evoke_am_resident_fold_checked_add(
    Size left,
    Size right,
    Size *result_out
)
{
    if (left > SIZE_MAX - right)
    {
        return false;
    }
    *result_out = left + right;
    return true;
}

static bool
evoke_am_resident_fold_checked_mul(
    Size left,
    Size right,
    Size *result_out
)
{
    if (left == 0 || right == 0)
    {
        *result_out = 0;
        return true;
    }
    if (left > SIZE_MAX / right)
    {
        return false;
    }
    *result_out = left * right;
    return true;
}

static bool
evoke_am_resident_fold_append_region(
    Size *cursor,
    Size count,
    Size width,
    uint64 *offset_out
)
{
    Size bytes;
    Size next;

    if (cursor == NULL || offset_out == NULL ||
        !evoke_am_resident_fold_checked_mul(count, width, &bytes))
    {
        return false;
    }
    *cursor = MAXALIGN(*cursor);
    *offset_out = bytes == 0 ? 0 : (uint64) *cursor;
    if (!evoke_am_resident_fold_checked_add(*cursor, bytes, &next))
    {
        return false;
    }
    *cursor = next;
    return true;
}

static bool
evoke_am_resident_fold_region_valid(
    const evoke_am_resident_fold_header *header,
    uint64 offset,
    uint64 count,
    Size width
)
{
    uint64 bytes;

    if (count == 0)
    {
        return offset == 0;
    }
    if (offset < MAXALIGN(sizeof(*header)) ||
        offset > header->total_size || count > UINT64_MAX / width)
    {
        return false;
    }
    bytes = count * width;
    return bytes <= header->total_size - offset;
}

static bool
evoke_am_resident_fold_header_valid(
    const void *block,
    Size mapped_size
)
{
    const evoke_am_resident_fold_header *header = block;

    if (block == NULL || mapped_size < sizeof(*header) ||
        header->magic != EVOKE_AM_RESIDENT_FOLD_MAGIC ||
        header->version != EVOKE_AM_RESIDENT_FOLD_VERSION ||
        header->total_size != mapped_size ||
        header->reserved != 0 ||
        header->corpus_document_count > header->num_docs ||
        (((header->flags & EVOKE_AM_RESIDENT_FOLD_HAS_EMPTY_TOKEN) != 0) &&
         header->empty_token_id >= header->vocab_size) ||
        (((header->flags & EVOKE_AM_RESIDENT_FOLD_HAS_EMPTY_TOKEN) == 0) &&
         header->empty_token_id != header->vocab_size) ||
        header->block_shift == 0 || header->block_shift >= 32 ||
        header->root.active_l0.record_count != 0 ||
        header->root.pending_l0.record_count != 0 ||
        !evoke_am_resident_fold_region_valid(
            header,
            header->terms_offset,
            header->vocab_size,
            sizeof(evoke_am_resident_fold_term)) ||
        !evoke_am_resident_fold_region_valid(
            header,
            header->extents_offset,
            header->extent_count,
            sizeof(evoke_am_resident_fold_extent)) ||
        !evoke_am_resident_fold_region_valid(
            header,
            header->doc_lengths_offset,
            header->num_docs,
            sizeof(uint32)) ||
        !evoke_am_resident_fold_region_valid(
            header,
            header->doc_frequencies_offset,
            header->vocab_size,
            sizeof(uint32)) ||
        !evoke_am_resident_fold_region_valid(
            header,
            header->document_tids_offset,
            header->num_docs,
            sizeof(ItemPointerData)) ||
        !evoke_am_resident_fold_region_valid(
            header,
            header->document_block_extrema_offset,
            header->document_block_count,
            sizeof(evoke_document_cow_length_extrema)) ||
        !evoke_am_resident_fold_region_valid(
            header,
            header->retired_document_ids_offset,
            header->retired_document_count,
            sizeof(uint32)) ||
        !evoke_am_resident_fold_region_valid(
            header,
            header->tie_break_keys_offset,
            header->num_docs,
            sizeof(uint64)) ||
        !evoke_am_resident_fold_region_valid(
            header,
            header->tie_break_order_offset,
            header->tie_break_order_count,
            sizeof(uint32)))
    {
        return false;
    }
    if ((header->flags & EVOKE_AM_RESIDENT_FOLD_HAS_VOCAB) != 0)
    {
        if (!evoke_am_resident_fold_region_valid(
                header,
                header->vocab_offsets_offset,
                header->vocab_size,
                sizeof(uint64)) ||
            !evoke_am_resident_fold_region_valid(
                header,
                header->sorted_vocab_ids_offset,
                header->vocab_size,
                sizeof(uint32)) ||
            !evoke_am_resident_fold_region_valid(
                header,
                header->vocab_bytes_offset,
                header->vocab_bytes_size,
                sizeof(uint8)))
        {
            return false;
        }
    }
    else if (header->vocab_offsets_offset != 0 ||
             header->sorted_vocab_ids_offset != 0 ||
             header->vocab_bytes_offset != 0 ||
             header->vocab_bytes_size != 0)
    {
        return false;
    }
    return true;
}

static bool
evoke_am_resident_fold_block_valid(const void *block, Size mapped_size)
{
    const evoke_am_resident_fold_header *header = block;
    const evoke_am_resident_fold_term *terms;
    const evoke_am_resident_fold_extent *extents;
    const uint64 *vocab_offsets;
    const uint32 *sorted_ids;
    uint64 extent_cursor = 0;

    if (!evoke_am_resident_fold_header_valid(block, mapped_size))
    {
        return false;
    }
    terms = (const void *) ((const char *) block + header->terms_offset);
    extents = (const void *) (
        (const char *) block + header->extents_offset
    );
    for (uint32 term_id = 0; term_id < header->vocab_size; term_id++)
    {
        if (terms[term_id].first_extent != extent_cursor ||
            terms[term_id].extent_count >
                header->extent_count - extent_cursor)
        {
            return false;
        }
        extent_cursor += terms[term_id].extent_count;
    }
    if (extent_cursor != header->extent_count)
    {
        return false;
    }
    for (uint32 extent_index = 0;
         extent_index < header->extent_count;
         extent_index++)
    {
        const evoke_am_resident_fold_extent *extent =
            &extents[extent_index];

        if (extent->reserved != 0 ||
            extent->kind < EVOKE_POSTING_EXTENT_LEXICAL_NEUTRAL ||
            extent->kind > EVOKE_POSTING_EXTENT_LEXICAL_IMPACT ||
            !evoke_am_resident_fold_region_valid(
                header,
                extent->indices_offset,
                extent->posting_count,
                sizeof(uint32)) ||
            (extent->data_offset != 0 &&
             !evoke_am_resident_fold_region_valid(
                 header,
                 extent->data_offset,
                 extent->posting_count,
                 sizeof(float))) ||
            (extent->term_frequencies_offset != 0 &&
             !evoke_am_resident_fold_region_valid(
                 header,
                 extent->term_frequencies_offset,
                 extent->posting_count,
                 sizeof(uint32))) ||
            (extent->values_offset != 0 &&
             !evoke_am_resident_fold_region_valid(
                 header,
                 extent->values_offset,
                 extent->posting_count,
                 sizeof(evoke_posting_value))) ||
            (extent->document_id_map_offset != 0 &&
             !evoke_am_resident_fold_region_valid(
                 header,
                 extent->document_id_map_offset,
                 extent->local_document_count,
                 sizeof(uint32))) ||
            !evoke_am_resident_fold_region_valid(
                header,
                extent->blocks_offset,
                extent->block_count,
                sizeof(evoke_posting_block_record)))
        {
            return false;
        }
    }
    if ((header->flags & EVOKE_AM_RESIDENT_FOLD_HAS_VOCAB) == 0)
    {
        return true;
    }
    vocab_offsets = (const void *) (
        (const char *) block + header->vocab_offsets_offset
    );
    sorted_ids = (const void *) (
        (const char *) block + header->sorted_vocab_ids_offset
    );
    for (uint32 index = 0; index < header->vocab_size; index++)
    {
        const char *token;
        Size remaining;

        if (vocab_offsets[index] < header->vocab_bytes_offset ||
            vocab_offsets[index] >=
                header->vocab_bytes_offset + header->vocab_bytes_size ||
            sorted_ids[index] >= header->vocab_size)
        {
            return false;
        }
        token = (const char *) block + vocab_offsets[index];
        remaining = (Size) (header->vocab_bytes_offset +
            header->vocab_bytes_size - vocab_offsets[index]);
        if (memchr(token, '\0', remaining) == NULL)
        {
            return false;
        }
        if (index > 0)
        {
            const char *prior = (const char *) block +
                vocab_offsets[sorted_ids[index - 1]];
            const char *current = (const char *) block +
                vocab_offsets[sorted_ids[index]];

            if (strcmp(prior, current) >= 0)
            {
                return false;
            }
        }
    }
    return true;
}

static int
evoke_am_resident_fold_compare_vocab(const void *left, const void *right)
{
    const evoke_am_resident_fold_vocab_item *left_item = left;
    const evoke_am_resident_fold_vocab_item *right_item = right;

    return strcmp(left_item->token, right_item->token);
}

static bool
evoke_am_resident_fold_layout(
    const evoke_segment_storage_snapshot *snapshot,
    evoke_am_resident_fold_header *header_out
)
{
    evoke_am_resident_fold_header header;
    Size cursor = MAXALIGN(sizeof(header));
    bool has_vocab;

    if (snapshot == NULL || header_out == NULL ||
        snapshot->root.active_l0.record_count != 0 ||
        snapshot->root.pending_l0.record_count != 0 ||
        snapshot->index_metadata.num_docs !=
            snapshot->manifest.document_slot_count ||
        snapshot->index_metadata.vocab_size !=
            snapshot->read_view.vocab_size ||
        snapshot->read_view.extent_count > UINT32_MAX ||
        snapshot->document_block_count > UINT32_MAX ||
        snapshot->retired_document_count > UINT32_MAX ||
        snapshot->document_tie_break_order_count > UINT32_MAX)
    {
        return false;
    }
    memset(&header, 0, sizeof(header));
    header.magic = EVOKE_AM_RESIDENT_FOLD_MAGIC;
    header.version = EVOKE_AM_RESIDENT_FOLD_VERSION;
    header.root = snapshot->root;
    header.params = snapshot->index_metadata.params;
    header.corpus_document_count = snapshot->corpus_stats.document_count;
    header.total_document_length =
        snapshot->corpus_stats.total_document_length;
    header.num_docs = snapshot->index_metadata.num_docs;
    header.vocab_size = snapshot->index_metadata.vocab_size;
    header.extent_count = snapshot->read_view.extent_count;
    header.document_block_count = (uint32) snapshot->document_block_count;
    header.retired_document_count = snapshot->retired_document_count;
    header.tie_break_order_count =
        snapshot->document_tie_break_order_count;
    header.empty_token_id = snapshot->index_metadata.has_empty_token
        ? snapshot->index_metadata.empty_token_id
        : snapshot->index_metadata.vocab_size;
    if (snapshot->index_metadata.has_empty_token)
    {
        if (snapshot->index_metadata.empty_token_id >= header.vocab_size)
        {
            return false;
        }
        header.flags |= EVOKE_AM_RESIDENT_FOLD_HAS_EMPTY_TOKEN;
    }
    header.block_shift = snapshot->query_contract.block_shift;
    memcpy(
        header.contract_hash,
        snapshot->manifest.contract_hash,
        sizeof(header.contract_hash)
    );
    if (snapshot->document_tie_break_order_is_identity)
    {
        header.flags |= EVOKE_AM_RESIDENT_FOLD_TIE_ORDER_IDENTITY;
    }
    has_vocab = header.vocab_size > 0 &&
        snapshot->index_metadata.vocab != NULL;
    if (has_vocab)
    {
        header.flags |= EVOKE_AM_RESIDENT_FOLD_HAS_VOCAB;
    }
    if (!evoke_am_resident_fold_append_region(
            &cursor,
            header.vocab_size,
            sizeof(evoke_am_resident_fold_term),
            &header.terms_offset) ||
        !evoke_am_resident_fold_append_region(
            &cursor,
            header.extent_count,
            sizeof(evoke_am_resident_fold_extent),
            &header.extents_offset) ||
        !evoke_am_resident_fold_append_region(
            &cursor,
            header.num_docs,
            sizeof(uint32),
            &header.doc_lengths_offset) ||
        !evoke_am_resident_fold_append_region(
            &cursor,
            header.vocab_size,
            sizeof(uint32),
            &header.doc_frequencies_offset) ||
        !evoke_am_resident_fold_append_region(
            &cursor,
            header.num_docs,
            sizeof(ItemPointerData),
            &header.document_tids_offset) ||
        !evoke_am_resident_fold_append_region(
            &cursor,
            header.document_block_count,
            sizeof(evoke_document_cow_length_extrema),
            &header.document_block_extrema_offset) ||
        !evoke_am_resident_fold_append_region(
            &cursor,
            header.retired_document_count,
            sizeof(uint32),
            &header.retired_document_ids_offset) ||
        !evoke_am_resident_fold_append_region(
            &cursor,
            header.num_docs,
            sizeof(uint64),
            &header.tie_break_keys_offset) ||
        !evoke_am_resident_fold_append_region(
            &cursor,
            header.tie_break_order_count,
            sizeof(uint32),
            &header.tie_break_order_offset))
    {
        return false;
    }
    if (has_vocab)
    {
        Size vocab_bytes = 0;

        for (uint32 term_id = 0; term_id < header.vocab_size; term_id++)
        {
            Size token_bytes;

            if (snapshot->index_metadata.vocab[term_id] == NULL)
            {
                return false;
            }
            token_bytes = strlen(snapshot->index_metadata.vocab[term_id]) + 1;
            if (!evoke_am_resident_fold_checked_add(
                    vocab_bytes,
                    token_bytes,
                    &vocab_bytes))
            {
                return false;
            }
        }
        header.vocab_bytes_size = vocab_bytes;
        if (!evoke_am_resident_fold_append_region(
                &cursor,
                header.vocab_size,
                sizeof(uint64),
                &header.vocab_offsets_offset) ||
            !evoke_am_resident_fold_append_region(
                &cursor,
                header.vocab_size,
                sizeof(uint32),
                &header.sorted_vocab_ids_offset) ||
            !evoke_am_resident_fold_append_region(
                &cursor,
                vocab_bytes,
                sizeof(uint8),
                &header.vocab_bytes_offset))
        {
            return false;
        }
    }
    for (uint32 extent_index = 0;
         extent_index < header.extent_count;
         extent_index++)
    {
        const evoke_posting_extent *extent =
            &snapshot->read_view.extents[extent_index];
        uint64 ignored;

        if (!evoke_am_resident_fold_append_region(
                &cursor,
                extent->len,
                sizeof(uint32),
                &ignored) ||
            (extent->data != NULL &&
             !evoke_am_resident_fold_append_region(
                 &cursor,
                 extent->len,
                 sizeof(float),
                 &ignored)) ||
            (extent->term_frequencies != NULL &&
             !evoke_am_resident_fold_append_region(
                 &cursor,
                 extent->len,
                 sizeof(uint32),
                 &ignored)) ||
            (extent->values != NULL &&
             !evoke_am_resident_fold_append_region(
                 &cursor,
                 extent->len,
                 sizeof(evoke_posting_value),
                 &ignored)) ||
            (extent->document_id_map != NULL &&
             !evoke_am_resident_fold_append_region(
                 &cursor,
                 extent->local_document_count,
                 sizeof(uint32),
                 &ignored)) ||
            !evoke_am_resident_fold_append_region(
                &cursor,
                extent->block_count,
                sizeof(evoke_posting_block_record),
                &ignored))
        {
            return false;
        }
    }
    header.total_size = MAXALIGN(cursor);
    *header_out = header;
    return true;
}

Size
evoke_am_resident_fold_required_size(
    const evoke_segment_storage_snapshot *snapshot
)
{
    evoke_am_resident_fold_header header;

    return evoke_am_resident_fold_layout(snapshot, &header)
        ? (Size) header.total_size
        : 0;
}

static bool
evoke_am_resident_fold_copy_region(
    void *block,
    Size total_size,
    Size *cursor,
    const void *source,
    Size count,
    Size width,
    uint64 *offset_out
)
{
    Size bytes;

    if (!evoke_am_resident_fold_append_region(
            cursor,
            count,
            width,
            offset_out) ||
        !evoke_am_resident_fold_checked_mul(count, width, &bytes) ||
        *cursor > total_size)
    {
        return false;
    }
    if (bytes > 0)
    {
        if (source == NULL)
        {
            return false;
        }
        memcpy((char *) block + *offset_out, source, bytes);
    }
    return true;
}

static bool
evoke_am_resident_fold_fill(
    void *block,
    Size block_size,
    const evoke_segment_storage_snapshot *snapshot
)
{
    evoke_am_resident_fold_header layout;
    evoke_am_resident_fold_header *header = block;
    evoke_am_resident_fold_term *terms;
    evoke_am_resident_fold_extent *extents;
    Size cursor;
    uint64 ignored;

    if (block == NULL ||
        !evoke_am_resident_fold_layout(snapshot, &layout) ||
        layout.total_size != block_size)
    {
        return false;
    }
    *header = layout;
    cursor = (Size) header->terms_offset;
    terms = (void *) ((char *) block + header->terms_offset);
    cursor += (Size) header->vocab_size * sizeof(*terms);
    extents = (void *) ((char *) block + header->extents_offset);
    cursor = (Size) header->extents_offset +
        (Size) header->extent_count * sizeof(*extents);
    if (!evoke_am_resident_fold_copy_region(
            block,
            block_size,
            &cursor,
            snapshot->index_metadata.doc_lengths,
            header->num_docs,
            sizeof(uint32),
            &ignored) ||
        ignored != header->doc_lengths_offset ||
        !evoke_am_resident_fold_copy_region(
            block,
            block_size,
            &cursor,
            snapshot->corpus_stats.doc_frequencies,
            header->vocab_size,
            sizeof(uint32),
            &ignored) ||
        ignored != header->doc_frequencies_offset ||
        !evoke_am_resident_fold_copy_region(
            block,
            block_size,
            &cursor,
            snapshot->doc_tids,
            header->num_docs,
            sizeof(ItemPointerData),
            &ignored) ||
        ignored != header->document_tids_offset ||
        !evoke_am_resident_fold_copy_region(
            block,
            block_size,
            &cursor,
            snapshot->document_block_extrema,
            header->document_block_count,
            sizeof(evoke_document_cow_length_extrema),
            &ignored) ||
        ignored != header->document_block_extrema_offset ||
        !evoke_am_resident_fold_copy_region(
            block,
            block_size,
            &cursor,
            snapshot->retired_document_ids,
            header->retired_document_count,
            sizeof(uint32),
            &ignored) ||
        ignored != header->retired_document_ids_offset ||
        !evoke_am_resident_fold_copy_region(
            block,
            block_size,
            &cursor,
            snapshot->document_tie_break_keys,
            header->num_docs,
            sizeof(uint64),
            &ignored) ||
        ignored != header->tie_break_keys_offset ||
        !evoke_am_resident_fold_copy_region(
            block,
            block_size,
            &cursor,
            snapshot->document_tie_break_order,
            header->tie_break_order_count,
            sizeof(uint32),
            &ignored) ||
        ignored != header->tie_break_order_offset)
    {
        return false;
    }
    if ((header->flags & EVOKE_AM_RESIDENT_FOLD_HAS_VOCAB) != 0)
    {
        uint64 *vocab_offsets;
        uint32 *sorted_ids;
        char *vocab_bytes;
        Size vocab_cursor = 0;
        evoke_am_resident_fold_vocab_item *items;

        if (!evoke_am_resident_fold_append_region(
                &cursor,
                header->vocab_size,
                sizeof(uint64),
                &ignored) ||
            ignored != header->vocab_offsets_offset)
        {
            return false;
        }
        vocab_offsets = (void *) ((char *) block + ignored);
        if (!evoke_am_resident_fold_append_region(
                &cursor,
                header->vocab_size,
                sizeof(uint32),
                &ignored) ||
            ignored != header->sorted_vocab_ids_offset)
        {
            return false;
        }
        sorted_ids = (void *) ((char *) block + ignored);
        if (!evoke_am_resident_fold_append_region(
                &cursor,
                header->vocab_bytes_size,
                sizeof(uint8),
                &ignored) ||
            ignored != header->vocab_bytes_offset)
        {
            return false;
        }
        vocab_bytes = (char *) block + ignored;
        items = palloc(
            sizeof(*items) * (Size) header->vocab_size
        );
        for (uint32 term_id = 0; term_id < header->vocab_size; term_id++)
        {
            const char *token = snapshot->index_metadata.vocab[term_id];
            Size token_size = strlen(token) + 1;

            vocab_offsets[term_id] =
                header->vocab_bytes_offset + vocab_cursor;
            memcpy(vocab_bytes + vocab_cursor, token, token_size);
            vocab_cursor += token_size;
            items[term_id].token = token;
            items[term_id].term_id = term_id;
        }
        if (vocab_cursor != header->vocab_bytes_size)
        {
            pfree(items);
            return false;
        }
        qsort(
            items,
            header->vocab_size,
            sizeof(*items),
            evoke_am_resident_fold_compare_vocab
        );
        for (uint32 index = 0; index < header->vocab_size; index++)
        {
            sorted_ids[index] = items[index].term_id;
        }
        pfree(items);
    }
    for (uint32 term_id = 0; term_id < header->vocab_size; term_id++)
    {
        const evoke_term_extent_list *source_term =
            &snapshot->read_view.terms[term_id];

        if (source_term->len > UINT32_MAX ||
            (source_term->len > 0 &&
             (source_term->extents < snapshot->read_view.extents ||
              source_term->extents + source_term->len >
                snapshot->read_view.extents +
                    snapshot->read_view.extent_count)))
        {
            return false;
        }
        terms[term_id].first_extent = source_term->len == 0
            ? term_id == 0
                ? 0
                : terms[term_id - 1].first_extent +
                    terms[term_id - 1].extent_count
            : (uint32) (
                source_term->extents - snapshot->read_view.extents
            );
        terms[term_id].extent_count = (uint32) source_term->len;
    }
    for (uint32 extent_index = 0;
         extent_index < header->extent_count;
         extent_index++)
    {
        const evoke_posting_extent *source =
            &snapshot->read_view.extents[extent_index];
        evoke_am_resident_fold_extent *target = &extents[extent_index];

        memset(target, 0, sizeof(*target));
        target->posting_count = source->len;
        target->document_id_base = source->document_id_base;
        target->local_document_count = source->local_document_count;
        target->block_count = source->block_count;
        target->block_shift = source->block_shift;
        target->kind = source->kind;
        if (!evoke_am_resident_fold_copy_region(
                block,
                block_size,
                &cursor,
                source->indices,
                source->len,
                sizeof(uint32),
                &target->indices_offset) ||
            (source->data != NULL &&
             !evoke_am_resident_fold_copy_region(
                 block,
                 block_size,
                 &cursor,
                 source->data,
                 source->len,
                 sizeof(float),
                 &target->data_offset)) ||
            (source->term_frequencies != NULL &&
             !evoke_am_resident_fold_copy_region(
                 block,
                 block_size,
                 &cursor,
                 source->term_frequencies,
                 source->len,
                 sizeof(uint32),
                 &target->term_frequencies_offset)) ||
            (source->values != NULL &&
             !evoke_am_resident_fold_copy_region(
                 block,
                 block_size,
                 &cursor,
                 source->values,
                 source->len,
                 sizeof(evoke_posting_value),
                 &target->values_offset)) ||
            (source->document_id_map != NULL &&
             !evoke_am_resident_fold_copy_region(
                 block,
                 block_size,
                 &cursor,
                 source->document_id_map,
                 source->local_document_count,
                 sizeof(uint32),
                 &target->document_id_map_offset)) ||
            !evoke_am_resident_fold_copy_region(
                block,
                block_size,
                &cursor,
                source->blocks,
                source->block_count,
                sizeof(evoke_posting_block_record),
                &target->blocks_offset))
        {
            return false;
        }
    }
    if (MAXALIGN(cursor) != block_size)
    {
        return false;
    }
    return evoke_am_resident_fold_block_valid(block, block_size);
}

bool
evoke_am_resident_fold_materialization_fits(
    uint64 relation_bytes,
    uint64 arena_bytes,
    uint64 physical_bytes,
    uint64 shared_buffer_bytes
)
{
    const uint64 minimum_headroom = UINT64_C(8) * 1024 * 1024 * 1024;
    uint64 safety_headroom;
    uint64 fixed_bytes;
    uint64 scratch_bytes;

    if (relation_bytes == 0 || physical_bytes == 0)
    {
        return false;
    }
    safety_headroom = Max(physical_bytes / 8, minimum_headroom);
    if (arena_bytes > physical_bytes ||
        shared_buffer_bytes > physical_bytes - arena_bytes)
    {
        return false;
    }
    fixed_bytes = arena_bytes + shared_buffer_bytes;
    if (safety_headroom > physical_bytes - fixed_bytes)
    {
        return false;
    }
    scratch_bytes = physical_bytes - fixed_bytes - safety_headroom;

    /*
     * The publication target is already charged to the fixed shared arena.
     * Keep one relation-sized private allowance for the decoded source while
     * the global reservation performs the exact fold-size admission check.
     */
    return relation_bytes <= scratch_bytes;
}

evoke_am_resident_fold_publish_result
evoke_am_resident_fold_publish(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    const evoke_segment_storage_snapshot *snapshot,
    Size *published_size_out
)
{
    evoke_am_preload_reservation reservation = {0};
    evoke_am_preload_reserve_result reserve_result;
    Size required_size;
    bool filled = false;
    bool published = false;

    if (published_size_out != NULL)
    {
        *published_size_out = 0;
    }
    required_size = evoke_am_resident_fold_required_size(snapshot);
    if (index_relation == NULL || meta == NULL || required_size == 0)
    {
        return EVOKE_AM_RESIDENT_FOLD_PUBLISH_FAILED;
    }
    reserve_result = evoke_am_preload_reserve(
        index_relation,
        meta,
        EVOKE_AM_PRELOAD_RESIDENT_FOLD,
        required_size,
        &reservation
    );
    if (reserve_result == EVOKE_AM_PRELOAD_RESERVE_READY)
    {
        if (published_size_out != NULL)
        {
            *published_size_out = required_size;
        }
        return EVOKE_AM_RESIDENT_FOLD_PUBLISH_READY;
    }
    if (reserve_result == EVOKE_AM_PRELOAD_RESERVE_LOADING)
    {
        return EVOKE_AM_RESIDENT_FOLD_PUBLISH_LOADING;
    }
    if (reserve_result == EVOKE_AM_PRELOAD_RESERVE_FAILED)
    {
        return EVOKE_AM_RESIDENT_FOLD_PUBLISH_OVERSIZED;
    }
    PG_TRY();
    {
        filled = evoke_am_resident_fold_fill(
            evoke_am_preload_reservation_payload(&reservation),
            required_size,
            snapshot
        );
        if (filled)
        {
            published = evoke_am_preload_reservation_commit(&reservation);
        }
        else
        {
            evoke_am_preload_reservation_abort(&reservation);
        }
    }
    PG_CATCH();
    {
        evoke_am_preload_reservation_abort(&reservation);
        PG_RE_THROW();
    }
    PG_END_TRY();
    if (!filled)
    {
        return EVOKE_AM_RESIDENT_FOLD_PUBLISH_FAILED;
    }
    if (!published)
    {
        return EVOKE_AM_RESIDENT_FOLD_PUBLISH_FAILED;
    }
    if (published_size_out != NULL)
    {
        *published_size_out = required_size;
    }
    return EVOKE_AM_RESIDENT_FOLD_PUBLISH_DONE;
}

void
evoke_am_resident_fold_view_init(evoke_am_resident_fold_view *view)
{
    if (view != NULL)
    {
        memset(view, 0, sizeof(*view));
    }
}

void
evoke_am_resident_fold_view_release(evoke_am_resident_fold_view *view)
{
    if (view == NULL)
    {
        return;
    }
    evoke_am_preload_lease_release(&view->private_lease);
    memset(view, 0, sizeof(*view));
}

bool
evoke_am_resident_fold_attach(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_resident_fold_view *view
)
{
    const evoke_am_resident_fold_header *header;
    const void *block;
    Size mapped_size;

    if (view == NULL)
    {
        return false;
    }
    evoke_am_resident_fold_view_init(view);
    if (index_relation == NULL || meta == NULL ||
        !evoke_am_preload_attach(
            index_relation,
            meta,
            EVOKE_AM_PRELOAD_RESIDENT_FOLD,
            &view->private_lease))
    {
        return false;
    }
    block = evoke_am_preload_lease_payload(&view->private_lease);
    mapped_size = evoke_am_preload_lease_payload_size(&view->private_lease);
    if (!evoke_am_resident_fold_header_valid(block, mapped_size))
    {
        evoke_am_resident_fold_view_release(view);
        return false;
    }
    header = block;
    view->private_header = header;
    view->private_terms = (const char *) block + header->terms_offset;
    view->private_extents = (const char *) block + header->extents_offset;
    if ((header->flags & EVOKE_AM_RESIDENT_FOLD_HAS_VOCAB) != 0)
    {
        view->private_vocab_offsets =
            (const char *) block + header->vocab_offsets_offset;
        view->private_sorted_vocab_ids =
            (const void *) (
                (const char *) block + header->sorted_vocab_ids_offset
            );
    }
    return true;
}

bool
evoke_am_resident_fold_lookup_token(
    const evoke_am_resident_fold_view *view,
    const char *token,
    uint32 *term_id_out
)
{
    const evoke_am_resident_fold_header *header;
    const uint64 *vocab_offsets;
    const uint32 *sorted_ids;
    uint32 low = 0;
    uint32 high;

    if (view == NULL || token == NULL || term_id_out == NULL ||
        view->private_header == NULL ||
        view->private_vocab_offsets == NULL ||
        view->private_sorted_vocab_ids == NULL)
    {
        return false;
    }
    header = view->private_header;
    vocab_offsets = view->private_vocab_offsets;
    sorted_ids = view->private_sorted_vocab_ids;
    high = header->vocab_size;
    while (low < high)
    {
        uint32 middle = low + (high - low) / 2;
        uint32 candidate_id = sorted_ids[middle];
        const char *candidate = (const char *) header +
            vocab_offsets[candidate_id];
        int comparison = strcmp(candidate, token);

        if (comparison < 0)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= header->vocab_size)
    {
        return false;
    }
    *term_id_out = sorted_ids[low];
    return strcmp(
        (const char *) header + vocab_offsets[*term_id_out],
        token
    ) == 0;
}

static const void *
evoke_am_resident_fold_pointer(
    const evoke_am_resident_fold_header *header,
    uint64 offset
)
{
    return offset == 0 ? NULL : (const char *) header + offset;
}

static void
evoke_am_resident_fold_query_free(evoke_am_resident_fold_query *query)
{
    if (query == NULL)
    {
        return;
    }
    if (query->original_term_ids != NULL)
    {
        pfree(query->original_term_ids);
    }
    if (query->mapped_query_ids != NULL)
    {
        pfree(query->mapped_query_ids);
    }
    if (query->doc_frequencies != NULL)
    {
        pfree(query->doc_frequencies);
    }
    if (query->extents != NULL)
    {
        pfree(query->extents);
    }
    if (query->terms != NULL)
    {
        pfree(query->terms);
    }
    memset(query, 0, sizeof(*query));
}

static bool
evoke_am_resident_fold_query_prepare(
    const evoke_am_resident_fold_view *view,
    const uint32 *query_ids,
    size_t query_len,
    evoke_am_resident_fold_query *query
)
{
    const evoke_am_resident_fold_header *header = view->private_header;
    const evoke_am_resident_fold_term *shared_terms = view->private_terms;
    const evoke_am_resident_fold_extent *shared_extents =
        view->private_extents;
    size_t capacity;

    memset(query, 0, sizeof(*query));
    evoke_index_init(&query->index);
    if (query_len == SIZE_MAX)
    {
        return false;
    }
    capacity = query_len + 1;
    if (capacity > MaxAllocSize / sizeof(uint32))
    {
        return false;
    }
    query->mapped_query_ids = palloc(sizeof(uint32) * capacity);
    query->original_term_ids = palloc(sizeof(uint32) * capacity);
    for (size_t query_index = 0; query_index < query_len; query_index++)
    {
        uint32 original_id = query_ids[query_index];
        size_t unique_index;

        query->mapped_query_ids[query_index] = UINT32_MAX;
        if (original_id >= header->vocab_size)
        {
            continue;
        }
        for (unique_index = 0;
             unique_index < query->unique_term_count;
             unique_index++)
        {
            if (query->original_term_ids[unique_index] == original_id)
            {
                break;
            }
        }
        if (unique_index == query->unique_term_count)
        {
            query->original_term_ids[query->unique_term_count++] =
                original_id;
            query->extent_count += shared_terms[original_id].extent_count;
        }
        query->mapped_query_ids[query_index] = (uint32) unique_index;
    }
    if (query->unique_term_count == 0 &&
        header->empty_token_id < header->vocab_size)
    {
        query->original_term_ids[0] = header->empty_token_id;
        query->unique_term_count = 1;
        query->extent_count =
            shared_terms[header->empty_token_id].extent_count;
    }
    if (query->unique_term_count > UINT32_MAX ||
        query->extent_count > MaxAllocSize / sizeof(*query->extents) ||
        query->unique_term_count >
            MaxAllocSize / sizeof(*query->terms))
    {
        return false;
    }
    if (query->unique_term_count == 0)
    {
        return true;
    }
    query->terms = palloc0(
        sizeof(*query->terms) * query->unique_term_count
    );
    if (query->extent_count > 0)
    {
        query->extents = palloc0(
            sizeof(*query->extents) * query->extent_count
        );
    }
    query->doc_frequencies = palloc(
        sizeof(*query->doc_frequencies) * query->unique_term_count
    );
    {
        const uint32 *shared_doc_frequencies =
            evoke_am_resident_fold_pointer(
                header,
                header->doc_frequencies_offset
            );
        size_t extent_cursor = 0;

        for (size_t unique_index = 0;
             unique_index < query->unique_term_count;
             unique_index++)
        {
            uint32 original_id = query->original_term_ids[unique_index];
            const evoke_am_resident_fold_term *shared_term =
                &shared_terms[original_id];

            query->doc_frequencies[unique_index] =
                shared_doc_frequencies[original_id];
            query->terms[unique_index].extents =
                &query->extents[extent_cursor];
            query->terms[unique_index].len = shared_term->extent_count;
            for (uint32 local_extent = 0;
                 local_extent < shared_term->extent_count;
                 local_extent++)
            {
                const evoke_am_resident_fold_extent *source =
                    &shared_extents[
                        shared_term->first_extent + local_extent
                    ];
                evoke_posting_extent *target =
                    &query->extents[extent_cursor++];

                target->data = evoke_am_resident_fold_pointer(
                    header,
                    source->data_offset
                );
                target->indices = evoke_am_resident_fold_pointer(
                    header,
                    source->indices_offset
                );
                target->term_frequencies =
                    evoke_am_resident_fold_pointer(
                        header,
                        source->term_frequencies_offset
                    );
                target->values = evoke_am_resident_fold_pointer(
                    header,
                    source->values_offset
                );
                target->document_id_map =
                    evoke_am_resident_fold_pointer(
                        header,
                        source->document_id_map_offset
                    );
                target->blocks = evoke_am_resident_fold_pointer(
                    header,
                    source->blocks_offset
                );
                target->len = source->posting_count;
                target->document_id_base = source->document_id_base;
                target->local_document_count =
                    source->local_document_count;
                target->block_count = source->block_count;
                target->block_shift = source->block_shift;
                target->kind = source->kind;
            }
        }
        if (extent_cursor != query->extent_count)
        {
            return false;
        }
    }
    query->index.params = header->params;
    query->index.num_docs = header->num_docs;
    query->index.vocab_size = (uint32) query->unique_term_count;
    query->index.doc_lengths = (void *)
        evoke_am_resident_fold_pointer(
            header,
            header->doc_lengths_offset
        );
    query->index.doc_frequencies = query->doc_frequencies;
    query->index.has_empty_token =
        query->unique_term_count == 1 &&
        query->original_term_ids[0] == header->empty_token_id;
    query->index.empty_token_id = 0;
    query->stats.document_count = header->corpus_document_count;
    query->stats.total_document_length = header->total_document_length;
    query->stats.doc_frequencies = query->doc_frequencies;
    query->stats.vocab_size = query->unique_term_count;
    return true;
}

static bool
evoke_am_resident_fold_query_has_blockmax(
    const evoke_am_resident_fold_header *header,
    const evoke_am_resident_fold_query *query
)
{
    if (header->document_block_count == 0 ||
        header->document_block_extrema_offset == 0)
    {
        return false;
    }
    for (size_t extent_index = 0;
         extent_index < query->extent_count;
         extent_index++)
    {
        if (query->extents[extent_index].len > 0 &&
            (query->extents[extent_index].blocks == NULL ||
             query->extents[extent_index].block_count == 0 ||
             query->extents[extent_index].block_shift !=
                header->block_shift))
        {
            return false;
        }
    }
    return true;
}

static evoke_status
evoke_am_resident_fold_topk_internal(
    const evoke_am_resident_fold_view *view,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_len,
    const uint8 *allowed_document_bitmap,
    size_t allowed_document_count,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *stats_out
)
{
    const evoke_am_resident_fold_header *header;
    evoke_am_resident_fold_query query;
    const uint32 *retired_ids;
    const uint64 *tie_break_keys;
    const uint32 *tie_break_order;
    float *scores = NULL;
    evoke_status status;

    if (result_out == NULL || view == NULL || view->private_header == NULL ||
        (query_len > 0 && query_ids == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    memset(result_out, 0, sizeof(*result_out));
    if (stats_out != NULL)
    {
        memset(stats_out, 0, sizeof(*stats_out));
    }
    header = view->private_header;
    if (allowed_document_bitmap != NULL &&
        (allowed_document_count > header->num_docs ||
         k > allowed_document_count))
    {
        return EVOKE_ERR_INVALID;
    }
    if (!evoke_am_resident_fold_query_prepare(
            view,
            query_ids,
            query_len,
            &query))
    {
        evoke_am_resident_fold_query_free(&query);
        return EVOKE_ERR_NOMEM;
    }
    if (query.unique_term_count == 0)
    {
        evoke_am_resident_fold_query_free(&query);
        return EVOKE_OK;
    }
    retired_ids = evoke_am_resident_fold_pointer(
        header,
        header->retired_document_ids_offset
    );
    tie_break_keys = evoke_am_resident_fold_pointer(
        header,
        header->tie_break_keys_offset
    );
    tie_break_order =
        (header->flags & EVOKE_AM_RESIDENT_FOLD_TIE_ORDER_IDENTITY) != 0
            ? NULL
            : evoke_am_resident_fold_pointer(
                header,
                header->tie_break_order_offset
            );
    if (evoke_am_resident_fold_query_has_blockmax(header, &query))
    {
        if (allowed_document_bitmap != NULL)
        {
            status =
                evoke_topk_from_weighted_ids_mixed_retired_blockmax_filtered_with_tie_breaks(
                    &query.index,
                    &query.stats,
                    query.terms,
                    query.unique_term_count,
                    evoke_am_resident_fold_pointer(
                        header,
                        header->document_block_extrema_offset
                    ),
                    header->document_block_count,
                    header->block_shift,
                    retired_ids,
                    header->retired_document_count,
                    tie_break_keys,
                    tie_break_order,
                    tie_break_order == NULL
                        ? 0
                        : header->tie_break_order_count,
                    allowed_document_bitmap,
                    allowed_document_count,
                    query.mapped_query_ids,
                    query_weights,
                    query_len,
                    Min(k, allowed_document_count),
                    result_out,
                    stats_out
                );
        }
        else
        {
            status =
                evoke_topk_from_weighted_ids_mixed_retired_blockmax_with_tie_breaks(
                &query.index,
                &query.stats,
                query.terms,
                query.unique_term_count,
                evoke_am_resident_fold_pointer(
                    header,
                    header->document_block_extrema_offset
                ),
                header->document_block_count,
                header->block_shift,
                retired_ids,
                header->retired_document_count,
                tie_break_keys,
                tie_break_order,
                tie_break_order == NULL
                    ? 0
                    : header->tie_break_order_count,
                query.mapped_query_ids,
                query_weights,
                query_len,
                Min(k, (size_t) header->num_docs),
                result_out,
                stats_out
            );
        }
    }
    else
    {
        if (allowed_document_bitmap != NULL)
        {
            evoke_am_resident_fold_query_free(&query);
            return EVOKE_ERR_INVALID;
        }
        status = evoke_scores_from_weighted_ids_mixed_retired(
            &query.index,
            &query.stats,
            query.terms,
            query.unique_term_count,
            retired_ids,
            header->retired_document_count,
            query.mapped_query_ids,
            query_weights,
            query_len,
            NULL,
            &scores
        );
        if (status == EVOKE_OK)
        {
            status = evoke_topk_with_tie_breaks(
                scores,
                header->num_docs,
                Min(k, (size_t) header->num_docs),
                true,
                tie_break_keys,
                result_out
            );
        }
        free(scores);
    }
    evoke_am_resident_fold_query_free(&query);
    return status;
}

evoke_status
evoke_am_resident_fold_topk(
    const evoke_am_resident_fold_view *view,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_len,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *stats_out
)
{
    return evoke_am_resident_fold_topk_internal(
        view,
        query_ids,
        query_weights,
        query_len,
        NULL,
        0,
        k,
        result_out,
        stats_out
    );
}

evoke_status
evoke_am_resident_fold_topk_filtered(
    const evoke_am_resident_fold_view *view,
    const uint32 *query_ids,
    const float *query_weights,
    size_t query_len,
    const uint8 *allowed_document_bitmap,
    size_t allowed_document_count,
    size_t k,
    evoke_topk_result *result_out,
    evoke_blockmax_stats *stats_out
)
{
    if (allowed_document_bitmap == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    return evoke_am_resident_fold_topk_internal(
        view,
        query_ids,
        query_weights,
        query_len,
        allowed_document_bitmap,
        allowed_document_count,
        k,
        result_out,
        stats_out
    );
}

const ItemPointerData *
evoke_am_resident_fold_document_tids(
    const evoke_am_resident_fold_view *view
)
{
    const evoke_am_resident_fold_header *header;

    if (view == NULL || view->private_header == NULL)
    {
        return NULL;
    }
    header = view->private_header;
    return evoke_am_resident_fold_pointer(
        header,
        header->document_tids_offset
    );
}

uint32
evoke_am_resident_fold_document_count(
    const evoke_am_resident_fold_view *view
)
{
    const evoke_am_resident_fold_header *header = view == NULL
        ? NULL
        : view->private_header;

    return header == NULL ? 0 : header->num_docs;
}

uint64
evoke_am_resident_fold_root_id(const evoke_am_resident_fold_view *view)
{
    const evoke_am_resident_fold_header *header = view == NULL
        ? NULL
        : view->private_header;

    return header == NULL ? 0 : header->root.root_id;
}
