#include "postgres.h"

#include <math.h>

#include "utils/memutils.h"

#include "evoke_am_hot_fold.h"
#include "evoke_core.h"
#include "evoke_segment_pages.h"
#include "evoke_segments.h"

#define EVOKE_AM_HOT_FOLD_MAX_TERMS UINT32_C(1024)

static bool
evoke_am_hot_fold_checked_add(
    size_t left,
    size_t right,
    size_t *result_out
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
evoke_am_hot_fold_checked_mul(
    size_t left,
    size_t right,
    size_t *result_out
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

static const evoke_am_hot_fold_header *
evoke_am_hot_fold_view_header(const evoke_am_hot_fold_view *view)
{
    return view == NULL ? NULL : view->private_header;
}

static const evoke_am_hot_fold_term *
evoke_am_hot_fold_view_terms(const evoke_am_hot_fold_view *view)
{
    return view == NULL ? NULL : view->private_terms;
}

static const evoke_am_hot_fold_entry *
evoke_am_hot_fold_view_entries(const evoke_am_hot_fold_view *view)
{
    return view == NULL ? NULL : view->private_entries;
}

static uint64
evoke_am_hot_fold_checksum(const void *block, Size size)
{
    const uint8 *bytes = block;
    const Size checksum_offset = offsetof(
        evoke_am_hot_fold_header,
        checksum
    );
    uint64 checksum = UINT64_C(14695981039346656037);

    for (Size index = 0; index < size; index++)
    {
        uint8 value = index >= checksum_offset &&
            index < checksum_offset + sizeof(uint64)
            ? 0
            : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static int
evoke_am_hot_fold_compare_terms(const void *left, const void *right)
{
    const evoke_am_hot_fold_term *left_term = left;
    const evoke_am_hot_fold_term *right_term = right;

    if (left_term->term_id < right_term->term_id)
    {
        return -1;
    }
    return left_term->term_id > right_term->term_id;
}

static int
evoke_am_hot_fold_compare_entries(const void *left, const void *right)
{
    const evoke_am_hot_fold_entry *left_entry = left;
    const evoke_am_hot_fold_entry *right_entry = right;

    if (left_entry->score > right_entry->score)
    {
        return -1;
    }
    if (left_entry->score < right_entry->score)
    {
        return 1;
    }
    if (left_entry->tie_break_key < right_entry->tie_break_key)
    {
        return -1;
    }
    if (left_entry->tie_break_key > right_entry->tie_break_key)
    {
        return 1;
    }
    if (left_entry->document_slot < right_entry->document_slot)
    {
        return -1;
    }
    return left_entry->document_slot > right_entry->document_slot;
}

static bool
evoke_am_hot_fold_block_header_validate(
    const void *block,
    Size mapped_size
)
{
    const evoke_am_hot_fold_header *header = block;
    size_t term_bytes;
    size_t entry_bytes;
    size_t expected_entries_offset;
    size_t expected_total_size;

    if (block == NULL || mapped_size < sizeof(*header) ||
        header->magic != EVOKE_AM_HOT_FOLD_MAGIC ||
        header->version != EVOKE_AM_HOT_FOLD_VERSION ||
        header->total_size != mapped_size ||
        header->term_count == 0 ||
        header->term_count > EVOKE_AM_HOT_FOLD_MAX_TERMS ||
        header->entry_count == 0 ||
        header->entry_count > SIZE_MAX ||
        header->terms_offset != MAXALIGN(sizeof(*header)) ||
        !evoke_am_hot_fold_checked_mul(
            header->term_count,
            sizeof(evoke_am_hot_fold_term),
            &term_bytes) ||
        !evoke_am_hot_fold_checked_add(
            header->terms_offset,
            term_bytes,
            &expected_entries_offset))
    {
        return false;
    }
    expected_entries_offset = MAXALIGN(expected_entries_offset);
    if (header->entries_offset != expected_entries_offset ||
        !evoke_am_hot_fold_checked_mul(
            (size_t) header->entry_count,
            sizeof(evoke_am_hot_fold_entry),
            &entry_bytes) ||
        !evoke_am_hot_fold_checked_add(
            header->entries_offset,
            entry_bytes,
            &expected_total_size) ||
        expected_total_size != mapped_size)
    {
        return false;
    }
    return true;
}

static bool
evoke_am_hot_fold_block_validate(const void *block, Size mapped_size)
{
    const evoke_am_hot_fold_header *header = block;
    const evoke_am_hot_fold_term *terms;
    const evoke_am_hot_fold_entry *entries;
    uint64 entry_cursor = 0;

    if (!evoke_am_hot_fold_block_header_validate(block, mapped_size) ||
        header->checksum != evoke_am_hot_fold_checksum(block, mapped_size))
    {
        return false;
    }
    terms = (const evoke_am_hot_fold_term *) (
        (const char *) block + header->terms_offset
    );
    entries = (const evoke_am_hot_fold_entry *) (
        (const char *) block + header->entries_offset
    );
    for (uint32 term_index = 0;
         term_index < header->term_count;
         term_index++)
    {
        const evoke_am_hot_fold_term *term = &terms[term_index];

        if (term->reserved != 0 || term->entry_count == 0 ||
            term->first_entry != entry_cursor ||
            term->entry_count > header->entry_count - entry_cursor ||
            (term_index > 0 &&
             terms[term_index - 1].term_id >= term->term_id))
        {
            return false;
        }
        for (uint64 entry_index = term->first_entry;
             entry_index < term->first_entry + term->entry_count;
             entry_index++)
        {
            const evoke_am_hot_fold_entry *entry = &entries[entry_index];

            if (!isfinite(entry->score) || entry->score <= 0.0f ||
                entry->tie_break_key == 0 || entry->reserved != 0 ||
                !ItemPointerIsValid(&entry->tid) ||
                (entry_index > term->first_entry &&
                 evoke_am_hot_fold_compare_entries(
                     &entries[entry_index - 1],
                     entry) > 0))
            {
                return false;
            }
        }
        entry_cursor += term->entry_count;
    }
    return entry_cursor == header->entry_count;
}

void
evoke_am_hot_fold_view_init(evoke_am_hot_fold_view *view)
{
    if (view != NULL)
    {
        memset(view, 0, sizeof(*view));
    }
}

void
evoke_am_hot_fold_view_release(evoke_am_hot_fold_view *view)
{
    if (view == NULL)
    {
        return;
    }
    evoke_am_preload_lease_release(&view->private_lease);
    memset(view, 0, sizeof(*view));
}

bool
evoke_am_hot_fold_attach(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_hot_fold_view *view
)
{
    Size mapped_size;
    const void *block;
    const evoke_am_hot_fold_header *header;

    if (view == NULL)
    {
        return false;
    }
    evoke_am_hot_fold_view_init(view);
    if (index_relation == NULL || meta == NULL ||
        !evoke_am_preload_attach(
            index_relation,
            meta,
            EVOKE_AM_PRELOAD_HOT_FOLD,
            &view->private_lease))
    {
        return false;
    }
    block = evoke_am_preload_lease_payload(&view->private_lease);
    mapped_size = evoke_am_preload_lease_payload_size(
        &view->private_lease
    );
    /*
     * The publishing worker validates checksum and ordering before setting
     * ready. An immutable exact-root lease makes this constant-time layout
     * validation sufficient on every query attach.
     */
    if (!evoke_am_hot_fold_block_header_validate(block, mapped_size))
    {
        evoke_am_hot_fold_view_release(view);
        return false;
    }
    header = block;
    view->private_header = header;
    view->private_terms = (const char *) block + header->terms_offset;
    view->private_entries = (const char *) block + header->entries_offset;
    return true;
}

const evoke_am_hot_fold_term *
evoke_am_hot_fold_find_term(
    const evoke_am_hot_fold_view *view,
    uint32 term_id
)
{
    const evoke_am_hot_fold_header *header =
        evoke_am_hot_fold_view_header(view);
    const evoke_am_hot_fold_term *terms =
        evoke_am_hot_fold_view_terms(view);
    uint32 low = 0;
    uint32 high;

    if (header == NULL || terms == NULL)
    {
        return NULL;
    }
    high = header->term_count;
    while (low < high)
    {
        uint32 middle = low + (high - low) / 2;
        uint32 candidate = terms[middle].term_id;

        if (candidate < term_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if (low >= header->term_count || terms[low].term_id != term_id)
    {
        return NULL;
    }
    return &terms[low];
}

const evoke_am_hot_fold_entry *
evoke_am_hot_fold_term_entries(
    const evoke_am_hot_fold_view *view,
    const evoke_am_hot_fold_term *term
)
{
    const evoke_am_hot_fold_header *header =
        evoke_am_hot_fold_view_header(view);
    const evoke_am_hot_fold_entry *entries =
        evoke_am_hot_fold_view_entries(view);

    if (header == NULL || entries == NULL || term == NULL ||
        term->first_entry > header->entry_count ||
        term->entry_count > header->entry_count - term->first_entry)
    {
        return NULL;
    }
    return &entries[term->first_entry];
}

static bool
evoke_am_hot_fold_build_block(
    const evoke_am_hot_fold_view *prior_view,
    uint32 term_id,
    const evoke_term_fold_bundle *impact_fold,
    const evoke_segment_storage_snapshot *snapshot,
    void **block_out,
    Size *size_out
)
{
    const evoke_am_hot_fold_header *prior_header =
        evoke_am_hot_fold_view_header(prior_view);
    const evoke_am_hot_fold_term *prior_terms =
        evoke_am_hot_fold_view_terms(prior_view);
    const evoke_am_hot_fold_entry *prior_entries =
        evoke_am_hot_fold_view_entries(prior_view);
    const evoke_am_hot_fold_term *prior_term = NULL;
    evoke_am_hot_fold_header *header;
    evoke_am_hot_fold_term *terms;
    evoke_am_hot_fold_entry *entries;
    void *block;
    size_t term_bytes;
    size_t entry_bytes;
    size_t cursor;
    size_t entries_offset;
    size_t total_size;
    uint64 live_entry_count = 0;
    uint64 output_entry_count;
    uint32 output_term_count;
    evoke_status status;

    if (block_out == NULL || size_out == NULL)
    {
        return false;
    }
    *block_out = NULL;
    *size_out = 0;
    if (impact_fold == NULL || snapshot == NULL ||
        impact_fold->run_count != 1 ||
        impact_fold->runs[0].term_id != term_id ||
        impact_fold->runs[0].kind !=
            EVOKE_POSTING_EXTENT_LEXICAL_IMPACT ||
        impact_fold->runs[0].posting_offset != 0 ||
        impact_fold->runs[0].posting_count !=
            impact_fold->posting_count ||
        evoke_method_requires_nonoccurrence(
            snapshot->index_metadata.params.method))
    {
        return false;
    }
    status = evoke_term_fold_bundle_validate(impact_fold);
    if (status != EVOKE_OK)
    {
        return false;
    }
    for (uint64 entry_index = 0;
         entry_index < impact_fold->posting_count;
         entry_index++)
    {
        uint32 document_slot = impact_fold->document_slots[entry_index];
        float score = impact_fold->values[entry_index].impact;

        if (document_slot >= snapshot->index_metadata.num_docs ||
            !isfinite(score) || score <= 0.0f)
        {
            return false;
        }
        if (ItemPointerIsValid(&snapshot->doc_tids[document_slot]))
        {
            if (snapshot->document_tie_break_keys[document_slot] == 0)
            {
                return false;
            }
            live_entry_count++;
        }
    }
    if (live_entry_count == 0)
    {
        return false;
    }
    if (prior_header != NULL)
    {
        prior_term = evoke_am_hot_fold_find_term(prior_view, term_id);
    }
    output_term_count = prior_header == NULL
        ? 1
        : prior_header->term_count + (prior_term == NULL ? 1 : 0);
    if (output_term_count > EVOKE_AM_HOT_FOLD_MAX_TERMS)
    {
        return false;
    }
    output_entry_count = live_entry_count;
    if (prior_header != NULL)
    {
        if (prior_header->entry_count > UINT64_MAX - output_entry_count)
        {
            return false;
        }
        output_entry_count += prior_header->entry_count;
        if (prior_term != NULL)
        {
            if (prior_term->entry_count > output_entry_count)
            {
                return false;
            }
            output_entry_count -= prior_term->entry_count;
        }
    }
    if (output_entry_count == 0 || output_entry_count > SIZE_MAX ||
        !evoke_am_hot_fold_checked_mul(
            output_term_count,
            sizeof(*terms),
            &term_bytes) ||
        !evoke_am_hot_fold_checked_mul(
            (size_t) output_entry_count,
            sizeof(*entries),
            &entry_bytes))
    {
        return false;
    }
    cursor = MAXALIGN(sizeof(*header));
    if (!evoke_am_hot_fold_checked_add(
            cursor,
            term_bytes,
            &entries_offset))
    {
        return false;
    }
    entries_offset = MAXALIGN(entries_offset);
    if (!evoke_am_hot_fold_checked_add(
            entries_offset,
            entry_bytes,
            &total_size) ||
        total_size > MaxAllocSize)
    {
        return false;
    }
    block = palloc0(total_size);
    header = block;
    terms = (evoke_am_hot_fold_term *) ((char *) block + cursor);
    entries = (evoke_am_hot_fold_entry *) (
        (char *) block + entries_offset
    );
    header->magic = EVOKE_AM_HOT_FOLD_MAGIC;
    header->version = EVOKE_AM_HOT_FOLD_VERSION;
    header->total_size = total_size;
    header->term_count = output_term_count;
    header->entry_count = output_entry_count;
    header->terms_offset = cursor;
    header->entries_offset = entries_offset;
    if (prior_header != NULL)
    {
        memcpy(
            terms,
            prior_terms,
            prior_header->term_count * sizeof(*terms)
        );
    }
    if (prior_term == NULL)
    {
        terms[output_term_count - 1].term_id = term_id;
    }
    qsort(
        terms,
        output_term_count,
        sizeof(*terms),
        evoke_am_hot_fold_compare_terms
    );
    cursor = 0;
    for (uint32 output_term_index = 0;
         output_term_index < output_term_count;
         output_term_index++)
    {
        evoke_am_hot_fold_term *term = &terms[output_term_index];
        uint64 source_first = term->first_entry;
        uint64 source_count = term->entry_count;

        term->first_entry = cursor;
        if (term->term_id == term_id)
        {
            term->entry_count = live_entry_count;
            for (uint64 input_index = 0;
                 input_index < impact_fold->posting_count;
                 input_index++)
            {
                uint32 document_slot =
                    impact_fold->document_slots[input_index];
                evoke_am_hot_fold_entry *entry;

                if (!ItemPointerIsValid(
                        &snapshot->doc_tids[document_slot]))
                {
                    continue;
                }
                entry = &entries[cursor++];
                entry->tie_break_key =
                    snapshot->document_tie_break_keys[document_slot];
                entry->document_slot = document_slot;
                entry->score = impact_fold->values[input_index].impact;
                entry->tid = snapshot->doc_tids[document_slot];
            }
            qsort(
                &entries[term->first_entry],
                (size_t) term->entry_count,
                sizeof(*entries),
                evoke_am_hot_fold_compare_entries
            );
        }
        else
        {
            if (prior_header == NULL || prior_entries == NULL ||
                source_count == 0 ||
                source_first > prior_header->entry_count ||
                source_count > prior_header->entry_count - source_first)
            {
                pfree(block);
                return false;
            }
            memcpy(
                &entries[cursor],
                &prior_entries[source_first],
                (size_t) source_count * sizeof(*entries)
            );
            cursor += source_count;
        }
    }
    if (cursor != output_entry_count)
    {
        pfree(block);
        return false;
    }
    header->checksum = evoke_am_hot_fold_checksum(block, total_size);
    if (!evoke_am_hot_fold_block_validate(block, total_size))
    {
        pfree(block);
        return false;
    }
    *block_out = block;
    *size_out = total_size;
    return true;
}

bool
evoke_am_hot_fold_publish(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_hot_fold_view *prior_view,
    uint32 term_id,
    const evoke_term_fold_bundle *impact_fold,
    const evoke_segment_storage_snapshot *snapshot
)
{
    evoke_am_preload_reservation reservation = {0};
    evoke_am_preload_reserve_result reserve_result;
    void *payload = NULL;
    Size payload_size = 0;
    bool published = false;

    if (index_relation == NULL || meta == NULL || prior_view == NULL ||
        !evoke_am_preload_cache_available() ||
        !evoke_am_hot_fold_build_block(
            prior_view,
            term_id,
            impact_fold,
            snapshot,
            &payload,
            &payload_size))
    {
        evoke_am_hot_fold_view_release(prior_view);
        return false;
    }
    evoke_am_hot_fold_view_release(prior_view);
    evoke_am_preload_retire_exact(
        index_relation,
        meta,
        EVOKE_AM_PRELOAD_HOT_FOLD
    );
    reserve_result = evoke_am_preload_reserve(
        index_relation,
        meta,
        EVOKE_AM_PRELOAD_HOT_FOLD,
        payload_size,
        &reservation
    );
    if (reserve_result != EVOKE_AM_PRELOAD_RESERVE_NEW)
    {
        pfree(payload);
        return reserve_result == EVOKE_AM_PRELOAD_RESERVE_READY;
    }
    PG_TRY();
    {
        memcpy(
            evoke_am_preload_reservation_payload(&reservation),
            payload,
            payload_size
        );
        published = evoke_am_preload_reservation_commit(&reservation);
    }
    PG_CATCH();
    {
        evoke_am_preload_reservation_abort(&reservation);
        pfree(payload);
        PG_RE_THROW();
    }
    PG_END_TRY();
    pfree(payload);
    return published;
}
