#include "evoke_semantic_accelerator_directory.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_ACCELERATOR_DIRECTORY_MAGIC UINT32_C(0x44414932)
#define EVOKE_ACCELERATOR_DIRECTORY_RETIREMENT_MIN_VERSION UINT16_C(5)
#define EVOKE_ACCELERATOR_DIRECTORY_RETIREMENT_MAX_VERSION UINT16_C(9)
#define EVOKE_ACCELERATOR_DIRECTORY_PHYSICAL_COST_VERSION UINT16_C(9)
#define EVOKE_ACCELERATOR_DIRECTORY_VERSION UINT16_C(10)
#define EVOKE_ACCELERATOR_DIRECTORY_SCOPE_HEADER_SIZE 128U
#define EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE 56U
#define EVOKE_ACCELERATOR_RETIREMENT_V5_FORWARD_ENTRY_SIZE 56U
#define EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE 64U
#define EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE 48U
#define EVOKE_ACCELERATOR_DIRECTORY_CHECKSUM_OFFSET 64U

static void evoke_accelerator_directory_read_ref(
    const uint8_t *bytes,
    evoke_segment_object_ref *ref
);

static bool evoke_accelerator_directory_ref_is_zero(
    const evoke_segment_object_ref *ref
);

static bool evoke_accelerator_directory_refs_overlap(
    const evoke_segment_object_ref *left,
    const evoke_segment_object_ref *right
);

static uint32_t
evoke_accelerator_directory_expected_forward_bound_shards(
    uint32_t vocab_size
)
{
    return vocab_size == 0
        ? 0
        : (vocab_size - 1U) /
            EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD + 1U;
}

static uint64_t
evoke_accelerator_directory_expected_forward_row_offsets(
    uint32_t document_count,
    uint32_t forward_chunk_count
)
{
    return (uint64_t) document_count + forward_chunk_count;
}

static bool
evoke_accelerator_directory_policy_is_queryable(uint32_t policy)
{
    return policy == EVOKE_SEMANTIC_ACCELERATOR_CURRENT_POLICY;
}

static void
evoke_accelerator_directory_write_u16(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t) (value & UINT16_C(0xFF));
    bytes[1] = (uint8_t) ((value >> 8) & UINT16_C(0xFF));
}

static void
evoke_accelerator_directory_write_u32(uint8_t *bytes, uint32_t value)
{
    size_t index;

    for (index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) ((value >> (index * 8)) & UINT32_C(0xFF));
    }
}

static void
evoke_accelerator_directory_write_u64(uint8_t *bytes, uint64_t value)
{
    size_t index;

    for (index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) ((value >> (index * 8)) & UINT64_C(0xFF));
    }
}

static uint16_t
evoke_accelerator_directory_read_u16(const uint8_t *bytes)
{
    return (uint16_t) bytes[0] |
        (uint16_t) ((uint16_t) bytes[1] << 8);
}

static uint32_t
evoke_accelerator_directory_read_u32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0] |
        ((uint32_t) bytes[1] << 8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static uint64_t
evoke_accelerator_directory_read_u64(const uint8_t *bytes)
{
    uint64_t value = 0;
    size_t index;

    for (index = 0; index < sizeof(value); index++)
    {
        value |= (uint64_t) bytes[index] << (index * 8);
    }
    return value;
}

static uint64_t
evoke_accelerator_directory_checksum(const uint8_t *bytes, size_t size)
{
    uint64_t checksum = UINT64_C(14695981039346656037);
    size_t index;

    for (index = 0; index < size; index++)
    {
        uint8_t value =
            index >= EVOKE_ACCELERATOR_DIRECTORY_CHECKSUM_OFFSET &&
            index < EVOKE_ACCELERATOR_DIRECTORY_CHECKSUM_OFFSET +
                sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= value;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

evoke_status
evoke_semantic_accelerator_directory_summary_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_accelerator_directory_summary *summary_out
)
{
    evoke_semantic_accelerator_directory_summary summary;
    size_t entry_bytes;
    size_t forward_bytes;
    size_t forward_term_work_bytes;
    size_t forward_chunk_cost_bytes;
    size_t forward_row_offset_bytes;
    size_t forward_term_bytes;
    size_t forward_bound_term_bytes;
    size_t forward_bound_ref_bytes;
    size_t expected_size;
    uint16_t version;
    uint16_t header_size;
    uint16_t term_entry_size;
    uint16_t forward_entry_size;

    if (bytes == NULL || summary_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (size < EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&summary, 0, sizeof(summary));
    version = evoke_accelerator_directory_read_u16(bytes + 4);
    header_size = evoke_accelerator_directory_read_u16(bytes + 6);
    term_entry_size = evoke_accelerator_directory_read_u16(bytes + 8);
    forward_entry_size = evoke_accelerator_directory_read_u16(bytes + 10);
    if (evoke_accelerator_directory_read_u32(bytes + 0) !=
            EVOKE_ACCELERATOR_DIRECTORY_MAGIC ||
        (version < EVOKE_ACCELERATOR_DIRECTORY_RETIREMENT_MIN_VERSION ||
         version > EVOKE_ACCELERATOR_DIRECTORY_VERSION) ||
        header_size != EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE ||
        size < header_size ||
        term_entry_size != EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE ||
        forward_entry_size !=
            (version == EVOKE_ACCELERATOR_DIRECTORY_RETIREMENT_MIN_VERSION
                ? EVOKE_ACCELERATOR_RETIREMENT_V5_FORWARD_ENTRY_SIZE
                : EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE))
    {
        return EVOKE_ERR_FORMAT;
    }
    summary.format_version = version;
    summary.header_size = header_size;
    summary.term_entry_size = term_entry_size;
    summary.forward_entry_size = forward_entry_size;
    summary.document_count = evoke_accelerator_directory_read_u32(bytes + 12);
    summary.vocab_size = evoke_accelerator_directory_read_u32(bytes + 16);
    summary.term_count = evoke_accelerator_directory_read_u32(bytes + 20);
    summary.forward_chunk_count =
        evoke_accelerator_directory_read_u32(bytes + 24);
    summary.forward_document_shift =
        evoke_accelerator_directory_read_u32(bytes + 28);
    summary.source_manifest_id =
        evoke_accelerator_directory_read_u64(bytes + 32);
    summary.source_authority_checksum =
        evoke_accelerator_directory_read_u64(bytes + 40);
    summary.owner_manifest_id =
        evoke_accelerator_directory_read_u64(bytes + 48);
    summary.total_size = evoke_accelerator_directory_read_u64(bytes + 56);
    summary.builder_policy_id =
        evoke_accelerator_directory_read_u32(bytes + 72);
    summary.retained_document_cap =
        evoke_accelerator_directory_read_u32(bytes + 76);
    summary.forward_term_work_count =
        version >= UINT16_C(7)
            ? summary.vocab_size
            : 0;
    summary.forward_chunk_cost_count =
        version >= EVOKE_ACCELERATOR_DIRECTORY_PHYSICAL_COST_VERSION
            ? summary.forward_chunk_count
            : 0;
    summary.forward_row_offset_count =
        version == EVOKE_ACCELERATOR_DIRECTORY_VERSION
            ? evoke_accelerator_directory_expected_forward_row_offsets(
                  summary.document_count,
                  summary.forward_chunk_count
              )
            : 0;
    summary.forward_term_bytes_count =
        version >= EVOKE_ACCELERATOR_DIRECTORY_PHYSICAL_COST_VERSION
            ? summary.vocab_size
            : 0;
    summary.forward_bound_term_bytes_count =
        version >= EVOKE_ACCELERATOR_DIRECTORY_PHYSICAL_COST_VERSION
            ? summary.vocab_size
            : 0;
    summary.forward_bound_shard_count =
        version >= UINT16_C(8)
            ? evoke_accelerator_directory_expected_forward_bound_shards(
                  summary.vocab_size
              )
            : 0;
    evoke_accelerator_directory_read_ref(bytes + 80, &summary.scope_object);
    evoke_accelerator_directory_read_ref(
        bytes + EVOKE_ACCELERATOR_DIRECTORY_SCOPE_HEADER_SIZE,
        &summary.tid_lookup_object
    );

    entry_bytes = (size_t) summary.term_count *
        summary.term_entry_size;
    forward_bytes = (size_t) summary.forward_chunk_count *
        summary.forward_entry_size;
    forward_term_work_bytes =
        (size_t) summary.forward_term_work_count * sizeof(uint64_t);
    forward_chunk_cost_bytes =
        (size_t) summary.forward_chunk_cost_count *
        2U * sizeof(uint64_t);
    if (summary.forward_row_offset_count > SIZE_MAX / sizeof(uint32_t))
    {
        return EVOKE_ERR_RANGE;
    }
    forward_row_offset_bytes =
        (size_t) summary.forward_row_offset_count * sizeof(uint32_t);
    forward_term_bytes =
        (size_t) summary.forward_term_bytes_count * sizeof(uint64_t);
    forward_bound_term_bytes =
        (size_t) summary.forward_bound_term_bytes_count * sizeof(uint64_t);
    forward_bound_ref_bytes =
        (size_t) summary.forward_bound_shard_count *
        EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE;
    if ((summary.term_count != 0 &&
         entry_bytes / summary.term_entry_size !=
             summary.term_count) ||
        (summary.forward_chunk_count != 0 &&
         forward_bytes / summary.forward_entry_size !=
             summary.forward_chunk_count) ||
        (summary.forward_term_work_count != 0 &&
         forward_term_work_bytes / sizeof(uint64_t) !=
             summary.forward_term_work_count) ||
        (summary.forward_chunk_cost_count != 0 &&
         forward_chunk_cost_bytes / (2U * sizeof(uint64_t)) !=
             summary.forward_chunk_cost_count) ||
        (summary.forward_row_offset_count != 0 &&
         forward_row_offset_bytes / sizeof(uint32_t) !=
             summary.forward_row_offset_count) ||
        (summary.forward_term_bytes_count != 0 &&
         forward_term_bytes / sizeof(uint64_t) !=
             summary.forward_term_bytes_count) ||
        (summary.forward_bound_term_bytes_count != 0 &&
         forward_bound_term_bytes / sizeof(uint64_t) !=
             summary.forward_bound_term_bytes_count) ||
        (summary.forward_bound_shard_count != 0 &&
         forward_bound_ref_bytes /
                EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE !=
             summary.forward_bound_shard_count) ||
        entry_bytes > SIZE_MAX - header_size ||
        forward_bytes >
            SIZE_MAX - header_size - entry_bytes ||
        forward_term_work_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes ||
        forward_chunk_cost_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes -
                forward_term_work_bytes ||
        forward_row_offset_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes -
                forward_term_work_bytes - forward_chunk_cost_bytes ||
        forward_term_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes -
                forward_term_work_bytes -
                forward_chunk_cost_bytes - forward_row_offset_bytes ||
        forward_bound_term_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes -
                forward_term_work_bytes -
                forward_chunk_cost_bytes - forward_row_offset_bytes -
                forward_term_bytes ||
        forward_bound_ref_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes -
                forward_term_work_bytes -
                forward_chunk_cost_bytes - forward_row_offset_bytes -
                forward_term_bytes -
                forward_bound_term_bytes)
    {
        return EVOKE_ERR_RANGE;
    }
    expected_size = header_size +
        entry_bytes + forward_bytes + forward_term_work_bytes +
        forward_chunk_cost_bytes + forward_row_offset_bytes +
        forward_term_bytes +
        forward_bound_term_bytes + forward_bound_ref_bytes;
    if (summary.source_manifest_id == 0 ||
        summary.source_authority_checksum == 0 ||
        summary.owner_manifest_id <= summary.source_manifest_id ||
        summary.document_count == 0 || summary.vocab_size == 0 ||
        summary.term_count == 0 || summary.total_size != expected_size ||
        ((summary.builder_policy_id == 0) !=
         (summary.retained_document_cap == 0)) ||
        (summary.forward_chunk_count > 0 &&
         (summary.forward_document_shift == 0 ||
          summary.forward_document_shift >= 32)) ||
        (!evoke_accelerator_directory_ref_is_zero(
                &summary.scope_object) &&
         (summary.scope_object.object_kind !=
            EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_SCOPE ||
          summary.scope_object.object_id != 1 ||
          summary.scope_object.owner_manifest_id !=
            summary.owner_manifest_id ||
          evoke_segment_object_ref_validate(
            &summary.scope_object,
            UINT32_MAX) != EVOKE_OK)) ||
        (!evoke_accelerator_directory_ref_is_zero(
                &summary.tid_lookup_object) &&
         (summary.tid_lookup_object.object_kind !=
            EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_TID_LOOKUP ||
          summary.tid_lookup_object.object_id != 1 ||
          summary.tid_lookup_object.owner_manifest_id !=
            summary.owner_manifest_id ||
          evoke_segment_object_ref_validate(
            &summary.tid_lookup_object,
            UINT32_MAX) != EVOKE_OK)))
    {
        return EVOKE_ERR_FORMAT;
    }
    *summary_out = summary;
    return EVOKE_OK;
}

bool
evoke_semantic_accelerator_directory_summary_format_is_current(
    const evoke_semantic_accelerator_directory_summary *summary
)
{
    return summary != NULL &&
        summary->format_version == EVOKE_ACCELERATOR_DIRECTORY_VERSION &&
        summary->header_size ==
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE &&
        summary->term_entry_size ==
            EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE &&
        summary->forward_entry_size ==
            EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE &&
        summary->forward_term_work_count == summary->vocab_size &&
        summary->forward_chunk_cost_count ==
            summary->forward_chunk_count &&
        summary->forward_row_offset_count ==
            evoke_accelerator_directory_expected_forward_row_offsets(
                summary->document_count,
                summary->forward_chunk_count
            ) &&
        summary->forward_term_bytes_count == summary->vocab_size &&
        summary->forward_bound_term_bytes_count == summary->vocab_size &&
        summary->forward_bound_shard_count ==
            evoke_accelerator_directory_expected_forward_bound_shards(
                summary->vocab_size
            );
}

bool
evoke_semantic_accelerator_directory_summary_is_current(
    const evoke_semantic_accelerator_directory_summary *summary
)
{
    return
        evoke_semantic_accelerator_directory_summary_format_is_current(
            summary
        ) &&
        summary->builder_policy_id ==
            EVOKE_SEMANTIC_ACCELERATOR_CURRENT_POLICY &&
        summary->retained_document_cap ==
            EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP &&
        evoke_semantic_accelerator_directory_summary_has_tid_lookup(summary);
}

bool
evoke_semantic_accelerator_directory_summary_has_complete_forward(
    const evoke_semantic_accelerator_directory_summary *summary
)
{
    uint64_t documents_per_chunk;
    uint64_t expected_chunks;

    if (!evoke_semantic_accelerator_directory_summary_is_current(summary) ||
        !evoke_accelerator_directory_policy_is_queryable(
            summary->builder_policy_id) ||
        summary->retained_document_cap !=
            EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP ||
        summary->forward_chunk_count == 0 ||
        summary->forward_document_shift == 0 ||
        summary->forward_document_shift >= 32)
    {
        return false;
    }
    documents_per_chunk = UINT64_C(1) <<
        summary->forward_document_shift;
    expected_chunks =
        ((uint64_t) summary->document_count + documents_per_chunk - 1) /
        documents_per_chunk;
    return expected_chunks == summary->forward_chunk_count;
}

evoke_status
evoke_semantic_accelerator_directory_forward_slice(
    const uint8_t *header_bytes,
    size_t header_size,
    evoke_semantic_accelerator_directory_summary *summary_out,
    size_t *offset_out,
    size_t *size_out
)
{
    evoke_semantic_accelerator_directory_summary summary;
    size_t offset;
    evoke_status status;

    if (summary_out == NULL || offset_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    status = evoke_semantic_accelerator_directory_summary_deserialize(
        header_bytes,
        header_size,
        &summary
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (!evoke_semantic_accelerator_directory_summary_has_complete_forward(
            &summary))
    {
        return EVOKE_ERR_FORMAT;
    }
    offset = summary.header_size +
        (size_t) summary.term_count * EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE;
    if (offset > summary.total_size)
    {
        return EVOKE_ERR_FORMAT;
    }
    *summary_out = summary;
    *offset_out = offset;
    *size_out =
        (size_t) summary.forward_chunk_count *
            EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE +
        (size_t) summary.forward_term_work_count * sizeof(uint64_t) +
        (size_t) summary.forward_chunk_cost_count *
            2U * sizeof(uint64_t) +
        (size_t) summary.forward_row_offset_count * sizeof(uint32_t) +
        (size_t) summary.forward_term_bytes_count * sizeof(uint64_t) +
        (size_t) summary.forward_bound_term_bytes_count *
            sizeof(uint64_t) +
        (size_t) summary.forward_bound_shard_count *
            EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE;
    return EVOKE_OK;
}

static void
evoke_accelerator_directory_write_ref(
    uint8_t *bytes,
    const evoke_segment_object_ref *ref
)
{
    evoke_accelerator_directory_write_u32(
        bytes + 0,
        (uint32_t) ref->object_kind
    );
    evoke_accelerator_directory_write_u32(bytes + 8, ref->start_block);
    evoke_accelerator_directory_write_u32(bytes + 12, ref->page_count);
    evoke_accelerator_directory_write_u64(bytes + 16, ref->object_id);
    evoke_accelerator_directory_write_u64(
        bytes + 24,
        ref->owner_manifest_id
    );
    evoke_accelerator_directory_write_u64(bytes + 32, ref->object_bytes);
    evoke_accelerator_directory_write_u64(bytes + 40, ref->object_checksum);
}

static void
evoke_accelerator_directory_read_ref(
    const uint8_t *bytes,
    evoke_segment_object_ref *ref
)
{
    memset(ref, 0, sizeof(*ref));
    ref->object_kind = (evoke_segment_object_kind)
        evoke_accelerator_directory_read_u32(bytes + 0);
    ref->start_block = evoke_accelerator_directory_read_u32(bytes + 8);
    ref->page_count = evoke_accelerator_directory_read_u32(bytes + 12);
    ref->object_id = evoke_accelerator_directory_read_u64(bytes + 16);
    ref->owner_manifest_id =
        evoke_accelerator_directory_read_u64(bytes + 24);
    ref->object_bytes = evoke_accelerator_directory_read_u64(bytes + 32);
    ref->object_checksum = evoke_accelerator_directory_read_u64(bytes + 40);
}

evoke_status
evoke_semantic_accelerator_forward_directory_deserialize(
    const evoke_semantic_accelerator_directory_summary *summary,
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_accelerator_directory *directory_out
)
{
    evoke_semantic_accelerator_directory directory;
    size_t forward_bytes;
    size_t forward_term_work_bytes;
    size_t forward_chunk_cost_bytes;
    size_t forward_row_offset_bytes;
    size_t forward_term_bytes;
    size_t forward_bound_term_bytes;
    size_t forward_bound_ref_bytes;
    size_t expected_size;

    if (summary == NULL || bytes == NULL || directory_out == NULL ||
        !evoke_semantic_accelerator_directory_summary_has_complete_forward(
            summary))
    {
        return EVOKE_ERR_INVALID;
    }
    forward_bytes = (size_t) summary->forward_chunk_count *
        EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE;
    forward_term_work_bytes =
        (size_t) summary->forward_term_work_count * sizeof(uint64_t);
    forward_chunk_cost_bytes =
        (size_t) summary->forward_chunk_cost_count *
        2U * sizeof(uint64_t);
    if (summary->forward_row_offset_count > SIZE_MAX / sizeof(uint32_t))
    {
        return EVOKE_ERR_RANGE;
    }
    forward_row_offset_bytes =
        (size_t) summary->forward_row_offset_count * sizeof(uint32_t);
    forward_term_bytes =
        (size_t) summary->forward_term_bytes_count * sizeof(uint64_t);
    forward_bound_term_bytes =
        (size_t) summary->forward_bound_term_bytes_count * sizeof(uint64_t);
    forward_bound_ref_bytes =
        (size_t) summary->forward_bound_shard_count *
        EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE;
    expected_size = forward_bytes + forward_term_work_bytes +
        forward_chunk_cost_bytes + forward_row_offset_bytes +
        forward_term_bytes + forward_bound_term_bytes +
        forward_bound_ref_bytes;
    if (size != expected_size)
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_semantic_accelerator_directory_init(&directory);
    directory.source_manifest_id = summary->source_manifest_id;
    directory.source_authority_checksum =
        summary->source_authority_checksum;
    directory.owner_manifest_id = summary->owner_manifest_id;
    directory.document_count = summary->document_count;
    directory.vocab_size = summary->vocab_size;
    directory.term_count = summary->term_count;
    directory.forward_chunk_count = summary->forward_chunk_count;
    directory.forward_document_shift = summary->forward_document_shift;
    directory.builder_policy_id = summary->builder_policy_id;
    directory.retained_document_cap = summary->retained_document_cap;
    directory.forward_bound_shard_count =
        summary->forward_bound_shard_count;
    directory.scope_object = summary->scope_object;
    directory.tid_lookup_object = summary->tid_lookup_object;
    directory.forward_chunks = calloc(
        directory.forward_chunk_count,
        sizeof(*directory.forward_chunks)
    );
    directory.forward_term_work = calloc(
        directory.vocab_size,
        sizeof(*directory.forward_term_work)
    );
    directory.forward_row_data_bytes = calloc(
        directory.forward_chunk_count,
        sizeof(*directory.forward_row_data_bytes)
    );
    directory.forward_transpose_fixed_bytes = calloc(
        directory.forward_chunk_count,
        sizeof(*directory.forward_transpose_fixed_bytes)
    );
    directory.forward_row_offsets = calloc(
        (size_t) summary->forward_row_offset_count,
        sizeof(*directory.forward_row_offsets)
    );
    directory.forward_term_bytes = calloc(
        directory.vocab_size,
        sizeof(*directory.forward_term_bytes)
    );
    directory.forward_bound_term_bytes = calloc(
        directory.vocab_size,
        sizeof(*directory.forward_bound_term_bytes)
    );
    if (directory.forward_bound_shard_count > 0)
    {
        directory.forward_bound_shards = calloc(
            directory.forward_bound_shard_count,
            sizeof(*directory.forward_bound_shards)
        );
    }
    if (directory.forward_chunks == NULL ||
        directory.forward_term_work == NULL ||
        directory.forward_row_data_bytes == NULL ||
        directory.forward_transpose_fixed_bytes == NULL ||
        directory.forward_row_offsets == NULL ||
        directory.forward_term_bytes == NULL ||
        directory.forward_bound_term_bytes == NULL ||
        (directory.forward_bound_shard_count > 0 &&
         directory.forward_bound_shards == NULL))
    {
        evoke_semantic_accelerator_directory_free(&directory);
        return EVOKE_ERR_NOMEM;
    }
    for (uint32_t index = 0;
         index < directory.forward_chunk_count;
         index++)
    {
        const uint8_t *entry = bytes +
            (size_t) index * EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE;
        evoke_semantic_accelerator_forward_entry *forward =
            &directory.forward_chunks[index];
        uint32_t expected_first = index == 0
            ? 0
            : directory.forward_chunks[index - 1U].first_document +
                directory.forward_chunks[index - 1U].document_count;

        forward->first_document =
            evoke_accelerator_directory_read_u32(entry + 0);
        forward->document_count =
            evoke_accelerator_directory_read_u32(entry + 4);
        forward->posting_count =
            evoke_accelerator_directory_read_u32(entry + 8);
        forward->row_data_offset =
            evoke_accelerator_directory_read_u32(entry + 12);
        evoke_accelerator_directory_read_ref(
            entry + 16,
            &forward->forward_object
        );
        if (forward->first_document != expected_first ||
            forward->document_count == 0 ||
            forward->document_count > directory.document_count ||
            forward->first_document >
                directory.document_count - forward->document_count ||
            forward->row_data_offset == 0 ||
            forward->forward_object.object_kind !=
                EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD ||
            forward->forward_object.object_id != (uint64_t) index + 1U ||
            forward->forward_object.owner_manifest_id !=
                directory.owner_manifest_id ||
            evoke_segment_object_ref_validate(
                &forward->forward_object,
                UINT32_MAX) != EVOKE_OK)
        {
            evoke_semantic_accelerator_directory_free(&directory);
            return EVOKE_ERR_FORMAT;
        }
    }
    for (uint64_t offset_index = 0;
         offset_index < summary->forward_row_offset_count;
         offset_index++)
    {
        directory.forward_row_offsets[offset_index] =
            evoke_accelerator_directory_read_u32(
                bytes + forward_bytes + forward_term_work_bytes +
                    forward_chunk_cost_bytes +
                    (size_t) offset_index * sizeof(uint32_t)
            );
    }
    if (directory.forward_chunks[
            directory.forward_chunk_count - 1U
        ].first_document + directory.forward_chunks[
            directory.forward_chunk_count - 1U
        ].document_count != directory.document_count)
    {
        evoke_semantic_accelerator_directory_free(&directory);
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t term_id = 0;
         term_id < directory.vocab_size;
         term_id++)
    {
        directory.forward_term_work[term_id] =
            evoke_accelerator_directory_read_u64(
                bytes + forward_bytes +
                    (size_t) term_id * sizeof(uint64_t)
            );
    }
    for (uint32_t chunk = 0;
         chunk < directory.forward_chunk_count;
         chunk++)
    {
        size_t offset = forward_bytes + forward_term_work_bytes +
            (size_t) chunk * 2U * sizeof(uint64_t);

        directory.forward_row_data_bytes[chunk] =
            evoke_accelerator_directory_read_u64(
                bytes + offset
            );
        directory.forward_transpose_fixed_bytes[chunk] =
            evoke_accelerator_directory_read_u64(
                bytes + offset + sizeof(uint64_t)
            );
        {
            const evoke_semantic_accelerator_forward_entry *forward =
                &directory.forward_chunks[chunk];
            uint64_t row_offset_index =
                (uint64_t) forward->first_document + chunk;
            uint32_t previous =
                directory.forward_row_offsets[row_offset_index];

            if (previous != 0 ||
                forward->row_data_offset >
                    forward->forward_object.object_bytes ||
                directory.forward_row_data_bytes[chunk] >
                    forward->forward_object.object_bytes -
                        forward->row_data_offset)
            {
                evoke_semantic_accelerator_directory_free(&directory);
                return EVOKE_ERR_FORMAT;
            }
            for (uint32_t row = 0; row < forward->document_count; row++)
            {
                uint32_t next = directory.forward_row_offsets[
                    row_offset_index + row + 1U
                ];

                if (next < previous)
                {
                    evoke_semantic_accelerator_directory_free(&directory);
                    return EVOKE_ERR_FORMAT;
                }
                previous = next;
            }
            if (previous != directory.forward_row_data_bytes[chunk])
            {
                evoke_semantic_accelerator_directory_free(&directory);
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    for (uint32_t term_id = 0;
         term_id < directory.vocab_size;
         term_id++)
    {
        size_t offset = forward_bytes + forward_term_work_bytes +
            forward_chunk_cost_bytes + forward_row_offset_bytes +
            (size_t) term_id * sizeof(uint64_t);

        directory.forward_term_bytes[term_id] =
            evoke_accelerator_directory_read_u64(bytes + offset);
        directory.forward_bound_term_bytes[term_id] =
            evoke_accelerator_directory_read_u64(
                bytes + offset + forward_term_bytes
            );
    }
    for (uint32_t index = 0;
         index < directory.forward_bound_shard_count;
         index++)
    {
        const uint8_t *entry = bytes + forward_bytes +
            forward_term_work_bytes + forward_chunk_cost_bytes +
            forward_row_offset_bytes + forward_term_bytes +
            forward_bound_term_bytes +
            (size_t) index * EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE;
        evoke_segment_object_ref *bound =
            &directory.forward_bound_shards[index];

        evoke_accelerator_directory_read_ref(entry, bound);
        if (bound->object_kind !=
                EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD_BOUND ||
            bound->object_id != (uint64_t) index + 1U ||
            bound->owner_manifest_id != directory.owner_manifest_id ||
            evoke_segment_object_ref_validate(bound, UINT32_MAX) != EVOKE_OK)
        {
            evoke_semantic_accelerator_directory_free(&directory);
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t previous = 0; previous < index; previous++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    bound,
                    &directory.forward_bound_shards[previous]))
            {
                evoke_semantic_accelerator_directory_free(&directory);
                return EVOKE_ERR_FORMAT;
            }
        }
        for (uint32_t forward = 0;
             forward < directory.forward_chunk_count;
             forward++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    bound,
                    &directory.forward_chunks[forward].forward_object))
            {
                evoke_semantic_accelerator_directory_free(&directory);
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    evoke_semantic_accelerator_directory_free(directory_out);
    *directory_out = directory;
    return EVOKE_OK;
}

static bool
evoke_accelerator_directory_refs_overlap(
    const evoke_segment_object_ref *left,
    const evoke_segment_object_ref *right
)
{
    uint64_t left_end = (uint64_t) left->start_block + left->page_count;
    uint64_t right_end = (uint64_t) right->start_block + right->page_count;

    return (uint64_t) left->start_block < right_end &&
        (uint64_t) right->start_block < left_end;
}

static bool
evoke_accelerator_directory_ref_is_zero(
    const evoke_segment_object_ref *ref
)
{
    static const evoke_segment_object_ref zero_ref = {0};

    return ref != NULL && memcmp(ref, &zero_ref, sizeof(*ref)) == 0;
}

void
evoke_semantic_accelerator_directory_init(
    evoke_semantic_accelerator_directory *directory
)
{
    if (directory != NULL)
    {
        memset(directory, 0, sizeof(*directory));
    }
}

void
evoke_semantic_accelerator_directory_free(
    evoke_semantic_accelerator_directory *directory
)
{
    if (directory == NULL)
    {
        return;
    }
    free(directory->terms);
    free(directory->forward_chunks);
    free(directory->forward_term_work);
    free(directory->forward_row_data_bytes);
    free(directory->forward_transpose_fixed_bytes);
    free(directory->forward_row_offsets);
    free(directory->forward_term_bytes);
    free(directory->forward_bound_term_bytes);
    free(directory->forward_bound_shards);
    memset(directory, 0, sizeof(*directory));
}

evoke_status
evoke_semantic_accelerator_directory_validate(
    const evoke_semantic_accelerator_directory *directory
)
{
    uint32_t index;

    if (directory == NULL || directory->source_manifest_id == 0 ||
        directory->source_authority_checksum == 0 ||
        directory->owner_manifest_id <= directory->source_manifest_id ||
        directory->document_count == 0 || directory->vocab_size == 0 ||
        directory->term_count == 0 || directory->terms == NULL ||
        ((directory->builder_policy_id == 0) !=
         (directory->retained_document_cap == 0)) ||
        (directory->forward_chunk_count > 0 &&
         (directory->forward_chunks == NULL ||
          directory->forward_document_shift == 0 ||
          directory->forward_document_shift >= 32)) ||
        ((directory->forward_row_offsets == NULL) !=
         (directory->forward_row_data_bytes == NULL)) ||
        (directory->forward_bound_shard_count == 0 &&
         directory->forward_bound_shards != NULL) ||
        (directory->forward_bound_shard_count > 0 &&
         (directory->forward_bound_shards == NULL ||
          directory->forward_bound_shard_count !=
            evoke_accelerator_directory_expected_forward_bound_shards(
                directory->vocab_size
            ))))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (!evoke_accelerator_directory_ref_is_zero(
            &directory->scope_object) &&
        (directory->scope_object.object_kind !=
            EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_SCOPE ||
         directory->scope_object.object_id != 1 ||
         directory->scope_object.owner_manifest_id !=
            directory->owner_manifest_id ||
         evoke_segment_object_ref_validate(
            &directory->scope_object,
            UINT32_MAX) != EVOKE_OK))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (!evoke_accelerator_directory_ref_is_zero(
            &directory->tid_lookup_object) &&
        (directory->tid_lookup_object.object_kind !=
            EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_TID_LOOKUP ||
         directory->tid_lookup_object.object_id != 1 ||
         directory->tid_lookup_object.owner_manifest_id !=
            directory->owner_manifest_id ||
         evoke_segment_object_ref_validate(
            &directory->tid_lookup_object,
            UINT32_MAX) != EVOKE_OK))
    {
        return EVOKE_ERR_FORMAT;
    }
    for (index = 0; index < directory->term_count; index++)
    {
        const evoke_semantic_accelerator_directory_entry *entry =
            &directory->terms[index];
        const evoke_segment_object_ref *ref = &entry->term_object;
        uint32_t other_index;

        if (entry->term_id >= directory->vocab_size ||
            (index > 0 &&
             directory->terms[index - 1].term_id >= entry->term_id) ||
            ref->object_kind !=
                EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_TERM ||
            ref->object_id != (uint64_t) entry->term_id + 1 ||
            ref->owner_manifest_id != directory->owner_manifest_id ||
            evoke_segment_object_ref_validate(ref, UINT32_MAX) != EVOKE_OK)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (other_index = 0; other_index < index; other_index++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    ref,
                    &directory->terms[other_index].term_object))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    for (index = 0; index < directory->forward_chunk_count; index++)
    {
        const evoke_semantic_accelerator_forward_entry *entry =
            &directory->forward_chunks[index];
        const evoke_segment_object_ref *ref = &entry->forward_object;
        uint32_t expected_first = index == 0
            ? 0
            : directory->forward_chunks[index - 1U].first_document +
                directory->forward_chunks[index - 1U].document_count;

        if (entry->first_document != expected_first ||
            entry->document_count == 0 ||
            entry->document_count > directory->document_count ||
            entry->first_document >
                directory->document_count - entry->document_count ||
            (directory->forward_row_offsets != NULL
                ? entry->row_data_offset == 0
                : entry->row_data_offset != 0) ||
            ref->object_kind !=
                EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD ||
            ref->object_id != (uint64_t) index + 1U ||
            ref->owner_manifest_id != directory->owner_manifest_id ||
            evoke_segment_object_ref_validate(ref, UINT32_MAX) != EVOKE_OK)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (directory->forward_row_offsets != NULL)
        {
            uint64_t offset_index = (uint64_t) entry->first_document + index;
            uint32_t previous = directory->forward_row_offsets[offset_index];

            if (previous != 0)
            {
                return EVOKE_ERR_FORMAT;
            }
            for (uint32_t row = 0; row < entry->document_count; row++)
            {
                uint32_t next = directory->forward_row_offsets[
                    offset_index + row + 1U
                ];

                if (next < previous)
                {
                    return EVOKE_ERR_FORMAT;
                }
                previous = next;
            }
            if (directory->forward_row_data_bytes != NULL &&
                previous != directory->forward_row_data_bytes[index])
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        for (uint32_t term_index = 0;
             term_index < directory->term_count;
             term_index++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    ref,
                    &directory->terms[term_index].term_object))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        if (!evoke_accelerator_directory_ref_is_zero(
                &directory->scope_object) &&
            evoke_accelerator_directory_refs_overlap(
                ref,
                &directory->scope_object))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (!evoke_accelerator_directory_ref_is_zero(
                &directory->tid_lookup_object) &&
            evoke_accelerator_directory_refs_overlap(
                ref,
                &directory->tid_lookup_object))
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t other = 0; other < index; other++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    ref,
                    &directory->forward_chunks[other].forward_object))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    if (!evoke_accelerator_directory_ref_is_zero(
            &directory->scope_object))
    {
        for (uint32_t term_index = 0;
             term_index < directory->term_count;
             term_index++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    &directory->scope_object,
                    &directory->terms[term_index].term_object))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    if (!evoke_accelerator_directory_ref_is_zero(
            &directory->tid_lookup_object))
    {
        for (uint32_t term_index = 0;
             term_index < directory->term_count;
             term_index++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    &directory->tid_lookup_object,
                    &directory->terms[term_index].term_object))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        for (uint32_t forward_index = 0;
             forward_index < directory->forward_chunk_count;
             forward_index++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    &directory->tid_lookup_object,
                    &directory->forward_chunks[
                        forward_index
                    ].forward_object))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        if (!evoke_accelerator_directory_ref_is_zero(
                &directory->scope_object) &&
            evoke_accelerator_directory_refs_overlap(
                &directory->tid_lookup_object,
                &directory->scope_object))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    for (index = 0;
         index < directory->forward_bound_shard_count;
         index++)
    {
        const evoke_segment_object_ref *ref =
            &directory->forward_bound_shards[index];

        if (ref->object_kind !=
                EVOKE_SEGMENT_OBJECT_SEMANTIC_ACCELERATOR_FORWARD_BOUND ||
            ref->object_id != (uint64_t) index + 1U ||
            ref->owner_manifest_id != directory->owner_manifest_id ||
            evoke_segment_object_ref_validate(ref, UINT32_MAX) != EVOKE_OK)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t other = 0; other < index; other++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    ref,
                    &directory->forward_bound_shards[other]))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        for (uint32_t term_index = 0;
             term_index < directory->term_count;
             term_index++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    ref,
                    &directory->terms[term_index].term_object))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        for (uint32_t forward_index = 0;
             forward_index < directory->forward_chunk_count;
             forward_index++)
        {
            if (evoke_accelerator_directory_refs_overlap(
                    ref,
                    &directory->forward_chunks[
                        forward_index
                    ].forward_object))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        if ((!evoke_accelerator_directory_ref_is_zero(
                 &directory->scope_object) &&
             evoke_accelerator_directory_refs_overlap(
                 ref,
                 &directory->scope_object)) ||
            (!evoke_accelerator_directory_ref_is_zero(
                 &directory->tid_lookup_object) &&
             evoke_accelerator_directory_refs_overlap(
                 ref,
                 &directory->tid_lookup_object)))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    if (directory->forward_chunk_count > 0)
    {
        const evoke_semantic_accelerator_forward_entry *last =
            &directory->forward_chunks[directory->forward_chunk_count - 1U];

        if (last->first_document + last->document_count !=
            directory->document_count)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    return EVOKE_OK;
}

const evoke_semantic_accelerator_directory_entry *
evoke_semantic_accelerator_directory_find(
    const evoke_semantic_accelerator_directory *directory,
    uint32_t term_id
)
{
    uint32_t low = 0;
    uint32_t high;

    if (directory == NULL || directory->terms == NULL)
    {
        return NULL;
    }
    high = directory->term_count;
    while (low < high)
    {
        uint32_t middle = low + (high - low) / 2;
        uint32_t candidate = directory->terms[middle].term_id;

        if (candidate < term_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return low < directory->term_count &&
        directory->terms[low].term_id == term_id
        ? &directory->terms[low]
        : NULL;
}

const evoke_semantic_accelerator_forward_entry *
evoke_semantic_accelerator_directory_find_forward(
    const evoke_semantic_accelerator_directory *directory,
    uint32_t document_id
)
{
    uint32_t low = 0;
    uint32_t high;

    if (directory == NULL || directory->forward_chunks == NULL ||
        document_id >= directory->document_count)
    {
        return NULL;
    }
    high = directory->forward_chunk_count;
    while (low < high)
    {
        uint32_t middle = low + (high - low) / 2U;
        const evoke_semantic_accelerator_forward_entry *entry =
            &directory->forward_chunks[middle];

        if (document_id < entry->first_document)
        {
            high = middle;
        }
        else if (document_id >= entry->first_document + entry->document_count)
        {
            low = middle + 1U;
        }
        else
        {
            return entry;
        }
    }
    return NULL;
}

const evoke_segment_object_ref *
evoke_semantic_accelerator_directory_find_forward_bound(
    const evoke_semantic_accelerator_directory *directory,
    uint32_t term_id
)
{
    uint32_t shard;

    if (!evoke_semantic_accelerator_directory_has_complete_forward_bounds(
            directory) ||
        term_id >= directory->vocab_size)
    {
        return NULL;
    }
    shard = term_id / EVOKE_SEMANTIC_FORWARD_BOUND_TERMS_PER_SHARD;
    return &directory->forward_bound_shards[shard];
}

bool
evoke_semantic_accelerator_directory_has_scope(
    const evoke_semantic_accelerator_directory *directory
)
{
    return directory != NULL &&
        directory->builder_policy_id ==
            EVOKE_SEMANTIC_ACCELERATOR_POLICY_SCOPE_FORWARD_INT8 &&
        !evoke_accelerator_directory_ref_is_zero(&directory->scope_object);
}

bool
evoke_semantic_accelerator_directory_summary_has_scope(
    const evoke_semantic_accelerator_directory_summary *summary
)
{
    return
        evoke_semantic_accelerator_directory_summary_format_is_current(
            summary
        ) &&
        summary->builder_policy_id ==
            EVOKE_SEMANTIC_ACCELERATOR_POLICY_SCOPE_FORWARD_INT8 &&
        !evoke_accelerator_directory_ref_is_zero(&summary->scope_object);
}

bool
evoke_semantic_accelerator_directory_has_tid_lookup(
    const evoke_semantic_accelerator_directory *directory
)
{
    return directory != NULL &&
        !evoke_accelerator_directory_ref_is_zero(
            &directory->tid_lookup_object
        );
}

bool
evoke_semantic_accelerator_directory_summary_has_tid_lookup(
    const evoke_semantic_accelerator_directory_summary *summary
)
{
    return
        evoke_semantic_accelerator_directory_summary_format_is_current(
            summary
        ) &&
        !evoke_accelerator_directory_ref_is_zero(
            &summary->tid_lookup_object
        );
}

bool
evoke_semantic_accelerator_directory_has_complete_forward(
    const evoke_semantic_accelerator_directory *directory
)
{
    return directory != NULL &&
        evoke_accelerator_directory_policy_is_queryable(
            directory->builder_policy_id) &&
        directory->retained_document_cap ==
            EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP &&
        directory->forward_term_work != NULL &&
        directory->forward_chunk_count > 0 &&
        evoke_semantic_accelerator_directory_validate(directory) == EVOKE_OK;
}

bool
evoke_semantic_accelerator_directory_has_complete_forward_bounds(
    const evoke_semantic_accelerator_directory *directory
)
{
    return directory != NULL &&
        directory->forward_bound_shard_count ==
            evoke_accelerator_directory_expected_forward_bound_shards(
                directory->vocab_size
            ) &&
        directory->forward_bound_shards != NULL;
}

bool
evoke_semantic_accelerator_directory_summary_has_complete_forward_bounds(
    const evoke_semantic_accelerator_directory_summary *summary
)
{
    return summary != NULL &&
        evoke_semantic_accelerator_directory_summary_format_is_current(
            summary
        ) &&
        summary->forward_bound_shard_count ==
            evoke_accelerator_directory_expected_forward_bound_shards(
                summary->vocab_size
            );
}

bool
evoke_semantic_accelerator_directory_is_current(
    const evoke_semantic_accelerator_directory *directory
)
{
    return directory != NULL &&
        directory->builder_policy_id ==
            EVOKE_SEMANTIC_ACCELERATOR_CURRENT_POLICY &&
        directory->retained_document_cap ==
            EVOKE_SEMANTIC_ACCELERATOR_RETAINED_DOCUMENT_CAP &&
        directory->forward_term_work != NULL &&
        directory->forward_row_data_bytes != NULL &&
        directory->forward_transpose_fixed_bytes != NULL &&
        directory->forward_row_offsets != NULL &&
        directory->forward_term_bytes != NULL &&
        directory->forward_bound_term_bytes != NULL &&
        evoke_semantic_accelerator_directory_has_complete_forward_bounds(
            directory
        ) &&
        evoke_semantic_accelerator_directory_has_tid_lookup(directory) &&
        evoke_semantic_accelerator_directory_validate(directory) == EVOKE_OK;
}

evoke_status
evoke_semantic_accelerator_directory_serialize(
    const evoke_semantic_accelerator_directory *directory,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes;
    size_t entry_bytes;
    size_t forward_bytes;
    size_t forward_term_work_bytes;
    size_t forward_chunk_cost_bytes;
    size_t forward_row_offset_bytes;
    size_t forward_term_bytes;
    size_t forward_bound_term_bytes;
    size_t forward_bound_ref_bytes;
    size_t total_size;
    uint32_t index;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_semantic_accelerator_directory_validate(directory);
    if (status != EVOKE_OK ||
        !evoke_semantic_accelerator_directory_is_current(directory))
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }
    entry_bytes = (size_t) directory->term_count *
        EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE;
    forward_bytes = (size_t) directory->forward_chunk_count *
        EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE;
    forward_term_work_bytes =
        (size_t) directory->vocab_size * sizeof(uint64_t);
    forward_chunk_cost_bytes =
        (size_t) directory->forward_chunk_count *
        2U * sizeof(uint64_t);
    if (evoke_accelerator_directory_expected_forward_row_offsets(
            directory->document_count,
            directory->forward_chunk_count
        ) > SIZE_MAX / sizeof(uint32_t))
    {
        return EVOKE_ERR_RANGE;
    }
    forward_row_offset_bytes =
        (size_t) evoke_accelerator_directory_expected_forward_row_offsets(
            directory->document_count,
            directory->forward_chunk_count
        ) * sizeof(uint32_t);
    forward_term_bytes =
        (size_t) directory->vocab_size * sizeof(uint64_t);
    forward_bound_term_bytes =
        (size_t) directory->vocab_size * sizeof(uint64_t);
    forward_bound_ref_bytes =
        (size_t) directory->forward_bound_shard_count *
        EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE;
    if ((directory->term_count != 0 &&
         entry_bytes / EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE !=
             directory->term_count) ||
        (directory->forward_chunk_count != 0 &&
         forward_bytes / EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE !=
             directory->forward_chunk_count) ||
        (directory->forward_chunk_count != 0 &&
         forward_chunk_cost_bytes / (2U * sizeof(uint64_t)) !=
             directory->forward_chunk_count) ||
        (forward_row_offset_bytes != 0 &&
         forward_row_offset_bytes / sizeof(uint32_t) !=
            evoke_accelerator_directory_expected_forward_row_offsets(
                directory->document_count,
                directory->forward_chunk_count
            )) ||
        (directory->vocab_size != 0 &&
         (forward_term_work_bytes / sizeof(uint64_t) !=
              directory->vocab_size ||
          forward_term_bytes / sizeof(uint64_t) !=
              directory->vocab_size ||
          forward_bound_term_bytes / sizeof(uint64_t) !=
              directory->vocab_size)) ||
        (directory->forward_bound_shard_count != 0 &&
         forward_bound_ref_bytes /
                EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE !=
             directory->forward_bound_shard_count) ||
        entry_bytes >
            SIZE_MAX - EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE ||
        forward_bytes >
            SIZE_MAX - EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE -
            entry_bytes ||
        forward_term_work_bytes >
            SIZE_MAX - EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE -
            entry_bytes - forward_bytes ||
        forward_chunk_cost_bytes >
            SIZE_MAX - EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE -
                entry_bytes - forward_bytes - forward_term_work_bytes ||
        forward_row_offset_bytes >
            SIZE_MAX - EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE -
                entry_bytes - forward_bytes - forward_term_work_bytes -
                forward_chunk_cost_bytes ||
        forward_term_bytes >
            SIZE_MAX - EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE -
                entry_bytes - forward_bytes - forward_term_work_bytes -
                forward_chunk_cost_bytes - forward_row_offset_bytes ||
        forward_bound_term_bytes >
            SIZE_MAX - EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE -
                entry_bytes - forward_bytes - forward_term_work_bytes -
                forward_chunk_cost_bytes - forward_row_offset_bytes -
                forward_term_bytes ||
        forward_bound_ref_bytes >
            SIZE_MAX - EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE -
                entry_bytes - forward_bytes - forward_term_work_bytes -
                forward_chunk_cost_bytes - forward_row_offset_bytes -
                forward_term_bytes -
                forward_bound_term_bytes)
    {
        return EVOKE_ERR_RANGE;
    }
    total_size = EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
        entry_bytes + forward_bytes + forward_term_work_bytes +
        forward_chunk_cost_bytes + forward_row_offset_bytes +
        forward_term_bytes +
        forward_bound_term_bytes +
        forward_bound_ref_bytes;
    bytes = calloc(total_size, 1);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    evoke_accelerator_directory_write_u32(
        bytes + 0,
        EVOKE_ACCELERATOR_DIRECTORY_MAGIC
    );
    evoke_accelerator_directory_write_u16(
        bytes + 4,
        EVOKE_ACCELERATOR_DIRECTORY_VERSION
    );
    evoke_accelerator_directory_write_u16(
        bytes + 6,
        EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE
    );
    evoke_accelerator_directory_write_u16(
        bytes + 8,
        EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE
    );
    evoke_accelerator_directory_write_u16(
        bytes + 10,
        EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE
    );
    evoke_accelerator_directory_write_u32(
        bytes + 12,
        directory->document_count
    );
    evoke_accelerator_directory_write_u32(bytes + 16, directory->vocab_size);
    evoke_accelerator_directory_write_u32(bytes + 20, directory->term_count);
    evoke_accelerator_directory_write_u32(
        bytes + 24,
        directory->forward_chunk_count
    );
    evoke_accelerator_directory_write_u32(
        bytes + 28,
        directory->forward_document_shift
    );
    evoke_accelerator_directory_write_u64(
        bytes + 32,
        directory->source_manifest_id
    );
    evoke_accelerator_directory_write_u64(
        bytes + 40,
        directory->source_authority_checksum
    );
    evoke_accelerator_directory_write_u64(
        bytes + 48,
        directory->owner_manifest_id
    );
    evoke_accelerator_directory_write_u64(bytes + 56, total_size);
    evoke_accelerator_directory_write_u32(
        bytes + 72,
        directory->builder_policy_id
    );
    evoke_accelerator_directory_write_u32(
        bytes + 76,
        directory->retained_document_cap
    );
    evoke_accelerator_directory_write_ref(
        bytes + 80,
        &directory->scope_object
    );
    evoke_accelerator_directory_write_ref(
        bytes + EVOKE_ACCELERATOR_DIRECTORY_SCOPE_HEADER_SIZE,
        &directory->tid_lookup_object
    );
    for (index = 0; index < directory->term_count; index++)
    {
        uint8_t *entry = bytes +
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
            (size_t) index * EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE;

        evoke_accelerator_directory_write_u32(
            entry + 0,
            directory->terms[index].term_id
        );
        evoke_accelerator_directory_write_ref(
            entry + 8,
            &directory->terms[index].term_object
        );
    }
    for (index = 0; index < directory->forward_chunk_count; index++)
    {
        uint8_t *entry = bytes +
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
            entry_bytes +
            (size_t) index * EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE;

        evoke_accelerator_directory_write_u32(
            entry + 0,
            directory->forward_chunks[index].first_document
        );
        evoke_accelerator_directory_write_u32(
            entry + 4,
            directory->forward_chunks[index].document_count
        );
        evoke_accelerator_directory_write_u32(
            entry + 8,
            directory->forward_chunks[index].posting_count
        );
        evoke_accelerator_directory_write_u32(
            entry + 12,
            directory->forward_chunks[index].row_data_offset
        );
        evoke_accelerator_directory_write_ref(
            entry + 16,
            &directory->forward_chunks[index].forward_object
        );
    }
    for (index = 0; index < directory->vocab_size; index++)
    {
        size_t metrics_offset =
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
            entry_bytes + forward_bytes + forward_term_work_bytes +
            forward_chunk_cost_bytes + forward_row_offset_bytes;

        evoke_accelerator_directory_write_u64(
            bytes + EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
                entry_bytes + forward_bytes +
                (size_t) index * sizeof(uint64_t),
            directory->forward_term_work[index]
        );

        evoke_accelerator_directory_write_u64(
            bytes + metrics_offset +
                (size_t) index * sizeof(uint64_t),
            directory->forward_term_bytes[index]
        );
        evoke_accelerator_directory_write_u64(
            bytes + metrics_offset + forward_term_bytes +
                (size_t) index * sizeof(uint64_t),
            directory->forward_bound_term_bytes[index]
        );
    }
    for (uint64_t offset_index = 0;
         offset_index <
            evoke_accelerator_directory_expected_forward_row_offsets(
                directory->document_count,
                directory->forward_chunk_count
            );
         offset_index++)
    {
        evoke_accelerator_directory_write_u32(
            bytes + EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
                entry_bytes + forward_bytes + forward_term_work_bytes +
                forward_chunk_cost_bytes +
                (size_t) offset_index * sizeof(uint32_t),
            directory->forward_row_offsets[offset_index]
        );
    }
    for (index = 0; index < directory->forward_chunk_count; index++)
    {
        size_t cost_offset =
            EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
            entry_bytes + forward_bytes + forward_term_work_bytes +
            (size_t) index * 2U * sizeof(uint64_t);

        evoke_accelerator_directory_write_u64(
            bytes + cost_offset,
            directory->forward_row_data_bytes[index]
        );
        evoke_accelerator_directory_write_u64(
            bytes + cost_offset + sizeof(uint64_t),
            directory->forward_transpose_fixed_bytes[index]
        );
    }
    for (index = 0;
         index < directory->forward_bound_shard_count;
         index++)
    {
        evoke_accelerator_directory_write_ref(
            bytes + EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE +
                entry_bytes + forward_bytes + forward_term_work_bytes +
                forward_chunk_cost_bytes + forward_row_offset_bytes +
                forward_term_bytes +
                forward_bound_term_bytes +
                (size_t) index *
                    EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE,
            &directory->forward_bound_shards[index]
        );
    }
    evoke_accelerator_directory_write_u64(
        bytes + EVOKE_ACCELERATOR_DIRECTORY_CHECKSUM_OFFSET,
        evoke_accelerator_directory_checksum(bytes, total_size)
    );
    *bytes_out = bytes;
    *size_out = total_size;
    return EVOKE_OK;
}

static evoke_status
evoke_semantic_accelerator_directory_deserialize_internal(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_accelerator_directory *directory_out,
    bool allow_retirement
)
{
    evoke_semantic_accelerator_directory directory;
    uint64_t total_size;
    size_t entry_bytes;
    size_t forward_bytes;
    size_t forward_term_work_bytes;
    size_t forward_chunk_cost_bytes;
    size_t forward_row_offset_bytes;
    size_t forward_term_bytes;
    size_t forward_bound_term_bytes;
    size_t forward_bound_ref_bytes;
    uint16_t version;
    uint16_t header_size;
    uint16_t forward_entry_size;
    uint32_t index;
    evoke_status status;

    if (bytes == NULL || directory_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (size < EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    version = evoke_accelerator_directory_read_u16(bytes + 4);
    header_size = evoke_accelerator_directory_read_u16(bytes + 6);
    forward_entry_size = evoke_accelerator_directory_read_u16(bytes + 10);
    if (evoke_accelerator_directory_read_u32(bytes + 0) !=
            EVOKE_ACCELERATOR_DIRECTORY_MAGIC ||
        (version != EVOKE_ACCELERATOR_DIRECTORY_VERSION &&
         (!allow_retirement ||
          version < EVOKE_ACCELERATOR_DIRECTORY_RETIREMENT_MIN_VERSION ||
          version > EVOKE_ACCELERATOR_DIRECTORY_RETIREMENT_MAX_VERSION)) ||
        header_size != EVOKE_SEMANTIC_ACCELERATOR_DIRECTORY_HEADER_SIZE ||
        size < header_size ||
        evoke_accelerator_directory_read_u16(bytes + 8) !=
            EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE ||
        forward_entry_size !=
            (version == EVOKE_ACCELERATOR_DIRECTORY_RETIREMENT_MIN_VERSION
                ? EVOKE_ACCELERATOR_RETIREMENT_V5_FORWARD_ENTRY_SIZE
                : EVOKE_ACCELERATOR_FORWARD_ENTRY_SIZE) ||
        evoke_accelerator_directory_read_u64(
            bytes + EVOKE_ACCELERATOR_DIRECTORY_CHECKSUM_OFFSET) !=
            evoke_accelerator_directory_checksum(bytes, size))
    {
        return EVOKE_ERR_FORMAT;
    }
    total_size = evoke_accelerator_directory_read_u64(bytes + 56);
    evoke_semantic_accelerator_directory_init(&directory);
    directory.document_count =
        evoke_accelerator_directory_read_u32(bytes + 12);
    directory.vocab_size = evoke_accelerator_directory_read_u32(bytes + 16);
    directory.term_count = evoke_accelerator_directory_read_u32(bytes + 20);
    directory.forward_chunk_count =
        evoke_accelerator_directory_read_u32(bytes + 24);
    directory.forward_document_shift =
        evoke_accelerator_directory_read_u32(bytes + 28);
    directory.source_manifest_id =
        evoke_accelerator_directory_read_u64(bytes + 32);
    directory.source_authority_checksum =
        evoke_accelerator_directory_read_u64(bytes + 40);
    directory.owner_manifest_id =
        evoke_accelerator_directory_read_u64(bytes + 48);
    directory.builder_policy_id =
        evoke_accelerator_directory_read_u32(bytes + 72);
    directory.retained_document_cap =
        evoke_accelerator_directory_read_u32(bytes + 76);
    evoke_accelerator_directory_read_ref(bytes + 80, &directory.scope_object);
    evoke_accelerator_directory_read_ref(
        bytes + EVOKE_ACCELERATOR_DIRECTORY_SCOPE_HEADER_SIZE,
        &directory.tid_lookup_object
    );
    entry_bytes = (size_t) directory.term_count *
        EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE;
    forward_bytes = (size_t) directory.forward_chunk_count *
        forward_entry_size;
    forward_term_work_bytes =
        version >= UINT16_C(7)
            ? (size_t) directory.vocab_size * sizeof(uint64_t)
            : 0;
    forward_chunk_cost_bytes =
        version >= EVOKE_ACCELERATOR_DIRECTORY_PHYSICAL_COST_VERSION
            ? (size_t) directory.forward_chunk_count *
                2U * sizeof(uint64_t)
            : 0;
    if (version == EVOKE_ACCELERATOR_DIRECTORY_VERSION &&
        evoke_accelerator_directory_expected_forward_row_offsets(
            directory.document_count,
            directory.forward_chunk_count
        ) > SIZE_MAX / sizeof(uint32_t))
    {
        return EVOKE_ERR_RANGE;
    }
    forward_row_offset_bytes =
        version == EVOKE_ACCELERATOR_DIRECTORY_VERSION
            ? (size_t)
                evoke_accelerator_directory_expected_forward_row_offsets(
                    directory.document_count,
                    directory.forward_chunk_count
                ) * sizeof(uint32_t)
            : 0;
    forward_term_bytes =
        version >= EVOKE_ACCELERATOR_DIRECTORY_PHYSICAL_COST_VERSION
            ? (size_t) directory.vocab_size * sizeof(uint64_t)
            : 0;
    forward_bound_term_bytes = forward_term_bytes;
    directory.forward_bound_shard_count =
        version >= UINT16_C(8)
            ? evoke_accelerator_directory_expected_forward_bound_shards(
                  directory.vocab_size
              )
            : 0;
    forward_bound_ref_bytes =
        (size_t) directory.forward_bound_shard_count *
        EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE;
    if (total_size != size ||
        directory.term_count >
            (size - header_size) /
                EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE ||
        directory.forward_chunk_count >
            (size - header_size) /
                forward_entry_size ||
        entry_bytes > SIZE_MAX - header_size ||
        forward_bytes > SIZE_MAX - header_size - entry_bytes ||
        forward_term_work_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes ||
        forward_chunk_cost_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes -
                forward_term_work_bytes ||
        forward_row_offset_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes -
                forward_term_work_bytes - forward_chunk_cost_bytes ||
        forward_term_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes -
                forward_term_work_bytes -
                forward_chunk_cost_bytes - forward_row_offset_bytes ||
        forward_bound_term_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes -
                forward_term_work_bytes -
                forward_chunk_cost_bytes - forward_row_offset_bytes -
                forward_term_bytes ||
        forward_bound_ref_bytes >
            SIZE_MAX - header_size - entry_bytes - forward_bytes -
                forward_term_work_bytes -
                forward_chunk_cost_bytes - forward_row_offset_bytes -
                forward_term_bytes -
                forward_bound_term_bytes ||
        header_size + entry_bytes + forward_bytes +
            forward_term_work_bytes + forward_chunk_cost_bytes +
            forward_row_offset_bytes + forward_term_bytes +
            forward_bound_term_bytes +
            forward_bound_ref_bytes != size)
    {
        return EVOKE_ERR_FORMAT;
    }
    directory.terms = calloc(directory.term_count, sizeof(*directory.terms));
    if (directory.terms == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    if (directory.forward_chunk_count > 0)
    {
        directory.forward_chunks = calloc(
            directory.forward_chunk_count,
            sizeof(*directory.forward_chunks)
        );
        if (directory.forward_chunks == NULL)
        {
            evoke_semantic_accelerator_directory_free(&directory);
            return EVOKE_ERR_NOMEM;
        }
    }
    for (index = 0; index < directory.term_count; index++)
    {
        const uint8_t *entry =
            bytes + header_size +
            (size_t) index * EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE;

        if (evoke_accelerator_directory_read_u32(entry + 4) != 0)
        {
            evoke_semantic_accelerator_directory_free(&directory);
            return EVOKE_ERR_FORMAT;
        }
        directory.terms[index].term_id =
            evoke_accelerator_directory_read_u32(entry + 0);
        evoke_accelerator_directory_read_ref(
            entry + 8,
            &directory.terms[index].term_object
        );
    }
    for (index = 0; index < directory.forward_chunk_count; index++)
    {
        const uint8_t *entry =
            bytes + header_size +
            (size_t) directory.term_count *
                EVOKE_ACCELERATOR_DIRECTORY_ENTRY_SIZE +
            (size_t) index * forward_entry_size;

        directory.forward_chunks[index].first_document =
            evoke_accelerator_directory_read_u32(entry + 0);
        directory.forward_chunks[index].document_count =
            evoke_accelerator_directory_read_u32(entry + 4);
        if (version != EVOKE_ACCELERATOR_DIRECTORY_RETIREMENT_MIN_VERSION)
        {
            directory.forward_chunks[index].posting_count =
                evoke_accelerator_directory_read_u32(entry + 8);
            directory.forward_chunks[index].row_data_offset =
                version == EVOKE_ACCELERATOR_DIRECTORY_VERSION
                    ? evoke_accelerator_directory_read_u32(entry + 12)
                    : 0;
            if (version != EVOKE_ACCELERATOR_DIRECTORY_VERSION &&
                evoke_accelerator_directory_read_u32(entry + 12) != 0)
            {
                evoke_semantic_accelerator_directory_free(&directory);
                return EVOKE_ERR_FORMAT;
            }
            evoke_accelerator_directory_read_ref(
                entry + 16,
                &directory.forward_chunks[index].forward_object
            );
        }
        else
        {
            directory.forward_chunks[index].posting_count = 0;
            evoke_accelerator_directory_read_ref(
                entry + 8,
                &directory.forward_chunks[index].forward_object
            );
        }
    }
    if (version >= UINT16_C(7))
    {
        directory.forward_term_work = calloc(
            directory.vocab_size,
            sizeof(*directory.forward_term_work)
        );
        if (directory.forward_term_work == NULL)
        {
            evoke_semantic_accelerator_directory_free(&directory);
            return EVOKE_ERR_NOMEM;
        }
        for (index = 0; index < directory.vocab_size; index++)
        {
            directory.forward_term_work[index] =
                evoke_accelerator_directory_read_u64(
                    bytes + header_size + entry_bytes + forward_bytes +
                        (size_t) index * sizeof(uint64_t)
                );
        }
    }
    if (version == EVOKE_ACCELERATOR_DIRECTORY_VERSION)
    {
        directory.forward_row_data_bytes = calloc(
            directory.forward_chunk_count,
            sizeof(*directory.forward_row_data_bytes)
        );
        directory.forward_transpose_fixed_bytes = calloc(
            directory.forward_chunk_count,
            sizeof(*directory.forward_transpose_fixed_bytes)
        );
        directory.forward_row_offsets = calloc(
            (size_t)
                evoke_accelerator_directory_expected_forward_row_offsets(
                    directory.document_count,
                    directory.forward_chunk_count
                ),
            sizeof(*directory.forward_row_offsets)
        );
        directory.forward_term_bytes = calloc(
            directory.vocab_size,
            sizeof(*directory.forward_term_bytes)
        );
        directory.forward_bound_term_bytes = calloc(
            directory.vocab_size,
            sizeof(*directory.forward_bound_term_bytes)
        );
        if (directory.forward_row_data_bytes == NULL ||
            directory.forward_transpose_fixed_bytes == NULL ||
            directory.forward_row_offsets == NULL ||
            directory.forward_term_bytes == NULL ||
            directory.forward_bound_term_bytes == NULL)
        {
            evoke_semantic_accelerator_directory_free(&directory);
            return EVOKE_ERR_NOMEM;
        }
        for (uint64_t offset_index = 0;
             offset_index <
                evoke_accelerator_directory_expected_forward_row_offsets(
                    directory.document_count,
                    directory.forward_chunk_count
                );
             offset_index++)
        {
            directory.forward_row_offsets[offset_index] =
                evoke_accelerator_directory_read_u32(
                    bytes + header_size + entry_bytes + forward_bytes +
                        forward_term_work_bytes +
                        forward_chunk_cost_bytes +
                        (size_t) offset_index * sizeof(uint32_t)
                );
        }
        for (index = 0; index < directory.forward_chunk_count; index++)
        {
            size_t offset = header_size + entry_bytes + forward_bytes +
                forward_term_work_bytes +
                (size_t) index * 2U * sizeof(uint64_t);

            directory.forward_row_data_bytes[index] =
                evoke_accelerator_directory_read_u64(bytes + offset);
            directory.forward_transpose_fixed_bytes[index] =
                evoke_accelerator_directory_read_u64(
                    bytes + offset + sizeof(uint64_t)
                );
        }
        for (index = 0; index < directory.vocab_size; index++)
        {
            size_t offset = header_size + entry_bytes + forward_bytes +
                forward_term_work_bytes + forward_chunk_cost_bytes +
                forward_row_offset_bytes +
                (size_t) index * sizeof(uint64_t);

            directory.forward_term_bytes[index] =
                evoke_accelerator_directory_read_u64(
                    bytes + offset
                );
            directory.forward_bound_term_bytes[index] =
                evoke_accelerator_directory_read_u64(
                    bytes + offset + forward_term_bytes
                );
        }
    }
    if (directory.forward_bound_shard_count > 0)
    {
        directory.forward_bound_shards = calloc(
            directory.forward_bound_shard_count,
            sizeof(*directory.forward_bound_shards)
        );
        if (directory.forward_bound_shards == NULL)
        {
            evoke_semantic_accelerator_directory_free(&directory);
            return EVOKE_ERR_NOMEM;
        }
        for (index = 0;
             index < directory.forward_bound_shard_count;
             index++)
        {
            evoke_accelerator_directory_read_ref(
                bytes + header_size + entry_bytes + forward_bytes +
                    forward_term_work_bytes +
                    forward_chunk_cost_bytes + forward_row_offset_bytes +
                    forward_term_bytes +
                    forward_bound_term_bytes +
                    (size_t) index *
                        EVOKE_ACCELERATOR_FORWARD_BOUND_REF_SIZE,
                &directory.forward_bound_shards[index]
            );
        }
    }
    status = evoke_semantic_accelerator_directory_validate(&directory);
    if (status != EVOKE_OK)
    {
        evoke_semantic_accelerator_directory_free(&directory);
        return status;
    }
    evoke_semantic_accelerator_directory_free(directory_out);
    *directory_out = directory;
    return EVOKE_OK;
}

evoke_status
evoke_semantic_accelerator_directory_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_accelerator_directory *directory_out
)
{
    return evoke_semantic_accelerator_directory_deserialize_internal(
        bytes,
        size,
        directory_out,
        false
    );
}

evoke_status
evoke_semantic_accelerator_directory_deserialize_retired(
    const uint8_t *bytes,
    size_t size,
    evoke_semantic_accelerator_directory *directory_out
)
{
    return evoke_semantic_accelerator_directory_deserialize_internal(
        bytes,
        size,
        directory_out,
        true
    );
}
