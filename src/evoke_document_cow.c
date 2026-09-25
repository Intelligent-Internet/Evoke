#include "evoke_document_cow.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define EVOKE_DOCUMENT_COW_MAGIC UINT32_C(0x44434932)
#define EVOKE_DOCUMENT_COW_VERSION UINT16_C(5)
#define EVOKE_DOCUMENT_COW_HEADER_SIZE 136U
#define EVOKE_DOCUMENT_COW_REF_SIZE 128U
#define EVOKE_DOCUMENT_COW_CHILD_SIZE 136U
#define EVOKE_DOCUMENT_COW_RECORD_SIZE 176U
#define EVOKE_DOCUMENT_COW_CHECKSUM_OFFSET 48U

#define EVOKE_DOCUMENT_COW_KNOWN_VERSION_FLAGS                         \
    (EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING |                    \
     EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE |                   \
     EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_QUARANTINED |                \
     EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID |                          \
     EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE |                        \
     EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED)

#define EVOKE_DOCUMENT_COW_KNOWN_RETIREMENT_FLAGS                      \
    EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID

#define EVOKE_DOCUMENT_COW_KNOWN_SEMANTIC_FLAGS                        \
    (EVOKE_SEMANTIC_STATE_FLAG_COMPLETE |                              \
     EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED |                           \
     EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID)

typedef struct evoke_document_cow_summary
{
    uint64_t first_document_slot;
    uint64_t document_slot_count;
    uint64_t live_document_count;
    uint64_t semantic_pending_count;
    int64_t earliest_retry_after;
    uint64_t bounded_document_count;
    uint32_t min_document_length;
    uint32_t max_document_length;
    uint64_t reusable_document_count;
    uint64_t first_reusable_document_slot;
    uint64_t min_live_born_sequence;
} evoke_document_cow_summary;

typedef struct evoke_document_cow_patch_context
{
    const evoke_document_cow_ref *old_root;
    uint64_t old_document_slot_count;
    uint64_t next_document_slot_count;
    const evoke_document_cow_record *updates;
    size_t update_count;
    uint64_t owner_manifest_id;
    evoke_document_cow_object_loader loader;
    void *loader_context;
    evoke_document_cow_tree *patch;
    evoke_document_cow_update_stats *stats;
} evoke_document_cow_patch_context;

static evoke_status evoke_document_cow_object_serialized_size(
    const evoke_document_cow_object *object,
    size_t *size_out
);

static void
evoke_document_cow_write_u16(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t) value;
    bytes[1] = (uint8_t) (value >> 8);
}

static void
evoke_document_cow_write_u32(uint8_t *bytes, uint32_t value)
{
    bytes[0] = (uint8_t) value;
    bytes[1] = (uint8_t) (value >> 8);
    bytes[2] = (uint8_t) (value >> 16);
    bytes[3] = (uint8_t) (value >> 24);
}

static void
evoke_document_cow_write_u64(uint8_t *bytes, uint64_t value)
{
    for (size_t index = 0; index < sizeof(value); index++)
    {
        bytes[index] = (uint8_t) (value >> (index * 8));
    }
}

static uint16_t
evoke_document_cow_read_u16(const uint8_t *bytes)
{
    return (uint16_t) bytes[0] |
        ((uint16_t) bytes[1] << 8);
}

static uint32_t
evoke_document_cow_read_u32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0] |
        ((uint32_t) bytes[1] << 8) |
        ((uint32_t) bytes[2] << 16) |
        ((uint32_t) bytes[3] << 24);
}

static uint64_t
evoke_document_cow_read_u64(const uint8_t *bytes)
{
    uint64_t value = 0;

    for (size_t index = 0; index < sizeof(value); index++)
    {
        value |= (uint64_t) bytes[index] << (index * 8);
    }
    return value;
}

static bool
evoke_document_cow_ref_is_bound(const evoke_document_cow_ref *ref)
{
    return ref->start_block != 0 &&
        ref->start_block != UINT32_MAX &&
        ref->page_count != 0 &&
        ref->object_bytes != 0 &&
        ref->checksum != 0;
}

static bool
evoke_document_cow_refs_equal(
    const evoke_document_cow_ref *left,
    const evoke_document_cow_ref *right
)
{
    return left->kind == right->kind &&
        left->start_block == right->start_block &&
        left->page_count == right->page_count &&
        left->reserved == right->reserved &&
        left->object_id == right->object_id &&
        left->owner_manifest_id == right->owner_manifest_id &&
        left->object_bytes == right->object_bytes &&
        left->checksum == right->checksum &&
        left->first_document_slot == right->first_document_slot &&
        left->document_slot_count == right->document_slot_count &&
        left->live_document_count == right->live_document_count &&
        left->semantic_pending_count == right->semantic_pending_count &&
        left->earliest_retry_after == right->earliest_retry_after &&
        left->bounded_document_count ==
            right->bounded_document_count &&
        left->min_document_length == right->min_document_length &&
        left->max_document_length == right->max_document_length &&
        left->reusable_document_count ==
            right->reusable_document_count &&
        left->first_reusable_document_slot ==
            right->first_reusable_document_slot &&
        left->min_live_born_sequence ==
            right->min_live_born_sequence;
}

bool
evoke_document_cow_versions_equal(
    const evoke_document_version_record *left,
    const evoke_document_version_record *right
)
{
    return left->document_slot == right->document_slot &&
        left->born_sequence == right->born_sequence &&
        left->record_xid == right->record_xid &&
        left->heap_block == right->heap_block &&
        left->document_length == right->document_length &&
        left->heap_offset == right->heap_offset &&
        left->flags == right->flags &&
        memcmp(
            left->semantic_input_fingerprint,
            right->semantic_input_fingerprint,
            EVOKE_DOCUMENT_FINGERPRINT_BYTES) == 0;
}

static bool
evoke_document_cow_retirement_equal(
    const evoke_document_retirement_record *left,
    const evoke_document_retirement_record *right
)
{
    return left->document_slot == right->document_slot &&
        left->retirement_sequence == right->retirement_sequence &&
        left->record_xid == right->record_xid &&
        left->document_length == right->document_length &&
        left->flags == right->flags &&
        left->reserved == right->reserved &&
        left->reserved2 == right->reserved2;
}

static bool
evoke_document_cow_semantic_state_equal(
    const evoke_semantic_state_record *left,
    const evoke_semantic_state_record *right
)
{
    return left->document_slot == right->document_slot &&
        left->transition_sequence == right->transition_sequence &&
        left->record_xid == right->record_xid &&
        left->error_code == right->error_code &&
        left->retry_after == right->retry_after &&
        left->pending_since == right->pending_since &&
        left->error_hash == right->error_hash &&
        left->flags == right->flags &&
        left->failure_count == right->failure_count &&
        left->reserved == right->reserved &&
        memcmp(
            left->semantic_input_fingerprint,
            right->semantic_input_fingerprint,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES) == 0;
}

bool
evoke_document_cow_records_equal(
    const evoke_document_cow_record *left,
    const evoke_document_cow_record *right
)
{
    return left != NULL && right != NULL &&
        evoke_document_cow_versions_equal(
            &left->version,
            &right->version) &&
        evoke_document_cow_retirement_equal(
            &left->retirement,
            &right->retirement) &&
        evoke_document_cow_semantic_state_equal(
            &left->semantic_state,
            &right->semantic_state) &&
        left->lexical_residency == right->lexical_residency &&
        left->semantic_residency == right->semantic_residency &&
        left->event_residency == right->event_residency;
}

static bool
evoke_document_cow_record_is_aborted(
    const evoke_document_cow_record *record
)
{
    return (record->version.flags &
            EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE) != 0;
}

bool
evoke_document_cow_record_is_l0_owned(
    const evoke_document_cow_record *record
)
{
    return record != NULL &&
        (record->version.flags &
         EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED) != 0;
}

static bool
evoke_document_cow_record_is_retired(
    const evoke_document_cow_record *record
)
{
    return record->retirement.retirement_sequence != 0;
}

static bool
evoke_document_cow_record_is_reusable(
    const evoke_document_cow_record *record
)
{
    return (evoke_document_cow_record_is_aborted(record) ||
            evoke_document_cow_record_is_retired(record)) &&
        !evoke_document_cow_record_is_l0_owned(record) &&
        record->lexical_residency == 0 &&
        record->semantic_residency == 0 &&
        record->event_residency == 0;
}

uint64_t
evoke_document_cow_record_last_sequence(
    const evoke_document_cow_record *record
)
{
    uint64_t sequence = record->version.born_sequence;

    if (record->retirement.retirement_sequence > sequence)
    {
        sequence = record->retirement.retirement_sequence;
    }
    if (record->semantic_state.transition_sequence > sequence)
    {
        sequence = record->semantic_state.transition_sequence;
    }
    return sequence;
}

static bool
evoke_document_cow_record_is_semantic_complete(
    const evoke_document_cow_record *record
)
{
    return
        (record->version.flags &
         EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE) != 0 ||
        (record->semantic_state.flags &
         EVOKE_SEMANTIC_STATE_FLAG_COMPLETE) != 0;
}

static bool
evoke_document_cow_semantic_transition_allowed(
    const evoke_document_cow_record *old_record,
    const evoke_document_cow_record *next_record
)
{
    const evoke_semantic_state_record *old_semantic =
        &old_record->semantic_state;
    const evoke_semantic_state_record *next_semantic =
        &next_record->semantic_state;

    if (old_semantic->transition_sequence == 0)
    {
        return next_semantic->transition_sequence == 0 ||
            !evoke_document_cow_record_is_semantic_complete(old_record);
    }
    if (evoke_document_cow_semantic_state_equal(
            old_semantic,
            next_semantic))
    {
        return true;
    }
    return (old_semantic->flags &
            EVOKE_SEMANTIC_STATE_FLAG_COMPLETE) == 0 &&
        next_semantic->transition_sequence >
            old_semantic->transition_sequence;
}

static bool
evoke_document_cow_record_is_actionable(
    const evoke_document_cow_record *record,
    int64_t now
)
{
    if ((record->version.flags &
         EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING) == 0 ||
        evoke_document_cow_record_is_aborted(record) ||
        evoke_document_cow_record_is_retired(record) ||
        evoke_document_cow_record_is_semantic_complete(record))
    {
        return false;
    }
    if ((record->semantic_state.flags &
         EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED) == 0)
    {
        return true;
    }
    return record->semantic_state.retry_after != INT64_MAX &&
        record->semantic_state.retry_after <= now;
}

static int64_t
evoke_document_cow_record_retry_after(
    const evoke_document_cow_record *record
)
{
    if ((record->version.flags &
         EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING) == 0 ||
        evoke_document_cow_record_is_aborted(record) ||
        evoke_document_cow_record_is_retired(record) ||
        evoke_document_cow_record_is_semantic_complete(record))
    {
        return INT64_MAX;
    }
    if ((record->semantic_state.flags &
         EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED) == 0)
    {
        return 0;
    }
    return record->semantic_state.retry_after;
}

static evoke_status
evoke_document_cow_record_validate(
    const evoke_document_cow_record *record,
    uint64_t expected_slot
)
{
    const evoke_document_version_record *version;
    const evoke_document_retirement_record *retirement;
    const evoke_semantic_state_record *semantic;
    uint16_t semantic_version_flags;
    uint16_t semantic_state_flags;

    if (record == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    version = &record->version;
    retirement = &record->retirement;
    semantic = &record->semantic_state;
    semantic_version_flags = version->flags &
        (EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING |
         EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE |
         EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_QUARANTINED);
    semantic_state_flags = semantic->flags &
        (EVOKE_SEMANTIC_STATE_FLAG_COMPLETE |
         EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED);

    if (version->document_slot != expected_slot ||
        version->born_sequence == 0 ||
        (version->flags & ~EVOKE_DOCUMENT_COW_KNOWN_VERSION_FLAGS) != 0 ||
        (version->flags &
         EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID) == 0 ||
        (semantic_version_flags != 0 &&
         semantic_version_flags !=
            EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_PENDING &&
         semantic_version_flags !=
            EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_COMPLETE &&
         semantic_version_flags !=
            EVOKE_DOCUMENT_VERSION_FLAG_SEMANTIC_QUARANTINED))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (evoke_document_cow_record_is_aborted(record))
    {
        static const uint8_t zero_fingerprint[
            EVOKE_DOCUMENT_FINGERPRINT_BYTES
        ] = {0};

        uint16_t expected_flags =
            EVOKE_DOCUMENT_VERSION_FLAG_ABORTED_HOLE |
            EVOKE_DOCUMENT_VERSION_FLAG_FROZEN_XID;

        if (evoke_document_cow_record_is_l0_owned(record))
        {
            expected_flags |= EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED;
        }
        if (version->flags != expected_flags ||
            version->record_xid != 0 ||
            version->heap_block != 0 ||
            version->heap_offset != 0 ||
            version->document_length != 0 ||
            semantic_version_flags != 0 ||
            memcmp(
                version->semantic_input_fingerprint,
                zero_fingerprint,
                sizeof(zero_fingerprint)) != 0 ||
            record->lexical_residency != 0 ||
            record->semantic_residency != 0 ||
            retirement->retirement_sequence != 0 ||
            semantic->transition_sequence != 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        return EVOKE_OK;
    }
    if (version->record_xid != 0 ||
        version->heap_offset == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (semantic_version_flags != 0 &&
        evoke_document_fingerprint_is_zero(
            version->semantic_input_fingerprint))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (evoke_document_cow_record_is_l0_owned(record) &&
        (record->lexical_residency != 0 ||
         record->semantic_residency != 0 ||
         record->event_residency != 0 ||
         semantic->transition_sequence != 0))
    {
        return EVOKE_ERR_FORMAT;
    }

    if (retirement->retirement_sequence != 0)
    {
        if (retirement->document_slot != expected_slot ||
            retirement->retirement_sequence <= version->born_sequence ||
            retirement->record_xid != 0 ||
            retirement->document_length != version->document_length ||
            (retirement->flags &
             ~EVOKE_DOCUMENT_COW_KNOWN_RETIREMENT_FLAGS) != 0 ||
            (retirement->flags &
             EVOKE_DOCUMENT_RETIREMENT_FLAG_FROZEN_XID) == 0 ||
            retirement->reserved != 0 ||
            retirement->reserved2 != 0)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    else if (retirement->document_slot != 0 ||
             retirement->record_xid != 0 ||
             retirement->document_length != 0 ||
             retirement->flags != 0 ||
             retirement->reserved != 0 ||
             retirement->reserved2 != 0)
    {
        return EVOKE_ERR_FORMAT;
    }

    if (semantic->transition_sequence != 0)
    {
        if (semantic->document_slot != expected_slot ||
            semantic->transition_sequence <= version->born_sequence ||
            semantic->record_xid != 0 ||
            semantic->reserved != 0 ||
            (semantic->flags &
             ~EVOKE_DOCUMENT_COW_KNOWN_SEMANTIC_FLAGS) != 0 ||
            (semantic->flags &
             EVOKE_SEMANTIC_STATE_FLAG_FROZEN_XID) == 0 ||
            (semantic_state_flags != EVOKE_SEMANTIC_STATE_FLAG_COMPLETE &&
             semantic_state_flags !=
                EVOKE_SEMANTIC_STATE_FLAG_QUARANTINED) ||
            memcmp(
                semantic->semantic_input_fingerprint,
                version->semantic_input_fingerprint,
                EVOKE_DOCUMENT_FINGERPRINT_BYTES) != 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (semantic_state_flags == EVOKE_SEMANTIC_STATE_FLAG_COMPLETE)
        {
            if (semantic->error_code != 0 ||
                semantic->retry_after != 0 ||
                semantic->pending_since != 0 ||
                semantic->error_hash != 0 ||
                semantic->failure_count != 0)
            {
                return EVOKE_ERR_FORMAT;
            }
        }
        else if (semantic->error_code == 0 ||
                 semantic->pending_since <= 0 ||
                 semantic->retry_after < semantic->pending_since ||
                 semantic->failure_count == 0)
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    else if (semantic->document_slot != 0 ||
             semantic->record_xid != 0 ||
             semantic->error_code != 0 ||
             semantic->retry_after != 0 ||
             semantic->pending_since != 0 ||
             semantic->error_hash != 0 ||
             semantic->flags != 0 ||
             semantic->failure_count != 0 ||
             semantic->reserved != 0 ||
             !evoke_document_fingerprint_is_zero(
                 semantic->semantic_input_fingerprint))
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

static evoke_status
evoke_document_cow_record_transition_validate(
    const evoke_document_cow_record *old_record,
    const evoke_document_cow_record *next_record
)
{
    const evoke_document_retirement_record *old_retirement;
    const evoke_document_retirement_record *next_retirement;

    if (old_record == NULL || next_record == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (evoke_document_cow_record_is_l0_owned(old_record) &&
        evoke_document_cow_record_is_l0_owned(next_record) &&
        evoke_document_cow_record_is_aborted(old_record) &&
        !evoke_document_cow_record_is_aborted(next_record))
    {
        return next_record->version.document_slot ==
                    old_record->version.document_slot &&
            next_record->version.born_sequence ==
                old_record->version.born_sequence &&
            next_record->retirement.retirement_sequence >
                next_record->version.born_sequence &&
            next_record->lexical_residency == 0 &&
            next_record->semantic_residency == 0 &&
            next_record->event_residency == 0 &&
            next_record->semantic_state.transition_sequence == 0
            ? EVOKE_OK
            : EVOKE_ERR_FORMAT;
    }
    if (evoke_document_cow_record_is_l0_owned(old_record) &&
        !evoke_document_cow_record_is_l0_owned(next_record))
    {
        evoke_document_version_record old_version = old_record->version;

        old_version.flags &= ~EVOKE_DOCUMENT_VERSION_FLAG_L0_OWNED;
        if (next_record->version.document_slot !=
                old_record->version.document_slot ||
            next_record->version.born_sequence !=
                old_record->version.born_sequence ||
            next_record->event_residency == 0 ||
            !evoke_document_cow_retirement_equal(
                &old_record->retirement,
                &next_record->retirement) ||
            !evoke_document_cow_semantic_transition_allowed(
                old_record,
                next_record))
        {
            return EVOKE_ERR_FORMAT;
        }
        if (evoke_document_cow_record_is_aborted(old_record))
        {
            return EVOKE_OK;
        }
        return evoke_document_cow_versions_equal(
            &old_version,
            &next_record->version)
            ? EVOKE_OK
            : EVOKE_ERR_FORMAT;
    }
    if (!evoke_document_cow_versions_equal(
            &old_record->version,
            &next_record->version))
    {
        bool l0_retired_transition =
            evoke_document_cow_record_is_l0_owned(next_record) &&
            next_record->retirement.retirement_sequence >
                next_record->version.born_sequence &&
            next_record->lexical_residency == 0 &&
            next_record->semantic_residency == 0 &&
            next_record->event_residency == 0;

        if (!evoke_document_cow_record_is_reusable(old_record) ||
            next_record->version.document_slot !=
                old_record->version.document_slot ||
            next_record->version.born_sequence <=
                evoke_document_cow_record_last_sequence(old_record) ||
            next_record->semantic_state.transition_sequence != 0 ||
            (!l0_retired_transition &&
             (next_record->retirement.retirement_sequence != 0 ||
              next_record->event_residency == 0)))
        {
            return EVOKE_ERR_FORMAT;
        }
        return EVOKE_OK;
    }
    old_retirement = &old_record->retirement;
    next_retirement = &next_record->retirement;
    if (old_retirement->retirement_sequence != 0 &&
        !evoke_document_cow_retirement_equal(
            old_retirement,
            next_retirement))
    {
        return EVOKE_ERR_FORMAT;
    }

    return evoke_document_cow_semantic_transition_allowed(
        old_record,
        next_record)
        ? EVOKE_OK
        : EVOKE_ERR_FORMAT;
}

static void
evoke_document_cow_summary_init(
    evoke_document_cow_summary *summary
)
{
    memset(summary, 0, sizeof(*summary));
    summary->earliest_retry_after = INT64_MAX;
    summary->first_reusable_document_slot = UINT64_MAX;
    summary->min_live_born_sequence = UINT64_MAX;
}

static void
evoke_document_cow_summary_add_record(
    evoke_document_cow_summary *summary,
    const evoke_document_cow_record *record
)
{
    int64_t retry_after =
        evoke_document_cow_record_retry_after(record);

    if (summary->document_slot_count == 0)
    {
        summary->first_document_slot = record->version.document_slot;
    }
    summary->document_slot_count++;
    if (!evoke_document_cow_record_is_aborted(record) &&
        !evoke_document_cow_record_is_retired(record))
    {
        summary->live_document_count++;
        if (record->version.born_sequence <
            summary->min_live_born_sequence)
        {
            summary->min_live_born_sequence =
                record->version.born_sequence;
        }
    }
    if (retry_after != INT64_MAX)
    {
        summary->semantic_pending_count++;
        if (retry_after < summary->earliest_retry_after)
        {
            summary->earliest_retry_after = retry_after;
        }
    }
    if (!evoke_document_cow_record_is_aborted(record))
    {
        uint32_t length = record->version.document_length;

        if (summary->bounded_document_count == 0)
        {
            summary->min_document_length = length;
            summary->max_document_length = length;
        }
        else
        {
            if (length < summary->min_document_length)
            {
                summary->min_document_length = length;
            }
            if (length > summary->max_document_length)
            {
                summary->max_document_length = length;
            }
        }
        summary->bounded_document_count++;
    }
    if (evoke_document_cow_record_is_reusable(record))
    {
        if (summary->reusable_document_count == 0)
        {
            summary->first_reusable_document_slot =
                record->version.document_slot;
        }
        summary->reusable_document_count++;
    }
}

static void
evoke_document_cow_summary_add_ref(
    evoke_document_cow_summary *summary,
    const evoke_document_cow_ref *ref
)
{
    if (summary->document_slot_count == 0)
    {
        summary->first_document_slot = ref->first_document_slot;
    }
    summary->document_slot_count += ref->document_slot_count;
    summary->live_document_count += ref->live_document_count;
    if (ref->min_live_born_sequence < summary->min_live_born_sequence)
    {
        summary->min_live_born_sequence = ref->min_live_born_sequence;
    }
    summary->semantic_pending_count += ref->semantic_pending_count;
    if (ref->earliest_retry_after < summary->earliest_retry_after)
    {
        summary->earliest_retry_after = ref->earliest_retry_after;
    }
    if (ref->bounded_document_count != 0)
    {
        if (summary->bounded_document_count == 0)
        {
            summary->min_document_length =
                ref->min_document_length;
            summary->max_document_length =
                ref->max_document_length;
        }
        else
        {
            if (ref->min_document_length <
                summary->min_document_length)
            {
                summary->min_document_length =
                    ref->min_document_length;
            }
            if (ref->max_document_length >
                summary->max_document_length)
            {
                summary->max_document_length =
                    ref->max_document_length;
            }
        }
        summary->bounded_document_count +=
            ref->bounded_document_count;
    }
    if (ref->reusable_document_count != 0)
    {
        if (summary->reusable_document_count == 0 ||
            ref->first_reusable_document_slot <
                summary->first_reusable_document_slot)
        {
            summary->first_reusable_document_slot =
                ref->first_reusable_document_slot;
        }
        summary->reusable_document_count +=
            ref->reusable_document_count;
    }
}

static void
evoke_document_cow_ref_set_summary(
    evoke_document_cow_ref *ref,
    const evoke_document_cow_summary *summary
)
{
    ref->first_document_slot = summary->first_document_slot;
    ref->document_slot_count = summary->document_slot_count;
    ref->live_document_count = summary->live_document_count;
    ref->semantic_pending_count = summary->semantic_pending_count;
    ref->earliest_retry_after = summary->earliest_retry_after;
    ref->bounded_document_count = summary->bounded_document_count;
    ref->min_document_length = summary->min_document_length;
    ref->max_document_length = summary->max_document_length;
    ref->reusable_document_count =
        summary->reusable_document_count;
    ref->first_reusable_document_slot =
        summary->first_reusable_document_slot;
    ref->min_live_born_sequence = summary->min_live_born_sequence;
}

static evoke_status
evoke_document_cow_ref_validate(
    const evoke_document_cow_ref *ref,
    bool require_storage
)
{
    if (ref == NULL ||
        (ref->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE &&
         ref->kind != EVOKE_DOCUMENT_COW_OBJECT_LEAF) ||
        ref->reserved != 0 ||
        ref->object_id == 0 ||
        ref->owner_manifest_id == 0 ||
        ref->document_slot_count > UINT32_MAX ||
        ref->live_document_count > ref->document_slot_count ||
        ref->bounded_document_count > ref->document_slot_count ||
        ref->reusable_document_count > ref->document_slot_count ||
        ref->live_document_count + ref->reusable_document_count >
            ref->document_slot_count ||
        ref->semantic_pending_count > ref->live_document_count ||
        (ref->semantic_pending_count == 0 &&
         ref->earliest_retry_after != INT64_MAX) ||
        (ref->semantic_pending_count != 0 &&
         ref->earliest_retry_after == INT64_MAX) ||
        (ref->bounded_document_count == 0 &&
         (ref->min_document_length != 0 ||
          ref->max_document_length != 0)) ||
        (ref->bounded_document_count != 0 &&
         ref->min_document_length > ref->max_document_length) ||
        (ref->reusable_document_count == 0 &&
         ref->first_reusable_document_slot != UINT64_MAX) ||
        (ref->reusable_document_count != 0 &&
         (ref->first_reusable_document_slot <
              ref->first_document_slot ||
          ref->first_document_slot >
              UINT64_MAX - ref->document_slot_count ||
          ref->first_reusable_document_slot >=
              ref->first_document_slot +
                  ref->document_slot_count)) ||
        (ref->live_document_count == 0 &&
         ref->min_live_born_sequence != UINT64_MAX) ||
        (ref->live_document_count != 0 &&
         (ref->min_live_born_sequence == 0 ||
          ref->min_live_born_sequence == UINT64_MAX)))
    {
        return EVOKE_ERR_FORMAT;
    }
    if ((ref->object_bytes == 0) != (ref->checksum == 0))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (require_storage && !evoke_document_cow_ref_is_bound(ref))
    {
        return EVOKE_ERR_FORMAT;
    }
    if ((ref->start_block == 0) != (ref->page_count == 0) ||
        (ref->start_block != 0 &&
         !evoke_document_cow_ref_is_bound(ref)))
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

void
evoke_document_cow_tree_init(evoke_document_cow_tree *tree)
{
    if (tree != NULL)
    {
        memset(tree, 0, sizeof(*tree));
    }
}

void
evoke_document_cow_tree_free(evoke_document_cow_tree *tree)
{
    if (tree == NULL)
    {
        return;
    }
    free(tree->objects);
    free(tree->retired_ranges);
    memset(tree, 0, sizeof(*tree));
}

static evoke_status
evoke_document_cow_tree_add_retired_ref(
    evoke_document_cow_tree *tree,
    const evoke_document_cow_ref *ref
)
{
    evoke_block_range *ranges;
    size_t next_capacity;

    if (tree == NULL || ref == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (!evoke_document_cow_ref_is_bound(ref))
    {
        return EVOKE_OK;
    }
    for (size_t index = 0; index < tree->retired_range_count; index++)
    {
        if (tree->retired_ranges[index].start_block == ref->start_block &&
            tree->retired_ranges[index].block_count == ref->page_count)
        {
            return EVOKE_OK;
        }
    }
    if (tree->retired_range_count == tree->retired_range_capacity)
    {
        next_capacity = tree->retired_range_capacity == 0
            ? 8
            : tree->retired_range_capacity * 2;
        if (next_capacity < tree->retired_range_capacity ||
            next_capacity > SIZE_MAX / sizeof(*tree->retired_ranges))
        {
            return EVOKE_ERR_RANGE;
        }
        ranges = realloc(
            tree->retired_ranges,
            next_capacity * sizeof(*tree->retired_ranges)
        );
        if (ranges == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        tree->retired_ranges = ranges;
        tree->retired_range_capacity = next_capacity;
    }
    tree->retired_ranges[tree->retired_range_count].start_block =
        ref->start_block;
    tree->retired_ranges[tree->retired_range_count].block_count =
        ref->page_count;
    tree->retired_range_count++;
    return EVOKE_OK;
}

static evoke_status
evoke_document_cow_tree_append_object(
    evoke_document_cow_tree *tree,
    const evoke_document_cow_object *source,
    evoke_document_cow_ref *ref_out
)
{
    evoke_document_cow_object object;
    evoke_document_cow_object *objects;
    size_t next_capacity;

    if (tree == NULL || source == NULL || ref_out == NULL ||
        tree->next_object_id == 0 ||
        tree->next_object_id == UINT64_MAX)
    {
        return EVOKE_ERR_INVALID;
    }
    if (tree->object_count == tree->object_capacity)
    {
        next_capacity = tree->object_capacity == 0
            ? 16
            : tree->object_capacity * 2;
        if (next_capacity < tree->object_capacity ||
            next_capacity > SIZE_MAX / sizeof(*tree->objects))
        {
            return EVOKE_ERR_RANGE;
        }
        objects = realloc(
            tree->objects,
            next_capacity * sizeof(*tree->objects)
        );
        if (objects == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        tree->objects = objects;
        tree->object_capacity = next_capacity;
    }

    object = *source;
    object.ref.object_id = tree->next_object_id++;
    object.ref.start_block = 0;
    object.ref.page_count = 0;
    object.ref.object_bytes = 0;
    object.ref.checksum = 0;
    tree->objects[tree->object_count++] = object;
    *ref_out = object.ref;
    return EVOKE_OK;
}

static evoke_document_cow_object *
evoke_document_cow_tree_find_object(
    evoke_document_cow_tree *tree,
    const evoke_document_cow_ref *ref
)
{
    evoke_document_cow_object *candidate;

    if (tree == NULL || ref == NULL ||
        ref->owner_manifest_id != tree->root.owner_manifest_id)
    {
        return NULL;
    }
    if (ref->object_id > 0 && ref->object_id <= tree->object_count)
    {
        candidate = &tree->objects[ref->object_id - 1];
        if (candidate->ref.object_id == ref->object_id &&
            candidate->ref.owner_manifest_id == ref->owner_manifest_id)
        {
            return candidate;
        }
    }
    for (size_t index = 0; index < tree->object_count; index++)
    {
        if (tree->objects[index].ref.object_id == ref->object_id &&
            tree->objects[index].ref.owner_manifest_id ==
                ref->owner_manifest_id)
        {
            return &tree->objects[index];
        }
    }
    return NULL;
}

static const evoke_document_cow_object *
evoke_document_cow_tree_find_const_object(
    const evoke_document_cow_tree *tree,
    const evoke_document_cow_ref *ref
)
{
    return evoke_document_cow_tree_find_object(
        (evoke_document_cow_tree *) tree,
        ref
    );
}

static evoke_status
evoke_document_cow_build_leaf(
    evoke_document_cow_tree *tree,
    const evoke_document_cow_record *records,
    size_t record_count,
    evoke_document_cow_ref *ref_out
)
{
    evoke_document_cow_object object;
    evoke_document_cow_summary summary;

    if (record_count == 0 ||
        record_count > EVOKE_DOCUMENT_COW_LEAF_RECORDS)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&object, 0, sizeof(object));
    object.ref.kind = EVOKE_DOCUMENT_COW_OBJECT_LEAF;
    object.ref.owner_manifest_id = tree->root.owner_manifest_id;
    object.value.leaf.base_document_slot =
        records[0].version.document_slot;
    object.value.leaf.record_count = (uint32_t) record_count;
    evoke_document_cow_summary_init(&summary);
    for (size_t index = 0; index < record_count; index++)
    {
        uint64_t expected_slot =
            object.value.leaf.base_document_slot + index;
        evoke_status status = evoke_document_cow_record_validate(
            &records[index],
            expected_slot
        );

        if (status != EVOKE_OK)
        {
            return status;
        }
        object.value.leaf.records[index] = records[index];
        evoke_document_cow_summary_add_record(
            &summary,
            &records[index]
        );
    }
    evoke_document_cow_ref_set_summary(&object.ref, &summary);
    return evoke_document_cow_tree_append_object(
        tree,
        &object,
        ref_out
    );
}

static uint32_t
evoke_document_cow_child_slot(uint64_t document_slot, uint16_t level)
{
    uint64_t leaf_index =
        document_slot / EVOKE_DOCUMENT_COW_LEAF_RECORDS;
    uint32_t shift = (uint32_t) level *
        EVOKE_DOCUMENT_COW_RADIX_BITS;

    return (uint32_t) (
        (leaf_index >> shift) &
        (EVOKE_DOCUMENT_COW_RADIX_FANOUT - 1)
    );
}

static evoke_status
evoke_document_cow_build_level(
    evoke_document_cow_tree *tree,
    uint16_t level,
    const evoke_document_cow_record *records,
    size_t record_count,
    evoke_document_cow_ref *ref_out
)
{
    evoke_document_cow_object object;
    evoke_document_cow_summary summary;
    size_t cursor = 0;

    memset(&object, 0, sizeof(object));
    object.ref.kind = EVOKE_DOCUMENT_COW_OBJECT_NODE;
    object.ref.owner_manifest_id = tree->root.owner_manifest_id;
    object.value.node.level = level;
    object.value.node.prefix =
        record_count == 0
            ? 0
            : records[0].version.document_slot /
                EVOKE_DOCUMENT_COW_LEAF_RECORDS >>
                ((uint32_t) (level + 1) *
                 EVOKE_DOCUMENT_COW_RADIX_BITS);
    evoke_document_cow_summary_init(&summary);

    while (cursor < record_count)
    {
        uint32_t slot = evoke_document_cow_child_slot(
            records[cursor].version.document_slot,
            level
        );
        size_t end = cursor + 1;
        evoke_document_cow_ref child_ref;
        evoke_status status;

        while (end < record_count &&
               evoke_document_cow_child_slot(
                   records[end].version.document_slot,
                   level) == slot)
        {
            end++;
        }
        if (object.value.node.child_count >=
            EVOKE_DOCUMENT_COW_RADIX_FANOUT)
        {
            return EVOKE_ERR_RANGE;
        }
        if (level == 0)
        {
            status = evoke_document_cow_build_leaf(
                tree,
                records + cursor,
                end - cursor,
                &child_ref
            );
        }
        else
        {
            status = evoke_document_cow_build_level(
                tree,
                level - 1,
                records + cursor,
                end - cursor,
                &child_ref
            );
        }
        if (status != EVOKE_OK)
        {
            return status;
        }
        object.value.node.children[
            object.value.node.child_count
        ].slot = (uint16_t) slot;
        object.value.node.children[
            object.value.node.child_count
        ].ref = child_ref;
        object.value.node.child_count++;
        evoke_document_cow_summary_add_ref(&summary, &child_ref);
        cursor = end;
    }
    evoke_document_cow_ref_set_summary(&object.ref, &summary);
    return evoke_document_cow_tree_append_object(
        tree,
        &object,
        ref_out
    );
}

evoke_status
evoke_document_cow_tree_build(
    const evoke_document_cow_record *records,
    uint64_t record_count,
    uint64_t owner_manifest_id,
    evoke_document_cow_tree *tree_out
)
{
    evoke_document_cow_tree tree;
    evoke_status status;

    if (tree_out == NULL || owner_manifest_id == 0 ||
        record_count > UINT32_MAX ||
        (record_count > 0 && records == NULL) ||
        record_count > SIZE_MAX / sizeof(*records))
    {
        return EVOKE_ERR_INVALID;
    }
    evoke_document_cow_tree_init(&tree);
    tree.document_slot_count = record_count;
    tree.next_object_id = 1;
    tree.root.owner_manifest_id = owner_manifest_id;
    status = evoke_document_cow_build_level(
        &tree,
        EVOKE_DOCUMENT_COW_RADIX_LEVELS - 1,
        records,
        (size_t) record_count,
        &tree.root
    );
    if (status != EVOKE_OK)
    {
        evoke_document_cow_tree_free(&tree);
        return status;
    }
    status = evoke_document_cow_tree_validate(&tree);
    if (status != EVOKE_OK)
    {
        evoke_document_cow_tree_free(&tree);
        return status;
    }
    evoke_document_cow_tree_free(tree_out);
    *tree_out = tree;
    return EVOKE_OK;
}

static evoke_status
evoke_document_cow_object_validate(
    const evoke_document_cow_object *object,
    bool require_storage
)
{
    evoke_document_cow_summary summary;

    if (object == NULL ||
        evoke_document_cow_ref_validate(
            &object->ref,
            require_storage) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_document_cow_summary_init(&summary);
    if (object->ref.kind == EVOKE_DOCUMENT_COW_OBJECT_LEAF)
    {
        const evoke_document_cow_leaf *leaf = &object->value.leaf;

        if (leaf->record_count == 0 ||
            leaf->record_count > EVOKE_DOCUMENT_COW_LEAF_RECORDS ||
            leaf->reserved != 0 ||
            leaf->base_document_slot %
                EVOKE_DOCUMENT_COW_LEAF_RECORDS != 0)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t index = 0; index < leaf->record_count; index++)
        {
            evoke_status status = evoke_document_cow_record_validate(
                &leaf->records[index],
                leaf->base_document_slot + index
            );

            if (status != EVOKE_OK)
            {
                return status;
            }
            evoke_document_cow_summary_add_record(
                &summary,
                &leaf->records[index]
            );
        }
    }
    else
    {
        const evoke_document_cow_node *node = &object->value.node;

        if (node->level >= EVOKE_DOCUMENT_COW_RADIX_LEVELS ||
            node->reserved != 0 ||
            node->child_count > EVOKE_DOCUMENT_COW_RADIX_FANOUT ||
            (node->level != EVOKE_DOCUMENT_COW_RADIX_LEVELS - 1 &&
             node->child_count == 0))
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t index = 0; index < node->child_count; index++)
        {
            const evoke_document_cow_child *child =
                &node->children[index];

            if (child->slot >= EVOKE_DOCUMENT_COW_RADIX_FANOUT ||
                child->reserved != 0 ||
                child->reserved2 != 0 ||
                (index > 0 &&
                 node->children[index - 1].slot >= child->slot) ||
                evoke_document_cow_ref_validate(
                    &child->ref,
                    require_storage) != EVOKE_OK ||
                (node->level == 0 &&
                 child->ref.kind != EVOKE_DOCUMENT_COW_OBJECT_LEAF) ||
                (node->level > 0 &&
                 child->ref.kind != EVOKE_DOCUMENT_COW_OBJECT_NODE))
            {
                return EVOKE_ERR_FORMAT;
            }
            evoke_document_cow_summary_add_ref(
                &summary,
                &child->ref
            );
        }
    }
    if (summary.first_document_slot !=
            object->ref.first_document_slot ||
        summary.document_slot_count !=
            object->ref.document_slot_count ||
        summary.live_document_count !=
            object->ref.live_document_count ||
        summary.semantic_pending_count !=
            object->ref.semantic_pending_count ||
        summary.earliest_retry_after !=
            object->ref.earliest_retry_after ||
        summary.bounded_document_count !=
            object->ref.bounded_document_count ||
        summary.min_document_length !=
            object->ref.min_document_length ||
        summary.max_document_length !=
            object->ref.max_document_length ||
        summary.reusable_document_count !=
            object->ref.reusable_document_count ||
        summary.first_reusable_document_slot !=
            object->ref.first_reusable_document_slot ||
        summary.min_live_born_sequence !=
            object->ref.min_live_born_sequence)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

static int
evoke_document_cow_compare_object_ids(
    const void *left_pointer,
    const void *right_pointer
)
{
    const uint64_t left = *(const uint64_t *) left_pointer;
    const uint64_t right = *(const uint64_t *) right_pointer;

    return (left > right) - (left < right);
}

evoke_status
evoke_document_cow_tree_validate(const evoke_document_cow_tree *tree)
{
    const evoke_document_cow_object *root_object;
    uint64_t *object_ids;

    if (tree == NULL ||
        tree->document_slot_count > UINT32_MAX ||
        tree->next_object_id == 0 ||
        tree->object_count == 0 ||
        tree->object_count > tree->object_capacity ||
        tree->root.kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        tree->root.owner_manifest_id == 0 ||
        tree->root.document_slot_count != tree->document_slot_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (tree->object_count > SIZE_MAX / sizeof(*object_ids))
    {
        return EVOKE_ERR_RANGE;
    }
    object_ids = malloc(tree->object_count * sizeof(*object_ids));
    if (object_ids == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    for (size_t index = 0; index < tree->object_count; index++)
    {
        const evoke_document_cow_object *object =
            &tree->objects[index];

        if (object->ref.object_id == 0 ||
            object->ref.object_id >= tree->next_object_id ||
            object->ref.owner_manifest_id !=
                tree->root.owner_manifest_id ||
            evoke_document_cow_object_validate(object, false) != EVOKE_OK)
        {
            free(object_ids);
            return EVOKE_ERR_FORMAT;
        }
        object_ids[index] = object->ref.object_id;
    }
    qsort(
        object_ids,
        tree->object_count,
        sizeof(*object_ids),
        evoke_document_cow_compare_object_ids
    );
    for (size_t index = 1; index < tree->object_count; index++)
    {
        if (object_ids[index - 1] == object_ids[index])
        {
            free(object_ids);
            return EVOKE_ERR_FORMAT;
        }
    }
    free(object_ids);
    root_object =
        evoke_document_cow_tree_find_const_object(tree, &tree->root);
    if (root_object == NULL ||
        root_object->value.node.level !=
            EVOKE_DOCUMENT_COW_RADIX_LEVELS - 1)
    {
        return EVOKE_ERR_FORMAT;
    }
    return EVOKE_OK;
}

typedef struct evoke_document_cow_external_walk
{
    uint64_t document_slot_count;
    uint64_t next_document_slot;
    uint64_t root_owner_manifest_id;
    size_t max_objects;
    evoke_document_cow_ref *visited;
    size_t visited_count;
    size_t visited_capacity;
    evoke_document_cow_object_loader loader;
    void *loader_context;
    evoke_document_cow_object_visitor visitor;
    void *visitor_context;
} evoke_document_cow_external_walk;

static int
evoke_document_cow_compare_ref_identity(
    const void *left_pointer,
    const void *right_pointer
)
{
    const evoke_document_cow_ref *left = left_pointer;
    const evoke_document_cow_ref *right = right_pointer;

    if (left->owner_manifest_id < right->owner_manifest_id)
    {
        return -1;
    }
    if (left->owner_manifest_id > right->owner_manifest_id)
    {
        return 1;
    }
    if (left->object_id < right->object_id)
    {
        return -1;
    }
    if (left->object_id > right->object_id)
    {
        return 1;
    }
    return 0;
}

static int
evoke_document_cow_compare_ref_storage(
    const void *left_pointer,
    const void *right_pointer
)
{
    const evoke_document_cow_ref *left = left_pointer;
    const evoke_document_cow_ref *right = right_pointer;

    if (left->start_block < right->start_block)
    {
        return -1;
    }
    if (left->start_block > right->start_block)
    {
        return 1;
    }
    if (left->page_count < right->page_count)
    {
        return -1;
    }
    if (left->page_count > right->page_count)
    {
        return 1;
    }
    return 0;
}

static bool
evoke_document_cow_identity_equal(
    const evoke_document_cow_ref *left,
    const evoke_document_cow_ref *right
)
{
    return left->owner_manifest_id == right->owner_manifest_id &&
        left->object_id == right->object_id;
}

static bool
evoke_document_cow_storage_ranges_overlap(
    const evoke_document_cow_ref *left,
    const evoke_document_cow_ref *right
)
{
    uint64_t left_end =
        (uint64_t) left->start_block + left->page_count;
    uint64_t right_end =
        (uint64_t) right->start_block + right->page_count;

    return (uint64_t) left->start_block < right_end &&
        (uint64_t) right->start_block < left_end;
}

static evoke_status
evoke_document_cow_external_validate_visited(
    evoke_document_cow_external_walk *walk
)
{
    if (walk == NULL || walk->visited_count == 0)
    {
        return EVOKE_ERR_FORMAT;
    }

    qsort(
        walk->visited,
        walk->visited_count,
        sizeof(*walk->visited),
        evoke_document_cow_compare_ref_identity
    );
    for (size_t index = 1; index < walk->visited_count; index++)
    {
        if (evoke_document_cow_identity_equal(
                &walk->visited[index - 1],
                &walk->visited[index]))
        {
            return EVOKE_ERR_FORMAT;
        }
    }

    qsort(
        walk->visited,
        walk->visited_count,
        sizeof(*walk->visited),
        evoke_document_cow_compare_ref_storage
    );
    for (size_t index = 1; index < walk->visited_count; index++)
    {
        if (evoke_document_cow_storage_ranges_overlap(
                &walk->visited[index - 1],
                &walk->visited[index]))
        {
            return EVOKE_ERR_FORMAT;
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_document_cow_external_visit_ref(
    evoke_document_cow_external_walk *walk,
    const evoke_document_cow_ref *ref
)
{
    evoke_document_cow_ref *resized;
    size_t next_capacity;

    if (walk->visited_count >= walk->max_objects ||
        ref->owner_manifest_id > walk->root_owner_manifest_id)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (walk->visited_count == walk->visited_capacity)
    {
        next_capacity = walk->visited_capacity == 0
            ? 64
            : walk->visited_capacity * 2;
        if (next_capacity < walk->visited_capacity ||
            next_capacity > walk->max_objects)
        {
            next_capacity = walk->max_objects;
        }
        if (next_capacity > SIZE_MAX / sizeof(*walk->visited))
        {
            return EVOKE_ERR_RANGE;
        }
        resized = realloc(
            walk->visited,
            next_capacity * sizeof(*walk->visited)
        );
        if (resized == NULL)
        {
            return EVOKE_ERR_NOMEM;
        }
        walk->visited = resized;
        walk->visited_capacity = next_capacity;
    }
    walk->visited[walk->visited_count++] = *ref;
    return EVOKE_OK;
}

static evoke_status
evoke_document_cow_external_walk_ref(
    evoke_document_cow_external_walk *walk,
    const evoke_document_cow_ref *ref,
    evoke_document_cow_object_kind expected_kind,
    uint16_t expected_level,
    uint64_t expected_prefix
)
{
    evoke_document_cow_object object;
    evoke_status status;

    status = evoke_document_cow_external_visit_ref(walk, ref);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = walk->loader(walk->loader_context, ref, &object);
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (!evoke_document_cow_refs_equal(&object.ref, ref) ||
        evoke_document_cow_object_validate(&object, true) != EVOKE_OK ||
        object.ref.kind != expected_kind)
    {
        return EVOKE_ERR_FORMAT;
    }

    if (expected_kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        const evoke_document_cow_node *node = &object.value.node;

        if (node->level != expected_level ||
            node->prefix != expected_prefix)
        {
            return EVOKE_ERR_FORMAT;
        }
        for (uint32_t index = 0; index < node->child_count; index++)
        {
            const evoke_document_cow_child *child =
                &node->children[index];
            uint64_t child_prefix;

            if (expected_prefix >
                (UINT64_MAX - child->slot) /
                    EVOKE_DOCUMENT_COW_RADIX_FANOUT)
            {
                return EVOKE_ERR_FORMAT;
            }
            child_prefix =
                expected_prefix * EVOKE_DOCUMENT_COW_RADIX_FANOUT +
                child->slot;
            status = evoke_document_cow_external_walk_ref(
                walk,
                &child->ref,
                expected_level == 0
                    ? EVOKE_DOCUMENT_COW_OBJECT_LEAF
                    : EVOKE_DOCUMENT_COW_OBJECT_NODE,
                expected_level == 0
                    ? 0
                    : (uint16_t) (expected_level - 1),
                child_prefix
            );
            if (status != EVOKE_OK)
            {
                return status;
            }
        }
    }
    else
    {
        const evoke_document_cow_leaf *leaf = &object.value.leaf;
        uint64_t base_document_slot;
        uint64_t expected_records;

        if (expected_prefix >
            UINT64_MAX / EVOKE_DOCUMENT_COW_LEAF_RECORDS)
        {
            return EVOKE_ERR_FORMAT;
        }
        base_document_slot =
            expected_prefix * EVOKE_DOCUMENT_COW_LEAF_RECORDS;
        if (base_document_slot >= walk->document_slot_count ||
            leaf->base_document_slot != base_document_slot ||
            base_document_slot != walk->next_document_slot)
        {
            return EVOKE_ERR_FORMAT;
        }
        expected_records =
            walk->document_slot_count - base_document_slot;
        if (expected_records > EVOKE_DOCUMENT_COW_LEAF_RECORDS)
        {
            expected_records = EVOKE_DOCUMENT_COW_LEAF_RECORDS;
        }
        if (leaf->record_count != expected_records)
        {
            return EVOKE_ERR_FORMAT;
        }
        walk->next_document_slot += expected_records;
    }
    if (walk->visitor != NULL)
    {
        return walk->visitor(walk->visitor_context, &object);
    }
    return EVOKE_OK;
}

static evoke_status
evoke_document_cow_walk_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_object_visitor visitor,
    void *visitor_context
)
{
    evoke_document_cow_external_walk walk;
    size_t leaf_count;
    evoke_status status;

    if (root == NULL || loader == NULL ||
        document_slot_count > UINT32_MAX ||
        root->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        root->document_slot_count != document_slot_count ||
        evoke_document_cow_ref_validate(root, true) != EVOKE_OK)
    {
        return EVOKE_ERR_INVALID;
    }
    leaf_count = (
        (size_t) document_slot_count +
        EVOKE_DOCUMENT_COW_LEAF_RECORDS - 1
    ) / EVOKE_DOCUMENT_COW_LEAF_RECORDS;
    if (leaf_count >
        (SIZE_MAX - 1) / (EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1))
    {
        return EVOKE_ERR_RANGE;
    }

    memset(&walk, 0, sizeof(walk));
    walk.document_slot_count = document_slot_count;
    walk.root_owner_manifest_id = root->owner_manifest_id;
    walk.max_objects =
        1 + leaf_count * (EVOKE_DOCUMENT_COW_RADIX_LEVELS + 1);
    walk.loader = loader;
    walk.loader_context = loader_context;
    walk.visitor = visitor;
    walk.visitor_context = visitor_context;
    status = evoke_document_cow_external_walk_ref(
        &walk,
        root,
        EVOKE_DOCUMENT_COW_OBJECT_NODE,
        EVOKE_DOCUMENT_COW_RADIX_LEVELS - 1,
        0
    );
    if (status == EVOKE_OK &&
        walk.next_document_slot != document_slot_count)
    {
        status = EVOKE_ERR_FORMAT;
    }
    if (status == EVOKE_OK)
    {
        status = evoke_document_cow_external_validate_visited(&walk);
    }
    free(walk.visited);
    return status;
}

evoke_status
evoke_document_cow_validate_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context
)
{
    return evoke_document_cow_walk_external(
        root,
        document_slot_count,
        loader,
        loader_context,
        NULL,
        NULL
    );
}

evoke_status
evoke_document_cow_visit_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_object_visitor visitor,
    void *visitor_context
)
{
    if (visitor == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    return evoke_document_cow_walk_external(
        root,
        document_slot_count,
        loader,
        loader_context,
        visitor,
        visitor_context
    );
}

typedef enum evoke_document_cow_born_heap_kind
{
    EVOKE_DOCUMENT_COW_BORN_HEAP_REF = 0,
    EVOKE_DOCUMENT_COW_BORN_HEAP_RECORD = 1
} evoke_document_cow_born_heap_kind;

typedef struct evoke_document_cow_born_heap_item
{
    uint64_t born_sequence;
    uint64_t document_slot;
    evoke_document_cow_born_heap_kind kind;
    union
    {
        evoke_document_cow_ref ref;
        evoke_document_cow_record record;
    } value;
} evoke_document_cow_born_heap_item;

typedef struct evoke_document_cow_born_heap
{
    evoke_document_cow_born_heap_item *items;
    size_t count;
    size_t capacity;
    uint64_t peak;
} evoke_document_cow_born_heap;

static bool
evoke_document_cow_born_item_precedes(
    const evoke_document_cow_born_heap_item *left,
    const evoke_document_cow_born_heap_item *right
)
{
    if (left->born_sequence != right->born_sequence)
    {
        return left->born_sequence < right->born_sequence;
    }
    if (left->kind != right->kind)
    {
        return left->kind < right->kind;
    }
    return left->document_slot < right->document_slot;
}

static evoke_status
evoke_document_cow_born_heap_reserve(
    evoke_document_cow_born_heap *heap,
    size_t additional
)
{
    size_t required;
    size_t capacity;
    evoke_document_cow_born_heap_item *items;

    if (heap == NULL || additional > SIZE_MAX - heap->count)
    {
        return EVOKE_ERR_RANGE;
    }
    required = heap->count + additional;
    if (required <= heap->capacity)
    {
        return EVOKE_OK;
    }
    capacity = heap->capacity == 0 ? 64 : heap->capacity;
    while (capacity < required)
    {
        if (capacity > SIZE_MAX / 2)
        {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(*items))
    {
        return EVOKE_ERR_RANGE;
    }
    items = realloc(heap->items, capacity * sizeof(*items));
    if (items == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    heap->items = items;
    heap->capacity = capacity;
    return EVOKE_OK;
}

static evoke_status
evoke_document_cow_born_heap_push(
    evoke_document_cow_born_heap *heap,
    const evoke_document_cow_born_heap_item *item
)
{
    size_t index;
    evoke_status status;

    status = evoke_document_cow_born_heap_reserve(heap, 1);
    if (status != EVOKE_OK)
    {
        return status;
    }
    index = heap->count++;
    heap->items[index] = *item;
    while (index > 0)
    {
        size_t parent = (index - 1) / 2;
        evoke_document_cow_born_heap_item temporary;

        if (!evoke_document_cow_born_item_precedes(
                &heap->items[index],
                &heap->items[parent]))
        {
            break;
        }
        temporary = heap->items[parent];
        heap->items[parent] = heap->items[index];
        heap->items[index] = temporary;
        index = parent;
    }
    if (heap->count > heap->peak)
    {
        heap->peak = heap->count;
    }
    return EVOKE_OK;
}

static bool
evoke_document_cow_born_heap_pop(
    evoke_document_cow_born_heap *heap,
    evoke_document_cow_born_heap_item *item_out
)
{
    size_t index = 0;

    if (heap == NULL || item_out == NULL || heap->count == 0)
    {
        return false;
    }
    *item_out = heap->items[0];
    heap->count--;
    if (heap->count == 0)
    {
        return true;
    }
    heap->items[0] = heap->items[heap->count];
    while (true)
    {
        size_t left = index * 2 + 1;
        size_t right = left + 1;
        size_t smallest = index;
        evoke_document_cow_born_heap_item temporary;

        if (left < heap->count &&
            evoke_document_cow_born_item_precedes(
                &heap->items[left],
                &heap->items[smallest]))
        {
            smallest = left;
        }
        if (right < heap->count &&
            evoke_document_cow_born_item_precedes(
                &heap->items[right],
                &heap->items[smallest]))
        {
            smallest = right;
        }
        if (smallest == index)
        {
            break;
        }
        temporary = heap->items[index];
        heap->items[index] = heap->items[smallest];
        heap->items[smallest] = temporary;
        index = smallest;
    }
    return true;
}

static evoke_status
evoke_document_cow_born_heap_push_ref(
    evoke_document_cow_born_heap *heap,
    const evoke_document_cow_ref *ref
)
{
    evoke_document_cow_born_heap_item item;

    if (ref->live_document_count == 0)
    {
        return EVOKE_OK;
    }
    memset(&item, 0, sizeof(item));
    item.born_sequence = ref->min_live_born_sequence;
    item.document_slot = ref->first_document_slot;
    item.kind = EVOKE_DOCUMENT_COW_BORN_HEAP_REF;
    item.value.ref = *ref;
    return evoke_document_cow_born_heap_push(heap, &item);
}

static evoke_status
evoke_document_cow_born_heap_push_record(
    evoke_document_cow_born_heap *heap,
    const evoke_document_cow_record *record
)
{
    evoke_document_cow_born_heap_item item;

    if (evoke_document_cow_record_is_aborted(record) ||
        evoke_document_cow_record_is_retired(record))
    {
        return EVOKE_OK;
    }
    memset(&item, 0, sizeof(item));
    item.born_sequence = record->version.born_sequence;
    item.document_slot = record->version.document_slot;
    item.kind = EVOKE_DOCUMENT_COW_BORN_HEAP_RECORD;
    item.value.record = *record;
    return evoke_document_cow_born_heap_push(heap, &item);
}

evoke_status
evoke_document_cow_collect_live_born_prefix_matching_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    size_t limit,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record_predicate predicate,
    void *predicate_context,
    evoke_document_cow_record *records_out,
    size_t record_capacity,
    size_t *record_count_out,
    evoke_document_cow_born_prefix_stats *stats_out
)
{
    evoke_document_cow_born_heap heap = {0};
    evoke_document_cow_born_prefix_stats stats = {0};
    size_t target_count;
    size_t record_count = 0;
    uint64_t previous_sequence = 0;
    uint64_t previous_slot = 0;
    bool previous_seen = false;
    evoke_status status;

    if (root == NULL || loader == NULL || record_count_out == NULL ||
        stats_out == NULL || document_slot_count == 0 ||
        document_slot_count > UINT32_MAX ||
        root->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        root->first_document_slot != 0 ||
        root->document_slot_count != document_slot_count ||
        evoke_document_cow_ref_validate(root, true) != EVOKE_OK ||
        limit > record_capacity ||
        (record_capacity > 0 && records_out == NULL))
    {
        return EVOKE_ERR_INVALID;
    }
    *record_count_out = 0;
    memset(stats_out, 0, sizeof(*stats_out));
    target_count = limit;
    if ((uint64_t) target_count > root->live_document_count)
    {
        target_count = (size_t) root->live_document_count;
    }
    if (target_count == 0)
    {
        return EVOKE_OK;
    }

    status = evoke_document_cow_born_heap_push_ref(&heap, root);
    while (status == EVOKE_OK && record_count < target_count)
    {
        evoke_document_cow_born_heap_item item;

        if (!evoke_document_cow_born_heap_pop(&heap, &item))
        {
            status = predicate == NULL ? EVOKE_ERR_FORMAT : EVOKE_OK;
            break;
        }
        if (item.kind == EVOKE_DOCUMENT_COW_BORN_HEAP_RECORD)
        {
            if (previous_seen &&
                (item.born_sequence < previous_sequence ||
                 (item.born_sequence == previous_sequence &&
                  item.document_slot <= previous_slot)))
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
            previous_sequence = item.born_sequence;
            previous_slot = item.document_slot;
            previous_seen = true;
            if (predicate != NULL &&
                !predicate(predicate_context, &item.value.record))
            {
                continue;
            }
            records_out[record_count++] = item.value.record;
            continue;
        }
        else
        {
            evoke_document_cow_object object;

            status = loader(loader_context, &item.value.ref, &object);
            if (status != EVOKE_OK)
            {
                break;
            }
            stats.objects_loaded++;
            if (!evoke_document_cow_refs_equal(
                    &item.value.ref,
                    &object.ref) ||
                evoke_document_cow_object_validate(
                    &object,
                    true) != EVOKE_OK)
            {
                status = EVOKE_ERR_FORMAT;
                break;
            }
            if (object.ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
            {
                for (uint32_t index = 0;
                     status == EVOKE_OK &&
                         index < object.value.node.child_count;
                     index++)
                {
                    status = evoke_document_cow_born_heap_push_ref(
                        &heap,
                        &object.value.node.children[index].ref
                    );
                }
            }
            else
            {
                stats.records_examined +=
                    object.value.leaf.record_count;
                for (uint32_t index = 0;
                     status == EVOKE_OK &&
                         index < object.value.leaf.record_count;
                     index++)
                {
                    status = evoke_document_cow_born_heap_push_record(
                        &heap,
                        &object.value.leaf.records[index]
                    );
                }
            }
        }
    }
    stats.heap_peak = heap.peak;
    free(heap.items);
    if (status != EVOKE_OK)
    {
        return status;
    }
    *record_count_out = record_count;
    *stats_out = stats;
    return EVOKE_OK;
}

evoke_status
evoke_document_cow_collect_live_born_prefix_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    size_t limit,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *records_out,
    size_t record_capacity,
    size_t *record_count_out,
    evoke_document_cow_born_prefix_stats *stats_out
)
{
    return evoke_document_cow_collect_live_born_prefix_matching_external(
        root,
        document_slot_count,
        limit,
        loader,
        loader_context,
        NULL,
        NULL,
        records_out,
        record_capacity,
        record_count_out,
        stats_out
    );
}

static evoke_status
evoke_document_cow_load_tree_object(
    void *context,
    const evoke_document_cow_ref *ref,
    evoke_document_cow_object *object_out
)
{
    const evoke_document_cow_tree *tree = context;
    const evoke_document_cow_object *object =
        evoke_document_cow_tree_find_const_object(tree, ref);

    if (object == NULL || object_out == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    *object_out = *object;
    return EVOKE_OK;
}

typedef struct evoke_document_cow_length_walk
{
    uint64_t first_document_slot;
    uint64_t end_document_slot;
    bool require_storage;
    evoke_document_cow_object_loader loader;
    void *loader_context;
    evoke_document_cow_length_extrema extrema;
} evoke_document_cow_length_walk;

static void
evoke_document_cow_length_add(
    evoke_document_cow_length_extrema *target,
    uint64_t document_count,
    uint32_t min_document_length,
    uint32_t max_document_length
)
{
    if (document_count == 0)
    {
        return;
    }
    if (target->document_count == 0)
    {
        target->min_document_length = min_document_length;
        target->max_document_length = max_document_length;
    }
    else
    {
        if (min_document_length < target->min_document_length)
        {
            target->min_document_length = min_document_length;
        }
        if (max_document_length > target->max_document_length)
        {
            target->max_document_length = max_document_length;
        }
    }
    target->document_count += document_count;
}

static evoke_status
evoke_document_cow_length_walk_ref(
    evoke_document_cow_length_walk *walk,
    const evoke_document_cow_ref *ref,
    bool summary_trusted
)
{
    uint64_t ref_end;

    if (walk == NULL || ref == NULL ||
        evoke_document_cow_ref_validate(
            ref,
            walk->require_storage) != EVOKE_OK ||
        ref->first_document_slot >
            UINT64_MAX - ref->document_slot_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    ref_end = ref->first_document_slot + ref->document_slot_count;
    if (ref_end <= walk->first_document_slot ||
        ref->first_document_slot >= walk->end_document_slot)
    {
        return EVOKE_OK;
    }
    if (summary_trusted &&
        walk->first_document_slot <= ref->first_document_slot &&
        ref_end <= walk->end_document_slot)
    {
        evoke_document_cow_length_add(
            &walk->extrema,
            ref->bounded_document_count,
            ref->min_document_length,
            ref->max_document_length
        );
        return EVOKE_OK;
    }

    {
        evoke_document_cow_object object;
        evoke_status status = walk->loader(
            walk->loader_context,
            ref,
            &object
        );

        if (status != EVOKE_OK ||
            !evoke_document_cow_refs_equal(&object.ref, ref) ||
            evoke_document_cow_object_validate(
                &object,
                walk->require_storage) != EVOKE_OK)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (walk->first_document_slot <=
                object.ref.first_document_slot &&
            ref_end <= walk->end_document_slot)
        {
            evoke_document_cow_length_add(
                &walk->extrema,
                object.ref.bounded_document_count,
                object.ref.min_document_length,
                object.ref.max_document_length
            );
            return EVOKE_OK;
        }
        if (object.ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
        {
            for (uint32_t index = 0;
                 index < object.value.node.child_count;
                 index++)
            {
                status = evoke_document_cow_length_walk_ref(
                    walk,
                    &object.value.node.children[index].ref,
                    true
                );
                if (status != EVOKE_OK)
                {
                    return status;
                }
            }
            return EVOKE_OK;
        }

        for (uint32_t index = 0;
             index < object.value.leaf.record_count;
             index++)
        {
            const evoke_document_cow_record *record =
                &object.value.leaf.records[index];
            uint64_t document_slot = record->version.document_slot;

            if (document_slot < walk->first_document_slot ||
                document_slot >= walk->end_document_slot ||
                evoke_document_cow_record_is_aborted(record))
            {
                continue;
            }
            evoke_document_cow_length_add(
                &walk->extrema,
                1,
                record->version.document_length,
                record->version.document_length
            );
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_document_cow_length_extrema_with_loader(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t first_document_slot,
    uint64_t range_document_slot_count,
    bool require_storage,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_length_extrema *extrema_out
)
{
    evoke_document_cow_length_walk walk;
    evoke_status status;

    if (root == NULL || loader == NULL || extrema_out == NULL ||
        document_slot_count > UINT32_MAX ||
        range_document_slot_count == 0 ||
        first_document_slot >= document_slot_count ||
        range_document_slot_count >
            document_slot_count - first_document_slot ||
        root->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        root->first_document_slot != 0 ||
        root->document_slot_count != document_slot_count ||
        evoke_document_cow_ref_validate(
            root,
            require_storage) != EVOKE_OK)
    {
        return EVOKE_ERR_INVALID;
    }

    memset(&walk, 0, sizeof(walk));
    walk.first_document_slot = first_document_slot;
    walk.end_document_slot =
        first_document_slot + range_document_slot_count;
    walk.require_storage = require_storage;
    walk.loader = loader;
    walk.loader_context = loader_context;
    status = evoke_document_cow_length_walk_ref(
        &walk,
        root,
        false
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    *extrema_out = walk.extrema;
    return EVOKE_OK;
}

evoke_status
evoke_document_cow_length_extrema_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t first_document_slot,
    uint64_t range_document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_length_extrema *extrema_out
)
{
    return evoke_document_cow_length_extrema_with_loader(
        root,
        document_slot_count,
        first_document_slot,
        range_document_slot_count,
        true,
        loader,
        loader_context,
        extrema_out
    );
}

evoke_status
evoke_document_cow_tree_length_extrema(
    const evoke_document_cow_tree *tree,
    uint64_t first_document_slot,
    uint64_t range_document_slot_count,
    evoke_document_cow_length_extrema *extrema_out
)
{
    if (tree == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    return evoke_document_cow_length_extrema_with_loader(
        &tree->root,
        tree->document_slot_count,
        first_document_slot,
        range_document_slot_count,
        false,
        evoke_document_cow_load_tree_object,
        (void *) tree,
        extrema_out
    );
}

typedef struct evoke_document_cow_block_walk
{
    uint32_t block_shift;
    bool require_storage;
    evoke_document_cow_object_loader loader;
    void *loader_context;
    evoke_document_cow_length_extrema *extrema;
    size_t extrema_count;
} evoke_document_cow_block_walk;

static evoke_status
evoke_document_cow_block_walk_ref(
    evoke_document_cow_block_walk *walk,
    const evoke_document_cow_ref *ref,
    bool summary_trusted
)
{
    uint64_t ref_end;
    uint64_t first_block;
    uint64_t last_block;

    if (walk == NULL || ref == NULL ||
        evoke_document_cow_ref_validate(
            ref,
            walk->require_storage) != EVOKE_OK ||
        ref->document_slot_count == 0 ||
        ref->first_document_slot >
            UINT64_MAX - ref->document_slot_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    ref_end = ref->first_document_slot + ref->document_slot_count;
    first_block = ref->first_document_slot >> walk->block_shift;
    last_block = (ref_end - 1) >> walk->block_shift;
    if (last_block >= walk->extrema_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    if (summary_trusted && first_block == last_block)
    {
        evoke_document_cow_length_add(
            &walk->extrema[first_block],
            ref->bounded_document_count,
            ref->min_document_length,
            ref->max_document_length
        );
        return EVOKE_OK;
    }

    {
        evoke_document_cow_object object;
        evoke_status status = walk->loader(
            walk->loader_context,
            ref,
            &object
        );

        if (status != EVOKE_OK ||
            !evoke_document_cow_refs_equal(&object.ref, ref) ||
            evoke_document_cow_object_validate(
                &object,
                walk->require_storage) != EVOKE_OK)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (first_block == last_block)
        {
            evoke_document_cow_length_add(
                &walk->extrema[first_block],
                object.ref.bounded_document_count,
                object.ref.min_document_length,
                object.ref.max_document_length
            );
            return EVOKE_OK;
        }
        if (object.ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
        {
            for (uint32_t index = 0;
                 index < object.value.node.child_count;
                 index++)
            {
                status = evoke_document_cow_block_walk_ref(
                    walk,
                    &object.value.node.children[index].ref,
                    true
                );
                if (status != EVOKE_OK)
                {
                    return status;
                }
            }
            return EVOKE_OK;
        }

        for (uint32_t index = 0;
             index < object.value.leaf.record_count;
             index++)
        {
            const evoke_document_cow_record *record =
                &object.value.leaf.records[index];
            uint64_t block_id =
                record->version.document_slot >> walk->block_shift;

            if (block_id >= walk->extrema_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            if (evoke_document_cow_record_is_aborted(record))
            {
                continue;
            }
            evoke_document_cow_length_add(
                &walk->extrema[block_id],
                1,
                record->version.document_length,
                record->version.document_length
            );
        }
    }
    return EVOKE_OK;
}

static evoke_status
evoke_document_cow_block_extrema_with_loader(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint32_t block_shift,
    bool require_storage,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_length_extrema **extrema_out,
    size_t *extrema_count_out
)
{
    evoke_document_cow_block_walk walk;
    evoke_document_cow_length_extrema *extrema;
    uint64_t block_size;
    uint64_t block_count;
    uint64_t bounded_document_count = 0;
    evoke_status status;

    if (root == NULL || loader == NULL || extrema_out == NULL ||
        extrema_count_out == NULL || document_slot_count == 0 ||
        document_slot_count > UINT32_MAX ||
        block_shift == 0 || block_shift >= 32 ||
        root->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        root->first_document_slot != 0 ||
        root->document_slot_count != document_slot_count ||
        evoke_document_cow_ref_validate(
            root,
            require_storage) != EVOKE_OK)
    {
        return EVOKE_ERR_INVALID;
    }
    *extrema_out = NULL;
    *extrema_count_out = 0;
    block_size = UINT64_C(1) << block_shift;
    block_count =
        (document_slot_count + block_size - 1) >> block_shift;
    if (block_count == 0 ||
        block_count > SIZE_MAX / sizeof(*extrema))
    {
        return EVOKE_ERR_RANGE;
    }
    extrema = calloc(
        (size_t) block_count,
        sizeof(*extrema)
    );
    if (extrema == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }

    memset(&walk, 0, sizeof(walk));
    walk.block_shift = block_shift;
    walk.require_storage = require_storage;
    walk.loader = loader;
    walk.loader_context = loader_context;
    walk.extrema = extrema;
    walk.extrema_count = (size_t) block_count;
    status = evoke_document_cow_block_walk_ref(
        &walk,
        root,
        false
    );
    if (status != EVOKE_OK)
    {
        free(extrema);
        return status;
    }
    for (size_t index = 0; index < (size_t) block_count; index++)
    {
        if (UINT64_MAX - bounded_document_count <
            extrema[index].document_count)
        {
            free(extrema);
            return EVOKE_ERR_RANGE;
        }
        bounded_document_count += extrema[index].document_count;
    }
    if (bounded_document_count != root->bounded_document_count)
    {
        free(extrema);
        return EVOKE_ERR_FORMAT;
    }
    *extrema_out = extrema;
    *extrema_count_out = (size_t) block_count;
    return EVOKE_OK;
}

evoke_status
evoke_document_cow_block_extrema_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint32_t block_shift,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_length_extrema **extrema_out,
    size_t *extrema_count_out
)
{
    return evoke_document_cow_block_extrema_with_loader(
        root,
        document_slot_count,
        block_shift,
        true,
        loader,
        loader_context,
        extrema_out,
        extrema_count_out
    );
}

evoke_status
evoke_document_cow_tree_block_extrema(
    const evoke_document_cow_tree *tree,
    uint32_t block_shift,
    evoke_document_cow_length_extrema **extrema_out,
    size_t *extrema_count_out
)
{
    if (tree == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    return evoke_document_cow_block_extrema_with_loader(
        &tree->root,
        tree->document_slot_count,
        block_shift,
        false,
        evoke_document_cow_load_tree_object,
        (void *) tree,
        extrema_out,
        extrema_count_out
    );
}

void
evoke_document_cow_block_extrema_free(
    evoke_document_cow_length_extrema *extrema
)
{
    free(extrema);
}

static evoke_status
evoke_document_cow_lookup_with_loader(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t document_slot,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out
)
{
    evoke_document_cow_ref current;

    if (root == NULL || loader == NULL || record_out == NULL ||
        document_slot >= document_slot_count ||
        root->document_slot_count != document_slot_count)
    {
        return EVOKE_ERR_INVALID;
    }
    current = *root;
    for (uint16_t expected_level =
             EVOKE_DOCUMENT_COW_RADIX_LEVELS - 1;
         ;
         expected_level--)
    {
        evoke_document_cow_object object;
        uint32_t child_slot;
        bool found = false;
        evoke_status status = loader(
            loader_context,
            &current,
            &object
        );

        if (status != EVOKE_OK ||
            evoke_document_cow_object_validate(
                &object,
                evoke_document_cow_ref_is_bound(&current)) != EVOKE_OK ||
            !evoke_document_cow_refs_equal(&object.ref, &current) ||
            object.ref.kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
            object.value.node.level != expected_level)
        {
            return EVOKE_ERR_FORMAT;
        }
        child_slot = evoke_document_cow_child_slot(
            document_slot,
            expected_level
        );
        for (uint32_t index = 0;
             index < object.value.node.child_count;
             index++)
        {
            if (object.value.node.children[index].slot == child_slot)
            {
                current = object.value.node.children[index].ref;
                found = true;
                break;
            }
        }
        if (!found)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (expected_level == 0)
        {
            evoke_document_cow_object leaf_object;
            uint64_t local_slot;

            status = loader(
                loader_context,
                &current,
                &leaf_object
            );
            if (status != EVOKE_OK ||
                evoke_document_cow_object_validate(
                    &leaf_object,
                    evoke_document_cow_ref_is_bound(&current)) !=
                    EVOKE_OK ||
                !evoke_document_cow_refs_equal(
                    &leaf_object.ref,
                    &current) ||
                leaf_object.ref.kind !=
                    EVOKE_DOCUMENT_COW_OBJECT_LEAF ||
                document_slot <
                    leaf_object.value.leaf.base_document_slot)
            {
                return EVOKE_ERR_FORMAT;
            }
            local_slot = document_slot -
                leaf_object.value.leaf.base_document_slot;
            if (local_slot >= leaf_object.value.leaf.record_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            *record_out =
                leaf_object.value.leaf.records[local_slot];
            return EVOKE_OK;
        }
    }
}

typedef struct evoke_document_cow_range_walk
{
    uint64_t first_document_slot;
    uint64_t end_document_slot;
    uint64_t next_document_slot;
    evoke_document_cow_object_loader loader;
    void *loader_context;
    evoke_document_cow_record *records;
} evoke_document_cow_range_walk;

static bool
evoke_document_cow_ref_overlaps_range(
    const evoke_document_cow_ref *ref,
    uint64_t first_document_slot,
    uint64_t end_document_slot
)
{
    uint64_t ref_end;

    if (ref == NULL || ref->document_slot_count == 0 ||
        ref->first_document_slot >
            UINT64_MAX - ref->document_slot_count)
    {
        return false;
    }
    ref_end = ref->first_document_slot + ref->document_slot_count;
    return ref->first_document_slot < end_document_slot &&
        ref_end > first_document_slot;
}

static evoke_status
evoke_document_cow_read_range_ref(
    evoke_document_cow_range_walk *walk,
    const evoke_document_cow_ref *ref,
    uint16_t expected_level
)
{
    evoke_document_cow_object object;
    evoke_status status;

    if (walk == NULL || ref == NULL ||
        !evoke_document_cow_ref_overlaps_range(
            ref,
            walk->first_document_slot,
            walk->end_document_slot))
    {
        return EVOKE_ERR_FORMAT;
    }
    status = walk->loader(walk->loader_context, ref, &object);
    if (status != EVOKE_OK ||
        evoke_document_cow_object_validate(
            &object,
            evoke_document_cow_ref_is_bound(ref)) != EVOKE_OK ||
        !evoke_document_cow_refs_equal(&object.ref, ref))
    {
        return status == EVOKE_OK ? EVOKE_ERR_FORMAT : status;
    }

    if (ref->kind == EVOKE_DOCUMENT_COW_OBJECT_LEAF)
    {
        const evoke_document_cow_leaf *leaf = &object.value.leaf;
        uint64_t leaf_end;
        uint64_t copy_first;
        uint64_t copy_end;
        size_t source_offset;
        size_t target_offset;
        size_t copy_count;

        if (expected_level != UINT16_MAX ||
            leaf->base_document_slot >
                UINT64_MAX - leaf->record_count)
        {
            return EVOKE_ERR_FORMAT;
        }
        leaf_end = leaf->base_document_slot + leaf->record_count;
        copy_first = leaf->base_document_slot;
        if (copy_first < walk->first_document_slot)
        {
            copy_first = walk->first_document_slot;
        }
        copy_end = leaf_end;
        if (copy_end > walk->end_document_slot)
        {
            copy_end = walk->end_document_slot;
        }
        if (copy_first >= copy_end ||
            copy_first != walk->next_document_slot)
        {
            return EVOKE_ERR_FORMAT;
        }
        source_offset = (size_t) (
            copy_first - leaf->base_document_slot
        );
        target_offset = (size_t) (
            copy_first - walk->first_document_slot
        );
        copy_count = (size_t) (copy_end - copy_first);
        memcpy(
            walk->records + target_offset,
            leaf->records + source_offset,
            copy_count * sizeof(*walk->records)
        );
        walk->next_document_slot = copy_end;
        return EVOKE_OK;
    }

    if (ref->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        expected_level == UINT16_MAX ||
        object.value.node.level != expected_level)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t index = 0;
         index < object.value.node.child_count;
         index++)
    {
        const evoke_document_cow_child *child =
            &object.value.node.children[index];
        uint64_t child_last;
        evoke_document_cow_object_kind expected_kind;

        if (!evoke_document_cow_ref_overlaps_range(
                &child->ref,
                walk->first_document_slot,
                walk->end_document_slot))
        {
            continue;
        }
        child_last = child->ref.first_document_slot +
            child->ref.document_slot_count - 1;
        expected_kind = expected_level == 0
            ? EVOKE_DOCUMENT_COW_OBJECT_LEAF
            : EVOKE_DOCUMENT_COW_OBJECT_NODE;
        if (child->ref.kind != expected_kind ||
            child->slot != evoke_document_cow_child_slot(
                child->ref.first_document_slot,
                expected_level) ||
            child->slot != evoke_document_cow_child_slot(
                child_last,
                expected_level))
        {
            return EVOKE_ERR_FORMAT;
        }
        status = evoke_document_cow_read_range_ref(
            walk,
            &child->ref,
            expected_level == 0
                ? UINT16_MAX
                : (uint16_t) (expected_level - 1)
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
    }
    return EVOKE_OK;
}

evoke_status
evoke_document_cow_read_range_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t first_document_slot,
    uint64_t range_document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *records_out
)
{
    evoke_document_cow_range_walk walk;
    evoke_status status;

    if (root == NULL || loader == NULL || records_out == NULL ||
        range_document_slot_count == 0 ||
        root->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        root->document_slot_count != document_slot_count ||
        first_document_slot >= document_slot_count ||
        range_document_slot_count >
            document_slot_count - first_document_slot ||
        range_document_slot_count >
            SIZE_MAX / sizeof(*records_out))
    {
        return EVOKE_ERR_INVALID;
    }
    memset(&walk, 0, sizeof(walk));
    walk.first_document_slot = first_document_slot;
    walk.end_document_slot = first_document_slot +
        range_document_slot_count;
    walk.next_document_slot = first_document_slot;
    walk.loader = loader;
    walk.loader_context = loader_context;
    walk.records = records_out;
    status = evoke_document_cow_read_range_ref(
        &walk,
        root,
        EVOKE_DOCUMENT_COW_RADIX_LEVELS - 1
    );
    if (status == EVOKE_OK &&
        walk.next_document_slot != walk.end_document_slot)
    {
        return EVOKE_ERR_FORMAT;
    }
    return status;
}

evoke_status
evoke_document_cow_tree_lookup(
    const evoke_document_cow_tree *tree,
    uint64_t document_slot,
    evoke_document_cow_record *record_out
)
{
    if (tree == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    return evoke_document_cow_lookup_with_loader(
        &tree->root,
        tree->document_slot_count,
        document_slot,
        evoke_document_cow_load_tree_object,
        (void *) tree,
        record_out
    );
}

static evoke_status
evoke_document_cow_find_actionable_ref(
    const evoke_document_cow_ref *ref,
    uint16_t expected_level,
    uint64_t first_document_slot,
    int64_t now,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out,
    bool *found_out
)
{
    evoke_document_cow_object object;
    uint64_t ref_end;
    evoke_status status;

    *found_out = false;
    if (ref->first_document_slot >
        UINT64_MAX - ref->document_slot_count)
    {
        return EVOKE_ERR_FORMAT;
    }
    ref_end = ref->first_document_slot + ref->document_slot_count;
    if (ref->semantic_pending_count == 0 ||
        ref->earliest_retry_after > now ||
        ref_end <= first_document_slot)
    {
        return EVOKE_OK;
    }
    status = loader(loader_context, ref, &object);
    if (status != EVOKE_OK ||
        evoke_document_cow_object_validate(
            &object,
            evoke_document_cow_ref_is_bound(ref)) != EVOKE_OK ||
        !evoke_document_cow_refs_equal(&object.ref, ref))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (ref->kind == EVOKE_DOCUMENT_COW_OBJECT_LEAF)
    {
        for (uint32_t index = 0;
             index < object.value.leaf.record_count;
             index++)
        {
            if (object.value.leaf.records[index].
                    version.document_slot < first_document_slot)
            {
                continue;
            }
            if (evoke_document_cow_record_is_actionable(
                    &object.value.leaf.records[index],
                    now))
            {
                *record_out = object.value.leaf.records[index];
                *found_out = true;
                return EVOKE_OK;
            }
        }
        return first_document_slot > ref->first_document_slot
            ? EVOKE_OK
            : EVOKE_ERR_FORMAT;
    }
    if (object.value.node.level != expected_level)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t index = 0;
         index < object.value.node.child_count;
         index++)
    {
        const evoke_document_cow_ref *child =
            &object.value.node.children[index].ref;

        if (child->semantic_pending_count == 0 ||
            child->earliest_retry_after > now)
        {
            continue;
        }
        status = evoke_document_cow_find_actionable_ref(
            child,
            expected_level == 0 ? 0 : expected_level - 1,
            first_document_slot,
            now,
            loader,
            loader_context,
            record_out,
            found_out
        );
        if (status != EVOKE_OK || *found_out)
        {
            return status;
        }
    }
    return first_document_slot > ref->first_document_slot
        ? EVOKE_OK
        : EVOKE_ERR_FORMAT;
}

evoke_status
evoke_document_cow_find_actionable_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    int64_t now,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out,
    bool *found_out
)
{
    return evoke_document_cow_find_actionable_external_from(
        root,
        document_slot_count,
        0,
        now,
        loader,
        loader_context,
        record_out,
        found_out
    );
}

evoke_status
evoke_document_cow_find_actionable_external_from(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t first_document_slot,
    int64_t now,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out,
    bool *found_out
)
{
    if (root == NULL || loader == NULL ||
        record_out == NULL || found_out == NULL ||
        root->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        root->document_slot_count != document_slot_count ||
        first_document_slot > document_slot_count)
    {
        return EVOKE_ERR_INVALID;
    }
    *found_out = false;
    if (first_document_slot == document_slot_count ||
        root->semantic_pending_count == 0 ||
        root->earliest_retry_after > now)
    {
        return EVOKE_OK;
    }
    return evoke_document_cow_find_actionable_ref(
        root,
        EVOKE_DOCUMENT_COW_RADIX_LEVELS - 1,
        first_document_slot,
        now,
        loader,
        loader_context,
        record_out,
        found_out
    );
}

static evoke_status
evoke_document_cow_find_reusable_ref_from(
    const evoke_document_cow_ref *ref,
    uint16_t expected_level,
    uint64_t first_document_slot,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out,
    bool *found_out
)
{
    evoke_document_cow_object object;
    evoke_status status;

    *found_out = false;
    if (ref->reusable_document_count == 0 ||
        ref->first_document_slot >
            UINT64_MAX - ref->document_slot_count ||
        ref->first_document_slot + ref->document_slot_count <=
            first_document_slot)
    {
        return EVOKE_OK;
    }
    status = loader(loader_context, ref, &object);
    if (status != EVOKE_OK ||
        evoke_document_cow_object_validate(
            &object,
            evoke_document_cow_ref_is_bound(ref)) != EVOKE_OK ||
        !evoke_document_cow_refs_equal(&object.ref, ref))
    {
        return EVOKE_ERR_FORMAT;
    }
    if (ref->kind == EVOKE_DOCUMENT_COW_OBJECT_LEAF)
    {
        uint64_t local_slot;
        uint64_t first_local_slot = 0;

        if (expected_level != UINT16_MAX)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (first_document_slot > object.value.leaf.base_document_slot)
        {
            first_local_slot = first_document_slot -
                object.value.leaf.base_document_slot;
        }
        for (local_slot = first_local_slot;
             local_slot < object.value.leaf.record_count;
             local_slot++)
        {
            if (!evoke_document_cow_record_is_reusable(
                    &object.value.leaf.records[local_slot]))
            {
                continue;
            }
            *record_out = object.value.leaf.records[local_slot];
            *found_out = true;
            return EVOKE_OK;
        }
        return EVOKE_OK;
    }
    if (ref->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        object.value.node.level != expected_level)
    {
        return EVOKE_ERR_FORMAT;
    }
    for (uint32_t index = 0;
         index < object.value.node.child_count;
         index++)
    {
        const evoke_document_cow_ref *child =
            &object.value.node.children[index].ref;

        bool child_found = false;

        if (child->reusable_document_count == 0 ||
            child->first_document_slot >
                UINT64_MAX - child->document_slot_count ||
            child->first_document_slot + child->document_slot_count <=
                first_document_slot)
        {
            continue;
        }
        status = evoke_document_cow_find_reusable_ref_from(
            child,
            child->kind == EVOKE_DOCUMENT_COW_OBJECT_LEAF
                ? UINT16_MAX
                : (uint16_t) (expected_level - 1),
            first_document_slot,
            loader,
            loader_context,
            record_out,
            &child_found
        );
        if (status != EVOKE_OK || child_found)
        {
            *found_out = child_found;
            return status;
        }
    }
    return EVOKE_OK;
}

evoke_status
evoke_document_cow_find_reusable_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out,
    bool *found_out
)
{
    return evoke_document_cow_find_reusable_external_from(
        root,
        document_slot_count,
        0,
        loader,
        loader_context,
        record_out,
        found_out
    );
}

evoke_status
evoke_document_cow_find_reusable_external_from(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t first_document_slot,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out,
    bool *found_out
)
{
    if (root == NULL || loader == NULL || record_out == NULL ||
        found_out == NULL ||
        root->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        root->document_slot_count != document_slot_count ||
        first_document_slot > document_slot_count)
    {
        return EVOKE_ERR_INVALID;
    }
    *found_out = false;
    if (first_document_slot == document_slot_count ||
        root->reusable_document_count == 0)
    {
        return EVOKE_OK;
    }
    return evoke_document_cow_find_reusable_ref_from(
        root,
        EVOKE_DOCUMENT_COW_RADIX_LEVELS - 1,
        first_document_slot,
        loader,
        loader_context,
        record_out,
        found_out
    );
}

evoke_status
evoke_document_cow_lookup_external(
    const evoke_document_cow_ref *root,
    uint64_t document_slot_count,
    uint64_t document_slot,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_record *record_out
)
{
    return evoke_document_cow_lookup_with_loader(
        root,
        document_slot_count,
        document_slot,
        loader,
        loader_context,
        record_out
    );
}

static evoke_status
evoke_document_cow_load_patch_source(
    const evoke_document_cow_patch_context *context,
    const evoke_document_cow_ref *ref,
    evoke_document_cow_object *object_out
)
{
    const evoke_document_cow_object *local;

    local = evoke_document_cow_tree_find_const_object(
        context->patch,
        ref
    );
    if (local != NULL)
    {
        *object_out = *local;
        return EVOKE_OK;
    }
    return context->loader(
        context->loader_context,
        ref,
        object_out
    );
}

static void
evoke_document_cow_refresh_leaf_summary(
    evoke_document_cow_object *object
)
{
    evoke_document_cow_summary summary;

    evoke_document_cow_summary_init(&summary);
    for (uint32_t index = 0;
         index < object->value.leaf.record_count;
         index++)
    {
        evoke_document_cow_summary_add_record(
            &summary,
            &object->value.leaf.records[index]
        );
    }
    evoke_document_cow_ref_set_summary(&object->ref, &summary);
}

static void
evoke_document_cow_refresh_node_summary(
    evoke_document_cow_object *object
)
{
    evoke_document_cow_summary summary;

    evoke_document_cow_summary_init(&summary);
    for (uint32_t index = 0;
         index < object->value.node.child_count;
         index++)
    {
        evoke_document_cow_summary_add_ref(
            &summary,
            &object->value.node.children[index].ref
        );
    }
    evoke_document_cow_ref_set_summary(&object->ref, &summary);
}

static evoke_status
evoke_document_cow_patch_leaf(
    evoke_document_cow_patch_context *context,
    const evoke_document_cow_ref *old_ref,
    const evoke_document_cow_record *updates,
    size_t update_count,
    evoke_document_cow_ref *ref_out
)
{
    evoke_document_cow_object object;
    uint64_t base_slot;
    evoke_status status;

    status = evoke_document_cow_load_patch_source(
        context,
        old_ref,
        &object
    );
    if (status != EVOKE_OK ||
        object.ref.kind != EVOKE_DOCUMENT_COW_OBJECT_LEAF ||
        !evoke_document_cow_refs_equal(&object.ref, old_ref) ||
        evoke_document_cow_object_validate(
            &object,
            evoke_document_cow_ref_is_bound(old_ref)) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_document_cow_tree_add_retired_ref(
        context->patch,
        old_ref
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    base_slot = object.value.leaf.base_document_slot;
    object.ref.owner_manifest_id = context->owner_manifest_id;
    object.ref.start_block = 0;
    object.ref.page_count = 0;
    object.ref.object_bytes = 0;
    object.ref.checksum = 0;

    for (size_t index = 0; index < update_count; index++)
    {
        uint64_t document_slot =
            updates[index].version.document_slot;
        uint64_t local_slot;

        if (document_slot < base_slot)
        {
            return EVOKE_ERR_FORMAT;
        }
        local_slot = document_slot - base_slot;
        if (local_slot > object.value.leaf.record_count ||
            local_slot >= EVOKE_DOCUMENT_COW_LEAF_RECORDS ||
            evoke_document_cow_record_validate(
                &updates[index],
                document_slot) != EVOKE_OK)
        {
            return EVOKE_ERR_FORMAT;
        }
        if (local_slot < object.value.leaf.record_count &&
            evoke_document_cow_record_transition_validate(
                &object.value.leaf.records[local_slot],
                &updates[index]) != EVOKE_OK)
        {
            return EVOKE_ERR_FORMAT;
        }
        object.value.leaf.records[local_slot] = updates[index];
        if (local_slot == object.value.leaf.record_count)
        {
            object.value.leaf.record_count++;
        }
    }
    evoke_document_cow_refresh_leaf_summary(&object);
    status = evoke_document_cow_tree_append_object(
        context->patch,
        &object,
        ref_out
    );
    if (status == EVOKE_OK)
    {
        context->stats->written_leaves++;
    }
    return status;
}

static evoke_status
evoke_document_cow_insert_child(
    evoke_document_cow_node *node,
    uint16_t child_slot,
    const evoke_document_cow_ref *child_ref
)
{
    uint32_t position = 0;

    while (position < node->child_count &&
           node->children[position].slot < child_slot)
    {
        position++;
    }
    if (position < node->child_count &&
        node->children[position].slot == child_slot)
    {
        node->children[position].ref = *child_ref;
        return EVOKE_OK;
    }
    if (node->child_count >= EVOKE_DOCUMENT_COW_RADIX_FANOUT)
    {
        return EVOKE_ERR_RANGE;
    }
    memmove(
        &node->children[position + 1],
        &node->children[position],
        (size_t) (node->child_count - position) *
            sizeof(*node->children)
    );
    memset(
        &node->children[position],
        0,
        sizeof(*node->children)
    );
    node->children[position].slot = child_slot;
    node->children[position].ref = *child_ref;
    node->child_count++;
    return EVOKE_OK;
}

static const evoke_document_cow_ref *
evoke_document_cow_find_child(
    const evoke_document_cow_node *node,
    uint16_t child_slot
)
{
    for (uint32_t index = 0; index < node->child_count; index++)
    {
        if (node->children[index].slot == child_slot)
        {
            return &node->children[index].ref;
        }
    }
    return NULL;
}

static evoke_status
evoke_document_cow_patch_level(
    evoke_document_cow_patch_context *context,
    const evoke_document_cow_ref *old_ref,
    uint16_t level,
    const evoke_document_cow_record *updates,
    size_t update_count,
    evoke_document_cow_ref *ref_out
)
{
    evoke_document_cow_object object;
    size_t cursor = 0;
    evoke_status status;

    status = evoke_document_cow_load_patch_source(
        context,
        old_ref,
        &object
    );
    if (status != EVOKE_OK ||
        object.ref.kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        object.value.node.level != level ||
        !evoke_document_cow_refs_equal(&object.ref, old_ref) ||
        evoke_document_cow_object_validate(
            &object,
            evoke_document_cow_ref_is_bound(old_ref)) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_document_cow_tree_add_retired_ref(
        context->patch,
        old_ref
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    object.ref.owner_manifest_id = context->owner_manifest_id;
    object.ref.start_block = 0;
    object.ref.page_count = 0;
    object.ref.object_bytes = 0;
    object.ref.checksum = 0;

    while (cursor < update_count)
    {
        uint16_t child_slot = (uint16_t)
            evoke_document_cow_child_slot(
                updates[cursor].version.document_slot,
                level
            );
        size_t end = cursor + 1;
        const evoke_document_cow_ref *old_child;
        evoke_document_cow_ref child_ref;

        while (end < update_count &&
               evoke_document_cow_child_slot(
                   updates[end].version.document_slot,
                   level) == child_slot)
        {
            end++;
        }
        old_child = evoke_document_cow_find_child(
            &object.value.node,
            child_slot
        );
        if (old_child == NULL)
        {
            if (updates[cursor].version.document_slot <
                context->old_document_slot_count)
            {
                return EVOKE_ERR_FORMAT;
            }
            if (level == 0)
            {
                status = evoke_document_cow_build_leaf(
                    context->patch,
                    updates + cursor,
                    end - cursor,
                    &child_ref
                );
                if (status == EVOKE_OK)
                {
                    context->stats->written_leaves++;
                }
            }
            else
            {
                size_t before =
                    context->patch->object_count;

                status = evoke_document_cow_build_level(
                    context->patch,
                    level - 1,
                    updates + cursor,
                    end - cursor,
                    &child_ref
                );
                if (status == EVOKE_OK)
                {
                    for (size_t index = before;
                         index < context->patch->object_count;
                         index++)
                    {
                        if (context->patch->objects[index].ref.kind ==
                            EVOKE_DOCUMENT_COW_OBJECT_LEAF)
                        {
                            context->stats->written_leaves++;
                        }
                        else
                        {
                            context->stats->written_nodes++;
                        }
                    }
                }
            }
        }
        else if (level == 0)
        {
            status = evoke_document_cow_patch_leaf(
                context,
                old_child,
                updates + cursor,
                end - cursor,
                &child_ref
            );
        }
        else
        {
            status = evoke_document_cow_patch_level(
                context,
                old_child,
                level - 1,
                updates + cursor,
                end - cursor,
                &child_ref
            );
        }
        if (status != EVOKE_OK)
        {
            return status;
        }
        status = evoke_document_cow_insert_child(
            &object.value.node,
            child_slot,
            &child_ref
        );
        if (status != EVOKE_OK)
        {
            return status;
        }
        cursor = end;
    }
    evoke_document_cow_refresh_node_summary(&object);
    status = evoke_document_cow_tree_append_object(
        context->patch,
        &object,
        ref_out
    );
    if (status == EVOKE_OK)
    {
        context->stats->written_nodes++;
    }
    return status;
}

static int
evoke_document_cow_compare_records(const void *left, const void *right)
{
    const evoke_document_cow_record *left_record = left;
    const evoke_document_cow_record *right_record = right;

    if (left_record->version.document_slot <
        right_record->version.document_slot)
    {
        return -1;
    }
    if (left_record->version.document_slot >
        right_record->version.document_slot)
    {
        return 1;
    }
    return 0;
}

evoke_status
evoke_document_cow_build_external_patch(
    const evoke_document_cow_ref *old_root,
    uint64_t old_document_slot_count,
    uint64_t next_document_slot_count,
    const evoke_document_cow_record *updates,
    size_t update_count,
    uint64_t owner_manifest_id,
    evoke_document_cow_object_loader loader,
    void *loader_context,
    evoke_document_cow_tree *patch_out,
    evoke_document_cow_update_stats *stats_out
)
{
    evoke_document_cow_patch_context context;
    evoke_document_cow_tree patch;
    evoke_document_cow_update_stats stats;
    evoke_document_cow_record *sorted = NULL;
    uint64_t expected_append_slot = old_document_slot_count;
    evoke_status status;

    if (old_root == NULL || patch_out == NULL || stats_out == NULL ||
        loader == NULL || owner_manifest_id == 0 ||
        owner_manifest_id <= old_root->owner_manifest_id ||
        update_count == 0 || updates == NULL ||
        old_root->kind != EVOKE_DOCUMENT_COW_OBJECT_NODE ||
        old_root->document_slot_count != old_document_slot_count ||
        next_document_slot_count < old_document_slot_count ||
        next_document_slot_count > UINT32_MAX ||
        update_count > SIZE_MAX / sizeof(*updates))
    {
        return EVOKE_ERR_INVALID;
    }
    sorted = malloc(update_count * sizeof(*sorted));
    if (sorted == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    memcpy(sorted, updates, update_count * sizeof(*sorted));
    qsort(
        sorted,
        update_count,
        sizeof(*sorted),
        evoke_document_cow_compare_records
    );
    for (size_t index = 0; index < update_count; index++)
    {
        uint64_t slot = sorted[index].version.document_slot;

        if (slot >= next_document_slot_count ||
            (index > 0 &&
             sorted[index - 1].version.document_slot == slot) ||
            evoke_document_cow_record_validate(
                &sorted[index],
                slot) != EVOKE_OK)
        {
            free(sorted);
            return EVOKE_ERR_FORMAT;
        }
        if (slot >= old_document_slot_count)
        {
            if (slot != expected_append_slot)
            {
                free(sorted);
                return EVOKE_ERR_FORMAT;
            }
            expected_append_slot++;
        }
    }
    if (expected_append_slot != next_document_slot_count)
    {
        free(sorted);
        return EVOKE_ERR_FORMAT;
    }

    evoke_document_cow_tree_init(&patch);
    memset(&stats, 0, sizeof(stats));
    patch.document_slot_count = next_document_slot_count;
    patch.next_object_id = 1;
    patch.root.owner_manifest_id = owner_manifest_id;
    memset(&context, 0, sizeof(context));
    context.old_root = old_root;
    context.old_document_slot_count = old_document_slot_count;
    context.next_document_slot_count = next_document_slot_count;
    context.updates = sorted;
    context.update_count = update_count;
    context.owner_manifest_id = owner_manifest_id;
    context.loader = loader;
    context.loader_context = loader_context;
    context.patch = &patch;
    context.stats = &stats;

    status = evoke_document_cow_patch_level(
        &context,
        old_root,
        EVOKE_DOCUMENT_COW_RADIX_LEVELS - 1,
        sorted,
        update_count,
        &patch.root
    );
    free(sorted);
    if (status != EVOKE_OK)
    {
        evoke_document_cow_tree_free(&patch);
        return status;
    }
    stats.changed_records = (uint32_t) update_count;
    if (patch.root.document_slot_count != next_document_slot_count ||
        evoke_document_cow_tree_validate(&patch) != EVOKE_OK)
    {
        evoke_document_cow_tree_free(&patch);
        return EVOKE_ERR_FORMAT;
    }
    for (size_t index = 0; index < patch.object_count; index++)
    {
        size_t object_bytes;

        status = evoke_document_cow_object_serialized_size(
            &patch.objects[index],
            &object_bytes
        );
        if (status != EVOKE_OK ||
            stats.written_bytes > UINT64_MAX - object_bytes)
        {
            evoke_document_cow_tree_free(&patch);
            return status == EVOKE_OK ? EVOKE_ERR_RANGE : status;
        }
        stats.written_bytes += object_bytes;
    }
    evoke_document_cow_tree_free(patch_out);
    *patch_out = patch;
    *stats_out = stats;
    return EVOKE_OK;
}

static uint64_t
evoke_document_cow_checksum(const uint8_t *bytes, size_t size)
{
    uint64_t checksum = UINT64_C(14695981039346656037);

    for (size_t index = 0; index < size; index++)
    {
        uint8_t byte =
            index >= EVOKE_DOCUMENT_COW_CHECKSUM_OFFSET &&
            index < EVOKE_DOCUMENT_COW_CHECKSUM_OFFSET +
                sizeof(uint64_t)
                ? 0
                : bytes[index];

        checksum ^= byte;
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static void
evoke_document_cow_serialize_ref(
    uint8_t *bytes,
    const evoke_document_cow_ref *ref
)
{
    memset(bytes, 0, EVOKE_DOCUMENT_COW_REF_SIZE);
    evoke_document_cow_write_u32(bytes + 0, (uint32_t) ref->kind);
    evoke_document_cow_write_u32(bytes + 4, ref->start_block);
    evoke_document_cow_write_u32(bytes + 8, ref->page_count);
    evoke_document_cow_write_u32(bytes + 12, ref->reserved);
    evoke_document_cow_write_u64(bytes + 16, ref->object_id);
    evoke_document_cow_write_u64(
        bytes + 24,
        ref->owner_manifest_id
    );
    evoke_document_cow_write_u64(bytes + 32, ref->object_bytes);
    evoke_document_cow_write_u64(bytes + 40, ref->checksum);
    evoke_document_cow_write_u64(
        bytes + 48,
        ref->first_document_slot
    );
    evoke_document_cow_write_u64(
        bytes + 56,
        ref->document_slot_count
    );
    evoke_document_cow_write_u64(
        bytes + 64,
        ref->live_document_count
    );
    evoke_document_cow_write_u64(
        bytes + 72,
        ref->semantic_pending_count
    );
    evoke_document_cow_write_u64(
        bytes + 80,
        (uint64_t) ref->earliest_retry_after
    );
    evoke_document_cow_write_u64(
        bytes + 88,
        ref->bounded_document_count
    );
    evoke_document_cow_write_u32(
        bytes + 96,
        ref->min_document_length
    );
    evoke_document_cow_write_u32(
        bytes + 100,
        ref->max_document_length
    );
    evoke_document_cow_write_u64(
        bytes + 104,
        ref->reusable_document_count
    );
    evoke_document_cow_write_u64(
        bytes + 112,
        ref->first_reusable_document_slot
    );
    evoke_document_cow_write_u64(
        bytes + 120,
        ref->min_live_born_sequence
    );
}

static void
evoke_document_cow_deserialize_ref(
    const uint8_t *bytes,
    evoke_document_cow_ref *ref
)
{
    memset(ref, 0, sizeof(*ref));
    ref->kind = (evoke_document_cow_object_kind)
        evoke_document_cow_read_u32(bytes + 0);
    ref->start_block = evoke_document_cow_read_u32(bytes + 4);
    ref->page_count = evoke_document_cow_read_u32(bytes + 8);
    ref->reserved = evoke_document_cow_read_u32(bytes + 12);
    ref->object_id = evoke_document_cow_read_u64(bytes + 16);
    ref->owner_manifest_id =
        evoke_document_cow_read_u64(bytes + 24);
    ref->object_bytes = evoke_document_cow_read_u64(bytes + 32);
    ref->checksum = evoke_document_cow_read_u64(bytes + 40);
    ref->first_document_slot =
        evoke_document_cow_read_u64(bytes + 48);
    ref->document_slot_count =
        evoke_document_cow_read_u64(bytes + 56);
    ref->live_document_count =
        evoke_document_cow_read_u64(bytes + 64);
    ref->semantic_pending_count =
        evoke_document_cow_read_u64(bytes + 72);
    ref->earliest_retry_after = (int64_t)
        evoke_document_cow_read_u64(bytes + 80);
    ref->bounded_document_count =
        evoke_document_cow_read_u64(bytes + 88);
    ref->min_document_length =
        evoke_document_cow_read_u32(bytes + 96);
    ref->max_document_length =
        evoke_document_cow_read_u32(bytes + 100);
    ref->reusable_document_count =
        evoke_document_cow_read_u64(bytes + 104);
    ref->first_reusable_document_slot =
        evoke_document_cow_read_u64(bytes + 112);
    ref->min_live_born_sequence =
        evoke_document_cow_read_u64(bytes + 120);
}

static void
evoke_document_cow_serialize_record(
    uint8_t *bytes,
    const evoke_document_cow_record *record
)
{
    const evoke_document_version_record *version = &record->version;
    const evoke_document_retirement_record *retirement =
        &record->retirement;
    const evoke_semantic_state_record *semantic =
        &record->semantic_state;

    memset(bytes, 0, EVOKE_DOCUMENT_COW_RECORD_SIZE);
    evoke_document_cow_write_u64(bytes + 0, version->document_slot);
    evoke_document_cow_write_u64(bytes + 8, version->born_sequence);
    evoke_document_cow_write_u32(bytes + 16, version->record_xid);
    evoke_document_cow_write_u32(bytes + 20, version->heap_block);
    evoke_document_cow_write_u32(bytes + 24, version->document_length);
    evoke_document_cow_write_u16(bytes + 28, version->heap_offset);
    evoke_document_cow_write_u16(bytes + 30, version->flags);
    memcpy(
        bytes + 32,
        version->semantic_input_fingerprint,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );

    evoke_document_cow_write_u64(
        bytes + 48,
        retirement->document_slot
    );
    evoke_document_cow_write_u64(
        bytes + 56,
        retirement->retirement_sequence
    );
    evoke_document_cow_write_u32(
        bytes + 64,
        retirement->record_xid
    );
    evoke_document_cow_write_u32(
        bytes + 68,
        retirement->document_length
    );
    evoke_document_cow_write_u16(bytes + 72, retirement->flags);
    evoke_document_cow_write_u16(bytes + 74, retirement->reserved);
    evoke_document_cow_write_u32(bytes + 76, retirement->reserved2);

    evoke_document_cow_write_u64(
        bytes + 80,
        semantic->document_slot
    );
    evoke_document_cow_write_u64(
        bytes + 88,
        semantic->transition_sequence
    );
    evoke_document_cow_write_u32(bytes + 96, semantic->record_xid);
    evoke_document_cow_write_u32(bytes + 100, semantic->error_code);
    evoke_document_cow_write_u64(
        bytes + 104,
        (uint64_t) semantic->retry_after
    );
    evoke_document_cow_write_u16(bytes + 112, semantic->flags);
    evoke_document_cow_write_u16(
        bytes + 114,
        semantic->failure_count
    );
    evoke_document_cow_write_u32(bytes + 116, semantic->reserved);
    memcpy(
        bytes + 120,
        semantic->semantic_input_fingerprint,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );
    evoke_document_cow_write_u64(
        bytes + 136,
        (uint64_t) semantic->pending_since
    );
    evoke_document_cow_write_u64(bytes + 144, semantic->error_hash);
    evoke_document_cow_write_u64(
        bytes + 152,
        record->lexical_residency
    );
    evoke_document_cow_write_u64(
        bytes + 160,
        record->semantic_residency
    );
    evoke_document_cow_write_u64(
        bytes + 168,
        record->event_residency
    );
}

static void
evoke_document_cow_deserialize_record(
    const uint8_t *bytes,
    evoke_document_cow_record *record
)
{
    evoke_document_version_record *version;
    evoke_document_retirement_record *retirement;
    evoke_semantic_state_record *semantic;

    memset(record, 0, sizeof(*record));
    version = &record->version;
    retirement = &record->retirement;
    semantic = &record->semantic_state;
    version->document_slot = evoke_document_cow_read_u64(bytes + 0);
    version->born_sequence = evoke_document_cow_read_u64(bytes + 8);
    version->record_xid = evoke_document_cow_read_u32(bytes + 16);
    version->heap_block = evoke_document_cow_read_u32(bytes + 20);
    version->document_length = evoke_document_cow_read_u32(bytes + 24);
    version->heap_offset = evoke_document_cow_read_u16(bytes + 28);
    version->flags = evoke_document_cow_read_u16(bytes + 30);
    memcpy(
        version->semantic_input_fingerprint,
        bytes + 32,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );

    retirement->document_slot =
        evoke_document_cow_read_u64(bytes + 48);
    retirement->retirement_sequence =
        evoke_document_cow_read_u64(bytes + 56);
    retirement->record_xid =
        evoke_document_cow_read_u32(bytes + 64);
    retirement->document_length =
        evoke_document_cow_read_u32(bytes + 68);
    retirement->flags = evoke_document_cow_read_u16(bytes + 72);
    retirement->reserved = evoke_document_cow_read_u16(bytes + 74);
    retirement->reserved2 = evoke_document_cow_read_u32(bytes + 76);

    semantic->document_slot =
        evoke_document_cow_read_u64(bytes + 80);
    semantic->transition_sequence =
        evoke_document_cow_read_u64(bytes + 88);
    semantic->record_xid =
        evoke_document_cow_read_u32(bytes + 96);
    semantic->error_code =
        evoke_document_cow_read_u32(bytes + 100);
    semantic->retry_after = (int64_t)
        evoke_document_cow_read_u64(bytes + 104);
    semantic->flags = evoke_document_cow_read_u16(bytes + 112);
    semantic->failure_count =
        evoke_document_cow_read_u16(bytes + 114);
    semantic->reserved = evoke_document_cow_read_u32(bytes + 116);
    memcpy(
        semantic->semantic_input_fingerprint,
        bytes + 120,
        EVOKE_DOCUMENT_FINGERPRINT_BYTES
    );
    semantic->pending_since = (int64_t)
        evoke_document_cow_read_u64(bytes + 136);
    semantic->error_hash = evoke_document_cow_read_u64(bytes + 144);
    record->lexical_residency =
        evoke_document_cow_read_u64(bytes + 152);
    record->semantic_residency =
        evoke_document_cow_read_u64(bytes + 160);
    record->event_residency =
        evoke_document_cow_read_u64(bytes + 168);
}

static evoke_status
evoke_document_cow_object_serialized_size(
    const evoke_document_cow_object *object,
    size_t *size_out
)
{
    size_t item_size;
    size_t item_count;

    if (object == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    if (object->ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        item_size = EVOKE_DOCUMENT_COW_CHILD_SIZE;
        item_count = object->value.node.child_count;
    }
    else if (object->ref.kind == EVOKE_DOCUMENT_COW_OBJECT_LEAF)
    {
        item_size = EVOKE_DOCUMENT_COW_RECORD_SIZE;
        item_count = object->value.leaf.record_count;
    }
    else
    {
        return EVOKE_ERR_FORMAT;
    }
    if (item_count >
        (SIZE_MAX - EVOKE_DOCUMENT_COW_HEADER_SIZE) / item_size)
    {
        return EVOKE_ERR_RANGE;
    }
    *size_out = EVOKE_DOCUMENT_COW_HEADER_SIZE +
        item_count * item_size;
    return EVOKE_OK;
}

static evoke_status
evoke_document_cow_object_serialize(
    const evoke_document_cow_object *object,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    uint8_t *bytes;
    size_t size;
    uint16_t level;
    uint32_t item_count;
    uint64_t key;
    evoke_status status;

    if (bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    *bytes_out = NULL;
    *size_out = 0;
    status = evoke_document_cow_object_validate(object, false);
    if (status != EVOKE_OK)
    {
        return status;
    }
    status = evoke_document_cow_object_serialized_size(object, &size);
    if (status != EVOKE_OK)
    {
        return status;
    }
    bytes = calloc(size, 1);
    if (bytes == NULL)
    {
        return EVOKE_ERR_NOMEM;
    }
    if (object->ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        level = object->value.node.level;
        item_count = object->value.node.child_count;
        key = object->value.node.prefix;
    }
    else
    {
        level = UINT16_MAX;
        item_count = object->value.leaf.record_count;
        key = object->value.leaf.base_document_slot;
    }

    evoke_document_cow_write_u32(bytes + 0, EVOKE_DOCUMENT_COW_MAGIC);
    evoke_document_cow_write_u16(
        bytes + 4,
        EVOKE_DOCUMENT_COW_VERSION
    );
    evoke_document_cow_write_u16(
        bytes + 6,
        EVOKE_DOCUMENT_COW_HEADER_SIZE
    );
    evoke_document_cow_write_u16(
        bytes + 8,
        (uint16_t) object->ref.kind
    );
    evoke_document_cow_write_u16(bytes + 10, level);
    evoke_document_cow_write_u32(bytes + 12, item_count);
    evoke_document_cow_write_u64(bytes + 16, key);
    evoke_document_cow_write_u64(
        bytes + 24,
        object->ref.object_id
    );
    evoke_document_cow_write_u64(
        bytes + 32,
        object->ref.owner_manifest_id
    );
    evoke_document_cow_write_u64(bytes + 40, size);
    evoke_document_cow_write_u64(
        bytes + 56,
        object->ref.first_document_slot
    );
    evoke_document_cow_write_u64(
        bytes + 64,
        object->ref.document_slot_count
    );
    evoke_document_cow_write_u64(
        bytes + 72,
        object->ref.live_document_count
    );
    evoke_document_cow_write_u64(
        bytes + 80,
        object->ref.semantic_pending_count
    );
    evoke_document_cow_write_u64(
        bytes + 88,
        (uint64_t) object->ref.earliest_retry_after
    );
    evoke_document_cow_write_u64(
        bytes + 96,
        object->ref.bounded_document_count
    );
    evoke_document_cow_write_u32(
        bytes + 104,
        object->ref.min_document_length
    );
    evoke_document_cow_write_u32(
        bytes + 108,
        object->ref.max_document_length
    );
    evoke_document_cow_write_u64(
        bytes + 112,
        object->ref.reusable_document_count
    );
    evoke_document_cow_write_u64(
        bytes + 120,
        object->ref.first_reusable_document_slot
    );
    evoke_document_cow_write_u64(
        bytes + 128,
        object->ref.min_live_born_sequence
    );

    if (object->ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        for (uint32_t index = 0; index < item_count; index++)
        {
            uint8_t *child_bytes =
                bytes + EVOKE_DOCUMENT_COW_HEADER_SIZE +
                (size_t) index * EVOKE_DOCUMENT_COW_CHILD_SIZE;
            const evoke_document_cow_child *child =
                &object->value.node.children[index];

            evoke_document_cow_write_u16(
                child_bytes + 0,
                child->slot
            );
            evoke_document_cow_write_u16(
                child_bytes + 2,
                child->reserved
            );
            evoke_document_cow_write_u32(
                child_bytes + 4,
                child->reserved2
            );
            evoke_document_cow_serialize_ref(
                child_bytes + 8,
                &child->ref
            );
        }
    }
    else
    {
        for (uint32_t index = 0; index < item_count; index++)
        {
            evoke_document_cow_serialize_record(
                bytes + EVOKE_DOCUMENT_COW_HEADER_SIZE +
                    (size_t) index *
                        EVOKE_DOCUMENT_COW_RECORD_SIZE,
                &object->value.leaf.records[index]
            );
        }
    }
    evoke_document_cow_write_u64(
        bytes + EVOKE_DOCUMENT_COW_CHECKSUM_OFFSET,
        evoke_document_cow_checksum(bytes, size)
    );
    *bytes_out = bytes;
    *size_out = size;
    return EVOKE_OK;
}

static void
evoke_document_cow_refresh_local_children(
    evoke_document_cow_tree *tree,
    evoke_document_cow_object *object
)
{
    if (object->ref.kind != EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        return;
    }
    for (uint32_t index = 0;
         index < object->value.node.child_count;
         index++)
    {
        evoke_document_cow_ref *child_ref =
            &object->value.node.children[index].ref;
        evoke_document_cow_object *child =
            evoke_document_cow_tree_find_object(tree, child_ref);

        if (child != NULL)
        {
            *child_ref = child->ref;
        }
    }
    evoke_document_cow_refresh_node_summary(object);
}

evoke_status
evoke_document_cow_tree_prepare_object_for_storage(
    evoke_document_cow_tree *tree,
    uint64_t object_id,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    evoke_document_cow_ref ref = {0};
    evoke_document_cow_object *object;
    uint8_t *bytes = NULL;
    size_t size = 0;
    uint64_t checksum;
    evoke_status status;

    if (tree == NULL || bytes_out == NULL || size_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    ref.object_id = object_id;
    ref.owner_manifest_id = tree->root.owner_manifest_id;
    object = evoke_document_cow_tree_find_object(tree, &ref);
    if (object == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    evoke_document_cow_refresh_local_children(tree, object);
    if (object->ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        for (uint32_t index = 0;
             index < object->value.node.child_count;
             index++)
        {
            if (!evoke_document_cow_ref_is_bound(
                    &object->value.node.children[index].ref))
            {
                return EVOKE_ERR_FORMAT;
            }
        }
    }
    status = evoke_document_cow_object_serialize(
        object,
        &bytes,
        &size
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    checksum = evoke_segment_blob_checksum(bytes, size);
    object->ref.object_bytes = size;
    object->ref.checksum = checksum;
    if (object->ref.object_id == tree->root.object_id)
    {
        tree->root = object->ref;
    }
    *bytes_out = bytes;
    *size_out = size;
    return EVOKE_OK;
}

evoke_status
evoke_document_cow_object_bind_storage(
    evoke_document_cow_object *object,
    const evoke_segment_object_ref *storage_ref
)
{
    if (object == NULL || storage_ref == NULL ||
        storage_ref->object_kind !=
            EVOKE_SEGMENT_OBJECT_DOCUMENT_DIRECTORY ||
        storage_ref->object_id != object->ref.object_id ||
        storage_ref->owner_manifest_id !=
            object->ref.owner_manifest_id ||
        storage_ref->object_bytes != object->ref.object_bytes ||
        storage_ref->object_checksum != object->ref.checksum ||
        storage_ref->start_block == 0 ||
        storage_ref->start_block == UINT32_MAX ||
        storage_ref->page_count == 0)
    {
        return EVOKE_ERR_FORMAT;
    }
    object->ref.start_block = storage_ref->start_block;
    object->ref.page_count = storage_ref->page_count;
    return EVOKE_OK;
}

evoke_status
evoke_document_cow_tree_bind_object_storage(
    evoke_document_cow_tree *tree,
    uint64_t object_id,
    const evoke_segment_object_ref *storage_ref
)
{
    evoke_document_cow_ref ref = {0};
    evoke_document_cow_object *object;
    evoke_status status;

    if (tree == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    ref.object_id = object_id;
    ref.owner_manifest_id = tree->root.owner_manifest_id;
    object = evoke_document_cow_tree_find_object(tree, &ref);
    if (object == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    status = evoke_document_cow_object_bind_storage(
        object,
        storage_ref
    );
    if (status != EVOKE_OK)
    {
        return status;
    }
    if (object->ref.object_id == tree->root.object_id)
    {
        tree->root = object->ref;
    }
    return EVOKE_OK;
}

evoke_status
evoke_document_cow_ref_as_segment_object_ref(
    const evoke_document_cow_ref *ref,
    evoke_segment_object_ref *storage_ref_out
)
{
    if (storage_ref_out == NULL ||
        evoke_document_cow_ref_validate(ref, true) != EVOKE_OK)
    {
        return EVOKE_ERR_INVALID;
    }
    memset(storage_ref_out, 0, sizeof(*storage_ref_out));
    storage_ref_out->object_kind =
        EVOKE_SEGMENT_OBJECT_DOCUMENT_DIRECTORY;
    storage_ref_out->start_block = ref->start_block;
    storage_ref_out->page_count = ref->page_count;
    storage_ref_out->object_id = ref->object_id;
    storage_ref_out->owner_manifest_id = ref->owner_manifest_id;
    storage_ref_out->object_bytes = ref->object_bytes;
    storage_ref_out->object_checksum = ref->checksum;
    return EVOKE_OK;
}

evoke_status
evoke_document_cow_tree_object_serialize(
    const evoke_document_cow_tree *tree,
    const evoke_document_cow_ref *ref,
    uint8_t **bytes_out,
    size_t *size_out
)
{
    const evoke_document_cow_object *object;

    if (tree == NULL || ref == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    object = evoke_document_cow_tree_find_const_object(tree, ref);
    if (object == NULL)
    {
        return EVOKE_ERR_FORMAT;
    }
    return evoke_document_cow_object_serialize(
        object,
        bytes_out,
        size_out
    );
}

evoke_status
evoke_document_cow_object_deserialize(
    const uint8_t *bytes,
    size_t size,
    evoke_document_cow_object *object_out
)
{
    evoke_document_cow_object object;
    uint16_t level;
    uint32_t item_count;
    uint64_t key;
    uint64_t expected_size;
    uint64_t stored_checksum;
    size_t item_size;

    if (bytes == NULL || object_out == NULL ||
        size < EVOKE_DOCUMENT_COW_HEADER_SIZE ||
        evoke_document_cow_read_u32(bytes + 0) !=
            EVOKE_DOCUMENT_COW_MAGIC ||
        evoke_document_cow_read_u16(bytes + 4) !=
            EVOKE_DOCUMENT_COW_VERSION ||
        evoke_document_cow_read_u16(bytes + 6) !=
            EVOKE_DOCUMENT_COW_HEADER_SIZE)
    {
        return EVOKE_ERR_FORMAT;
    }
    memset(&object, 0, sizeof(object));
    object.ref.kind = (evoke_document_cow_object_kind)
        evoke_document_cow_read_u16(bytes + 8);
    level = evoke_document_cow_read_u16(bytes + 10);
    item_count = evoke_document_cow_read_u32(bytes + 12);
    key = evoke_document_cow_read_u64(bytes + 16);
    object.ref.object_id = evoke_document_cow_read_u64(bytes + 24);
    object.ref.owner_manifest_id =
        evoke_document_cow_read_u64(bytes + 32);
    expected_size = evoke_document_cow_read_u64(bytes + 40);
    stored_checksum = evoke_document_cow_read_u64(bytes + 48);
    object.ref.first_document_slot =
        evoke_document_cow_read_u64(bytes + 56);
    object.ref.document_slot_count =
        evoke_document_cow_read_u64(bytes + 64);
    object.ref.live_document_count =
        evoke_document_cow_read_u64(bytes + 72);
    object.ref.semantic_pending_count =
        evoke_document_cow_read_u64(bytes + 80);
    object.ref.earliest_retry_after = (int64_t)
        evoke_document_cow_read_u64(bytes + 88);
    object.ref.bounded_document_count =
        evoke_document_cow_read_u64(bytes + 96);
    object.ref.min_document_length =
        evoke_document_cow_read_u32(bytes + 104);
    object.ref.max_document_length =
        evoke_document_cow_read_u32(bytes + 108);
    object.ref.reusable_document_count =
        evoke_document_cow_read_u64(bytes + 112);
    object.ref.first_reusable_document_slot =
        evoke_document_cow_read_u64(bytes + 120);
    object.ref.min_live_born_sequence =
        evoke_document_cow_read_u64(bytes + 128);
    if (object.ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        item_size = EVOKE_DOCUMENT_COW_CHILD_SIZE;
        if (level >= EVOKE_DOCUMENT_COW_RADIX_LEVELS ||
            item_count > EVOKE_DOCUMENT_COW_RADIX_FANOUT)
        {
            return EVOKE_ERR_FORMAT;
        }
        object.value.node.level = level;
        object.value.node.prefix = key;
        object.value.node.child_count = item_count;
    }
    else if (object.ref.kind == EVOKE_DOCUMENT_COW_OBJECT_LEAF)
    {
        item_size = EVOKE_DOCUMENT_COW_RECORD_SIZE;
        if (level != UINT16_MAX ||
            item_count == 0 ||
            item_count > EVOKE_DOCUMENT_COW_LEAF_RECORDS)
        {
            return EVOKE_ERR_FORMAT;
        }
        object.value.leaf.base_document_slot = key;
        object.value.leaf.record_count = item_count;
    }
    else
    {
        return EVOKE_ERR_FORMAT;
    }
    if (item_count >
            (SIZE_MAX - EVOKE_DOCUMENT_COW_HEADER_SIZE) / item_size ||
        expected_size !=
            EVOKE_DOCUMENT_COW_HEADER_SIZE +
                (size_t) item_count * item_size ||
        expected_size != size ||
        stored_checksum != evoke_document_cow_checksum(bytes, size))
    {
        return EVOKE_ERR_FORMAT;
    }
    object.ref.object_bytes = size;
    object.ref.checksum = evoke_segment_blob_checksum(bytes, size);

    if (object.ref.kind == EVOKE_DOCUMENT_COW_OBJECT_NODE)
    {
        for (uint32_t index = 0; index < item_count; index++)
        {
            const uint8_t *child_bytes =
                bytes + EVOKE_DOCUMENT_COW_HEADER_SIZE +
                (size_t) index * EVOKE_DOCUMENT_COW_CHILD_SIZE;
            evoke_document_cow_child *child =
                &object.value.node.children[index];

            child->slot =
                evoke_document_cow_read_u16(child_bytes + 0);
            child->reserved =
                evoke_document_cow_read_u16(child_bytes + 2);
            child->reserved2 =
                evoke_document_cow_read_u32(child_bytes + 4);
            evoke_document_cow_deserialize_ref(
                child_bytes + 8,
                &child->ref
            );
        }
    }
    else
    {
        for (uint32_t index = 0; index < item_count; index++)
        {
            evoke_document_cow_deserialize_record(
                bytes + EVOKE_DOCUMENT_COW_HEADER_SIZE +
                    (size_t) index *
                        EVOKE_DOCUMENT_COW_RECORD_SIZE,
                &object.value.leaf.records[index]
            );
        }
    }
    if (evoke_document_cow_object_validate(&object, false) != EVOKE_OK)
    {
        return EVOKE_ERR_FORMAT;
    }
    *object_out = object;
    return EVOKE_OK;
}
