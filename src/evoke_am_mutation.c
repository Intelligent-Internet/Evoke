#include "postgres.h"

#include <errno.h>
#include <stdlib.h>

#include "access/generic_xlog.h"
#include "access/transam.h"
#include "access/xact.h"
#include "access/xlog.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "storage/lwlock.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/rel.h"

#include "evoke_am_maintenance.h"
#include "evoke_am_meta.h"
#include "evoke_am_mutation.h"
#include "evoke_core.h"
#include "evoke_segment_pages.h"

#define EVOKE_AM_FROZEN_XID_AUDIT_SETTING \
    "evoke.test_require_frozen_delta_xids"
#define EVOKE_AM_FROZEN_XID_AUDIT_DESCRIPTION "delta XID freeze audit"

static bool
evoke_am_frozen_xid_audit_enabled(void)
{
    const char *setting = GetConfigOptionByName(
        EVOKE_AM_FROZEN_XID_AUDIT_SETTING,
        NULL,
        true
    );
    bool enabled = false;

    if (setting == NULL || setting[0] == '\0')
    {
        return false;
    }
    if (!parse_bool(setting, &enabled))
    {
        ereport(
            ERROR,
            (
                errmsg(
                    "%s must be boolean",
                    EVOKE_AM_FROZEN_XID_AUDIT_SETTING
                )
            )
        );
    }
    if (!enabled)
    {
        return false;
    }
    if (!superuser_arg(GetOuterUserId()))
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
                errmsg(
                    "%s failure injection is superuser-only",
                    EVOKE_AM_FROZEN_XID_AUDIT_DESCRIPTION
                )
            )
        );
    }
    return true;
}

uint32
evoke_am_active_l0_rotation_record_limit(void)
{
    const char *setting = GetConfigOptionByName(
        "evoke.test_convergent_l0_rotation_records",
        NULL,
        true
    );
    char *end = NULL;
    unsigned long long value;

    if (setting == NULL || setting[0] == '\0')
    {
        return EVOKE_ACTIVE_L0_ROTATION_RECORDS;
    }
    if (!superuser_arg(GetOuterUserId()))
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
                errmsg(
                    "evoke convergent-L0 test threshold is "
                    "superuser-only"
                )
            )
        );
    }
    errno = 0;
    value = strtoull(setting, &end, 10);
    if (errno != 0 || end == setting || *end != '\0' ||
        value == 0 || value > EVOKE_ACTIVE_L0_MAX_RECORDS)
    {
        ereport(
            ERROR,
            (errmsg(
                "evoke.test_convergent_l0_rotation_records must be "
                "between 1 and %u",
                EVOKE_ACTIVE_L0_MAX_RECORDS
            ))
        );
    }
    return (uint32) value;
}

bool
evoke_am_active_l0_checkpoint_due(const evoke_segment_read_root *root)
{
    return root != NULL && root->active_l0.record_count > 0 &&
        (root->active_l0.record_count >=
             evoke_am_active_l0_rotation_record_limit() ||
         root->active_l0.page_count >= EVOKE_ACTIVE_L0_ROTATION_PAGES);
}

uint32
evoke_am_accelerator_refresh_record_limit(void)
{
    return evoke_am_active_l0_rotation_record_limit();
}

bool
evoke_am_delta_record_states(
    const TransactionId *record_xids,
    evoke_am_delta_xid_state *states_out,
    uint32 record_count
)
{
    volatile bool truncation_lock_held = false;
    bool has_normal_xid = false;

    if ((record_xids == NULL || states_out == NULL) &&
        record_count > 0)
    {
        ereport(ERROR, (errmsg("evoke delta XID batch is invalid")));
    }
    for (uint32 i = 0; i < record_count; i++)
    {
        TransactionId record_xid = record_xids[i];

        if (TransactionIdIsNormal(record_xid) &&
            !TransactionIdIsCurrentTransactionId(record_xid))
        {
            has_normal_xid = true;
            break;
        }
    }
    if (has_normal_xid && evoke_am_frozen_xid_audit_enabled())
    {
        ereport(ERROR, (errmsg("evoke delta record XID is not frozen")));
    }

    PG_TRY();
    {
        if (has_normal_xid)
        {
            LWLockAcquire(XactTruncationLock, LW_SHARED);
            truncation_lock_held = true;
        }
        for (uint32 i = 0; i < record_count; i++)
        {
            TransactionId record_xid = record_xids[i];

            if (record_xid == EVOKE_AM_ABORTED_DELTA_XID)
            {
                states_out[i] = EVOKE_AM_DELTA_XID_ABORTED;
            }
            else if (!TransactionIdIsValid(record_xid) ||
                     record_xid == FrozenTransactionId ||
                     !TransactionIdIsNormal(record_xid))
            {
                states_out[i] = EVOKE_AM_DELTA_XID_COMMITTED;
            }
            else if (TransactionIdIsCurrentTransactionId(record_xid))
            {
                states_out[i] = EVOKE_AM_DELTA_XID_UNRESOLVED;
            }
            else if (TransactionIdPrecedes(
                         record_xid,
                         TransamVariables->oldestClogXid) ||
                     TransactionIdDidCommit(record_xid))
            {
                states_out[i] = EVOKE_AM_DELTA_XID_COMMITTED;
            }
            else if (TransactionIdDidAbort(record_xid))
            {
                states_out[i] = EVOKE_AM_DELTA_XID_ABORTED;
            }
            else
            {
                states_out[i] = EVOKE_AM_DELTA_XID_UNRESOLVED;
            }
        }
    }
    PG_FINALLY();
    {
        if (truncation_lock_held)
        {
            LWLockRelease(XactTruncationLock);
        }
    }
    PG_END_TRY();
    return has_normal_xid;
}

bool
evoke_am_l0_expected_source_matches(
    uint8 record_kind,
    const evoke_document_cow_record *expected,
    const evoke_document_cow_record *current
)
{
    if (record_kind != EVOKE_L0_RECORD_RETIRE)
    {
        return evoke_document_cow_records_equal(expected, current);
    }

    /*
     * Semantic completion and quarantine may change the mutable state of a
     * live record after VACUUM has proved its heap TID dead. RETIRE guards
     * therefore bind to the immutable slot incarnation rather than to that
     * complete mutable state. Born sequence plus heap TID prevents a reused
     * slot from inheriting the retirement.
     */
    return expected != NULL && current != NULL &&
        expected->version.document_slot ==
            current->version.document_slot &&
        expected->version.born_sequence ==
            current->version.born_sequence &&
        expected->version.heap_block == current->version.heap_block &&
        expected->version.heap_offset == current->version.heap_offset &&
        expected->version.document_length ==
            current->version.document_length &&
        memcmp(
            expected->version.semantic_input_fingerprint,
            current->version.semantic_input_fingerprint,
            EVOKE_DOCUMENT_FINGERPRINT_BYTES) == 0;
}

static void
evoke_am_set_generic_page_content_len(Page page, Size content_len)
{
    PageHeader header = (PageHeader) page;
    Size content_start = MAXALIGN(SizeOfPageHeaderData);
    Size content_end;

    if (content_len > BLCKSZ - content_start)
    {
        ereport(ERROR, (errmsg("evoke page content is too large")));
    }
    content_end = content_start + content_len;
    if (content_end > header->pd_upper)
    {
        ereport(ERROR, (errmsg("evoke page content overlaps page storage")));
    }
    header->pd_lower = (LocationIndex) content_end;
}

static Size
evoke_am_l0_page_content_bytes(void)
{
    return BLCKSZ - MAXALIGN(SizeOfPageHeaderData);
}

static Size
evoke_am_l0_page_payload_capacity(void)
{
    return evoke_am_l0_page_content_bytes() -
        EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE;
}

void
evoke_am_l0_require_meta_root(
    const evoke_am_meta_page *meta,
    const evoke_am_meta_page *expected
)
{
    if (meta == NULL || expected == NULL ||
        meta->magic != EVOKE_AM_MAGIC ||
        meta->version != EVOKE_AM_VERSION ||
        meta->page_kind != EVOKE_AM_PAGE_META ||
        !evoke_am_meta_uses_convergent_segment_storage(meta) ||
        meta->source_type != expected->source_type ||
        memcmp(
            meta->segment_read_root_bytes,
            expected->segment_read_root_bytes,
            sizeof(meta->segment_read_root_bytes)
        ) != 0)
    {
        ereport(
            ERROR,
            (errmsg("evoke convergent segment root changed during append"))
        );
    }
}

static void
evoke_am_l0_read_tail_header(
    Page page,
    const evoke_segment_read_root *root,
    evoke_active_l0_page_header *header_out
)
{
    PageHeader postgres_header = (PageHeader) page;
    const uint8 *content;
    Size minimum_content;
    evoke_status status;

    if (page == NULL || root == NULL || header_out == NULL ||
        root->active_l0.record_count == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke active L0 tail request")));
    }
    content = (const uint8 *) PageGetContents(page);
    status = evoke_active_l0_page_header_deserialize(
        content,
        EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE,
        evoke_am_l0_page_content_bytes(),
        header_out
    );
    if (status != EVOKE_OK)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke active L0 page header"),
                errdetail(
                    "Validation failed: %s.",
                    evoke_strerror(status)
                )
            )
        );
    }
    minimum_content =
        MAXALIGN(SizeOfPageHeaderData) +
        EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE +
        header_out->used_bytes;
    if (postgres_header->pd_lower < minimum_content ||
        header_out->segment_id != root->active_l0.segment_id ||
        header_out->ordinal + 1 != root->active_l0.page_count ||
        header_out->max_sequence != root->active_l0.max_sequence ||
        header_out->next_block != EVOKE_ACTIVE_L0_NO_NEXT_BLOCK)
    {
        ereport(ERROR, (errmsg("evoke active L0 tail does not match root")));
    }
    status = evoke_active_l0_page_payload_validate(
        header_out,
        content + EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE,
        header_out->used_bytes
    );
    if (status != EVOKE_OK)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke active L0 page payload"),
                errdetail(
                    "Validation failed: %s.",
                    evoke_strerror(status)
                )
            )
        );
    }
}

static void
evoke_am_l0_advance_root(
    evoke_segment_read_root *root,
    const evoke_l0_record *record,
    Size record_size,
    const evoke_l0_chain_write_result *chain,
    uint32 published_block_high_watermark
)
{
    evoke_active_l0_frontier *frontier;
    bool advances_document_high_watermark;
    bool was_empty;
    evoke_status status;

    if (root == NULL || record == NULL ||
        (record->kind != EVOKE_L0_RECORD_UPSERT &&
         record->kind != EVOKE_L0_RECORD_RETIRE &&
         record->kind != EVOKE_L0_RECORD_SEMANTIC_COMPLETE &&
         record->kind != EVOKE_L0_RECORD_SEMANTIC_QUARANTINE) ||
        record->sequence != root->next_sequence ||
        root->next_sequence == UINT64_MAX ||
        record_size == 0 ||
        published_block_high_watermark <= 1)
    {
        ereport(ERROR, (errmsg("invalid evoke active L0 root advance")));
    }
    advances_document_high_watermark =
        record->kind == EVOKE_L0_RECORD_UPSERT &&
        record->document_slot == root->next_document_slot;
    if ((record->kind == EVOKE_L0_RECORD_UPSERT &&
         (record->document_slot > root->next_document_slot ||
          (advances_document_high_watermark &&
           root->next_document_slot >= UINT32_MAX))) ||
        (record->kind != EVOKE_L0_RECORD_UPSERT &&
         record->document_slot >= root->next_document_slot))
    {
        ereport(ERROR, (errmsg("invalid evoke active L0 document slot")));
    }

    frontier = &root->active_l0;
    was_empty = frontier->record_count == 0;
    if (frontier->record_count == UINT32_MAX ||
        frontier->payload_bytes > UINT64_MAX - (uint64) record_size)
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                errmsg("evoke active L0 frontier is exhausted")
            )
        );
    }

    if (chain != NULL)
    {
        if (chain->page_count == 0 ||
            chain->head_block == 0 ||
            chain->tail_block == 0 ||
            (uint64) frontier->page_count + chain->page_count >
                EVOKE_ACTIVE_L0_MAX_PAGES)
        {
            ereport(ERROR, (errmsg("invalid evoke active L0 chain")));
        }
        if (was_empty)
        {
            frontier->head_block = chain->head_block;
            frontier->min_sequence = record->sequence;
        }
        frontier->tail_block = chain->tail_block;
        frontier->page_count += chain->page_count;
    }
    else if (was_empty)
    {
        ereport(ERROR, (errmsg("empty evoke active L0 requires a chain")));
    }

    frontier->max_sequence = record->sequence;
    frontier->record_count++;
    frontier->payload_bytes += (uint64) record_size;
    root->next_sequence++;
    if (advances_document_high_watermark)
    {
        root->next_document_slot++;
    }
    if (record->kind == EVOKE_L0_RECORD_UPSERT)
    {
        root->reusable_document_slot_cursor =
            (uint32) (record->document_slot + 1);
    }
    root->published_block_high_watermark =
        published_block_high_watermark;
    status = evoke_segment_read_root_validate(root);
    if (status != EVOKE_OK)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke active L0 read root"),
                errdetail(
                    "Validation failed: %s.",
                    evoke_strerror(status)
                )
            )
        );
    }
}

void
evoke_am_l0_store_root(
    Page meta_page,
    const evoke_segment_read_root *root
)
{
    evoke_am_meta_page *meta;
    evoke_status status;

    meta = (evoke_am_meta_page *) PageGetContents(meta_page);
    status = evoke_segment_read_root_serialize(
        root,
        meta->segment_read_root_bytes,
        sizeof(meta->segment_read_root_bytes)
    );
    if (status != EVOKE_OK)
    {
        ereport(
            ERROR,
            (
                errmsg("failed to serialize evoke active L0 read root"),
                errdetail(
                    "Validation failed: %s.",
                    evoke_strerror(status)
                )
            )
        );
    }
    /*
     * Linked L0 is the complete mutation-debt authority for automatic v3
     * policies.  Clear counters retained for manual consistency whenever a
     * page-native root advances so retired counter accounting cannot survive a
     * rotate, seal, or COW publication.
     */
    meta->pending_write_tuples = 0;
    meta->pending_delete_tuples = 0;
    evoke_am_set_generic_page_content_len(meta_page, sizeof(*meta));
}

bool
evoke_am_mutation_publish_cow_manifest(
    Relation index_relation,
    const evoke_segment_read_root *build_root,
    const evoke_segment_manifest *next_manifest,
    const evoke_segment_cow_result *cow_result,
    bool seal_pending,
    bool require_exact_frontiers,
    bool publish_fsm_handoff
)
{
    evoke_am_meta_page latest_meta;
    evoke_segment_read_root latest_root;
    evoke_segment_read_root next_root;
    Buffer meta_buffer;
    BlockNumber nblocks;
    volatile bool append_locked = false;
    bool published = false;
    evoke_status status;
    XLogRecPtr publish_lsn = InvalidXLogRecPtr;

    if (index_relation == NULL || build_root == NULL ||
        next_manifest == NULL || cow_result == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke COW manifest publication")));
    }

    evoke_am_lock_append(index_relation);
    append_locked = true;
    PG_TRY();
    {
        evoke_am_read_meta(index_relation, &latest_meta);
        status = evoke_am_segment_read_root_from_meta(
            &latest_meta,
            &latest_root
        );
        if (!evoke_am_meta_uses_convergent_segment_storage(
                &latest_meta) ||
            status != EVOKE_OK)
        {
            ereport(ERROR, (errmsg("invalid evoke latest segment root")));
        }
        if (latest_root.root_id != build_root->root_id ||
            latest_root.next_segment_id !=
                build_root->next_segment_id ||
            !evoke_segment_object_ref_equal(
                &latest_root.manifest,
                &build_root->manifest) ||
            (require_exact_frontiers &&
             (!evoke_active_l0_frontier_equal(
                  &latest_root.active_l0,
                  &build_root->active_l0) ||
              !evoke_active_l0_frontier_equal(
                  &latest_root.pending_l0,
                  &build_root->pending_l0))) ||
            (seal_pending &&
             !evoke_active_l0_frontier_equal(
                &latest_root.pending_l0,
                &build_root->pending_l0)))
        {
            published = false;
        }
        else
        {
            nblocks = RelationGetNumberOfBlocks(index_relation);
            if (nblocks > UINT32_MAX)
            {
                ereport(
                    ERROR,
                    (errmsg("evoke relation block range is exhausted"))
                );
            }
            status = evoke_segment_manifest_validate_published(
                next_manifest,
                &cow_result->manifest,
                (uint32) nblocks
            );
            if (status != EVOKE_OK)
            {
                evoke_am_maintenance_codec_error(
                    "validate the published manifest",
                    status
                );
            }
            next_root = latest_root;
            status = seal_pending
                ? evoke_segment_read_root_seal_pending(
                      &next_root,
                      &cow_result->manifest,
                      (uint32) nblocks
                  )
                : evoke_segment_read_root_replace_manifest(
                      &next_root,
                      &cow_result->manifest,
                      (uint32) nblocks
                  );
            if (status != EVOKE_OK)
            {
                evoke_am_maintenance_codec_error(
                    "advance the segment root",
                    status
                );
            }
            meta_buffer = ReadBufferExtended(
                index_relation,
                MAIN_FORKNUM,
                0,
                RBM_NORMAL,
                NULL
            );
            LockBuffer(meta_buffer, BUFFER_LOCK_EXCLUSIVE);
            evoke_am_l0_require_meta_root(
                (const evoke_am_meta_page *) PageGetContents(
                    BufferGetPage(meta_buffer)
                ),
                &latest_meta
            );
            {
                GenericXLogState *xlog_state =
                    GenericXLogStart(index_relation);
                Page next_meta_page = GenericXLogRegisterBuffer(
                    xlog_state,
                    meta_buffer,
                    0
                );
                evoke_am_meta_page *next_meta =
                    (evoke_am_meta_page *) PageGetContents(
                        next_meta_page
                    );

                next_meta->num_docs =
                    (uint32) next_manifest->visible_document_count;
                evoke_am_l0_store_root(next_meta_page, &next_root);
                publish_lsn = GenericXLogFinish(xlog_state);
            }
            UnlockReleaseBuffer(meta_buffer);
            if (RelationNeedsWAL(index_relation) &&
                publish_lsn != InvalidXLogRecPtr)
            {
                /*
                 * The root is the commit point for every earlier immutable
                 * object. Make successful maintenance durable before exposing
                 * its result or handing unreachable pages to the FSM.
                 */
                XLogFlush(publish_lsn);
            }
            if (publish_fsm_handoff)
            {
                evoke_segment_pages_publish_fsm_handoff(
                    index_relation,
                    next_manifest->manifest_id,
                    cow_result
                );
            }
            published = true;
        }
    }
    PG_FINALLY();
    {
        if (append_locked)
        {
            evoke_am_unlock_append(index_relation);
        }
    }
    PG_END_TRY();
    return published;
}

XLogRecPtr
evoke_am_l0_rotate_active_locked(
    Relation indexRelation,
    evoke_am_meta_page *expected_meta,
    evoke_segment_read_root *root
)
{
    Buffer meta_buffer;
    evoke_segment_read_root next_root;
    evoke_status status;
    XLogRecPtr rotation_lsn = InvalidXLogRecPtr;

    if (indexRelation == NULL || expected_meta == NULL || root == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke active L0 rotation")));
    }
    next_root = *root;
    status = evoke_segment_read_root_rotate_l0(&next_root);
    if (status != EVOKE_OK)
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                errmsg("evoke active L0 rotation is unavailable"),
                errdetail(
                    "Validation failed: %s.",
                    evoke_strerror(status)
                )
            )
        );
    }

    meta_buffer = ReadBufferExtended(
        indexRelation,
        MAIN_FORKNUM,
        0,
        RBM_NORMAL,
        NULL
    );
    LockBuffer(meta_buffer, BUFFER_LOCK_EXCLUSIVE);
    evoke_am_l0_require_meta_root(
        (const evoke_am_meta_page *) PageGetContents(
            BufferGetPage(meta_buffer)
        ),
        expected_meta
    );
    {
        GenericXLogState *state = GenericXLogStart(indexRelation);
        Page next_meta_page = GenericXLogRegisterBuffer(
            state,
            meta_buffer,
            0
        );

        evoke_am_l0_store_root(next_meta_page, &next_root);
        rotation_lsn = GenericXLogFinish(state);
    }
    UnlockReleaseBuffer(meta_buffer);

    status = evoke_segment_read_root_serialize(
        &next_root,
        expected_meta->segment_read_root_bytes,
        sizeof(expected_meta->segment_read_root_bytes)
    );
    if (status != EVOKE_OK)
    {
        ereport(ERROR, (errmsg("failed to retain rotated evoke L0 root")));
    }
    *root = next_root;
    return rotation_lsn;
}

static bool
evoke_am_l0_tail_can_fit(
    const evoke_active_l0_page_header *header,
    Size record_size
)
{
    Size required;

    if (header == NULL ||
        record_size > SIZE_MAX - EVOKE_L0_FRAME_HEADER_SIZE)
    {
        return false;
    }
    required = EVOKE_L0_FRAME_HEADER_SIZE + record_size;
    return header->used_bytes <= evoke_am_l0_page_payload_capacity() &&
        required <=
            evoke_am_l0_page_payload_capacity() - header->used_bytes;
}

static void
evoke_am_l0_append_inline(
    Relation indexRelation,
    const evoke_am_meta_page *expected_meta,
    const evoke_segment_read_root *expected_root,
    const evoke_l0_record *record,
    const uint8 *record_bytes,
    Size record_size
)
{
    Buffer meta_buffer = InvalidBuffer;
    Buffer tail_buffer = InvalidBuffer;
    evoke_active_l0_page_header header;
    evoke_segment_read_root next_root;
    evoke_l0_frame_header frame;
    uint8 frame_bytes[EVOKE_L0_FRAME_HEADER_SIZE];
    evoke_status status;

    meta_buffer = ReadBufferExtended(
        indexRelation,
        MAIN_FORKNUM,
        0,
        RBM_NORMAL,
        NULL
    );
    LockBuffer(meta_buffer, BUFFER_LOCK_EXCLUSIVE);
    evoke_am_l0_require_meta_root(
        (const evoke_am_meta_page *) PageGetContents(
            BufferGetPage(meta_buffer)
        ),
        expected_meta
    );
    tail_buffer = ReadBufferExtended(
        indexRelation,
        MAIN_FORKNUM,
        expected_root->active_l0.tail_block,
        RBM_NORMAL,
        NULL
    );
    LockBuffer(tail_buffer, BUFFER_LOCK_EXCLUSIVE);
    evoke_am_l0_read_tail_header(
        BufferGetPage(tail_buffer),
        expected_root,
        &header
    );
    if (!evoke_am_l0_tail_can_fit(&header, record_size) ||
        header.frame_count == UINT32_MAX)
    {
        UnlockReleaseBuffer(tail_buffer);
        UnlockReleaseBuffer(meta_buffer);
        ereport(ERROR, (errmsg("evoke active L0 tail capacity changed")));
    }

    memset(&frame, 0, sizeof(frame));
    frame.flags = EVOKE_L0_FRAME_FLAG_START | EVOKE_L0_FRAME_FLAG_END;
    frame.sequence = record->sequence;
    frame.record_checksum = evoke_segment_blob_checksum(
        record_bytes,
        record_size
    );
    frame.record_bytes = (uint32) record_size;
    frame.fragment_bytes = (uint32) record_size;
    status = evoke_l0_frame_header_serialize(
        &frame,
        frame_bytes,
        sizeof(frame_bytes)
    );
    if (status != EVOKE_OK)
    {
        UnlockReleaseBuffer(tail_buffer);
        UnlockReleaseBuffer(meta_buffer);
        ereport(ERROR, (errmsg("failed to encode evoke active L0 frame")));
    }

    next_root = *expected_root;
    evoke_am_l0_advance_root(
        &next_root,
        record,
        record_size,
        NULL,
        (uint32) RelationGetNumberOfBlocks(indexRelation)
    );
    {
        GenericXLogState *state = GenericXLogStart(indexRelation);
        Page next_meta_page = GenericXLogRegisterBuffer(
            state,
            meta_buffer,
            0
        );
        Page next_tail_page = GenericXLogRegisterBuffer(
            state,
            tail_buffer,
            0
        );
        uint8 *content = (uint8 *) PageGetContents(next_tail_page);
        Size offset =
            EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE + header.used_bytes;

        memcpy(content + offset, frame_bytes, sizeof(frame_bytes));
        memcpy(
            content + offset + sizeof(frame_bytes),
            record_bytes,
            record_size
        );
        header.max_sequence = record->sequence;
        header.frame_count++;
        header.used_bytes +=
            (uint32) (sizeof(frame_bytes) + record_size);
        header.payload_checksum = evoke_segment_blob_checksum(
            content + EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE,
            header.used_bytes
        );
        status = evoke_active_l0_page_header_serialize(
            &header,
            evoke_am_l0_page_content_bytes(),
            content,
            EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE
        );
        if (status != EVOKE_OK)
        {
            ereport(ERROR, (errmsg("failed to update evoke active L0 tail")));
        }
        evoke_am_set_generic_page_content_len(
            next_tail_page,
            EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE + header.used_bytes
        );
        evoke_am_l0_store_root(next_meta_page, &next_root);
        (void) GenericXLogFinish(state);
    }
    UnlockReleaseBuffer(tail_buffer);
    UnlockReleaseBuffer(meta_buffer);
}

static void
evoke_am_l0_publish_chain(
    Relation indexRelation,
    const evoke_am_meta_page *expected_meta,
    const evoke_segment_read_root *expected_root,
    const evoke_l0_record *record,
    Size record_size,
    const evoke_l0_chain_write_result *chain
)
{
    Buffer meta_buffer = InvalidBuffer;
    Buffer tail_buffer = InvalidBuffer;
    evoke_active_l0_page_header tail_header;
    evoke_segment_read_root next_root;
    BlockNumber nblocks;
    evoke_status status;

    nblocks = RelationGetNumberOfBlocks(indexRelation);
    if (nblocks > UINT32_MAX ||
        chain == NULL ||
        chain->tail_block >= nblocks)
    {
        ereport(ERROR, (errmsg("invalid evoke active L0 publication")));
    }

    meta_buffer = ReadBufferExtended(
        indexRelation,
        MAIN_FORKNUM,
        0,
        RBM_NORMAL,
        NULL
    );
    LockBuffer(meta_buffer, BUFFER_LOCK_EXCLUSIVE);
    evoke_am_l0_require_meta_root(
        (const evoke_am_meta_page *) PageGetContents(
            BufferGetPage(meta_buffer)
        ),
        expected_meta
    );
    if (expected_root->active_l0.record_count != 0)
    {
        tail_buffer = ReadBufferExtended(
            indexRelation,
            MAIN_FORKNUM,
            expected_root->active_l0.tail_block,
            RBM_NORMAL,
            NULL
        );
        LockBuffer(tail_buffer, BUFFER_LOCK_EXCLUSIVE);
        evoke_am_l0_read_tail_header(
            BufferGetPage(tail_buffer),
            expected_root,
            &tail_header
        );
    }

    next_root = *expected_root;
    evoke_am_l0_advance_root(
        &next_root,
        record,
        record_size,
        chain,
        (uint32) nblocks
    );
    {
        GenericXLogState *state = GenericXLogStart(indexRelation);
        Page next_meta_page = GenericXLogRegisterBuffer(
            state,
            meta_buffer,
            0
        );

        if (BufferIsValid(tail_buffer))
        {
            Page next_tail_page = GenericXLogRegisterBuffer(
                state,
                tail_buffer,
                0
            );
            uint8 *content =
                (uint8 *) PageGetContents(next_tail_page);

            tail_header.next_block = chain->head_block;
            status = evoke_active_l0_page_header_serialize(
                &tail_header,
                evoke_am_l0_page_content_bytes(),
                content,
                EVOKE_ACTIVE_L0_PAGE_HEADER_SIZE
            );
            if (status != EVOKE_OK)
            {
                ereport(
                    ERROR,
                    (errmsg("failed to link evoke active L0 chain"))
                );
            }
        }
        evoke_am_l0_store_root(next_meta_page, &next_root);
        (void) GenericXLogFinish(state);
    }
    if (BufferIsValid(tail_buffer))
    {
        UnlockReleaseBuffer(tail_buffer);
    }
    UnlockReleaseBuffer(meta_buffer);
}

static uint32
evoke_am_l0_record_chain_pages(Size record_size)
{
    Size fragment_capacity;
    uint64 page_count;

    fragment_capacity = evoke_am_l0_page_payload_capacity() -
        EVOKE_L0_FRAME_HEADER_SIZE;
    if (record_size == 0 || fragment_capacity == 0)
    {
        ereport(ERROR, (errmsg("invalid evoke active L0 record size")));
    }
    page_count = 1 +
        ((uint64) record_size - 1) / (uint64) fragment_capacity;
    if (page_count > EVOKE_ACTIVE_L0_MAX_PAGES)
    {
        ereport(
            ERROR,
            (
                errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                errmsg("evoke active L0 record exceeds the page bound")
            )
        );
    }
    return (uint32) page_count;
}

static uint64
evoke_am_l0_choose_upsert_document_slot(
    Relation index_relation,
    const evoke_segment_read_root *root
)
{
    evoke_segment_manifest manifest;
    evoke_document_cow_record reusable;
    uint64 selected_slot = root->next_document_slot;

    evoke_segment_manifest_init(&manifest);
    PG_TRY();
    {
        evoke_segment_pages_load_maintenance_manifest(
            index_relation,
            root,
            &manifest
        );
        if (root->reusable_document_slot_cursor <
                manifest.document_slot_count &&
            evoke_segment_pages_find_reusable_document_from(
                index_relation,
                root,
                &manifest,
                root->reusable_document_slot_cursor,
                &reusable))
        {
            if (reusable.version.document_slot >=
                    root->next_document_slot ||
                reusable.version.document_slot <
                    root->reusable_document_slot_cursor)
            {
                ereport(
                    ERROR,
                    (errmsg("invalid evoke reusable document slot"))
                );
            }
            selected_slot = reusable.version.document_slot;
        }
    }
    PG_FINALLY();
    {
        evoke_segment_manifest_free(&manifest);
    }
    PG_END_TRY();
    return selected_slot;
}

bool
evoke_am_mutation_append_l0_record(
    Relation indexRelation,
    evoke_l0_record *record,
    const evoke_document_cow_record *expected_source,
    uint64 expected_l0_born_sequence,
    bool *maintenance_due_out
)
{
    evoke_am_meta_page meta;
    evoke_segment_read_root root;
    evoke_segment_manifest source_manifest;
    evoke_l0_chain_write_result chain;
    uint8 *record_bytes = NULL;
    size_t record_size = 0;
    volatile bool append_locked = false;
    bool appended = false;
    evoke_status status;

    if (maintenance_due_out == NULL)
    {
        ereport(ERROR, (errmsg("invalid evoke maintenance outcome")));
    }
    *maintenance_due_out = false;
    if (indexRelation == NULL || record == NULL ||
        (record->kind != EVOKE_L0_RECORD_UPSERT &&
         record->kind != EVOKE_L0_RECORD_RETIRE &&
         record->kind != EVOKE_L0_RECORD_SEMANTIC_COMPLETE &&
         record->kind != EVOKE_L0_RECORD_SEMANTIC_QUARANTINE))
    {
        ereport(ERROR, (errmsg("invalid evoke active L0 record")));
    }
    if ((expected_source != NULL &&
         ((record->kind != EVOKE_L0_RECORD_RETIRE &&
           record->kind != EVOKE_L0_RECORD_SEMANTIC_COMPLETE &&
           record->kind != EVOKE_L0_RECORD_SEMANTIC_QUARANTINE) ||
          record->document_slot !=
            expected_source->version.document_slot)) ||
        (expected_l0_born_sequence != 0 &&
         (expected_source != NULL ||
          record->kind != EVOKE_L0_RECORD_RETIRE)))
    {
        ereport(ERROR, (errmsg("invalid evoke L0 source guard")));
    }

    evoke_segment_manifest_init(&source_manifest);
    evoke_am_pin_maintenance_xact(indexRelation);
    evoke_am_lock_append(indexRelation);
    append_locked = true;
    PG_TRY();
    {
        bool inline_append = false;
        bool rotate_active = false;
        uint32 chain_pages;
        uint32 rotation_record_limit;

        evoke_am_read_meta(indexRelation, &meta);
        if (!evoke_am_meta_uses_convergent_segment_storage(&meta))
        {
            ereport(ERROR, (errmsg("evoke convergent segment root is missing")));
        }
        status = evoke_am_segment_read_root_from_meta(&meta, &root);
        if (status != EVOKE_OK)
        {
            ereport(ERROR, (errmsg("invalid evoke convergent segment root")));
        }
        if (expected_source != NULL)
        {
            evoke_document_cow_record current_source;

            if (record->document_slot >= root.next_document_slot)
            {
                goto append_done;
            }
            evoke_segment_pages_load_maintenance_manifest(
                indexRelation,
                &root,
                &source_manifest
            );
            if (record->document_slot >=
                source_manifest.document_slot_count)
            {
                goto append_done;
            }
            evoke_segment_pages_load_document_record(
                indexRelation,
                &root,
                &source_manifest,
                record->document_slot,
                &current_source
            );
            if (!evoke_am_l0_expected_source_matches(
                    record->kind,
                    expected_source,
                    &current_source) ||
                (record->kind == EVOKE_L0_RECORD_RETIRE &&
                 current_source.retirement.retirement_sequence != 0))
            {
                goto append_done;
            }
        }
        else if (expected_l0_born_sequence != 0)
        {
            evoke_document_cow_record current_source;

            /*
             * One VACUUM reservation can retire several dead L0 upserts in
             * the same COW publication.  The physical RETIRE record names
             * only the slot used to reserve that shared sequence, so guard
             * every later reservation against the current slot incarnation.
             */
            if (record->document_slot >= root.next_document_slot)
            {
                goto append_done;
            }
            evoke_segment_pages_load_maintenance_manifest(
                indexRelation,
                &root,
                &source_manifest
            );
            if (record->document_slot <
                source_manifest.document_slot_count)
            {
                evoke_segment_pages_load_document_record(
                    indexRelation,
                    &root,
                    &source_manifest,
                    record->document_slot,
                    &current_source
                );
                if (current_source.version.born_sequence >
                        expected_l0_born_sequence ||
                    (current_source.version.born_sequence ==
                         expected_l0_born_sequence &&
                     current_source.retirement.retirement_sequence != 0))
                {
                    goto append_done;
                }
            }
        }
        if (root.next_sequence == UINT64_MAX)
        {
            ereport(
                ERROR,
                (
                    errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                    errmsg("evoke active L0 append is unavailable")
                )
            );
        }

        record->sequence = root.next_sequence;
        if (record->kind == EVOKE_L0_RECORD_UPSERT)
        {
            record->document_slot =
                evoke_am_l0_choose_upsert_document_slot(
                    indexRelation,
                    &root
                );
            if (record->document_slot == root.next_document_slot &&
                root.next_document_slot >= UINT32_MAX)
            {
                ereport(
                    ERROR,
                    (
                        errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                        errmsg("evoke document slot capacity is exhausted")
                    )
                );
            }
        }
        else if (record->document_slot >= root.next_document_slot)
        {
            ereport(
                ERROR,
                (errmsg("invalid evoke active L0 retirement slot"))
            );
        }
        status = evoke_l0_record_serialize(
            record,
            &record_bytes,
            &record_size
        );
        if (status != EVOKE_OK)
        {
            ereport(
                ERROR,
                (
                    errmsg("failed to serialize evoke active L0 record"),
                    errdetail(
                        "Validation failed: %s.",
                        evoke_strerror(status)
                    )
                )
            );
        }
        if (record_size > UINT32_MAX ||
            record_size < EVOKE_L0_RECORD_HEADER_SIZE)
        {
            ereport(ERROR, (errmsg("evoke active L0 record is too large")));
        }
        chain_pages = evoke_am_l0_record_chain_pages(record_size);
        rotation_record_limit =
            evoke_am_active_l0_rotation_record_limit();

        if (root.active_l0.record_count != 0)
        {
            Buffer tail_buffer;
            Page tail_page;
            evoke_active_l0_page_header tail_header;

            tail_buffer = ReadBufferExtended(
                indexRelation,
                MAIN_FORKNUM,
                root.active_l0.tail_block,
                RBM_NORMAL,
                NULL
            );
            LockBuffer(tail_buffer, BUFFER_LOCK_SHARE);
            tail_page = BufferGetPage(tail_buffer);
            evoke_am_l0_read_tail_header(
                tail_page,
                &root,
                &tail_header
            );
            inline_append = evoke_am_l0_tail_can_fit(
                &tail_header,
                record_size
            );
            UnlockReleaseBuffer(tail_buffer);
        }
        rotate_active =
            root.active_l0.record_count != 0 &&
            (
                root.active_l0.record_count >= rotation_record_limit ||
                (
                    !inline_append &&
                    (uint64) root.active_l0.page_count + chain_pages >
                        EVOKE_ACTIVE_L0_ROTATION_PAGES
                )
            );

        if (rotate_active && root.pending_l0.segment_id != 0)
        {
            /*
             * Soft rotation leaves a second bounded frontier as ingress
             * headroom. Pending sealing is urgent worker work; foreground
             * writers never compact or infer.
             */
            rotate_active = false;
            *maintenance_due_out = true;
        }

        if (rotate_active)
        {
            (void) evoke_am_l0_rotate_active_locked(
                indexRelation,
                &meta,
                &root
            );
            inline_append = false;
        }
        if (root.active_l0.record_count >=
                EVOKE_ACTIVE_L0_MAX_RECORDS ||
            root.active_l0.payload_bytes >
                UINT64_MAX - (uint64) record_size ||
            (!inline_append &&
             (uint64) root.active_l0.page_count + chain_pages >
                 EVOKE_ACTIVE_L0_MAX_PAGES))
        {
            if (root.pending_l0.segment_id != 0)
            {
                ereport(
                    ERROR,
                    (
                        errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                        errmsg(
                            "evoke active L0 hard frontier is exhausted "
                            "while pending L0 is unsealed"
                        ),
                        errhint(
                            "Allow urgent background maintenance to seal "
                            "the pending segment before retrying."
                        )
                    )
                );
            }
            ereport(
                ERROR,
                (errmsg("evoke active L0 hard frontier is exhausted"))
            );
        }

        if (inline_append)
        {
            evoke_am_l0_append_inline(
                indexRelation,
                &meta,
                &root,
                record,
                record_bytes,
                record_size
            );
        }
        else
        {
            memset(&chain, 0, sizeof(chain));
            evoke_segment_pages_write_l0_record_chain(
                indexRelation,
                root.active_l0.segment_id,
                root.active_l0.page_count,
                record->sequence,
                record_bytes,
                record_size,
                &chain
            );
            evoke_am_l0_publish_chain(
                indexRelation,
                &meta,
                &root,
                record,
                record_size,
                &chain
            );
        }
        if (root.pending_l0.segment_id != 0 ||
            root.active_l0.record_count + UINT64_C(1) >=
                rotation_record_limit ||
            (uint64) root.active_l0.page_count +
                (inline_append ? 0 : chain_pages) >=
                    EVOKE_ACTIVE_L0_ROTATION_PAGES)
        {
            *maintenance_due_out = true;
        }
        appended = true;

append_done:
        ;
    }
    PG_FINALLY();
    {
        if (record_bytes != NULL)
        {
            free(record_bytes);
        }
        if (append_locked)
        {
            evoke_am_unlock_append(indexRelation);
        }
        evoke_segment_manifest_free(&source_manifest);
    }
    PG_END_TRY();
    return appended;
}
