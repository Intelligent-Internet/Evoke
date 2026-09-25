#include "postgres.h"

#include "access/xlog.h"
#include "access/xloginsert.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "storage/lock.h"
#include "utils/builtins.h"
#include "utils/rel.h"

#include "evoke_am_meta.h"
#include "evoke_am_maintenance.h"
#include "evoke_am_options.h"
#include "evoke_segment_pages.h"

#define EVOKE_AM_GENERATION_BARRIER_LOCK_TAG UINT32_C(0x32534255)

StaticAssertDecl(
    sizeof(evoke_am_meta_page) <= BLCKSZ - MAXALIGN(SizeOfPageHeaderData),
    "evoke metapage payload exceeds one PostgreSQL page"
);

bool
evoke_am_meta_uses_convergent_segment_storage(
    const evoke_am_meta_page *meta
)
{
    return meta != NULL &&
        meta->storage_version == EVOKE_AM_STORAGE_CONVERGENT_SEGMENTS;
}

bool
evoke_am_generation_identity_matches(
    const evoke_am_meta_page *cached,
    const evoke_am_meta_page *current
)
{
    if (cached == NULL || current == NULL)
    {
        return false;
    }

    if (cached->magic != current->magic ||
        cached->version != current->version ||
        cached->page_kind != current->page_kind ||
        cached->cache_epoch != current->cache_epoch ||
        cached->storage_version != current->storage_version ||
        cached->source_type != current->source_type)
    {
        return false;
    }

    if (!evoke_am_meta_uses_convergent_segment_storage(current))
    {
        return false;
    }

    /* The checked v3 read-root blob is the complete physical identity. */
    return memcmp(
        cached->segment_read_root_bytes,
        current->segment_read_root_bytes,
        sizeof(current->segment_read_root_bytes)
    ) == 0;
}

void
evoke_am_require_convergent_segment_storage(
    const evoke_am_meta_page *meta
)
{
    if (evoke_am_meta_uses_convergent_segment_storage(meta))
    {
        return;
    }

    ereport(
        ERROR,
        (
            errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
            errmsg("unsupported evoke index storage layout"),
            errdetail(
                "Storage version %u is not part of the current product path.",
                meta == NULL ? 0 : meta->storage_version
            ),
            errhint("REINDEX the evoke index to publish page-native v3 storage.")
        )
    );
}

evoke_status
evoke_am_segment_read_root_from_meta(
    const evoke_am_meta_page *meta,
    evoke_segment_read_root *root_out
)
{
    if (!evoke_am_meta_uses_convergent_segment_storage(meta) ||
        root_out == NULL)
    {
        return EVOKE_ERR_INVALID;
    }
    return evoke_segment_read_root_deserialize(
        meta->segment_read_root_bytes,
        sizeof(meta->segment_read_root_bytes),
        root_out
    );
}

BlockNumber
evoke_am_relation_nblocks(Relation indexRelation)
{
    BlockNumber nblocks;

    nblocks = RelationGetNumberOfBlocks(indexRelation);
    if (nblocks != 0)
    {
        return nblocks;
    }

    /*
     * Full rewrites truncate then extend the relation. The smgr nblocks cache
     * can transiently retain the post-truncate zero size after new pages are
     * visible. Close/reopen only on this rare zero-size path, so normal query
     * scans keep the cheap cached nblocks check.
     */
    RelationCloseSmgr(indexRelation);
    return RelationGetNumberOfBlocks(indexRelation);
}

static void
evoke_am_lock_generation_barrier_oid(Oid index_oid)
{
    LOCKTAG tag;

    /*
     * Protect generation identity and retired segments without placing a
     * heavyweight relation lock in PostgreSQL's CREATE/REINDEX lock graph.
     * Readers hold ShareLock while using a snapped generation. Publication,
     * truncation, and in-place normalization require ExclusiveLock.
     */
    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_GENERATION_BARRIER_LOCK_TAG,
        index_oid,
        0
    );
    (void) LockAcquire(&tag, ExclusiveLock, false, false);
}

static void
evoke_am_unlock_generation_barrier_oid(Oid index_oid)
{
    LOCKTAG tag;

    SET_LOCKTAG_ADVISORY(
        tag,
        MyDatabaseId,
        EVOKE_AM_GENERATION_BARRIER_LOCK_TAG,
        index_oid,
        0
    );
    if (!LockRelease(&tag, ExclusiveLock, false))
    {
        ereport(
            ERROR,
            (errmsg("evoke generation barrier is not held"))
        );
    }
}

void
evoke_am_lock_generation_barrier(Relation indexRelation)
{
    evoke_am_lock_generation_barrier_oid(RelationGetRelid(indexRelation));
}

void
evoke_am_unlock_generation_barrier(Relation indexRelation)
{
    evoke_am_unlock_generation_barrier_oid(RelationGetRelid(indexRelation));
}

void
evoke_am_mark_buffer_dirty_with_wal(
    Relation indexRelation,
    Buffer buffer
)
{
    Assert(CritSectionCount > 0);
    MarkBufferDirty(buffer);
    if (RelationNeedsWAL(indexRelation))
    {
        (void) log_newpage_buffer(buffer, false);
    }
}

void
evoke_am_write_page_at(
    Relation indexRelation,
    BlockNumber blkno,
    const void *contents,
    Size len
)
{
    Buffer buffer;
    BlockNumber nblocks;
    Page page;

    nblocks = RelationGetNumberOfBlocks(indexRelation);
    if (blkno < nblocks)
    {
        buffer = ReadBufferExtended(
            indexRelation,
            MAIN_FORKNUM,
            blkno,
            RBM_NORMAL,
            NULL
        );
    }
    else if (blkno == nblocks)
    {
        buffer = ReadBufferExtended(
            indexRelation,
            MAIN_FORKNUM,
            P_NEW,
            RBM_NORMAL,
            NULL
        );
    }
    else
    {
        ereport(ERROR, (errmsg("evoke segment write skipped a block")));
    }
    LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
    page = BufferGetPage(buffer);
    START_CRIT_SECTION();
    PageInit(page, BLCKSZ, 0);
    memcpy(PageGetContents(page), contents, len);
    /* Custom payload pages do not use the standard item-page layout. */
    evoke_am_mark_buffer_dirty_with_wal(indexRelation, buffer);
    END_CRIT_SECTION();
    UnlockReleaseBuffer(buffer);
}

void
evoke_am_write_init_page_at(
    Relation indexRelation,
    BlockNumber blkno,
    const void *contents,
    Size len
)
{
    Buffer buffer;
    BlockNumber nblocks;
    Page page;

    nblocks = RelationGetNumberOfBlocksInFork(
        indexRelation,
        INIT_FORKNUM
    );
    if (blkno != nblocks)
    {
        ereport(
            ERROR,
            (errmsg("evoke initialization fork write is not contiguous"))
        );
    }

    /*
     * PostgreSQL copies this fork over MAIN after crash recovery for an
     * UNLOGGED relation.  The fork must therefore contain a complete, valid
     * empty generation and, unlike ordinary UNLOGGED pages, be WAL-logged.
     */
    buffer = ExtendBufferedRel(
        BMR_REL(indexRelation),
        INIT_FORKNUM,
        NULL,
        EB_LOCK_FIRST | EB_SKIP_EXTENSION_LOCK
    );
    if (BufferGetBlockNumber(buffer) != blkno)
    {
        UnlockReleaseBuffer(buffer);
        ereport(
            ERROR,
            (errmsg("unexpected evoke initialization fork size"))
        );
    }

    page = BufferGetPage(buffer);
    START_CRIT_SECTION();
    PageInit(page, BLCKSZ, 0);
    memcpy(PageGetContents(page), contents, len);
    MarkBufferDirty(buffer);
    (void) log_newpage_buffer(buffer, false);
    END_CRIT_SECTION();
    UnlockReleaseBuffer(buffer);
}

void
evoke_am_write_new_page(Relation indexRelation, const void *contents, Size len)
{
    evoke_am_write_page_at(
        indexRelation,
        RelationGetNumberOfBlocks(indexRelation),
        contents,
        len
    );
}

void
evoke_am_payload_health(
    Relation indexRelation,
    const evoke_am_meta_page *meta,
    evoke_am_payload_health_state *health_out
)
{
    evoke_segment_read_root root;
    BlockNumber nblocks;
    evoke_status status;

    memset(health_out, 0, sizeof(*health_out));
    health_out->status = "ok";
    health_out->reason = "ok";
    if (meta == NULL ||
        !evoke_am_meta_uses_convergent_segment_storage(meta))
    {
        health_out->corrupt = true;
        health_out->rebuild_required = true;
        health_out->status = "corrupt";
        health_out->reason = meta == NULL
            ? "missing_meta"
            : "unsupported_storage_layout";
        return;
    }

    nblocks = evoke_am_relation_nblocks(indexRelation);
    status = evoke_am_segment_read_root_from_meta(meta, &root);
    health_out->capacity_bytes =
        (uint64) nblocks * (uint64) BLCKSZ;
    if (status != EVOKE_OK)
    {
        health_out->corrupt = true;
        health_out->rebuild_required = true;
        health_out->status = "corrupt";
        health_out->reason = "invalid_segment_read_root";
        return;
    }
    health_out->expected_bytes =
        (uint64) root.published_block_high_watermark *
        (uint64) BLCKSZ;
    if (root.published_block_high_watermark > (uint32) nblocks)
    {
        health_out->corrupt = true;
        health_out->rebuild_required = true;
        health_out->status = "corrupt";
        health_out->reason = "segment_root_out_of_bounds";
        return;
    }
    if ((meta->flags & EVOKE_AM_FLAG_CORRUPT) != 0)
    {
        health_out->corrupt = true;
        health_out->rebuild_required = true;
        health_out->status = "corrupt";
        health_out->reason = "marked_corrupt";
        return;
    }
    if ((meta->flags & EVOKE_AM_FLAG_REBUILD_REQUIRED) != 0)
    {
        health_out->rebuild_required = true;
        health_out->status = "rebuild_required";
        health_out->reason = "marked_rebuild_required";
        return;
    }
    if ((meta->flags & EVOKE_AM_FLAG_STALE) != 0)
    {
        health_out->rebuild_required = true;
        health_out->status = "stale";
        health_out->reason = "stale";
    }
}

uint64
evoke_am_segment_object_bytes_add(uint64 left, uint64 right)
{
    return evoke_u64_saturating_add(left, right);
}

uint64
evoke_am_convergent_object_bytes(
    Relation indexRelation,
    const evoke_am_meta_page *meta
)
{
    evoke_segment_read_root root;

    if (evoke_am_segment_read_root_from_meta(meta, &root) != EVOKE_OK)
    {
        ereport(ERROR, (errmsg("invalid evoke convergent segment root")));
    }
    (void) indexRelation;

    /*
     * The page-native generation may own posting objects through COW leaves,
     * not only through manifest segment descriptors. The published high-water
     * mark is the O(1), conservative runtime working-set estimate. Explicit
     * generation diagnostics report exact reachable bytes instead.
     */
    return (uint64) root.published_block_high_watermark * BLCKSZ;
}

void
evoke_am_convergent_mutation_debt_read(
    Relation index_relation,
    const evoke_am_meta_page *meta,
    evoke_am_convergent_mutation_debt *debt_out
)
{
    evoke_segment_read_root root;
    evoke_l0_storage_snapshot snapshot;
    evoke_status status;

    if (index_relation == NULL || meta == NULL || debt_out == NULL ||
        !evoke_am_meta_uses_convergent_segment_storage(meta))
    {
        ereport(ERROR, (errmsg("invalid evoke linked L0 debt request")));
    }
    memset(debt_out, 0, sizeof(*debt_out));
    if (evoke_am_get_consistency(index_relation) ==
        EVOKE_AM_CONSISTENCY_MANUAL)
    {
        debt_out->upserts = meta->pending_write_tuples;
        debt_out->retirements = meta->pending_delete_tuples;
    }
    status = evoke_am_segment_read_root_from_meta(meta, &root);
    if (status != EVOKE_OK)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke linked L0 debt root"),
                errdetail(
                    "Validation failed: %s.",
                    evoke_strerror(status)
                )
            )
        );
    }

    evoke_l0_storage_snapshot_init(&snapshot);
    PG_TRY();
    {
        evoke_segment_pages_load_l0_snapshot(
            index_relation,
            &root,
            &snapshot
        );
        for (uint32 record_index = 0;
             record_index < snapshot.record_count;
             record_index++)
        {
            if ((record_index & 1023) == 0)
            {
                CHECK_FOR_INTERRUPTS();
            }
            switch (snapshot.records[record_index].view.kind)
            {
                case EVOKE_L0_RECORD_UPSERT:
                    debt_out->upserts = evoke_u32_saturating_add(
                        debt_out->upserts,
                        1
                    );
                    break;
                case EVOKE_L0_RECORD_RETIRE:
                    debt_out->retirements =
                        evoke_u32_saturating_add(
                            debt_out->retirements,
                            1
                        );
                    break;
                case EVOKE_L0_RECORD_SEMANTIC_COMPLETE:
                case EVOKE_L0_RECORD_SEMANTIC_QUARANTINE:
                    break;
                case EVOKE_L0_RECORD_INVALID:
                default:
                    ereport(
                        ERROR,
                        (errmsg("invalid evoke linked L0 debt record"))
                    );
            }
        }
        debt_out->records = snapshot.record_count;
        debt_out->bytes = snapshot.payload_bytes;
    }
    PG_FINALLY();
    {
        evoke_l0_storage_snapshot_free(&snapshot);
    }
    PG_END_TRY();
}

static void
evoke_am_normalize_meta_storage(
    evoke_am_meta_page *meta,
    BlockNumber nblocks
)
{
    evoke_segment_read_root read_root;
    evoke_status root_status;

    if (meta == NULL)
    {
        return;
    }
    if (evoke_am_meta_uses_convergent_segment_storage(meta))
    {
        /*
         * The fixed root blob is the only physical authority in v3. Retired
         * fixed-layout fields must not silently describe a second
         * base/delta/semantic layout beside it. Pending counters are
         * non-physical manual-mode debt: manual writes leave the root
         * unchanged while reporting that an explicit refresh is required.
         */
        if (meta->tid_bytes_len != 0 ||
            meta->index_bytes_len != 0 ||
            meta->delta_record_count != 0 ||
            meta->delta_bytes_len != 0 ||
            meta->active_generation != 0 ||
            meta->active_start_blkno != 0 ||
            meta->active_data_pages != 0 ||
            meta->delta_start_blkno != 0 ||
            meta->delta_data_pages != 0 ||
            meta->semantic_start_blkno != 0 ||
            meta->semantic_data_pages != 0 ||
            meta->semantic_record_count != 0 ||
            meta->semantic_bytes_len != 0 ||
            meta->semantic_signature[0] != '\0')
        {
            meta->flags |= EVOKE_AM_FLAG_CORRUPT |
                EVOKE_AM_FLAG_REBUILD_REQUIRED;
            return;
        }
        root_status = evoke_am_segment_read_root_from_meta(
            meta,
            &read_root
        );
        if (root_status != EVOKE_OK ||
            read_root.published_block_high_watermark >
                (uint32) nblocks)
        {
            meta->flags |= EVOKE_AM_FLAG_CORRUPT |
                EVOKE_AM_FLAG_REBUILD_REQUIRED;
        }
        return;
    }

    meta->flags |= EVOKE_AM_FLAG_REBUILD_REQUIRED;
}

void
evoke_am_read_meta(
    Relation indexRelation,
    evoke_am_meta_page *meta_out
)
{
    Buffer buffer;
    Page page;
    evoke_am_meta_page *meta;

    if (evoke_am_relation_nblocks(indexRelation) == 0)
    {
        ereport(
            ERROR,
            (
                errmsg(
                    "evoke index relation %u (%s) is empty",
                    RelationGetRelid(indexRelation),
                    RelationGetRelationName(indexRelation)
                )
            )
        );
    }

    buffer = ReadBufferExtended(indexRelation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
    LockBuffer(buffer, BUFFER_LOCK_SHARE);
    page = BufferGetPage(buffer);
    meta = (evoke_am_meta_page *) PageGetContents(page);

    if (meta->magic != EVOKE_AM_MAGIC ||
        meta->version != EVOKE_AM_VERSION ||
        meta->page_kind != EVOKE_AM_PAGE_META)
    {
        uint32 magic = meta->magic;
        uint32 version = meta->version;
        uint32 page_kind = meta->page_kind;
        Oid relid = RelationGetRelid(indexRelation);
        const char *relname = RelationGetRelationName(indexRelation);

        UnlockReleaseBuffer(buffer);
        if (magic == EVOKE_AM_MAGIC &&
            page_kind == EVOKE_AM_PAGE_META &&
            version != EVOKE_AM_VERSION)
        {
            ereport(
                ERROR,
                (
                    errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                    errmsg(
                        "unsupported evoke index metapage version %u",
                        version
                    ),
                    errdetail(
                        "The current product format requires metapage "
                        "version %u.",
                        EVOKE_AM_VERSION
                    ),
                    errhint("REINDEX the evoke index to publish the current "
                            "sole-authority format.")
                )
            );
        }
        ereport(
            ERROR,
            (
                errmsg(
                    "invalid evoke index metapage for relation %u (%s): "
                    "magic=%u version=%u page_kind=%u",
                    relid,
                    relname != NULL ? relname : "<unknown>",
                    magic,
                    version,
                    page_kind
                )
            )
        );
    }

    *meta_out = *meta;
    evoke_am_normalize_meta_storage(
        meta_out,
        evoke_am_relation_nblocks(indexRelation)
    );
    UnlockReleaseBuffer(buffer);
}

bool
evoke_am_try_read_current_meta(
    Relation indexRelation,
    evoke_am_meta_page *meta_out
)
{
    Buffer buffer;
    Page page;
    evoke_am_meta_page *meta;

    if (indexRelation == NULL || meta_out == NULL ||
        evoke_am_relation_nblocks(indexRelation) == 0)
    {
        return false;
    }

    buffer = ReadBufferExtended(indexRelation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
    LockBuffer(buffer, BUFFER_LOCK_SHARE);
    page = BufferGetPage(buffer);
    meta = (evoke_am_meta_page *) PageGetContents(page);

    if (meta->magic != EVOKE_AM_MAGIC ||
        meta->version != EVOKE_AM_VERSION ||
        meta->page_kind != EVOKE_AM_PAGE_META)
    {
        UnlockReleaseBuffer(buffer);
        return false;
    }

    *meta_out = *meta;
    evoke_am_normalize_meta_storage(
        meta_out,
        evoke_am_relation_nblocks(indexRelation)
    );
    UnlockReleaseBuffer(buffer);
    return true;
}

static void
evoke_am_update_meta_flags(
    Relation indexRelation,
    uint16 set_mask,
    uint16 clear_mask
)
{
    Buffer buffer;
    Page page;
    evoke_am_meta_page *meta;
    uint16 next_flags;

    evoke_am_lock_append(indexRelation);
    buffer = ReadBufferExtended(indexRelation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
    LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
    page = BufferGetPage(buffer);
    meta = (evoke_am_meta_page *) PageGetContents(page);

    if (meta->magic != EVOKE_AM_MAGIC ||
        meta->version != EVOKE_AM_VERSION ||
        meta->page_kind != EVOKE_AM_PAGE_META)
    {
        UnlockReleaseBuffer(buffer);
        evoke_am_unlock_append(indexRelation);
        ereport(ERROR, (errmsg("invalid evoke index metapage")));
    }

    next_flags = (uint16) ((meta->flags | set_mask) & ~clear_mask);
    if (meta->flags != next_flags)
    {
        START_CRIT_SECTION();
        meta->flags = next_flags;
        evoke_am_mark_buffer_dirty_with_wal(indexRelation, buffer);
        END_CRIT_SECTION();
    }
    UnlockReleaseBuffer(buffer);
    evoke_am_unlock_append(indexRelation);
}

void
evoke_am_note_maintenance_activity(
    Relation indexRelation,
    uint32 pending_write_add,
    uint32 pending_delete_add
)
{
    Buffer buffer;
    Page page;
    evoke_am_meta_page *meta;

    if (RelationGetNumberOfBlocks(indexRelation) == 0)
    {
        return;
    }

    /*
     * Automatic v3 maintenance derives exact debt from linked-L0. These
     * metapage counters remain only for manual-consistency stale accounting,
     * where no query-visible L0 record is published by foreground mutation.
     */
    if (evoke_am_maintenance_tracking_enabled(indexRelation))
    {
        return;
    }

    evoke_am_pin_maintenance_xact(indexRelation);
    evoke_am_lock_append(indexRelation);
    buffer = ReadBufferExtended(indexRelation, MAIN_FORKNUM, 0, RBM_NORMAL, NULL);
    LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
    page = BufferGetPage(buffer);
    meta = (evoke_am_meta_page *) PageGetContents(page);

    if (meta->magic != EVOKE_AM_MAGIC ||
        meta->version != EVOKE_AM_VERSION ||
        meta->page_kind != EVOKE_AM_PAGE_META)
    {
        UnlockReleaseBuffer(buffer);
        evoke_am_unlock_append(indexRelation);
        ereport(ERROR, (errmsg("invalid evoke index metapage")));
    }

    START_CRIT_SECTION();
    meta->pending_write_tuples = evoke_u32_saturating_add(
        meta->pending_write_tuples,
        pending_write_add
    );
    meta->pending_delete_tuples = evoke_u32_saturating_add(
        meta->pending_delete_tuples,
        pending_delete_add
    );
    evoke_am_mark_buffer_dirty_with_wal(indexRelation, buffer);
    END_CRIT_SECTION();
    UnlockReleaseBuffer(buffer);
    evoke_am_unlock_append(indexRelation);
}

void
evoke_am_mark_stale(Relation indexRelation)
{
    evoke_am_meta_page meta;

    if (RelationGetNumberOfBlocks(indexRelation) == 0)
    {
        return;
    }

    evoke_am_read_meta(indexRelation, &meta);
    if ((meta.flags & EVOKE_AM_FLAG_STALE) != 0)
    {
        return;
    }

    evoke_am_update_meta_flags(
        indexRelation,
        EVOKE_AM_FLAG_STALE,
        0
    );
}

void
evoke_am_get_stats(
    Relation indexRelation,
    IndexBulkDeleteResult *stats
)
{
    evoke_am_meta_page meta;

    if (RelationGetNumberOfBlocks(indexRelation) == 0)
    {
        memset(stats, 0, sizeof(*stats));
        stats->num_pages = 0;
        return;
    }

    evoke_am_read_meta(indexRelation, &meta);
    stats->num_pages = RelationGetNumberOfBlocks(indexRelation);
    stats->num_index_tuples = meta.num_docs;
}

uint64
evoke_am_next_rebuild_count(Relation indexRelation)
{
    evoke_am_meta_page meta;

    if (RelationGetNumberOfBlocks(indexRelation) == 0)
    {
        return 1;
    }

    evoke_am_read_meta(indexRelation, &meta);
    if (meta.rebuild_count == UINT64_MAX)
    {
        return UINT64_MAX;
    }

    return meta.rebuild_count + 1;
}

uint16
evoke_am_next_cache_epoch(Relation indexRelation)
{
    evoke_am_meta_page meta;

    if (RelationGetNumberOfBlocks(indexRelation) == 0)
    {
        return 1;
    }

    evoke_am_read_meta(indexRelation, &meta);
    if (meta.cache_epoch == UINT16_MAX)
    {
        return 1;
    }

    return (uint16) (meta.cache_epoch + 1);
}

void
evoke_am_publish_rebuild_meta(
    Relation indexRelation,
    ForkNumber fork_number,
    const evoke_segment_read_root *root,
    Oid source_type,
    uint32 num_docs,
    uint16 cache_epoch,
    uint16 flags,
    uint64 rebuild_count
)
{
    Buffer meta_buffer;
    Page meta_page_raw;
    evoke_am_meta_page *meta;
    evoke_am_meta_page next_meta;
    uint8 root_bytes[EVOKE_SEGMENT_READ_ROOT_SERIALIZED_SIZE];
    evoke_status status;
    XLogRecPtr publish_lsn = InvalidXLogRecPtr;
    BlockNumber nblocks;

    if ((fork_number != MAIN_FORKNUM &&
         fork_number != INIT_FORKNUM) ||
        root == NULL || rebuild_count == 0)
    {
        ereport(
            ERROR,
            (errmsg("invalid evoke convergent segment publication"))
        );
    }
    status = evoke_segment_read_root_serialize(
        root,
        root_bytes,
        sizeof(root_bytes)
    );
    if (status != EVOKE_OK)
    {
        ereport(
            ERROR,
            (
                errmsg("invalid evoke convergent segment root"),
                errdetail("Validation failed: %s.", evoke_strerror(status))
            )
        );
    }
    nblocks = RelationGetNumberOfBlocksInFork(
        indexRelation,
        fork_number
    );
    if (nblocks > UINT32_MAX ||
        root->published_block_high_watermark != (uint32) nblocks)
    {
        ereport(
            ERROR,
            (errmsg("evoke convergent segment high-water mark is stale"))
        );
    }

    meta_buffer = ReadBufferExtended(
        indexRelation,
        fork_number,
        0,
        RBM_NORMAL,
        NULL
    );
    LockBuffer(meta_buffer, BUFFER_LOCK_EXCLUSIVE);
    meta_page_raw = BufferGetPage(meta_buffer);
    meta = (evoke_am_meta_page *) PageGetContents(meta_page_raw);
    if (meta->magic != EVOKE_AM_MAGIC ||
        meta->version != EVOKE_AM_VERSION ||
        meta->page_kind != EVOKE_AM_PAGE_META)
    {
        UnlockReleaseBuffer(meta_buffer);
        ereport(ERROR, (errmsg("invalid evoke index metapage")));
    }

    memset(&next_meta, 0, sizeof(next_meta));
    next_meta.magic = EVOKE_AM_MAGIC;
    next_meta.version = EVOKE_AM_VERSION;
    next_meta.page_kind = EVOKE_AM_PAGE_META;
    next_meta.flags = flags;
    next_meta.cache_epoch = cache_epoch;
    next_meta.source_type = source_type;
    next_meta.num_docs = num_docs;
    next_meta.rebuild_count = rebuild_count;
    next_meta.storage_version = EVOKE_AM_STORAGE_CONVERGENT_SEGMENTS;
    memcpy(
        next_meta.segment_read_root_bytes,
        root_bytes,
        sizeof(next_meta.segment_read_root_bytes)
    );

    START_CRIT_SECTION();
    *meta = next_meta;
    MarkBufferDirty(meta_buffer);
    if (fork_number == INIT_FORKNUM || RelationNeedsWAL(indexRelation))
    {
        (void) log_newpage_buffer(meta_buffer, false);
        publish_lsn = PageGetLSN(meta_page_raw);
    }
    END_CRIT_SECTION();
    UnlockReleaseBuffer(meta_buffer);

    if (publish_lsn != InvalidXLogRecPtr)
    {
        XLogFlush(publish_lsn);
    }
}

bool
evoke_am_meta_has_pending_maintenance(
    const evoke_am_meta_page *meta
)
{
    return meta != NULL &&
        (meta->pending_write_tuples > 0 || meta->pending_delete_tuples > 0);
}
